import copy
from functools import partial

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F

from holoformer.datasets.hf_datasets import HfDatasetDataModule
from holoformer.models import hrr
from holoformer.models.position import (
    HolographicPositionalEncoding, PositionalEncoding
)
from .top_k_sampling import top_k_top_p_filtering


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class HolographicQKV(nn.Module):
    def __init__(self, dims, heads, causal=True, unit=True):
        super().__init__()
        self.causal = causal
        self.unit = unit
        self.heads = heads
        self.query = nn.Sequential(
            nn.Linear(dims, dims),
        )
        self.key = nn.Sequential(
            nn.Linear(dims, dims),
        )
        self.value = nn.Sequential(
            nn.Linear(dims, dims),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('linear')
            )
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x.shape ~= (batch, sequence, embedding)
        """
        batch, seq, dims = x.shape
        head_dims = dims // self.heads
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q, k, v = map(
            lambda x: x.view(batch, seq, self.heads, head_dims),
            (q, k, v)
        )
        if self.unit:
            q, k, v = map(hrr.unit_projection, (q, k, v))
        x_k = hrr.bind(k, v)
        if self.causal:
            s = x_k.cumsum(dim=1)
        else:
            s = x_k.sum(dim=1, keepdims=True)
        values = hrr.unbind(s, q)
        values = values.view(batch, seq, dims)
        return values


class HoloformerEncoderLayer(nn.Module):
    def __init__(
        self, dims, ff_dims, dropout,
        mixer=HolographicQKV, **kwargs
    ):
        super().__init__()
        self.mixer = nn.Sequential(
            mixer(dims),
            nn.Dropout(dropout),
        )
        # self.ln1 = nn.LayerNorm(dims)
        # self.ln2 = nn.LayerNorm(dims)
        self.ln1 = hrr.unit_projection
        self.ln2 = hrr.unit_projection
        self.mlp = nn.Sequential(
            nn.Linear(dims, 4 * dims),
            nn.GELU(),
            nn.Linear(4 * dims, dims),
            nn.Dropout(dropout),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)

    def forward(self, x, **kwargs):
        x = x + self.mixer(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HoloformerMLM(pl.LightningModule):
    """Holoformer using masked language model"""
    def __init__(self, tokenizer, data_dims=100, ff_dims=512, layers=4,
                 lr=0.001, weight_decay=0.1, dropout=0.1,
                 activation=nn.ReLU, pad_token_id=0,
                 update_embedding=False, lr_warmup_steps=3,
                 opt_betas=(0.9, 0.95), heads=8, max_seq_len=256,
                 pe_type='holo', p_mask=0.15, p_random_mask=0.2,
                 p_unmask=0.2, mask_token_id=1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.data_dims = data_dims
        self.ff_dims = ff_dims
        self.opt_betas = opt_betas
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type
        self.embedding = nn.Embedding(
            len(tokenizer), data_dims, padding_idx=pad_token_id,
        )
        self.embedding.weight.data = hrr.init(self.embedding.weight.data.shape)
        self.embedding.requires_grad_(update_embedding)

        if pe_type == 'holo':
            self.positional_encoding = HolographicPositionalEncoding(data_dims)
            self.positional_encoding.requires_grad_(update_embedding)
        else:
            self.positional_encoding = PositionalEncoding(data_dims)
        self.output_token = self.get_output_head(len(tokenizer))

        self.register_buffer('presence_embeddings', hrr.init_ortho(
            (2, data_dims)
        ))# ).unsqueeze(1).unsqueeze(1))

        mixer = partial(HolographicQKV, heads=heads, causal=False)
        transformer_layer = HoloformerEncoderLayer(
            data_dims, ff_dims, dropout=dropout, activation=activation,
            mixer=mixer
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.hrr_dist = Normal(0., 1. / data_dims)
        self.update_embedding = update_embedding
        self.f_loss = nn.CrossEntropyLoss()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('leaky_relu')
            )
            nn.init.zeros_(m.bias)

    def get_output_head(self, output_dims):
        return nn.Sequential(
            nn.Linear(self.data_dims, output_dims)
        )

    def forward(self, x, **kwargs):
        encoded = self.encode_sequence(x)
        # encoded = encoded / (torch.norm(encoded, dim=-1, keepdim=True) + 1e-8)
        return self.output_token(encoded)

    def encode_sequence(self, x):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        # present_emb = self.presence_embeddings[1]
        # embedded = hrr.unbind(embedded, present_emb)
        encoded = self.encoder(embedded)
        return encoded

    def outputs_to_ids(self, x, temperature=0., top_p=0.9, top_k=0):
        if temperature:
            x = x / temperature
        if top_p or top_k:
            return top_k_top_p_filtering(x[:, -1], top_p=top_p, top_k=top_k)
        elif temperature:
            dist = Categorical(logits=x)
            return dist.sample()
        else:
            return x.argmax(-1)

    def training_step(self, batch, batch_idx):
        metrics, losses = self._shared_step(batch, batch_idx)
        self.log_dict(metrics)
        return losses

    def validation_step(self, batch, batch_idx):
        metrics, losses = self._shared_step(batch, batch_idx)
        metrics = {
            f'val_{k}': v for k, v in metrics.items()
        }
        self.log_dict(metrics)
        return losses

    def _shared_step(self, data, batch_idx):
        all_tokens = data['input_ids'].clone()
        masked_tokens, mask = self.mask_tokens(all_tokens)
        recon_tokens = self(masked_tokens)
        all_tokens[~mask] = -100  # Don't calculate loss for the unmasked
        recon_loss = F.cross_entropy(
            recon_tokens.permute(0, 2, 1), all_tokens, ignore_index=-100
        )

        embedding_loss = torch.tensor(0, device=self.device)
        positional_loss = torch.tensor(0, device=self.device)
        # if self.update_embedding:
        #     embedding_loss = hrr.unit_regularization(self.embedding.weight).mean()
        #     positional_loss = self.positional_encoding.loss(all_tokens).mean()

        loss = recon_loss + embedding_loss + positional_loss
        metrics = dict(
            loss=loss,
            recon_loss=recon_loss,
            embedding_loss=embedding_loss,
            positional_loss=positional_loss,
        )
        losses = dict(
            loss=loss
        )
        return metrics, losses

    def mask_tokens(self, all_tokens):
        mask = torch.rand(*all_tokens.shape, device=self.device)
        mask = (mask < self.hparams.p_mask) * (all_tokens != self.pad_token_id)
        masked_tokens = all_tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        # leave some tokens unmasked
        unmask = torch.rand_like(all_tokens, dtype=torch.float32)
        unmask = (unmask < self.hparams.p_unmask) * mask
        masked_tokens[unmask] = all_tokens[unmask]
        # assign random tokens
        random_mask = torch.rand_like(all_tokens, dtype=torch.float32)
        random_mask = (random_mask < self.hparams.p_random_mask) * mask
        random_indices = torch.randint_like(all_tokens, 1, len(self.tokenizer))
        masked_tokens[random_mask] = random_indices[random_mask]
        return masked_tokens, mask

    def configure_optimizers(self):
        """
        From minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        if (
            hasattr(self, 'positional_encoding') and
            hasattr(self.positional_encoding, 'embeddings')
        ):
            no_decay.add('positional_encoding.embeddings')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, betas=self.opt_betas)
        return optimizer

    @classmethod
    def add_argparse_args(self, p):
        p.add_argument('--data_dims', default=128, type=int)
        p.add_argument('--ff_dims', default=1024, type=int)
        p.add_argument('--heads', default=8, type=int)
        p.add_argument('--lr', default=0.0003, type=float)
        p.add_argument('--weight_decay', default=0.1, type=float)
        p.add_argument('--layers', default=4, type=int)
        p.add_argument('--dropout', default=0.1, type=float)
        p.add_argument('--batch_size', default=32, type=int)
        p.add_argument('--lr_warmup_steps', default=3, type=int)
        p.add_argument('--update_embedding', action='store_true')
        p.add_argument('--opt_betas', default=(0.9, 0.95), type=parse_csv_arg(float))
        p.add_argument('--pe_type', default='cos')
        return p


def parse_csv_arg(type_):
    def f(v):
        return tuple(map(type_, v.split(',')))
    return f


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('tokenizer_name')
    p.add_argument('--max_seq_len', default=256, type=int)
    p.add_argument('--p_print', default=0.01, type=float)

    p = HoloformerMLM.add_argparse_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    print('Building DataModule')
    dm = HfDatasetDataModule(**vars(args))
    dm.setup('fit')
    num_tokens = len(dm.tokenizer)

    print('Building model')
    mask_token_id, pad_token_id = dm.tokenizer.convert_tokens_to_ids([
        '[MASK]', '[PAD]'
    ])
    model = HoloformerMLM(
        tokenizer=dm.tokenizer,
        num_tokens=num_tokens,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        **vars(args)
    )

    print('Set up Trainer')
    model_checkpoint = ModelCheckpoint()
    callbacks = [model_checkpoint]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks
    )
    trainer.fit(model, dm)
