import copy
from functools import partial

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
from torch.distributions import Normal
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from holoformer.datasets.hf_datasets import HfDatasetDataModule
from holoformer.models import hrr
from holoformer.models.callbacks.ar import AutoRegressiveTextBatch
from holoformer.models.position import (
    HolographicPositionalEncoding, PositionalEncoding
)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class HolographicQKV(nn.Module):
    def __init__(self, dims, ff_dims):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(dims, dims),
        #    nn.Tanh(),
        )
        self.key = nn.Sequential(
            nn.Linear(dims, dims),
        #    nn.Tanh(),
        )
        self.value = nn.Sequential(
            nn.Linear(dims, dims),
        #    nn.Tanh(),
        )

    def forward(self, x):
        """
        x.shape ~= (batch, sequence, embedding)
        """
        query = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        # query = hrr.unit_projection(self.query(x))
        # keys = hrr.unit_projection(self.key(x))
        # values = hrr.unit_projection(self.value(x))
        x_k = hrr.bind(keys, values)
        s = x_k.sum(dim=1, keepdim=True)
        values = hrr.unbind(s, query)
        return values


class CausalHolographicQKV(nn.Module):
    def __init__(self, dims, ff_dims, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Sequential(
            nn.Linear(dims, dims),
        #    nn.Tanh(),
        )
        self.key = nn.Sequential(
            nn.Linear(dims, dims),
        #    nn.Tanh(),
        )
        self.value = nn.Sequential(
            nn.Linear(dims, dims),
        #    nn.Tanh(),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('tanh')
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
        q, k, v = map(hrr.unit_projection, (q, k, v))
        x_k = hrr.bind(k, v)
        s = x_k.cumsum(dim=1)
        values = hrr.unbind(s, q)
        values = values.view(batch, seq, dims)
        return values


class HoloformerEncoderLayer(nn.Module):
    def __init__(
        self, dims, ff_dims, dropout,
        mixer=CausalHolographicQKV, **kwargs
    ):
        super().__init__()
        self.mixer = nn.Sequential(
            mixer(dims, ff_dims),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(dims)
        self.ln2 = nn.LayerNorm(dims)
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


class HoloformerAR(pl.LightningModule):
    """Auto-regressive holoformer"""
    def __init__(self, tokenizer, data_dims=100, ff_dims=512, layers=4,
                 lr=0.001, weight_decay=0.1, dropout=0.1,
                 activation=nn.ReLU, pad_token_id=0,
                 update_embedding=False, lr_warmup_steps=3,
                 opt_betas=(0.9, 0.95), heads=8, max_seq_len=256,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.data_dims = data_dims
        self.ff_dims = ff_dims
        self.opt_betas = opt_betas
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(
            len(tokenizer), data_dims, padding_idx=pad_token_id,
        )
        self.embedding.weight.data = hrr.init(self.embedding.weight.data.shape)
        self.embedding.requires_grad_(update_embedding)

        self.positional_encoding = HolographicPositionalEncoding(data_dims)
        self.positional_encoding.requires_grad_(update_embedding)
        self.output_token = nn.Sequential(
            nn.Linear(data_dims, data_dims),
            nn.LeakyReLU(),
            nn.Linear(data_dims, num_tokens)
        )
        self.output_token.apply(self.init_weights)

        self.register_buffer('presence_embeddings', hrr.init_ortho(
            (2, data_dims)
        ).unsqueeze(1).unsqueeze(1))

        mixer = partial(CausalHolographicQKV, heads=heads)
        transformer_layer = HoloformerEncoderLayer(
            data_dims, ff_dims, dropout=dropout, activation=activation,
            mixer=mixer
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.hrr_dist = Normal(0., 1. / data_dims)
        self.update_embedding = update_embedding
        self.ce_loss = nn.CrossEntropyLoss()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('leaky_relu')
            )
            nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        # present_emb = self.presence_embeddings[1]
        # embedded = hrr.unbind(embedded, present_emb)
        y = self.encoder(embedded)
        #y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return self.output_token(y)

    def embeddings_to_ids(self, x):
        return x.argmax(-1)

    def generate(self, prompt, max_length=None, temperature=1.):
        length = prompt.shape[1]
        if max_length is None:
            max_length = self.max_seq_len
        tokens = prompt
        while length < max_length:
            p_tokens = self(tokens)
            next_tokens = self.embeddings_to_ids(p_tokens)
            tokens = torch.cat([tokens, next_tokens[:, -1:]], dim=1)
            length += 1
        return tokens

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
        p_tokens = self(all_tokens[:, :-1])
        target_tokens = all_tokens[:, 1:]
        recon_loss = F.cross_entropy(
            p_tokens.permute(0, 2, 1), target_tokens
        )

        embedding_loss = torch.tensor(0, device=self.device)
        positional_loss = torch.tensor(0, device=self.device)
        if self.update_embedding:
            embedding_loss = hrr.unit_regularization(self.embedding.weight).mean()
            positional_loss = self.positional_encoding.loss(all_tokens).mean()

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=self.opt_betas,
            # weight_decay=self.weight_decay
        )
        return optimizer
        def lr_update(epoch):
            if epoch < self.hparams.lr_warmup_steps:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.lr_warmup_steps - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_update
        )

        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'loss',
                }
            ]
        )

    @classmethod
    def add_argparse_args(self, p):
        p.add_argument('--data_dims', default=128, type=int)
        p.add_argument('--ff_dims', default=512, type=int)
        p.add_argument('--heads', default=8, type=int)
        p.add_argument('--lr', default=0.001, type=float)
        p.add_argument('--weight_decay', default=1e-4, type=float)
        p.add_argument('--layers', default=4, type=int)
        p.add_argument('--dropout', default=0.1, type=float)
        p.add_argument('--batch_size', default=32, type=int)
        p.add_argument('--lr_warmup_steps', default=3, type=int)
        p.add_argument('--update_embedding', action='store_true')
        p.add_argument('--opt_betas', default=(0.9, 0.95), type=parse_csv_arg(float))
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

    p = HoloformerAR.add_argparse_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    print('Building DataModule')
    dm = HfDatasetDataModule(**vars(args))
    dm.setup('fit')
    num_tokens = len(dm.tokenizer)

    print('Building model')
    pad_token_id, = dm.tokenizer.convert_tokens_to_ids([
        '[PAD]'
    ])
    model = HoloformerAR(
        tokenizer=dm.tokenizer,
        num_tokens=num_tokens,
        pad_token_id=pad_token_id,
        **vars(args)
    )

    print('Set up Trainer')
    model_checkpoint = ModelCheckpoint()
    callbacks = [model_checkpoint, AutoRegressiveTextBatch(p_print=0.01)]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks
    )
    trainer.fit(model, dm)
