import copy

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
from holoformer.models.callbacks.mlm import EchoMLMTextBatch
from holoformer.models.position import (
    HolographicPositionalEncoding, PositionalEncoding
)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class HolographicMixer(nn.Module):
    def __init__(self, dims, ff_dims):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Tanh(),
        )
        self.key = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Tanh(),
        )
        self.value = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        x.shape ~= (batch, sequence, embedding)
        """
        query = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        remove_query = self.remove_query(x)
        # query = hrr.unit_projection(self.query(x))
        # keys = hrr.unit_projection(self.key(x))
        # values = hrr.unit_projection(self.value(x))
        # remove_query = hrr.unit_projection(self.remove_query(x))
        x_k = hrr.bind(keys, values)
        s = x_k.sum(dim=1, keepdim=True)
        values = hrr.unbind(s, query)
        x = x + values
        remove_values = hrr.unbind(s, remove_query)
        removes = hrr.bind(remove_query, remove_values)
        x = x - removes
        return x


class HoloformerEncoderLayer(nn.Module):
    def __init__(self, dims, ff_dims, dropout, activation=nn.ReLU, **kwargs):
        super().__init__()
        self.mixer = nn.Sequential(
            HolographicMixer(dims, ff_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        return self.mixer(x)


class HoloformerMLM(pl.LightningModule):
    def __init__(self, tokenizer, data_dims=100, ff_dims=512, layers=4,
                 lr=0.001, weight_decay=1e-5, dropout=0.1,
                 activation=nn.ReLU, pad_token_id=0, mask_token_id=1,
                 update_embedding=False, p_mask=0.15, p_random_mask=0.2,
                 p_unmask=0.2, lr_warmup_steps=3,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.data_dims = data_dims
        self.ff_dims = ff_dims
        self.p_mask = p_mask
        self.p_random_mask = p_random_mask
        self.p_unmask = p_unmask
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            len(tokenizer), data_dims, padding_idx=pad_token_id,
        )
        self.embedding.weight.data = hrr.init(self.embedding.weight.data.shape)
        self.embedding.requires_grad_(update_embedding)

        self.positional_encoding = HolographicPositionalEncoding(data_dims)
        self.positional_encoding.requires_grad_(update_embedding)
        self.output_token = nn.Linear(
            data_dims, num_tokens,
        )
        self.register_buffer('presence_embeddings', hrr.init_ortho(
            (2, data_dims)
        ).unsqueeze(1).unsqueeze(1))

        transformer_layer = HoloformerEncoderLayer(
            data_dims, ff_dims, dropout=dropout, activation=activation
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.hrr_dist = Normal(0., 1. / data_dims)
        self.update_embedding = update_embedding
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        present_emb = self.presence_embeddings[1]
        embedded = hrr.unbind(embedded, present_emb)
        y = self.encoder(embedded)
        return self.output_token(y)

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
            recon_tokens.permute(0, 2, 1), all_tokens
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

    def mask_tokens(self, all_tokens):
        mask = torch.rand(*all_tokens.shape, device=self.device)
        mask = (mask < self.p_mask) * (all_tokens != self.pad_token_id)
        masked_tokens = all_tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        # leave some tokens unmasked
        unmask = torch.rand_like(all_tokens, dtype=torch.float32)
        unmask = (unmask < self.p_unmask) * mask
        masked_tokens[unmask] = all_tokens[unmask]
        # assign random tokens
        random_mask = torch.rand_like(all_tokens, dtype=torch.float32)
        random_mask = (random_mask < self.p_random_mask) * mask
        random_indices = torch.randint_like(all_tokens, 1, len(self.tokenizer))
        masked_tokens[random_mask] = random_indices[random_mask]
        return masked_tokens, mask

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr,  # weight_decay=self.weight_decay
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
        p.add_argument('--dims', default=100, type=int)
        p.add_argument('--ff_dims', default=512, type=int)
        p.add_argument('--lr', default=0.001, type=float)
        p.add_argument('--weight_decay', default=1e-4, type=float)
        p.add_argument('--layers', default=4, type=int)
        p.add_argument('--dropout', default=0.1, type=float)
        p.add_argument('--batch_size', default=32, type=int)
        p.add_argument('--lr_warmup_steps', default=3, type=int)
        p.add_argument('--update_embedding', action='store_true')
        return p


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('tokenizer_name')
    p.add_argument('--max_seq_len', default=256, type=int)

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
        num_tokens=num_tokens, mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        **vars(args)
    )

    print('Set up Trainer')
    model_checkpoint = ModelCheckpoint()
    callbacks = [model_checkpoint, EchoMLMTextBatch(p_print=0.1)]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks
    )
    trainer.fit(model, dm)
