import copy

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
import torch.nn.functional as F

from holoformer.datasets.hf_datasets import HfDatasetDataModule
from holoformer.models import hrr
from holoformer.models.position import PositionalEncoding


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class HolographicMixer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(dims, dims),
            nn.LayerNorm(dims),
        )
        self.key = nn.Sequential(
            nn.Linear(dims, dims),
            nn.LayerNorm(dims),
        )

    def forward(self, x):
        """
        x.shape ~= (batch, sequence, embedding)
        """
        query = self.query(x)
        keys = self.key(x)
        x = hrr.bind(keys, x)
        s = x.sum(dim=1, keepdim=True)
        values = hrr.unbind(s, query)
        return x + values


class HoloformerFeedForward(nn.Module):
    def __init__(self, dims, ff_dims, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims, ff_dims),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(ff_dims, dims),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class HoloformerEncoderLayer(nn.Module):
    def __init__(self, dims, ff_dims, dropout, activation=nn.ReLU, **kwargs):
        super().__init__()
        self.mixer = nn.Sequential(
            HolographicMixer(dims),
            nn.Dropout(dropout),
        )
        self.feed_forward = nn.Sequential(
            HoloformerFeedForward(
                dims, ff_dims, dropout=dropout, activation=activation
            ),
        )
        self.norm0 = nn.LayerNorm(dims)
        self.norm1 = nn.LayerNorm(dims)

    def forward(self, x, **kwargs):
        x = self.norm0(x + self.mixer(x))
        x = self.norm1(x + self.feed_forward(x))
        return x


class Holoformer(pl.LightningModule):
    def __init__(self, num_tokens, data_dims=100, ff_dims=512, layers=4,
                 lr=0.001, weight_decay=1e-5, dropout=0.1,
                 activation=nn.ReLU, pad_token_id=0, mask_token_id=1,
                 update_embedding=False,
                 vanilla=False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.data_dims = data_dims
        self.ff_dims = ff_dims
        self.mask_token_id = mask_token_id
        self.embedding = nn.Embedding(
            num_tokens, data_dims, padding_idx=pad_token_id,
        )
        self.embedding.weight.data = hrr.init(self.embedding.weight.data.shape)
        self.embedding.requires_grad_(update_embedding)

        self.positional_encoding = PositionalEncoding(data_dims)
        self.output_token = nn.Linear(
            data_dims, num_tokens,
        )
        if vanilla:
            transformer_layer = HoloformerEncoderLayer(
                data_dims, ff_dims, dropout=dropout, activation=activation
            )
        else:
            transformer_layer = nn.TransformerEncoderLayer(
                data_dims, 1, dim_feedforward=ff_dims,
                dropout=dropout
            )
        self.encoder = nn.TransformerEncoder(transformer_layer, layers)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, **kwargs):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        embedded = hrr.unit_projection(embedded)
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
        all_tokens = data['input_ids']
        mask = torch.rand(*all_tokens.shape, device=self.device) < 0.15
        masked_tokens = all_tokens.clone()
        masked_tokens[mask] = self.mask_token_id

        recon_tokens = self(masked_tokens)
        # num_tokens = recon_tokens.shape[-1]
        # recon_tokens = torch.masked_select(recon_tokens, mask.unsqueeze(-1))
        # recon_tokens = recon_tokens.reshape(-1, num_tokens)
        # original_tokens = torch.masked_select(all_tokens, mask)
        # loss = F.cross_entropy(
        #     recon_tokens, original_tokens
        # )
        loss = F.cross_entropy(
            recon_tokens.permute(0, 2, 1), all_tokens
        )

        metrics = dict(
            loss=loss,
        )
        losses = dict(
            loss=loss,
        )
        return metrics, losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr,  # weight_decay=self.weight_decay
        )
        return optimizer

    @classmethod
    def add_argparse_args(self, p):
        p.add_argument('--dims', default=100, type=int)
        p.add_argument('--ff_dims', default=512, type=int)
        p.add_argument('--lr', default=0.001, type=float)
        p.add_argument('--weight_decay', default=1e-4, type=float)
        p.add_argument('--layers', default=4, type=int)
        p.add_argument('--dropout', default=0.1, type=float)
        p.add_argument('--vanilla', action='store_true')
        return p


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('tokenizer')

    p = Holoformer.add_argparse_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    print('Building DataModule')
    dm = HfDatasetDataModule(args.dataset)
    dm.setup('fit')
    num_tokens = len(dm.tokenizer)

    print('Building model')
    mask_token_id, pad_token_id = dm.tokenizer.convert_tokens_to_ids([
        '[MASK]', '[PAD]'
    ])
    model = Holoformer(
        num_tokens=num_tokens, mask_token_id=mask_token_id,
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
