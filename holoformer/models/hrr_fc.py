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
from holoformer.models.callbacks.mlm import EchoMLMFullyReducedTextBatch
from holoformer.models.position import (
    HolographicPositionalEncoding, PositionalEncoding
)


class HoloEncoderFC(nn.Module):
    def __init__(self, dims, ff_dims, num_layers, dropout, activation=nn.ReLU,
                 **kwargs):
        super().__init__()
        layers = [
            nn.Linear(dims, ff_dims),
            activation(),
        ]
        for _ in range(num_layers):
            layers += [
                nn.Linear(ff_dims, ff_dims),
                activation(),
            ]
        layers += [
            nn.Linear(ff_dims, dims)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return self.net(x)


class HoloFCMLM(pl.LightningModule):
    def __init__(self, tokenizer, data_dims=100, ff_dims=512, layers=4,
                 lr=0.001, weight_decay=1e-5, dropout=0.1,
                 activation=nn.ReLU, pad_token_id=0, mask_token_id=1,
                 update_embedding=True, p_mask=0.15, p_random_mask=0.2,
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

        self.encoder = HoloEncoderFC(
            data_dims, ff_dims, layers, dropout=dropout, activation=activation
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.hrr_dist = Normal(0., 1. / data_dims)
        self.update_embedding = update_embedding

    def forward(self, x, **kwargs):
        embedded = self.embed_sequence(x)
        embedded = embedded.sum(1)
        y = self.encoder(embedded)
        return y

    def embed_sequence(self, x):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        return embedded

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
        recon_tokens_hrr = self(masked_tokens)
        target_tokens = self.embed_sequence(all_tokens)

        all_embeddings = self.embedding.weight.sum(0, keepdim=True)
        all_positions = self.positional_encoding.embeddings.sum(1)
        all_emb_pos = hrr.bind(all_embeddings, all_positions)

        absent_emb_pos = all_emb_pos - target_tokens.sum(1)
        absent_emb_pos = absent_emb_pos / (torch.linalg.norm(absent_emb_pos, dim=-1, keepdim=True) + 1e-8)
        recon_tokens_hrr = recon_tokens_hrr / (torch.linalg.norm(recon_tokens_hrr, dim=-1, keepdim=True) + 1e-8)
        target_tokens = target_tokens / (torch.linalg.norm(target_tokens, dim=-1, keepdim=True) + 1e-8)

        present_loss = (1 - torch.matmul(target_tokens, recon_tokens_hrr.unsqueeze(-1))).mean()
        absent_loss = torch.matmul(target_tokens, absent_emb_pos.unsqueeze(-1)).mean()

        # regularize embeddings
        embedding_loss = torch.tensor(0, device=self.device)
        positional_loss = torch.tensor(0, device=self.device)
        if self.update_embedding:
            embedding_loss = hrr.unit_regularization(self.embedding.weight).mean()
            positional_loss = self.positional_encoding.loss(all_tokens).mean()

        loss = present_loss + absent_loss + embedding_loss + positional_loss
        metrics = dict(
            loss=loss,
            present_loss=present_loss,
            absent_loss=absent_loss,
            embedding_loss=embedding_loss,
            positional_loss=positional_loss,
        )
        losses = dict(
            loss=loss
        )
        return metrics, losses

    def extract_sequence(self, s):
        tokens_hrr = self.positional_encoding.unbind_reduced(s)
        token_embeddings = self.embedding.weight
        tokens_hrr = tokens_hrr / (torch.linalg.norm(tokens_hrr, dim=-1, keepdim=True) + 1e-8)
        token_embeddings = token_embeddings / (torch.linalg.norm(token_embeddings, dim=-1, keepdim=True) + 1e-8)

        p_tokens = torch.matmul(tokens_hrr, token_embeddings.T.unsqueeze(0))
        p_tokens = p_tokens.argmax(-1)
        return p_tokens

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
            self.parameters(), lr=self.lr,
        )
        return optimizer

    @classmethod
    def add_argparse_args(self, p):
        p.add_argument('--dims', default=100, type=int)
        p.add_argument('--ff_dims', default=200, type=int)
        p.add_argument('--lr', default=0.001, type=float)
        p.add_argument('--weight_decay', default=1e-4, type=float)
        p.add_argument('--layers', default=4, type=int)
        p.add_argument('--dropout', default=0.1, type=float)
        p.add_argument('--batch_size', default=32, type=int)
        p.add_argument('--lr_warmup_steps', default=3, type=int)
        return p


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('tokenizer_name')

    p = HoloFCMLM.add_argparse_args(p)
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
    model = HoloFCMLM(
        tokenizer=dm.tokenizer,
        num_tokens=num_tokens, mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        **vars(args)
    )

    print('Set up Trainer')
    model_checkpoint = ModelCheckpoint()
    callbacks = [model_checkpoint, EchoMLMFullyReducedTextBatch()]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks
    )
    trainer.fit(model, dm)
