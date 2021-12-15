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
from holoformer.models.callbacks.mlm import EchoFullyReducedTextBatch
from holoformer.models.position import (
    HolographicPositionalEncoding, PositionalEncoding
)


class HoloReduceExpand(pl.LightningModule):
    """
    Learn token and position embeddings which allow for reducing
    a sequence into a single vector and then re-expanding to
    its original form.
    """
    def __init__(self, tokenizer, dims=100,
                 lr=0.001, weight_decay=1e-5, dropout=0.1,
                 pad_token_id=0,
                 update_embedding=True,
                 max_seq_len=256,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.data_dims = dims
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            len(tokenizer), dims, padding_idx=pad_token_id,
        )
        self.embedding.weight.data = hrr.init(self.embedding.weight.data.shape)
        self.embedding.requires_grad_(update_embedding)

        self.positional_encoding = HolographicPositionalEncoding(dims, max_len=max_seq_len)
        self.positional_encoding.requires_grad_(update_embedding)

        self.lr = lr
        self.hrr_dist = Normal(0., 1. / dims)
        self.update_embedding = update_embedding

    def forward(self, x, **kwargs):
        embedded = self.embed_sequence(x)
        embedded = embedded.sum(1)
        return embedded

    def embed_sequence(self, x):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        # print(torch.norm(embedded, dim=-1))
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
        embedded = self.embed_sequence(all_tokens)
        recon_tokens_hrr = embedded.sum(1)
        target_tokens = embedded

        all_embeddings = self.embedding.weight.sum(0, keepdim=True)
        all_positions = self.positional_encoding.embeddings.sum(1)
        all_emb_pos = hrr.bind(all_embeddings, all_positions)

        absent_emb_pos = all_emb_pos - target_tokens.sum(1)
        absent_emb_pos = absent_emb_pos / (torch.linalg.norm(absent_emb_pos, dim=-1, keepdim=True) + 1e-8)
        recon_tokens_hrr = recon_tokens_hrr / (torch.linalg.norm(recon_tokens_hrr, dim=-1, keepdim=True) + 1e-8)
        target_tokens = target_tokens / (torch.linalg.norm(target_tokens, dim=-1, keepdim=True) + 1e-8)

        present_loss = (1 - torch.matmul(target_tokens, recon_tokens_hrr.unsqueeze(-1)))
        present_loss = present_loss.squeeze(-1).abs().sum(1).mean()
        absent_loss = torch.matmul(target_tokens, absent_emb_pos.unsqueeze(-1))
        absent_loss = absent_loss.abs().squeeze(-1).sum(1).mean()

        # regularize embeddings
        embedding_loss = torch.tensor(0, device=self.device)
        positional_loss = torch.tensor(0, device=self.device)
        embedding_uniqueness_loss = torch.tensor(0, device=self.device)
        if self.update_embedding:
            # embedding_loss = torch.abs(1 - torch.linalg.norm(self.embedding.weight, dim=-1)).mean()
            # positional_loss = torch.abs(1 - torch.linalg.norm(self.positional_encoding.embeddings, dim=-1)).mean()
            embedding_loss = hrr.unit_regularization(self.embedding.weight).mean()
            positional_loss = self.positional_encoding.loss(all_tokens).mean()

            unique_tokens = torch.unique(
                all_tokens,
                sorted=False, return_inverse=False, return_counts=False
            )
            unique_embeddings = self.embedding(unique_tokens)
            embedding_bt_loss = unique_nonzero_loss(unique_embeddings).mean()
            position_bt_loss = unique_nonzero_loss(self.positional_encoding.embeddings[0]).mean()

        loss = present_loss + absent_loss + embedding_loss + positional_loss + embedding_bt_loss + position_bt_loss
        metrics = dict(
            loss=loss,
            present_loss=present_loss,
            absent_loss=absent_loss,
            embedding_loss=embedding_loss,
            positional_loss=positional_loss,
            embedding_uniqueness_loss=embedding_uniqueness_loss,
        )
        losses = dict(
            loss=loss
        )
        return metrics, losses

    def extract_sequence(self, s):
        tokens_hrr = self.positional_encoding.unbind_reduced(s)
        return self.extract_tokens(tokens_hrr)

    def extract_tokens(self, tokens_hrr):
        token_embeddings = self.embedding.weight
        tokens_hrr = tokens_hrr / (torch.linalg.norm(tokens_hrr, dim=-1, keepdim=True) + 1e-8)
        token_embeddings = token_embeddings / (torch.linalg.norm(token_embeddings, dim=-1, keepdim=True) + 1e-8)

        p_tokens = torch.matmul(tokens_hrr, token_embeddings.T.unsqueeze(0))
        p_tokens = p_tokens.argmax(-1)
        return p_tokens

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
        p.add_argument('--max_seq_len', default=20, type=int)

        return p


def unique_nonzero_loss(x, lambda_=1.):
    c = torch.matmul(x, x.T)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    return on_diag + lambda_ * off_diag


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('tokenizer_name')

    p = HoloReduceExpand.add_argparse_args(p)
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
    model = HoloReduceExpand(
        tokenizer=dm.tokenizer,
        num_tokens=num_tokens, mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        **vars(args)
    )

    print('Set up Trainer')
    model_checkpoint = ModelCheckpoint()
    callbacks = [model_checkpoint, EchoFullyReducedTextBatch()]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks
    )
    trainer.fit(model, dm)
