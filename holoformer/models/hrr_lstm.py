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
from holoformer.models.callbacks.ar import AutoRegressiveTextBatch
from holoformer.models.position import (
    HolographicPositionalEncoding, PositionalEncoding
)


class HoloformerLSTM(pl.LightningModule):
    def __init__(self, tokenizer, data_dims=100, hidden_dims=512, layers=1,
                 lr=0.001, weight_decay=1e-5, dropout=0.1,
                 activation=nn.ReLU, pad_token_id=0, mask_token_id=1,
                 update_embedding=True, lr_warmup_steps=3,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.data_dims = data_dims
        self.hidden_dims = hidden_dims
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.register_buffer('presence_embeddings', hrr.init_ortho(
            (2, data_dims)
        ).unsqueeze(1).unsqueeze(1))
        self.embedding = nn.Embedding(
            len(tokenizer), data_dims, padding_idx=pad_token_id,
        )
        self.embedding.weight.data = hrr.init(self.embedding.weight.data.shape)
        self.embedding.requires_grad_(update_embedding)

        self.rnn = nn.LSTM(
            data_dims, hidden_dims, num_layers=layers, dropout=dropout,
            batch_first=True
        )
        self.output_token = nn.Linear(
            hidden_dims, data_dims,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.hrr_dist = Normal(0., 1. / data_dims)
        self.update_embedding = update_embedding

    def forward(self, x, **kwargs):
        embedded = self.embedding(x)
        encoded, hidden = self.rnn(embedded)
        return self.output_token(encoded), hidden

    def embeddings_to_ids(self, emb):
        batch_size, seq_len = emb.shape[:2]
        present_emb = self.presence_embeddings[1]
        present_emb = present_emb.repeat(batch_size, seq_len, 1)
        p_present = hrr.unbind(emb, present_emb)
        p_present = p_present / (torch.norm(p_present, dim=-1, keepdim=True) + 1e-8)
        all_embeddings = self.embedding.weight.data.unsqueeze(0)
        all_embeddings /= (torch.norm(all_embeddings, dim=-1, keepdim=True) + 1e-8)
        cos_present = torch.matmul(
            all_embeddings, p_present.unsqueeze(-2).transpose(-1, -2)
        ).squeeze(-1)
        tokens = cos_present.argmax(-1)
        return tokens

    def decode_embeddings(self, emb):
        ids = self.embeddings_to_ids(emb)
        tokens = self.tokenizer.decode(ids)
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
        recon_embedding, _ = self(all_tokens)
        recon_embedding = recon_embedding[:, :-1]
        batch_size, seq_len = recon_embedding.shape[:2]
        target_embeddings = self.embedding(all_tokens[:, 1:])
        target_embeddings = target_embeddings / (torch.norm(target_embeddings, dim=-1, keepdim=True) + 1e-8)
        # all_embeddings = self.embedding.weight.data.sum(0, keepdim=True)
        # all of the embeddings except the target
        # negative_target_embeddings = all_embeddings.unsqueeze(1).repeat(batch_size, seq_len, 1) - target_embeddings

        absent_emb, present_emb = self.presence_embeddings
        absent_emb = absent_emb.repeat(batch_size, seq_len, 1)
        p_absent = hrr.unbind(recon_embedding, absent_emb)
        p_absent = p_absent / (torch.norm(p_absent, dim=-1, keepdim=True) + 1e-8)
        cos_absent = torch.einsum('bij,bij->bi', target_embeddings, p_absent)
        # cos_absent = torch.matmul(
        #     target_embeddings, p_absent.unsqueeze(1).transpose(-1, -2)
        # ).squeeze(-1)
        J_n = torch.mean(torch.abs(cos_absent))

        present_emb = present_emb.repeat(batch_size, seq_len, 1)
        p_present = hrr.unbind(recon_embedding, present_emb)
        p_present = p_present / (torch.norm(p_present, dim=-1, keepdim=True) + 1e-8)
        cos_present = torch.einsum('bij,bij->bi', target_embeddings, p_present)
        # cos_present = torch.matmul(
        #     target_embeddings.unsqueeze(-1), p_present.unsqueeze(-2).transpose(-1, -2)
        # ).squeeze(-1)
        J_p = torch.mean(1 - torch.abs(cos_present))

        hrr_loss = J_n + J_p

        embedding_loss = torch.tensor(0, device=self.device)
        if self.update_embedding:
            embedding_loss = hrr.unit_regularization(self.embedding.weight).mean()

        loss = hrr_loss + embedding_loss
        metrics = dict(
            loss=loss,
            recon_loss=hrr_loss,
            embedding_loss=embedding_loss
        )
        losses = dict(
            loss=loss
        )
        return metrics, losses

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
        p.add_argument('--hidden_dims', default=200, type=int)
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
    p.add_argument('--max_seq_len', default=256, type=int)

    p = HoloformerLSTM.add_argparse_args(p)
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
    model = HoloformerLSTM(
        tokenizer=dm.tokenizer,
        num_tokens=num_tokens, mask_token_id=mask_token_id,
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
