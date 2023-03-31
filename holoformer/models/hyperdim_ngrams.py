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


class HyperdimensionalNGramPretraining(pl.LightningModule):
    def __init__(self, tokenizer, data_dims=100, hidden_dims=512, layers=1,
                 ngram_size=3,
                 lr=0.001, weight_decay=1e-5, dropout=0.1,
                 activation=nn.ReLU, pad_token_id=0, mask_token_id=1,
                 lr_warmup_steps=3,
                 emb_loss_w=0.1,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.data_dims = data_dims
        self.ngram_size = ngram_size
        self.hyper_dims = ngram_size * len(tokenizer)
        self.hidden_dims = hidden_dims
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.emb_loss_w = emb_loss_w

        self.embedding = nn.Embedding(
            len(tokenizer), data_dims, padding_idx=pad_token_id,
        )

        self.output_tokens = nn.Linear(
            data_dims, self.hyper_dims, bias=False
        )
        self.output_tokens.requires_grad_(False)

        self.register_buffer(
            'output_token_offsets',
            torch.full((1, ngram_size), len(self.tokenizer)) *
            torch.arange(ngram_size)
        )

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, **kwargs):
        embedded = self.embedding(x)
        encoded, hidden = self.rnn(embedded)
        return self.output_token(encoded), hidden

    def embeddings_to_ids(self, emb):
        batch_size, seq_len = emb.shape[:2]

        batch_ngram_embedding_keys = self.ngram_embeddings.repeat(batch_size, seq_len, 1)
        p_present = hrr.unbind(emb, batch_ngram_embedding_keys)
        p_present = p_present / (torch.norm(p_present, dim=-1, keepdim=True) + 1e-8)
        all_embeddings = self.embedding.weight.data.unsqueeze(0)
        all_embeddings = all_embeddings / (torch.norm(all_embeddings, dim=-1, keepdim=True) + 1e-8)
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

        token_ngrams = torch.split(all_tokens, self.ngram_size, dim=1)
        # TODO: pad the last ngram where needed
        summed_embedded_ngrams = []
        ngram_target_indices = []
        for i in range(len(token_ngrams) - 1):
            ngram = token_ngrams[i]
            summed_embedded_ngrams.append(
                self.embedding(ngram).sum(1)
            )
            ngram_target_indices.append(
                 ngram + self.output_token_offsets
            )
        summed_embedded_ngrams = torch.stack(summed_embedded_ngrams)
        ngram_target_indices = torch.stack(ngram_target_indices)
        recon_tokens = self.output_tokens(summed_embedded_ngrams)
        dense_shape = recon_tokens.shape[:2] + (self.hyper_dims,)
        dense_targets = torch.zeros(dense_shape, device=self.device, dtype=torch.long)
        dense_targets = dense_targets.view(-1, self.hyper_dims)
        ngram_target_indices = ngram_target_indices.view(-1, self.ngram_size)
        dense_targets.scatter_(1, ngram_target_indices, torch.ones_like(ngram_target_indices))
        recon_tokens = recon_tokens.view(-1, self.hyper_dims)

        loss = F.binary_cross_entropy_with_logits(recon_tokens, dense_targets.float())
        metrics = dict(
            loss=loss,
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
        p.add_argument('--data_dims', default=100, type=int)
        p.add_argument('--hidden_dims', default=200, type=int)
        p.add_argument('--lr', default=0.001, type=float)
        p.add_argument('--weight_decay', default=1e-4, type=float)
        p.add_argument('--layers', default=4, type=int)
        p.add_argument('--dropout', default=0.1, type=float)
        p.add_argument('--batch_size', default=32, type=int)
        p.add_argument('--lr_warmup_steps', default=3, type=int)
        p.add_argument('--emb_loss_w', default=0.1, type=float)
        p.add_argument('--update_embedding', action='store_true')
        return p


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('tokenizer_name')
    p.add_argument('--max_seq_len', default=256, type=int)

    p = HyperdimensionalNGramPretraining.add_argparse_args(p)
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
    model = HyperdimensionalNGramPretraining(
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
