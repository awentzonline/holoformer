import copy
from functools import partial

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F

from holoformer.datasets.glue import GLUEDataModule
from .hrr_xformer_masked import HoloformerMLM


class HoloformerGLUE(pl.LightningModule):
    """Holoformer using pre-trained masked language model to solve GLUE"""
    def __init__(self, encoder, num_labels, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.classifier_logits = nn.Sequential(
            nn.Linear(encoder.data_dims, encoder.data_dims * 4),
            nn.LeakyReLU(),
            nn.Linear(encoder.data_dims * 4, num_labels),
        )

    def forward(self, x, **kwargs):
        encoded = self.encode_sequence(x)
        logits = self.classifier_logits(encoded[:, 0])
        return logits

    def _shared_step(self, data, batch_idx):
        p_labels = self(data['input_ids'])
        loss = F.cross_entropy(p_labels, data['labels'])

        metrics = dict(
            loss=loss,
        )
        losses = dict(
            loss=loss
        )
        return metrics, losses

    @classmethod
    def add_argparse_args(self, p):
        return super().add_argparse_args(p)


def parse_csv_arg(type_):
    def f(v):
        return tuple(map(type_, v.split(',')))
    return f


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('task_name')
    p.add_argument('tokenizer_name')
    p.add_argument('encoder')
    p.add_argument('--max_seq_len', default=128, type=int)

    p = HoloformerGLUE.add_argparse_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args = p.parse_args()

    print('Building DataModule')
    dm = GLUEDataModule(**vars(args))
    dm.setup('fit')
    num_tokens = len(dm.tokenizer)

    print('Building model')
    mask_token_id, pad_token_id = dm.tokenizer.convert_tokens_to_ids([
        '[MASK]', '[PAD]'
    ])
    encoder = HoloformerMLM.load_from_checkpoint(args.encoder)
    model = HoloformerGLUE(
        encoder,
        num_labels=dm.num_labels,
        **vars(args)
    )

    print('Set up Trainer')
    model_checkpoint = ModelCheckpoint()
    callbacks = [model_checkpoint]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks
    )
    trainer.fit(model, dm)
