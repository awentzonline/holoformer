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
from .hrr_loss import hrr_xml_loss
from .hrr_xformer_ar import HoloformerAR


class Dot(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, x):
        y = torch.einsum('bse,ne->bsn', x, self.weights)
        return y


class HoloformerARXML(HoloformerAR):
    """Auto-regressive holoformer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_token = Dot(self.embedding.weight)
        self.output_token.requires_grad_(self.update_embedding)

    def _shared_step(self, data, batch_idx):
        all_tokens = data['input_ids'].clone()
        p_tokens = self.encode_sequence(all_tokens[:, :-1])
        target_tokens = all_tokens[:, 1:]
        dims = p_tokens.shape[-1]
        recon_loss = hrr_xml_loss(
            p_tokens.reshape(-1, dims), target_tokens.reshape(-1),
            self.embedding
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


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('tokenizer_name')
    p.add_argument('--max_seq_len', default=256, type=int)
    p.add_argument('--p_print', default=0.01, type=float)

    p = HoloformerARXML.add_argparse_args(p)
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
    model = HoloformerARXML(
        tokenizer=dm.tokenizer,
        num_tokens=num_tokens,
        pad_token_id=pad_token_id,
        **vars(args)
    )

    print('Set up Trainer')
    model_checkpoint = ModelCheckpoint()
    callbacks = [model_checkpoint, AutoRegressiveTextBatch(p_print=args.p_print)]
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks
    )
    trainer.fit(model, dm)
