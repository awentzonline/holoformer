import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
import torch.nn.functional as F
from transformers.optimization import get_linear_schedule_with_warmup

from holoformer.datasets.glue import GLUEDataModule
from .hrr_xformer_masked import HoloformerMLM


class HoloformerGLUE(pl.LightningModule):
    """Holoformer using pre-trained masked language model to solve GLUE"""
    def __init__(
        self, encoder, num_labels, lr=0.0001, weight_decay=0.001,
        warmup_steps=3, opt_betas=(0.9, 0.95), total_steps=1000000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore='encoder')
        self.encoder = encoder
        self.num_labels = num_labels
        self.lr = lr
        self.classifier_logits = nn.Sequential(
            nn.Linear(encoder.data_dims, encoder.data_dims * 4),
            nn.LeakyReLU(),
            nn.Linear(encoder.data_dims * 4, num_labels),
        )
        self.classifier_logits.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('linear')
            )
            nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        encoded = self.encoder.encode_sequence(x)
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
            hasattr(self.encoder, 'positional_encoding') and
            hasattr(self.encoder.positional_encoding, 'embeddings')
        ):
            no_decay.add('encoder.positional_encoding.embeddings')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.lr, betas=self.hparams.opt_betas)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @classmethod
    def add_argparse_args(self, p):
        p.add_argument('--lr', default=0.0001, type=float)
        p.add_argument('--weight_decay', default=0.01, type=float)
        p.add_argument('--batch_size', default=32, type=int)
        p.add_argument('--opt_betas', default=(0.9, 0.999), type=parse_csv_arg(float))
        p.add_argument('--warmup_steps', default=0.1, type=float)
        return p


def parse_csv_arg(type_):
    def f(v):
        return tuple(map(type_, v.split(',')))
    return f


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('task_name')
    p.add_argument('tokenizer_name')
    p.add_argument('encoder_path')
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
    encoder = HoloformerMLM.load_from_checkpoint(args.encoder_path)
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
