import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class EchoMLMTextBatch(Callback):
    def __init__(self, p_print=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_print = p_print

    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() > self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)
        masked_tokens, _ = model.mask_tokens(all_tokens)
        recon_tokens = model(masked_tokens)[0]
        recon_tokens = torch.argmax(recon_tokens, dim=-1)
        print(model.tokenizer.decode(recon_tokens))


class EchoMLMReducedTextBatch(Callback):
    def __init__(self, p_print=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_print = p_print

    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() > self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)
        masked_tokens, _ = model.mask_tokens(all_tokens)
        recon_tokens = model(masked_tokens)[0]
        # TODO: decode from reduced rep?
        recon_tokens = torch.argmax(recon_tokens, dim=-1)
        print(model.tokenizer.decode(recon_tokens))
