import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class AutoRegressiveTeacherTextBatch(Callback):
    def __init__(self, p_print=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_print = p_print

    @torch.no_grad()
    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() > self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)
        recon_tokens = model(all_tokens)
        if isinstance(recon_tokens, tuple):
            recon_tokens = recon_tokens[0]
        recon_tokens = model.embeddings_to_ids(recon_tokens)[0]
        print(model.tokenizer.decode(recon_tokens))


class AutoRegressiveTextBatch(Callback):
    def __init__(self, p_print=0.01, prompt_len=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_print = p_print
        self.prompt_len = prompt_len

    @torch.no_grad()
    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() < self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)[0:1, :self.prompt_len]
        recon_tokens = model.generate(all_tokens)
        print(model.tokenizer.decode(recon_tokens[0]))
