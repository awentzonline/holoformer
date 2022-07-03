import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class AutoRegressiveTextBatch(Callback):
    def __init__(
        self, p_print=0.01, prompt_len=10, top_k=0., top_p=0.9, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p_print = p_print
        self.prompt_len = prompt_len
        self.top_k = top_k
        self.top_p = top_p

    @torch.no_grad()
    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() > self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)[0:1, :self.prompt_len]
        recon_tokens = model.generate(
            all_tokens, top_k=self.top_k, top_p=self.top_p
        )
        print(model.tokenizer.decode(recon_tokens[0]))
