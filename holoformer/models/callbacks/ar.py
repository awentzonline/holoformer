import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class AutoRegressiveTextBatch(Callback):
    def __init__(self, p_print=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_print = p_print

    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() > self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)
        recon_tokens, _ = model(all_tokens)
        recon_tokens = model.embeddings_to_ids(recon_tokens)[0]
        print(model.tokenizer.decode(recon_tokens))
