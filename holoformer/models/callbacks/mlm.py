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
        mask = torch.rand(*all_tokens.shape, device=model.device) < 0.15
        masked_tokens = all_tokens.clone()
        masked_tokens[mask] = model.mask_token_id
        recon_tokens = model(masked_tokens)[0]
        recon_tokens = torch.argmax(recon_tokens, dim=-1)
        print(model.tokenizer.decode(recon_tokens))
