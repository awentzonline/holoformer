import torch
from pytorch_lightning.callbacks import Callback


class EchoMLMTextBatch(Callback):
    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        all_tokens = batch['input_ids']
        mask = torch.rand(*all_tokens.shape, device=model.device) < 0.15
        masked_tokens = all_tokens.clone()
        masked_tokens[mask] = model.mask_token_id
        recon_tokens = model(masked_tokens)[0]
        print(recon_tokens.shape)
        recon_tokens = torch.argmax(recon_tokens, dim=-1)
        print(recon_tokens.shape)
        print(model.tokenizer.decode(recon_tokens))
