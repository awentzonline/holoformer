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
        recon_tokens = model(masked_tokens)
        recon_tokens = model.positional_encoding.unbind_positions(recon_tokens)
        recon_tokens = recon_tokens / (torch.norm(recon_tokens, dim=-1, keepdim=True) + 1e-8)
        recon_tokens = model.lookup_embeddings(recon_tokens[0])
        recon_tokens = torch.argmax(recon_tokens, dim=-1)
        print(model.tokenizer.decode(recon_tokens))


class EchoMLMFullyReducedTextBatch(Callback):
    def __init__(self, p_print=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_print = p_print

    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() > self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)
        masked_tokens, _ = model.mask_tokens(all_tokens)
        recon_tokens = model(masked_tokens)
        recon_tokens = model.extract_sequence(recon_tokens)[0]
        print(model.tokenizer.decode(recon_tokens))


class EchoFullyReducedTextBatch(Callback):
    def __init__(self, p_print=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_print = p_print

    def on_train_batch_end(self, trainer, model, outputs, batch, *args, **kwargs):
        if np.random.uniform() > self.p_print:
            return

        all_tokens = batch['input_ids'].to(model.device)

        # reconstructing embedding works
        # recon_tokens = model.embedding(all_tokens)
        # recon_tokens = model.extract_tokens(recon_tokens)[0]

        # bind/unbind positions reconstruct embedding works
        # recon_tokens = model.embed_sequence(all_tokens)
        # recon_tokens = model.positional_encoding.unbind_positions(recon_tokens)
        # recon_tokens = model.extract_tokens(recon_tokens)[0]

        # fully reduced:
        recon_tokens = model(all_tokens)
        recon_tokens = model.extract_sequence(recon_tokens)[0]

        print(model.tokenizer.decode(recon_tokens))
