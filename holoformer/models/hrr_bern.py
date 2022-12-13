"""
Adapted from: https://gitlab.com/speech-interaction-technology-aalto-university/nsvq
"""
import torch
from torch.distributions import Bernoulli
from torch import nn

from holoformer.models import hrr


class HRRBernoulliSampler(nn.Module):
    def __init__(self, num_embeddings, dims, eps=1e-12):
        super().__init__()
        self.eps = eps
        codebooks = hrr.init((num_embeddings, dims))
        self.register_buffer('codebooks', codebooks)

    def forward(self, x, norm_x=False, norm_vecs=True, norm_both=False):
        vecs = self.codebooks
        if norm_x or norm_both:
            x = x / torch.linalg.norm(x, dim=-1, keepdims=True)
        if norm_vecs or norm_both:
            vecs = vecs / torch.linalg.norm(vecs, dim=-1, keepdims=True)
        probs = torch.matmul(x, vecs.T).abs()
        dist = Bernoulli(probs=probs)
        x_sample = dist.sample()
        ste_probs = (x_sample - probs).detach() + probs
        return ste_probs
