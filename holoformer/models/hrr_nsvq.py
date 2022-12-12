"""
Adapted from: https://gitlab.com/speech-interaction-technology-aalto-university/nsvq
"""
import torch
from torch.distributions import Normal
from torch import nn

from holoformer.models import hrr


class HRRNSVQ(nn.Module):
    def __init__(self, num_embeddings, dims, eps=1e-12):
        super().__init__()

        self.eps = eps
        codebooks = hrr.init((num_embeddings, dims))
        self.register_buffer('codebooks', codebooks)

    def forward(self, x):
        distances = (
            torch.sum(x ** 2, dim=1, keepdim=True) -
            2 * (torch.matmul(x, self.codebooks.t())) +
            torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )

        min_indices = torch.argmin(distances, dim=1)

        best_entries = self.codebooks[min_indices]
        random_vector = Normal(0, 1).sample(x.shape).to(x.device)

        norm_best_entries = (x - best_entries).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (norm_best_entries / norm_random_vector + self.eps) * random_vector

        quantized_input = x + vq_error

        return quantized_input


class HRRNSVQMultiLabel(nn.Module):
    def __init__(self, num_embeddings, dims, eps=1e-12):
        super().__init__()

        self.eps = eps
        codebooks = hrr.init((num_embeddings, dims))
        codebooks = codebooks / torch.linalg.norm(codebooks, dim=-1, keepdim=True)
        self.register_buffer('codebooks', codebooks)

    def forward(self, x, threshold=0.3):
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        similarities = torch.matmul(x / (x_norm + self.eps), self.codebooks.t())
        #print(similarities.min(), similarities.mean(), similarities.max())
        is_similar = (similarities > threshold).float()
        #print(is_similar.mean(), is_similar.sum())
        best_entries = torch.einsum('id, bsi -> bsd', self.codebooks, is_similar)

        random_vector = Normal(0, 1).sample(x.shape).to(x.device)

        norm_best_entries = (x - best_entries).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (norm_best_entries / (norm_random_vector + self.eps)) * random_vector

        quantized_input = x + vq_error

        return quantized_input
