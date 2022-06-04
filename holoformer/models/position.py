import math

import torch
from torch import nn

from holoformer.models import hrr


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class HolographicPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.embeddings = nn.Parameter(
            hrr.init((1, max_len, d_model))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        embeddings = self.embeddings[:, :seq_len]
        #embeddings = hrr.unit_projection(embeddings)
        y = hrr.bind(embeddings, x)
        return y

    def loss(self, x):
        seq_len = x.shape[1]
        embeddings = self.embeddings[:, :seq_len]
        return hrr.unit_regularization(embeddings)

    def get_embeddings(self, indices):
        return self.embeddings[:, indices]

    def unbind_positions(self, x):
        batch_size, seq_len = x.shape[:2]
        embeddings = self.embeddings[:, :seq_len]
        y = hrr.unbind(x, embeddings)
        return y

    def unbind_reduced(self, s):
        s = s.unsqueeze(1)
        y = hrr.unbind(s, self.embeddings)
        return y
