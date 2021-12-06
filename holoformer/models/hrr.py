"""
HHR ops from https://arxiv.org/pdf/2109.02157.pdf
"""
import torch
from torch.fft import fft, ifft


def bind(a, b):
    return torch.real(ifft(torch.multiply(fft(a), fft(b))))

def unbind(s, a):
    return bind(s, inverse(a))

def inverse(a):
    a = torch.flip(a, dims=[-1])
    return torch.roll(a, 1, dims=-1)

def unit_projection(a, eps=1e-5):
    a_hat = fft(a)
    a_hat = a_hat / (a_hat.abs() + eps)
    return torch.real(ifft(a_hat))

def init(shape):
    a = torch.randn(*shape) / shape[-1]
    return unit_projection(a)
