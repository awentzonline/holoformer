"""
HHR ops from https://arxiv.org/pdf/2109.02157.pdf
"""
import torch
from torch.distributions import Normal
#from torch.fft import fft, ifft


def fft(x):
    return torch.fft.fft(x, norm=None)


def ifft(x):
    return torch.fft.ifft(x, norm=None)


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


def unit_regularization(v):
    v_hat = fft(v)
    v_hat = v_hat * v_hat.abs()
    x = torch.real(ifft(v_hat))
    dist = Normal(0., 1. / v.shape[-1])
    nlp = -dist.log_prob(x)
    return nlp
