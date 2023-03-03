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


# def unit_projection(a, eps=1e-5):
#     a_hat = fft(a)
#     a_hat = a_hat / (a_hat.abs() + eps)
#     return torch.real(ifft(a_hat))


def unit_projection(x):
    c = fft(x)
    c_ish = c / torch.norm(c, dim=-1, keepdim=True)
    output = ifft(c_ish)
    return torch.real(output)


def init(shape):
    a = torch.randn(*shape) / shape[-1]
    return unit_projection(a)


def init_ortho(shape):
    """
    Generate n vectors of size dims that are orthogonal to each other.
    """
    num_vectors, dims = shape
    # Intializing class vectors.
    vecs = torch.randn(dims, num_vectors, dtype=torch.float)

    # Using QR decomposition to get orthogonal vectors.
    vecs, _ = torch.qr(vecs)
    vecs = vecs.t()
    vecs = vecs / torch.norm(vecs, dim=-1, keepdim=True)
    return vecs


def unit_regularization(v):
    v_hat = fft(v)
    v_hat = v_hat * torch.norm(v_hat, dim=-1, keepdim=True)
    x = torch.real(ifft(v_hat))
    dist = Normal(0., 1. / v.shape[-1])
    nlp = -dist.log_prob(x)
    return nlp


# @torch.jit.script
def key_value_query(
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
    causal: bool = True, norm: bool = False
):
    """
    Since F(x) + F(y) = F(x + y) we can avoid a couple of fft/ifft calls
    when calculating the reduced key/value vectors and unbinding the
    query values. Also, norm(fft(inv(q))) == fft(inv(norm(q))) so
    this is also kept in the frequency domain as opposed to using
    the `unit_projection` function which is ifft(norm(fft(q))).
    """
    k, v, q = fft(k), fft(v), fft(inverse(q))
    if norm:
        k = unit_projection(k, dim=-1)
        q = unit_projection(q, dim=-1)
        v = unit_projection(v, dim=-1)
        # k = F.normalize(k, dim=-1)
        # q = F.normalize(q, dim=-1)
        # v = F.normalize(v, dim=-1)
    kv = torch.multiply(k, v)
    if causal:
        r = kv.cumsum(dim=1)
    else:
        r = kv.sum(dim=1, keepdim=True)
    # unbind values for each query
    qv = torch.real(ifft(torch.multiply(r, q)))
    return qv
