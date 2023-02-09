###
# Author: Kai Li
# Date: 2021-06-09 20:24:51
# LastEditors: Kai Li
# LastEditTime: 2021-06-20 19:27:13
###

import torch
import torch.nn as nn
from typing import List
from torch.nn.modules.batchnorm import _BatchNorm


def norm(x, dims: List[int], EPS: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt(var2 + EPS)
    return value


def glob_norm(x, ESP: float = 1e-8):
    dims: List[int] = torch.arange(1, len(x.shape)).tolist()
    return norm(x, dims, ESP)


class MLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)

    def forward(self, x, EPS: float = 1e-8):
        pass


class GlobalLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        value = glob_norm(x, EPS)
        return self.apply_gain_and_bias(value)


class ChannelLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumulateLN(MLayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        batch, channels, time = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=1)
        cnt = torch.arange(
            start=channels, end=channels * (time + 1), step=channels, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class BatchNorm(_BatchNorm):
    """Wrapper class for pytorch BatchNorm1D and BatchNorm2D"""

    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError("expected 4D or 3D input (got {}D input)".format(input.dim()))


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


# Aliases.
gLN = GlobalLN
cLN = ChannelLN
LN = CumulativeLayerNorm
cgLN = CumulateLN
bN = BatchNorm
BatchNorm1d = nn.BatchNorm1d


def get(identifier):
    """Returns a norm class from a string. Returns its input if it
    is callable (already a :class:`._LayerNorm` for example).

    Args:
        identifier (str or Callable or None): the norm identifier.

    Returns:
        :class:`._LayerNorm` or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
