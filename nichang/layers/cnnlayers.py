###
# Author: Kai Li
# Date: 2021-06-10 10:21:13
# LastEditors: Kai Li
# LastEditTime: 2021-09-13 20:30:22
###

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import normalizations, activations


class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., : -self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        hid_chan,
        skip_out_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = normalizations.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride=1,
        groups=1,
        dilation=1,
        padding=0,
        norm_type="gLN",
        act_type="prelu",
    ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv1d(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=True,
            groups=groups,
        )
        self.norm = normalizations.get(norm_type)(out_chan)
        self.act = activations.get(act_type)()

    def forward(self, x):
        output = self.conv(x)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride=1,
        groups=1,
        dilation=1,
        padding=0,
        norm_type="gLN",
    ):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel_size, stride, padding, dilation, bias=True, groups=groups
        )
        self.norm = normalizations.get(norm_type)(out_chan)

    def forward(self, x):
        output = self.conv(x)
        return self.norm(output)

class AudioSubnetwork(nn.Module):
    def __init__(
        self, in_chan=128, out_chan=512, upsampling_depth=4, norm_type="gLN", act_type="prelu"
    ):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            groups=1,
            dilation=1,
            padding=0,
            norm_type=norm_type,
            act_type=act_type,
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(
            ConvNorm(
                out_chan,
                out_chan,
                kernel_size=5,
                stride=1,
                groups=out_chan,
                dilation=1,
                padding=((5 - 1) // 2) * 1,
                norm_type=norm_type,
            )
        )
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                ConvNorm(
                    out_chan,
                    out_chan,
                    kernel_size=5,
                    stride=2,
                    groups=out_chan,
                    dilation=1,
                    padding=((5 - 1) // 2) * 1,
                    norm_type=norm_type,
                )
            )
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j - i == 1:
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNorm(
                            out_chan,
                            out_chan,
                            kernel_size=5,
                            stride=2,
                            groups=out_chan,
                            dilation=1,
                            padding=((5 - 1) // 2) * 1,
                            norm_type=norm_type,
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth - 1:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 2, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
                    )
                )
            else:
                self.concat_layer.append(
                    ConvNormAct(
                        out_chan * 3, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
                    )
                )
        self.last_layer = nn.Sequential(
            ConvNormAct(
                out_chan * upsampling_depth, out_chan, 1, 1, norm_type=norm_type, act_type=act_type
            )
        )
        self.res_conv = nn.Conv1d(out_chan, in_chan, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat(
                (
                    self.fuse_layers[i][0](output[i - 1])
                    if i - 1 >= 0
                    else torch.Tensor().to(output1.device),
                    output[i],
                    F.interpolate(output[i + 1], size=wav_length, mode="nearest")
                    if i + 1 < self.depth
                    else torch.Tensor().to(output1.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layer[i](y))

        wav_length = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=wav_length, mode="nearest")

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual