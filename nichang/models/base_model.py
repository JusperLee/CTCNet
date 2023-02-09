###
# Author: Kai Li
# Date: 2021-06-17 23:08:32
# LastEditors: Kai Li
# LastEditTime: 2021-08-30 18:02:00
###
import torch
import torch.nn as nn
import math
from ..utils.separator import Separator, separate
from ..layers import activations, normalizations
from ..utils.torch_utils import shape_reconstructed, pad_x_to_y


def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


def pad_to_appropriate_length(x, lcm):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padded_x = torch.zeros(
            list(appropriate_shape[:-1]) + [appropriate_shape[-1] + lcm - values_to_pad],
            dtype=torch.float32,
        ).to(x.device)
        padded_x[..., : x.shape[-1]] = x
        return padded_x
    return x


class BaseModel(nn.Module):
    def __init__(self, sample_rate, in_chan=1):
        super().__init__()
        self._sample_rate = sample_rate
        self._in_chan = in_chan

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample_rate(
        self,
    ):
        return self._sample_rate

    def separate(self, *args, **kwargs):
        return separate(*args, **kwargs)

    def forward_wav(self, wav, *args, **kwargs):
        return self(wav, *args, **kwargs)

    @staticmethod
    def load_state_dict_in_audio(model, pretrained_dict):
        model_dict = model.state_dict()
        update_dict = {}
        for k, v in pretrained_dict.items():
            if "audio_model" in k:
                update_dict[k[12:]] = v
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        return model

    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get
        conf = torch.load(
            pretrained_model_conf_or_path, map_location="cpu"
        )  # Attempt to find the model and instantiate it.
        model_class = get(conf["model_name"])
        # model_class = get("Conv_TasNet")
        model = model_class(*args, **kwargs)
        model.load_state_dict(conf["state_dict"])
        return model

    def serialize(self):
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError


class BaseEncoderMaskerDecoder(BaseModel):
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()

        # stft
        self.mag = None
        self.phase = None

    def forward(self, wav, is_stft=False):
        shape = wav.shape
        wav = _unsqueeze_to_3d(wav)

        enc_w = self.forward_encoder(wav, is_stft=is_stft)
        masks = self.forward_masker(enc_w)
        ests_wav = self.forward_decoder(masks, is_stft=is_stft)

        reconstructed = pad_x_to_y(ests_wav, wav)
        return shape_reconstructed(reconstructed, shape)

    def forward_encoder(self, wav, is_stft=False):
        if is_stft:
            spec = self.encoder(wav)
            spec_real, spec_imag = spec[:, :, :, 0], spec[:, :, :, 1]
            self.mag = torch.sqrt(torch.square(spec_real) + torch.square(spec_imag))
            self.phase = spec / (self.mag.unsqueeze(-1) + 1e-12)
            return self.enc_activation(self.mag)
        enc_w = self.encoder(wav)
        return self.enc_activation(enc_w)

    def forward_masker(self, enc_w):
        return self.apply_masks(enc_w, self.masker(enc_w))

    def forward_decoder(self, masks, is_stft):
        # masks: [b, spk, f, t]
        # phase: [b, f, t, 2]
        if is_stft:
            outputs = []
            for i in range(masks.shape[1]):
                outputs.append(
                    self.encoder.inverse(masks[:, i, :, :].unsqueeze(-1) * self.phase).unsqueeze(1)
                )
            return torch.cat(outputs, dim=1)
        return self.decoder(masks)

    def apply_masks(self, enc_w, masks):
        return masks * enc_w.unsqueeze(1)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args
