import numpy as np
import torch
from torch import nn

__all__ = ['get_sinusoid_encoding_table', 'AVSpectralTransformer']


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class BaseFusionModule(nn.Module):
    def forward(self, v_feats: torch.Tensor, a_feats: torch.Tensor, *args) -> torch.Tensor:
        """
        :param v_feats: Visual features from tgt speaker. BxTxC
        :param a_feats: Audio features from the mixture. BxT'xC' (T may be equal to T' for some models)
        :param args: Place-holder for other models which require extra info
        :return: Complex mask which applied over the mixture estimate the clean audio.
        """
        raise NotImplementedError

    def generate_square_subsequent_mask(self, sz: int, device: torch.device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


from .av_sttrans import AVSpectralTransformer