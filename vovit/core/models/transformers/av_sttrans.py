"""
Inspired by https://arxiv.org/pdf/1911.09783.pdf
            WildMix Dataset and Spectro-Temporal Transformer Model
            for Monoaural Audio Source Separation
Source code (not used but can help others)
https://github.com/tianjunm/monaural-source-separation/blob/fd773aec28d4dee54746e340c30c855b59b5f6ab/models/stt_aaai.py
"""

import torch
from einops import rearrange
from torch import nn
from torch.nn import TransformerEncoderLayer as TorchTFL, TransformerEncoder, LayerNorm

from . import BaseFusionModule, get_sinusoid_encoding_table


class TransformerEncoderLayer(TorchTFL):
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)


def build_encoder(d_model, nhead, dim_feedforward, dropout, num_encoder_layers):
    assert (d_model % nhead) == 0, f'Transformers d_model must be divisible by nhead but' \
                                   f' {d_model}/{nhead}={d_model // nhead}R{d_model % nhead}'
    layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu')
    encoder = TransformerEncoder(layer, num_encoder_layers, norm=LayerNorm(d_model))
    return encoder


class STEncoder(nn.Module):
    """
    Builds either an spectral encoder or a temporal encoder.
    """

    def __init__(self, n_temp_feats, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float):
        super(STEncoder, self).__init__()
        self.encoder = build_encoder(d_model, nhead, dim_feedforward, dropout, num_encoder_layers)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_temp_feats + 1, d_model, padding_idx=0),
            freeze=True)

    def forward(self, av_feats: torch.Tensor) -> torch.Tensor:
        """
        :param av_feats: AudioVisual signal of shape BxTxC if temporal or BxCxT if spectral
        """
        B, T, C = av_feats.shape
        pos = torch.arange(1, T + 1, device=av_feats.device)
        pos = self.pos_emb(pos)
        av_feats += pos
        av_feats = self.encoder(rearrange(av_feats, 'b t c -> t b c'))
        return av_feats


class SpectroTemporalEncoder(nn.Module):
    def __init__(self, *, n_temp_feats, n_channels, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float):
        super(SpectroTemporalEncoder, self).__init__()
        self.temporal_enc = STEncoder(n_temp_feats, n_channels, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.spectral_enc = STEncoder(n_channels, n_temp_feats, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.av_dim_adaptation = nn.Sequential(nn.LazyLinear(n_channels), nn.LeakyReLU(0.1))

    def forward(self, av_feats):
        """
        :param av_feats: BxTxC
        :return: TxBxC set of features
        """
        av_feats = self.av_dim_adaptation(av_feats)
        temp_feats = self.temporal_enc(av_feats)

        # Note that the spectral encoder is gonna permute b t c-> c b t
        spectral_feats = self.spectral_enc(rearrange(av_feats, 'b t c -> b c t'))
        spectral_feats = rearrange(spectral_feats, 'c b t -> t b c')
        feats = spectral_feats + temp_feats
        # feats = spectral_feats
        return feats



class AVSpectralTransformer(BaseFusionModule):
    def __init__(self, *, n_temp_feats, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float, spec_freq_size: int):
        super(AVSpectralTransformer, self).__init__()
        self.encoder = SpectroTemporalEncoder(n_temp_feats=n_temp_feats, n_channels=d_model, nhead=nhead,
                                              num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
                                              dropout=dropout)
        self.decoder = build_encoder(d_model, nhead, dim_feedforward, dropout, num_decoder_layers)
        self.feats2mask = nn.Sequential(nn.LazyLinear(2 * d_model), nn.LeakyReLU(0.5),
                                        nn.Linear(2 * d_model, spec_freq_size))
        self.d_model = d_model
        self.spec_freq_size = spec_freq_size

    def forward(self, v_feats: torch.Tensor, a_feats: torch.Tensor, *args) -> torch.Tensor:
        B, T, C = a_feats.shape
        av_feats = torch.cat([v_feats, a_feats], dim=-1)

        memory = self.encoder(av_feats)
        latent_feats = self.decoder(memory)
        latent_feats = self.feats2mask(latent_feats)
        mask = rearrange(latent_feats, 't b c -> b c t ').reshape(B, 2, self.spec_freq_size // 2, T)
        return mask


class ConvTemporalEncoder(nn.Module):
    def __init__(self, *, n_temp_feats, n_channels, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float):
        super(ConvTemporalEncoder, self).__init__()
        self.temporal_enc = STEncoder(n_temp_feats, n_channels, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.spectral_enc = STEncoder(n_channels, n_temp_feats, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.av_dim_adaptation = nn.Sequential(nn.LazyLinear(n_channels), nn.LeakyReLU(0.1))

    def forward(self, av_feats):
        """
        :param av_feats: BxTxC
        :return: TxBxC set of features
        """
        av_feats = self.av_dim_adaptation(av_feats)
        temp_feats = self.temporal_enc(av_feats)

        # Note that the spectral encoder is gonna permute b t c-> c b t
        spectral_feats = self.spectral_enc(rearrange(av_feats, 'b t c -> b c t'))
        spectral_feats = rearrange(spectral_feats, 'c b t -> t b c')
        feats = spectral_feats + temp_feats
        # feats = spectral_feats
        return feats
