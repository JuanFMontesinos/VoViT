import os
import inspect
from copy import copy

import torch
from einops import rearrange
from torch import nn, istft
from torchaudio.functional import spectrogram

from . import fourier_defaults, VIDEO_FRAMERATE

from .weights import WEIGHTS_PATH
from .utils import load_weights
from .modules.spec2vec import Spec2Vec
from .modules.st_gcn import ST_GCN
from .transformers import *

DURATION = {'vovit_speech': 2, 'vovit_singing_voice': 4}

_base_args = {'d_model': 512,
              'nhead': 8,
              'dim_feedforward': 1024,
              'dropout': 0.3,
              'spec_freq_size': fourier_defaults['sp_freq_shape'],
              'skeleton_pooling': 'AdaptativeAP',
              "graph_kwargs": {
                  "graph_cfg": {
                      "layout": "acappella",
                      "strategy": "spatial",
                      "max_hop": 1,
                      "dilation": 1},
                  "edge_importance_weighting": "dynamic",
                  "dropout": False,
                  "dilated": False}
              }
_stt_args = {'fusion_module': 'spectral_transformer'}
_singing_voice_args = {'num_encoder_layers': 4,
                       'num_decoder_layers': 4,
                       'n_temp_feats': 64 * DURATION['vovit_singing_voice']
                       }
_speech_args = {'num_encoder_layers': 10,
                'num_decoder_layers': 10,
                'n_temp_feats': 64 * DURATION['vovit_speech']}


def copyupdt(original: dict, *args):
    assert isinstance(original, dict)
    new_dic = copy(original)
    for arg in args:
        assert isinstance(arg, dict)
        new_dic.update(arg)
    return new_dic


vovit_speech_args = copyupdt(_base_args, _stt_args, _speech_args)
vovit_singing_voice_args = copyupdt(_base_args, _stt_args, _singing_voice_args)


def complex_product(x, y):
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.shape[-1] == 2, "Last dimension must be 2"
    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack([real, imag], dim=-1)


def complex_division(x, y):
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.shape[-1] == 2, "Last dimension must be 2"
    real = (x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1]) / (y[..., 0] ** 2 + y[..., 1] ** 2)
    imag = (x[..., 1] * y[..., 0] - x[..., 0] * y[..., 1]) / (y[..., 0] ** 2 + y[..., 1] ** 2)
    return torch.stack([real, imag], dim=-1)


class AudioPreprocessor(nn.Module):
    def __init__(self, *,
                 debug: dict,
                 audio_length: int, audio_samplerate: int,
                 n_fft: int, hop_length: int, sp_freq_shape: int,
                 downsample_coarse: bool):
        super(AudioPreprocessor, self).__init__()

        self.downsample_coarse = downsample_coarse
        self.debug = debug

        self._audio_samplerate = audio_samplerate
        self._audio_length = audio_length
        self._n_fft = n_fft
        self._sp_freq_shape = sp_freq_shape
        self._hop_length = hop_length
        self.register_buffer('_window', torch.hann_window(self._n_fft), persistent=False)

    def wav2sp(self, x):
        # CUDNN does not support half complex numbers for non-power2 windows
        # Casting to float32 is a workaround
        dtype = x.dtype
        x = x.float()
        s = spectrogram(x, pad=0, window=self._window.float(), win_length=self._n_fft,
                        n_fft=self._n_fft, hop_length=self._hop_length,
                        power=None, normalized=False, return_complex=False)
        return s.to(dtype)

    def istft(self, x):
        if not x.is_complex():
            x = x.float()
        return istft(x, n_fft=self._n_fft, hop_length=self._hop_length, length=self._audio_length,
                     window=self._window.float())

    def sp2wav(self, inference_mask, mixture, compute_wav):
        if self.downsample_coarse:
            inference_mask = torch.nn.functional.upsample(rearrange(inference_mask, 'b f t c -> b c f t'),
                                                          scale_factor=(2, 1), mode='nearest').squeeze(
                1)
            inference_mask = rearrange(inference_mask, 'b c f t -> b f t c')
        estimated_sp = complex_product(inference_mask, mixture)
        if not compute_wav:
            return None, estimated_sp
        estimated_wav = self.istft(estimated_sp)
        return estimated_wav, estimated_sp

    def preprocess_audio(self, *src: list, n_sources=2):
        """
        Inputs contains the following keys:
           audio: the main audio waveform of shape N,M
           audio_acmt: the secondary audio waveform of shame N,M
           src: If using inference on real mixtures, the mixture audio waveform of shape N,M
        """

        self.n_sources = n_sources
        # Inference in case of a real sample
        sp_mix_raw = self.wav2sp(src[0]).contiguous() / self.n_sources

        if self.downsample_coarse:
            # Contiguous required to address memory problems in certain gpus
            sp_mix = sp_mix_raw[:, ::2, ...].contiguous()  # BxFxTx2
        x = rearrange(sp_mix, 'b f t c -> b c f t')
        output = {'mixture': x, 'sp_mix_raw': sp_mix_raw}

        return output

    def get_inference_mask(self, logits_mask, sp_mix):
        sp_mix = rearrange(sp_mix, 'b c f t -> b f t c')
        inference_mask = rearrange(logits_mask, 'b c f t -> b f t c')
        inference_mask = self.n_sources * inference_mask
        target_sp = complex_product(inference_mask, sp_mix)
        return inference_mask, target_sp


class AudioVisualNetwork(nn.Module):
    def __init__(self, *,
                 audio_kwargs,
                 video_temporal_features: int,
                 landmarks_enabled: bool,
                 n=1, **kwargs):
        """
        :param audio_model:
        :param audio_kwargs:
        :param video_enabled: bool Whether to use video features or not
        :param video_temporal_features: int Amount of visual temporal features to use (controls video upsampling and v
        video max pool)
        :param landmarks_enabled: bool Whether to use landmark and graph-cnn
        :param single_frame_enabled:  bool Whether to use appearance features extracted by a cnn or not
        :param single_emb_enabled: bool Whether to use appearance features pre-computed by a cnn or not
        :param n: This model works on 4s audio tracks. n is a multiplier for larger tracks 8s track--> n=2
        """
        super(AudioVisualNetwork, self).__init__()
        self.audio_processor = self.ap = AudioPreprocessor(**audio_kwargs)
        # Placeholder
        self._n = n
        self.feat_num = 0

        # Flags
        self.video_temporal_features = video_temporal_features
        self.landmarks_enabled = landmarks_enabled

        self._define_graph_network(kwargs)
        self._define_audio_network(kwargs)

    def _define_audio_network(self, kwargs):
        # Defining audio model /Stratum
        self.audio_network = SAplusF(input_dim=self.feat_num, **kwargs)

    def _define_graph_network(self, kwargs):
        if self.landmarks_enabled:
            self.feat_num += 256
            # Graph convolutional network for skeleton analysis
            if kwargs['skeleton_pooling'] == 'AdaptativeAP':
                self.pool = nn.AdaptiveAvgPool2d((None, 1))
            elif kwargs['skeleton_pooling'] == 'AdaptativeMP':
                self.pool = nn.AdaptiveMaxPool2d((None, 1))
            elif kwargs['skeleton_pooling'] == 'linear':
                self.pool = nn.Linear(self.graph_net.heads[0].graph.num_node, 1, bias=False)
            else:
                raise ValueError(
                    'VnNet pooling type: %s not implemented. Choose between AdaptativeMP,AdaptativeMP or linear' %
                    kwargs['skeleton_pooling'])

            if kwargs['graph_kwargs']['graph_cfg']['layout'] == 'upperbody_with_hands':
                in_channels = 3
            elif kwargs['graph_kwargs']['graph_cfg']['layout'] == 'acappella':
                in_channels = 2
            else:
                raise NotImplementedError
            flag = self.video_temporal_features < 50

            self.graph_net = ST_GCN(in_channels=in_channels, **kwargs['graph_kwargs'], temporal_downsample=flag)

    def forward(self, inputs: dict,
                compute_wav=True):
        # Placeholder
        output = {'logits_mask': None,
                  'inference_mask': None,
                  'loss_mask': None,
                  'gt_mask': None,
                  'separation_loss': None,
                  'alignment_loss': None,
                  'estimated_sp': None,
                  'estimated_wav': None}

        landmarks = inputs['landmarks']

        # ==========================================

        audio_feats = self.audio_processor.preprocess_audio(inputs['src'])

        """
        mixture: ready to fed the network
        sources raw: list of all the independent sources before downsampling
        weight: gradient penalty term for the loss
        sp_mix_raw: mixture spectrogram before downsampling
        """

        # ==========================================
        # Generating visual features
        visual_features = self.forward_visual(landmarks)

        # ==========================================

        # Computing audiovisual prediction
        pred = self.forward_audiovisual(audio_feats, visual_features)

        # ==========================================

        logits_mask = pred
        output['logits_mask'] = logits_mask
        inference_mask, target_sp = self.audio_processor.get_inference_mask(logits_mask, audio_feats['mixture'])
        output['inference_mask'] = inference_mask
        # Upsampling must be carried out on the mask, NOT the spectrogram
        # https://www.juanmontesinos.com/posts/2021/02/08/bss-masking/

        estimated_wav, estimated_sp = self.ap.sp2wav(inference_mask,
                                                     audio_feats['sp_mix_raw'],
                                                     compute_wav)
        output['estimated_sp'] = estimated_sp
        output['estimated_wav'] = estimated_wav

        output['mix_sp'] = torch.view_as_complex(audio_feats['sp_mix_raw'])
        return output

    def forward_audiovisual(self, audio_feats, visual_features):
        pred = self.audio_network(audio_feats, visual_features)
        return pred

    def forward_visual(self, landmarks):
        sk_features = self.graph_net(landmarks)
        sk_features = self.pool(sk_features).squeeze(3)
        sk_features = torch.nn.functional.interpolate(sk_features,
                                                      size=self.video_temporal_features * self._n)
        sk_features = rearrange(sk_features, 'b c t -> b t c')
        return sk_features


class SAplusF(nn.Module):
    def __init__(self, *,
                 fusion_module: str,
                 input_dim: int,
                 last_shape=8,
                 **kwargs):
        super(SAplusF, self).__init__()
        self.audio_net = Spec2Vec(last_shape)
        self.fusion_module = self._set_fusion_module(fusion_module, input_dim * (last_shape + 1), **kwargs)

    def _set_fusion_module(self, *args, **kwargs):
        transformer_kw = {}
        for arg in inspect.getfullargspec(AVSpectralTransformer.__init__).kwonlyargs:
            transformer_kw[arg] = kwargs[arg]
        fusion_module = AVSpectralTransformer(**transformer_kw)

        return fusion_module

    def forward(self, audio: dict, video_feats):
        """
        :param audio: spectrogram of the mixture B C F T
        :param video_feats: video features of the tgt speaker
        :param audio_clean: spectrogram of the tgt speaker (if training) B C F T
        """
        # input_audio will be (N,2,256,256)
        # input video feats [N, 256, 256]
        input_audio = rearrange(audio['mixture'], 'b c f t -> b c t f')  # Freqxtime required for audio network
        audio_feats = self.audio_net(input_audio)  # [N, 2048, 256, 1]
        # audio_feats will be (N,8*256,256,1)
        audio_feats = rearrange(audio_feats.squeeze(-1), 'b feats t -> b t feats')
        complex_mask = self.fusion_module(video_feats, audio_feats, audio)

        return complex_mask


class RefinementAVSE_LowLatency(nn.Module):

    def __init__(self, av_se: nn.Module):
        super(RefinementAVSE_LowLatency, self).__init__()
        from flerken.models import UNet
        self.av_se = av_se
        self.unet = UNet(mode='upsample', architecture='sop', layer_kernels="ssssst",
                         output_channels=1,
                         film=None,
                         useBN=True,
                         activation=torch.sigmoid,
                         layer_channels=[32, 64, 128, 256, 256, 256, 512])

    def forward_avse(self, inputs, compute_istft: bool):
        self.av_se.eval()
        output = self.av_se(inputs, compute_wav=compute_istft)
        return output

    def forward(self, *args, **kwargs):
        return self.inference(*args, **kwargs)

    def inference(self, inputs: dict, n_iter=1):
        with torch.no_grad():
            output = self.forward_avse(inputs, compute_istft=False)
            estimated_sp = output['estimated_sp']
            for i in range(n_iter):
                estimated_sp_mag = estimated_sp.norm(dim=-1)
                mask = self.unet(estimated_sp_mag[:, None].contiguous())[:, 0]
                estimated_sp = mask.unsqueeze(-1) * estimated_sp
                output[f'estimated_sp_{i}'] = estimated_sp
            output['ref_mask'] = mask
            output['ref_est_sp'] = estimated_sp
            output['ref_est_wav'] = self.av_se.ap.istft(output['ref_est_sp'])
            return output


class VoViT(nn.Module):
    def __init__(self, *, model_name: str, debug: dict, pretrained: bool):
        super(VoViT, self).__init__()
        self.pretrained = pretrained
        self.debug = debug

        assert model_name.lower() in ['vovit_speech', 'vovit_singing_voice']
        self.avse = self._instantiate_avse_model(debug, model_name.lower())

    def _instantiate_avse_model(self, debug, model_name: str):
        audio_kw = copy(fourier_defaults)
        sr = fourier_defaults['audio_samplerate']
        audio_kw.update({'audio_length': DURATION[model_name] * sr - 1, 'downsample_coarse': True, 'debug': debug})

        model = AudioVisualNetwork(audio_kwargs=audio_kw,
                                   video_temporal_features=64 * DURATION[model_name],
                                   landmarks_enabled=True,
                                   **globals()[model_name + '_args'])
        if self.pretrained:
            state_dict = load_weights(os.path.join(WEIGHTS_PATH, model_name + '.pth'))
            model.load_state_dict(state_dict)
            print('VoViT pre-trained weights loaded')
        if model_name == 'vovit_speech':
            model = RefinementAVSE_LowLatency(model)
            if self.pretrained:
                state_dict = load_weights(os.path.join(WEIGHTS_PATH, 'refinement_avse_low_latency.pth'))
                model.unet.load_state_dict(state_dict)
                print('Lead Voice enhancer pre-trained weights loaded')

        return model

    def forward(self, mixture, landmarks):
        """
        mixture: torch.Tensor (B,N)
        """
        inputs = {'src': mixture, 'landmarks': landmarks}
        return self.avse(inputs)
