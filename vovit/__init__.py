import os
import yaml

import torch
from einops import rearrange
from numpy import load as np_load
from .core import VoViT, core_path, VIDEO_FRAMERATE
from .core.kabsch import register_sequence_of_landmarks
from .core.functionals import *

from . import utils


class End2EndVoViT(torch.nn.Module):
    def __init__(self, *, model_name: str, debug: dict, pretrained: bool = True,
                 extract_landmarks: bool = False, detect_faces: bool = False):
        super().__init__()
        self.extract_landmarks = extract_landmarks
        self.detect_faces = detect_faces
        self.vovit = VoViT(model_name=model_name, debug=debug, pretrained=pretrained)

        if self.extract_landmarks:
            from .core.landmark_estimator.TDDFA_GPU import TDDFA
            cfg = yaml.load(open(utils.DEFAULT_CFG_PATH), Loader=yaml.SafeLoader)
            cfg['checkpoint_fp'] = os.path.join(utils.LANDMARK_LIB_PATH, 'weights', 'mb1_120x120.pth')
            cfg['bfm_fp'] = os.path.join(utils.LANDMARK_LIB_PATH, 'configs', 'bfm_noneck_v3.pkl')
            self.face_extractor = TDDFA(**cfg)
        self.register_buffer('mean_face',
                             torch.from_numpy(np_load(os.path.join(core_path, 'speech_mean_face.npy'))).float(),
                             persistent=False)

    def forward(self, mixture, visuals, extract_landmarks=False):
        """
        :param mixture: torch.Tensor of shape (B,N)
        :param visuals: torch.Tensor of shape (B,C,H,W) BGR format required
        :return:
        """
        if self.detect_faces:
            raise NotImplementedError
        else:
            cropped_video = visuals
        if extract_landmarks:
            ld = self.face_extractor(cropped_video)
            avg = (ld[:-2] + ld[1:-1] + ld[2:]) / 3
            ld[:-2] = avg
        else:
            ld = cropped_video
        # Registering the face
        if not ld.is_floating_point():
            ld = ld.float()
        ld = torch.stack([register_sequence_of_landmarks(x[..., :48], self.mean_face[:, :48],
                                                         per_frame=True,
                                                         display_sequence=x) for x in ld])
        ld = rearrange(ld, 'b t c j ->b c t j')[:, :2].unsqueeze(-1)
        mixture = cast_dtype(mixture, raise_error=True)  # Cast integers to float
        mixture /= mixture.abs().max()

        return self.vovit(mixture, ld)

    def forward_unlimited(self, mixture, visuals):
        """
        Allows to run inference in an unlimited duration samples (up to gpu memory constrains)
        The results will be trimmed to multiples of 2 seconds (e.g. if your audio is 8.5 seconds long,
        the result will be trimmed to 8 seconds)
        Args:
            visuals: raw video if self.extract_landmarks is True, precomputed_landmarks otherwise.
                    lanmarks are uint16 tensors of shape (T,3,68)
                    raw video are uint8 RGB tensors of shape (T,H,W,3) (values between 0-255)
            mixture: tensor of shape (N)
        """
        fps = VIDEO_FRAMERATE
        length = self.vovit.avse.av_se.ap._audio_length
        n_chunks = visuals.shape[0] // (fps * 2)
        if self.extract_landmarks:
            visuals = self.face_extractor(visuals)
            avg = (visuals[:-2] + visuals[1:-1] + visuals[2:]) / 3
            visuals[:-2] = avg
        visuals = visuals[:n_chunks * fps * 2].view(n_chunks, fps * 2, 3, 68)
        mixture = mixture[:n_chunks * length].view(n_chunks, -1)
        pred = self.forward(mixture, visuals)
        pred_unraveled = {}
        for k, v in pred.items():
            if v is None:
                continue
            if v.is_complex():  # Complex spectrogram
                pred_unraveled[k] = rearrange(v, 'b f t ->  f (b t)')
            if v.ndim == 4:  # Two-channels mask
                idx = v.shape.index(2)
                if idx == 1:
                    string = 'b c f t ->  c f (b t)'
                elif idx == 3:
                    string = 'b f t c->  f (b t) c'
                else:
                    raise ValueError('Unknown shape')
                pred_unraveled[k] = rearrange(v, string)
            if v.ndim == 2:  # Waveforms
                pred_unraveled[k] = v.flatten()
        return pred_unraveled
