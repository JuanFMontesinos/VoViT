"""
Source code from cleardusk
Optimized by Juan F. Montesinos to work with batches in GPU
"""
import os.path as osp
import torch
from torch import nn
from torchvision.transforms import Compose

import models
from bfm import BFMModel
from utils.io import _load
from utils.functions import (
    crop_video, reshape_fortran, parse_roi_box_from_bbox,
)
from utils.tddfa_util import (
    load_model, _batched_parse_param, batched_similar_transform,
    ToTensorGjz, NormalizeGjz
)

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA(nn.Module):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        self.size = kvs.get('size', 120)

        # load BFM
        self.bfm = BFMModel(
            bfm_fp=kvs.get('bfm_fp', make_abs_path('configs/bfm_noneck_v3.pkl')),
            shape_dim=kvs.get('shape_dim', 40),
            exp_dim=kvs.get('exp_dim', 10)
        )
        self.tri = self.bfm.tri

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # load model, default output is dimension with length 62 = 12(pose) + 40(shape) +10(expression)
        model = getattr(models, kvs.get('arch'))(
            num_classes=kvs.get('num_params', 62),
            widen_factor=kvs.get('widen_factor', 1),
            size=self.size,
            mode=kvs.get('mode', 'small')
        )
        model = load_model(model, kvs.get('checkpoint_fp'))


        self.model = model

        # data normalization
        self.transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, self.transform_normalize])
        self.transform = transform

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = torch.from_numpy(r.get('mean'))
        self.param_std = torch.from_numpy(r.get('std'))
        self.param_mean = self.param_mean
        self.param_std = self.param_std



    def batched_inference(self, video_ori, bbox, **kvs):
        """The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :param objs: left, top, right, bottom = bbox (think in lines like y=25, not points)
        :param kvs: options
        :return: param list and roi_box list
        """
        roi_box = parse_roi_box_from_bbox(bbox)
        video = crop_video(video_ori, roi_box)
        img = torch.nn.functional.interpolate(video, size=(self.size, self.size), mode='bilinear', align_corners=False)

        inp = self.transform_normalize(img)
        param = self.model(inp)

        param = param * self.param_std + self.param_mean  # re-scale

        return param, roi_box

    def batched_recon_vers(self, param, roi_box, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size
        R, offset, alpha_shp, alpha_exp = _batched_parse_param(param)
        if dense_flag:
            tensor = self.bfm.u + self.bfm.w_shp @ alpha_shp + self.bfm.w_exp @ alpha_exp
        else:
            tensor = self.bfm.u_base + self.bfm.w_shp_base @ alpha_shp + self.bfm.w_exp_base @ alpha_exp
        pts3d = R @ reshape_fortran(tensor, (param.shape[0], 3, -1)) + offset
        pts3d = batched_similar_transform(pts3d, roi_box, size)

        return pts3d
