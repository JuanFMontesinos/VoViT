# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import os.path as osp
import numpy as np
import torch
from utils.io import _load

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(torch.nn.Module):
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        super(BFMModel, self).__init__()
        bfm = _load(bfm_fp)
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            self.tri = _load(make_abs_path('../configs/tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
        else:
            self.tri = bfm.get('tri')

        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]


        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
        self.u_base = torch.from_numpy(self.u_base).cuda()
        self.w_shp_base = torch.from_numpy(self.w_shp_base).cuda()
        self.w_exp_base = torch.from_numpy(self.w_exp_base).cuda()

