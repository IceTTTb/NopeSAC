import torch
import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.config import configurable
from typing import Callable, Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
import logging
import numpy as np
import quaternion

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def get_assignment_matrix(log_scores_padded, match_threshold):
    max0, max1 = log_scores_padded[:, :-1, :-1].max(2), log_scores_padded[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = log_scores_padded.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))

    assignment_matrix = torch.zeros_like(log_scores_padded)  # 1, n1 + 1, n2 + 1
    assert assignment_matrix.shape[0] == 1
    indices0_ = indices0.clone()
    indices0_[indices0_ == -1] = assignment_matrix.shape[-1] - 1
    idxs0 = torch.arange(0, indices0_.shape[1]).to(indices0_.device)
    assignment_matrix[0, idxs0, indices0_] = 1
    assignment_matrix = assignment_matrix[:, :-1, :-1]  # 1, n1, n2

    return assignment_matrix

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.LeakyReLU(inplace=True),
    )

def angle_error_vec(v1, v2):
    return 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(v1, v2), axis=1)), -1.0, 1.0)) * 180 / np.pi

def build_rot_matrix_from_angle(A1, A2, A3):
    """
    A1: bs
    A2: bs
    A3: bs
    """
    A1 = A1 / 180. * np.pi
    A2 = A2 / 180. * np.pi
    A3 = A3 / 180. * np.pi
    inf = 1e4
    device = A1.device
    bs = A1.shape[0]

    R1 = torch.tensor([[1, 0,     0],
                       [0, inf, inf],
                       [0, inf, inf]]).reshape(1, 3, 3).repeat(bs, 1, 1).to(device=device, dtype=A1.dtype)
    R2 = torch.tensor([[inf, 0, inf],
                       [0,   1,   0],
                       [inf, 0, inf]]).reshape(1, 3, 3).repeat(bs, 1, 1).to(device=device, dtype=A1.dtype)
    R3 = torch.tensor([[inf, inf, 0],
                       [inf, inf, 0],
                       [0,   0,   1]]).reshape(1, 3, 3).repeat(bs, 1, 1).to(device=device, dtype=A1.dtype)

    cosA1 = torch.cos(A1)  # bs
    sinA1 = torch.sin(A1)  # bs
    R1[:, 1, 1] = cosA1
    R1[:, 1, 2] = -sinA1
    R1[:, 2, 1] = sinA1
    R1[:, 2, 2] = cosA1
    R1 = R1.contiguous()

    cosA2 = torch.cos(A2)  # bs
    sinA2 = torch.sin(A2)  # bs
    R2[:, 0, 0] = cosA2
    R2[:, 0, 2] = sinA2
    R2[:, 2, 0] = -sinA2
    R2[:, 2, 2] = cosA2
    R2 = R2.contiguous()

    cosA3 = torch.cos(A3)  # bs
    sinA3 = torch.sin(A3)  # bs
    R3[:, 0, 0] = cosA3
    R3[:, 0, 1] = -sinA3
    R3[:, 1, 0] = sinA3
    R3[:, 1, 1] = cosA3
    R3 = R3.contiguous()

    R = R1 @ R2 @ R3

    return R  # bs, 3, 3

def generate_rand_qua(bs, rA1=20., rA2=300., rA3=20.):
    A1 = (torch.rand(bs) - 0.5) * rA1
    A2 = (torch.rand(bs) - 0.5) * rA2
    A3 = (torch.rand(bs) - 0.5) * rA3

    Rm = build_rot_matrix_from_angle(A1, A2, A3)
    debug_rot = quaternion.from_rotation_matrix(Rm.cpu().numpy())
    debug_rot = quaternion.as_float_array(debug_rot)
    debug_rot = torch.tensor(debug_rot)

    debug_rot_sig = debug_rot[:, 0:1]  # bs, 1
    debug_rot_sig = ((debug_rot_sig >= 0).float() - 0.5) * 2.
    debug_rot = debug_rot * debug_rot_sig

    return debug_rot

def quaternion2rotmatrix(quan):
    assert isinstance(quan, torch.Tensor)
    assert quan.shape[-1] == 4
    if quan.dim() == 1:
        quan = quan.unsqueeze(0)  # 1, 4
    elif quan.dim() == 2:
        pass  # bs, 4
    else:
        raise NotImplementedError

    bs = quan.shape[0]
    rot_matrix = torch.zeros([bs, 3, 3]).to(device=quan.device, dtype=quan.dtype)  # bs, 3, 3

    w = quan[:, 0]  # bs
    x = quan[:, 1]
    y = quan[:, 2]
    z = quan[:, 3]

    m1 = 1 - 2 * y * y - 2 * z * z
    m2 = 2 * x * y - 2 * w * z
    m3 = 2 * x * z + 2 * w * y
    m4 = 2 * x * y + 2 * w * z
    m5 = 1 - 2 * x * x - 2 * z * z
    m6 = 2 * y * z - 2 * w * x
    m7 = 2 * x * z - 2 * w * y
    m8 = 2 * y * z + 2 * w * x
    m9 = 1 - 2 * x * x - 2 * y * y

    rot_matrix[:, 0, 0] = m1
    rot_matrix[:, 0, 1] = m2
    rot_matrix[:, 0, 2] = m3

    rot_matrix[:, 1, 0] = m4
    rot_matrix[:, 1, 1] = m5
    rot_matrix[:, 1, 2] = m6

    rot_matrix[:, 2, 0] = m7
    rot_matrix[:, 2, 1] = m8
    rot_matrix[:, 2, 2] = m9

    rot_matrix = rot_matrix.contiguous()  # bs, 3, 3

    return rot_matrix

def warp_plane_param_to_global(geo_sequence_local, rot_quan, tran):
    """
        rot_quan: bs, n, 4
        tran: bs, n, 3
    """
    bs, n, _ = rot_quan.shape

    # get normal and offset
    plane0 = geo_sequence_local[:, :, :3].unsqueeze(1).repeat(1, n, 1, 1).view(bs * n, n, 3)  # bs*n, n, 3

    # get normal and offset
    plane1 = geo_sequence_local[:, :, 3:].unsqueeze(1).repeat(1, n, 1, 1).view(bs * n, n, 3)  # bs*n, n, 3

    # get rot matrix
    rot_quan = rot_quan.view(-1, 4)  # bs*n, 4
    rot_matrix = quaternion2rotmatrix(rot_quan)  # bs*n, 3, 3

    tran = tran.unsqueeze(2).repeat(1, 1, n, 1).view(bs * n, n, 3)  # bs*n, n, 3

    # convert plane of the first view
    start = tran  # bs*n, n, 3
    end = plane0 * (torch.tensor([1, -1, -1]).reshape(1, 1, 3).to(tran.device))  # suncg2habitat   # bs*n, n, 3
    end = end.permute(0, 2, 1)  # bs*n, 3, n
    end = (torch.bmm(rot_matrix, end)).permute(0, 2, 1) + tran  # cam2world  # bs*n, n, 3
    a = end  # bs*n, n, 3
    b = end - start  # bs*n, n, 3
    plane0 = ((a * b).sum(dim=-1) / (torch.norm(b, dim=-1) + 1e-5) ** 2).view(bs * n, n, 1) * b  # bs*n, n, 3
    plane0 = plane0.reshape(bs, n, n, 3).contiguous()

    # convert plane of the second view
    plane1 = plane1 * (torch.tensor([1, -1, -1]).reshape(1, 1, 3).to(plane1.device))  # bs*n, n, 3
    plane1 = plane1.reshape(bs, n, n, 3).contiguous()

    return plane0, plane1

def QuaternionMultiplication(q1, q2):
    """
    q1: bs, 4
    q2: bs, 4
    """
    w1 = q1[:, 0]  # bs,
    w2 = q2[:, 0]  # bs,

    x1 = q1[:, 1]
    x2 = q2[:, 1]

    y1 = q1[:, 2]
    y2 = q2[:, 2]

    z1 = q1[:, 3]
    z2 = q2[:, 3]

    w_new = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x_new = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y_new = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z_new = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q_new = torch.stack((w_new, x_new, y_new, z_new), dim=-1)  # bs, 4
    q_new = q_new.contiguous()

    return q_new

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, init_on=True):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        if init_on:
            for layer in self.layers:
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BasePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        self.in_features = self.in_features[1:]
        feature_channels = feature_channels[1:]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y)

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)

class CameraPoseLoss(nn.Module):
    def __init__(self):
        super(CameraPoseLoss, self).__init__()
        self.norm = 2

    def forward(self, est_pose, gt_pose, reduce=True, mask=None):
        if reduce:
            if mask is None:
                # Position loss
                l_x = torch.norm(gt_pose[:, 0:3] - est_pose[:, 0:3], dim=1, p=self.norm).mean()
                # Orientation loss (normalized to unit norm)
                l_q = torch.norm(F.normalize(gt_pose[:, 3:], p=2, dim=1) - F.normalize(est_pose[:, 3:], p=2, dim=1),
                                 dim=1, p=self.norm).mean()
            else:
                l_x = torch.norm(gt_pose[:, 0:3] - est_pose[:, 0:3], dim=1, p=self.norm)
                l_q = torch.norm(F.normalize(gt_pose[:, 3:], p=2, dim=1) - F.normalize(est_pose[:, 3:], p=2, dim=1),
                                 dim=1, p=self.norm)
                l_x = (l_x[mask > 0]).mean()
                l_q = (l_q[mask > 0]).mean()
            return l_x, l_q
        else:
            l_x = torch.norm(gt_pose[:, 0:3] - est_pose[:, 0:3], dim=1, p=self.norm)
            l_q = torch.norm(F.normalize(gt_pose[:, 3:], p=2, dim=1) - F.normalize(est_pose[:, 3:], p=2, dim=1),
                             dim=1, p=self.norm)
            return l_x, l_q
