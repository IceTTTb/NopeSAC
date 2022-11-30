import torch
from torch import nn
from torch.nn import functional as F
from detectron2.utils.registry import Registry
from copy import deepcopy
import logging
import numpy as np

from ..transformer.gnn import LocalFeatureTransformer

logger = logging.getLogger(__name__)

__all__ = ["build_matching_head", "MATCHING_HEAD_REGISTRY"]

MATCHING_HEAD_REGISTRY = Registry("MATCHING_HEAD")
MATCHING_HEAD_REGISTRY.__doc__ = """
Registry for plane matching head
"""

def build_matching_head(cfg):
    return MATCHING_HEAD_REGISTRY.get("MatchingHead")(cfg)

@MATCHING_HEAD_REGISTRY.register()
class MatchingHead(nn.Module):
    def __init__(self, cfg):
        super(MatchingHead, self).__init__()
        self.cfg = cfg
        self.offset_multiplier = cfg.MODEL.MATCHING_HEAD.OFFSET_MULTIPLIER
        self.normal_multiplier = cfg.MODEL.MATCHING_HEAD.NORMAL_MULTIPLIER
        gnn_config = {
            "d_model": 256,
            "nhead": 8,
            "layer_names": ['self', 'cross'] * 9,
        }
        self.gnn = LocalFeatureTransformer(gnn_config)
        self.planeDesc_proj = nn.Conv1d(256, 256, kernel_size=1, bias=True)
        self.bin_score = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.sinkhorn_iterations = 200
        self.max_length = cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES
        self.mask_on = True
        self.planeApp_proj = nn.Conv1d(256, 256, kernel_size=1, bias=True)

    def forward(self, planeApp1, planeApp2, matcher_inputCam, parameters1_local, parameters2_local,
                indices1=None, indices2=None, gt_corr_matrix=None,
                suffix="", normal_decay=1.0, offset_deacy=1.0):
        sinkhorn_iterations = self.sinkhorn_iterations
        bs, n1, _ = planeApp1.shape
        n2 = planeApp2.shape[1]

        # compute row and col masks
        if indices1 is not None and indices2 is not None and self.mask_on:
            row_masks = torch.zeros(size=(bs, n1), dtype=torch.bool).cuda()
            col_masks = torch.zeros(size=(bs, n2), dtype=torch.bool).cuda()
            for bi in range(bs):
                matched_predPlane_idx1 = indices1[bi][0]
                matched_predPlane_idx2 = indices2[bi][0]
                row_masks[bi][matched_predPlane_idx1] = True
                col_masks[bi][matched_predPlane_idx2] = True
            padded_row_masks = torch.zeros(size=(bs, n1 + 1), dtype=torch.bool).cuda()
            padded_row_masks[:, :n1] = ~row_masks
            padded_col_masks = torch.zeros(size=(bs, n2 + 1), dtype=torch.bool).cuda()
            padded_col_masks[:, :n2] = ~col_masks
            padded_gt_valid_corr_masks = ~torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))
        else:
            row_masks = None
            col_masks = None
            padded_gt_valid_corr_masks = None
        if padded_gt_valid_corr_masks is not None and gt_corr_matrix is not None:
            gt_corr_matrix = torch.logical_and(padded_gt_valid_corr_masks, gt_corr_matrix)

        if matcher_inputCam is None:
            offset_dist = torch.zeros(bs, n1, n2).to(dtype=planeApp1.dtype, device=planeApp1.device)
            normal_dist = torch.zeros(bs, n1, n2).to(dtype=planeApp1.dtype, device=planeApp1.device)
        else:
            # warp plane parameters 2
            parameters2_warped = self.warp_single_view_plane_param_to_global(
                parameters2_local, pose_n=1)[:, 0, :, :]  # bs, n2, 3
            offset2_warped = torch.norm(parameters2_warped, dim=2, keepdim=True, p=2)  # bs, n2, 1
            normal2_warped = F.normalize(parameters2_warped, dim=-1, p=2)  # bs, n2, 3
            # warp plane parameters 1
            parameters1_warped_r = self.warp_single_view_plane_param_to_global(
                parameters1_local, matcher_inputCam[:, 3:].unsqueeze(1), matcher_inputCam[:, :3].unsqueeze(1) * 0.)[:,
                                   0, :, :]  # bs, n1, 3
            normal1_warped_r = F.normalize(parameters1_warped_r, dim=-1, p=2)  # bs, n1, 3
            nTn_r = torch.bmm(normal1_warped_r, normal2_warped.transpose(1, 2))  # bs, n1, n2
            normal_dist = torch.acos(torch.clamp(nTn_r, -1, 1))  # in rad
            normal_dist = normal_dist / np.pi * 180.
            parameters1_warped_rt = self.warp_single_view_plane_param_to_global(
                parameters1_local, matcher_inputCam[:, 3:].unsqueeze(1), matcher_inputCam[:, :3].unsqueeze(1))[:, 0, :,
                                    :]  # bs, n1, 3
            offset1_warped_rt = torch.norm(parameters1_warped_rt, dim=2, keepdim=True, p=2)  # bs, n1, 1
            normal1_warped_rt = F.normalize(parameters1_warped_rt, dim=-1, p=2)  # bs, n1, 3
            nTn_rt = torch.bmm(normal1_warped_rt, normal2_warped.transpose(1, 2))  # bs, n1, n2
            offset_dist = torch.abs(offset1_warped_rt - offset2_warped.transpose(1, 2))  # b, n1, n2
            offset_dist[nTn_rt < 0] = torch.abs(offset1_warped_rt + offset2_warped.transpose(1, 2))[nTn_rt < 0]
            offset_dist = torch.clamp(offset_dist, min=1e-10, max=5.)

        offset_dist = offset_dist.detach() * normal_decay
        normal_dist = normal_dist.detach() * offset_deacy

        if self.planeApp_proj is not None:
            planeApp1 = self.planeApp_proj(planeApp1.permute(0, 2, 1)).permute(0, 2, 1)  # b, n1, 256
            planeApp2 = self.planeApp_proj(planeApp2.permute(0, 2, 1)).permute(0, 2, 1)  # b, n2, 256

        plane_geoFeat1 = planeApp1  # b, n1, 256
        plane_geoFeat2 = planeApp2  # b, n2, 256
        plane_desc1, plane_desc2 = self.gnn(plane_geoFeat1, plane_geoFeat2, mask0=row_masks,
                                            mask1=col_masks)  # b, n1, 256; b, n2, 256
        plane_desc1 = self.planeDesc_proj(plane_desc1.permute(0, 2, 1))  # b, 256, n1
        plane_desc2 = self.planeDesc_proj(plane_desc2.permute(0, 2, 1))  # b, 256, n2

        # Compute matching descriptor distance.
        log_scores = torch.einsum('bdn,bdm->bnm', plane_desc1, plane_desc2)
        log_scores = log_scores / 256 ** .5
        log_scores_App = log_scores
        if offset_dist is not None:
            log_scores = log_scores - offset_dist / self.offset_multiplier
        if normal_dist is not None:
            log_scores = log_scores - normal_dist / self.normal_multiplier

        # run optimal transport
        log_scores_padded = log_optimal_transport_withMask(
            log_scores,
            self.bin_score,
            iters=sinkhorn_iterations,
            row_masks=row_masks,
            col_masks=col_masks
        )
        losses = {}
        if self.training:
            losses_emb = self.embedding_loss_forward(log_scores_padded, gt_corr_matrix)
            losses["losses_emb_%s" % (suffix)] = losses_emb
        return losses, log_scores_padded

    def embedding_loss_forward(self, pred_log_score_matrix, gt_corr_matrix, ite=0):
        pred_log_score_matrix = torch.clamp(pred_log_score_matrix, max=0.)
        gt_corr_matrix = gt_corr_matrix > 0
        loss = torch.mean(-pred_log_score_matrix[gt_corr_matrix]) * 2
        return loss

    def warp_single_view_plane_param_to_global(self, plane, rot_quan=None, tran=None, pose_n=None):
        """
        plane: bs, plane_n, 3
        rot_quan: bs, pose_n, 4
        tran: bs, pose_n, 3
        """
        if rot_quan is not None and tran is not None:
            bs, pose_n, _ = rot_quan.shape
            plane_n = plane.shape[1]

            # get normal and offset
            plane0 = plane.unsqueeze(1).repeat(1, pose_n, 1, 1).view(bs*pose_n, plane_n, 3)  # bs*pose_n, plane_n, 3

            # get rot matrix
            rot_quan = rot_quan.view(-1, 4)  # bs*pose_n, 4
            rot_matrix = self.quaternion2rotmatrix(rot_quan)  # bs*pose_n, 3, 3

            tran = tran.unsqueeze(2).repeat(1, 1, plane_n, 1).view(bs*pose_n, plane_n, 3)  # bs*pose_n, plane_n, 3

            # convert plane of the first view
            start = tran  # bs*pose_n, plane_n, 3
            end = plane0 * (torch.tensor([1, -1, -1]).reshape(1, 1, 3).to(tran.device))  # suncg2habitat # bs*pose_n, plane_n, 3
            end = end.permute(0, 2, 1)  # bs*pose_n, 3, plane_n
            end = (torch.bmm(rot_matrix, end)).permute(0, 2, 1) + tran  # cam2world  # bs*pose_n, plane_n, 3
            a = end  # bs*pose_n, plane_n, 3
            b = end - start  # bs*pose_n, plane_n, 3
            plane0 = ((a * b).sum(dim=-1) / (torch.norm(b, dim=-1) + 1e-5) ** 2).view(bs*pose_n, plane_n, 1) * b  # bs*pose_n, plane_n, 3
            plane0 = plane0.reshape(bs, pose_n, plane_n, 3).contiguous()

            return plane0
        else:
            assert pose_n is not None
            bs = plane.shape[0]
            plane_n = plane.shape[1]
            # convert plane of the second view
            plane1 = plane.unsqueeze(1).repeat(1, pose_n, 1, 1).view(bs*pose_n, plane_n, 3)  # bs*pose_n, plane_n, 3
            plane1 = plane1 * (torch.tensor([1, -1, -1]).reshape(1, 1, 3).to(plane1.device))  # bs*n, n, 3
            plane1 = plane1.reshape(bs, pose_n, plane_n, 3).contiguous()

            return plane1

    def quaternion2rotmatrix(self, quan):
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



def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N

    return Z

def log_optimal_transport_withMask(scores, alpha, iters: int, row_masks=None, col_masks=None):
    """
    https://github.com/qinzheng93/GeoTransformer/blob/73e1439a7a/geotransformer/modules/sinkhorn/learnable_sinkhorn.py
    Sinkhorn Optimal Transport (SuperGlue style) forward.
            Args:
                scores: torch.Tensor (B, M, N)
                row_masks: torch.Tensor (B, M)
                col_masks: torch.Tensor (B, N)
            Returns:
                matching_scores: torch.Tensor (B, M+1, N+1)
    """
    batch_size, num_row, num_col = scores.shape
    inf = 1e5

    if row_masks is None:
        row_masks = torch.ones(size=(batch_size, num_row), dtype=torch.bool).cuda()
    if col_masks is None:
        col_masks = torch.ones(size=(batch_size, num_col), dtype=torch.bool).cuda()

    padded_row_masks = torch.zeros(size=(batch_size, num_row + 1), dtype=torch.bool).cuda()
    padded_row_masks[:, :num_row] = ~row_masks
    padded_col_masks = torch.zeros(size=(batch_size, num_col + 1), dtype=torch.bool).cuda()
    padded_col_masks[:, :num_col] = ~col_masks
    padded_score_masks = torch.logical_or(padded_row_masks.unsqueeze(2), padded_col_masks.unsqueeze(1))

    padded_col = alpha.expand(batch_size, num_row, 1)
    padded_row = alpha.expand(batch_size, 1, num_col + 1)
    padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)
    padded_scores.masked_fill_(padded_score_masks, -inf)

    num_valid_row = row_masks.float().sum(1)
    num_valid_col = col_masks.float().sum(1)
    norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

    log_mu = torch.empty(size=(batch_size, num_row + 1)).cuda()
    log_mu[:, :num_row] = norm.unsqueeze(1)
    log_mu[:, num_row] = torch.log(num_valid_col) + norm
    log_mu[padded_row_masks] = -inf

    log_nu = torch.empty(size=(batch_size, num_col + 1)).cuda()
    log_nu[:, :num_col] = norm.unsqueeze(1)
    log_nu[:, num_col] = torch.log(num_valid_row) + norm
    log_nu[padded_col_masks] = -inf

    outputs = log_sinkhorn_iterations(padded_scores, log_mu, log_nu, iters)
    outputs = outputs - norm.unsqueeze(1).unsqueeze(2)

    return outputs

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
