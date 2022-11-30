# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py

import torch
import torch.nn.functional as F
from torch import nn
from detectron2.utils.comm import get_world_size
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    """
    This class computes the loss for PlaneTR.
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, losses_aux, K_inv_dot_xy_1=None):
        """Create the criterion.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.losses_aux = losses_aux
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks, aux=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, aux=False):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_mask_logits" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_mask_logits"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def loss_centers(self, outputs, targets, indices, num_planes, aux=False):
        assert 'pred_centers' in outputs and targets[0]["plane_centers"] is not None
        idx = self._get_src_permutation_idx(indices)

        src_center = outputs['pred_centers'][idx]  # N, 2
        target_center = torch.cat([tgt["plane_centers"][i] for tgt, (_, i) in zip(targets, indices)], dim=0)

        delta_xy = torch.abs(target_center - src_center)  # N, 2
        dist = torch.norm(delta_xy, dim=-1)  # N
        loss_center_l2 = torch.mean(dist)

        losses = {}
        losses['loss_center_ins'] = loss_center_l2

        if targets[0]["pixel_centers"] is not None and "pixel_centers" in outputs:
            gt_plane_pixel_centers = torch.stack([tgt["pixel_centers"] for tgt in targets], dim=0)  # b, 2, h, w
            pixel_center = outputs['pixel_centers']  # b, 2, h, w
            pixel_center = F.interpolate(pixel_center, size=gt_plane_pixel_centers.shape[-2:], mode="bilinear", align_corners=False)
            assert gt_plane_pixel_centers.shape[0] == pixel_center.shape[0]

            pixel_dist = torch.norm(torch.abs(gt_plane_pixel_centers - pixel_center), dim=1, keepdim=True)  #b, 1, h, w

            if "valid_region" in targets[0] and targets[0]["valid_region"] is not None:
                valid_region = torch.stack([tgt["valid_region"] for tgt in targets], dim=0)
                mask = valid_region > 0
            else:
                mask = torch.ones_like(pixel_dist) > 0

            loss_pixel_center = torch.mean(pixel_dist[mask])
            losses['loss_center_pixel'] = loss_pixel_center

        return losses

    def loss_params(self, outputs, targets, indices, num_planes, aux=False):
        assert 'pred_params' in outputs and targets[0]["plane_params"] is not None
        idx = self._get_src_permutation_idx(indices)
        src_param = outputs['pred_params'][idx]  # N, 3
        target_param = torch.cat([tgt["plane_params"][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        # l1 loss
        loss_param_l1 = torch.mean(torch.sum(torch.abs(target_param - src_param), dim=1))
        # cos loss
        similarity = torch.nn.functional.cosine_similarity(src_param, target_param, dim=1)  # N
        loss_param_cos = torch.mean(1 - similarity)

        losses = {}
        losses['loss_param_l1'] = loss_param_l1
        losses['loss_param_cos'] = loss_param_cos

        if aux:
            return losses

        # --------------------------- Q loss
        gt_depths = torch.stack([tgt["depth"] for tgt in targets], dim=0)  # b, 1, h, w
        assert gt_depths.dim() == 4 or gt_depths.dim() == 3
        if gt_depths.dim() == 3:
            gt_depths = gt_depths.unsqueeze(1)
        b, _, h, w = gt_depths.shape
        ho, wo = h, w
        downsample_on = False

        loss_q = 0.
        assert b == len(targets)
        for bi in range(b):
            num_planes = targets[bi]["labels"].shape[0]
            segmentation = targets[bi]['masks']  # gt_n, h, w
            depth = gt_depths[bi]  # 1, h, w
            k_inv_dot_xy1_map = targets[bi]['k_inv_dot_xy1'].clone().view(3, h, w).to(gt_depths.device)  # 3, h, w
            gt_pts_map = k_inv_dot_xy1_map * depth  # 3, h, w
            if downsample_on:
                gt_pts_map = F.interpolate(gt_pts_map.unsqueeze(0), scale_factor=0.5, mode='nearest')[0]
                segmentation = F.interpolate(segmentation.float().unsqueeze(0), scale_factor=0.5, mode='nearest')[0]
                ho, wo = segmentation.shape[1], segmentation.shape[2]
            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]
            assert idx_tgt.max() + 1 == num_planes

            all_pts = gt_pts_map.reshape(3, -1)
            gt_masks = segmentation[idx_tgt] > 0  # gt_n, h, w
            gt_masks = gt_masks.float()
            # if gt_masks.sum() < 1:
            if gt_masks.sum() < 1:
                loss_bi_new = 0.
            else:
                # gt
                gt_params = targets[bi]['plane_params'][idx_tgt]  # gt_n, 3; normal * offset
                gt_offsets = torch.norm(gt_params, dim=-1, keepdim=True)  # gt_n, 1
                gt_normals = gt_params / gt_offsets  # gt_n, 3
                gt_params_new = gt_normals / gt_offsets  # gt_n, 3
                gt_dist = torch.abs(
                    torch.matmul(gt_params_new, all_pts).reshape(gt_params_new.shape[0], ho, wo) - 1.0)  # gt_n, h, w
                gt_dist_masked = gt_dist * gt_masks  # gt_n, h, w
                gt_dist_err_map = gt_dist_masked.sum(dim=0)  # h, w
                valid_region = (gt_dist_err_map < 0.2) & (gt_masks.sum(dim=0) > 0)
                if valid_region.sum() == 0:
                    loss_bi_new = 0.
                else:
                    pred_params = outputs['pred_params'][bi][idx_out]  # gt_n, 3
                    pred_offsets = torch.norm(pred_params, dim=-1, keepdim=True)  # gt_n, 1
                    pred_normals = pred_params / pred_offsets  # gt_n, 3
                    pred_params_new = pred_normals / pred_offsets  # gt_n, 3
                    dist = torch.abs(
                        torch.matmul(pred_params_new, all_pts).reshape(pred_params.shape[0], ho, wo) - 1.0)  # gt_n, h, w
                    dist_masked = dist * gt_masks
                    # loss_bi_new = dist_masked.sum() / (gt_masks.sum())
                    loss_bi_new = torch.mean(dist_masked.sum(dim=0)[valid_region])

            loss_q += loss_bi_new

        loss_q = loss_q / b
        losses['loss_q'] = loss_q

        return losses

    def loss_depth(self, outputs, targets, indices, num_planes, aux=False):
        gt_pixel_depth = torch.stack([tgt['depth'] for tgt in targets], dim=0)
        pred_pixel_depth = outputs['pixel_depth']
        pred_pixel_depth = F.interpolate(pred_pixel_depth, size=gt_pixel_depth.shape[-2:], mode="bilinear", align_corners=False)
        mask = (gt_pixel_depth > 1e-4).float()
        loss = torch.sum(torch.abs(pred_pixel_depth[:, 0] - gt_pixel_depth) * mask) / torch.clamp(mask.sum(), min=1)

        losses = {'loss_depth_pixel': loss}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, aux=False):
        loss_map = {"labels": self.loss_labels,
                    "masks": self.loss_masks,
                    "centers": self.loss_centers,
                    "params": self.loss_params,
                    "depth": self.loss_depth,
                    }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, aux=aux)

    def forward(self, outputs, targets, matcher_on=True, losses_on=True):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if matcher_on is not True:
            return {}, None

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        if losses_on is not True:
            return {}, indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses_aux:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, num_masks, aux=True)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses, indices
