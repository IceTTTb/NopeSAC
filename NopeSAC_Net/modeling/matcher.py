# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

def batch_dice_loss(inputs, targets):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
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
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 20, cost_dice: float = 1, cost_center: float = 0.5,
                 cost_param: float = 1., cost_param_offset: float = 0.01, cost_param_normal_angle: float = 0.0028, param_on: bool = True):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_center = cost_center
        self.cost_param_offset = cost_param_offset
        self.cost_param_normal_angle = cost_param_normal_angle

        self.param_on = param_on
        self.cost_param = cost_param

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0 or cost_center != 0 or cost_param != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Work out the mask padding size
        masks = [v["masks"] for v in targets]
        h_max = max([m.shape[1] for m in masks])
        w_max = max([m.shape[2] for m in masks])

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_mask = outputs["pred_mask_logits"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Compute the L2 cost between centers
            if 'pred_centers' in outputs.keys() and targets[0]["plane_centers"] is not None:
                out_center = outputs["pred_centers"][b]  # [num_queries, 2]
                tgt_center = targets[b]["plane_centers"]  # [num_target_planes, 2]
                cost_center = torch.cdist(out_center, tgt_center, p=2)  # num_queries * num_target_planes
            else:
                raise

            if self.param_on and 'pred_params' in outputs.keys() and targets[0]["plane_params"] is not None:
                out_param = outputs["pred_params"][b]  # [num_queries, 3]
                tgt_param = targets[b]["plane_params"]  # [num_target_planes, 3]
                cost_param = torch.cdist(out_param, tgt_param, p=1)  # num_queries * num_target_planes

                out_param_normal = F.normalize(out_param, dim=-1)  # [num_queries, 3]
                tgt_param_normal = F.normalize(tgt_param, dim=-1)  # [num_target_planes, 3]
                cost_param_normal_cos = torch.matmul(out_param_normal, tgt_param_normal.permute(1, 0))  # num_queries * num_target_planes
                cost_param_normal_cos = torch.clamp(cost_param_normal_cos, max=0.999999, min=-0.999999)
                cost_param_normal_angle = torch.acos(cost_param_normal_cos) * 180.0 / np.pi

                out_param_offset = torch.norm(out_param, p=2, dim=-1, keepdim=True)  # [num_queries, 1]
                tgt_param_offset = torch.norm(tgt_param, p=2, dim=-1, keepdim=True)  # [num_target_planes, 1]
                cost_param_offset = torch.cdist(out_param_offset, tgt_param_offset, p=1)  # num_queries * num_target_planes
            else:
                raise

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_center * cost_center
                + self.cost_param * cost_param
                + self.cost_param_offset * cost_param_offset +
                + self.cost_param_normal_angle * cost_param_normal_angle
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
