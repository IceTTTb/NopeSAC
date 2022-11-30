# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
import detectron2.utils.comm as comm
from detectron2.layers import FrozenBatchNorm2d
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import get_world_size, get_rank

from NopeSAC_Net.modeling.criterion import SetCriterion
from NopeSAC_Net.modeling.matcher import HungarianMatcher
from NopeSAC_Net.modeling.planeTR_net import build_planeTR_head
from NopeSAC_Net.modeling.camera_net import build_camera_head
from NopeSAC_Net.modeling.matching_net import build_matching_head

import time
import numpy as np
import cv2
import pycocotools.mask as mask_util
import os
import quaternion
import pickle
__all__ = ["PlaneTR_NopeSAC"]

@META_ARCH_REGISTRY.register()
class PlaneTR_NopeSAC(nn.Module):
    """
    Main architecture.
    """
    @configurable
    def __init__(
        self,
        *,
        criterion: nn.Module,
        num_queries: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        device,
        cfg
    ):
        super().__init__()
        # set seed
        torch.manual_seed(get_rank() * 2 + 40)
        # model cfg
        self.cfg = cfg
        self.mask_on = cfg.MODEL.MASK_ON
        self.embedding_on = cfg.MODEL.EMBEDDING_ON
        self.depth_on = cfg.MODEL.DEPTH_ON
        self.camera_on = cfg.MODEL.CAMERA_ON
        self.camera_refine_on = cfg.MODEL.CAMERA_HEAD.REFINE_ON
        self.camera_cls_on = cfg.MODEL.CAMERA_HEAD.CLASSIFICATION_ON

        # 3D plane detection and matching modules
        self.backbone = build_backbone(cfg)
        # plane head
        self.sem_seg_head = build_planeTR_head(cfg, self.backbone.output_shape())
        if self.embedding_on:
            self.matching_head = build_matching_head(cfg)
        if not self.mask_on:
            self.embedding_on = False

        # camera head
        if self.camera_on:
            self.cam_backbone = None
            self.camera_head_list = nn.ModuleList([])
            camera_head_reg = build_camera_head(cfg, self.backbone.output_shape())
            self.camera_head_list.append(camera_head_reg)
        else:
            self.camera_refine_on = False

        # planeTR criterion
        self.criterion = criterion
        # planeTR query number
        self.num_queries = num_queries

        # image normalizer
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        pixel_mean = torch.Tensor(pixel_mean).to(device).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # postprocess setting
        self.overlap_threshold = cfg.TEST.OVERLAP_THRESHOLD
        self.plane_score_threshold = cfg.TEST.PLANE_SCORE_THRESHOLD
        self.mask_prob_threshold = cfg.TEST.MASK_PROB_THRESHOLD

        # debug setting
        self.debug_iter = 0
        self.run_iter = 0
        self.infer_iter = 0

        # losses setting
        self.matcher_on = cfg.MODEL.HUNGARIAN_MATCHER_ON
        self.loss_detection_on = cfg.MODEL.LOSS_DETECTION_ON
        self.loss_camera_on = cfg.MODEL.LOSS_CAMERA_ON
        self.loss_matching_on = cfg.MODEL.LOSS_EMB_ON

        # freeze model
        self._freeze = cfg.MODEL.FREEZE
        for layers in self._freeze:
            layer = layers.split(".")
            final = self
            for l in layer:
                final = getattr(final, l)
            for params in final.parameters():
                params.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(final)

        # camera info
        kmeans_trans_path = self.cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH
        kmeans_rots_path = self.cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH
        assert os.path.exists(kmeans_trans_path)
        assert os.path.exists(kmeans_rots_path)
        with open(kmeans_trans_path, "rb") as f:
            self.kmeans_trans = pickle.load(f)
        f.close()
        with open(kmeans_rots_path, "rb") as f:
            self.kmeans_rots = pickle.load(f)
        f.close()

        self.precompute_xy_map()
        self.pre_defined_k_inv_dot_xy1 = get_coordinate_map(device=self.device, h=480, w=640)  # 3, h, w

    @classmethod
    def from_config(cls, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.SEM_SEG_HEAD.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.SEM_SEG_HEAD.NO_OBJECT_WEIGHT
        dice_weight = cfg.MODEL.SEM_SEG_HEAD.DICE_WEIGHT
        mask_weight = cfg.MODEL.SEM_SEG_HEAD.MASK_WEIGHT
        param_weight_L1 = cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_L1
        param_weight_cos = cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_COS
        param_weight_q = cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_Q
        param_weight_offset = cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_OFFSET
        param_weight_angle = cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_ANGLE
        center_ins_weight = cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_CENTER_INS

        # building criterion
        use_param_in_matcher = cfg.MODEL.SEM_SEG_HEAD.PARAM_IN_MATCHER
        param_hm_weight_L1 = cfg.MODEL.SEM_SEG_HEAD.PARAM_HM_WEIGHT_L1
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_center=center_ins_weight,
            cost_param=param_hm_weight_L1,
            cost_param_offset=param_weight_offset,
            cost_param_normal_angle=param_weight_angle,
            param_on=use_param_in_matcher,
        )
        weight_dict = {'loss_ce': 1,
                       'loss_param_l1': param_weight_L1,
                       'loss_param_cos': param_weight_cos,
                       'loss_q': param_weight_q,
                       'loss_center_ins': center_ins_weight,
                       'loss_center_pixel': 1.0,
                       'loss_depth_pixel': 1.0,
                       'loss_mask': mask_weight,
                       'loss_dice': dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.SEM_SEG_HEAD.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"]
        losses_aux = ['labels', 'masks']
        if cfg.MODEL.SEM_SEG_HEAD.CENTER_ON:
            losses.append("centers")
            losses_aux.append("centers")
        if cfg.MODEL.SEM_SEG_HEAD.PARAM_ON:
            losses.append("params")
            losses_aux.append("params")
        if cfg.MODEL.DEPTH_ON:
            losses.append("depth")
        device = torch.device(cfg.MODEL.DEVICE)
        criterion = SetCriterion(
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            losses_aux=losses_aux
        )
        return {
            "criterion": criterion,
            "num_queries": cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "device": device,
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # training or inference
        if not self.training:
            if comm.is_main_process():
                self.infer_iter += 1
            res = self.inference(batched_inputs)
            return res
        if comm.is_main_process():
            self.run_iter += 1

        # build single inputs
        batched_inputs_single = {"0": [], "1": []}
        for batched_input in batched_inputs:
            for i in ["0", "1"]:
                batched_inputs_single[i].append(batched_input[i])
        losses = {}

        # ***********************************************************************
        # single view plane detection
        # ***********************************************************************
        losses1, cam_feats1, query_feat1, indices1, plane_outputs1 = self.forward_single(
            batched_inputs_single["0"], view="v0", stereoInputs=batched_inputs, run_iter=self.run_iter)
        losses2, cam_feats2, query_feat2, indices2, plane_outputs2 = self.forward_single(
            batched_inputs_single["1"], view="v1", stereoInputs=batched_inputs, run_iter=self.run_iter)
        # update plane detection losses
        if self.mask_on and self.loss_detection_on:
            for key in losses1.keys():
                losses[key] = (losses1[key] + losses2[key]) / 2

        # ***********************************************************************
        # plane matching
        # ***********************************************************************
        gt_pose = self.process_camera(batched_inputs)
        if plane_outputs1 is not None:
            plane_param1 = plane_outputs1['pred_params']  # bs, num_query, 3
            plane_param2 = plane_outputs2['pred_params']  # bs, num_query, 3
        else:
            plane_param1 = plane_param2 = None
        if self.embedding_on:
            if self.matcher_on:
                # assignment between the pred planes and the gt planes is needed
                if indices1 is None or indices2 is None:
                    # do Hungarian matching for cross-view plane matching
                    raise NotImplementedError
                # prepare gt corr matrix of pred plane sets from two views
                gt_corr_matrix = self.process_plane_corr_matrix(batched_inputs, indices1, indices2)  # bs, n1+1, n2+1
            else:
                gt_corr_matrix = None

            if self.loss_matching_on:
                assert self.matcher_on
                # assignment between the pred planes and the gt planes is needed
                if indices1 is None or indices2 is None:
                    # do Hungarian matching for cross-view plane matching
                    raise NotImplementedError
                # prepare gt corr matrix of pred plane sets from two views
                # gt_corr_matrix = self.process_plane_corr_matrix(batched_inputs, indices1, indices2)  # bs, n1+1, n2+1

                # matching forward
                losses_matching, log_scores_padded = self.matching_head(
                    query_feat1, query_feat2, gt_pose, plane_param1, plane_param2,
                    indices1=indices1,
                    indices2=indices2,
                    gt_corr_matrix=gt_corr_matrix,
                    suffix="0",
                )
                # update matching loss
                losses.update(losses_matching)
        else:
            gt_corr_matrix = None

        # ***********************************************************************
        # get camera pose from plane matching results
        # ***********************************************************************
        if self.camera_on:
            assert self.loss_camera_on, "camera loss should be used in current version"
            # forward
            losses_cam_init, trans_list, rot_list, log_scores_padded_list, _, initPose_ref_outputs = \
                self.camera_head_list[0](
                    cam_feats1,
                    cam_feats2,
                    plane_param1,
                    plane_param2,
                    gt_pose=gt_pose,
                    gt_corr_matrix=gt_corr_matrix,
                    ite=self.run_iter,
                    batched_inputs=batched_inputs
            )
            if self.loss_camera_on:
                losses.update(losses_cam_init)

        return losses

    def forward_single(self, batched_inputs, view="v0", stereoInputs=None, run_iter=0):
        images = self.preprocess_image(batched_inputs)
        features = None
        if self.camera_on:
            if self.cam_backbone is None:
                features = self.backbone(images.tensor)
                cam_feats = features
            else:
                cam_feats = self.cam_backbone(images.tensor)
        else:
            cam_feats = None
        if self.mask_on is not True:
            query_feat = None
            return {}, cam_feats, query_feat, None, None

        if features is None:
            features = self.backbone(images.tensor)
        outputs, query_feat = self.sem_seg_head(features)

        # mask classification target
        if "instances" in batched_inputs[0]:
            targets = self.prepare_targets(batched_inputs, images)
        else:
            targets = None

        # bipartite matching-based loss
        losses, indices = self.criterion(outputs, targets, matcher_on=self.matcher_on, losses_on=self.loss_detection_on)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses, cam_feats, query_feat, indices, outputs

    def inference(self, batched_inputs, detected_instances=None):
        assert not self.training
        assert len(batched_inputs) == 1
        assert detected_instances is None
        self.debug_iter += 1
        if self.cfg.DEBUG_CAMERA_ON:
            print('**********************> ', self.debug_iter)
        batched_inputs_single = {"0": [], "1": []}
        for batched_input in batched_inputs:
            for i in ["0", "1"]:
                batched_inputs_single[i].append(batched_input[i])

        # ***********************************************************************
        # single view plane detection
        # ***********************************************************************
        pred_depth1, processed_results1, camera_input1, query_feat1, ori_results1 = self.inference_single(
            batched_inputs_single["0"], view="v0", stereoInputs=batched_inputs)
        pred_depth2, processed_results2, camera_input2, query_feat2, ori_results2 = self.inference_single(
            batched_inputs_single["1"], view="v1", stereoInputs=batched_inputs)

        # ***********************************************************************
        # get camera pose from plane matching
        # ***********************************************************************
        if self.camera_on:
            gt_pose = None
            if processed_results1[0] is not None:
                plane_param1 = processed_results1[0]['pred_plane'].unsqueeze(0).to(device=self.device)  # bs, n1, 3
                plane_param2 = processed_results2[0]['pred_plane'].unsqueeze(0).to(device=self.device)  # bs, n2, 3
                selected_query_feat1 = processed_results1[0]["pred_plane_feats"]
                selected_query_feat2 = processed_results2[0]["pred_plane_feats"]
            else:
                plane_param1 = None
                plane_param2 = None
                selected_query_feat1 = None
                selected_query_feat2 = None
            emb_affinity_matrix = [None] * len(batched_inputs)

            output_cameras, trans_list, rot_list, log_scores_padded_list, output_planeAss, pose_ref_outputs = \
                self.camera_head_list[0](
                    camera_input1, camera_input2, plane_param1, plane_param2,
                    planeApp1=selected_query_feat1,
                    planeApp2=selected_query_feat2,
                    gt_pose=gt_pose,
                    batched_inputs=batched_inputs,
                    matching_net=self.matching_head
                )
            batched_output_cameras = []
            for i in range(len(batched_inputs)):
                output_cameras_i = {}
                for key, value in output_cameras.items():
                    output_cameras_i[key] = {"tran": value['tran'][i].cpu().numpy(),
                                             "rot": value['rot'][i].cpu().numpy()}
                batched_output_cameras.append(output_cameras_i)
            if len(output_planeAss) > 0:
                batched_output_planeAss = []
                for i in range(len(batched_inputs)):
                    output_ass_i = {}
                    for key, value in output_planeAss.items():
                        output_ass_i[key] = value[i]
                    batched_output_planeAss.append(output_ass_i)
            else:
                batched_output_planeAss = [None] * len(batched_inputs)
        else:
            trans_zero = np.array([0., 0., 0.]).reshape(3)
            rot_zero = np.array([1., 0., 0., 0.]).reshape(4)
            batched_output_cameras = []
            for i in range(len(batched_inputs)):
                output_cameras_i = {"camera": {"tran": trans_zero, "rot": rot_zero}}
                batched_output_cameras.append(output_cameras_i)

        # ***********************************************************************
        # output
        # ***********************************************************************
        if self.embedding_on:
            results = []
            for pred1, pred2, out_ass, d1, d2, out_cams, m_emb in zip(
                    processed_results1,
                    processed_results2,
                    batched_output_planeAss,
                    pred_depth1,
                    pred_depth2,
                    batched_output_cameras,
                    emb_affinity_matrix
            ):
                results.append(
                    {
                        "0": pred1,
                        "1": pred2,
                        "pred_aff": m_emb,
                        "depth": {"0": d1, "1": d2},  # 480, 640
                    }
                )
                results[-1].update(out_cams)
                results[-1].update(out_ass)
        else:
            results = []
            for pred1, pred2, d1, d2, out_cams in zip(
                    processed_results1,
                    processed_results2,
                    pred_depth1,
                    pred_depth2,
                    batched_output_cameras
            ):
                results.append(
                    {
                        "0": pred1,
                        "1": pred2,
                        "depth": {"0": d1, "1": d2},  # 480, 640
                    }
                )
                results[-1].update(out_cams)

        return results

    def inference_single(self, batched_inputs, view="0", stereoInputs=None):
        assert len(batched_inputs) == 1
        t_s = time.time()
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if self.cam_backbone is None:
            cam_feats = features
        else:
            cam_feats = self.cam_backbone(images.tensor)
        pred_depth = [None] * len(batched_inputs)
        if self.mask_on is not True:
            processed_results = [None] * len(batched_inputs)
            return pred_depth, processed_results, cam_feats, None, None

        pred_plane_dict, query_feat = self.sem_seg_head(features)

        b, n, _ = pred_plane_dict['pred_logits'].shape
        if "pred_params" not in pred_plane_dict:
            pred_plane_dict['pred_params'] = torch.randn([b, n, 3]).to(self.device)
        processed_results = self._postprocess_planeHeadMask(pred_plane_dict, pred_depth, batched_inputs,
                                                            images.image_sizes, query_feat)
        return pred_depth, processed_results, cam_feats, query_feat, pred_plane_dict

    def prepare_targets(self, batched_inputs, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in batched_inputs:
            instances = targets_per_image['instances']
            if 'plane_masks' in targets_per_image:
                padded_masks = targets_per_image["plane_masks"].tensor.to(self.device)  # shape: [n, h, w]
                plane_num = padded_masks.shape[0]
                assert plane_num == len(instances)
                if plane_num > 50:
                    plane_num = 50
            else:
                semantic_map = targets_per_image['semantic_map'].to(self.device)
                plane_ids = torch.unique(semantic_map)  # 0: non-plane
                if plane_ids[0] == 0:
                    plane_ids = plane_ids[1:]
                plane_num = len(plane_ids)
                assert plane_num == len(instances)
                if plane_num > 50:
                    plane_num = 50
                plane_ids = plane_ids[:plane_num]
                padded_masks = (plane_ids.reshape(-1, 1, 1) == semantic_map.unsqueeze(0)).to(self.device)  # shape: [n, h, w]

            if self.cfg.MODEL.SEM_SEG_HEAD.CENTER_ON:
                plane_xy_map = self.normalized_xy_map_tensor.unsqueeze(0).to(self.device) * padded_masks.unsqueeze(
                    1).to(torch.float32)  # n, 2, h, w
                plane_centers = plane_xy_map.flatten(2, 3).sum(-1) / padded_masks.unsqueeze(1).flatten(2, 3).sum(-1)
                padded_pixel_centers = plane_centers.unsqueeze(-1).unsqueeze(-1) * padded_masks.unsqueeze(1).to(
                    torch.float32)  # n, 2, h, w
                pixel_centers = padded_pixel_centers.sum(0)  # 2, h, w
            else:
                pixel_centers = None
                plane_centers = None

            gt_classes = instances.gt_classes.to(self.device)  # shape: [gt_n]  value: 0 -> plane
            depth = targets_per_image["depth"].to(self.device)
            if 'camera_K' in targets_per_image.keys():
                camera_K = targets_per_image['camera_K']  # 3, 3
                k_inv_dot_xy1 = get_coordinate_map(device=self.device, h=h, w=w, K_matrix=camera_K)  # 3, h, w
            else:
                k_inv_dot_xy1 = self.pre_defined_k_inv_dot_xy1

            if self.cfg.MODEL.SEM_SEG_HEAD.PARAM_ON:
                gt_planes = instances.gt_planes.to(self.device)[:plane_num]   # shape: [gt_n, 3]
            else:
                gt_planes = None
            new_targets.append(
                {
                    "labels": gt_classes[:plane_num],
                    "masks": padded_masks[:plane_num],
                    "plane_centers": plane_centers,  # plane_centers,  # shape: [gt_n, 2]
                    "pixel_centers": pixel_centers,
                    "plane_params": gt_planes,  # plane normal * plane offset
                    "depth": depth,
                    "k_inv_dot_xy1": k_inv_dot_xy1,  # 3, h, w
                }
            )
        return new_targets

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = {}
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def process_depth(self, batched_inputs):
        depth = [x["depth"].to(self.device) for x in batched_inputs]
        depth = torch.stack(depth)
        return depth

    def process_camera(self, batched_inputs):
        if self.cfg.MODEL.CAMERA_HEAD.NAME in ["PlaneCameraHead"]:
            gt_pose = []
            for per_stereo_input in batched_inputs:
                rel_pose = per_stereo_input["rel_pose"]
                rot = np.array(rel_pose["rotation"]).astype(np.float32)
                rot = torch.from_numpy(rot).reshape(1, 4).to(self.device)
                if rot[0, 0] < 0 and not self.camera_cls_on:
                    rot = -rot
                tran = torch.tensor(rel_pose["position"]).reshape(1, 3).to(self.device)
                gt_p = torch.cat([tran, rot], dim=-1)  # 1, 7
                gt_pose.append(gt_p)
            gt_pose = torch.cat(gt_pose, dim=0)  # bs, 7
            return gt_pose
        else:
            raise NotImplementedError

    def process_plane_corr_matrix(self, batched_inputs, matched_idx1, matched_idx2):
        # get corr idx of gt planes
        bs = len(batched_inputs)
        gt_corrIdx_view1 = []
        gt_corrIdx_view2 = []
        gt_corr_batchIdX = []
        for i in range(bs):
            gt_corrs = batched_inputs[i]['gt_corrs']
            gt_corrs = torch.tensor(gt_corrs)  # n_gt, 2
            gt_corrs_mask = (gt_corrs[:, 0] < 50) & ((gt_corrs[:, 1] < 50))  # n
            gt_corrs = gt_corrs[gt_corrs_mask]
            bi_gt_corrIdx_view1 = gt_corrs[:, 0].contiguous().to(device=self.device)
            bi_gt_corrIdx_view2 = gt_corrs[:, 1].contiguous().to(device=self.device)
            bi_idx = torch.full_like(bi_gt_corrIdx_view1, i)
            gt_corrIdx_view1.append(bi_gt_corrIdx_view1)
            gt_corrIdx_view2.append(bi_gt_corrIdx_view2)
            gt_corr_batchIdX.append(bi_idx)
        gt_corrIdx_view1 = torch.cat(gt_corrIdx_view1)
        gt_corrIdx_view2 = torch.cat(gt_corrIdx_view2)
        gt_corr_batchIdX = torch.cat(gt_corr_batchIdX)

        # get corr idx of pred planes
        batch_idx1 = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_idx1)]).to(device=self.device)
        gt_plane_idx1 = torch.cat([tgt for (_, tgt) in matched_idx1]).to(device=self.device)
        pred_plane_idx1 = torch.cat([src for (src, _) in matched_idx1]).to(device=self.device)

        gt2pred1 = -torch.ones(bs, self.num_queries).to(device=self.device, dtype=pred_plane_idx1.dtype)  # bs, nq
        gt2pred1[batch_idx1, gt_plane_idx1] = pred_plane_idx1
        gt2pred1[gt2pred1 == -1] = self.num_queries

        pred_corrIdx_view1 = gt2pred1[gt_corr_batchIdX, gt_corrIdx_view1]

        # get corr idx of pred planes
        try:
            batch_idx2 = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_idx2)]).to(device=self.device)
        except:
            import pdb; pdb.set_trace()
        gt_plane_idx2 = torch.cat([tgt for (_, tgt) in matched_idx2]).to(device=self.device)
        pred_plane_idx2 = torch.cat([src for (src, _) in matched_idx2]).to(device=self.device)

        gt2pred2 = -torch.ones(bs, self.num_queries).to(device=self.device, dtype=pred_plane_idx2.dtype)  # bs, nq
        gt2pred2[batch_idx2, gt_plane_idx2] = pred_plane_idx2
        gt2pred2[gt2pred2 == -1] = self.num_queries

        pred_corrIdx_view2 = gt2pred2[gt_corr_batchIdX, gt_corrIdx_view2]

        # get gt corr matrix of pred planes
        corr_matrix_of_predPlanes = torch.zeros(bs, self.num_queries+1, self.num_queries+1).to(device=self.device)
        corr_matrix_of_predPlanes[gt_corr_batchIdX, pred_corrIdx_view1, pred_corrIdx_view2] = 1

        sum_row = 1 - torch.sum(corr_matrix_of_predPlanes[:, :-1, :], dim=1, keepdim=True)
        sum_col = 1 - torch.sum(corr_matrix_of_predPlanes[:, :, :-1], dim=2, keepdim=True)
        corr_matrix_of_predPlanes[:, -1:, :] = sum_row
        corr_matrix_of_predPlanes[:, :, -1:] = sum_col
        corr_matrix_of_predPlanes[:, -1, -1] = 0  # b, max_n+1, max_n+1
        corr_matrix_of_predPlanes = corr_matrix_of_predPlanes > 0

        return corr_matrix_of_predPlanes

    def _postprocess_planeHeadMask(
            self, planeTR_outputs, pred_depth, batched_inputs, image_sizes, query_feat_in, mask_threshold=0.5, nms=False
    ):
        """
        Rescale the output instances to the target size.
        """
        bs = planeTR_outputs['pred_logits'].shape[0]
        results = []
        query_feat = query_feat_in.clone()

        for i in range(bs):
            height = batched_inputs[i].get("height", image_sizes[i][0])
            width = batched_inputs[i].get("width", image_sizes[i][1])

            res_bi = {}
            res_bi["image_id"] = batched_inputs[i]["image_id"]
            res_bi["file_name"] = batched_inputs[i]["file_name"]

            # decompose outputs
            pred_logits = planeTR_outputs['pred_logits'][i].detach()  # num_queries, 2
            pred_param = planeTR_outputs['pred_params'][i].detach()  # num_queries, 3
            pred_masks_logits = planeTR_outputs['pred_mask_logits'][i].detach()  # nq, h, w
            pred_masks_prob = torch.sigmoid(pred_masks_logits)  # nq, h, w
            pred_masks_prob = F.interpolate(pred_masks_prob[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
            oriIdx = torch.arange(0, self.num_queries).to(pred_logits.device)

            # remove non-plane instance
            pred_prob = F.softmax(pred_logits, dim=-1)  # num_queries, 2
            score, labels = pred_prob.max(dim=-1)
            label_mask = (labels == 0) & (score > self.plane_score_threshold)

            zero_flag = False
            if sum(label_mask) == 0:
                _, max_pro_idx = pred_prob[:, 0].max(dim=0)
                label_mask[max_pro_idx] = 1
                score[max_pro_idx] = pred_prob[max_pro_idx, 0]
                zero_flag = True

            """"""
            valid_param = pred_param[label_mask, :]  # valid_plane_num, 3
            valid_plane_prob = score[label_mask]  # valid_plane_num
            valid_masks_prob_ori = pred_masks_prob[label_mask]  # valid_plane_num, h, w
            valid_masks_prob = valid_plane_prob.view(-1, 1, 1) * valid_masks_prob_ori  # valid_plane_num, h, w
            valid_plane_num = valid_param.shape[0]
            valid_plane_feat = query_feat[i, label_mask]  # valid_plane_num, c
            valid_plane_oriIdx = oriIdx[label_mask]

            assert valid_plane_num > 0
            # get plane segmentation
            valid_mask_ids = valid_masks_prob.argmax(0)  # h, w
            """"""
            processed_plane = []
            instances = []
            valid_query_feats = []
            valid_plane_idxs = []
            valid_plane_masks = []
            valid_plane_ins_centers = []
            max_overlap_id = 0
            max_overlap = 0.
            for pi in range(valid_plane_num):
                plane_mask_pi = (valid_mask_ids == pi) & (valid_masks_prob[pi] > self.mask_prob_threshold)  # h, w
                plane_mask_pi_np = np.asfortranarray(plane_mask_pi.cpu().numpy())

                mask_area = plane_mask_pi.sum().item()
                original_area = (valid_masks_prob_ori[pi] >= self.mask_prob_threshold).sum().item()
                if zero_flag is False:
                    if mask_area < 1 or original_area < 1:
                        continue
                    overlap = mask_area / original_area
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_overlap_id = pi
                    if overlap < self.overlap_threshold:
                        continue
                else:
                    if mask_area == 0:
                        plane_mask_pi_np[0, 0] = 1
                        plane_mask_pi[0, 0] = 1
                rle_pi = mask_util.encode(plane_mask_pi_np)
                bbox_i = mask_util.toBbox(rle_pi).tolist()  # x, y, w, h
                plane_param_pi = valid_param[pi].cpu()  # shape: [3]
                score_pi = valid_plane_prob[pi]
                segmentation = {}
                segmentation["size"] = [height, width]
                segmentation["counts"] = rle_pi["counts"]

                processed_plane.append(plane_param_pi)
                ins_pi = {}
                ins_pi["image_id"] = batched_inputs[i]["image_id"]
                ins_pi["file_name"] = batched_inputs[i]["file_name"]
                ins_pi["category_id"] = 0
                ins_pi["score"] = score_pi.item()
                ins_pi["segmentation"] = segmentation
                ins_pi["bbox"] = bbox_i
                ins_pi["bbox_mode"] = 1
                instances.append(ins_pi)
                valid_query_feat_pi = valid_plane_feat[pi]  # c
                valid_query_feats.append(valid_query_feat_pi)
                valid_plane_idxs.append(valid_plane_oriIdx[pi])
                valid_plane_masks.append(plane_mask_pi)

                # get plane center
                plane_mask = plane_mask_pi_np.astype(np.float)
                pixel_num = plane_mask.sum()
                x_map = self.normalized_xy_map[0] * plane_mask
                y_map = self.normalized_xy_map[1] * plane_mask
                x_sum = x_map.sum()
                y_sum = y_map.sum()
                plane_x = x_sum / (pixel_num + 1e-10)
                plane_y = y_sum / (pixel_num + 1e-10)

                instance_center = np.zeros([2]).astype(np.float32)
                instance_center[0] = plane_x
                instance_center[1] = plane_y
                valid_plane_ins_centers.append(instance_center)

            if len(instances) == 0:
                pi = max_overlap_id
                plane_mask_pi = valid_mask_ids == pi  # h, w
                plane_mask_pi = np.asfortranarray(plane_mask_pi.cpu().numpy())
                mask_area = plane_mask_pi.sum()
                original_area = (valid_masks_prob_ori[pi] >= 0.5).sum().item()
                rle_pi = mask_util.encode(plane_mask_pi)
                bbox_i = mask_util.toBbox(rle_pi).tolist()  # x, y, w, h
                plane_param_pi = valid_param[pi].cpu()  # shape: [3]
                score_pi = valid_plane_prob[pi]

                segmentation = {}
                segmentation["size"] = [height, width]
                segmentation["counts"] = rle_pi["counts"]

                processed_plane.append(plane_param_pi)
                ins_pi = {}
                ins_pi["image_id"] = batched_inputs[i]["image_id"]
                ins_pi["file_name"] = batched_inputs[i]["file_name"]
                ins_pi["category_id"] = 0
                ins_pi["score"] = score_pi.item()
                ins_pi["segmentation"] = segmentation
                ins_pi["bbox"] = bbox_i
                ins_pi["bbox_mode"] = 1

                instances.append(ins_pi)

                valid_query_feat_pi = valid_plane_feat[pi]  # c
                valid_query_feats.append(valid_query_feat_pi)

                valid_plane_idxs.append(valid_plane_oriIdx[pi])

                valid_plane_masks.append(torch.from_numpy(plane_mask_pi))

                # get plane center
                plane_mask = plane_mask_pi.astype(np.float)
                pixel_num = plane_mask.sum()
                x_map = self.normalized_xy_map[0] * plane_mask
                y_map = self.normalized_xy_map[1] * plane_mask
                x_sum = x_map.sum()
                y_sum = y_map.sum()
                plane_x = x_sum / pixel_num
                plane_y = y_sum / pixel_num

                instance_center = np.zeros([2]).astype(np.float32)
                instance_center[0] = plane_x
                instance_center[1] = plane_y
                valid_plane_ins_centers.append(instance_center)

            res_bi["instances"] = instances
            processed_plane = torch.stack(processed_plane, dim=0)  # n, 3
            res_bi["pred_plane"] = processed_plane
            valid_query_feats = torch.stack(valid_query_feats, dim=0).unsqueeze(0).contiguous()  # 1, n, c
            res_bi["pred_plane_feats"] = valid_query_feats
            res_bi["pred_plane_oriIdxs"] = valid_plane_idxs
            valid_plane_masks = torch.stack(valid_plane_masks, dim=0)  # n, h, w
            res_bi["pred_plane_masks"] = valid_plane_masks

            valid_plane_ins_centers = torch.tensor(valid_plane_ins_centers).reshape(-1, 2)
            res_bi["pred_plane_ins_center"] = valid_plane_ins_centers
            results.append(res_bi)

        return results

    def precompute_xy_map(self, h=480, w=640):
        xy_map = np.zeros((2, h, w)).astype(np.float32)
        for y in range(h):
            for x in range(w):
                xy_map[0, y, x] = float(x) / w
                xy_map[1, y, x] = float(y) / h
        self.normalized_xy_map = xy_map
        self.normalized_xy_map_tensor = torch.from_numpy(self.normalized_xy_map).to(self.device)

# ------------------------------debug tools ----------------------------
def get_coordinate_map(device, h=480, w=640, K_matrix=None):
    if K_matrix is None:
        focal_length = 517.97
        offset_x = 320
        offset_y = 240
        K = [[focal_length, 0, offset_x], [0, focal_length, offset_y], [0, 0, 1]]
        K_matrix = torch.tensor(K).to(dtype=torch.float32)
    else:
        K_matrix = torch.from_numpy(K_matrix).to(dtype=torch.float32)
        assert K_matrix.dim() == 2
    K = K_matrix.to(device)
    K_inv = K.inverse()

    x = torch.arange(w, dtype=torch.float32).view(1, w) / w * 640
    y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * 480

    x = x.to(device)
    y = y.to(device)
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32).to(device)))  # (3, h, w)
    xy1 = xy1.view(3, -1)  # (3, h*w)

    k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w)
    return k_inv_dot_xy1.reshape(3, h, w)
