import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
from detectron2.utils.registry import Registry
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.layers import FrozenBatchNorm2d
from .camera_modules import *
import quaternion
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ["build_camera_head", "CAMERA_HEAD_REGISTRY", "PlaneCameraHead"]

CAMERA_HEAD_REGISTRY = Registry("CAMERA_HEAD")
CAMERA_HEAD_REGISTRY.__doc__ = """
Registry for camera head.
The call is expected to return an :class:`nn.module`.
"""

def build_camera_head(cfg, input_shape):
    """
    Build CameraHeads defined by `cfg.MODEL.CAMERA_HEADS.NAME`.
    """
    name = cfg.MODEL.CAMERA_HEAD.NAME
    return CAMERA_HEAD_REGISTRY.get(name)(cfg, input_shape)

@CAMERA_HEAD_REGISTRY.register()
class PlaneCameraHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super(PlaneCameraHead, self).__init__()
        self.cfg = cfg
        self.plane_matcher_on = cfg.MODEL.EMBEDDING_ON and cfg.MODEL.MASK_ON
        self.rand_cam_on = cfg.MODEL.CAMERA_HEAD.RAND_ON
        self.cam_rec_on = cfg.MODEL.CAMERA_HEAD.CAM_REC_ON
        self.cam_ref_on = cfg.MODEL.CAMERA_HEAD.REFINE_ON
        self.use_sparsePlane_Top1Cam_testSet = cfg.MODEL.CAMERA_HEAD.INFERENCE_SP_TOPCAM_ON
        self.num_queries = cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES
        self.cam_loss = CameraPoseLoss()
        self.plane_cam_weight = cfg.MODEL.CAMERA_HEAD.PLANE_CAM_WEIGHT
        self.plane_cam_weight_predplane = cfg.MODEL.CAMERA_HEAD.PLANE_CAM_WEIGHT_PREDPLANE
        self.initial_cam_weight = cfg.MODEL.CAMERA_HEAD.INITIAL_CAM_WEIGHT
        self.inference_out_cam_type = cfg.MODEL.CAMERA_HEAD.INFERENCE_OUT_CAM_TYPE
        self.matching_score_threshold = cfg.TEST.MATCHING_SCORE_THRESHOLD
        self.rand_bs = 64
        self.warp_plane_in_cam_ref_on = cfg.MODEL.CAMERA_HEAD.WARP_PLANE_IN_CAM_REF_ON

        # read sparseplane top1 cam
        self.sparsePlaneTop1Cam_dict = None
        if self.use_sparsePlane_Top1Cam_testSet:
            with open(cfg.MODEL.CAMERA_HEAD.INFERENCE_SP_TOPCAM_PATH, "rb") as f:
                self.sparsePlaneTop1Cam_dict = pickle.load(f)

        # pixel camera head (Pose Regression Network)
        self.__initial_PixelCameraHead(cfg, input_shape)

        # pose reg head
        self.trans = nn.Linear(256, 3)
        self.rots = nn.Linear(256, 4)

        # plane rec head (Arbitrary Initialization Module)
        if self.cam_rec_on:
            self.__initial_RotRecHead()
            self.__initial_TransRecHead()

        # plane camera heads (NOPESAC Pose Refinement)
        if self.cam_ref_on:
            self.__initial_PlaneCamRefHead()

    def __initial_PixelCameraHead(self, cfg, input_shape):
        self.pixel_decoder = BasePixelDecoder(cfg, input_shape)
        self.convs_backbone = nn.Sequential(
            conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )
        for block in self.convs_backbone:
            if isinstance(block, nn.modules.container.Sequential):
                for layer in block:
                    if isinstance(layer, nn.Conv2d):
                        weight_init.c2_msra_fill(layer)
        self.convs_trans = nn.Sequential(
            conv2d(in_channels=300, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )
        self.convs_rots = nn.Sequential(
            conv2d(in_channels=300, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )
        self.fc_trans = nn.Linear(768, 256)
        self.fc_rots = nn.Linear(768, 256)

    def __initial_RotRecHead(self):
        self.rot_emb_proj = MLP(4, 256, 256, 6)

    def __initial_TransRecHead(self):
        self.trans_emb_proj = MLP(3, 256, 256, 6)

    def __initial_PlaneCamRefHead(self):
        self.geo_encoder = MLP(8, 1024, 1024, 6)
        self.geo_proj_s1 = MLP(1024, 1024, 1024, 3)

        self.decoder_rot = MLP(1024, 512, 256, 6)

        self.geo_proj_s2 = MLP(1024 + 256, 1024, 1024, 3)
        self.decoder_tran = MLP(1024, 512, 256, 6)

        self.decoder_rot2 = MLP(512, 512, 256, 3)
        self.decoder_tran2 = MLP(512, 512, 256, 3)

        self.normal_score_proj = MLP(self.num_queries, 128, 64, 3)
        self.rot_score_reg = nn.Linear(64, 1)

        self.param_score_proj = MLP(self.num_queries, 128, 64, 3)
        self.trans_score_reg = nn.Linear(64, 1)

    def forward(self, features1, features2, planeParam1, planeParam2, planeApp1=None, planeApp2=None,
                gt_pose=None, gt_corr_matrix=None, batched_inputs=None, ite=0, matching_net=None):
        if self.training:
            assert gt_pose is not None
        else:
            return self.inference_Joint(features1, features2, planeParam1, planeParam2, planeApp1, planeApp2,
                                        gt_pose=gt_pose,
                                        gt_corr_matrix=gt_corr_matrix,
                                        batched_inputs=batched_inputs,
                                        matching_net=matching_net)
        losses = {}
        log_scores_padded_list = []
        assignment_matrix_list = []
        trans_list = []
        rot_list = []
        out_cam_type = "soft"
        pose_ref_dict = {}

        # ****************************** pose refinement with initial pose **************************************
        losses_init, trans_list_init, rot_list_init, initPose_ref_outputs = self.forward_withInitialCam_Joint(
            features1, features2, gt_pose, batched_inputs, out_cam_type,
            planeParam1=planeParam1,
            planeParam2=planeParam2,
            assignment_matrix=gt_corr_matrix[:, :-1, :-1] if gt_corr_matrix is not None else None,
            iter=ite
        )
        losses.update(losses_init)
        pose_ref_dict["initial"] = {
            "trans": trans_list_init,
            "rot": rot_list_init
        }
        # ****************************** AIM **************************************
        if self.rand_cam_on:
            losses_rand = self.forward_withRandCam_Joint(
                gt_pose, suffix='N1'
            )
            losses.update(losses_rand)

        trans_list = trans_list + trans_list_init
        rot_list = rot_list + rot_list_init

        for key in list(losses.keys()):
            if not losses[key].requires_grad:
                losses.pop(key)
                continue
            if torch.isnan(losses[key]):
                print("*" * 20, ite)
                import pdb; pdb.set_trace()

        return losses, trans_list, rot_list, log_scores_padded_list, assignment_matrix_list, initPose_ref_outputs

    def forward_withInitialCam_Joint(self, features1, features2, gt_pose, batched_inputs, out_cam_type='soft',
                                     planeParam1=None, planeParam2=None, assignment_matrix=None, iter=0):
        losses = {}
        device = gt_pose.device
        pose_ref_outputs = None
        score_sequence, matched_nums = None, None

        initial_trans_feat_list = []
        initial_rot_feat_list = []
        initial_trans_list = []
        initial_rot_list = []

        rec_initial_trans_feat_list = []
        rec_initial_rot_feat_list = []
        rec_initial_trans_list = []
        rec_initial_rot_list = []

        trans_list = []
        rot_list = []

        # ****************************** get initial camera pose ******************************
        loss_pixelCam, pixel_initial_cam, pixel_pose_feats = self.__forward_PixelCameraHead(features1, features2, gt_pose=gt_pose)
        losses.update(loss_pixelCam)
        pixel_initial_trans = pixel_initial_cam['pred_trans']  # bs, 3
        pixel_initial_rot = pixel_initial_cam['pred_rot']  # bs, 4

        # record initial pose and feat
        initial_trans_list.append(pixel_initial_trans)
        initial_rot_list.append(pixel_initial_rot)
        initial_trans_feat_list.append(pixel_pose_feats["trans_feat"])
        initial_rot_feat_list.append(pixel_pose_feats["rots_feat"])

        # record pose
        trans_list.append(pixel_initial_trans)
        rot_list.append(pixel_initial_rot)

        if self.cam_rec_on:
            # ****************************** get rec initial rot **************************************
            losses_init_rot_rec, init_rec_rot, init_rec_rot_feat = self.__forward_RotRecHead(
                input_rot=pixel_initial_rot, suffix="_initCamRec")
            losses.update(losses_init_rot_rec)
            # ****************************** get rec initial trans **************************************
            losses_init_trans_rec, init_rec_trans, init_rec_trans_feat = self.__forward_TransRecHead(
                input_trans=pixel_initial_trans, suffix="_initCamRec")
            losses.update(losses_init_trans_rec)

            # record pose and feat
            rec_initial_trans_list.append(init_rec_trans)
            rec_initial_rot_list.append(init_rec_rot)
            rec_initial_trans_feat_list.append(init_rec_trans_feat)
            rec_initial_rot_feat_list.append(init_rec_rot_feat)

            # record pose
            trans_list.append(init_rec_trans)
            rot_list.append(init_rec_rot)

        if self.cam_ref_on:
            # ****************************** get local geo sequence
            geo_sequence_local_GTParam, score_sequence, matched_nums, _ = self.get_gt_geo_sequence(
                batched_inputs, pred_cams=None, device=device)
            geo_sequence_local = geo_sequence_local_GTParam
        else:
            return losses, trans_list, rot_list, pose_ref_outputs

        loop_n1 = 1 if self.cam_ref_on else 0
        loop_n2 = 1 if self.cam_ref_on else 0
        loop_n3 = 1 if self.cam_ref_on else 0

        # ****************************** pose refinement based on initial pose with gt plane param *******
        losses_n1, record_trans_list, record_rot_list = self.forawrd_refineLoop(
            loop_n1, initial_trans_list, initial_rot_list, initial_trans_feat_list, initial_rot_feat_list,
            geo_sequence_local, matched_nums, out_cam_type=out_cam_type, gt_pose=gt_pose, suffix='initCamRef',
            weight=self.plane_cam_weight
        )
        losses.update(losses_n1)
        trans_list = trans_list + record_trans_list
        rot_list = rot_list + record_rot_list

        # ****************************** pose refinement based on rec initial pose with gt plane param *******
        if self.cam_rec_on:
            losses_n2, record_trans_list, record_rot_list = self.forawrd_refineLoop(
                loop_n2, rec_initial_trans_list, rec_initial_rot_list, rec_initial_trans_feat_list, rec_initial_rot_feat_list,
                geo_sequence_local, matched_nums, out_cam_type=out_cam_type, gt_pose=gt_pose, suffix='initRecCamRef',
                weight=self.plane_cam_weight
            )
            losses.update(losses_n2)
            trans_list = trans_list + record_trans_list
            rot_list = rot_list + record_rot_list

        if assignment_matrix is None:
            return losses, trans_list, rot_list, pose_ref_outputs

        # ****************************** pose refinement based on initial pose with pred plane param *******
        geo_sequence_local_PredParam, score_sequence, matched_nums = self.get_pred_geo_sequence(
            planes1=planeParam1, planes2=planeParam2,
            pred_assignment_matrix=assignment_matrix,
            pred_score_matrix=assignment_matrix,
            pred_cams=None, device=device)
        geo_sequence_local = geo_sequence_local_PredParam

        aux_initial_trans_list = [initial_trans_list[0]]
        aux_initial_rot_list = [initial_rot_list[0]]
        aux_initial_trans_feat_list = [initial_trans_feat_list[0]]
        aux_initial_rot_feat_list = [initial_rot_feat_list[0]]
        losses_n3_initial, record_trans_list, record_rot_list = self.forawrd_refineLoop(
            loop_n3, aux_initial_trans_list, aux_initial_rot_list,
            aux_initial_trans_feat_list, aux_initial_rot_feat_list,
            geo_sequence_local, matched_nums,
            out_cam_type=out_cam_type, gt_pose=gt_pose, suffix='initCamRef_Aux',
            weight=self.plane_cam_weight_predplane
        )
        losses.update(losses_n3_initial)
        trans_list = trans_list + record_trans_list
        rot_list = rot_list + record_rot_list

        # ****************************** pose refinement based on rec initial pose with pred plane param *******
        if self.cam_rec_on:
            aux_rec_trans_list = [rec_initial_trans_list[0]]
            aux_rec_rot_list = [rec_initial_rot_list[0]]
            aux_rec_trans_feat_list = [rec_initial_trans_feat_list[0]]
            aux_rec_rot_feat_list = [rec_initial_rot_feat_list[0]]
            losses_n3_rec, record_trans_list, record_rot_list = self.forawrd_refineLoop(
                loop_n3, aux_rec_trans_list, aux_rec_rot_list,
                aux_rec_trans_feat_list, aux_rec_rot_feat_list,
                geo_sequence_local, matched_nums,
                out_cam_type=out_cam_type, gt_pose=gt_pose, suffix='initRecCamRef_Aux',
                weight=self.plane_cam_weight_predplane
            )
            losses.update(losses_n3_rec)
            trans_list = trans_list + record_trans_list
            rot_list = rot_list + record_rot_list

        return losses, trans_list, rot_list, pose_ref_outputs

    def forward_withRandCam_Joint(self, gt_pose, suffix='N1', LBS_on=True):
        losses = {}
        device = gt_pose.device
        bs = gt_pose.shape[0]
        if bs > self.rand_bs:
            repeat_rat = 1
        else:
            repeat_rat = self.rand_bs // bs
        if not self.cam_rec_on:
            return losses
        if LBS_on:
            # # ****************************** train rec rot with rand input **************************************
            losses_rand_rot_rec, _, _ = self.__forward_RotRecHead(
                input_rot=None, suffix="_randCamRecLBS_%s"%(suffix), batch_size=bs*repeat_rat, device=device)
            losses.update(losses_rand_rot_rec)
            # ****************************** train rec trans with rand input **************************************
            losses_rand_trans_rec, _, _ = self.__forward_TransRecHead(
                input_trans=None, suffix="_randCamRecLBS_%s"%(suffix), batch_size=bs*repeat_rat, device=device)
            losses.update(losses_rand_trans_rec)
        return losses

    def forawrd_refineLoop(self, loop, trans_list, rot_list, trans_feat_list, rot_feat_list, geo_sequence_local, matched_nums,
                           out_cam_type='soft', gt_pose=None, suffix='', weight=1.0):
        losses = {}
        record_trans_list = []
        record_rot_list = []
        assert loop <= 1, "does not support the case of loop > 1 now"
        for ite in range(loop):
            input_camera1 = {
                "tran": trans_list[ite].detach(),  # bs, 3
                "rot": rot_list[ite].detach()  # bs, 4
            }
            geo_sequence_global = self.get_gt_global_geo_sequence(
                geo_sequence_local, pred_cams=input_camera1)

            input_camera2 = {
                "tran": torch.zeros_like(trans_list[ite]).detach(),  # bs, 3
                "rot": rot_list[ite].detach()  # bs, 4
            }
            geo_sequence_global_aux = self.get_gt_global_geo_sequence(
                geo_sequence_local, pred_cams=input_camera2)

            sig_seq = (geo_sequence_global[:, :, 0:1] * geo_sequence_global_aux[:, :, 0:1]) >= 0
            sig_seq = (sig_seq.float() - 0.5) * 2.  # -1 or 1  bs, n, 1

            losses_pose_ref, pose_ref_outputs = self.__forward_PlaneCamRefHead(
                trans_feat_list[ite],
                rot_feat_list[ite],
                geo_sequence_global,
                gt_pose=gt_pose,
                geo_sequence_local=geo_sequence_local,
                matched_nums=matched_nums,
                suffix=suffix,
                out_cam_type=out_cam_type,
                weight=weight,
                sig_seq=sig_seq,
                initial_trans=trans_list[ite],
                initial_rot=rot_list[ite]
            )
            losses.update(losses_pose_ref)

            if "pred_rot_avg" in pose_ref_outputs and "pred_trans_avg" in pose_ref_outputs:
                # record pose
                record_rot_list.append(pose_ref_outputs["pred_rot_avg"])
                record_trans_list.append(pose_ref_outputs["pred_trans_avg"])

            ref_rot_soft = pose_ref_outputs["pred_rot"]
            ref_trans_soft = pose_ref_outputs["pred_trans"]

            # record pose
            record_rot_list.append(ref_rot_soft)
            record_trans_list.append(ref_trans_soft)

        return losses, record_trans_list, record_rot_list

    def inference_Joint(self, cam_feats1, cam_feats2, planeParam1, planeParam2, planeApp1, planeApp2,
                        gt_corr_matrix=None, batched_inputs=None, gt_pose=None, matching_net=None):
        device = cam_feats1['res2'].device
        trans_list = []
        rot_list = []
        matcher_inputCam_list = []
        initial_rot_feat_list = []
        initial_rot_list = []
        initial_trans_feat_list = []
        initial_trans_list = []

        output_cameras = {}
        output_cameras["camera_zero"] = {"tran": torch.tensor([0., 0., 0.]).reshape(1, 3).to(device).float(),
                                         "rot": torch.tensor([1., 0., 0., 0.]).reshape(1, 4).to(device).float()}

        out_cam_type = self.inference_out_cam_type
        if not self.cam_ref_on:
            out_cam_type = "initial"

        if self.use_sparsePlane_Top1Cam_testSet:
            print("Note: using top1 cam")
            img_id1 = batched_inputs[0]['0']['image_id']
            img_id2 = batched_inputs[0]['1']['image_id']
            cam_id = img_id1 + '-' + img_id2
            top1_cam = self.sparsePlaneTop1Cam_dict[cam_id]
            top1_cam_rot = torch.tensor(top1_cam['rotation']).reshape(1, 4).to(device=device, dtype=planeApp1.dtype)
            top1_cam_rot = F.normalize(top1_cam_rot, dim=-1, p=2)
            if top1_cam_rot[0, 0] < 0:
                top1_cam_rot = -top1_cam_rot
            top1_cam_trans = torch.tensor(top1_cam['position']).reshape(1, 3).to(device=device, dtype=planeApp1.dtype)
            initial_rot = top1_cam_rot
            initial_trans = top1_cam_trans
        else:
            _, pixel_initial_cam, pixel_pose_feats = self.__forward_PixelCameraHead(cam_feats1, cam_feats2, gt_pose=gt_pose)
            initial_trans = pixel_initial_cam['pred_trans']  # bs, 3
            initial_rot = pixel_initial_cam['pred_rot']  # bs, 4
            if initial_rot[0, 0] < 0:
                initial_rot = -initial_rot

        # record pose
        trans_list.append(initial_trans)
        rot_list.append(initial_rot)
        output_cameras["camera_init"] = {"tran": initial_trans, "rot": initial_rot}

        if not self.plane_matcher_on:
            print("Note: using initial camera")
            output_cameras["camera"] = {"tran": trans_list[-1], "rot": rot_list[-1]}
            return output_cameras, trans_list, rot_list, [], {}, None

        if self.cam_rec_on:
            # ****************************** get rec rot
            _, rec_initial_rot, rec_initial_rot_feat = self.__forward_RotRecHead(initial_rot)
            # ****************************** get rec pose
            _, rec_initial_trans, rec_initial_trans_feat = self.__forward_TransRecHead(initial_trans)
            # get matcher input pose
            rec_pose = torch.cat([rec_initial_trans, rec_initial_rot], dim=-1)  # bs, 7
            matcher_inputCam_list.append(rec_pose)
            # record pose
            trans_list.append(rec_initial_trans)
            rot_list.append(rec_initial_rot)
            output_cameras["camera_initRec"] = {"tran": rec_initial_trans, "rot": rec_initial_rot}
            # record pose and pose feat
            initial_rot_list.append(rec_initial_rot)
            initial_trans_list.append(rec_initial_trans)
            initial_rot_feat_list.append(rec_initial_rot_feat)
            initial_trans_feat_list.append(rec_initial_trans_feat)
        else:
            rec_pose = torch.cat([initial_trans, initial_rot], dim=-1)  # bs, 7
            matcher_inputCam_list.append(rec_pose)
            assert not self.use_sparsePlane_Top1Cam_testSet
            initial_rot_list.append(initial_rot)
            initial_rot_feat_list.append(pixel_pose_feats["rots_feat"])
            initial_trans_list.append(initial_trans)
            initial_trans_feat_list.append(pixel_pose_feats["trans_feat"])

        use_gt_matcher = self.cfg.TEST.POSE_REFINEMENT_WITH_GT_MATCHERS
        pose_ref_outputs = None
        log_scores_padded_list = []
        assignment_matrix_list = []
        output_planeAss = {}

        # calculate warped plane parameters
        parameters2_warped = self.warp_single_view_plane_param_to_global(
            planeParam2, pose_n=1)[:, 0, :, :]  # bs, n2, 3
        offset2_warped = torch.norm(parameters2_warped, dim=2, keepdim=True, p=2)  # bs, n2, 1
        normal2_warped = F.normalize(parameters2_warped, dim=-1, p=2)  # bs, n2, 3

        # ****************************** matching planes **************************************
        matcher_inputCam = matcher_inputCam_list[-1]
        match_threshold = self.matching_score_threshold
        normal_deacy = 1.0
        offset_decay = 1.0
        # matching forward
        _, log_scores_padded = matching_net(
            planeApp1, planeApp2, matcher_inputCam, planeParam1, planeParam2,
            gt_corr_matrix=None,
            normal_decay=normal_deacy, offset_deacy=offset_decay
        )
        log_scores_padded_list.append(log_scores_padded)
        # get assignment matrix
        scores_matrix = (log_scores_padded[:, :-1, :-1]).contiguous().exp()
        assignment_matrix = get_assignment_matrix(log_scores_padded, match_threshold=match_threshold)
        assignment_matrix_list.append(assignment_matrix)
        output_planeAss["pred_assignment_beforeRef0"] = assignment_matrix.clone()
        if out_cam_type == "initial":
            print("Note: using initial camera")
            output_planeAss["pred_assignment"] = assignment_matrix.clone()
            output_cameras["camera"] = {"tran": trans_list[0], "rot": rot_list[0]}
            return output_cameras, trans_list, rot_list, log_scores_padded_list, output_planeAss, pose_ref_outputs
        else:
            print("Note: using ref camera (%s) from initial cam" % (out_cam_type))

        # get geo local sequence
        geo_sequence_local, _, _ = self.get_pred_geo_sequence(
            planes1=planeParam1, planes2=planeParam2,
            pred_assignment_matrix=assignment_matrix,
            pred_score_matrix=scores_matrix,
            pred_cams=None, device=device)

        # ****************************** rot refinement **************************************
        if use_gt_matcher:
            print("Warning: you are using gt plane mathes for pose refinement")
            noise_on = self.cfg.TEST.POSE_REFINEMENT_WITH_GT_NOISE_MATCHERS
            if noise_on:
                print("------>Using gt plane mathes for pose refinement with noises!")
            noise_scale_offset = self.cfg.TEST.POSE_REFINEMENT_WITH_GT_NOISE_MATCHERS_OFFSET_SCALE
            noise_normal_scale = self.cfg.TEST.POSE_REFINEMENT_WITH_GT_NOISE_MATCHERS_NORMAL_SCALE / 180. * np.pi
            geo_sequence_local, _, _, gauss_noise = self.get_gt_geo_sequence(
                batched_inputs, pred_cams=None, device=device, gauss_noise_on=noise_on, pre_gauss_noise=None,
                scale_offset=noise_scale_offset, scale_normal=noise_normal_scale)
            input_camera = {
                "tran": initial_trans_list[-1],  # bs, 3
                "rot": initial_rot_list[-1]  # bs, 4
            }
            geo_sequence_global, score_sequence, matched_nums, _ = self.get_gt_geo_sequence(
                batched_inputs, pred_cams=input_camera, device=device, gauss_noise_on=noise_on,
                pre_gauss_noise=gauss_noise,
                scale_offset=noise_scale_offset, scale_normal=noise_normal_scale)
            input_camera2 = {
                "tran": torch.zeros_like(initial_trans_list[-1]),  # bs, 3
                "rot": initial_rot_list[-1].detach()  # bs, 4
            }
            geo_sequence_global_aux, _, _, _ = self.get_gt_geo_sequence(
                batched_inputs, pred_cams=input_camera2, device=device, gauss_noise_on=noise_on,
                pre_gauss_noise=gauss_noise,
                scale_offset=noise_scale_offset, scale_normal=noise_normal_scale)
            sig_seq = (geo_sequence_global[:, :, 0:1] * geo_sequence_global_aux[:, :, 0:1]) >= 0
            sig_seq = (sig_seq.float() - 0.5) * 2.  # -1 or 1  bs, n, 1
        else:
            # get geo global sequence
            input_camera = {
                "tran": initial_trans_list[-1],  # bs, 3
                "rot": initial_rot_list[-1]  # bs, 4
            }
            # in inference stage, we use pred assignment_matrix and pred scores_matrix
            geo_sequence_global, score_sequence, matched_nums = self.get_pred_geo_sequence(
                planes1=planeParam1, planes2=planeParam2,
                pred_assignment_matrix=assignment_matrix,
                # pred_score_matrix=scores_matrix,
                pred_cams=input_camera, device=device)
            input_camera2 = {
                "tran": torch.zeros_like(initial_trans_list[-1]),  # bs, 3
                "rot": initial_rot_list[-1].detach()  # bs, 4
            }
            geo_sequence_global_aux, _, _ = self.get_pred_geo_sequence(
                planes1=planeParam1, planes2=planeParam2,
                pred_assignment_matrix=assignment_matrix,
                pred_cams=input_camera2, device=device)
            sig_seq = (geo_sequence_global[:, :, 0:1] * geo_sequence_global_aux[:, :, 0:1]) >= 0
            sig_seq = (sig_seq.float() - 0.5) * 2.  # -1 or 1  bs, n, 1

        _, pose_ref_outputs = self.__inference_PlaneCamRefHead(
            initial_trans_feat_list[-1],
            initial_rot_feat_list[-1],
            geo_sequence_global,
            score_sequence,
            gt_pose=None,
            geo_sequence_local=geo_sequence_local,
            matched_nums=matched_nums,
            out_cam_type=out_cam_type,
            sig_seq=sig_seq,
            initial_rot=initial_rot_list[-1],
            initial_trans=initial_trans_list[-1]
        )

        if "pred_rot_avg" in pose_ref_outputs and "pred_trans_avg" in pose_ref_outputs:
            # record pose
            trans_list.append(pose_ref_outputs["pred_trans_avg"])
            rot_list.append(pose_ref_outputs["pred_rot_avg"])
            output_cameras["camera_avgRef0"] = {"tran": trans_list[-1], "rot": rot_list[-1]}

        # get refined pose
        ref_trans_soft = pose_ref_outputs["pred_trans"]
        ref_rot_soft = pose_ref_outputs["pred_rot"]

        # record pose
        trans_list.append(ref_trans_soft)
        rot_list.append(ref_rot_soft)
        output_cameras["camera_softRef0"] = {"tran": trans_list[-1], "rot": rot_list[-1]}

        if ref_rot_soft[0, 0] < 0 and self.cam_rec_on:
            ref_rot_soft = - ref_rot_soft

        pose_ref_outputs['sig_seq'] = sig_seq[0, :matched_nums[0], 0]

        # ---------------------------------------update Ass. matrix
        # calculate normal dist
        parameters1_warped_r = self.warp_single_view_plane_param_to_global(
            planeParam1, ref_rot_soft.unsqueeze(1), ref_trans_soft.unsqueeze(1) * 0.)[:, 0, :, :]  # bs, n1, 3
        normal1_warped_r = F.normalize(parameters1_warped_r, dim=-1, p=2)  # bs, n1, 3
        nTn_r = torch.bmm(normal1_warped_r, normal2_warped.transpose(1, 2))  # bs, n1, n2
        normal_dist = torch.acos(torch.clamp(nTn_r, -1, 1))  # in rad
        normal_dist = normal_dist / np.pi * 180.  # b, n1, n2
        # calculate offset dist
        parameters1_warped_rt = self.warp_single_view_plane_param_to_global(
            planeParam1, ref_rot_soft.unsqueeze(1), ref_trans_soft.unsqueeze(1))[:, 0, :, :]  # bs, n1, 3
        offset1_warped_rt = torch.norm(parameters1_warped_rt, dim=2, keepdim=True, p=2)  # bs, n1, 1
        normal1_warped_rt = F.normalize(parameters1_warped_rt, dim=-1, p=2)  # bs, n1, 3
        nTn_rt = torch.bmm(normal1_warped_rt, normal2_warped.transpose(1, 2))  # bs, n1, n2
        offset_dist = torch.abs(offset1_warped_rt - offset2_warped.transpose(1, 2))  # b, n1, n2
        offset_dist[nTn_rt < 0] = torch.abs(offset1_warped_rt + offset2_warped.transpose(1, 2))[nTn_rt < 0]
        offset_dist = torch.clamp(offset_dist, min=1e-4, max=10)  # b, n1, n2
        # calculate mask
        normal_dist_mask = normal_dist < 45.
        offset_dist_mask = offset_dist < 1.
        Ass_mask = normal_dist_mask & offset_dist_mask
        assignment_matrix = assignment_matrix * Ass_mask.float()
        assignment_matrix_list.append(assignment_matrix)
        output_planeAss["pred_assignment_afterRef0"] = assignment_matrix.clone()
        output_planeAss["pred_assignment"] = assignment_matrix.clone()

        matcher_inputCam_list.append(torch.cat([ref_trans_soft, ref_rot_soft], dim=-1))

        output_cameras["camera"] = {"tran": trans_list[-1], "rot": rot_list[-1]}

        if 'all_pred_trans' in pose_ref_outputs:
            output_cameras["camera_onePP"] = {
                "tran": pose_ref_outputs['all_pred_trans'],  # b, m, 3
                "rot": pose_ref_outputs['all_pred_rots']
            }
        return output_cameras, trans_list, rot_list, log_scores_padded_list, output_planeAss, pose_ref_outputs

    def __forward_PixelCameraHead(self, features1, features2, gt_pose=None):
        # feature net
        x1_0 = self.pixel_decoder.forward_features(features1)
        x2_0 = self.pixel_decoder.forward_features(features2)
        x1 = self.convs_backbone(x1_0)  # b, 256, h, w
        x2 = self.convs_backbone(x2_0)  # b, 256, h, w
        bs, _, h1, w1 = x1.shape
        bs, _, h2, w2 = x2.shape

        # get aff matrix
        aff = self.compute_corr_softmax(x1, x2)  # bs, w*h, h, w (e.g., bs, 20*15, 15, 20)

        # get trans feature
        trans_feat_0 = self.convs_trans(aff)  # (bs, 128, 2, 3)
        trans_feat_0 = torch.flatten(trans_feat_0, 1)  # (bs, 768)
        trans_feat = F.relu(self.fc_trans(trans_feat_0))  # (bs, 256)

        # get rots feature
        rots_feat_0 = self.convs_rots(aff)  # (bs, 128, 2, 3)
        rots_feat_0 = torch.flatten(rots_feat_0, 1)  # (bs, 768)
        rots_feat = F.relu(self.fc_rots(rots_feat_0))  # (bs, 256)

        # get refined trans
        trans = self.trans(trans_feat)  # directly regress pose
        # # get refined rots
        rots_norm = F.normalize(self.rots(rots_feat), p=2, dim=1)  # directly regress pose

        pred_cam = {"pred_trans": trans, "pred_rot": rots_norm}
        pose_feats = {"trans_feat": trans_feat, "rots_feat": rots_feat}

        losses = {}
        if self.training:
            # cam loss
            pose_pred_ref = torch.cat((trans, rots_norm), dim=-1)  # bs, 7  trans + rot
            loss_tran, loss_rot = self.cam_loss(pose_pred_ref, gt_pose)
            losses = {
                "loss_tran_pixelReg": loss_tran * self.initial_cam_weight,
                "loss_rot_pixelReg": loss_rot * self.initial_cam_weight,
            }
            if torch.isnan(loss_tran) or torch.isnan(loss_rot):
                import pdb; pdb.set_trace()
        return losses, pred_cam, pose_feats

    def __forward_RotRecHead(self, input_rot=None, suffix="_rec", batch_size=32, device=None, mask=None, loss_on=True):
        if input_rot is None:
            assert device is not None
            rot_vec_rand = (torch.rand(batch_size, 3) * 2. - 1) * 2.5  # -2.5 ~ 2.5
            input_rot = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vec_rand))  # bs, 4
            input_rot = torch.from_numpy(input_rot).to(device=device, dtype=torch.float32)
            input_rot = F.normalize(input_rot, dim=1, p=2)
        else:
            input_rot = input_rot.detach().clone()  # bs, 4

        sig = ((input_rot[:, 0:1] >= 0.).float() - 0.5) * 2.
        input_rot = input_rot * sig

        if loss_on:
            input_rot_emb = input_rot.detach()  # bs, 4
            rot_feat = F.relu(self.rot_emb_proj(input_rot_emb))
            pred_rot = F.normalize(self.rots(rot_feat), p=2, dim=1)  # bs, 4
            losses = {}
            if self.training:
                if mask is None:
                    loss_rot = torch.norm(F.normalize(input_rot, p=2, dim=1) - pred_rot, dim=1, p=2).mean()
                else:
                    loss_rot = (torch.norm(F.normalize(input_rot, p=2, dim=1) - pred_rot, dim=1, p=2)[mask > 0]).mean()
                losses["loss_rot%s" % (suffix)] = loss_rot
            return losses, pred_rot, rot_feat
        else:
            return {}, input_rot, None

    def __forward_TransRecHead(self, input_trans=None, suffix="_rec", batch_size=32, device=None, mask=None, loss_on=True):
        if input_trans is None:
            assert device is not None
            input_trans = (torch.rand(batch_size, 3).to(device) - 0.5) * 5.
        else:
            input_trans = input_trans.detach().clone() + 1e-10  # bs, 3

        if loss_on:
            input_trans_emb = input_trans.detach()
            trans_feat = F.relu(self.trans_emb_proj(input_trans_emb))  # bs, 256
            pred_trans = self.trans(trans_feat)  # bs, 3
            losses = {}
            if self.training:
                if mask is None:
                    loss_trans = torch.norm(input_trans - pred_trans, dim=1, p=2).mean()
                else:
                    loss_trans = (torch.norm(input_trans - pred_trans, dim=1, p=2)[mask > 0]).mean()

                losses["loss_trans%s" % (suffix)] = loss_trans

            return losses, pred_trans, trans_feat
        else:
            return {}, input_trans, None

    def __forward_PlaneCamRefHead(self, initial_trans_feat, initial_rot_feat, geo_sequence_global_init,
                                  gt_pose=None, geo_sequence_local=None, suffix="",
                                  matched_nums=None, out_cam_type='soft', weight=1.0, sig_seq=None,
                                  initial_trans=None, initial_rot=None, iter=0):
        assert sig_seq is not None
        bs, max_n, _ = geo_sequence_global_init.shape
        # -------------------------------------------------------geo encoding------------------------------------
        if self.warp_plane_in_cam_ref_on:
            # get normal and offset
            geo_sequence_0 = geo_sequence_global_init[:, :, :3]  # bs, n, 3
            offset_0 = torch.norm(geo_sequence_0, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_0 = geo_sequence_0 / (offset_0 + 1e-10)  # bs, n, 3
            # get normal and offset
            geo_sequence_1 = geo_sequence_global_init[:, :, 3:]  # bs, n, 3
            offset_1 = torch.norm(geo_sequence_1, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_1 = geo_sequence_1 / (offset_1 + 1e-10)  # bs, n, 3
            offset_0 = offset_0 * sig_seq
            normal_0 = normal_0 * sig_seq
        else:
            # get normal and offset
            geo_sequence_0 = geo_sequence_local[:, :, :3]  # bs, n, 3
            offset_0 = torch.norm(geo_sequence_0, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_0 = geo_sequence_0 / (offset_0 + 1e-10)  # bs, n, 3
            # get normal and offset
            geo_sequence_1 = geo_sequence_local[:, :, 3:]  # bs, n, 3
            offset_1 = torch.norm(geo_sequence_1, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_1 = geo_sequence_1 / (offset_1 + 1e-10)  # bs, n, 3

        # calculate geo feat
        geo_sequence_new = torch.cat((normal_0, offset_0, normal_1, offset_1), dim=-1)  # bs, n, 8
        geo_fea_all = self.geo_encoder(geo_sequence_new)  # bs, n, 1024
        geo_fea_s1_all = self.geo_proj_s1(geo_fea_all)  # bs, n, 1024

        geo_fea_rot_all = self.decoder_rot(geo_fea_s1_all)  # bs, n, 256
        geo_fea_s2_all = self.geo_proj_s2(torch.cat([geo_fea_s1_all, geo_fea_rot_all], dim=-1))  # bs, n, 1024
        geo_fea_tran_all = self.decoder_tran(geo_fea_s2_all)  # bs, n, 256

        # ----------------------------------------calculate one plane pose embedding------------------------------------
        # generate mask
        matching_mask = torch.zeros(bs, max_n + 1, max_n).to(device=initial_rot_feat.device,
                                                             dtype=torch.float32)  # bs, n+1, n
        for i in range(bs):
            m = matched_nums[i]
            matching_mask[i, :m + 1, :m] = 1.
        matching_mask = matching_mask.contiguous().detach()
        # padding initial pose
        initial_trans_feat_pad = initial_trans_feat.unsqueeze(1).repeat(1, max_n, 1).view(bs * max_n, -1)  # b*n, 256
        initial_rot_feat_pad = initial_rot_feat.unsqueeze(1).repeat(1, max_n, 1).view(bs * max_n, -1)  # b*n, 256
        # calculate one plane pose embedding
        fused_rots_feat_all = torch.cat((initial_rot_feat_pad, geo_fea_rot_all.view(bs * max_n, -1)),dim=-1)  # bs*n, 512
        fused_rots_feat_all = F.relu(self.decoder_rot2(fused_rots_feat_all))  # bs*n, 256
        fused_trans_feat_all = torch.cat((initial_trans_feat_pad, geo_fea_tran_all.view(bs * max_n, -1)),dim=-1)  # bs*n, 512
        fused_trans_feat_all = F.relu(self.decoder_tran2(fused_trans_feat_all))  # bs*n, 256

        # ------------------------------------------calculate one plane rot and scoring-------------------------------
        # rot estimate
        rots_ref_all = F.normalize(self.rots(fused_rots_feat_all), dim=-1, p=2).view(bs, max_n, 4)  # bs, n, 4
        rots_ref_all = torch.cat([initial_rot.unsqueeze(1), rots_ref_all], dim=1)  # bs, n+1, 4
        # calculate normal dist
        trans_zero_all = torch.zeros(bs, max_n + 1, 3).to(device=rots_ref_all.device, dtype=torch.float32)
        plane1_mid_r = self.warp_single_view_plane_param_to_global(geo_sequence_local[:, :, 3:],
                                                                   pose_n=1)  # bs, 1, n, 3
        plane1_mid_r = plane1_mid_r.repeat(1, max_n + 1, 1, 1)  # bs, n+1, n, 3
        plane0_mid_r = self.warp_single_view_plane_param_to_global(geo_sequence_local[:, :, :3], rots_ref_all,
                                                                   trans_zero_all)  # bs, n+1, n, 3
        normal0_mid_r = F.normalize(plane0_mid_r, p=2, dim=-1)  # bs, n+1, n, 3
        normal1_mid_r = F.normalize(plane1_mid_r, p=2, dim=-1)  # bs, n+1, n, 3
        dist_normalAngle_mid = torch.acos(
            torch.clamp(torch.sum(normal0_mid_r * normal1_mid_r, dim=-1), min=-1., max=1.)) / np.pi * 180.
        dist_normalL2_mid_r = torch.norm(normal0_mid_r - normal1_mid_r, p=2, dim=-1)  # bs, n+1, n ; 0 ~ +
        dist_normalL2_mid_r = dist_normalL2_mid_r * matching_mask  # bs, n+1, n ; 0 ~ +
        score_normalL2_mid_r = torch.exp(-dist_normalL2_mid_r) * matching_mask  # bs, n+1, n ; 0 ~ +
        score_normalL2_mid_r = self.normal_score_proj(score_normalL2_mid_r)  # bs, n+1, 64

        score_soft_rot_temp = self.rot_score_reg(score_normalL2_mid_r)  # bs, n+1, 1
        score_soft_rot = torch.zeros_like(score_soft_rot_temp)
        for i in range(bs):
            m = matched_nums[i]
            score_soft_rot[i, :m + 1] = score_soft_rot_temp[i, :m + 1].softmax(0)  # bs, n+1, 1
        score_soft_rot = torch.clamp(score_soft_rot, max=0.9, min=0.01)
        score_soft_rot = score_soft_rot * matching_mask[:, :, 0:1]  # bs, n+1, 1
        score_soft_rot = score_soft_rot / (score_soft_rot.sum(dim=1, keepdim=True) + 1e-10)

        # -----------------------------------------calculate one plane trans and scoring -------------------------------
        # trans estimate
        trans_ref_all = self.trans(fused_trans_feat_all).view(bs, max_n, 3)  # bs, n, 3
        trans_ref_all = torch.cat([initial_trans.unsqueeze(1), trans_ref_all], dim=1)  # bs, n+1, 3
        # calculate trans cost
        plane0_mid_rt = self.warp_single_view_plane_param_to_global(geo_sequence_local[:, :, :3], rots_ref_all,
                                                                    trans_ref_all)  # bs, n+1, n, 3
        plane1_mid_rt = plane1_mid_r  # bs, n+1, n, 3
        offset0_mid_rt = torch.norm(plane0_mid_rt, p=2, dim=-1)  # bs, n+1, n
        offset1_mid_rt = torch.norm(plane1_mid_rt, p=2, dim=-1)  # bs, n+1, n
        normal0_mid_rt = F.normalize(plane0_mid_rt, p=2, dim=-1)  # bs, n+1, n, 3
        normal1_mid_rt = F.normalize(plane1_mid_rt, p=2, dim=-1)  # bs, n+1, n, 3
        nTn = torch.sum(normal0_mid_rt * normal1_mid_rt, dim=-1)  # bs, n+1, n
        dist_offset_mid_rt = torch.abs(offset0_mid_rt - offset1_mid_rt)  # bs, n+1, n ; 0 ~ +
        dist_offset_mid_rt[nTn < 0] = torch.abs(offset0_mid_rt + offset1_mid_rt)[nTn < 0]
        dist_l2_mid_ori = torch.norm(plane0_mid_rt - plane1_mid_rt, p=2, dim=-1)  # bs, n+1, n ; 0 ~ +
        # use l2 dist to re-score translation
        dist_l2_mid = dist_l2_mid_ori * matching_mask  # bs, n+1, n ; 0 ~ +
        dist_l2_mid = torch.exp(-dist_l2_mid) * matching_mask  # bs, n+1, n ; 0 ~ +
        dist_l2_mid = self.param_score_proj(dist_l2_mid)  # bs, n+1, 64

        score_soft_trans_temp = self.trans_score_reg(dist_l2_mid)  # bs, n+1, 1
        score_soft_trans = torch.zeros_like(score_soft_trans_temp)
        for i in range(bs):
            m = matched_nums[i]
            score_soft_trans[i, :m + 1] = score_soft_trans_temp[i, :m + 1].softmax(0)  # bs, n+1, 1
        score_soft_trans = torch.clamp(score_soft_trans, max=0.9, min=0.01)
        score_soft_trans = score_soft_trans * matching_mask[:, :, 0:1]  # bs, n+1, 1
        score_soft_trans = score_soft_trans / (score_soft_trans.sum(dim=1, keepdim=True) + 1e-10)

        # -----------------------------------------pose selection-------------------------------
        # ---------------------------  Avg.
        score_avgAll_trans = torch.ones_like(score_soft_trans) * matching_mask[:, :, 0:1]  # bs, n+1, 1
        score_avgAll_trans = score_avgAll_trans / (score_avgAll_trans.sum(dim=1, keepdim=True) + 1e-10)  # bs, n+1, 1
        score_avgAll_rot = torch.ones_like(score_soft_rot) * matching_mask[:, :, 0:1]  # bs, n+1, 1
        score_avgAll_rot = score_avgAll_rot / (score_avgAll_rot.sum(dim=1, keepdim=True) + 1e-10)  # bs, n+1, 1
        fused_trans_feat_avg = (
            fused_trans_feat_all.reshape(bs, max_n, 256) * score_avgAll_trans[:, 1:] / score_avgAll_trans[:, 1:].sum(dim=1, keepdim=True)).sum(dim=1)  # b, 256
        fused_rots_feat_avg = (
            fused_rots_feat_all.reshape(bs, max_n, 256) * score_avgAll_rot[:, 1:] / score_avgAll_rot[:, 1:].sum(dim=1, keepdim=True)).sum(dim=1)  # b, 256
        rots_ref_avg = F.normalize(self.rots(fused_rots_feat_avg), dim=-1, p=2)
        trans_ref_avg = self.trans(fused_trans_feat_avg)

        # --------------------------- Soft
        fused_trans_feat_soft = torch.cat((initial_trans_feat.unsqueeze(1), fused_trans_feat_all.reshape(bs, max_n, 256)), dim=1)  # b, n+1, c
        fused_trans_feat_soft = torch.sum(fused_trans_feat_soft * score_soft_trans, dim=1)
        fused_rots_feat_soft = torch.cat((initial_rot_feat.unsqueeze(1), fused_rots_feat_all.reshape(bs, max_n, 256)), dim=1)
        fused_rots_feat_soft = torch.sum(fused_rots_feat_soft * score_soft_rot, dim=1)
        rots_ref_soft = F.normalize(self.rots(fused_rots_feat_soft), dim=-1, p=2)
        trans_ref_soft = self.trans(fused_trans_feat_soft)

        pred_cam = {"pred_trans": trans_ref_soft, "pred_rot": rots_ref_soft,
                    "pred_trans_avg": trans_ref_avg, "pred_rot_avg": rots_ref_avg,
                    "all_pred_trans": trans_ref_all[0:1, :matched_nums[0] + 1],
                    "all_pred_rots": rots_ref_all[0:1, :matched_nums[0] + 1],
                    "score_soft_rot": score_soft_rot[0:1, :matched_nums[0] + 1],
                    "score_soft_offset": score_soft_trans[0:1, :matched_nums[0] + 1],
                    "l2_dist": dist_l2_mid_ori[0:1, :matched_nums[0] + 1, :matched_nums[0]],
                    "normal_dist": dist_normalAngle_mid[0:1, :matched_nums[0] + 1, :matched_nums[0]],
                    "offset_dist": dist_offset_mid_rt[0:1, :matched_nums[0] + 1, :matched_nums[0]],
                    }

        # -----------------------------------------loss calculation -------------------------------
        losses = {}
        if self.training:
            pose_pred_ref = torch.cat((trans_ref_avg, rots_ref_avg), dim=-1)  # bs, 7  trans + rot
            pose_pred_ref_soft = torch.cat((trans_ref_soft, rots_ref_soft), dim=-1)  # bs, 7  trans + rot

            loss_tran_ref, loss_rot_ref = self.cam_loss(pose_pred_ref, gt_pose)
            loss_tran_ref_soft, loss_rot_ref_soft = self.cam_loss(pose_pred_ref_soft, gt_pose)

            batch_idx = torch.arange(bs).to(device=gt_pose.device, dtype=torch.int64)
            # get best rot idx
            onePP_rot_err = torch.norm(
                F.normalize(gt_pose[:, 3:].unsqueeze(1), p=2, dim=-1) - F.normalize(rots_ref_all, p=2, dim=-1),
                dim=-1, p=2)  # bs, n+1
            onePP_rot_err.masked_fill_(matching_mask[:, :, 0] < 0.5, 1e10)
            min_onePP_rot_err, min_onePP_rot_err_idx = onePP_rot_err.detach().min(dim=-1)  # bs
            loss_rotIdx = torch.abs(1.0 - score_soft_rot.squeeze(-1)[batch_idx, min_onePP_rot_err_idx]).mean()

            # get best trans idx
            onePP_trans_err = torch.norm(
                gt_pose[:, :3].unsqueeze(1) - trans_ref_all,
                dim=-1, p=2)  # bs, n+1
            onePP_trans_err.masked_fill_(matching_mask[:, :, 0] < 0.5, 1e10)
            min_onePP_trans_err, min_onePP_trans_err_idx = onePP_trans_err.detach().min(dim=-1)  # bs
            loss_transIdx = torch.abs(1.0 - score_soft_trans.squeeze(-1)[batch_idx, min_onePP_trans_err_idx]).mean()

            offset_errs = 0.
            for bi in range(bs):
                offset_errs += torch.diag(dist_l2_mid_ori[bi, 1:]).sum() / matched_nums[bi]
            paramL2_errs = offset_errs / bs

            losses = {
                "loss_tran_planeAvgReg_%s" % (suffix): loss_tran_ref * weight,
                "loss_rot_planeAvgReg_%s" % (suffix): loss_rot_ref * weight,
                "loss_tran_planeSoftReg_%s" % (suffix): loss_tran_ref_soft * weight,
                "loss_rot_planeSoftReg_%s" % (suffix): loss_rot_ref_soft * weight,
                "loss_rotIdx_%s" % (suffix): loss_rotIdx * 0.01 * weight,
                "loss_transIdx_%s" % (suffix): loss_transIdx * 0.02 * weight,
                "loss_paramL2_dist_%s" % (suffix): paramL2_errs * 0.1 * weight,
            }

        return losses, pred_cam

    def __inference_PlaneCamRefHead(self, initial_trans_feat, initial_rot_feat, geo_sequence_global_init,
                                        score_sequence_avg=None, gt_pose=None, geo_sequence_local=None, suffix="",
                                        matched_nums=None, out_cam_type='soft', weight=1.0, sig_seq=None,
                                        initial_trans=None, initial_rot=None):
        assert sig_seq is not None
        assert out_cam_type in ['avg-all', 'soft', 'max-score', 'min-cost']
        assert not self.training
        bs, max_n, _ = geo_sequence_global_init.shape

        # -------------------------------------------------------geo encoding------------------------------------
        if self.warp_plane_in_cam_ref_on:
            # get normal and offset
            geo_sequence_0 = geo_sequence_global_init[:, :, :3]  # bs, n, 3
            offset_0 = torch.norm(geo_sequence_0, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_0 = geo_sequence_0 / (offset_0 + 1e-10)  # bs, n, 3
            # get normal and offset
            geo_sequence_1 = geo_sequence_global_init[:, :, 3:]  # bs, n, 3
            offset_1 = torch.norm(geo_sequence_1, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_1 = geo_sequence_1 / (offset_1 + 1e-10)  # bs, n, 3
            offset_0 = offset_0 * sig_seq
            normal_0 = normal_0 * sig_seq
        else:
            # get normal and offset
            geo_sequence_0 = geo_sequence_local[:, :, :3]  # bs, n, 3
            offset_0 = torch.norm(geo_sequence_0, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_0 = geo_sequence_0 / (offset_0 + 1e-10)  # bs, n, 3
            # get normal and offset
            geo_sequence_1 = geo_sequence_local[:, :, 3:]  # bs, n, 3
            offset_1 = torch.norm(geo_sequence_1, p=2, dim=-1, keepdim=True)  # bs, n, 1
            normal_1 = geo_sequence_1 / (offset_1 + 1e-10)  # bs, n, 3

        # calculate geo feat
        geo_sequence_new = torch.cat((normal_0, offset_0, normal_1, offset_1), dim=-1)  # bs, n, 8
        geo_fea_all = self.geo_encoder(geo_sequence_new)  # bs, n, 1024
        geo_fea_s1_all = self.geo_proj_s1(geo_fea_all)  # bs, n, 1024
        geo_fea_rot_all = self.decoder_rot(geo_fea_s1_all)  # bs, n, 256
        geo_fea_s2_all = self.geo_proj_s2(torch.cat([geo_fea_s1_all, geo_fea_rot_all], dim=-1))  # bs, n, 1024
        geo_fea_tran_all = self.decoder_tran(geo_fea_s2_all)  # bs, n, 256

        if matched_nums[0] == 0:
            losses = {}
            pred_cam = {"pred_trans": initial_trans, "pred_rot": initial_rot,
                        "pred_trans_avg": initial_trans, "pred_rot_avg": initial_rot,
                        }
            return losses, pred_cam

        # ----------------------------------------calculate one plane pose embedding------------------------------------
        # generate mask
        matching_mask = torch.zeros(bs, max_n + 1, max_n).to(device=initial_rot_feat.device,
                                                             dtype=torch.float32)  # bs, n+1, n
        for i in range(bs):
            m = matched_nums[i]
            matching_mask[i, :m + 1, :m] = 1.
        matching_mask = matching_mask.contiguous().detach()
        # padding initial pose
        initial_trans_feat_pad = initial_trans_feat.unsqueeze(1).repeat(1, max_n, 1).view(bs * max_n, -1)  # b*n, 256
        initial_rot_feat_pad = initial_rot_feat.unsqueeze(1).repeat(1, max_n, 1).view(bs * max_n, -1)  # b*n, 256
        # calculate one plane pose embedding
        fused_rots_feat_all = torch.cat((initial_rot_feat_pad, geo_fea_rot_all.view(bs * max_n, -1)),dim=-1)  # bs*n, 512
        fused_rots_feat_all = F.relu(self.decoder_rot2(fused_rots_feat_all))  # bs*n, 256
        fused_trans_feat_all = torch.cat((initial_trans_feat_pad, geo_fea_tran_all.view(bs * max_n, -1)),dim=-1)  # bs*n, 512
        fused_trans_feat_all = F.relu(self.decoder_tran2(fused_trans_feat_all))  # bs*n, 256

        # ------------------------------------------calculate one plane rot and scoring-------------------------------
        # rot estimate
        rots_ref_all = F.normalize(self.rots(fused_rots_feat_all), dim=-1, p=2).view(bs, max_n, 4)  # bs, n, 4
        rots_ref_all = torch.cat([initial_rot.unsqueeze(1), rots_ref_all], dim=1)  # bs, n+1, 4
        # calculate normal dist
        trans_zero_all = torch.zeros(bs, max_n + 1, 3).to(device=rots_ref_all.device, dtype=torch.float32)
        plane1_mid_r = self.warp_single_view_plane_param_to_global(geo_sequence_local[:, :, 3:],
                                                                   pose_n=1)  # bs, 1, n, 3
        plane1_mid_r = plane1_mid_r.repeat(1, max_n + 1, 1, 1)  # bs, n+1, n, 3
        plane0_mid_r = self.warp_single_view_plane_param_to_global(geo_sequence_local[:, :, :3], rots_ref_all,
                                                                   trans_zero_all)  # bs, n+1, n, 3
        normal0_mid_r = F.normalize(plane0_mid_r, p=2, dim=-1)  # bs, n+1, n, 3
        normal1_mid_r = F.normalize(plane1_mid_r, p=2, dim=-1)  # bs, n+1, n, 3
        dist_normalAngle_mid = torch.acos(
            torch.clamp(torch.sum(normal0_mid_r * normal1_mid_r, dim=-1), min=-1., max=1.)) / np.pi * 180.
        dist_normalL2_mid_r = torch.norm(normal0_mid_r - normal1_mid_r, p=2, dim=-1)  # bs, n+1, n ; 0 ~ +
        dist_normalL2_mid_r = dist_normalL2_mid_r * matching_mask  # bs, n+1, n ; 0 ~ +
        dist_normalL2_sum = dist_normalL2_mid_r.sum(-1)  # bs, n+1
        score_normalL2_mid_r = torch.exp(-dist_normalL2_mid_r) * matching_mask  # bs, n+1, n ; 0 ~ +
        score_normalL2_mid_r = self.normal_score_proj(score_normalL2_mid_r)  # bs, n+1, 64

        score_soft_rot_temp = self.rot_score_reg(score_normalL2_mid_r)  # bs, n+1, 1
        score_soft_rot = torch.zeros_like(score_soft_rot_temp)
        for i in range(bs):
            m = matched_nums[i]
            score_soft_rot[i, :m + 1] = score_soft_rot_temp[i, :m + 1].softmax(0)  # bs, n+1, 1
        score_soft_rot = score_soft_rot * matching_mask[:, :, 0:1]  # bs, n+1, 1

        # -----------------------------------------calculate one plane trans and scoring -------------------------------
        # trans estimate
        trans_ref_all = self.trans(fused_trans_feat_all).view(bs, max_n, 3)  # bs, n, 3
        trans_ref_all = torch.cat([initial_trans.unsqueeze(1), trans_ref_all], dim=1)  # bs, n+1, 3
        # calculate trans cost
        plane0_mid_rt = self.warp_single_view_plane_param_to_global(geo_sequence_local[:, :, :3], rots_ref_all,
                                                                    trans_ref_all)  # bs, n+1, n, 3
        plane1_mid_rt = plane1_mid_r  # bs, n+1, n, 3
        offset0_mid_rt = torch.norm(plane0_mid_rt, p=2, dim=-1)  # bs, n+1, n
        offset1_mid_rt = torch.norm(plane1_mid_rt, p=2, dim=-1)  # bs, n+1, n
        normal0_mid_rt = F.normalize(plane0_mid_rt, p=2, dim=-1)  # bs, n+1, n, 3
        normal1_mid_rt = F.normalize(plane1_mid_rt, p=2, dim=-1)  # bs, n+1, n, 3
        nTn = torch.sum(normal0_mid_rt * normal1_mid_rt, dim=-1)  # bs, n+1, n
        dist_offset_mid_rt = torch.abs(offset0_mid_rt - offset1_mid_rt)  # bs, n+1, n ; 0 ~ +
        dist_offset_mid_rt[nTn < 0] = torch.abs(offset0_mid_rt + offset1_mid_rt)[nTn < 0]
        dist_l2_mid_ori = torch.norm(plane0_mid_rt - plane1_mid_rt, p=2, dim=-1)  # bs, n+1, n ; 0 ~ +
        dist_l2_mid_ori_sum = (dist_l2_mid_ori * matching_mask).sum(-1)
        # use l2 dist to re-score translation
        dist_l2_mid = dist_l2_mid_ori * matching_mask  # bs, n+1, n ; 0 ~ +
        dist_l2_mid = torch.exp(-dist_l2_mid) * matching_mask  # bs, n+1, n ; 0 ~ +
        dist_l2_mid = self.param_score_proj(dist_l2_mid)  # bs, n+1, 64

        score_soft_trans_temp = self.trans_score_reg(dist_l2_mid)  # bs, n+1, 1
        score_soft_trans = torch.zeros_like(score_soft_trans_temp)
        for i in range(bs):
            m = matched_nums[i]
            score_soft_trans[i, :m + 1] = score_soft_trans_temp[i, :m + 1].softmax(0)  # bs, n+1, 1
        score_soft_trans = score_soft_trans * matching_mask[:, :, 0:1]  # bs, n+1, 1

        # -----------------------------------------pose selection-------------------------------
        # ---------------------------  Avg.
        score_avgAll_trans = torch.ones_like(score_soft_trans) * matching_mask[:, :, 0:1]  # bs, n+1, 1
        score_avgAll_trans = score_avgAll_trans / (score_avgAll_trans.sum(dim=1, keepdim=True) + 1e-10)  # bs, n+1, 1
        score_avgAll_rot = torch.ones_like(score_soft_rot) * matching_mask[:, :, 0:1]  # bs, n+1, 1
        score_avgAll_rot = score_avgAll_rot / (score_avgAll_rot.sum(dim=1, keepdim=True) + 1e-10)  # bs, n+1, 1

        if matched_nums[0] > 1:
            fused_trans_feat_avg = torch.cat(
                (initial_trans_feat.unsqueeze(1), fused_trans_feat_all.reshape(bs, max_n, 256)), dim=1)  # b, n+1, c
            fused_trans_feat_avg = torch.sum(fused_trans_feat_avg * score_avgAll_trans, dim=1)  # b, c
            fused_rots_feat_avg = torch.cat(
                (initial_rot_feat.unsqueeze(1), fused_rots_feat_all.reshape(bs, max_n, 256)), dim=1)
            fused_rots_feat_avg = torch.sum(fused_rots_feat_avg * score_avgAll_rot, dim=1)  # b, c
        else:
            fused_trans_feat_avg = (
                fused_trans_feat_all.reshape(bs, max_n, 256) * score_avgAll_trans[:, 1:] / score_avgAll_trans[:,1:].sum(dim=1, keepdim=True)).sum(dim=1)  # b, 256
            fused_rots_feat_avg = (
                fused_rots_feat_all.reshape(bs, max_n, 256) * score_avgAll_rot[:, 1:] / score_avgAll_rot[:, 1:].sum(dim=1, keepdim=True)).sum(dim=1)  # b, 256

        rots_ref_avg = F.normalize(self.rots(fused_rots_feat_avg), dim=-1, p=2)
        trans_ref_avg = self.trans(fused_trans_feat_avg)

        if matched_nums[0] <= 1:
            pred_cam = {"pred_trans": trans_ref_avg, "pred_rot": rots_ref_avg,
                        "pred_trans_avg": trans_ref_avg, "pred_rot_avg": rots_ref_avg,
                        }
            if torch.isnan(rots_ref_avg.sum()):
                import pdb;
                pdb.set_trace()
            return {}, pred_cam

        if out_cam_type == 'avg-all':
            trans_ref_final = trans_ref_avg
            rots_ref_final = rots_ref_avg
        elif out_cam_type == 'soft':
            # --------------------------- Soft
            fused_trans_feat_soft = torch.cat((initial_trans_feat.unsqueeze(1), fused_trans_feat_all.reshape(bs, max_n, 256)), dim=1)  # b, n+1, c
            fused_trans_feat_soft = torch.sum(fused_trans_feat_soft * score_soft_trans, dim=1)
            fused_rots_feat_soft = torch.cat((initial_rot_feat.unsqueeze(1), fused_rots_feat_all.reshape(bs, max_n, 256)), dim=1)
            fused_rots_feat_soft = torch.sum(fused_rots_feat_soft * score_soft_rot, dim=1)
            rots_ref_final = F.normalize(self.rots(fused_rots_feat_soft), dim=-1, p=2)
            trans_ref_final = self.trans(fused_trans_feat_soft)
        elif out_cam_type == 'min-cost':
            print("Warning: you are using pose with min geo cost!")
            _, selected_rot_idxs = dist_normalL2_sum[0:1, :matched_nums[0] + 1].min(-1)  # b, n+1
            rots_ref_final = rots_ref_all[0:1, selected_rot_idxs[0]]  # 1, 4
            _, selected_trans_idxs = dist_l2_mid_ori_sum[0:1, :matched_nums[0] + 1].min(-1)  # b, n+1
            trans_ref_final = trans_ref_all[0:1, selected_trans_idxs[0]]  # 1, 4
        elif out_cam_type == 'max-score':
            print("Warning: you are using pose with highest score!")
            _, selected_rot_idxs = score_soft_rot[0:1, :matched_nums[0] + 1, 0].max(-1)  # b, n+1
            rots_ref_final = rots_ref_all[0:1, selected_rot_idxs[0]]  # 1, 4
            _, selected_trans_idxs = score_soft_trans[0:1, :matched_nums[0] + 1, 0].max(-1)  # b, n+1
            trans_ref_final = trans_ref_all[0:1, selected_trans_idxs[0]]  # 1, 4

        pred_cam = {"pred_trans": trans_ref_final, "pred_rot": rots_ref_final,
                    "pred_trans_avg": trans_ref_avg, "pred_rot_avg": rots_ref_avg,
                    "all_pred_trans": trans_ref_all[0:1, :matched_nums[0] + 1],
                    "all_pred_rots": rots_ref_all[0:1, :matched_nums[0] + 1],
                    "score_soft_rot": score_soft_rot[0:1, :matched_nums[0] + 1],
                    "score_soft_offset": score_soft_trans[0:1, :matched_nums[0] + 1],
                    "l2_dist": dist_l2_mid_ori[0:1, :matched_nums[0] + 1, :matched_nums[0]],
                    "normal_dist": dist_normalAngle_mid[0:1, :matched_nums[0] + 1, :matched_nums[0]],
                    "offset_dist": dist_offset_mid_rt[0:1, :matched_nums[0] + 1, :matched_nums[0]],
                    }
        losses = {}

        if torch.isnan(rots_ref_final.sum()):
            import pdb; pdb.set_trace()
        return losses, pred_cam

    def compute_corr_softmax(self, im_feature1, im_feature2, att_map=None):
        _, _, h1, w1 = im_feature1.size()
        _, _, h2, w2 = im_feature2.size()
        im_feature2 = im_feature2.transpose(2, 3)  # _, _, w2, h2
        im_feature2_vec = im_feature2.contiguous().view(
            im_feature2.size(0), im_feature2.size(1), -1
        )  # b, c, w2h2
        im_feature2_vec = im_feature2_vec.transpose(1, 2)  # b, w2h2, c
        im_feature1_vec = im_feature1.contiguous().view(
            im_feature1.size(0), im_feature1.size(1), -1
        )  # b, c, h1w1
        corrfeat = torch.matmul(im_feature2_vec, im_feature1_vec)  # b, w2h2, h1w1

        corrfeat = corrfeat.view(corrfeat.size(0), h2 * w2, h1, w1)
        corrfeat = F.softmax(corrfeat, dim=1)

        return corrfeat

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

    def get_gt_geo_sequence(self, batched_inputs, pred_cams=None, gauss_noise_on=False, pre_gauss_noise=None,
                            scale_offset=0.1, scale_normal=10 / 180. * np.pi, device=None):
        geo_res = []
        score_res = []
        matched_num = []
        gauss_noises = None

        if pred_cams is not None:
            pred_trans = pred_cams['tran']  # b, 3
            pred_rots = pred_cams['rot']  # b, 3
        else:
            pred_trans = [None] * len(batched_inputs)
            pred_rots = [None] * len(batched_inputs)

        for per_target, pred_t, pred_r in zip(batched_inputs, pred_trans, pred_rots):
            if 'instances' in per_target['0']:
                plane1 = per_target['0']['instances'].gt_planes.to(device)  # n1, 3
                plane2 = per_target['1']['instances'].gt_planes.to(device)  # n2, 3
                n1 = plane1.shape[0]
                n2 = plane2.shape[0]
                if n1 > 50:
                    n1 = 50
                    plane1 = plane1[:n1]
                if n2 > 50:
                    n2 = 50
                    plane2 = plane2[:n2]
            else:
                n1 = len(per_target['0']['annotations'])
                plane1 = [ann['plane'] for ann in per_target['0']['annotations']]
                plane1 = torch.tensor(plane1).to(device=device)  # n1, 3

                n2 = len(per_target['1']['annotations'])
                plane2 = [ann['plane'] for ann in per_target['1']['annotations']]
                plane2 = torch.tensor(plane2).to(device=device)  # n1, 3

                if n1 > 50:
                    n1 = 50
                    plane1 = plane1[:n1]
                if n2 > 50:
                    n2 = 50
                    plane2 = plane2[:n2]

            gt_corrs = per_target["gt_corrs"]
            gt_corrs = torch.tensor(gt_corrs)  # m, 2

            idx1 = gt_corrs[:, 0]  # m
            idx2 = gt_corrs[:, 1]  # m
            idx_mask = (idx1 < 50) & (idx2 < 50)
            idx1 = idx1[idx_mask]
            idx2 = idx2[idx_mask]

            m = idx1.shape[0]

            if gauss_noise_on:
                if pre_gauss_noise is None:
                    import time
                    np.random.seed(int(time.time()))
                    noise1_offset = np.random.normal(loc=0.0, scale=scale_offset, size=(n1, 1))
                    noise1_offset = torch.from_numpy(noise1_offset).to(device=device, dtype=torch.float32)
                    noise1_normal = np.random.normal(loc=0.0, scale=scale_normal, size=(n1, 3)).astype(np.float32)

                    noise2_offset = np.random.normal(loc=0.0, scale=scale_offset, size=(n2, 1))
                    noise2_offset = torch.from_numpy(noise2_offset).to(device=device, dtype=torch.float32)
                    noise2_normal = np.random.normal(loc=0.0, scale=scale_normal, size=(n2, 3)).astype(np.float32)
                else:
                    noise1_offset = pre_gauss_noise["n1_offset"]
                    noise1_normal = pre_gauss_noise["n1_normal"]
                    noise2_offset = pre_gauss_noise["n2_offset"]
                    noise2_normal = pre_gauss_noise["n2_normal"]

                noise1_rot_matrix = build_rot_matrix_from_angle(torch.tensor(noise1_normal[:, 0]*180/np.pi).to(device),
                                                                torch.tensor(noise1_normal[:, 1]*180/np.pi).to(device),
                                                                torch.tensor(noise1_normal[:, 2]*180/np.pi).to(device)).to(
                    device=device, dtype=torch.float32)
                noise2_rot_matrix = build_rot_matrix_from_angle(torch.tensor(noise2_normal[:, 0] * 180 / np.pi).to(device),
                                                                torch.tensor(noise2_normal[:, 1] * 180 / np.pi).to(device),
                                                                torch.tensor(noise2_normal[:, 2] * 180 / np.pi).to(device)).to(
                    device=device, dtype=torch.float32)
                normal1 = torch.bmm(noise1_rot_matrix, F.normalize(plane1, dim=-1).unsqueeze(-1)).squeeze(-1)
                normal1 = F.normalize(normal1, dim=-1)
                offset1 = torch.norm(plane1, dim=-1, keepdim=True) + noise1_offset
                plane1 = offset1 * normal1

                normal2 = torch.bmm(noise2_rot_matrix, F.normalize(plane2, dim=-1).unsqueeze(-1)).squeeze(-1)
                normal2 = F.normalize(normal2, dim=-1)
                offset2 = torch.norm(plane2, dim=-1, keepdim=True) + noise2_offset
                plane2 = offset2 * normal2

                gauss_noises = {"n1_offset": noise1_offset,
                                "n1_normal": noise1_normal,
                                "n2_offset": noise2_offset,
                                "n2_normal": noise2_normal}

            # ---------------------------------------warp plane param
            if pred_cams is not None:
                if isinstance(pred_t, torch.Tensor):
                    tran = pred_t
                    rot = self.quaternion2rotmatrix(pred_r).squeeze(0)  # 3, 3
                else:
                    tran = torch.FloatTensor(pred_t).to(device)
                    rot = quaternion.from_float_array(pred_r)
                    rot = torch.FloatTensor(quaternion.as_rotation_matrix(rot)).to(device)

                # plane of the first view
                start = torch.ones((len(plane1), 3)).to(device) * tran
                end = plane1 * torch.tensor([1, -1, -1]).to(device)  # suncg2habitat
                end = (
                        torch.mm(
                            rot,
                            (end).T,
                        ).T
                        + tran
                )  # cam2world
                a = end
                b = end - start
                plane1 = ((a * b).sum(dim=1) / (torch.norm(b, dim=1) + 1e-5) ** 2).view(-1, 1) * b

                # plane of the second view
                plane2 = plane2 * torch.FloatTensor([1, -1, -1]).to(device)
                # ---------------------------------------warp plane param


            plane1_pad = plane1.unsqueeze(1).expand(n1, n2, 3)
            plane2_pad = plane2.unsqueeze(0).expand(n1, n2, 3)

            plane_cat = torch.cat((plane1_pad, plane2_pad), dim=-1)  # n1, n2, 6

            geo_matrix = torch.zeros(50, 50, 6).to(dtype=plane_cat.dtype, device=device)
            score_matrix = torch.zeros(50, 50).to(dtype=plane_cat.dtype, device=device)

            geo_matrix[:n1, :n2, :] = plane_cat

            geo_matrix = geo_matrix.contiguous()

            score_matrix[idx1, idx2] = 1
            score_matrix = score_matrix.unsqueeze(-1).contiguous()  # 50, 50, 1

            geo_sequence = geo_matrix[idx1, idx2]
            score_sequence = score_matrix[idx1, idx2]

            geo_sequence_pad = torch.zeros([50, 6]).to(dtype=geo_sequence.dtype, device=device)
            geo_sequence_pad[:m] = geo_sequence
            geo_sequence_pad = geo_sequence_pad.contiguous()
            score_sequence_pad = torch.zeros([50, 1]).to(dtype=geo_sequence.dtype, device=device)
            score_sequence_pad[:m] = score_sequence
            score_sequence_pad = score_sequence_pad.contiguous()

            geo_res.append(geo_sequence_pad)
            score_res.append(score_sequence_pad)
            matched_num.append(m)

        geo_res = torch.stack(geo_res, dim=0)  # bs, 50, 6
        score_res = torch.stack(score_res, dim=0)  # bs, 50, 1

        return geo_res, score_res, matched_num, gauss_noises

    def get_gt_global_geo_sequence(self, gt_geo_sequence_local, pred_cams):
        pred_trans = pred_cams['tran']  # b, 3
        pred_rots = pred_cams['rot']  # b, 3

        plane1 = gt_geo_sequence_local[:, :, :3]  # b, n, 3
        plane2 = gt_geo_sequence_local[:, :, 3:]  # b, n, 3

        plane1_global = self.warp_single_view_plane_param_to_global(
            plane1, pred_rots.unsqueeze(1), pred_trans.unsqueeze(1))[:, 0, :, :]  # b, n, 3

        plane2_global = self.warp_single_view_plane_param_to_global(
            plane2, pose_n=1)[:, 0, :, :]  # b, n, 3

        geo_res = torch.cat([plane1_global, plane2_global], dim=-1)  # b, n, 6

        return geo_res

    def get_pred_geo_sequence(self, planes1, planes2, pred_assignment_matrix, pred_score_matrix=None, pred_cams=None, device=None):
        if pred_score_matrix is None:
            pred_score_matrix = [None] * len(planes1)

        if pred_cams is not None:
            pred_trans = pred_cams['tran']  # b, 3
            pred_rots = pred_cams['rot']  # b, 3
        else:
            pred_trans = [None] * len(planes1)
            pred_rots = [None] * len(planes1)

        geo_res = []
        score_res = []
        matched_num = []
        for plane1, plane2, assignment_matrix, score_matrix, pred_t, pred_r in zip(planes1, planes2, pred_assignment_matrix, pred_score_matrix, pred_trans, pred_rots):
            assert plane1.shape[0] == assignment_matrix.shape[0] and plane2.shape[0] == assignment_matrix.shape[1]
            idxs = torch.nonzero(assignment_matrix)  # m, 2
            m = idxs.shape[0]
            idx1 = idxs[:, 0]  # m
            idx2 = idxs[:, 1]  # m
            m_plane1 = plane1[idx1]  # m, 3
            m_plane2 = plane2[idx2]  # m, 3
            # ---------------------------------------warp plane param
            if pred_cams is not None:
                if self.training:
                    assert isinstance(pred_t, torch.Tensor) and isinstance(pred_r, torch.Tensor), "In training stage, you must input a tensor!"
                if isinstance(pred_t, torch.Tensor):
                    tran = pred_t
                    rot = self.quaternion2rotmatrix(pred_r).squeeze(0)  # 3, 3
                else:
                    tran = torch.FloatTensor(pred_t).to(device)
                    rot = quaternion.from_float_array(pred_r)
                    rot = torch.FloatTensor(quaternion.as_rotation_matrix(rot)).to(device)

                # plane of the first view
                start = torch.ones((len(m_plane1), 3)).to(device) * tran
                end = m_plane1 * torch.tensor([1, -1, -1]).to(device)  # suncg2habitat
                end = (
                        torch.mm(
                            rot,
                            (end).T,
                        ).T
                        + tran
                )  # cam2world
                a = end
                b = end - start
                m_plane1 = ((a * b).sum(dim=1) / (torch.norm(b, dim=1) + 1e-5) ** 2).view(-1, 1) * b

                # plane of the second view
                m_plane2 = m_plane2 * torch.FloatTensor([1, -1, -1]).to(device)
                # ---------------------------------------warp plane param

            geo_sequence = torch.cat((m_plane1, m_plane2), dim=-1)  # m, 6

            geo_sequence_pad = torch.zeros([self.num_queries, 6]).to(dtype=geo_sequence.dtype, device=device)
            geo_sequence_pad[:m] = geo_sequence
            geo_sequence_pad = geo_sequence_pad.contiguous()

            score_sequence_pad = torch.zeros([self.num_queries, 1]).to(dtype=geo_sequence.dtype, device=device)
            if score_matrix is None:
                score_sequence_pad[:m] = 1
            else:
                score_sequence = score_matrix[idx1, idx2]  # m
                score_sequence_pad[:m] = score_sequence.view(m, 1)
            score_sequence_pad = score_sequence_pad.contiguous()

            geo_res.append(geo_sequence_pad)
            score_res.append(score_sequence_pad)
            matched_num.append(m)

        geo_res = torch.stack(geo_res, dim=0)  # bs, 50, 6
        score_res = torch.stack(score_res, dim=0)  # bs, 50, 1

        return geo_res, score_res, matched_num

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