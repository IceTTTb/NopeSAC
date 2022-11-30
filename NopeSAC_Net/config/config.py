# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def get_sparseplane_cfg_defaults(cfg):
    """
    Add config for NopeSAC.
    """
    # solver config
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.SEM_SEG_HEAD_MULTIPLIER = 1.0
    cfg.SOLVER.PLANE_MATCHER_HEAD_MULTIPLIER = 1.0

    # mask_former model config
    cfg.MODEL.FREEZE = []
    cfg.MODEL.DEPTH_ON = False
    cfg.MODEL.EMBEDDING_ON = False
    cfg.MODEL.CAMERA_ON = False
    cfg.MODEL.MASK_ON = True

    # loss types
    cfg.MODEL.HUNGARIAN_MATCHER_ON = True
    cfg.MODEL.LOSS_DETECTION_ON = True
    cfg.MODEL.LOSS_CAMERA_ON = False
    cfg.MODEL.LOSS_EMB_ON = False

    # PlaneTR config
    # loss weights
    cfg.MODEL.SEM_SEG_HEAD.DEEP_SUPERVISION = True
    cfg.MODEL.SEM_SEG_HEAD.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.DICE_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.MASK_WEIGHT = 20.0
    cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_L1 = 0.5
    cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_COS = 10.
    cfg.MODEL.SEM_SEG_HEAD.PARAM_HM_WEIGHT_L1 = 0.5
    cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_Q = 1.0
    cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_CENTER_INS = 0.5
    cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_ANGLE = 0.0028
    cfg.MODEL.SEM_SEG_HEAD.PARAM_WEIGHT_OFFSET = 0.01
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.CENTER_ON = False
    cfg.MODEL.SEM_SEG_HEAD.PARAM_ON = False
    cfg.MODEL.SEM_SEG_HEAD.PARAM_IN_MATCHER = True
    cfg.MODEL.SEM_SEG_HEAD.NHEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.ENC_LAYERS = 6
    cfg.MODEL.SEM_SEG_HEAD.DEC_LAYERS = 6
    cfg.MODEL.SEM_SEG_HEAD.NUM_OBJECT_QUERIES = 50
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIM = 256

    # ------------------------------------------------------------------------ #
    # Camera Predict Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.CAMERA_BRANCH = "CACHED"  # For inference
    cfg.MODEL.CAMERA_HEAD = CN()
    cfg.MODEL.CAMERA_HEAD.NAME = ""
    cfg.MODEL.CAMERA_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH = "./camCls/kmeans_trans_32.pkl"
    cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH = "./camCls/kmeans_rots_32.pkl"
    cfg.MODEL.CAMERA_HEAD.TRANS_CLASS_NUM = 32
    cfg.MODEL.CAMERA_HEAD.ROTS_CLASS_NUM = 32
    cfg.MODEL.CAMERA_HEAD.FEATURE_SIZE = 64
    cfg.MODEL.CAMERA_HEAD.BACKBONE_FEATURE = "res3"
    cfg.MODEL.CAMERA_HEAD.REFINE_ON = False
    cfg.MODEL.CAMERA_HEAD.CAM_REC_ON = False
    cfg.MODEL.CAMERA_HEAD.RAND_ON = False
    cfg.MODEL.CAMERA_HEAD.PIXEL_CAM_FIX_ON = False
    cfg.MODEL.CAMERA_HEAD.INFERENCE_OUT_CAM_TYPE = "soft"
    cfg.MODEL.CAMERA_HEAD.INITIAL_CAM_WEIGHT = 1.0
    cfg.MODEL.CAMERA_HEAD.PLANE_CAM_WEIGHT = 1.0
    cfg.MODEL.CAMERA_HEAD.PLANE_CAM_WEIGHT_PREDPLANE = 0.1
    cfg.MODEL.CAMERA_HEAD.CLASSIFICATION_ON = False
    cfg.MODEL.CAMERA_HEAD.INFERENCE_SP_TOPCAM_ON = False
    cfg.MODEL.CAMERA_HEAD.INFERENCE_SP_TOPCAM_PATH = ''
    cfg.MODEL.CAMERA_HEAD.WARP_PLANE_IN_CAM_REF_ON = True

    # ------------------------------------------------------------------------ #
    # Matching Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.MATCHING_HEAD = CN()
    cfg.MODEL.MATCHING_HEAD.NAME = ""
    cfg.MODEL.MATCHING_HEAD.INITIAL_CAM_ON = True
    cfg.MODEL.MATCHING_HEAD.OFFSET_MULTIPLIER = 4.
    cfg.MODEL.MATCHING_HEAD.NORMAL_MULTIPLIER = 8.

    # ------------------------------------------------------------------------ #
    # Test config
    # ------------------------------------------------------------------------ #
    cfg.TEST.EVAL_GT_BOX = False
    cfg.TEST.OVERLAP_THRESHOLD = 0.6
    cfg.TEST.PLANE_SCORE_THRESHOLD = 0.6
    cfg.TEST.MASK_PROB_THRESHOLD = 0.5
    cfg.TEST.EVAL_FULL_SCENE = False
    cfg.TEST.MATCHING_SCORE_THRESHOLD = 0.2
    cfg.TEST.POSE_REFINEMENT_WITH_GT_MATCHERS = False
    cfg.TEST.POSE_REFINEMENT_WITH_GT_NOISE_MATCHERS = False
    cfg.TEST.POSE_REFINEMENT_WITH_GT_NOISE_MATCHERS_OFFSET_SCALE = 0.1
    cfg.TEST.POSE_REFINEMENT_WITH_GT_NOISE_MATCHERS_NORMAL_SCALE = 10.

    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
    cfg.DATALOADER.AUGMENTATION = False

    # ------------------------------------------------------------------------ #
    # Debug config
    # ------------------------------------------------------------------------ #
    cfg.DEBUG_ON = False
    cfg.DEBUG_CAMERA_ON = False

    cfg.SEED = 42
    cfg.FIX_SEED = True

    cfg.DATASETS.ROOT_DIR = ""

