import numpy as np
import os
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
import copy
from typing import Any, Dict, List, Set
import itertools
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.solver.build import get_default_optimizer_params
# required so that .register() calls are executed in module scope

import NopeSAC_Net.modeling  # noqa
from NopeSAC_Net.config import get_sparseplane_cfg_defaults
from NopeSAC_Net.data import PlaneRCNNMapper as dataMapper
from NopeSAC_Net.evaluation import MP3DEvaluator

import random
import numpy as np
import shutil
import time

import logging
logger = logging.getLogger(__name__)
if not logger.isEnabledFor(logging.INFO):
    setup_logger(name=__name__)

def copy_all_code(src_dir, dst_dir, include_dir=['configs', 'NopeSAC_Net', 'tools', 'configsV2', 'configsV3']):
    for files in os.listdir(src_dir):
        name = os.path.join(src_dir, files)
        back_name = os.path.join(dst_dir, files)
        if os.path.isfile(name):
            if 'LICENSE' in name:
                continue
            if not os.path.exists(back_name):
                shutil.copy(name, back_name)
            else:
                raise
        else:
            if files in include_dir:
                if not os.path.exists(back_name):
                    # os.makedirs(back_name)
                    shutil.copytree(name, back_name)
                else:
                    raise

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type  # defined in register_mp3d.py
        if evaluator_type == "mp3d":
            return MP3DEvaluator(dataset_name, cfg, True, output_dir=cfg.OUTPUT_DIR)
        else:
            raise ValueError("The evaluator type is wrong")

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=dataMapper(cfg, False, dataset_names=(dataset_name,)),
        )

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_names = cfg.DATASETS.TRAIN
        return build_detection_train_loader(
            cfg, mapper=dataMapper(cfg, True, dataset_names=dataset_names)
        )

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        # print('weight_decay_norm = ', weight_decay_norm)
        # print()
        # import pdb; pdb.set_trace()

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        count = 0.
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER

                if "sem_seg_head" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.SEM_SEG_HEAD_MULTIPLIER

                if "plane_matcher_net" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.PLANE_MATCHER_HEAD_MULTIPLIER

                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed

                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def vis(cls, cfg):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):

        Returns:
            dict: a dict of result metrics
        """
        # todo
        return {}

    @classmethod
    def test(cls, cfg, model):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):

        Returns:
            dict: a dict of result metrics
        """
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
        return results


def setup(args):
    cfg = get_cfg()
    get_sparseplane_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "meshrcnn" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="planeTR"
    )
    return cfg


def main(args):
    cfg = setup(args)
    if comm.is_main_process() and not args.eval_only:
        our_dir = os.path.join(cfg.OUTPUT_DIR, 'code')
        copy_all_code('./', our_dir)
    if cfg.FIX_SEED:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    # print(cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        #todo
        # res = Trainer.vis(cfg)

        return res

    trainer = Trainer(cfg)

    print("# of layers require gradient:")
    for c in trainer.checkpointer.model.named_children():
        grad = np.array(
            [
                param.requires_grad
                for param in getattr(trainer.checkpointer.model, c[0]).parameters()
            ]
        )
        print(c[0], grad.sum())
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
