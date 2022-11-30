import copy
import itertools
import json
import logging
import numpy as np
import pickle
import os
from collections import OrderedDict, Counter
from scipy.special import softmax
import detectron2.utils.comm as comm
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
# from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import setup_logger, create_small_table
from detectron2.utils.visualizer import Visualizer


from pycocotools.coco import COCO
from iopath.common.file_io import PathManager, file_lock
from sklearn.metrics import average_precision_score, roc_auc_score

from .detectron2coco import convert_to_coco_dict
import NopeSAC_Net.utils.VOCap as VOCap
from NopeSAC_Net.utils.metrics import compare_planes
from NopeSAC_Net.visualization import create_instances, get_labeled_seg, draw_match
import cv2
import time

logger = logging.getLogger(__name__)
if not logger.isEnabledFor(logging.INFO):
    setup_logger(name=__name__)

class MP3DEvaluator(COCOEvaluator):
    """
    Evaluate object proposal, instance detection, segmentation and affinity
    outputs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self.cfg = cfg

        self._logger2 = setup_logger(
            output=os.path.join(cfg.OUTPUT_DIR, "metrics.txt"), distributed_rank=comm.get_rank(), name="planeTR"
        )
        self._logger2.info("ckpt Dir: %s" % (cfg.MODEL.WEIGHTS))

        self.debug_on = cfg.DEBUG_ON
        self._tasks = self._tasks_from_config(cfg)
        self._plane_tasks = self._specific_tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self.eval_full_scene = cfg.TEST.EVAL_FULL_SCENE

        self._cpu_device = torch.device("cpu")
        self._device = cfg.MODEL.DEVICE
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._coco_api = COCO(self._siamese_to_coco(self._metadata.json_file))
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self._kpt_oks_sigmas = None
        self._filter_iou = 0.7
        self._filter_score = 0.7
        self.load_input_dataset(dataset_name)

        if self.cfg.MODEL.CAMERA_ON:
            kmeans_trans_path = cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH
            kmeans_rots_path = cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH
            assert os.path.exists(kmeans_trans_path)
            assert os.path.exists(kmeans_rots_path)
            with open(kmeans_trans_path, "rb") as f:
                self.kmeans_trans = pickle.load(f)
            with open(kmeans_rots_path, "rb") as f:
                self.kmeans_rots = pickle.load(f)

    def load_input_dataset(self, dataset):
        dataset_dict = {}
        dataset_list = list(DatasetCatalog.get(dataset))
        for dic in dataset_list:
            key0 = dic["0"]["image_id"]
            key1 = dic["1"]["image_id"]
            key = key0 + "__" + key1
            for i in range(len(dic["0"]["annotations"])):
                dic["0"]["annotations"][i]["bbox_mode"] = BoxMode(
                    dic["0"]["annotations"][i]["bbox_mode"]
                )
            for i in range(len(dic["1"]["annotations"])):
                dic["1"]["annotations"][i]["bbox_mode"] = BoxMode(
                    dic["1"]["annotations"][i]["bbox_mode"]
                )
            dataset_dict[key] = dic
        self.dataset_dict = dataset_dict

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ()
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        return tasks

    def _specific_tasks_from_config(self, cfg):
        tasks = ()
        if cfg.MODEL.EMBEDDING_ON and cfg.MODEL.MASK_ON:
            tasks = tasks + ("embedding",)
        if cfg.MODEL.CAMERA_ON:
            tasks = tasks + ("camera",)
        return tasks

    def _siamese_to_coco(self, siamese_json):
        assert self._output_dir
        save_json = os.path.join(self._output_dir, "siamese2coco.json")
        pm = PathManager()
        pm.mkdirs(os.path.dirname(save_json))
        with file_lock(save_json):
            if pm.exists(save_json):
                logger.warning(
                    f"Using previously cached COCO format annotations at '{save_json}'. "
                    "You need to clear the cache file if your dataset has been modified."
                )
            else:
                logger.info(
                    f"Converting annotations of dataset '{siamese_json}' to COCO format ...)"
                )
                with pm.open(siamese_json, "r") as f:
                    siamese_data = json.load(f)
                coco_data = {"data": []}
                exist_imgid = set()
                for key, datas in siamese_data.items():
                    # copy 'info', 'categories'
                    if key != "data":
                        coco_data[key] = datas
                    else:
                        for data in datas:
                            for i in range(2):
                                img_data = data[str(i)]
                                if img_data["image_id"] in exist_imgid:
                                    continue
                                else:
                                    exist_imgid.add(img_data["image_id"])
                                    coco_data[key].append(img_data)
                self._logger.info(f"Number of unique images: {len(exist_imgid)}.")
                coco_data = convert_to_coco_dict(coco_data["data"], self._metadata)
                with pm.open(save_json, "w") as f:
                    json.dump(coco_data, f)
        return save_json

    def _siamese_to_single(self, siamese_predictions):
        single_predictions = []
        exist_imgid = set()
        for pred in siamese_predictions:
            for i in range(2):
                single_pred = pred[str(i)]["instances"]
                if len(single_pred) == 0:
                    continue
                imgid = single_pred[0]["image_id"]
                if imgid in exist_imgid:
                    continue
                exist_imgid.add(imgid)
                single_predictions.append(pred[str(i)])
        return single_predictions

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"0": {}, "1": {}}
            tmp_instances = {"0": {}, "1": {}}
            for i in range(2):
                prediction[str(i)]["image_id"] = input[str(i)]["image_id"]
                prediction[str(i)]["file_name"] = input[str(i)]["file_name"]

                if output[str(i)] is not None and "instances" in output[str(i)]:
                    instances = output[str(i)]["instances"]
                    prediction[str(i)]["instances"] = instances

                if "annotations" in input[str(i)] and "segm" in self._tasks:
                    tmp_instances[str(i)]["gt_bbox"] = [
                        ann["bbox"] for ann in input[str(i)]["annotations"]
                    ]
                    if len(input[str(i)]["annotations"]) > 0:
                        tmp_instances[str(i)]["gt_bbox"] = np.array(
                            tmp_instances[str(i)]["gt_bbox"]
                        ).reshape(-1, 4)  # xywh from coco
                        original_mode = input[str(i)]["annotations"][0]["bbox_mode"]
                        tmp_instances[str(i)]["gt_bbox"] = BoxMode.convert(
                            tmp_instances[str(i)]["gt_bbox"],
                            BoxMode(original_mode),
                            BoxMode.XYXY_ABS,
                        )
                        prediction[str(i)]["pred_plane"] = output[str(i)]["pred_plane"]

                if output["depth"][str(i)] is not None:
                    prediction[str(i)]["pred_depth"] = output["depth"][str(i)].to(
                        self._cpu_device
                    )
                    depth_rst = get_depth_err(
                        output["depth"][str(i)], input[str(i)]["depth"].to(self._device)
                    )
                    prediction[str(i)]["depth_l1_dist"] = depth_rst.to(self._cpu_device)

            if "camera" in self._plane_tasks:
                gt_cam = {
                    "tran": input["rel_pose"]["position"],
                    "rot": input["rel_pose"]["rotation"],
                    "tran_cls": input["rel_pose"]["tran_cls"],
                    "rot_cls": input["rel_pose"]["rot_cls"],
                }
                for key in output:
                    if "camera" in key and "cls" not in key:
                        prediction[key] = {
                            "pred": output[key],
                            "gts": gt_cam,
                        }
                    elif "camera" in key and "cls" in key:
                        prediction[key] = {
                            "logits": {
                                'tran': output[key]['tran_cls_logit'].to(self._cpu_device),
                                'rot': output[key]['rot_cls_logit'].to(self._cpu_device),
                            },
                            "gts": gt_cam,
                        }
            if "embedding" in self._plane_tasks:
                for key in output:
                    if "assignment" in key:
                        prediction[key] = output[key].to(self._cpu_device)
            if "pred_aff" in output and output["pred_aff"] is not None:
                tmp_instances["pred_aff"] = output["pred_aff"].to(self._cpu_device)
            prediction["corrs"] = tmp_instances
            self._predictions.append(prediction)

    def get_optimized_dict(self, predictions):
        return_dict = {}
        for idx, prediction in enumerate(predictions):
            best_assignment = prediction["pred_assignment"].numpy()
            n_corr = best_assignment.sum()

            camera_dict = prediction["camera"]
            position = camera_dict["pred"]["tran"]
            rotation = camera_dict["pred"]["rot"]
            best_camera = {
                "position": position,
                "rotation": rotation
            }

            aux_cams = {}
            for key in prediction:
                if "camera" in key:
                    aux_cams[key] = {
                        "position": prediction[key]["pred"]["tran"],
                        "rotation": prediction[key]["pred"]["rot"]
                    }

            gt_position = camera_dict["gts"]["tran"]
            gt_rotation = camera_dict["gts"]["rot"]
            gt_camera = {
                "position": gt_position,
                "rotation": gt_rotation
            }

            plane_param0 = prediction["0"]["pred_plane"].cpu().numpy()
            plane_param1 = prediction["1"]["pred_plane"].cpu().numpy()
            plane_param_override = {
                "0": plane_param0,
                "1": plane_param1
            }

            img_id0 = prediction["0"]["image_id"]
            img_id1 = prediction["1"]["image_id"]
            image_ids = {
                "0": img_id0,
                "1": img_id1
            }

            optimized_dict = {
                "n_corr": n_corr,
                "cost": 0.1,
                "best_camera": best_camera,
                "gt_camera": gt_camera,
                "best_assignment": best_assignment,
                "plane_param_override": plane_param_override,
                "image_ids": image_ids
            }
            return_dict[idx] = optimized_dict

        return return_dict

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            gt_corrs = self._gt_corrs

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self.eval_full_scene:
            pm = PathManager()
            pm.mkdirs(self._output_dir)

            # save instance prediction
            file_path = os.path.join(self._output_dir, "NopeSAC_instances_predictions.pth")
            with pm.open(file_path, "wb") as f:
                torch.save(predictions, f)

            # save optimized dict
            optimized_dict = self.get_optimized_dict(predictions)
            save_dict(optimized_dict, self._output_dir, 'continuous')

        self._results = OrderedDict()
        if "segm" in self._tasks:
            single_predictions = self._siamese_to_single(predictions)

            if "instances" in single_predictions[0]:
                self._eval_plane(single_predictions)
            if "depth_l1_dist" in single_predictions[0]:
                self._eval_depth(single_predictions)

        if "embedding" in self._plane_tasks:
            self._eval_matching(predictions)

        if "camera" in self._plane_tasks:
            for key in predictions[0]:
                if 'onePP' in key:
                    continue
                if "camera" in key and 'cls' not in key:
                    self._eval_camera_reg(predictions, camera_name=key)
                elif "camera" in key and 'cls' in key:
                    raise NotImplementedError

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def xyz2class(self, x, y, z):
        return self.kmeans_trans.predict([[x, y, z]])

    def quat2class(self, w, xi, yi, zi):
        return self.kmeans_rots.predict([[w, xi, yi, zi]])

    def class2xyz(self, cls):
        assert (cls >= 0).all() and (cls < self.kmeans_trans.n_clusters).all()
        return self.kmeans_trans.cluster_centers_[cls]

    def class2quat(self, cls):
        assert (cls >= 0).all() and (cls < self.kmeans_rots.n_clusters).all()
        return self.kmeans_rots.cluster_centers_[cls]

    def _eval_camera_reg(self, predictions, camera_name="camera"):
        gt_tran = np.vstack([p[camera_name]["gts"]["tran"] for p in predictions])    # n_prediction, 3
        gt_rot = np.vstack([p[camera_name]["gts"]["rot"] for p in predictions])   # n_prediction, 4

        pred_tran = np.vstack([p[camera_name]["pred"]["tran"] for p in predictions])   # n_pred, 3
        pred_rot = np.vstack([p[camera_name]["pred"]["rot"] for p in predictions])   # n_pred, 4

        top1_error = {
            "tran": np.linalg.norm(gt_tran - pred_tran, axis=1),
            "rot": angle_error_vec(pred_rot, gt_rot),
        }
        top1_accuracy = {
            "tran": (top1_error["tran"] < 1.0).sum() / len(top1_error["tran"]),
            "rot": (top1_error["rot"] < 30).sum() / len(top1_error["rot"]),
        }
        top1_accuracy2 = {
            "tran": (top1_error["tran"] < 0.5).sum() / len(top1_error["tran"]),
            "rot": (top1_error["rot"] < 15).sum() / len(top1_error["rot"]),
        }
        top1_accuracy3 = {
            "tran": (top1_error["tran"] < 0.2).sum() / len(top1_error["tran"]),
            "rot": (top1_error["rot"] < 10).sum() / len(top1_error["rot"]),
        }

        camera_metrics = {
            f"T median err": np.median(top1_error["tran"]),
            f"T mean err": np.mean(top1_error["tran"]),
            f"T err < 1.0": top1_accuracy["tran"] * 100,
            f"T err < 0.5": top1_accuracy2["tran"] * 100,
            f"T err < 0.2": top1_accuracy3["tran"] * 100,

            f"R median err": np.median(top1_error["rot"]),
            f"R mean err": np.mean(top1_error["rot"]),
            f"R err < 30": top1_accuracy["rot"] * 100,
            f"R err < 15": top1_accuracy2["rot"] * 100,
            f"R err < 10": top1_accuracy3["rot"] * 100,
        }

        self._logger2.info("%s metrics (final output mode -> %s): \n"%(camera_name, self.cfg.MODEL.CAMERA_HEAD.INFERENCE_OUT_CAM_TYPE) + create_small_table(camera_metrics))

        self._results.update(camera_metrics)

        summary = {}
        return summary

    def _eval_plane(self, predictions):
        results = evaluate_for_planes(
            predictions,
            self._coco_api,
            self._metadata,
            device=self._device,
            _logger=self._logger2
        )
        self._results.update(results)

    def _eval_matching(self, predictions):
        results = evaluate_for_matchings(
            predictions,
            self.dataset_dict,
            device=self._device,
            _logger=self._logger2
        )
        self._results.update(results)

    def _eval_depth(self, predictions):
        depth_l1_dist = [p["depth_l1_dist"] for p in predictions]
        result = {f"depth_l1_dist": np.mean(depth_l1_dist)}
        self._logger2.info("Depth metrics: \n" + create_small_table(result))
        self._results.update(result)


def l1LossMask(pred, gt, mask):
    """L1 loss with a mask"""
    return torch.sum(torch.abs(pred - gt) * mask) / torch.clamp(mask.sum(), min=1)


def get_depth_err(pred_depth, gt_depth, device=None):
    l1dist = l1LossMask(pred_depth, gt_depth, (gt_depth > 1e-4).float())
    return l1dist


def angle_error_vec(v1, v2):
    assert v1.ndim == 2 and v2.ndim == 2
    return 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(v1, v2), axis=1)), -1.0, 1.0)) * 180 / np.pi

def evaluate_for_planes(
    predictions,
    dataset,
    metadata,
    iou_thresh=0.5,
    normal_threshold=30,
    offset_threshold=0.3,
    device=None,
    _logger=None,
):
    if device is None:
        device = torch.device("cpu")
    # classes
    cat_ids = sorted(dataset.getCatIds())  # [1]
    reverse_id_mapping = {
        v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
    }  # {0: 1}

    # initialize tensors to record mask AP, number of gt positives
    mask_apscores, mask_aplabels = {}, {}
    plane_apscores, plane_aplabels = {}, {}
    plane_normal_apscores, plane_normal_aplabels = {}, {}
    plane_offset_apscores, plane_offset_aplabels = {}, {}
    plane_offset_errs, plane_normal_errs = [], []
    npos = {}
    for cat_id in cat_ids:
        mask_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mask_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]

        plane_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        plane_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]

        plane_normal_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        plane_normal_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]

        plane_offset_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        plane_offset_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]

        npos[cat_id] = 0.0

    # number of gt positive instances per class
    for gt_ann in dataset.dataset["annotations"]:
        gt_label = gt_ann["category_id"]
        npos[gt_label] += 1.0

    for prediction in predictions:
        original_id = prediction["image_id"]
        image_width = dataset.loadImgs([original_id])[0]["width"]
        image_height = dataset.loadImgs([original_id])[0]["height"]
        if "instances" not in prediction:
            continue

        num_img_preds = len(prediction["instances"])
        if num_img_preds == 0:
            continue

        mask_apscores_temp, mask_aplabels_temp = {}, {}
        for cat_id in cat_ids:
            mask_apscores_temp[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
            mask_aplabels_temp[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]

        # predictions
        scores, labels, masks_rles = [], [], []
        for ins in prediction["instances"]:
            scores.append(ins["score"])  # sorted from high to low
            labels.append(ins["category_id"])  # [0, 0, ..., 0]
            masks_rles.append(ins["segmentation"])
        planes = prediction["pred_plane"]

        # ground truth
        # anotations corresponding to original_id (aka coco image_id)
        gt_ann_ids = dataset.getAnnIds(imgIds=[original_id])
        gt_anns = dataset.loadAnns(gt_ann_ids)
        # get original ground truth mask, label & mesh
        gt_labels, gt_mask_rles, gt_planes = [], [], []
        for ann in gt_anns:
            gt_labels.append(ann["category_id"])  # [1, 1, 1, ..., 1]
            if isinstance(ann["segmentation"], list):
                polygons = [np.array(p, dtype=np.float64) for p in ann["segmentation"]]
                try:
                    rles = mask_util.frPyObjects(polygons, image_height, image_width)
                except:
                    polygons = []
                    for p in ann["segmentation"]:
                        if len(p) > 4:
                            polygons.append(np.array(p, dtype=np.float64))
                    # exit()
                rle = mask_util.merge(rles)
            elif isinstance(ann["segmentation"], dict):  # RLE
                rle = ann["segmentation"]
            else:
                raise TypeError(
                    f"Unknown segmentation type {type(ann['segmentation'])}!"
                )
            gt_mask_rles.append(rle)
            gt_planes.append(ann["plane"])

        # mask iou
        miou = mask_util.iou(masks_rles, gt_mask_rles, [0] * len(gt_mask_rles))  # shape: [n_pred, n_gt]
        m_iou_tensor = torch.from_numpy(miou)

        plane_metrics = compare_planes(planes, gt_planes)  # dict: {'norm': tensor[n_pred, n_gt], 'offset': tensor[n_pred, n_gt]}

        # sort predictions in descending order
        scores = torch.tensor(np.array(scores), dtype=torch.float32)
        scores_sorted, idx_sorted = torch.sort(scores, descending=True)
        # record assigned gt.
        mask_covered = []
        plane_covered = []
        plane_normal_covered = []
        plane_offset_covered = []

        for pred_id in range(num_img_preds):
            # remember we only evaluate the preds that have overlap more than
            # iou_filter with the ground truth prediction

            # Assign pred to gt
            gt_id = torch.argmax(m_iou_tensor[idx_sorted[pred_id]])  # idx of gt plane which has max iou with current predicted plane
            gt_label = gt_labels[gt_id]  # 1
            # map to dataset category id
            pred_label = reverse_id_mapping[labels[idx_sorted[pred_id]]]
            pred_miou = miou[idx_sorted[pred_id], gt_id]
            pred_score = scores[idx_sorted[pred_id]].view(1).to(device)

            normal = plane_metrics["norm"][idx_sorted[pred_id], gt_id].item()
            offset = plane_metrics["offset"][idx_sorted[pred_id], gt_id].item()
            plane_offset_errs.append(offset)
            plane_normal_errs.append(normal)

            # mask
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)  # true positive & false positive, default: 0 (false positive)
            if (
                (pred_label == gt_label)
                and (pred_miou > iou_thresh)
                and (gt_id not in mask_covered)  # a gt plane can only be assigned to one predicted plane
            ):
                tpfp[0] = 1  # it is a true positive
                mask_covered.append(gt_id)
            mask_apscores[pred_label].append(pred_score)
            mask_aplabels[pred_label].append(tpfp)

            mask_apscores_temp[pred_label].append(pred_score)
            mask_aplabels_temp[pred_label].append(tpfp)

            # plane mask+normal+offset
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (normal < normal_threshold)
                and (offset < offset_threshold)
                and (pred_miou > iou_thresh)
                and (gt_id not in plane_covered)  # a gt plane can only be assigned to one predicted plane
            ):
                tpfp[0] = 1  # it is a true positive
                plane_covered.append(gt_id)
            plane_apscores[pred_label].append(pred_score)
            plane_aplabels[pred_label].append(tpfp)

            # plane mask+normal
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (normal < normal_threshold)
                and (pred_miou > iou_thresh)
                and (gt_id not in plane_normal_covered)  # a gt plane can only be assigned to one predicted plane
            ):
                tpfp[0] = 1  # it is a true positive
                plane_normal_covered.append(gt_id)
            plane_normal_apscores[pred_label].append(pred_score)
            plane_normal_aplabels[pred_label].append(tpfp)

            # plane mask+offset
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_miou > iou_thresh)
                and (offset < offset_threshold)
                and (gt_id not in plane_offset_covered)  # a gt plane can only be assigned to one predicted plane
            ):
                tpfp[0] = 1  # it is a true positive
                plane_offset_covered.append(gt_id)
            plane_offset_apscores[pred_label].append(pred_score)
            plane_offset_aplabels[pred_label].append(tpfp)

    # check things for eval
    # assert npos.sum() == len(dataset.dataset["annotations"])
    # convert to tensors
    detection_metrics = {}
    maskap, planeap, planeNormalAP, planeOffsetAP = 0.0, 0.0, 0.0, 0.0
    valid = 0.0
    for cat_id in cat_ids:
        cat_name = dataset.loadCats([cat_id])[0]["name"]  # 'plane'
        if npos[cat_id] == 0:
            continue  # it means there is no plane in the input dataset
        valid += 1

        cat_mask_ap = VOCap.compute_ap(
            torch.cat(mask_apscores[cat_id]),
            torch.cat(mask_aplabels[cat_id]),
            npos[cat_id],
        ).item()  # mask ap of all predicted planes on the whole dataset
        maskap += cat_mask_ap
        detection_metrics["mask_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_mask_ap

        cat_plane_ap = VOCap.compute_ap(
            torch.cat(plane_apscores[cat_id]),
            torch.cat(plane_aplabels[cat_id]),
            npos[cat_id],
        ).item()
        planeap += cat_plane_ap
        detection_metrics[
            "plane_ap@iou%.1fnormal%.1foffset%.1f - %s"
            % (iou_thresh, normal_threshold, offset_threshold, cat_name)
        ] = cat_plane_ap

        cat_plane_normal_ap = VOCap.compute_ap(
            torch.cat(plane_normal_apscores[cat_id]),
            torch.cat(plane_normal_aplabels[cat_id]),
            npos[cat_id],
        ).item()
        planeNormalAP += cat_plane_normal_ap
        detection_metrics[
            "plane_ap@iou%.1fnormal%.1f - %s"
            % (iou_thresh, normal_threshold, cat_name)
            ] = cat_plane_normal_ap

        cat_plane_offset_ap = VOCap.compute_ap(
            torch.cat(plane_offset_apscores[cat_id]),
            torch.cat(plane_offset_aplabels[cat_id]),
            npos[cat_id],
        ).item()
        planeOffsetAP += cat_plane_offset_ap
        detection_metrics[
            "plane_ap@iou%.1foffset%.1f - %s"
            % (iou_thresh, offset_threshold, cat_name)
            ] = cat_plane_offset_ap

    detection_metrics["mask_ap@%.1f" % iou_thresh] = maskap / valid  # mean ap on all classes
    detection_metrics[
        "plane_ap@iou%.1fnormal%.1foffset%.1f"
        % (iou_thresh, normal_threshold, offset_threshold)
    ] = (planeap / valid)  # mean ap on all classes
    detection_metrics[
        "plane_ap@iou%.1fnormal%.1f"
        % (iou_thresh, normal_threshold)
        ] = (planeNormalAP / valid)  # mean ap on all classes
    detection_metrics[
        "plane_ap@iou%.1foffset%.1f"
        % (iou_thresh, normal_threshold)
        ] = (planeOffsetAP / valid)  # mean ap on all classes

    # logger.info("Detection metrics: \n" + create_small_table(detection_metrics))
    _logger.info("Detection metrics: \n" + create_small_table(detection_metrics))

    plane_metrics = {}
    plane_normal_errs = np.array(plane_normal_errs)
    plane_offset_errs = np.array(plane_offset_errs)
    plane_metrics["%normal<10"] = (
        sum(plane_normal_errs < 10) / len(plane_normal_errs) * 100
    )
    plane_metrics["%normal<30"] = (
        sum(plane_normal_errs < 30) / len(plane_normal_errs) * 100
    )
    plane_metrics["%offset<0.5"] = (
        sum(plane_offset_errs < 0.5) / len(plane_offset_errs) * 100
    )
    plane_metrics["%offset<0.3"] = (
        sum(plane_offset_errs < 0.3) / len(plane_offset_errs) * 100
    )
    plane_metrics["mean_normal"] = plane_normal_errs.mean()
    plane_metrics["median_normal"] = np.median(plane_normal_errs)
    plane_metrics["mean_offset"] = plane_offset_errs.mean()
    plane_metrics["median_offset"] = np.median(plane_offset_errs)
    # logger.info("Plane metrics: \n" + create_small_table(plane_metrics))
    _logger.info("Plane metrics: \n" + create_small_table(plane_metrics))
    plane_metrics.update(detection_metrics)
    return plane_metrics


def evaluate_for_matchings(
    predictions,
    dataset_dict,
    iou_thresh=0.5,
    device=None,
    _logger=None,
):
    match_statistics = {}
    for key in predictions[0]:
        if "assignment" in key:
            match_statistics[key] = {
                "all_correct_num": 0,
                "all_matched_num": 0
            }

    all_gt_num = 0
    for pred in predictions:
        img_id0 = pred['0']['image_id']
        img_id1 = pred['1']['image_id']
        img_pair_id = img_id0 + '__' + img_id1

        gt_input_pair = dataset_dict[img_pair_id]
        gt_corr = gt_input_pair['gt_corrs']  # e.g., [[0, 0], [1, 2], [3, 5], ...]
        all_gt_num += len(gt_corr)

        matched_iou_list = []
        matched_gtidx_list = []
        for img_idx in ['0', '1']:
            # get gt masks
            gt_anns = gt_input_pair[img_idx]['annotations']
            gt_mask_rles = []
            for ann in gt_anns:
                image_height = ann['height']
                image_width = ann['width']
                if isinstance(ann["segmentation"], list):
                    polygons = [np.array(p, dtype=np.float64) for p in ann["segmentation"]]
                    try:
                        rles = mask_util.frPyObjects(polygons, image_height, image_width)
                    except:
                        polygons = []
                        for p in ann["segmentation"]:
                            if len(p) > 4:
                                polygons.append(np.array(p, dtype=np.float64))
                            else:
                                raise ValueError
                    rle = mask_util.merge(rles)
                elif isinstance(ann["segmentation"], dict):  # RLE
                    rle = ann["segmentation"]
                else:
                    raise TypeError(
                        f"Unknown segmentation type {type(ann['segmentation'])}!"
                    )
                gt_mask_rles.append(rle)

            # get predicted masks
            pred_mask_rles = []
            for pred_ins in pred[img_idx]["instances"]:
                pred_mask_rles.append(pred_ins["segmentation"])

            # get mask ious
            miou = mask_util.iou(pred_mask_rles, gt_mask_rles, [0] * len(gt_mask_rles))  # shape: [n_pred, n_gt]
            m_iou_tensor = torch.from_numpy(miou)  # shape: [n_pred, n_gt]

            # assign pred mask to gt mask
            matched_iou, matched_gtidx = m_iou_tensor.max(-1)  # idx of gt plane which has max iou with predicted plane
            matched_iou_list.append(matched_iou)
            matched_gtidx_list.append(matched_gtidx)

        for key in pred:
            if "assignment" in key:
                pred_corr_matrix = pred[key]  # pred_n, pred_n
                pred_corr = torch.nonzero(pred_corr_matrix).reshape(-1, 2).detach().cpu().numpy()  # n, 2
                pred_matched_num = pred_corr.shape[0]
                correct_num = 0
                for i in range(pred_matched_num):
                    m_idxs = pred_corr[i]
                    pred_idx0 = m_idxs[0]
                    pred_idx1 = m_idxs[1]
                    if matched_iou_list[0][pred_idx0] >= 0.5 and matched_iou_list[1][pred_idx1] >= 0.5:
                        gt_idx0 = matched_gtidx_list[0][pred_idx0]
                        gt_idx1 = matched_gtidx_list[1][pred_idx1]
                        if [gt_idx0, gt_idx1] in gt_corr:
                            correct_num += 1

                match_statistics[key]["all_matched_num"] += pred_matched_num
                match_statistics[key]["all_correct_num"] += correct_num

    for key in match_statistics:
        all_correct_num = match_statistics[key]['all_correct_num']
        all_matched_num = match_statistics[key]['all_matched_num']
        precision = float(all_correct_num) / float(all_matched_num)
        recall = float(all_correct_num) / float(all_gt_num)
        F_score = 2 * precision * recall / (precision + recall)

        matching_metrics = {}
        matching_metrics['precision'] = precision
        matching_metrics['recall'] = recall
        matching_metrics['F-score'] = F_score
        matching_metrics['TP'] = all_correct_num
        matching_metrics['Pred. Num.'] = all_matched_num
        matching_metrics['GT Num.'] = all_gt_num
        _logger.info("Plane metrics (%s): \n"%(key) + create_small_table(matching_metrics))

    return matching_metrics


def save_dict(return_dict, folder, prefix=None):
    os.makedirs(folder, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if prefix is None:
        save_path = os.path.join(folder, f"optimized_{timestr}.pkl")
    else:
        save_path = os.path.join(folder, prefix + ".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(return_dict.copy(), f)