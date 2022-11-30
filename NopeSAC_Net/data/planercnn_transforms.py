import copy
import numpy as np
import os
import torch
import pickle
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    polygons_to_bitmask,
)
import pycocotools.mask as mask_util
from PIL import Image
import torchvision.transforms as transforms
from . import GaussianBlur
import cv2

__all__ = ["PlaneRCNNMapper"]


def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def annotations_to_instances(
    annos, image_size, mask_format="polygon", max_num_planes=20
):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Args:
        annos (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width
    Returns:
        Instances: It will contains fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """
    boxes = [
        BoxMode.convert(obj["bbox"], BoxMode(obj["bbox_mode"]), BoxMode.XYXY_ABS)
        for obj in annos
    ]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except:
                segms = []
                for obj in annos:
                    seg_new = []
                    for seg in obj["segmentation"]:
                        if len(seg) > 4:
                            seg_new.append(seg)
                    segms.append(seg_new)
                masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert (
                        segm.ndim == 2
                    ), "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "plane" in annos[0]:
        plane = [torch.tensor(obj["plane"]) for obj in annos]
        plane_idx = [torch.tensor([i]) for i in range(len(plane))]
        target.gt_planes = torch.stack(plane, dim=0)
        target.gt_plane_idx = torch.stack(plane_idx, dim=0)
    return target


class PlaneRCNNMapper:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.

    Note that for our existing models, mean/std normalization is done by the model instead of here.
    """

    def __init__(self, cfg, is_train=True, dataset_names=None):
        self.cfg = cfg
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.depth_on       = cfg.MODEL.DEPTH_ON
        self.camera_on      = cfg.MODEL.CAMERA_ON
        self.mask_on = True
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self._eval_gt_box = cfg.TEST.EVAL_GT_BOX
        self._augmentation = cfg.DATALOADER.AUGMENTATION
        self.root_dir = cfg.DATASETS.ROOT_DIR
        self.dataset_names = dataset_names
        assert dataset_names is not None

        self.use_mp3d = False
        self.use_scannet = False
        for data_name in self.dataset_names:
            if 'mp3d' in data_name:
                self.use_mp3d = True
            if 'scannet' in data_name:
                self.use_scannet = True

        assert (self.use_scannet & self.use_mp3d) is False

        # fmt: on
        if self.load_proposals:
            raise ValueError("Loading proposals not yet supported")

        self.is_train = is_train

        if self.camera_on:
            kmeans_trans_path = cfg.MODEL.CAMERA_HEAD.KMEANS_TRANS_PATH
            kmeans_rots_path = cfg.MODEL.CAMERA_HEAD.KMEANS_ROTS_PATH
            assert os.path.exists(kmeans_trans_path)
            assert os.path.exists(kmeans_rots_path)
            with open(kmeans_trans_path, "rb") as f:
                self.kmeans_trans = pickle.load(f)
            with open(kmeans_rots_path, "rb") as f:
                self.kmeans_rots = pickle.load(f)

        if self._augmentation:
            color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            augmentation = [
                transforms.RandomApply([color_jitter], p=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
            ]
            self.img_transform = transforms.Compose(augmentation)

        self.precompute_xy_map()

    def precompute_xy_map(self, h=480, w=640):
        xy_map = np.zeros((2, h, w)).astype(np.float32)
        for y in range(h):
            for x in range(w):
                xy_map[0, y, x] = float(x) / w
                xy_map[1, y, x] = float(y) / h
        self.normalized_xy_map = xy_map

    def __call__(self, dataset_dict):
        if self.use_mp3d:
            return self.call_mp3d(dataset_dict)
        else:
            return self.call_scannet(dataset_dict)


    def call_mp3d(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        for i in range(2):
            dataset_dict[str(i)]["file_name"] = dataset_dict[str(i)]["file_name"].replace(
                '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20/', self.root_dir)
            image = utils.read_image(
                dataset_dict[str(i)]["file_name"], format=self.img_format
            )
            utils.check_image_size(dataset_dict[str(i)], image)
            if self.is_train and self._augmentation:
                image = Image.fromarray(image)
                dataset_dict[str(i)]["image"] = self.img_transform(image) * 255.0
                image_shape = dataset_dict[str(i)]["image"].shape[1:]
            else:
                image_shape = image.shape[:2]
                dataset_dict[str(i)]["image"] = torch.as_tensor(
                    image.transpose(2, 0, 1).astype("float32")
                )
            # Can use uint8 if it turns out to be slow some day
            if self.mask_on:
                # load mask maps
                house, img_id = dataset_dict[str(i)]["image_id"].split("_", 1)
                masks_path = os.path.join(
                    self.root_dir,
                    "observations",
                    house,
                    img_id + ".pkl",
                )
                with open(masks_path, "rb") as f:
                    obs = pickle.load(f)
                # This assertion is to check dataset is clean
                # assert((obs['color_sensor'][:,:,:3][:,:,::-1].transpose(2, 0, 1)-dataset_dict[str(i)]["image"].numpy()).sum()==0)
                semantic_map = obs["semantic_sensor"]
                dataset_dict[str(i)]["semantic_map"] = torch.as_tensor(
                    semantic_map.astype("int32")
                )

            if self.mask_on:
                # load depth map
                house, img_id = dataset_dict[str(i)]["image_id"].split("_", 1)
                depth_path = os.path.join(
                    self.root_dir,
                    "observations",
                    house,
                    img_id + ".pkl",
                )
                with open(depth_path, "rb") as f:
                    obs = pickle.load(f)
                # This assertion is to check dataset is clean
                # assert((obs['color_sensor'][:,:,:3][:,:,::-1].transpose(2, 0, 1)-dataset_dict[str(i)]["image"].numpy()).sum()==0)
                depth = obs["depth_sensor"]
                dataset_dict[str(i)]["depth"] = torch.as_tensor(
                    depth.astype("float32")
                )

            if self.camera_on:
                relative_pose = dataset_dict["rel_pose"]
                x, y, z = relative_pose["position"]
                w, xi, yi, zi = relative_pose["rotation"]
                dataset_dict["rel_pose"]["tran_cls"] = torch.LongTensor(
                    self.xyz2class(x, y, z)
                )
                dataset_dict["rel_pose"]["rot_cls"] = torch.LongTensor(
                    self.quat2class(w, xi, yi, zi)
                )

        if not self.is_train and not self._eval_gt_box:
            return dataset_dict

        if not self._eval_gt_box:
            for i in range(2):
                if "annotations" in dataset_dict[str(i)]:
                    annos = [
                        self.transform_annotations(obj)
                        for obj in dataset_dict[str(i)].pop("annotations")
                        if obj.get("iscrowd", 0) == 0
                    ]
                    # Should not be empty during training
                    instances = annotations_to_instances(annos, image_shape)
                    dataset_dict[str(i)]["instances"] = instances[
                        instances.gt_boxes.nonempty()
                    ]
        else:
            for i in range(2):
                if "annotations" in dataset_dict[str(i)]:
                    annos = [
                        self.transform_annotations(obj)
                        for obj in dataset_dict[str(i)]["annotations"]
                        if obj.get("iscrowd", 0) == 0
                    ]
                    # Should not be empty during training
                    instances = annotations_to_instances(annos, image_shape)
                    dataset_dict[str(i)]["instances"] = instances[
                        instances.gt_boxes.nonempty()
                    ]

        return dataset_dict

    def call_scannet(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        for i in range(2):
            image = utils.read_image(
                dataset_dict[str(i)]["file_name"], format=self.img_format
            )
            image = cv2.resize(image, (640, 480))
            utils.check_image_size(dataset_dict[str(i)], image)
            if self.is_train and self._augmentation:
                image = Image.fromarray(image)
                dataset_dict[str(i)]["image"] = self.img_transform(image) * 255.0
                image_shape = dataset_dict[str(i)]["image"].shape[1:]
            else:
                image_shape = image.shape[:2]
                dataset_dict[str(i)]["image"] = torch.as_tensor(
                    image.transpose(2, 0, 1).astype("float32")
                )

            # Can use uint8 if it turns out to be slow some day
            if True:
                # load mask maps
                image_id = dataset_dict[str(i)]["image_id"]
                scene_idx, image_idx = image_id.split('-')
                masks_path = os.path.join(
                    self.root_dir,
                    "twoView_Anns",
                    scene_idx,
                    image_idx + ".pkl",
                )
                with open(masks_path, "rb") as f:
                    obs = pickle.load(f)

                plane_masks = obs['plane_masks']

                plane_num = len(plane_masks)
                assert plane_num == dataset_dict[str(i)]["gt_plane_num"]

                dataset_dict[str(i)]["plane_masks"] = plane_masks
                camera_K = obs['camera_K']  # 3, 3
                dataset_dict[str(i)]["camera_K"] = camera_K

            if True:
                # load depth map
                depth_path = dataset_dict[str(i)]["file_name"]
                depth_path = depth_path.replace('color', 'depth').replace('.jpg', '.png')
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
                dataset_dict[str(i)]["depth"] = torch.as_tensor(
                    depth.astype("float32")
                )

            if self.camera_on:
                relative_pose = dataset_dict["rel_pose"]
                x, y, z = relative_pose["position"]
                w, xi, yi, zi = relative_pose["rotation"]
                dataset_dict["rel_pose"]["tran_cls"] = torch.LongTensor(
                    self.xyz2class(x, y, z)
                )
                dataset_dict["rel_pose"]["rot_cls"] = torch.LongTensor(
                    self.quat2class(w, xi, yi, zi)
                )

        if not self.is_train and not self._eval_gt_box:
            return dataset_dict

        if not self._eval_gt_box:
            for i in range(2):
                if "annotations" in dataset_dict[str(i)]:
                    annos = [
                        self.transform_annotations(obj)
                        for obj in dataset_dict[str(i)].pop("annotations")
                        if obj.get("iscrowd", 0) == 0
                    ]
                    # Should not be empty during training
                    instances = annotations_to_instances(annos, image_shape)
                    dataset_dict[str(i)]["instances"] = instances[
                        instances.gt_boxes.nonempty()
                    ]
        else:
            for i in range(2):
                if "annotations" in dataset_dict[str(i)]:
                    annos = [
                        self.transform_annotations(obj)
                        for obj in dataset_dict[str(i)]["annotations"]
                        if obj.get("iscrowd", 0) == 0
                    ]
                    # Should not be empty during training
                    instances = annotations_to_instances(annos, image_shape)
                    dataset_dict[str(i)]["instances"] = instances[
                        instances.gt_boxes.nonempty()
                    ]

        return dataset_dict


    def transform_annotations(self, annotation, transforms=None, image_size=None):
        """
        Apply image transformations to the annotations.
        After this method, the box mode will be set to XYXY_ABS.
        """
        return annotation

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
