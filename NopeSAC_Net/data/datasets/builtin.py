import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from NopeSAC_Net.data.datasets import load_mp3d_json


def get_mp3d_metadata():
    meta = [
        {"name": "plane", "color": [230, 25, 75], "id": 1},  # noqa
    ]
    return meta


SPLITS = {
    "mp3d_val": ("mp3d", "mp3d_planercnn_json/cached_set_val.json"),
    "mp3d_test": ("mp3d", "mp3d_planercnn_json/cached_set_test.json"),
    "mp3d_train": ("mp3d", "mp3d_planercnn_json/cached_set_train.json"),
    "scannet_train": ("scannet", "scannet_json/cached_set_trainV2.json"),
    "scannet_test": ("scannet", "scannet_json/cached_set_testV2.json"),
}

def register_mp3d(dataset_name, json_file, image_root):
    if 'mp3d' in dataset_name:
        root = "./datasets/mp3d_dataset/"
    elif 'scannet' in dataset_name:
        root = "./datasets/scannet_dataset/"
    else:
        raise NotImplementedError

    DatasetCatalog.register(
        dataset_name, lambda: load_mp3d_json(json_file, dataset_name)
    )
    things_ids = [k["id"] for k in get_mp3d_metadata()]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    thing_classes = [k["name"] for k in get_mp3d_metadata()]
    thing_colors = [k["color"] for k in get_mp3d_metadata()]
    json_file = os.path.join(root, json_file)
    image_root = os.path.join(root, image_root)
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="mp3d", **metadata
    )


for key, (data_root, anno_file) in SPLITS.items():
    register_mp3d(key, anno_file, data_root)
