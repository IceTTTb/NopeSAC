import numpy as np
import argparse, os, cv2, torch, pickle, quaternion

import pycocotools.mask as mask_util

from scipy.linalg import eigh
from scipy.ndimage.measurements import center_of_mass

from detectron2.utils.visualizer import Visualizer
from pytorch3d.structures import join_meshes_as_batch

from NopeSAC_Net.utils.mesh_utils import (
    save_obj,
    get_camera_meshes,
    transform_meshes,
    rotate_mesh_for_webview,
    get_plane_params_in_global,
    get_plane_params_in_local,
)
from NopeSAC_Net.utils.vis import get_single_image_mesh_plane
from NopeSAC_Net.visualization import create_instances, get_labeled_seg, draw_match

from detectron2.evaluation.coco_evaluation import instances_to_coco_json


def process(output, input, show_gt=False):
    _cpu_device = "cpu"


    if show_gt:
        # raise
        prediction = {"0": {}, "1": {}}
        for i in range(2):
            # instances_pred = output[str(i)]["instances"]
            gt_anns = input[str(i)]['annotations']
            instances = []
            planes = []
            for ann in gt_anns:
                image_height = ann['height']
                image_width = ann['width']

                if isinstance(ann["segmentation"], list):
                    polygons = [np.array(p, dtype=np.float64) for p in ann["segmentation"]]
                    rles = mask_util.frPyObjects(polygons, image_height, image_width)
                    rle = mask_util.merge(rles)
                elif isinstance(ann["segmentation"], dict):  # RLE
                    rle = ann["segmentation"]
                else:
                    raise TypeError(
                        f"Unknown segmentation type {type(ann['segmentation'])}!"
                    )
                ins = {"image_id": ann['image_id'],
                       "file_name": input[str(i)]['file_name'],
                       'category_id': ann['category_id'],
                       'score': 1.0,
                       'segmentation': rle}
                instances.append(ins)
                planes.append(ann["plane"])

            prediction[str(i)]["instances"] = instances
            prediction[str(i)]["pred_plane"] = torch.tensor(planes)

        gt_corr = input['gt_corrs']
        gt_corr = np.array(gt_corr)  # n, 2
        idxs0 = gt_corr[:, 0]
        idxs1 = gt_corr[:, 1]
        pn0 = len(prediction['0']["instances"])
        pn1 = len(prediction['1']["instances"])
        ass_matrix = np.zeros((pn0, pn1))
        ass_matrix[idxs0, idxs1] = 1
        prediction["corrs"] = torch.from_numpy(ass_matrix)

        camera_dict = {
            "pred": {
                "tran": np.array(input["rel_pose"]["position"]),
                "rot": np.array(input["rel_pose"]["rotation"])
            },
            "gt": {
                "tran": np.array(input["rel_pose"]["position"]),
                "rot": np.array(input["rel_pose"]["rotation"])
            }
        }
        prediction["camera"] = camera_dict
    else:
        prediction = {"0": {}, "1": {}}
        for i in range(2):
            if "instances" in output[str(i)]:
                instances = output[str(i)]["instances"]
                prediction[str(i)]["instances"] = instances
                prediction[str(i)]["pred_plane"] = output[str(i)]["pred_plane"].to(_cpu_device)

            if output["depth"][str(i)] is not None:
                prediction[str(i)]["pred_depth"] = output["depth"][str(i)].to(_cpu_device)
        if "pred_assignment" in output:
            pred_ass = output["pred_assignment"].to(_cpu_device)
            prediction["corrs"] = pred_ass

        camera_dict = {
            "pred": {
                "tran": output["camera"]["tran"],
                "rot": output["camera"]["rot"],
            },
            "gt": {
                "tran": np.array(input["rel_pose"]["position"]),
                "rot": np.array(input["rel_pose"]["rotation"])
            }
        }
        prediction["camera"] = camera_dict

    return prediction


def save_matching(
    img_file1,
    img_file2,
    pred_dict,
    assignment,
    output_dir,
    prefix="",
    paper_img=False,
    score_threshold=0.7,
):
    """
    fp: whether show fp or fn
    gt_box: whether use gtbox
    """
    image_paths = {"0": img_file1, "1": img_file2}
    blended = {}
    # centroids for matching
    centroids = {"0": [], "1": []}

    idxs1, idxs2 = np.where(assignment > 0)
    matched_num = idxs1.shape[0]
    idxs_all = [idxs1, idxs2]

    for i in range(2):
        img = cv2.imread(image_paths[str(i)], cv2.IMREAD_COLOR)[:, :, ::-1]
        img = cv2.resize(img, (640, 480))
        height, width, _ = img.shape
        vis = Visualizer(img)

        plane = pred_dict[str(i)]["pred_plane"].numpy()  # n, 3
        ins = pred_dict[str(i)]["instances"]
        ins_new = []
        plane_new = np.zeros_like(plane)

        for pi in range(matched_num):
            ins_new.append(ins[idxs_all[i][pi]])
            plane_new[pi] = plane[idxs_all[i][pi]]

        idx_temp = 0
        for ri in range(len(ins)):
            if ri in idxs_all[i]:
                continue
            ins_new.append(ins[ri])
            plane_new[matched_num + idx_temp] = plane[ri]
            idx_temp += 1

        assert matched_num + idx_temp == len(ins)

        p_instance = create_instances(
            pred_dict[str(i)]["instances"],
            img.shape[:2],
            pred_planes=pred_dict[str(i)]["pred_plane"].numpy(),
            conf_threshold=score_threshold,
        )

        p_instance_align = create_instances(
            ins_new,
            img.shape[:2],
            pred_planes=plane_new,
            conf_threshold=0.5,
        )  # <class 'detectron2.structures.instances.Instances'>

        seg_blended = get_labeled_seg(
            p_instance_align, score_threshold, vis, paper_img=paper_img
        )
        blended[str(i)] = seg_blended
        # centroid of mask
        for ann in pred_dict[str(i)]["instances"]:
            M = center_of_mass(mask_util.decode(ann["segmentation"]))
            centroids[str(i)].append(M[::-1])  # reverse for opencv
        centroids[str(i)] = np.array(centroids[str(i)])

    pred_corr_list = np.array(torch.FloatTensor(assignment).nonzero().tolist())

    correct_list_pred = [True for pair in pred_corr_list]
    pred_matching_fig = draw_match(
        blended["0"],
        blended["1"],
        centroids["0"],
        centroids["1"],
        np.array(pred_corr_list),
        correct_list_pred,
        vertical=False,
    )
    os.makedirs(output_dir, exist_ok=True)
    pred_matching_fig.save(os.path.join(output_dir, prefix + ".png"))


def merge_plane_params_from_local_params(plane_locals, corr_list, camera_pose):
    """
    input: plane parameters in camera frame
    output: merged plane parameters using corr_list
    """
    param1, param2 = plane_locals["0"], plane_locals["1"]
    param1_global = get_plane_params_in_global(param1, camera_pose)
    param2_global = get_plane_params_in_global(
        param2, {"position": np.array([0, 0, 0]), "rotation": np.quaternion(1, 0, 0, 0)}
    )
    param1_global, param2_global = merge_plane_params_from_global_params(
        param1_global, param2_global, corr_list
    )
    param1 = get_plane_params_in_local(param1_global, camera_pose)
    param2 = get_plane_params_in_local(
        param2_global,
        {"position": np.array([0, 0, 0]), "rotation": np.quaternion(1, 0, 0, 0)},
    )
    return {"0": param1, "1": param2}


def merge_plane_params_from_global_params(param1, param2, corr_list):
    """
    input: plane parameters in global frame
    output: merged plane parameters using corr_list
    """
    pred = {"0": {}, "1": {}}
    pred["0"]["offset"] = np.maximum(
        np.linalg.norm(param1, ord=2, axis=1), 1e-5
    ).reshape(-1, 1)
    pred["0"]["normal"] = param1 / pred["0"]["offset"]
    pred["1"]["offset"] = np.maximum(
        np.linalg.norm(param2, ord=2, axis=1), 1e-5
    ).reshape(-1, 1)
    pred["1"]["normal"] = param2 / pred["1"]["offset"]
    for ann_id in corr_list:
        # average normal
        normal_pair = np.vstack(
            (pred["0"]["normal"][ann_id[0]], pred["1"]["normal"][ann_id[1]])
        )
        w, v = eigh(normal_pair.T @ normal_pair)
        avg_normals = v[:, np.argmax(w)]
        if (avg_normals @ normal_pair.T).sum() < 0:
            avg_normals = -avg_normals
        # average offset
        avg_offset = (
            pred["0"]["offset"][ann_id[0]] + pred["1"]["offset"][ann_id[1]]
        ) / 2
        avg_plane = avg_normals * avg_offset
        param1[ann_id[0]] = avg_plane
        param2[ann_id[1]] = avg_plane
    return param1, param2


def save_pair_objects(
    img_file1,
    img_file2,
    p_instances,
    output_dir,
    prefix="",
    pred_camera=None,
    plane_param_override=None,
    show_camera=True,
    corr_list=[],
    webvis=False,
    save_mesh=True
):
    """
    if tran_topk == -2 and rot_topk == -2, then pred_camera should not be None, this is used for non-binned camera.
    if exclude is not None, exclude some instances to make fig 2.
    idx=7867
    exclude = {
        '0': [2,3,4,5,6,7],
        '1': [0,1,2,4,5,6,7],
    }
    """
    image_paths = {"0": img_file1, "1": img_file2}
    meshes_list = []
    # map_files = []
    uv_maps = []
    cam_list = []
    # get plane parameters
    plane_locals = {}
    for i in range(2):
        if plane_param_override is None:
            plane_locals[str(i)] = p_instances[str(i)].pred_planes
        else:
            plane_locals[str(i)] = plane_param_override[str(i)]
    # get camera 1 to 2
    camera1to2 = {
        "position": np.array(pred_camera["position"]),
        "rotation": quaternion.from_float_array(pred_camera["rotation"]),
    }

    # Merge planes if they are in correspondence
    if len(corr_list) != 0:
        plane_locals = merge_plane_params_from_local_params(
            plane_locals, corr_list, camera1to2
        )

    os.makedirs(output_dir, exist_ok=True)
    for i in range(2):
        if i == 0:
            camera_info = camera1to2
        else:
            camera_info = {
                "position": np.array([0, 0, 0]),
                "rotation": np.quaternion(1, 0, 0, 0),
            }
        p_instance = p_instances[str(i)]
        plane_params = plane_locals[str(i)]
        segmentations = p_instance.pred_masks
        meshes, uv_map = get_single_image_mesh_plane(
            plane_params,
            segmentations,
            img_file=image_paths[str(i)],
            height=480,
            width=640,
            webvis=False,
        )
        uv_maps.extend(uv_map)
        meshes = transform_meshes(meshes, camera_info)
        meshes_list.append(meshes)
        cam_list.append(camera_info)

    joint_mesh = join_meshes_as_batch(meshes_list)
    if webvis:
        joint_mesh = rotate_mesh_for_webview(joint_mesh)

    # add camera into the mesh
    if show_camera:
        cam_meshes = get_camera_meshes(cam_list)
        if webvis:
            cam_meshes = rotate_mesh_for_webview(cam_meshes)
    else:
        cam_meshes = None

    # save obj
    if len(prefix) == 0:
        prefix = "pred"
    save_obj(
        folder=output_dir,
        prefix=prefix,
        meshes=joint_mesh,
        cam_meshes=cam_meshes,
        decimal_places=10,
        blend_flag=True,
        map_files=None,
        uv_maps=uv_maps,
        save_mesh=save_mesh
    )
