import torch
import pickle
import pycocotools.mask as mask_util
import numpy as np
import cv2
import os
import quaternion
from pytorch3d.structures import join_meshes_as_batch
from NopeSAC_Net.data import PlaneRCNNMapper
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage.measurements import center_of_mass

from detectron2.utils.visualizer import Visualizer
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


from tqdm import tqdm
import argparse

def load_predictions(predictions_file_name, opt_dict_file_name):
    predictions = torch.load(predictions_file_name)
    with open(opt_dict_file_name, "rb") as f:
        optimized_dict = pickle.load(f)
    return predictions, optimized_dict

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
    # import pdb; pdb.set_trace()
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
        factor=1
    )
    os.makedirs(output_dir, exist_ok=True)
    pred_matching_fig.save(os.path.join(output_dir, prefix + ".png"))

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
    save_mesh=True,
    camera_K=-1
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
            camera_K=camera_K
        )
        uv_maps.extend(uv_map)
        meshes = transform_meshes(meshes, camera_info)
        meshes_list.append(meshes)
        cam_list.append(camera_info)

    joint_mesh = join_meshes_as_batch(meshes_list)
    # import pdb;pdb.set_trace()
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


def load_input_dataset(dataset):
    dataset_dict = {}
    dataset_list = list(DatasetCatalog.get(dataset))
    i = 0
    for dic in tqdm(dataset_list):
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
        if 'mp3d' in dataset:
            dic['0']["file_name"] = dic['0']["file_name"].replace(
                '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20/',
                'datasets/mp3d_dataset/')
            dic['1']["file_name"] = dic['1']["file_name"].replace(
                '/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20/',
                'datasets/mp3d_dataset/')
            dic['camera_K'] = None
        else:
            scene_id, sceneimgidx = dic['0']['image_id'].split('-')
            camera_K_path = os.path.join('datasets/scannet_dataset/twoView_Anns/', scene_id, sceneimgidx + '.pkl')

            dic['camera_K'] = None
            if True:
                with open(camera_K_path, "rb") as f:
                    obs = pickle.load(f)
                cam_K = np.array(obs['camera_K'])  # 3, 3
                f.close()
                assert cam_K is not None
                dic['camera_K'] = cam_K

        dataset_dict[key] = dic
        i = i +1

    return dataset_dict

def angle_error_vec(v1, v2):
    v1 = v1.reshape(1, 4)
    v2 = v2.reshape(1, 4)
    return 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(v1, v2), axis=1)), -1.0, 1.0)) * 180 / np.pi

def vis(input, output_dir, camera_K, opt_dict=None, gt_on=True, merge_on=True, save_match=True, show_camera=True, save_mesh=True, online=False, prefix='', pIdx=0):
    prediction = {"0": {}, "1": {}}
    if gt_on:
        for i in range(2):
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
        camera_dict = {
            "pred": {
                "tran": np.array(input["rel_pose"]["position"]),
                "rot": np.array(input["rel_pose"]["rotation"])
            },
        }
        # prediction["camera"] = camera_dict
        # prediction["corrs"] = torch.from_numpy(ass_matrix)
        # best_ass = prediction["corrs"].numpy().astype(np.int32)
        best_ass = ass_matrix.astype(np.int32)
        if not merge_on:
            best_ass = best_ass * 0
        optimized_dict = {
            "best_assignment": best_ass,
            "best_camera": {"position": camera_dict["pred"]["tran"],
                            "rotation": camera_dict["pred"]["rot"]},
            "plane_param_override": {"0": prediction['0']["pred_plane"].numpy(),
                                     "1": prediction['1']["pred_plane"].numpy()},
        }
    else:
        assert opt_dict is not None
        prediction = {"0": {}, "1": {}}
        for i in range(2):
            if "instances" in input[str(i)]:
                instances = input[str(i)]["instances"]
                prediction[str(i)]["instances"] = instances
                prediction[str(i)]["pred_plane"] = input[str(i)]["pred_plane"]
            if "depth" in input and input["depth"][str(i)] is not None:
                prediction[str(i)]["pred_depth"] = input["depth"][str(i)]
        optimized_dict = opt_dict

    im0 = input['0']['file_name'].replace(
                        "/data/datasets/tanbin/planceRCNN_data/data/ScanNet/",
                        "/remote-home/share/datasets/ScanNetV2_Plane/ScanNet/")
    im1 = input['1']['file_name'].replace(
                        "/data/datasets/tanbin/planceRCNN_data/data/ScanNet/",
                        "/remote-home/share/datasets/ScanNetV2_Plane/ScanNet/")

    image_paths = {"0": im0, "1": im1}

    p_instances = {}
    idxs1, idxs2 = np.where(optimized_dict['best_assignment'] > 0)
    matched_num = idxs1.shape[0]
    idxs_all = [idxs1, idxs2]
    seg_blends = []

    os.makedirs(os.path.join(output_dir), exist_ok=True)

    for i in range(2):
        # import pdb; pdb.set_trace()
        img = cv2.imread(image_paths[str(i)], cv2.IMREAD_COLOR)
        try:
            img = cv2.resize(img, (640, 480))
        except:
            import pdb; pdb.set_trace()
        vis = Visualizer(img)
        plane = prediction[str(i)]["pred_plane"].numpy()  # n, 3
        ins = prediction[str(i)]["instances"]
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
            plane_new[matched_num+idx_temp] = plane[ri]
            idx_temp += 1
        assert matched_num + idx_temp == len(ins)

        p_instance = create_instances(
            prediction[str(i)]["instances"],
            img.shape[:2],
            pred_planes=prediction[str(i)]["pred_plane"].numpy(),
            conf_threshold=0.5,
        )  # <class 'detectron2.structures.instances.Instances'>

        p_instance_align = create_instances(
            ins_new,
            img.shape[:2],
            pred_planes=plane_new,
            conf_threshold=0.5,
        )  # <class 'detectron2.structures.instances.Instances'>

        p_instances[str(i)] = p_instance
        seg_blended = get_labeled_seg(p_instance_align, 0.5, vis, paper_img=True)
        if not online and save_mesh:
            os.makedirs(os.path.join(output_dir), exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"{pIdx}view{i}_blended.jpg"), seg_blended)
            cv2.imwrite(os.path.join(output_dir, f"{pIdx}view{i}_rgb.jpg"), img)

        seg_blends.append(seg_blended)

    if online:
        plt.figure(figsize=(8, 3))
        plt.axis('off')

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(seg_blends[0])

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(seg_blends[1])

        plt.show()
        plt.close()

        return

    # visualize
    if save_match:
        save_matching(
            im0,
            im1,
            prediction,
            optimized_dict["best_assignment"],
            output_dir,
            prefix="%dcorr"%(pIdx),
            paper_img=True,
        )


    if save_mesh or show_camera:
        # save original image (resized)
        cv2.imwrite(os.path.join(output_dir, "%dview0.jpg" % (pIdx)), cv2.resize(cv2.imread(im0), (640, 480)))
        cv2.imwrite(os.path.join(output_dir, "%dview1.jpg" % (pIdx)), cv2.resize(cv2.imread(im1), (640, 480)))

        # save 3D planes
        save_pair_objects(
            os.path.join(output_dir, "%dview0.jpg"%(pIdx)),
            os.path.join(output_dir, "%dview1.jpg"%(pIdx)),
            p_instances,
            os.path.join(output_dir),
            prefix="refined%s"%(prefix),
            pred_camera=optimized_dict["best_camera"],
            plane_param_override=optimized_dict["plane_param_override"],
            show_camera=show_camera,
            corr_list=np.argwhere(optimized_dict["best_assignment"]),
            webvis=False,
            save_mesh=save_mesh,
            camera_K=camera_K
        )


def vis_3DPlanes(GTs, pred_NopeSACs, opt_NopeSACs, root_dir, final_mesh_on=True):
    i = 0
    for key, gt in GTs.items():
        print("saving results of image pair %d"%(i))
        pred_NopeSAC = pred_NopeSACs[i]
        opt_NopeSAC = opt_NopeSACs[i]
        if not final_mesh_on:
            pred_NopeSAC['pred_assignment'] = pred_NopeSAC['pred_assignment_beforeRef0']
            opt_NopeSAC['best_assignment'] = pred_NopeSAC['pred_assignment_beforeRef0'].numpy()

        # check key
        key_NopeSAC = pred_NopeSAC['0']['image_id'] + "__" + pred_NopeSAC['1']['image_id']
        assert key == key_NopeSAC
        camera_K = gt['camera_K']

        if 'camera_onePP' not in pred_NopeSAC:
            i = i + 1
            continue

        onePP_trans = pred_NopeSAC['camera_onePP']['pred']['tran']  # m+1, 3
        onePP_rots = pred_NopeSAC['camera_onePP']['pred']['rot']  # m+1, 4
        onePPnum = onePP_rots.shape[0]

        out_dir_NopeSAC = os.path.join(root_dir, "matchers")
        vis(pred_NopeSAC, out_dir_NopeSAC, camera_K=camera_K, opt_dict=opt_NopeSAC, gt_on=False, online=False,
            save_mesh=False, show_camera=False, pIdx=i)

        # vis gt
        out_dir_gt = os.path.join(root_dir, "%d" % (i), "GT")
        vis(gt, out_dir_gt, camera_K, gt_on=True, show_camera=False, prefix='GT', pIdx=i)
        vis(gt, out_dir_gt + 'Cam', camera_K, gt_on=True, save_mesh=False, save_match=False, prefix='GTCam', pIdx=i)

        # vis NopeSAC mesh
        out_dir_NopeSAC = os.path.join(root_dir, "%d" % (i), "NopeSAC")
        vis(pred_NopeSAC, out_dir_NopeSAC, camera_K=camera_K, opt_dict=opt_NopeSAC, gt_on=False,
            show_camera=False, prefix='Final', pIdx=i)

        # vis refined cam
        vis(pred_NopeSAC, out_dir_NopeSAC + 'Cam_final', camera_K=camera_K, opt_dict=opt_NopeSAC, gt_on=False,
            save_mesh=False, save_match=False, prefix='CamFinal', pIdx=i)

        # vis onePP cam (including initial camera)
        for pi in range(onePPnum):
            cam_pi = {"position": onePP_trans[pi],
                      "rotation": onePP_rots[pi]}
            opt_NopeSAC_camPi = opt_NopeSAC.copy()
            opt_NopeSAC_camPi['best_camera'] = cam_pi
            vis(pred_NopeSAC, out_dir_NopeSAC + 'Cam_onePP%d' % (pi), camera_K=camera_K, opt_dict=opt_NopeSAC_camPi,
                gt_on=False, save_mesh=False, save_match=False, prefix='Cam_onePP%d' % (pi), pIdx=i)
        i = i + 1

        import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--dataset", default='mp3d_test', help="dataset name")
    # parser.add_argument("--NopeSAC_pred", required=True, help="path to instances_predictions.pth")
    # parser.add_argument("--NopeSAC_opt", required=True, help="path to continuous.pkl")
    args = parser.parse_args()

    file_paths = {
        'mp3d_test': {
            "NopeSAC_pred": "results/mp3d_testSet/NopeSAC_instances_predictions.pth",
            "NopeSAC_opt": "results/mp3d_testSet/continuous.pkl",
            },
        'scannet_test': {
            "NopeSAC_pred": "results/scannet_testSet/NopeSAC_instances_predictions.pth",
            "NopeSAC_opt": "results/scannet_testSet/continuous.pkl",
            }
    }
    dataset = args.dataset

    # load gts
    print("loading GT.....")
    inputs = load_input_dataset(dataset)
    out_dir = os.path.join('vis_res', dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("loading predictions")
    pred_NopeSAC, opt_NopeSAC = load_predictions(file_paths[dataset]['NopeSAC_pred'], file_paths[dataset]['NopeSAC_opt'])

    vis_3DPlanes(inputs, pred_NopeSAC, opt_NopeSAC, out_dir, final_mesh_on=False)



