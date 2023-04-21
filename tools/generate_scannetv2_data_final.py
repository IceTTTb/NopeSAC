import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from detectron2.data import detection_utils as utils
import torch
import json
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    polygons_to_bitmask,
)
import pycocotools.mask as mask_util
import quaternion
import pickle
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = torch.arange(640, dtype=torch.float32).view(1, 640) / 640 * 640
y = torch.arange(480, dtype=torch.float32).view(480, 1) / 480 * 480
x = x.cuda()
y = y.cuda()
xx = x.repeat(480, 1)
yy = y.repeat(1, 640)
xy1 = torch.stack((xx, yy, torch.ones((480, 640), dtype=torch.float32).cuda()))  # (3, h, w)
xy1 = xy1.view(3, -1)  # (3, h*w)

def angle_error_qua(v1, v2):
    return 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(v1, v2), axis=1)), -1.0, 1.0)) * 180 / np.pi

## The function to compute plane depths from plane parameters
def calcPlaneDepths(planes, width, height, camera, max_depth=10):
    fx = camera[0]
    fy = camera[1]
    offset_x = camera[2]
    offset_y = camera[3]
    K = [[fx, 0, offset_x],
         [0, fy, offset_y],
         [0, 0, 1]]
    K = np.array(K).reshape(3, 3)
    K = torch.from_numpy(K).cuda().float()
    K_inv = K.inverse()
    k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w)
    planes = torch.from_numpy(planes).cuda().float()
    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)  # n, 1
    planeNormals = planes / (planeOffsets + 1e-10)  # n, 3
    planeNormals = torch.cat([planeNormals[:, 0:1], -planeNormals[:, 2:3], planeNormals[:, 1:2]], dim=-1)
    normalXYZ = torch.matmul(planeNormals, k_inv_dot_xy1)  # 3, hw
    normalXYZ = normalXYZ.permute(1, 0).reshape(h, w, -1)
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = torch.clamp(planeDepths, min=0, max=max_depth)
        pass

    return planeDepths

## Clean segmentation
def cleanSegmentation(image, planes, plane_info, segmentation, depth, camera, planeAreaThreshold=200,
                      planeWidthThreshold=10, depthDiffThreshold=0.1, validAreaThreshold=0.5, brightThreshold=20,
                      confident_labels={}, return_plane_depths=False):
    planeDepths_ = calcPlaneDepths(planes, segmentation.shape[1], segmentation.shape[0], camera).permute((2, 0, 1))

    # ------------------------------------------------------------------------
    seg_idx_list = np.unique(segmentation)
    planeDepths = planeDepths_
    segmentation = torch.from_numpy(segmentation).cuda()
    newSegmentation = -torch.ones_like(segmentation)
    image = torch.from_numpy(image).cuda().float()
    depth = torch.from_numpy(depth).cuda()
    validMask = (torch.norm(image, dim=-1, p=2) > brightThreshold) & (depth > 1e-4)
    depthDiffMask = (torch.abs(planeDepths - depth) < depthDiffThreshold) | (depth < 1e-4)
    for segmentIndex in seg_idx_list:
        if segmentIndex < 0:
            continue
        segmentMask = segmentation == segmentIndex
        try:
            plane_info[segmentIndex][0][1]
        except:
            print('invalid plane info')
            print(plane_info)
            print(len(plane_info), len(planes), segmentation.min(), segmentation.max())
            print(segmentIndex)
            print(plane_info[segmentIndex])
            exit(1)
        if plane_info[segmentIndex][0][1] in confident_labels:
            if segmentMask.sum() > planeAreaThreshold:
                newSegmentation[segmentMask] = segmentIndex
                pass
            continue
        oriArea = segmentMask.sum()
        segmentMask = segmentMask & (depthDiffMask[segmentIndex])
        newArea = (segmentMask & validMask).sum()
        if newArea < oriArea * validAreaThreshold:
            continue
        segmentMask = segmentMask.cpu().numpy().astype(np.uint8)
        segmentMask = cv2.dilate(segmentMask, np.ones((3, 3)))
        numLabels, components = cv2.connectedComponents(segmentMask)
        for label in range(1, numLabels):
            mask = components == label
            ys, xs = mask.nonzero()
            area = float(len(xs))
            if area < planeAreaThreshold * 2.:
                continue
            size_y = ys.max() - ys.min() + 1
            size_x = xs.max() - xs.min() + 1
            length = np.linalg.norm([size_x, size_y])
            if area / length < planeWidthThreshold:
                continue
            mask = torch.from_numpy(mask).cuda()
            newSegmentation[mask] = segmentIndex
            continue
        continue
    newSegmentation = newSegmentation.cpu().numpy()

    if return_plane_depths:
        return newSegmentation, planeDepths.cpu().numpy()
    return newSegmentation

def transformPlanes(transformation, planes):
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)

    centers = planes
    centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1)
    newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
    newCenters = newCenters[:, :3] / newCenters[:, 3:4]

    refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
    refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
    newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
    newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

    planeNormals = newRefPoints - newCenters
    planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
    planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
    newPlanes = planeNormals * planeOffsets
    return newPlanes

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap

def get_mask_blened_image(image, plane_masks):
    plane_masks = torch.from_numpy(plane_masks)  # n, 480, 640
    plane_seg = torch.zeros((480, 640), dtype=torch.int)
    for i in range(plane_masks.shape[0]):
        mask = plane_masks[i]
        plane_seg[mask > 0] = i + 1  # plane idx starts from 1, thus 0 means non-plane
    segmentation = plane_seg.numpy()  # 480, 640
    colors = labelcolormap(256)
    seg = np.stack([colors[segmentation, 0], colors[segmentation, 1], colors[segmentation, 2]], axis=2)
    blend_seg = (seg * 0.7 + image * 0.3).astype(np.uint8)
    seg_mask = (segmentation > 0).astype(np.uint8)
    seg_mask = seg_mask[:, :, np.newaxis]
    blend_seg = blend_seg * seg_mask + image.astype(np.uint8) * (1 - seg_mask)
    return blend_seg

def polygonFromMask(maskedArr):
    # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        import pdb; pdb.set_trace()
        raise ValueError
    return segmentation

def maskFromPolygon(polygon_list):
    polygons = [
        np.array(p, dtype=np.float64) for p in polygon_list
    ]
    rles = mask_util.frPyObjects(
        polygons,
        480,
        640,
    )
    rle = mask_util.merge(rles)
    bm = mask_util.decode(rle)  # h, w
    return bm

def vis(sample1, sample2):
    plt.figure()
    plt.axis('off')
    for i, sample in enumerate([sample1, sample2]):
        filename = sample["file_name"]
        img = utils.read_image(filename, format='RGB')
        img = cv2.resize(img, (640, 480))

        plt.subplot(1, 2, i+1)
        plt.axis('off')
        plt.imshow(img)

    plt.show()
    import pdb; pdb.set_trace()

def get_candidate_imgpairs(scenePath, numImages, sample_step):
    trans = []
    rots = []
    sampled_img_idxs = []
    for img_idx in range(0, numImages, sample_step):
        # get path of image1 and image2
        image_name1 = str(img_idx) + '.jpg'
        image_path1 = os.path.join(scenePath, 'color', image_name1)

        # check plane and pose info of image1 and image2
        poses = []
        poses_inv = []

        # get extrinsics
        posePath = image_path1.replace('color', 'pose').replace('.jpg', '.txt')
        extrinsics_inv = []
        with open(posePath, 'r') as f:
            for line in f:
                extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                continue
            pass
        if len(extrinsics_inv) == 0:
            break

        extrinsics_inv = np.array(extrinsics_inv).reshape((4, 4))
        extrinsics = np.linalg.inv(extrinsics_inv)
        poses.append(extrinsics)
        poses_inv.append(extrinsics_inv)

        extrinsics = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
        extrinsics_inv = np.linalg.inv(extrinsics_inv)
        poses.append(extrinsics)
        poses_inv.append(extrinsics_inv)

        # check rel pose
        try:
            T_rel = np.matmul(poses[1], poses_inv[0])
            R = T_rel[:3, :3].reshape(3, 3)
            t = T_rel[:3, 3].reshape(3, 1)

            mA = np.array([[1., 0., 0.], [0, -1, 0], [0, 0, -1]]).astype(T_rel.dtype)
            mA_inv = np.linalg.inv(mA)

            R_ = np.matmul(mA, R)
            R_ = np.matmul(R_, mA_inv)
            t_ = np.matmul(mA, t)

            position = t_.reshape(-1).tolist()
            rotation_qua = quaternion.from_rotation_matrix(R_)
            rotation_qua = quaternion.as_float_array(rotation_qua)
            rotation_qua = rotation_qua.tolist()

            trans.append(torch.tensor(position).reshape(1, 3))
            rots.append(torch.tensor(rotation_qua).reshape(1, 4))
            sampled_img_idxs.append(img_idx)
        except:
            continue

    if len(trans) == 0:
        return None, sampled_img_idxs

    trans = torch.cat(trans, dim=0)  # n, 3
    rots = torch.cat(rots, dim=0)  # n, 4

    trans_err_matrix = torch.cdist(trans, trans, p=2)
    rot_err_matrix = 2 * torch.acos(
        torch.clamp(torch.matmul(rots, rots.permute(1, 0)).abs(), max=1., min=-1.)) * 180 / np.pi

    m1 = (rot_err_matrix > 15) & (trans_err_matrix > 0.2)
    m2 = (rot_err_matrix > 10) & (trans_err_matrix > 0.5)

    mask = m1 | m2

    if mask.sum() == 0:
        return None, sampled_img_idxs

    idx_pairs = torch.nonzero(mask)

    return idx_pairs, sampled_img_idxs

def vis_corr(batched_inputs, target_masks0, target_masks1):
    colors = labelcolormap(256)

    plt.figure(figsize=(8, 5))
    plt.axis('off')

    # -------------------------------------------------------------------------------------------------------------
    # ********************* get target info of view 0
    target_input0 = batched_inputs[0]["0"]
    target_masks0 = target_masks0.tensor  # shape: [n, h, w]
    file_name0 = target_input0["file_name"]
    image0 = cv2.imread(file_name0, cv2.IMREAD_COLOR)
    image0 = cv2.resize(image0, (640, 480))
    target_seg0 = torch.zeros((480, 640), dtype=torch.int)
    for i in range(target_masks0.shape[0]):
        mask = target_masks0[i]
        target_seg0[mask > 0] = i + 1  # plane idx starts from 1, thus 0 means non-plane
    target_seg_ori0 = target_seg0.clone().numpy()
    target_seg0 = target_seg0.numpy()  # 480, 640
    target_seg0 = np.stack([colors[target_seg0, 0], colors[target_seg0, 1], colors[target_seg0, 2]], axis=2)
    target_blend_seg0 = (target_seg0 * 0.7 + image0 * 0.3).astype(np.uint8)
    target_seg_mask0 = (target_seg_ori0 > 0).astype(np.uint8)
    target_seg_mask0 = target_seg_mask0[:, :, np.newaxis]
    target_blend_seg0 = target_blend_seg0 * target_seg_mask0 + image0.astype(np.uint8) * (1 - target_seg_mask0)

    # ********************* get target info of view 1
    target_input1 = batched_inputs[0]["1"]
    target_masks1 = target_masks1.tensor  # shape: [n, h, w]
    file_name1 = target_input1["file_name"]
    image1 = cv2.imread(file_name1, cv2.IMREAD_COLOR)
    image1 = cv2.resize(image1, (640, 480))
    target_seg1 = torch.zeros((480, 640), dtype=torch.int)
    for i in range(target_masks1.shape[0]):
        mask = target_masks1[i]
        target_seg1[mask > 0] = i + 1  # plane idx starts from 1, thus 0 means non-plane
    target_seg_ori1 = target_seg1.clone().numpy()
    target_seg1 = target_seg1.numpy()  # 480, 640
    target_seg1 = np.stack([colors[target_seg1, 0], colors[target_seg1, 1], colors[target_seg1, 2]], axis=2)
    target_blend_seg1 = (target_seg1 * 0.7 + image1 * 0.3).astype(np.uint8)
    target_seg_mask1 = (target_seg_ori1 > 0).astype(np.uint8)
    target_seg_mask1 = target_seg_mask1[:, :, np.newaxis]
    target_blend_seg1 = target_blend_seg1 * target_seg_mask1 + image1.astype(np.uint8) * (1 - target_seg_mask1)

    # ********************* get coordinate map
    h, w, _ = image0.shape
    xx = torch.arange(0, w).view(1, -1).repeat(h, 1)  # H, W
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w)  # H, W
    xy_grid = torch.stack((xx, yy), dim=0)  # 2, H, W
    xy_grid = xy_grid.numpy()

    # ********************* draw target corr map
    target_corr_image = np.concatenate((target_blend_seg0, target_blend_seg1), axis=1)  # h, 2w, 3
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title('target correspondence', size=8)
    plt.imshow(target_corr_image)
    target_NumPlanes0 = target_masks0.shape[0]
    target_NumPlanes1 = target_masks1.shape[0]
    target_corr = batched_inputs[0]["gt_corrs"]
    for i in range(1, target_NumPlanes0 + 1):
        for j in range(1, target_NumPlanes1 + 1):
            if [i - 1, j - 1] not in target_corr:
                continue
            mask0 = (target_seg_ori0 == i)
            pixel_num0 = mask0.sum()
            if pixel_num0 < 10:
                continue
            masked_xy_map0 = xy_grid * (mask0.astype(xy_grid.dtype)[np.newaxis, :, :])
            center0 = masked_xy_map0.reshape(2, -1).sum(1) / pixel_num0
            center0 = np.round(center0).astype(np.int)

            mask1 = (target_seg_ori1 == j)
            pixel_num1 = mask1.sum()
            if pixel_num1 < 10:
                continue
            masked_xy_map1 = xy_grid * (mask1.astype(xy_grid.dtype)[np.newaxis, :, :])
            center1 = masked_xy_map1.reshape(2, -1).sum(1) / pixel_num1
            center1 = np.round(center1).astype(np.int)
            center1[0] += w

            plt.plot([center0[0], center1[0]],
                     [center0[1], center1[1]], color='black', linewidth=1,
                     linestyle="-")
            plt.plot(center0[0], center0[1], 'c.')
            plt.plot(center1[0], center1[1], 'c.')
            plt.text(center0[0], center0[1], 'p%d' % (i - 1), fontsize=8, verticalalignment="bottom",
                     horizontalalignment="center")
            plt.text(center1[0], center1[1], 'p%d' % (j - 1), fontsize=8, verticalalignment="bottom",
                     horizontalalignment="center")

    # ********************* draw images
    zero_pad = np.zeros([h, 10, 3]).astype(image0.dtype)
    cat_image = np.concatenate((image0, zero_pad, image1), axis=1)  # h, 2w, 3
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(cat_image)

    plt.show()
    plt.close()

def vis_depth_img(img, depth, plane_depth):
    plt.figure(figsize=(8, 5))
    plt.axis('off')

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('image', size=8)
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('depth', size=8)
    plt.imshow(depth)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('depth plane', size=8)
    plt.imshow(plane_depth)

    plt.show()

def get_scene_info(scenePath):
    global_plane_info = np.load(scenePath + '/annotation/plane_info.npy', allow_pickle=True)
    global_planes = np.load(scenePath + '/annotation/planes.npy', allow_pickle=True)
    assert global_plane_info.shape[0] == global_planes.shape[0]

    # ***********************************************   calculate K & K_inv
    camera_vec = np.zeros(6)  # fx, fy, mx, my, W, H
    with open(scenePath + '/' + scene_name + '.txt') as f:
        for line in f:
            line = line.strip()
            tokens = [token for token in line.split(' ') if token.strip() != '']
            if tokens[0] == "fx_depth":
                camera_vec[0] = float(tokens[2])
            if tokens[0] == "fy_depth":
                camera_vec[1] = float(tokens[2])
            if tokens[0] == "mx_depth":
                camera_vec[2] = float(tokens[2])
            if tokens[0] == "my_depth":
                camera_vec[3] = float(tokens[2])
            elif tokens[0] == "colorWidth":
                colorWidth = int(tokens[2])
            elif tokens[0] == "colorHeight":
                colorHeight = int(tokens[2])
            elif tokens[0] == "depthWidth":
                depthWidth = int(tokens[2])
            elif tokens[0] == "depthHeight":
                depthHeight = int(tokens[2])
            elif tokens[0] == "numDepthFrames":
                numImages = int(tokens[2])
    camera_vec[4] = depthWidth
    camera_vec[5] = depthHeight

    fx = camera_vec[0]
    fy = camera_vec[1]
    offset_x = camera_vec[2]
    offset_y = camera_vec[3]
    K = [[fx, 0, offset_x],
         [0, fy, offset_y],
         [0, 0, 1]]
    camera_K = np.array(K).reshape(3, 3)

    return camera_vec, camera_K, depthWidth, depthHeight, numImages-10, global_planes, global_plane_info

confidentClasses = {'wall': True,
                            'floor': True,
                            'cabinet': True,
                            'bed': True,
                            'chair': False,
                            'sofa': False,
                            'table': True,
                            'door': True,
                            'window': True,
                            'bookshelf': False,
                            'picture': True,
                            'counter': True,
                            'blinds': False,
                            'desk': True,
                            'shelf': False,
                            'shelves': False,
                            'curtain': False,
                            'dresser': True,
                            'pillow': False,
                            'mirror': False,
                            'entrance': True,
                            'floor mat': True,
                            'clothes': False,
                            'ceiling': True,
                            'book': False,
                            'books': False,
                            'refridgerator': True,
                            'television': True,
                            'paper': False,
                            'towel': False,
                            'shower curtain': False,
                            'box': True,
                            'whiteboard': True,
                            'person': False,
                            'night stand': True,
                            'toilet': False,
                            'sink': False,
                            'lamp': False,
                            'bathtub': False,
                            'bag': False,
                            'otherprop': False,
                            'otherstructure': False,
                            'otherfurniture': False,
                            'unannotated': False,
                            '': False
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='ScanNet/')
    parser.add_argument('--data_type', type=str, default='', help='train / test')

    args = parser.parse_args()

    root_dir = os.path.join(args.data_path, 'scans')
    scene_names = os.listdir(root_dir)
    scene_names_train = scene_names[:1210]
    scene_names_test = scene_names[1210:]
    max_per_scene = 40
    if args.data_type == 'train':
        dType = 'cached_set_train'
        dScene = scene_names_train
        sample_step = 20
        max_hit_num = 1
    elif args.data_type == 'test':
        dType = 'cached_set_test'
        dScene = scene_names_test
        sample_step = 40
        max_hit_num = 2
    else:
        raise NotImplementedError

    trans_errs = []
    rot_errs = []

    classLabelMap = {}
    with open(os.path.join(args.data_path, "scannetv2-labels.combined.tsv")) as info_file:
        line_index = 0
        for line in info_file:
            if line_index > 0:
                line = line.split('\t')
                key = line[1].strip()
                if line[4].strip() != '':
                    label = int(line[4].strip())
                else:
                    label = -1
                    pass
                classLabelMap[key] = label
                classLabelMap[key + 's'] = label
                classLabelMap[key + 'es'] = label
                pass
            line_index += 1
            continue
        pass

    confident_labels = {}
    for name, confidence in confidentClasses.items():
        if confidence and name in classLabelMap:
            confident_labels[classLabelMap[name]] = True
            pass
        continue

    # generate data list
    data = []
    data_len = 0.
    for si in tqdm(range(len(dScene))):
        # ***********************************************   get scene info
        scene_name = dScene[si]
        scenePath = os.path.join(root_dir, scene_name)
        camera_vec, camera_K, depthWidth, depthHeight, numImages, global_planes, global_plane_info = get_scene_info(scenePath)

        # ***********************************************   get candidate pair idx
        candidate_imgpair_idx, sampled_img_idxs = get_candidate_imgpairs(scenePath, numImages, sample_step)
        if candidate_imgpair_idx is None:
            continue

        # ***********************************************   get image pairs
        selected_pair_num = 0
        hits = torch.zeros(len(sampled_img_idxs))
        print("candidate num:", len(candidate_imgpair_idx))
        for cpi in tqdm(range(len(candidate_imgpair_idx))):
            pair_idx = candidate_imgpair_idx[cpi]
            if selected_pair_num >= max_per_scene:
                break
            if hits[pair_idx[0]] >= max_hit_num or hits[pair_idx[1]] >= max_hit_num:
                continue

            img_idx1 = sampled_img_idxs[pair_idx[0]]
            img_idx2 = sampled_img_idxs[pair_idx[1]]

            image_name1 = str(img_idx1) + '.jpg'
            image_name2 = str(img_idx2) + '.jpg'
            image_path1 = os.path.join(scenePath, 'color', image_name1)
            image_path2 = os.path.join(scenePath, 'color', image_name2)

            # check plane and pose info of image1 and image2
            plane_idxs = []
            poses = []
            poses_inv = []
            planes_ori_list = []
            seg_list = []
            for view_idx, img_file in enumerate([image_path1, image_path2]):
                # get pose
                posePath = img_file.replace('color', 'pose').replace('.jpg', '.txt')
                extrinsics_inv = []
                with open(posePath, 'r') as f:
                    for line in f:
                        extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                        continue
                    pass
                if len(extrinsics_inv) == 0:
                    hits[pair_idx[view_idx]] += 100
                    break
                extrinsics_inv = np.array(extrinsics_inv).reshape((4, 4))
                extrinsics = np.linalg.inv(extrinsics_inv)
                poses.append(extrinsics)
                extrinsics_inv = np.linalg.inv(extrinsics)
                poses_inv.append(extrinsics_inv)

                # get plane masks
                segmentationPath = img_file.replace('color', 'annotation/segmentation').replace('.jpg', '.png')
                segmentation = cv2.imread(segmentationPath, -1).astype(np.int32)
                segmentation = (segmentation[:, :, 2] * 256 * 256 + segmentation[:, :, 1] * 256 + segmentation[:, :, 0]) // 100 - 1
                segments, counts = np.unique(segmentation, return_counts=True)
                segmentList = zip(segments.tolist(), counts.tolist())
                segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]
                segmentList = sorted(segmentList, key=lambda x: -x[1])

                newPlanes = []
                newPlaneInfo_global = []
                newSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)

                newIndex = 0
                planeAreaThreshold = 800
                for oriIndex, count in segmentList:
                    if count < planeAreaThreshold:
                        continue
                    if oriIndex >= len(global_planes):
                        continue
                    if np.linalg.norm(global_planes[oriIndex]) < 1e-4:
                        continue
                    newPlanes.append(global_planes[oriIndex])
                    newSegmentation[segmentation == oriIndex] = newIndex
                    newPlaneInfo_global.append(global_plane_info[oriIndex] + [oriIndex])
                    newIndex += 1
                    continue
                if newIndex < 2 or newIndex > 22:
                    hits[pair_idx[view_idx]] += 100
                    break
                (h, w) = newSegmentation.shape
                if (newSegmentation > -1).sum() / (h * w) < 0.7:
                    hits[pair_idx[view_idx]] += 100
                    break
                segmentation = newSegmentation
                planes = np.array(newPlanes)
                plane_info_global = newPlaneInfo_global

                image = cv2.imread(img_file)
                depthPath = img_file.replace('color', 'depth').replace('.jpg', '.png')
                try:
                    depth = cv2.imread(depthPath, -1).astype(np.float32) / 1000.0
                except:
                    import pdb;
                    pdb.set_trace()
                image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
                # warp plane
                extrinsics_ = extrinsics.copy()
                temp = extrinsics_[1].copy()
                extrinsics_[1] = extrinsics_[2]
                extrinsics_[2] = -temp
                planes = transformPlanes(extrinsics_, planes)
                # clean seg
                segmentation, plane_depths = cleanSegmentation(image, planes, newPlaneInfo_global, segmentation,
                                                               depth,
                                                               camera_vec,
                                                               planeAreaThreshold=800,
                                                               planeWidthThreshold=30,
                                                               confident_labels=confident_labels,
                                                               return_plane_depths=True)
                # check plane num
                clean_seg_idx = np.unique(segmentation).tolist()
                if len(clean_seg_idx) - 1 < 2:
                    hits[pair_idx[view_idx]] += 100
                    break
                # check depth
                masks = (np.expand_dims(segmentation, -1) == np.arange(len(planes))).astype(np.float32)
                plane_depth = (plane_depths.transpose((1, 2, 0)) * masks).sum(2)
                plane_mask = masks.max(2)
                plane_mask *= (depth > 1e-4).astype(np.float32)
                plane_area = plane_mask.sum()
                depth_error = (np.abs(plane_depth - depth) * plane_mask).sum() / max(plane_area, 1)
                if depth_error > 0.1:
                    hits[pair_idx[view_idx]] += 100
                    break

                # used to debug
                # vis_depth_img(image, depth, plane_depth)
                # print(np.unique(segmentation))
                # import pdb;pdb.set_trace()

                # check plane area
                if (segmentation > -1).sum() / (h * w) < 0.7:
                    hits[pair_idx[view_idx]] += 100
                    break

                # renew segmentation and planeInfo
                newSegmentation_filter = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
                newPlanes_filter = []
                newPlane_globalIdx_filter = []
                newIndex = 0
                for seg_idx in clean_seg_idx:
                    if seg_idx == -1:
                        continue
                    if (segmentation == seg_idx).sum() < planeAreaThreshold:
                        continue
                    newSegmentation_filter[segmentation == seg_idx] = newIndex
                    newPlanes_filter.append(planes[seg_idx])
                    newPlane_globalIdx_filter.append(plane_info_global[seg_idx][-1])
                    newIndex += 1
                plane_idxs.append(newPlane_globalIdx_filter)
                planes_ori_list.append(np.array(newPlanes_filter))
                seg_list.append(newSegmentation_filter)

            if len(plane_idxs) < 2:
                continue

            # check matched plane nums
            gt_corrs = []
            plane_idx1 = plane_idxs[0]
            plane_idx2 = plane_idxs[1]
            for i in range(len(plane_idx1)):
                local_idx1 = i
                global_idx1 = plane_idx1[i]
                for j in range(len(plane_idx2)):
                    local_idx2 = j
                    global_idx2 = plane_idx2[j]
                    if global_idx1 == global_idx2:
                        gt_corrs.append([local_idx1, local_idx2])
            if min(len(plane_idx1), len(plane_idx2)) > 10 and len(gt_corrs) / min(len(plane_idx1), len(plane_idx2)) > 0.7:
                continue
            elif len(gt_corrs) < 3 or len(gt_corrs) > 7:
                continue
            else:
                pass

            # check rel pose
            try:
                T_rel = np.matmul(poses[1], poses_inv[0])
                R = T_rel[:3, :3].reshape(3, 3)
                t = T_rel[:3, 3].reshape(3, 1)

                mA = np.array([[1., 0., 0.], [0, -1, 0], [0, 0, -1]]).astype(T_rel.dtype)
                mA_inv = np.linalg.inv(mA)

                R_ = np.matmul(mA, R)
                R_ = np.matmul(R_, mA_inv)
                t_ = np.matmul(mA, t)

                position = t_.reshape(-1).tolist()
                rotation_qua = quaternion.from_rotation_matrix(R_)
                rotation_qua = quaternion.as_float_array(rotation_qua)

                if rotation_qua[0] < 0.:
                    rotation_qua = -rotation_qua

                rotation_qua = rotation_qua.tolist()
                rel_pose = {
                    "position": position,
                    "rotation": rotation_qua
                }

                trans_err = torch.norm(torch.tensor(position).reshape(1, -1), dim=-1).numpy()

                rq1 = np.array([1., 0., 0., 0.]).reshape(1, -1)
                rq0 = np.array(rotation_qua).reshape(1, -1)
                rot_err = angle_error_qua(rq0, rq1)
            except:
                continue

            ''''''
            # get plane and image info of image1 and image2
            sample_list = []
            sample_list_extra = []
            for img_file, plo, extr, seg in zip([image_path1, image_path2], planes_ori_list, poses, seg_list):
                # read image
                im = utils.read_image(img_file, format='RGB')
                im = cv2.resize(im, (640, 480))
                im = torch.as_tensor(im.transpose(2, 0, 1).astype("float32"))

                # get plane info
                planes = np.concatenate([plo[:, 0:1], -plo[:, 2:3], plo[:, 1:2]], axis=-1)

                masks = (np.expand_dims(seg, -1) == np.arange(len(planes))).astype(np.float32)
                masks = masks.transpose(2, 0, 1)  # n, h, w
                plane_num = planes.shape[0]
                bit_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )

                # build annotations of the image
                annotations = []
                masks_fromPoly = []
                for pi in range(plane_num):
                    m_i = masks[pi]  # h, w
                    area = m_i.sum()
                    rle_pi = mask_util.encode(np.asfortranarray(m_i > 0))
                    bbox_i = mask_util.toBbox(rle_pi).tolist()  # x, y, w, h
                    polygons_pi = polygonFromMask(m_i.astype(np.uint8))  # list

                    # get anno of per image
                    ann_pi = {
                        "id": pi,
                        "image_id": '',
                        "category_id": 0,
                        "area": float(area),
                        "segmentation": polygons_pi,
                        "width": 640,
                        "height": 480,
                        "plane": planes[pi].tolist(),
                        "iscrowd": 0,
                        "bbox": bbox_i,
                        "bbox_mode": 1,
                    }
                    annotations.append(ann_pi)

                img_id = img_file.split('/')[-1].split('.')[0]
                ann_file_scene_path = os.path.join(args.data_path, 'twoView_Anns/', scene_name)

                sample = {"image_id": scene_name + "-" + img_id,
                          "file_name": img_file,
                          "height": 480,
                          "width": 640,
                          "annotations": annotations,
                          "gt_plane_num": plane_num,
                          }
                sample_extra = {
                    "plane_masks": bit_masks,
                    "camera_K": camera_K  # 3, 3
                }

                sample_list.append(sample)
                sample_list_extra.append((ann_file_scene_path, img_id, sample_extra))

            assert len(sample_list) == 2 and len(sample_list_extra) == 2

            batched_input = {"0": sample_list[0],
                             "1": sample_list[1],
                             "rel_pose": rel_pose,
                             "gt_corrs": gt_corrs}
            data.append(batched_input)

            print("writing...")
            for (ann_scene_path, image_id, sample_ex) in sample_list_extra:
                if not os.path.exists(ann_scene_path):
                    os.makedirs(ann_scene_path)
                ann_file_path = os.path.join(ann_scene_path, image_id + '.pkl')
                with open(ann_file_path, "wb") as f:
                    pickle.dump(sample_ex.copy(), f)

            # vis
            # print(pair_idx)
            # vis_corr([batched_input], sample_list_extra[0][2]['plane_masks'], sample_list_extra[1][2]['plane_masks'])
            # import pdb; pdb.set_trace()

            hits[pair_idx[0]] += 1
            hits[pair_idx[1]] += 1
            selected_pair_num += 1

            trans_errs.append(trans_err[0])
            rot_errs.append(rot_err[0])

        data_len += selected_pair_num
        print("")
        print("selected num (all num): %d (%d)" % (selected_pair_num, data_len))
        print("")

    tran_acc = sum(_ < 1 for _ in trans_errs) / len(trans_errs)
    rot_acc = sum(_ < 30 for _ in rot_errs) / len(rot_errs)

    tran_acc2 = sum(_ < 0.5 for _ in trans_errs) / len(trans_errs)
    rot_acc2 = sum(_ < 15 for _ in rot_errs) / len(rot_errs)

    tran_acc3 = sum(_ < 0.2 for _ in trans_errs) / len(trans_errs)
    rot_acc3 = sum(_ < 10 for _ in rot_errs) / len(rot_errs)

    tran_acc4 = sum(_ < 0.1 for _ in trans_errs) / len(trans_errs)
    rot_acc4 = sum(_ < 5 for _ in rot_errs) / len(rot_errs)

    median_tran_err = np.median(np.array(trans_errs))
    mean_tran_err = np.mean(np.array(trans_errs))
    median_rot_err = np.median(np.array(rot_errs))
    mean_rot_err = np.mean(np.array(rot_errs))

    print(
        "Mean Error [tran, rot]: {:.2f}, {:.2f}".format(mean_tran_err, mean_rot_err)
    )
    print(
        "Median Error [tran, rot]: {:.2f}, {:.2f}".format(
            median_tran_err, median_rot_err
        )
    )
    print(
        "Accuracy [tran(1m), rot(30')]: {:.1f}, {:.1f}".format(tran_acc * 100, rot_acc * 100)
    )
    print(
        "Accuracy [tran(0.5m), rot(15')]: {:.1f}, {:.1f}".format(tran_acc2 * 100, rot_acc2 * 100)
    )
    print(
        "Accuracy [tran(0.2m), rot(10')]: {:.1f}, {:.1f}".format(tran_acc3 * 100, rot_acc3 * 100)
    )
    print(
        "Accuracy [tran(0.1m), rot(5')]: {:.1f}, {:.1f}".format(tran_acc4 * 100, rot_acc4 * 100)
    )

    out_dict = {
        'categories': [{'id': 0, 'name': 'plane'}],
        'data': data
    }
    print("data len = ", len(data))
    print("writing...")
    json_path = os.path.join(args.data_path, 'scannet_json')
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    file_path = os.path.join(json_path, '%s.json' % (dType))
    with open(file_path, "w") as f:
        json.dump(out_dict, f)
        print('finished')