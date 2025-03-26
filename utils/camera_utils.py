#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from PIL import Image
from utils.graphics_utils import fov2focal
import torch.nn.functional as F
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    loaded_mask = None
    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        import torch
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        # loaded_mask = None

        # use masked image to train
        if args.name == 'gaussianobject':
            # NOTE: ref GaussianObject
            if cam_info.mask is not None:
                resized_mask = resize_mask_image(cam_info.mask, resolution)
                loaded_mask = torch.from_numpy(resized_mask).unsqueeze(0)
            else:
                loaded_mask = None
        else:
            if cam_info.mask is not None:
                loaded_mask = PILtoTorch(cam_info.mask, resolution)
        gt_image = resized_image_rgb

    return Camera(resolution, colmap_id=cam_info.uid, K=cam_info.K, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, gt_depth=cam_info.depth,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def resize_mask_image(mask, resolution):
    """
    Resize the image to the specified resolution.

    Args:
        mask (np.array): Input mask image as a NumPy array with shape (h, w).
        resolution (tuple): Target resolution as a tuple (width, height).

    Returns:
        np.array: Resized mask image as a NumPy array with shape (h, w).
    """
    # Ensure that resolution is in (width, height) format
    width, height = resolution

    # Resize the mask image
    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return resized_mask