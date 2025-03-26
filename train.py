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

import os
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
import torch
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss, ssim, bce_loss, generate_sharp_decay_weight_matrix, normal_from_depth_image, tv_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image, compute_diff_with_mask
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from imgviz import depth2rgb
import cv2
import numpy as np
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
if not os.path.exists(os.path.join(ROOT_PATH, 'debug_data')):
    os.makedirs(os.path.join(ROOT_PATH, 'debug_data'))
if not os.path.exists(os.path.join(ROOT_PATH, 'debug_data', 'rgb_depth_image')):
    os.makedirs(os.path.join(ROOT_PATH, 'debug_data', 'rgb_depth_image'))

def training(dataset_name, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    dataset.name = dataset_name
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    runtime_data_dir = os.path.join(dataset.model_path, 'runtime_data')
    if not os.path.exists(runtime_data_dir):
        os.makedirs(runtime_data_dir)
    
    start_normal_iter = 7000
    start_dist_iter = 3000
    use_depth_loss = False
    use_normal_loss = False
    use_alpha_loss = True
        
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, rend_alpha = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['rend_alpha']
        
        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else None
        assert gt_alpha_mask is not None, "gt_alpha_mask is None"
        # gt_image = gt_image * (gt_alpha_mask > 0).expand_as(gt_image) # only consider the pixels where the alpha mask is > 0
        Ll1 = l1_loss(image, gt_image, gt_alpha_mask)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > start_normal_iter else 0.0
        lambda_dist = opt.lambda_dist if iteration > start_dist_iter else 0.0
        
        # depth loss
        gt_depth = viewpoint_cam.gt_depth.cuda() if viewpoint_cam.gt_depth is not None else None
        if use_depth_loss and gt_depth is not None:
            gt_depth_mask = gt_depth > 0
            Ldepth = l1_loss(render_pkg["render_depth"], gt_depth, gt_depth_mask)
            # ref [DROID-Splat](https://arxiv.org/pdf/2411.17660)
            depth_loss = opt.lambda_depth * Ldepth
        else:
            depth_loss = torch.tensor(0.0, device=gt_image.device)
        
        loss = rgb_loss + depth_loss

        if use_alpha_loss:
            # weight matrix
            # weight_matrix = generate_sharp_decay_weight_matrix(gt_alpha_mask, D_threshold=20, sigma=5)
            # weight_matrix = weight_matrix.to(gt_alpha_mask.device)
            # alpha_weighted = gt_alpha_mask * weight_matrix
            Ll1_alpha = bce_loss(rend_alpha, gt_alpha_mask)
            alpha_loss = opt.lambda_alpha * Ll1_alpha
            loss += alpha_loss

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        if use_normal_loss:
            depth_normal = normal_from_depth_image(gt_depth, 
                                                viewpoint_cam.K[0, 0], 
                                                viewpoint_cam.K[1, 1], 
                                                viewpoint_cam.K[0, 2], 
                                                viewpoint_cam.K[1, 2],
                                                img_size=(gt_depth.shape[2], gt_depth.shape[1]),
                                                c2w=viewpoint_cam.world_view_transform.inverse().transpose(0, 1),
                                                device=gt_depth.device)
            depth_normal = depth_normal.permute(2, 0, 1)
            normal_error = (1 - (rend_normal * depth_normal).sum(dim=0))[None]
        else:
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()
        
        with torch.no_grad():
            # save image and gt_image
            # if iteration > start_normal_iter and use_depth_loss and use_alpha_loss:
            #     print(f'rgb_loss: {rgb_loss.mean()}, depth_loss: {depth_loss.mean()}, alpha_loss: {alpha_loss.mean()}, normal_loss: {normal_loss.mean()}, dist_loss: {dist_loss.mean()}')
            save_image_flag = True
            if save_image_flag and iteration % 1000 == 0:
                image_np = image.permute(1,2,0).detach().cpu().numpy()
                image_np = np.clip(image_np, 0, 1)
                image_np = (image_np * 255).astype(np.uint8)
                gt_image_np = gt_image.permute(1,2,0).detach().cpu().numpy()
                gt_image_np = (gt_image_np * 255).astype(np.uint8)
                rgb_image = np.hstack((image_np, gt_image_np))
                
                # depth with magma colormap
                depth_np = render_pkg["render_depth"].squeeze().detach().cpu().numpy()
                depth_vis = depth2rgb(depth_np, min_value=depth_np.min(), max_value=depth_np.max(), colormap="magma")
                if use_depth_loss:
                    gt_depth_np = gt_depth.squeeze().detach().cpu().numpy()
                    gt_depth_vis = depth2rgb(gt_depth_np, min_value=gt_depth_np.min(), max_value=gt_depth_np.max(), colormap="magma")
                else:
                    gt_depth_vis = np.zeros_like(depth_vis)
                depth_image = np.hstack((depth_vis, gt_depth_vis))
                
                # mask
                mask_np = rend_alpha.squeeze().detach().cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
                if use_alpha_loss:
                    gt_mask_np = gt_alpha_mask.squeeze().detach().cpu().numpy()
                    gt_mask_np = (gt_mask_np * 255).astype(np.uint8)
                else:
                    gt_mask_np = np.zeros_like(mask_np)
                mask_np = cv2.cvtColor(np.expand_dims(mask_np, axis=-1), cv2.COLOR_GRAY2BGR)
                gt_mask_np = cv2.cvtColor(np.expand_dims(gt_mask_np, axis=-1), cv2.COLOR_GRAY2BGR)
                mask_image = np.hstack((mask_np, gt_mask_np))
                
                # normal
                rend_normal_np = rend_normal.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                if opt.lambda_normal > 0:
                    if use_normal_loss:
                        surf_normal_np = depth_normal.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                    else:
                        surf_normal_np = surf_normal.permute(1, 2, 0).detach().cpu().numpy()
                else:
                    surf_normal_np = np.zeros_like(rend_normal_np)
                rend_normal_vis = ((rend_normal_np + 1) / 2.0 * 255).clip(0, 255).astype(np.uint8)
                surf_normal_vis = ((surf_normal_np + 1) / 2.0 * 255).clip(0, 255).astype(np.uint8)
                normal_image = np.hstack((rend_normal_vis, surf_normal_vis))
                
                # 改进的差异图像计算方法
                # RGB差异
                def compute_improved_diff(img1, img2, mask=None):
                    if mask is None:
                        mask = np.ones(img1.shape[:2], dtype=bool)
                    if len(img1.shape) == 3 and img1.shape[2] == 3:
                        # RGB图像：计算每个像素的L2距离（欧几里得距离）
                        diff = np.sqrt(np.sum((img1.astype(float) - img2.astype(float))**2, axis=2))
                    else:
                        # 灰度图像或深度图：计算绝对差异
                        diff = np.abs(img1.astype(float) - img2.astype(float))
                    
                    # 归一化并应用mask
                    if np.max(diff) > 0:
                        diff = diff / np.max(diff) * 255
                    diff = diff * mask
                    
                    # 应用magma colormap
                    import matplotlib.pyplot as plt
                    from matplotlib import cm
                    magma_cmap = cm.get_cmap('magma')
                    diff_colored = magma_cmap(diff.astype(np.uint8)/255.0)[:,:,:3] * 255
                    return diff_colored.astype(np.uint8)
                
                mask = gt_alpha_mask.squeeze().detach().cpu().numpy() > 0.5
                
                # 计算改进的差异图像
                rgb_diff_image = compute_improved_diff(image_np, gt_image_np, mask)

                # 深度差异
                if use_depth_loss:
                    depth_mask = gt_depth_mask.squeeze().detach().cpu().numpy()
                    depth_loss_image = compute_improved_diff(depth_vis, gt_depth_vis, depth_mask)
                else:
                    depth_loss_image = np.zeros_like(rgb_diff_image)

                # Alpha差异
                if use_alpha_loss:
                    alpha_loss_image = compute_improved_diff(mask_np, gt_mask_np, mask)
                else:
                    alpha_loss_image = np.zeros_like(rgb_diff_image)

                # 法线差异
                normal_loss_image = compute_improved_diff(rend_normal_vis, surf_normal_vis, mask)
                
                rgb_image_with_loss = np.hstack((rgb_image, rgb_diff_image))
                depth_image_with_loss = np.hstack((depth_image, depth_loss_image))
                mask_image_with_loss = np.hstack((mask_image, alpha_loss_image))
                normal_image_with_loss = np.hstack((normal_image, normal_loss_image))
                
                runtime_image = np.vstack((rgb_image_with_loss, mask_image_with_loss, depth_image_with_loss, normal_image_with_loss))
                cv2.imwrite(os.path.join(runtime_data_dir, f'{iteration}.png'),cv2.cvtColor(runtime_image, cv2.COLOR_BGR2RGB))
            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(dataset, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args, opt):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    with open(os.path.join(args.model_path, "opt_args"), 'w') as opt_log_f:
        opt_dict = {attr: getattr(opt, attr) for attr in dir(opt) 
                   if not attr.startswith('_') and not callable(getattr(opt, attr))}
        opt_log_f.write(str(opt_dict))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(dataset, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()[::len(scene.getTestCameras()) // 50 if len(scene.getTestCameras()) > 50 else 1][:50]}, 
                        {'name': 'train', 'cameras' : scene.getTrainCameras()[::len(scene.getTrainCameras()) // 50 if len(scene.getTrainCameras()) > 50 else 1][:50]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                rgb_l1_test = 0.0
                depth_l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                gaussians_count = scene.gaussians._xyz.shape[0]
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # save image and gt_image
                    directory = f'{dataset.model_path}/check_render_rgb/{config["name"]}/{iteration}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torchvision.utils.save_image(image, f'{directory}/image_{idx}.png')
                    render_depth = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render_depth"]
                    if viewpoint.gt_depth is not None:
                        gt_depth_image = viewpoint.gt_depth.to("cuda")
                    directory = f'{dataset.model_path}/check_render_depth/{config["name"]}/{iteration}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torchvision.utils.save_image(render_depth, f'{directory}/depth_{idx}.png')
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    rgb_l1_test += l1_loss(image, gt_image).mean().double()
                    if viewpoint.gt_depth is not None:
                        gt_depth_mask = gt_depth_image > 0
                        depth_l1_test += l1_loss(render_depth, gt_depth_image, gt_depth_mask).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                rgb_l1_test /= len(config['cameras'])
                depth_l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                results_file = 'results_test.txt' if config['name'] == 'test' else 'results_train.txt'
                with open(f'{dataset.model_path}/{results_file}', 'a') as f:
                    f.write("\n[ITER {}] Evaluating {}: RGBL1 {} DepthL1 {} PSNR {} SSIM {} LPIPS {} Gaussians Count {}\n".format(iteration, config['name'], rgb_l1_test, depth_l1_test, psnr_test, ssim_test, lpips_test, gaussians_count))

                torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--dataset", type=str, default="custom", 
                        choices=["custom", "gaussianobject"],
                        help="Dataset to use for training")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args.dataset, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
