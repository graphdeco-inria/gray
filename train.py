import os
import shutil
from gray.config import *
import json


@dataclass
class ParseConfigPath:
    config: Annotated[Optional[str], arg(aliases=["-c"])] = None


# * Parse Config, optionally extending from a provided config file
first_parse, extra_args = tyro.cli(ParseConfigPath, return_unknown_args=True)
default = (
    RaytracerConfig(**json.load(open(first_parse.config, "r"))) if first_parse.config else None
)
cfg = tyro.cli(Config, args=extra_args, default=default)

# * Confirm overwrite if model path already exists
if os.path.exists(cfg.model_path) and not cfg.yes:
    response = (
        input(f"Output folder '{cfg.model_path}' already exists. Overwrite? [y/N]: ")
        .strip()
        .lower()
    )
    if response == "y":
        shutil.rmtree(cfg.model_path, ignore_errors=True)
    else:
        exit(0)

# * Wait until after Config parsing to import slower modules
from gray.imports import *
from gray.prelude import *
from gray.memory import GpuMemoryMonitor
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from torch.utils.tensorboard import SummaryWriter

# * Read scene and prep preview cameras
set_seeds(0)
scene = SceneInfo.from_colmap(cfg)
cam0 = scene.train_cameras[0]
if cfg.preview_train_image_name:
    cam0 = {cam.image_name: cam for cam in scene.train_cameras}[cfg.preview_train_image_name]
if scene.test_cameras:
    test_cam0 = scene.test_cameras[0]
    if cfg.preview_test_image_name:
        test_cam0 = {cam.image_name: cam for cam in scene.test_cameras}[cfg.preview_test_image_name]

# *** Init gaussians and raytracer
raytracer = Raytracer.from_point_cloud(
    cfg, scene.point_cloud, cam0.image_width, cam0.image_height
)
if cfg.exposure_comp_enabled:
    raytracer.init_exposure_comp(scene.train_cameras)
gaussians = raytracer.cuda_module.get_gaussians()

# * Setup output folder
print("Output folder: {}".format(cfg.model_path))
os.makedirs(cfg.model_path, exist_ok=True)
cli_json_path = os.path.join(cfg.model_path, "config.json")
with open(cli_json_path, "w") as f:
    json.dump(vars(cfg), f, indent=4)
cameras_json_path = os.path.join(cfg.model_path, "cameras.json")
with open(cameras_json_path, "w") as f:
    json.dump([cam.to_json() for cam in scene.train_cameras + scene.test_cameras], f, indent=4)
with open(os.path.join(cfg.model_path, "preview_cameras.json"), "w") as f:
    json.dump(
        {
            "train": cam0.image_name,
            "test": test_cam0.image_name if scene.test_cameras else None,
        },
        f,
        indent=4,
    )

# * Setup viewer
if cfg.viewer:
    from viewer.types import ViewerMode
    from view import GaussianViewer

    viewer = GaussianViewer(raytracer, scene.train_cameras, scene.test_cameras, training=True)
    viewer_thd = Thread(target=viewer.run, daemon=True)
    viewer_thd.start()

# * Setup learning rate schedules
schedule_mean = get_expon_lr_func(
    lr_init=cfg.lr_mean_init * scene.point_cloud.radius,
    lr_final=cfg.lr_mean_final * scene.point_cloud.radius,
    lr_delay_mult=cfg.lr_schedule_delay_mult,
    max_steps=cfg.iterations,
)
schedule_dc = get_expon_lr_func(
    lr_init=cfg.lr_sh_dc_init,
    lr_final=cfg.lr_sh_dc_final,
    lr_delay_mult=cfg.lr_schedule_delay_mult,
    max_steps=cfg.iterations,
)
schedule_rotation = get_expon_lr_func(
    lr_init=cfg.lr_rotation_init,
    lr_final=cfg.lr_rotation_final,
    lr_delay_mult=cfg.lr_schedule_delay_mult,
    max_steps=cfg.iterations,
)
schedule_scale = get_expon_lr_func(
    lr_init=cfg.lr_scale_init,
    lr_final=cfg.lr_scale_final,
    lr_delay_mult=cfg.lr_schedule_delay_mult,
    max_steps=cfg.iterations,
)
schedule_opacity = get_expon_lr_func(
    lr_init=cfg.lr_opacity_init,
    lr_final=cfg.lr_opacity_final,
    lr_delay_mult=cfg.lr_schedule_delay_mult,
    max_steps=cfg.iterations,
)
schedule_exposure_comp = get_expon_lr_func(
    lr_init=cfg.exposure_comp_lr_init,
    lr_final=cfg.exposure_comp_lr_final,
    lr_delay_mult=cfg.exposure_comp_lr_delay_mult,
    max_steps=cfg.iterations,
)

# * Setup logging
writer = SummaryWriter(log_dir=cfg.model_path)
l1_avg = 0.0
psnr_avg = 0.0
last_psnr_avg = None
losses_log = open(os.path.join(cfg.model_path, f"losses.csv"), "w")
print("iteration l1 psnr", file=losses_log, flush=True)
psnr_log = open(os.path.join(cfg.model_path, f"psnr.csv"), "w")
print("iteration train test", file=psnr_log, flush=True)
ssim_log = open(os.path.join(cfg.model_path, f"ssim.csv"), "w")
print("iteration train test", file=ssim_log, flush=True)
time_log = open(os.path.join(cfg.model_path, f"time.csv"), "w")
print("iteration elapsed_time", file=time_log, flush=True)
num_gaussians_log = open(os.path.join(cfg.model_path, f"num_gaussians.csv"), "w")
print("iteration num_gaussians", file=num_gaussians_log, flush=True)
traversal_stats_log = open(os.path.join(cfg.model_path, f"traversal_stats.csv"), "w")
print("iteration,num_hit_per_ray,num_accum_per_ray", file=traversal_stats_log, flush=True)
geometry_stats_log = open(os.path.join(cfg.model_path, f"geometry_stats.csv"), "w")
print("iteration,opacity,scale,anisotropy", file=geometry_stats_log, flush=True)
preview_psnr_log = open(os.path.join(cfg.model_path, f"preview_psnr.csv"), "w")
print("iteration train test", file=preview_psnr_log, flush=True)
preview_ssim_log = open(os.path.join(cfg.model_path, f"preview_ssim.csv"), "w")
print("iteration train test", file=preview_ssim_log, flush=True)
executor = ThreadPoolExecutor()
memory_monitor = GpuMemoryMonitor().start()

# *** Training loop
iteration = 1
start = time.time()
progress_bar = tqdm(total=cfg.iterations, desc="Training progress", initial=1)
while iteration < cfg.iterations + 1:
    camera_pool = scene.train_cameras.copy()
    random.shuffle(camera_pool)
    while camera_pool:
        # * Save preview images rendered at fixed viewpoints
        if iteration in cfg.preview_iters:
            raytracer.set_render_resolution(cam0.image_width, cam0.image_height)
            views = [
                ("train", cam0, scene.train_images, "preview_train_psnr", "preview_train_ssim")
            ]
            if scene.test_cameras:
                views.append(
                    ("test", test_cam0, scene.test_images, "preview_test_psnr", "preview_test_ssim")
                )

            psnrs, ssims = [], []

            for label, cam, images_dict, track_psnr_name, track_ssim_name in views:
                with torch.no_grad():
                    preview_render = raytracer(cam).clamp(0, 1)
                preview_target = images_dict[cam.image_name]

                preview_error = (preview_render - preview_target).abs()
                if preview_error.amax() > 0:
                    preview_error = preview_error / preview_error.amax()

                preview = torch.cat([preview_render, preview_target, preview_error], dim=-2)
                preview_path = os.path.join(cfg.model_path, f"preview_{label}_{iteration:05d}.png")
                executor.submit(save_image, preview, preview_path)

                preview_psnr = psnr(preview_render[None], preview_target[None]).item()
                writer.add_scalar(track_psnr_name, preview_psnr, iteration)
                psnrs.append(preview_psnr)

                preview_ssim = ssim(
                    preview_render[None], preview_target[None], downsample=False
                ).item()
                writer.add_scalar(track_ssim_name, preview_ssim, iteration)
                ssims.append(preview_ssim)

                if cfg.render_depth:
                    fb = raytracer.cuda_module.get_framebuffer()
                    depth_img = fb.output_depth.moveaxis(-1, 0) / fb.output_depth.amax()
                    depth_path = os.path.join(
                        cfg.model_path, f"preview_{label}_depth_{iteration:05d}.png"
                    )
                    executor.submit(save_image, depth_img, depth_path)

            print(
                f"{iteration} " + " ".join(f"{p:02.2f}" for p in psnrs),
                file=preview_psnr_log,
                flush=True,
            )
            print(
                f"{iteration} " + " ".join(f"{s:0.4f}" for s in ssims),
                file=preview_ssim_log,
                flush=True,
            )

        # * Acquire viewer lock
        if cfg.viewer:
            viewer.gaussian_lock.acquire()

        # * Increment SH degree
        if cfg.sh and iteration % cfg.sh_increment_interval == 0:
            gaussians.increment_sh_degree()

        # * Update learning rate schedules
        if cfg.lr_mean_final != cfg.lr_mean_init:
            gaussians.lr_mean.fill_(schedule_mean(iteration - 1))
        if cfg.lr_rotation_final != cfg.lr_rotation_init:
            gaussians.lr_rotation.fill_(schedule_rotation(iteration - 1))
        if cfg.lr_scale_final != cfg.lr_scale_init:
            gaussians.lr_scale.fill_(schedule_scale(iteration - 1))
        if cfg.lr_opacity_final != cfg.lr_opacity_init:
            gaussians.lr_opacity.fill_(schedule_opacity(iteration - 1))
        if cfg.lr_sh_dc_final != cfg.lr_sh_dc_init:
            gaussians.lr_sh_dc.fill_(schedule_dc(iteration - 1))
        if cfg.exposure_comp_enabled:
            raytracer.exposure_comp.set_lr(schedule_exposure_comp(iteration - 1))

        # * Warmup at half resolution
        if iteration <= cfg.half_res_iters:
            raytracer.set_render_resolution(cam0.image_width // 2, cam0.image_height // 2)
            images = scene.train_images_halfres
            batch_size = cfg.half_res_batch_size
        else:
            raytracer.set_render_resolution(cam0.image_width, cam0.image_height)
            images = scene.train_images
            batch_size = 1

        # *** Forward pass
        batch = [camera_pool.pop() for _ in range(min(batch_size, len(camera_pool)))]
        for camera in batch:
            render_unclamped = raytracer(camera)
            render = render_unclamped.clamp(0, 1)
            if cfg.exposure_comp_enabled:
                render_unclamped = raytracer.exposure_comp(render_unclamped, camera.image_name)

            # * Compute loss
            target = images[camera.image_name]
            loss = F.l1_loss(render_unclamped, target)
            if cfg.lambda_ssim > 0.0:
                from fused_ssim import fused_ssim

                ssim_score = fused_ssim(render_unclamped[None], target[None])
                loss = (1.0 - cfg.lambda_ssim) * loss + cfg.lambda_ssim * (1.0 - ssim_score)

            # *** Backward pass and optimization step
            raytracer.backward(loss / batch_size)
        raytracer.step()

        # * Scale decay
        if cfg.scale_decay < 1.0:
            gaussians.scale.add_(math.log(cfg.scale_decay))

        needs_rebuild = False

        # * Pruning
        if (
            cfg.pruning
            and iteration >= cfg.pruning_from_iter
            and iteration % cfg.pruning_interval == 0
        ):
            raytracer.prune(iteration)
            needs_rebuild = True

        # * Its valuable for preformance to perform full rebuilds periodically
        if cfg.rebuild_interval > 0 and iteration % cfg.rebuild_interval == 0:
            needs_rebuild = True

        # * Rebuild BVH
        if needs_rebuild:
            raytracer.cuda_module.rebuild_bvh()

        # * Log training curve
        training_l1 = F.l1_loss(render, target).item()
        training_psnr = psnr(render[None], target[None]).item()
        l1_avg += training_l1 / cfg.log_loss_interval
        psnr_avg += training_psnr / cfg.log_loss_interval
        if iteration % cfg.log_loss_interval == 0 or iteration == 1:
            print(
                f"{iteration:05d} {l1_avg:.8f} {psnr_avg:.8f}",
                file=losses_log,
                flush=True,
            )
            writer.add_scalar("l1_during_training", l1_avg, iteration)
            writer.add_scalar("psnr_during_training", psnr_avg, iteration)
            last_psnr_avg = psnr_avg
            l1_avg = 0.0
            psnr_avg = 0.0
            print(f"{iteration:05d} {gaussians.mean.shape[0]}", file=num_gaussians_log, flush=True)
            writer.add_scalar("num_gaussians", gaussians.mean.shape[0], iteration)

        # * Log traversal stats
        if iteration % cfg.log_stats_interval == 0:
            stats = raytracer.cuda_module.get_stats()
            hits = stats.num_gaussians_hit.float().mean() / cfg.log_stats_interval
            trav = stats.num_gaussians_accumulated.float().mean() / cfg.log_stats_interval
            stats.reset()
            print(
                f"{iteration} {hits:.2f} {trav:.2f}",
                file=traversal_stats_log,
                flush=True,
            )
            writer.add_scalar("avg_num_hit_per_ray", hits, iteration)
            writer.add_scalar("avg_num_accum_per_ray", trav, iteration)

        # * Log opacity and scale stats
        if iteration % cfg.log_stats_interval == 0 or iteration == 1:
            opacities = gaussians.opacity.detach().sigmoid()
            scales = gaussians.scale.detach().exp()
            opacity_mean = float(opacities.mean())
            opacity_std = float(opacities.std())
            scale_mean = float(scales.mean())
            scale_std = float(scales.std())
            scale_max = float(scales.amax(dim=1).mean())
            scale_min = float(scales.amin(dim=1).mean())
            anisotropy = scale_max / scale_min if scale_min != 0 else float("inf")
            anisotropy_std = float(scales.std() / scale_mean) if scale_mean != 0 else float("inf")
            print(
                f"{iteration} {opacity_mean:.4f}±{opacity_std:.4f} {scale_mean:.4f}±{scale_std:.4f} {anisotropy:.4f}±{anisotropy_std:.4f}",
                file=geometry_stats_log,
                flush=True,
            )
            writer.add_scalar("opacity_mean", opacity_mean, iteration)
            writer.add_scalar("opacity_std", opacity_std, iteration)
            writer.add_scalar("scale_mean", scale_mean, iteration)
            writer.add_scalar("scale_std", scale_std, iteration)
            writer.add_scalar("anisotropy", anisotropy, iteration)
            writer.add_scalar("anisotropy_std", anisotropy_std, iteration)

        # * Evaluate PSNR
        start_val = time.time()
        if iteration in cfg.test_iters:
            raytracer.set_render_resolution(cam0.image_width, cam0.image_height)
            print(f"{iteration:05d}", end="", file=psnr_log)
            print(f"{iteration:05d}", end="", file=ssim_log)
            scores = {}
            for split, cams, images in [
                ("train", scene.train_cameras, scene.train_images),
                ("test", scene.test_cameras, scene.test_images),
            ]:
                if not cams:
                    continue
                with torch.no_grad():
                    renders = [
                        (raytracer(cam).clamp(0, 1).cpu()[None] * 255).floor() / 255 for cam in cams
                    ]
                gts = [images[cam.image_name].cpu()[None] for cam in cams]
                renders = torch.cat(renders, dim=0)
                gts = torch.cat(gts, dim=0)
                psnr_split = mean(
                    [
                        psnr(renders[idx][None].cuda(), gts[idx][None].cuda()).item()
                        for idx in range(len(cams))
                    ]
                )
                ssim_split = mean(
                    [
                        ssim(
                            renders[idx][None].cuda(), gts[idx][None].cuda(), downsample=False
                        ).item()
                        for idx in range(len(cams))
                    ]
                )
                writer.add_scalar(f"psnr_eval_on_{split}", psnr_split, iteration)
                writer.add_scalar(f"ssim_eval_on_{split}", ssim_split, iteration)
                print(f" {psnr_split:02.2f}", end="", file=psnr_log)
                print(f" {ssim_split:0.4f}", end="", file=ssim_log)
                print(
                    f"[ITER {iteration}] {split.capitalize()} PSNR {psnr_split:02.2f} SSIM {ssim_split:0.4f}"
                )
            print(file=psnr_log, flush=True)
            print(file=ssim_log, flush=True)

            # * Log elapsed time
            start += time.time() - start_val  # * remove time spent for evaluation
            elapsed_time = time.gmtime(time.time() - start)
            timestamp = time.strftime("%H:%M:%S", elapsed_time)
            print(f"{iteration:05d} {timestamp}", file=time_log, flush=True)
            writer.add_scalar("elapsed_time", time.time() - start, iteration)

        # * Save gaussian .ply file
        if iteration in cfg.save_iters:
            print(f"[ITER {iteration}] Saving Gaussians")
            raytracer.save_safetensors(cfg.model_path, iteration)

        # * Release viewer lock
        if cfg.viewer:
            viewer.gaussian_lock.release()

        # * End training if max iterations reached
        iteration += 1
        if iteration > cfg.iterations:
            break

        # * Update progress
        progress_bar.update()
        progress_bar.set_postfix(
            {"gaussians": gaussians.mean.shape[0], "psnr": f"{last_psnr_avg:.2f}"}
        )

progress_bar.close()
writer.close()
timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
print(f"Training complete ({timestamp})")

# * Record peak GPU memory use
peak_mib = memory_monitor.stop()
memory_monitor.save(cfg.model_path)
print(f"Peak GPU memory use: {peak_mib / 1024:.2f} GB")

if cfg.viewer:
    viewer_thd.join()
