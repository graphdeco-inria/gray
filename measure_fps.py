from gray.imports import *
from gray.prelude import *

from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class RenderCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]
    context: Annotated[Literal["train", "test"], arg(aliases=["-x"])] = "test"

    iteration: Annotated[int, arg(aliases=["-t"])] = -1

    # * Optional changes to this image size
    width: Optional[int] = None
    height: Optional[int] = None
    fov_y: Optional[float] = None
    scale: Optional[float] = None


# * Parse Config
cli, unknown_args = tyro.cli(RenderCLI, return_unknown_args=True)

# * Load the config from JSON and allow for Config overrides
saved_cli_path = os.path.join(cli.model_path, "config.json")
cfg = tyro.cli(Config, args=unknown_args, default=Config(**json.load(open(saved_cli_path, "r"))))

# * Make it possible to point directly to a gaussians file
if cli.model_path.endswith(".safetensors"):
    iteration = cfg.iteration
    save_path = cli.model_path
else:
    iteration = search_for_max_iteration(cli.model_path)
    save_path = os.path.join(cli.model_path, f"gaussians_{iteration:05d}.safetensors")

# * Load the scene and raytracer
scene = SceneInfo.from_colmap(cfg)
cam0 = scene.train_cameras[0]
raytracer = Raytracer.from_safetensors(
    cfg,
    save_path,
    int((cli.width or cam0.image_width) * (cli.scale or 1.0)),
    int(cli.height or cam0.image_height * (cli.scale or 1.0)),
)

# * Warmup caches
if cli.context == "train":
    cameras, images = scene.train_cameras, scene.train_images
else:
    cameras, images = scene.test_cameras, scene.test_images

for cam in cameras:
    with torch.no_grad():
        raytracer(cam)
# * Measure FPS
print("Measuring FPS at iteration", iteration)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
start.record()
for i, cam in enumerate(cameras):
    target = images[cam.image_name]
    if cli.fov_y is not None:
        cam.fov_y = cli.fov_y

    if cli.context == "train":
        render_unclamped = raytracer(cam)
        loss = F.l1_loss(render_unclamped, target)
        if cfg.lambda_ssim > 0.0:
            from fused_ssim import fused_ssim

            ssim_score = fused_ssim(render_unclamped[None].clamp(0), target[None].clamp(0))
            loss = (1.0 - cfg.lambda_ssim) * loss + cfg.lambda_ssim * (1.0 - ssim_score)
        raytracer.backward_and_step(loss)
    else:
        with torch.no_grad():
            raytracer(cam)
end.record()
torch.cuda.synchronize()
secs = start.elapsed_time(end) / 1000.0
print(f"Average {cli.context} FPS: {len(cameras) / secs:.2f}")
if cli.context == "test":
    with open(os.path.join(cli.model_path, f"fps.csv"), "w") as f:
        f.write(f"{len(cameras) / secs:.2f}\n")

    fps_json_path = os.path.join(cli.model_path, "fps_by_scale.json")
    if os.path.exists(fps_json_path):
        with open(fps_json_path, "r") as f:
            fps_data = json.load(f)
    else:
        fps_data = {}

    scale_key = str(cli.scale or 1.0)
    fps_data[scale_key] = len(cameras) / secs

    with open(fps_json_path, "w") as f:
        json.dump(fps_data, f, indent=4)
