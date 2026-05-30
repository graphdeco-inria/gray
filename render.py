from gray.imports import *
from gray.prelude import *

from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class RenderCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]

    iteration: Annotated[int, arg(aliases=["-t"])] = -1
    splits: List[Literal["train", "test"]] = field(default_factory=lambda: ["test"])

    # * Optional changes to this image size
    width: Optional[int] = None
    height: Optional[int] = None
    fov_y: Optional[float] = None

    # Optional per-frame znear overrides
    znear_list: Optional[List[float]] = None
    znear: float = 0.0


# * Parse Config
cli, unknown_args = tyro.cli(RenderCLI, return_unknown_args=True)

# * Load the config from JSON and allow for Config overrides
saved_cli_path = os.path.join(cli.model_path, "config.json")
cfg = tyro.cli(Config, args=unknown_args, default=Config(**json.load(open(saved_cli_path, "r"))))

# * Make it possible to point directly to a gaussians file
if cli.model_path.endswith(".safetensors"):
    iteration = cfg.iteration
    save_path = cli.model_path
elif cli.iteration != -1:
    iteration = cli.iteration
    save_path = os.path.join(cli.model_path, f"gaussians_{iteration:05d}.safetensors")
else:
    iteration = search_for_max_iteration(cli.model_path)
    save_path = os.path.join(cli.model_path, f"gaussians_{iteration:05d}.safetensors")

# * Load the scene and raytracer
# * Fall back to the cameras saved in the model's cameras.json when the colmap
# * dataset is unavailable (e.g. rendering pretrained scenes without the data).
try:
    scene = SceneInfo.from_colmap(cfg)
except FileNotFoundError:
    print(
        f"Colmap dataset not found at '{cfg.source_path}'; "
        f"falling back to cameras saved in the model's cameras.json"
    )
    scene = SceneInfo.from_cameras_json(cli.model_path)
cam0 = scene.train_cameras[0]
raytracer = Raytracer.from_safetensors(
    cfg,
    save_path,
    cli.width or cam0.image_width,
    cli.height or cam0.image_height,
    inference_only=True,
)

# * Render images
print("Rendering iteration", iteration)
executor = ThreadPoolExecutor()
for split in cli.splits:
    dir_name = os.path.join(cli.model_path, split, f"{iteration:05d}")
    os.makedirs(os.path.join(dir_name, "renders"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "gt"), exist_ok=True)

    if split == "train":
        cameras, images = scene.train_cameras, scene.train_images
    elif split == "test":
        cameras, images = scene.test_cameras, scene.test_images

    if cli.znear_list is not None and len(cli.znear_list) != len(cameras):
        raise ValueError(
            f"Expected {len(cameras)} znear values for split '{split}', got {len(cli.znear_list)}"
        )

    futures = []

    for i, cam in enumerate(cameras):
        gt = images.get(cam.image_name)
        if cli.fov_y is not None:
            cam.fov_y = cli.fov_y

        with torch.no_grad():
            znear = cli.znear_list[i] if cli.znear_list is not None else cli.znear
            render = raytracer(cam, znear=znear).clamp(0, 1)

        futures.append(
            executor.submit(save_image, render, os.path.join(dir_name, "renders", f"{i:05d}.png"))
        )
        # * Ground truth is only available when the dataset images are present.
        if gt is not None:
            futures.append(
                executor.submit(save_image, gt, os.path.join(dir_name, "gt", f"{i:05d}.png"))
            )

    for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Saving {split} images"):
        pass
