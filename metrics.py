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


from gray.imports import *
from gray.utils import set_seeds
from gray.config import Config

from piq import psnr, ssim, LPIPS
import warnings
from concurrent.futures import ThreadPoolExecutor

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lpips = LPIPS()


@dataclass
class MetricsCLI:
    model_paths: Annotated[List[str], tyro.conf.arg(aliases=["-m"])]


# * Parse Config
cfg = tyro.cli(MetricsCLI)
set_seeds(0)

full_dict = {}
per_view_dict = {}
full_dict_polytopeonly = {}
per_view_dict_polytopeonly = {}

for scene_dir in cfg.model_paths:
    print("Scene:", scene_dir)

    # * Initialize per-scene dictionaries
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):
        print("Iterations:", method)

        # * Initialize per-method dictionaries
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}
        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"

        # * Read images
        renders = []
        gts = []
        image_names = []

        def load_image_pair(fname):
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            render_tensor = TF.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
            gt_tensor = TF.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
            return render_tensor, gt_tensor, fname

        fnames = os.listdir(renders_dir)
        renders = []
        gts = []
        image_names = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_image_pair, fnames))
            for render_tensor, gt_tensor, fname in results:
                renders.append(render_tensor)
                gts.append(gt_tensor)
                image_names.append(fname)

        # * Evaluate metrics
        ssim_scores = []
        psnr_scores = []
        lpips_scores = []
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssim_scores.append(ssim(renders[idx], gts[idx], downsample=False).item())
            psnr_scores.append(psnr(renders[idx], gts[idx]).item())
            lpips_scores.append(lpips(renders[idx], gts[idx]).item())

        # * Compute averages
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssim_scores).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnr_scores).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpips_scores).mean(), ".5"))

        # * Store average metrics
        full_dict[scene_dir][method].update(
            {
                "SSIM": torch.tensor(ssim_scores).mean().item(),
                "PSNR": torch.tensor(psnr_scores).mean().item(),
                "LPIPS": torch.tensor(lpips_scores).mean().item(),
            }
        )

        # * Store per-view metrics
        per_view_dict[scene_dir][method].update(
            {
                "SSIM": {
                    name: value
                    for value, name in zip(torch.tensor(ssim_scores).tolist(), image_names)
                },
                "PSNR": {
                    name: value
                    for value, name in zip(torch.tensor(psnr_scores).tolist(), image_names)
                },
                "LPIPS": {
                    name: value
                    for value, name in zip(torch.tensor(lpips_scores).tolist(), image_names)
                },
            }
        )

    # * Save results
    with open(scene_dir + "/results.json", "w") as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", "w") as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
