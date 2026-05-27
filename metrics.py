from gray.imports import *

import warnings
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from piq import LPIPS


@dataclass
class MetricsCLI:
    model_paths: Annotated[List[str], arg(aliases=["-m"])]
    batch_size: int = 1


# * Parse Config
cli = tyro.cli(MetricsCLI)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lpips_fn = LPIPS(reduction="none").cuda()


class ImagePairDataset(Dataset):
    def __init__(self, renders_dir: Path, gt_dir: Path):
        self.renders_dir = renders_dir
        self.gt_dir = gt_dir
        self.fnames = sorted(os.listdir(renders_dir))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        render = read_image(str(self.renders_dir / fname), ImageReadMode.RGB).float() / 255.0
        gt = read_image(str(self.gt_dir / fname), ImageReadMode.RGB).float() / 255.0
        return render, gt, fname


avg_metrics = {}
per_image_metrics = {}

for model_path in cli.model_paths:
    print("Scene:", model_path)
    avg_metrics[model_path] = {}
    per_image_metrics[model_path] = {}

    test_dir = Path(model_path) / "test"

    for method in sorted(os.listdir(test_dir)):
        print("Iterations:", method)
        method_dir = test_dir / method
        dataset = ImagePairDataset(method_dir / "renders", method_dir / "gt")
        loader = DataLoader(dataset, batch_size=cli.batch_size, num_workers=4, pin_memory=True)

        ssim_scores = []
        psnr_scores = []
        lpips_scores = []
        image_names = []

        for renders_batch, gts_batch, fnames_batch in tqdm(loader, desc="Metric evaluation progress"):
            renders_batch = renders_batch.cuda()
            gts_batch = gts_batch.cuda()

            ssim_scores.extend(ssim(renders_batch, gts_batch, downsample=False, reduction="none").tolist())
            psnr_scores.extend(psnr(renders_batch, gts_batch, reduction="none").tolist())
            lpips_scores.extend(lpips_fn(renders_batch, gts_batch).tolist())
            image_names.extend(fnames_batch)

        ssim_mean = torch.tensor(ssim_scores).mean().item()
        psnr_mean = torch.tensor(psnr_scores).mean().item()
        lpips_mean = torch.tensor(lpips_scores).mean().item()

        print("  SSIM : {:>12.7f}".format(ssim_mean))
        print("  PSNR : {:>12.7f}".format(psnr_mean))
        print("  LPIPS: {:>12.7f}".format(lpips_mean))

        avg_metrics[model_path][method] = {"SSIM": ssim_mean, "PSNR": psnr_mean, "LPIPS": lpips_mean}
        per_image_metrics[model_path][method] = {
            "SSIM": {name: val for name, val in zip(image_names, ssim_scores)},
            "PSNR": {name: val for name, val in zip(image_names, psnr_scores)},
            "LPIPS": {name: val for name, val in zip(image_names, lpips_scores)},
        }

    # * Save results
    with open(model_path + "/results.json", "w") as fp:
        json.dump(avg_metrics[model_path], fp, indent=True)
    with open(model_path + "/per_view.json", "w") as fp:
        json.dump(per_image_metrics[model_path], fp, indent=True)
