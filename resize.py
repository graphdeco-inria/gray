import os
import sys
from pathlib import Path
from typing import List
from PIL import Image
import tyro
from tyro.conf import arg
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Annotated
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class Config:
    source_path: Annotated[str, arg(aliases=["-s"])]
    images_dir: Annotated[str, arg(aliases=["-i"])] = "images"
    yes: Annotated[bool, arg(aliases=["-y"])] = (
        False  # * allow overwriting existing directories without prompt
    )

    downsizing_factors: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    max_size: int = 1600

def resize_image(img_name, src_dir, out_dirs, downsizing_factors, max_size):
    img_path = src_dir / img_name
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return f"Skipping {img_name}: {e}"
    w, h = img.size
    for factor, out_dir in zip(downsizing_factors, out_dirs):
        scale = 1.0 / factor
        target_w = int(w * scale)
        target_h = int(h * scale)

        # * Enforce max_size while preserving aspect ratio
        if target_w > max_size or target_h > max_size:
            if w >= h:
                new_w = max_size
                new_h = int(h * (max_size / w))
            else:
                new_h = max_size
                new_w = int(w * (max_size / h))
        else:
            new_w, new_h = target_w, target_h

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        out_path = (out_dir / img_name).with_suffix(".png")
        img_resized.save(out_path)


cfg = tyro.cli(Config)

args = cfg
src_dir = Path(args.source_path) / args.images_dir
out_dirs = [src_dir.parent / f"{args.images_dir}_{factor}" for factor in args.downsizing_factors]

# * Check for existing output directories
existing = [str(d) for d in out_dirs if d.exists()]
if existing and not args.yes:
    print(f"The following output directories already exist: {', '.join(existing)}")
    resp = input("Overwrite them? [y/N]: ").strip().lower()
    if resp != "y":
        print("Aborted.")
        sys.exit(1)

# * Remove existing output directories
for d in out_dirs:
    if d.exists():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()

# * Create output directories
for d in out_dirs:
    d.mkdir(parents=True, exist_ok=True)

# * Process images
img_list = [img_name for img_name in os.listdir(src_dir) if (src_dir / img_name).is_file()]
with ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(resize_image, img_name, src_dir, out_dirs, args.downsizing_factors, args.max_size): img_name
        for img_name in img_list
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing images"):
        err = future.result()
        if err:
            print(err)
