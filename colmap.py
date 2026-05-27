import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pycolmap
import tyro
from PIL import Image
from tqdm import tqdm
from tyro.conf import arg


@dataclass
class CLI:
    source_path: Annotated[str, arg(aliases=["-s"])]
    camera: str = "OPENCV"
    gpu: bool = True
    delete_input: bool = False

cli = tyro.cli(CLI)

src = Path(cli.source_path)

assert (src / "input").is_dir(), f"Input directory not found: {src / 'input'}"
(src / "distorted" / "sparse").mkdir(parents=True, exist_ok=True)

# * Feature extraction
device = pycolmap.Device.cuda if cli.gpu else pycolmap.Device.cpu
pycolmap.extract_features(
    database_path=src / "distorted" / "database.db",
    image_path=src / "input",
    camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=pycolmap.ImageReaderOptions(camera_model=cli.camera),
    extraction_options=pycolmap.FeatureExtractionOptions(use_gpu=cli.gpu),
    device=device,
)

# * Feature matching
pycolmap.match_exhaustive(
    database_path=src / "distorted" / "database.db",
    device=device,
)

# * Incremental mapping (bundle adjustment)
maps = pycolmap.incremental_mapping(
    database_path=src / "distorted" / "database.db",
    image_path=src / "input",
    output_path=src / "distorted" / "sparse",
    options=pycolmap.IncrementalPipelineOptions(
        ba_global_function_tolerance=1e-6 # * speeds up bundle adjustment 
    ),
)
if not maps:
    logging.error("Incremental mapping failed. Exiting.")
    raise SystemExit(1)

# * Image undistortion
pycolmap.undistort_images(
    output_path=src,
    input_path=src / "distorted" / "sparse" / "0",
    image_path=src / "input",
    output_type="COLMAP",
)

# * Flatten sparse output into sparse/0
(src / "sparse" / "0").mkdir(parents=True, exist_ok=True)
for f in (src / "sparse").iterdir():
    if f.name == "0":
        continue
    shutil.move(str(f), str(src / "sparse" / "0" / f.name))

# * Cleanup
shutil.rmtree(src / "distorted")
shutil.rmtree(src / "stereo", ignore_errors=True)
for f in src.glob("*.sh"):
    f.unlink()
if cli.delete_input:
    shutil.rmtree(src / "input")
