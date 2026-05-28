
# GRay: Ray Tracing 3D Gaussians Near the Speed of Splats
Yohan Poirier-Ginter, Jean-François Lalonde, George Drettakis

[Website](https://repo-sam.inria.fr/nerphys/gray/) | [Paper](https://repo-sam.inria.fr/nerphys/gray/content/paper.pdf) | [Video](https://www.youtube.com/watch?v=ei-9wyzlaho) | [NERPHYS](https://project.inria.fr/nerphys/) | [Pretrained Models](https://repo-sam.inria.fr/nerphys/gray/pretrained.html) 

GRay is a fast ray tracer for 3D Gaussians that can be used as a ray-tracing-based alternative to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), much like [3DGRT](https://github.com/nv-tlabs/3dgrut). By leveraging [dense initialization](https://github.com/CompVis/EDGS) and other techniques including methods developped in our [previous project](https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/), GRay optimizes nearly 10× faster than 3DGRT on an RTX 4090.


## Installation
Using the [`uv`](https://github.com/astral-sh/uv) package manager (installable with `curl -LsSf https://astral.sh/uv/install.sh | sh`), run
```bash
git submodule update --init --recursive   # pull submodules
bash install.sh                           # create environment & install dependencies
source .venv/*/activate                   # activate environment
bash ./make.sh                            # compile the cuda raytracer into `build/`
```

This codebase requires a graphics card supporting OptiX 8 and a local CUDA 12 toolkit installation exposing `nvcc`.

We are working on Windows support using WSL. Please report any issues if you attempt working on Windows.


## Viewing Pretrained Models
The pretrained models are [available online](https://repo-sam.inria.fr/nerphys/gray/pretrained.html) and can be downloaded in batch with `bash scripts/download_all_pretrained_scenes.sh`. You can open them in the interactive viewer with

```bash
MODEL_DIR=output/pretrained/bicycle

python view.py -m $MODEL_DIR
```


## Easy Setup
This section explains how to easily reproduce the results from our paper.

First, run
```bash 
bash scripts/full_dataset_preparation.sh
```
to download, resize, and preprocess all 13 scenes used for evaluation and place them in `data/`.

Then run
```bash
bash scripts/run_all_scenes.sh output/
```
to train and evaluate all scenes and put them into `output/`.

You can then run 
```bash 
python collect_results.py output/
```
to collect all metrics in a table. 

Here are the expected results from the latest version of the code:

| PSNR | SSIM | LPIPS | Time | FPS |
| ---: | ---: | ---: | ---: | ---: |
| 26.45 | 0.818 | 0.238 | 05:30 | 250 |


## Step-by-Step Workflow
This section explains how to run scenes step-by-step. You can skip it if you followed the automated reproduction steps above.

### 1. Download Scenes
You can download the [MipNerf360](https://jonbarron.info/mipnerf360/), [Tanks and Temples](https://www.tanksandtemples.org/), and [Deep Blending](https://github.com/Phog/DeepBlending) scenes used for benchmarking with

```bash 
bash scripts/download_all_scenes.sh
```
This will place them in `data/`; for example, `data/360_v2/bicycle` will contain the files for the MipNeRF 360 bicycle scene.

You can also use any COLMAP scene or create your own with the provided `convert.py` utility. Its usage is explained in the [3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting).

### 2. Resize Images
This codebase expects your images to already be sized to the correct resolution in `.png`. Resizing can be done with the preprocessing script
```bash
SCENE_DIR=data/360_v2/bicycle

python resize.py -s $SCENE_DIR -y
```
which will downsize your images by factors of 2, 4, and 8 while also limiting their size to max 1600 pixels like 3DGS does. You can resize all benchmarking scenes with
```bash 
bash scripts/resize_all_scenes.sh
```
which will produce the subdirectories `images_1` (original size clamped to max 1600 pixels), `images_2` (half resolution), etc.

### 3. Create the Dense Initialization Point Cloud
This project uses [dense initialization](https://github.com/CompVis/EDGS) for its initial point cloud. You can create these point clouds for any scene with:

```bash
SCENE_DIR=data/360_v2/bicycle
INDOORS_OR_OUTDOORS=indoors

python third_party/edgs.py -s $SCENE_DIR --roma_model $INDOORS_OR_OUTDOORS
```
Here you must select which type of scene you are dealing with (`indoors` or `outdoors`) to choose the correct [RoMA](https://github.com/Parskatt/RoMa) network used for dense matching. The point cloud will be saved to `$SCENE_DIR/point_cloud.safetensors`.

You can create point clouds for all benchmarking scenes with
```bash 
bash scripts/create_all_dense_point_clouds.sh
```

### 4. Train, Render, and Evaluate
The configuration is mostly unchanged from 3DGS, with some minor differences.
Run a full training and evaluation pass with:
```bash
SCENE_DIR=data/360_v2/bicycle
DOWNSAMPLING_LEVEL=4
OUTPUT_DIR=out/bicycle

python train.py -m $OUTPUT_DIR -s $SCENE_DIR -r $DOWNSAMPLING_LEVEL 
python render.py -m $OUTPUT_DIR
python metrics.py -m $OUTPUT_DIR
python measure_fps.py -m $OUTPUT_DIR
```

The `run.sh` utility chains all steps and takes the output directory as its first argument, e.g.
```bash
bash run.sh $OUTPUT_DIR -s $SCENE_DIR -r $DOWNSAMPLING_LEVEL
```
The viewer can also be enabled during training with the `--viewer` flag.


## Details
This section clarifies technical details and additional features.

### Running COLMAP
You can use your own COLMAP files under the standard layout expected by 3DGS. 

We also provide a script for running COLMAP from `pycolmap-cuda12` (already installed). First, place your images under `data/$SCENE/input` and then run
``` 
python run_colmap.py -s data/$SCENE
``` 
GPU support is only for Linux; on Windows you can either install the CPU-only version `pycolmap`, or use the script from the 3DGS codebase.

Once COLMAP has run successfully, you will need to resize the images and run dense initialization as explained earlier. 

### Memory Use
We use per-pixel linked lists to store intersected Gaussians and data for the backward pass. You can control their size with the flags `--ppll_forward_size` and `--ppll_backward_size`. You might need to increase the defaults for your own scenes, or you might be able to reduce them. Running the standard scenes with the current settings requires 24GB of VRAM.

### PyTorch Integration
While most code is CUDA-side, including the loss computation and optimizer step, nearly all memory is allocated in tensors and exposed to PyTorch via [pybind](https://github.com/pybind/pybind11). As such, most configuration can be adjusted via the command line without recompiling, and many intermediate results can be inspected in Python for debugging.

The main ray tracer's CUDA module exposes objects that group relevant data tensors. For instance, the camera can be inspected with
```python
camera = raytracer.cuda_module.get_camera()
```
and data is provided to the CUDA module by modifying its values in-place.

Note that the backward pass relies on the data from the forward pass staying unmodified (camera, framebuffer, etc.).

### Quality Presets
Preset configurations are available: adding the flag `-c configs/lq.json` selects a lower level of quality, and the flag `-c configs/hq.json` selects a high level of quality. The default quality level is `mq` (medium quality). The hyperparameters used are detailed in the paper.

### Compatibility with 3DGS
The gaussians produced by this method are incompatible with 3DGS; in theory, the differences could be resolved by modifying both methods (refer to the paper for a short discussion on page 14), but this has not been done in practice. The file format was changed to `.safetensors` which is simpler and faster.

### Evaluation
Metric computation was moved to the [PIQ](https://github.com/photosynthesis-team/piq) library since the LPIPS metric was incorrect in the original 3DGS codebase. PSNRs and SSIM scores were verified to match.

### MLP Support
This codebase also features MLP support, although we did not use MLPs in the paper.

Two types of MLPs are supported:
- Pre-processing MLPs (`pre_mlp`) which transform features into per-gaussian channels, before they are rendered into pixels.
- Post-processing MLPs (`post_mlp`) which transform per-pixel channels into a final color.

If you wish to use [`tinycudann`](https://github.com/nvlabs/tiny-cuda-nn), you can optionally install it with `uv sync --extra tcnn` and enable it with `--tcnn`.

### Remote Viewer
The viewer can be used remotely, in which case a server renders the images and delivers them to the client via Websocket. Launch the server with
```
python view.py --server -m $MODEL_DIR
```
On the client, you can install the minimal required dependencies with `uv venv && source .venv/*/activate && uv pip install -r viewer/requirements.txt` and then run 
```
python view.py --client $SERVER_IP
```
The client does not require a GPU and all platforms are supported (Linux/Windows/Max). However note the remote viewer has low framerates due to network overhead.

### Fast Viewer
Besides the Python viewer, which is designed for ease of development, a faster C++ viewer is also provided. It can be included during the build with
```
bash make.sh -DGRAY_BUILD_FAST_VIEWER=ON
```
and launched with 
```bash
MODEL_DIR=output/pretrained/bicycle

build/fast_viewer -m $MODEL_DIR
```
The fast viewer does not include a UI and provides only minimal camera keyboard and mouse controls. Its performance can also be measured using the `--benchmark` and `--benchmark-test-cameras` flags.

### Depth Map Rendering
You can render depth maps with `--render_depth`.

### Bugfixes
We fixed a minor bug in how the bin size was computed for initialization binning. As such, the default value for `init_bin_size` differs from the value reported in the paper and quantitative results may differ by negligible amounts (< 0.1 dB).


## Troubleshooting
Please report any problems you encounter with installation in the GitHub issues.

If your scene is very large, you might get better results by disabling initialization binning with `--no_init_binning`.

This code was designed for scenes with around 200-300 images and pinhole cameras; we are working on support for larger scenes. Alternative camera models are not currently provided but should be straightforward to implement.

You will likely encounter floaters which are a known limitation of dense initialization.


## License
The original code in this repository is licensed under the MIT License.

Some files are derived from third-party sources and remain under their original licenses. Those files include license notices in their headers.

This includes, but is not limited to:
- The [GraphDeco viewer](https://github.com/graphdeco-inria/graphdecoviewer) which is under Apache 2.0.
- The [dense initialization](https://github.com/CompVis/EDGS) script `third_party/edgs.py` under the copyright license of its original authors.


## BibTeX
```
@article{poirierginter2026gray,
    author = {Poirier-Ginter, Yohan and Lalonde, Jean-Fran\c{c}ois and Drettakis, George},
    title = {GRay: Ray Tracing 3D Gaussians Near the Speed of Splats},
    year = {2026},
    issue_date = {May 2026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {9},
    number = {1},
    url = {https://doi.org/10.1145/3804496},
    doi = {10.1145/3804496},
    journal = {Proc. ACM Comput. Graph. Interact. Tech.},
    month = may,
    articleno = {14},
    numpages = {19}
}
```


## Acknowledgments
Thanks to [Jeffrey Hu](https://jefequien.github.io/) for helping with the code and pointing us towards dense initialization.

Thanks to [Ishaan Shah](https://ishaanshah.xyz/) for the Gaussian Viewer.

> This research was co-funded by the European Union (EU) ERC Advanced Grant NERPHYS No 101141721. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the EU or the European Research Council. Neither the EU nor the granting authority can be held responsible for them. Experiments presented in this paper were carried out using the Grid'5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations. This research was also supported by NSERC grant RGPIN-2020-04799 and the Digital Research Alliance Canada. The authors are grateful to Adobe and NVIDIA for generous donations.
