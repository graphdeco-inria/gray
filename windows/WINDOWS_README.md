
# Windows Instructions

Windows is supported and was tested with Visual Studio. 
PowerShell versions of all bash scripts are available in the `windows/` and  `scripts/windows/` directories.
We no longer recommend WSL.

Note that `pycolmap` does not support GPU on Windows.
If you need to run COLMAP yourself we recommend you install it directly and use the script in the 3DGS codebase.

Otherwise all features should be supported; please report any issues you encounter.

## Installation

Using the [`uv`](https://github.com/astral-sh/uv) package manager (installable with `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`), run
```bash
git submodule update --init --recursive   # pull submodules
.\windows\install.ps1                     # create environment & install dependencies
.\.venv\Scripts\Activate.ps1              # activate environment
.\windows\make.ps1                        # compile the cuda raytracer into `build/`
```

## Running code

Follow the Linux instructions in the README but use the `.ps1` scripts instead of their `.sh` equivalents.

## Troubleshooting

As usual make sure `cl`, `nvcc`, `cmake` are exposed.

If `fused-ssim` fails to compile in `install.sh` try replacing `--torch-backend=auto` with your specific cuda version e.g. `--torch-backend cu128`.