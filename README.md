# image-stitcher

High-performance image stitching for microscopy with GPU acceleration support.

## Features

- Automatic selection of CuPy, PyTorch, or NumPy backends
- GPU Acceleration: significant performance improvements for large-scale image registration  
- Graceful fallback to CPU processing if GPU is unavailable
- Works on Linux, our system of preference, and Windows

## Setup

### Pre-setup for linux (e.g. ubuntu)

On linux, ensure you have the necessary system dependencies installed and
conda environment setup. We provide a setup script for ubuntu 22.04
that does this automatically:

```bash
./setup_ubuntu_22_04.sh
```

This will set up a conda environment called `image-stitcher` with all the required dependencies,
including PyTorch with CUDA support for GPU acceleration.

### GPU Acceleration (Optional)

For maximum performance, install CuPy for NVIDIA GPU acceleration:

```bash
# Install CuPy for best GPU performance (Ubuntu only, compatible with Driver 535.x)
./setup_gpu_ubuntu_22_04.sh
```

The system will automatically detect and use the best available backend:
- **CuPy** (preferred) - Best performance for NVIDIA GPUs
- **PyTorch** (included) - Good performance, wider hardware support
- **NumPy** (fallback) - CPU-only, always available

### Test Your Installation

```bash
conda activate image-stitcher
python test_dependencies.py
```
### Pre-setup for Non-Ubuntu OS

For other environments, you will need to manually replicate the setup steps in `./setup_ubuntu_22_04.sh`.

We use [BaSiCPy](https://basicpy.readthedocs.io/en/latest/installation.html) which can be finicky.  Specifically
with respect to its jax depencency.  So far what has worked best is making sure to follow the exact suggestions
in their installation page with respect to installing jax separately, or just running `pip install basicpy`.

## Running The Stitcher
### Run the GUI

Activate the `image-stitcher` conda environment, then run `run_gui` script:
```
./run_gui
```

To run the gui manually without the helper script, you can run the following from this directory:
```
python -m image_stitcher.stitcher_gui
```

#### GPU Acceleration in GUI

The GUI includes a **Compute Backend** dropdown that allows you to select the tensor backend:

- **Auto (Recommended)** - Automatically selects the best available backend
- **NumPy (CPU)** - CPU-only processing using NumPy
- **PyTorch (GPU/CPU)** - Uses PyTorch with GPU acceleration if available
- **CuPy (NVIDIA GPU)** - Maximum performance with NVIDIA GPU acceleration

The dropdown shows the actual backend status (GPU/CPU).. Backend selection affects the registration process performance significantly.

## Running via the CLI

You can run registration from the command line via the stitcher_cli module and
its various configuration options. Run (with the conda `image-stitcher` environment activated):
```
python -m image_stitcher.stitcher_cli --help
```
to see the options and their documentation.

## Devtools

This repository is set up with [ruff](https://docs.astral.sh/ruff/) for linting
and formatting, and [mypy](https://mypy.readthedocs.io) for type checking. The
shell scripts in the dev directory can be used to invoke these tools and should
be run from the repository root.

You can install them with `pip install mypy ruff`
