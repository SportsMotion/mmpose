# SportsMotion MMPose

Fork of [MMPose](https://github.com/open-mmlab/mmpose) for fine-tuning custom keypoint models for sports motion analysis.

Currently training **COCO29** (17 body + 6 foot + 6 hand keypoints) using RTMPose, with the goal of extending to larger keypoint sets (e.g. 35+ keypoints) as needed for specific sports biomechanics use cases.

## What this repo does

- **Fine-tune RTMPose** on custom keypoint definitions using transfer learning from pretrained body models (Halpe26)
- **Extend to new keypoint formats** — the dataset config and training pipeline are designed to be adapted to any number of keypoints (29, 35, etc.) by modifying the dataset definition and annotation files
- **Export to ONNX** for deployment, with corrected preprocessing that matches mmpose's affine transforms
- **3D pose estimation** using RTMPose3D with multi-camera calibration support

## Environment Setup

This repo uses [uv](https://docs.astral.sh/uv/) for fast, reproducible environment setup. The original mmpose dependency chain (PyTorch + CUDA, mmcv, mmengine, mmdet, xtcocotools) is notoriously painful to install — `setup_env.sh` handles all of it in one script.

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.11
- CUDA 12.x toolkit (for building mmcv ops)

### Install

```bash
git clone git@github.com:SportsMotion/mmpose.git
cd mmpose
./setup_env.sh
```

This will:
1. Create a `.venv` with all Python dependencies (PyTorch, mmengine, mmdet, onnxruntime, etc.)
2. Build mmcv from source with CUDA ops
3. Install mmpose in editable mode
4. Verify everything works

```bash
source .venv/bin/activate
```

### What `setup_env.sh` solves

| Problem | Solution |
|---------|----------|
| PyTorch CUDA wheels need a special index | `pyproject.toml` uses `[[tool.uv.index]]` for pytorch-cu128 |
| numpy 2.x breaks xtcocotools ABI | Pinned `numpy>=1.22,<2.0` |
| mmcv doesn't declare build deps (pkg_resources) | Pins `setuptools<70` + builds with `--no-build-isolation` |
| mmcv needs CUDA toolkit to compile ops | Sets `CUDA_HOME` and `FORCE_CUDA=1` |
| albumentations 2.x breaks mmpose augmentation API | Pinned `albumentations>=1.0.0,<2.0.0` |
| Legacy checkpoints fail with PyTorch 2.6+ `weights_only=True` | Re-saves checkpoints during setup |

## Training

### Quick start (COCO29)

```bash
python tools/train.py \
  configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py \
  --work-dir work_dirs/rtmpose-l_coco29
```

### Training a different keypoint format

To train on a different number of keypoints (e.g. 35):

1. **Create a dataset definition** in `configs/_base_/datasets/` (see `coco29.py` as a template — define keypoint names, skeleton, sigmas, joint weights)
2. **Prepare annotations** in COCO format with your keypoints (see [Training Guide](docs/CUSTOM_RTM_TRAINING.md) for the exact JSON schema)
3. **Copy and modify the training config** — update `num_keypoints`, `data_root`, `ann_file`, and `metainfo` path
4. **Run training** with `tools/train.py`

The config uses a **frozen backbone + head-only training** strategy by default, which is fast (~1.7GB VRAM) and works well for adapting to new keypoint sets.

See [docs/CUSTOM_RTM_TRAINING.md](docs/CUSTOM_RTM_TRAINING.md) for the full guide on dataset format, expanding with synthetic data, and training options.

## Project Structure

```
configs/
  _base_/datasets/coco29.py                    # Keypoint definition (names, skeleton, sigmas)
  body_2d_keypoint/rtmpose/coco29/              # 2D RTMPose-L training config
  body_3d_keypoint/rtmpose3d/coco29/            # 3D RTMPose3D training config

scripts/
  inference/                                     # Video inference (PyTorch & ONNX, 2D & 3D)
  export/                                        # ONNX model export
  preprocessing_mmpose.py                        # Shared preprocessing (affine transforms, SimCC decode)

tools/
  train.py                                       # mmpose training entry point
  convert_multicam_to_coco29_3d.py               # Multi-camera 3D annotation converter

projects/rtmpose3d/                              # 3D pose estimator modules (head, loss, codec)

docs/
  CUSTOM_RTM_TRAINING.md                         # Dataset format & training guide
  ONNX_FIX_SUMMARY.md                           # ONNX preprocessing corrections

mmpose/                                          # Core library
  engine/hooks/freeze_backbone_hook.py           # Staged backbone freezing for transfer learning
  datasets/datasets/body3d/coco29_3d_dataset.py  # 3D dataset class
```

## Documentation

- [Training Guide](docs/CUSTOM_RTM_TRAINING.md) — dataset JSON format, adding synthetic data, training commands and config options
- [ONNX Preprocessing Fix](docs/ONNX_FIX_SUMMARY.md) — how we reduced ONNX inference error by 60%

## Based on

[MMPose v1.3.2](https://github.com/open-mmlab/mmpose) — Apache 2.0 License
