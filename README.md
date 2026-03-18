# MMPose — COCO29 Fork

Custom fork of [MMPose](https://github.com/open-mmlab/mmpose) extended with **29-keypoint pose estimation** (COCO body + feet + hands) for sports motion analysis.

## What's different from upstream

- **COCO29 dataset** (17 body + 6 foot + 6 hand keypoints)
- **RTMPose-L training config** with transfer learning from Halpe26
- **RTMPose3D config** for 3D pose estimation
- **FreezeBackboneHook** for staged transfer learning
- **ONNX export & inference** with corrected preprocessing
- **Multi-camera 3D annotation tools**

## Quick Start

```bash
# Setup environment (requires uv, Python 3.11, CUDA 12.x)
./setup_env.sh
source .venv/bin/activate

# Train
python tools/train.py \
  configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py \
  --work-dir work_dirs/rtmpose-l_coco29
```

## Project Structure

```
configs/
  _base_/datasets/coco29.py                              # 29-keypoint dataset definition
  body_2d_keypoint/rtmpose/coco29/                        # 2D training config
  body_3d_keypoint/rtmpose3d/coco29/                      # 3D training config

scripts/
  inference/                                               # Video inference (PyTorch & ONNX)
    inference_coco29.py, inference_coco29_onnx.py
    inference_rtmpose3d.py, inference_rtmpose3d_onnx.py
  export/                                                  # Model export to ONNX
    rtm2onnx.py, rtm3d2onnx.py
  preprocessing_mmpose.py                                  # Shared preprocessing utilities

tools/
  train.py                                                 # Training script (from mmpose)
  convert_multicam_to_coco29_3d.py                         # Multi-camera 3D data conversion
  example_convert_multicam.py                              # Example usage

projects/rtmpose3d/                                        # 3D pose estimator modules

docs/
  CUSTOM_RTM_TRAINING.md                                   # Dataset format & training guide
  ONNX_FIX_SUMMARY.md                                     # ONNX preprocessing fix details

mmpose/                                                    # Core library (+ custom hooks/datasets)
```

## Documentation

- **[Training Guide](docs/CUSTOM_RTM_TRAINING.md)** — dataset format, expanding with synthetic data, training commands
- **[ONNX Fix](docs/ONNX_FIX_SUMMARY.md)** — preprocessing corrections for ONNX inference

## Based on

[MMPose v1.3.2](https://github.com/open-mmlab/mmpose) — Apache 2.0 License
