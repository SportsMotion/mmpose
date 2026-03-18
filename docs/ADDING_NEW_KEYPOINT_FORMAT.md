# Adding a New Keypoint Format

Step-by-step guide to training an RTMPose model on a custom keypoint definition, using a hypothetical **COCO52** (52 keypoints) as an example. The same process works for any number of keypoints.

## Overview

There are 4 files to create/modify, then you train and export:

```
1. configs/_base_/datasets/coco52.py              ← keypoint definition
2. Annotation JSONs (coco52_train.json, etc.)      ← your dataset
3. configs/body_2d_keypoint/rtmpose/coco52/...py   ← training config
4. Train → Export to ONNX → Inference
```

---

## Step 1: Dataset Definition

Create `configs/_base_/datasets/coco52.py`. This tells mmpose what your keypoints are, how they connect, and how to evaluate them.

```python
dataset_info = dict(
    dataset_name='coco52',
    paper_info=dict(
        author='Your Name',
        title='COCO52 custom keypoints',
        container='Custom',
        year='2026',
        homepage='',
    ),

    # Define every keypoint
    keypoint_info={
        0: dict(name='nose',           id=0,  color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye',       id=1,  color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye',      id=2,  color=[51, 153, 255], type='upper', swap='left_eye'),
        # ... define all 52 keypoints ...
        # For left/right pairs, set swap= to the other side's name
        # For midline keypoints (nose, neck, etc.), set swap=''
        51: dict(name='right_pinky_tip', id=51, color=[255, 128, 0], type='upper', swap='left_pinky_tip'),
    },

    # Define skeleton connections (1-indexed!)
    skeleton_info={
        0: dict(link=('nose', 'left_eye'),         id=0, color=[51, 153, 255]),
        1: dict(link=('nose', 'right_eye'),        id=1, color=[51, 153, 255]),
        # ... all bone connections ...
    },

    # Per-keypoint loss weight (higher = model focuses more on this keypoint)
    joint_weights=[
        1.0, 1.0, 1.0, ...  # 52 values
    ],

    # Per-keypoint sigma for OKS evaluation
    # Smaller = tighter evaluation threshold (use ~0.025 for precise points like fingertips,
    # ~0.079 for mid-body joints, ~0.107 for hips)
    sigmas=[
        0.026, 0.025, 0.025, ...  # 52 values
    ],
)
```

### Key rules

- **`swap`**: Must be the exact `name` of the mirrored keypoint. This is used for horizontal flip augmentation. Midline keypoints (nose, neck) get `swap=''`.
- **`skeleton_info`**: Uses **keypoint names** (not indices). These are only for visualization.
- **`joint_weights`**: Higher values make the loss penalize errors on those keypoints more. Use 1.0 for standard joints, 1.2-1.5 for important ones you want higher accuracy on.
- **`sigmas`**: Controls the OKS evaluation sensitivity per keypoint. Smaller sigma = stricter. Use similar values from COCO/Halpe for body parts, smaller values (0.025) for precise landmarks like fingertips.

---

## Step 2: Prepare Your Dataset

Your annotation files must be in COCO keypoint format.

### Directory layout

```
/path/to/your/dataset/
├── coco52_train.json
├── coco52_val.json
├── train/
│   ├── image_0001.jpg
│   └── ...
└── val/
    ├── image_5001.jpg
    └── ...
```

### Annotation JSON structure

```json
{
  "images": [
    {
      "id": 0,
      "file_name": "train/image_0001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 1,
      "keypoints": [x0, y0, v0, x1, y1, v1, ..., x51, y51, v51],
      "num_keypoints": 52,
      "bbox": [x, y, width, height],
      "area": 50000.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person",
      "keypoints": ["nose", "left_eye", ..., "right_pinky_tip"],
      "skeleton": [[1, 2], [1, 3], ...]
    }
  ]
}
```

### Keypoint array format

The `keypoints` field is a flat array: `[x0, y0, v0, x1, y1, v1, ...]`

- **x, y**: pixel coordinates (absolute, not normalized)
- **v** (visibility): `0` = not labeled, `1` = labeled but occluded, `2` = labeled and visible

For 52 keypoints, this array has **156 values** (52 * 3).

### Bbox format

`[x, y, width, height]` — top-left corner, not center.

If you only have keypoints, compute the bbox:

```python
def keypoints_to_bbox(kps, padding=1.2):
    xs = [kps[i] for i in range(0, len(kps), 3) if kps[i+2] > 0]
    ys = [kps[i+1] for i in range(0, len(kps), 3) if kps[i+2] > 0]
    if not xs:
        return [0, 0, 0, 0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w, h = (x_max - x_min) * padding, (y_max - y_min) * padding
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    return [cx - w/2, cy - h/2, w, h]
```

### Categories keypoints and skeleton

The `keypoints` list in `categories` must be in the **exact same order** as your keypoint indices (index 0 = first name, etc.). The `skeleton` uses **1-indexed** pairs.

---

## Step 3: Training Config

Create `configs/body_2d_keypoint/rtmpose/coco52/rtmpose-l_8xb256-420e_coco52-256x192.py`.

The easiest way: **copy the coco29 config and change these values**:

```bash
mkdir -p configs/body_2d_keypoint/rtmpose/coco52
cp configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py \
   configs/body_2d_keypoint/rtmpose/coco52/rtmpose-l_8xb256-420e_coco52-256x192.py
```

Then edit the new file — here's every line that needs to change:

### 3a. Number of keypoints

```python
num_keypoints = 52  # was 29
```

### 3b. Dataset paths

```python
data_root = '/path/to/your/dataset/'  # where your images and JSONs live
```

### 3c. Annotation filenames

In `train_dataloader` and `val_dataloader`:

```python
ann_file='coco52_train.json',   # was coco29_train.json
```

```python
ann_file='coco52_val.json',     # was coco29_val.json
```

### 3d. Metainfo path

In both dataloaders:

```python
metainfo=dict(from_file='configs/_base_/datasets/coco52.py'),  # was coco29.py
```

### 3e. Pretrained weights

The `load_from` line loads a pretrained checkpoint. The backbone weights will transfer, but the **head will be randomly initialized** because the number of output keypoints changed.

```python
# Any RTMPose-L pretrained checkpoint works — the backbone is the same
load_from = 'checkpoints/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth'
```

### That's it

Everything else (model architecture, augmentation pipeline, optimizer, scheduler, hooks) stays the same. The `out_channels=num_keypoints` in the head config automatically picks up the new value.

---

## Step 4: Train

```bash
python tools/train.py \
  configs/body_2d_keypoint/rtmpose/coco52/rtmpose-l_8xb256-420e_coco52-256x192.py \
  --work-dir work_dirs/rtmpose-l_coco52
```

### Useful overrides

```bash
# Fewer epochs for testing
--cfg-options train_cfg.max_epochs=10

# Smaller batch size if OOM
--cfg-options train_batch_size=64

# Different dataset location
--cfg-options data_root=/data/coco52/

# Unfreeze backbone after 50 epochs (fine-tune full model)
--cfg-options freeze_backbone_epochs=50
```

### Monitor

```bash
tensorboard --logdir work_dirs/rtmpose-l_coco52
```

### Output

```
work_dirs/rtmpose-l_coco52/
├── best_PCK_epoch_XX.pth    ← best checkpoint (use this)
├── epoch_XX.pth             ← periodic checkpoints
└── YYYYMMDD_HHMMSS/
    └── vis_data/
        └── scalars.json     ← loss, PCK, lr per step
```

---

## Step 5: Export to ONNX

Edit `scripts/export/rtm2onnx.py` — update the `main()` function with your paths:

```python
def main():
    config_path = 'configs/body_2d_keypoint/rtmpose/coco52/rtmpose-l_8xb256-420e_coco52-256x192.py'
    checkpoint_path = 'work_dirs/rtmpose-l_coco52/best_PCK_epoch_XX.pth'
    output_path = 'work_dirs/rtmpose-l_coco52/rtmpose_coco52.onnx'
    input_shape = (1, 3, 256, 192)
    ...
```

Run:

```bash
python scripts/export/rtm2onnx.py
```

This produces an ONNX model with:
- **Input**: `(N, 3, 256, 192)` — N is dynamic (batch)
- **Output**: `simcc_x (N, 52, 384)` and `simcc_y (N, 52, 512)` — SimCC classification logits

---

## Step 6: Run Inference with ONNX

### Minimal example

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('rtmpose_coco52.onnx')

# Preprocess your image to (1, 3, 256, 192) float32, BGR, ImageNet-normalized
input_img = preprocess(image, bbox)  # see scripts/preprocessing_mmpose.py

simcc_x, simcc_y = session.run(None, {'input': input_img})

# Decode keypoints
x = np.argmax(simcc_x, axis=2) / 2.0  # divide by simcc_split_ratio
y = np.argmax(simcc_y, axis=2) / 2.0
scores = np.max(simcc_x, axis=2)  # confidence per keypoint

keypoints = np.stack([x, y], axis=-1)  # (N, 52, 2) in input crop space
```

### Correct preprocessing

Use `scripts/preprocessing_mmpose.py` for MMPose-compatible preprocessing (affine transforms with aspect ratio fixing). See [ONNX_FIX_SUMMARY.md](ONNX_FIX_SUMMARY.md) for why this matters — naive crop+resize causes 5-11px errors, proper affine transforms bring it down to ~2px.

For a full video inference example, see `scripts/inference/inference_coco29_onnx.py` — adapt it by changing the number of keypoints and skeleton definition.

---

## Checklist

- [ ] `configs/_base_/datasets/coco52.py` — keypoint names, swap pairs, skeleton, sigmas, weights
- [ ] `coco52_train.json` / `coco52_val.json` — COCO format annotations with 52 keypoints
- [ ] Images in `train/` and `val/` subdirectories
- [ ] Training config with `num_keypoints=52`, correct `data_root`, `ann_file`, `metainfo`
- [ ] Pretrained checkpoint in `checkpoints/`
- [ ] Train: `python tools/train.py configs/.../coco52/... --work-dir work_dirs/rtmpose-l_coco52`
- [ ] Export: update paths in `scripts/export/rtm2onnx.py` and run
- [ ] Inference: adapt `scripts/inference/inference_coco29_onnx.py` for 52 keypoints
