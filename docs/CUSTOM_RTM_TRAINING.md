# Custom RTMPose COCO29 Training Guide

Training RTMPose-Large on the COCO29 dataset (17 body + 6 foot + 6 hand keypoints) with transfer learning from Halpe26.

## Dataset Format

The dataset uses **COCO keypoint format** — a single JSON annotation file per split, with images in subdirectories.

### Directory Structure

```
/media/marco/T7/dataset/COCO29/
├── coco29_train.json          # Training annotations
├── coco29_val.json            # Validation annotations
├── train/                     # Training images
│   ├── seq001_cam01_frame000000.jpg
│   ├── seq001_cam01_frame000100.jpg
│   └── ...
└── val/                       # Validation images
    ├── seq010_cam01_frame000000.jpg
    └── ...
```

Current dataset size:
- **Train**: 5,538 images / 7,990 annotations (~1.4 persons/image)
- **Val**: 1,211 images / 1,400 annotations (~1.2 persons/image)
- Images are 3840x2160 from multi-camera setups (139 sequence-camera combos)

### Annotation JSON Structure

Each annotation file has three top-level keys:

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

#### `images` — one entry per image

```json
{
  "id": 0,
  "file_name": "train/seq001_cam01_frame000000.jpg",
  "width": 3840,
  "height": 2160
}
```

- `id`: unique integer, used to link annotations to images
- `file_name`: path relative to `data_root` (the dataset directory)
- `width`, `height`: image dimensions in pixels

Optional fields (used in our data but not required by mmpose):
- `frame_id`, `video_id`: useful for tracking/temporal data

#### `annotations` — one entry per person

```json
{
  "id": 0,
  "image_id": 0,
  "category_id": 1,
  "keypoints": [1189.83, 563.84, 2.0, 1200.12, 550.33, 2.0, ...],
  "num_keypoints": 29,
  "bbox": [980.55, 527.17, 461.42, 542.94],
  "area": 250527.99,
  "iscrowd": 0
}
```

- `id`: unique integer across all annotations
- `image_id`: links to the corresponding image
- `category_id`: always `1` (person)
- `keypoints`: flat array of `[x1, y1, v1, x2, y2, v2, ...]` — 87 values for 29 keypoints
- `num_keypoints`: number of keypoints (always 29 for this dataset)
- `bbox`: bounding box as `[x, y, width, height]` (top-left corner)
- `area`: bbox area in pixels (width * height)
- `iscrowd`: `0` for individual annotations, `1` for crowd (ignore during training)

#### Keypoint visibility flags

Each keypoint has a visibility value `v`:

| Value | Meaning |
|-------|---------|
| `0` | Not labeled / not visible (keypoint is ignored during training) |
| `1` | Labeled but occluded (used for training, lower confidence) |
| `2` | Labeled and visible (full confidence) |

Distribution in current training set: ~79% visible, ~6% occluded, ~15% not labeled.

#### `categories` — keypoint definition

```json
{
  "id": 1,
  "name": "person",
  "supercategory": "person",
  "keypoints": [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    "L_BigToe", "R_BigToe", "L_SmallToe", "R_SmallToe", "L_Heel", "R_Heel",
    "L_Thumb2Knuckles", "R_Thumb2Knuckles",
    "L_Index1Knuckles", "R_Index1Knuckles",
    "L_Pinky1Knuckles", "R_Pinky1Knuckles"
  ],
  "skeleton": [
    [1,2], [1,3], [2,4], [3,5],
    [6,7], [6,12], [7,13], [12,13],
    [6,8], [8,10], [7,9], [9,11],
    [12,14], [14,16], [13,15], [15,17],
    [16,18], [16,20], [16,22], [18,20],
    [17,19], [17,21], [17,23], [19,21],
    [10,24], [10,26], [10,28], [24,26], [26,28],
    [11,25], [11,27], [11,29], [25,27], [27,29]
  ]
}
```

Note: skeleton indices are **1-indexed** (COCO convention).

### 29 Keypoint Order (0-indexed)

| Index | Name | Group |
|-------|------|-------|
| 0 | Nose | Body |
| 1 | L_Eye | Body |
| 2 | R_Eye | Body |
| 3 | L_Ear | Body |
| 4 | R_Ear | Body |
| 5 | L_Shoulder | Body |
| 6 | R_Shoulder | Body |
| 7 | L_Elbow | Body |
| 8 | R_Elbow | Body |
| 9 | L_Wrist | Body |
| 10 | R_Wrist | Body |
| 11 | L_Hip | Body |
| 12 | R_Hip | Body |
| 13 | L_Knee | Body |
| 14 | R_Knee | Body |
| 15 | L_Ankle | Body |
| 16 | R_Ankle | Body |
| 17 | L_BigToe | Foot |
| 18 | R_BigToe | Foot |
| 19 | L_SmallToe | Foot |
| 20 | R_SmallToe | Foot |
| 21 | L_Heel | Foot |
| 22 | R_Heel | Foot |
| 23 | L_Thumb2Knuckles | Hand |
| 24 | R_Thumb2Knuckles | Hand |
| 25 | L_Index1Knuckles | Hand |
| 26 | R_Index1Knuckles | Hand |
| 27 | L_Pinky1Knuckles | Hand |
| 28 | R_Pinky1Knuckles | Hand |

---

## Expanding the Dataset with Synthetic Data

To add new data (synthetic or real), you need to:

1. **Add images** to `train/` (or `val/`)
2. **Append entries** to the annotation JSON

### Step 1: Prepare your new images

Place images into the `train/` or `val/` subdirectory. Use any naming convention, but avoid collisions with existing filenames.

```
train/
├── seq001_cam01_frame000000.jpg   # existing
├── ...
├── synth_001_frame0000.jpg        # new synthetic data
├── synth_001_frame0001.jpg
└── ...
```

### Step 2: Create annotations for new data

For each person in each new image, you need:
- The bounding box `[x, y, width, height]`
- All 29 keypoints as `[x, y, visibility]` triplets
- Set visibility to `0` for any keypoints you don't have labels for

### Step 3: Merge into the existing annotation file

```python
import json

# Load existing annotations
with open('/media/marco/T7/dataset/COCO29/coco29_train.json') as f:
    dataset = json.load(f)

# Find the next available IDs
next_img_id = max(img['id'] for img in dataset['images']) + 1
next_ann_id = max(ann['id'] for ann in dataset['annotations']) + 1

# Add new images
new_images = []
new_annotations = []

for i, (img_path, width, height, persons) in enumerate(your_synthetic_data):
    img_id = next_img_id + i
    new_images.append({
        'id': img_id,
        'file_name': f'train/{img_path}',  # relative to data_root
        'width': width,
        'height': height,
    })

    for person in persons:
        # person['keypoints'] = [x0, y0, v0, x1, y1, v1, ...] (87 values)
        # person['bbox'] = [x, y, w, h]
        kps = person['keypoints']
        bbox = person['bbox']
        new_annotations.append({
            'id': next_ann_id,
            'image_id': img_id,
            'category_id': 1,
            'keypoints': kps,
            'num_keypoints': 29,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': 0,
        })
        next_ann_id += 1

# Merge
dataset['images'].extend(new_images)
dataset['annotations'].extend(new_annotations)

# Save
with open('/media/marco/T7/dataset/COCO29/coco29_train.json', 'w') as f:
    json.dump(dataset, f)

print(f'Added {len(new_images)} images, {len(new_annotations)} annotations')
print(f'Total: {len(dataset["images"])} images, {len(dataset["annotations"])} annotations')
```

### Tips for synthetic data

- Make sure keypoint coordinates are in **pixel space** (absolute, not normalized)
- The bbox should tightly enclose all visible keypoints with some padding
- If your synthetic data only has a subset of keypoints (e.g. no foot labels), set those visibility flags to `0` — the model will learn to ignore them for those samples
- Keep the `categories` section unchanged — it defines the keypoint schema
- Aim for a reasonable ratio of synthetic to real data (start with 1:1 or less)

### Computing bbox from keypoints

If you have keypoints but no bounding box:

```python
def keypoints_to_bbox(keypoints, padding=1.2):
    """Convert [x0,y0,v0, x1,y1,v1, ...] to [x, y, w, h] with padding."""
    xs, ys = [], []
    for i in range(0, len(keypoints), 3):
        if keypoints[i+2] > 0:  # visible or occluded
            xs.append(keypoints[i])
            ys.append(keypoints[i+1])
    if not xs:
        return [0, 0, 0, 0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    # Add padding
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w *= padding
    h *= padding
    return [cx - w/2, cy - h/2, w, h]
```

---

## Training

### Prerequisites

1. Environment set up via `./setup_env.sh` (creates `.venv/`)
2. Dataset at `/media/marco/T7/dataset/COCO29/` (or update `data_root` in config)
3. Pretrained checkpoint at `checkpoints/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth`

### Run training

```bash
.venv/bin/python tools/train.py \
  configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py \
  --work-dir work_dirs/rtmpose-l_coco29
```

### Key config parameters

These are in `configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_epochs` | 130 | Total training epochs |
| `train_batch_size` | 128 | Batch size (reduce if OOM) |
| `base_lr` | 4e-3 | Learning rate (auto-scaled by batch size) |
| `freeze_backbone_epochs` | 130 | Epochs to keep backbone frozen (head-only training) |
| `data_root` | `/media/marco/T7/dataset/COCO29/` | Path to dataset |
| `val_interval` | 5 | Validate every N epochs |

### Override config from command line

```bash
# Shorter training run
.venv/bin/python tools/train.py \
  configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py \
  --work-dir work_dirs/rtmpose-l_coco29 \
  --cfg-options train_cfg.max_epochs=50 train_batch_size=64

# Different dataset location (e.g. on cloud)
.venv/bin/python tools/train.py \
  configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py \
  --work-dir work_dirs/rtmpose-l_coco29 \
  --cfg-options data_root=/data/COCO29/
```

### Resume from checkpoint

```bash
.venv/bin/python tools/train.py \
  configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py \
  --work-dir work_dirs/rtmpose-l_coco29 \
  --resume work_dirs/rtmpose-l_coco29/epoch_20.pth
```

### Training output

```
work_dirs/rtmpose-l_coco29/
├── rtmpose-l_8xb256-420e_coco29-256x192.py   # Saved config snapshot
├── epoch_10.pth                                # Periodic checkpoints
├── epoch_20.pth
├── best_PCK_epoch_15.pth                       # Best model by PCK accuracy
├── last_checkpoint                             # Pointer to latest checkpoint
└── 20260318_115206/                            # Run logs
    ├── 20260318_115206.log                     # Text log
    └── vis_data/
        ├── scalars.json                        # Metrics (loss, PCK, lr)
        └── events.out.tfevents.*               # TensorBoard events
```

### Monitor training

```bash
# TensorBoard
tensorboard --logdir work_dirs/rtmpose-l_coco29

# Or check the log directly
tail -f work_dirs/rtmpose-l_coco29/*/$(ls -t work_dirs/rtmpose-l_coco29/*/*.log | head -1)
```

### What to expect

- **Epoch 1**: loss ~3.2, PCK ~0.87 (backbone pretrained, head random)
- **Epoch 10-20**: loss drops to ~0.7, PCK improves
- **VRAM**: ~1.7 GB with backbone frozen (batch 128 on RTX 3090 Ti)
- **Speed**: ~20s/epoch on RTX 3090 Ti with current dataset size

### Training strategy

The config uses a two-stage approach:

1. **Stage 1** (epochs 0 → `max_epochs - 10`): Strong augmentations (large rotation ±80°, scale 0.6-1.4, coarse dropout 50%)
2. **Stage 2** (last 10 epochs): Reduced augmentations (rotation ±60°, scale 0.75-1.25, dropout 30%)

The backbone is **frozen for all 130 epochs** (head-only training). This is set by `freeze_backbone_epochs=130`. To fine-tune the full model, reduce this value:

```bash
# Freeze backbone for 50 epochs, then fine-tune everything
--cfg-options freeze_backbone_epochs=50
```
