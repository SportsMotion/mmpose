# ONNX Preprocessing Fix Summary

## Problem Identified

The ONNX inference script had **5-11 pixel errors** compared to PyTorch, with the largest errors on extremities (hands/feet). This was caused by incorrect preprocessing.

## Root Cause

### Original (Incorrect) Preprocessing
```python
# Simple crop + resize
cropped = image[y1:y2, x1:x2]
resized = cv2.resize(cropped, (192, 256))
```

**Issues:**
- No aspect ratio fixing
- Direct resize distorts proportions
- Doesn't match MMPose's preprocessing pipeline

### Correct MMPose Preprocessing

MMPose uses a more sophisticated pipeline:

1. **Convert bbox to center-scale format**
   - Represents bbox as (center_x, center_y, scale_w, scale_h)
   - Normalized by pixel_std=200.0

2. **Fix aspect ratio to 3:4 (192:256)**
   ```python
   if bbox_w > bbox_h * aspect_ratio:
       bbox_h = bbox_w / aspect_ratio
   else:
       bbox_w = bbox_h * aspect_ratio
   ```

3. **Apply padding** (1.0x for inference, 1.25x for training)

4. **Affine transformation** instead of simple resize
   - Uses `cv2.warpAffine()` with `cv2.INTER_LINEAR`
   - Preserves proportions and geometry
   - 3-point affine matrix (rotation, scale, translation)

5. **Inverse transform for keypoints**
   - Keypoints decoded in model space (0-192, 0-256)
   - Transformed back to original image space using inverse affine matrix

## Results

### Before Fix
- Mean error: **5.28 pixels**
- Max error: **11.39 pixels**
- 13/29 keypoints with >5px error

### After Fix
- Mean error: **2.11 pixels**
- Max error: **4.01 pixels**
- 0/29 keypoints with >5px error
- **60% error reduction!**

## Files Created

1. **`preprocessing_mmpose.py`** - MMPose-compatible preprocessing functions
   - `bbox_xyxy2cs()` - Convert bbox to center-scale
   - `get_warp_matrix()` - Create affine transformation matrix
   - `preprocess_image_mmpose()` - Full preprocessing pipeline
   - `transform_keypoints_inverse()` - Transform keypoints back to image space
   - `decode_simcc()` - Decode SimCC outputs

2. **`inference_coco29_onnx.py`** - Updated ONNX inference script
   - Now uses MMPose-compatible preprocessing
   - Produces results within 1-4 pixels of PyTorch

3. **`compare_methods.py`** - Comparison tool
   - Compares PyTorch vs ONNX vs rtmlib
   - Generates per-keypoint error analysis
   - Creates visualization overlay

## Usage

### Run ONNX inference with corrected preprocessing:
```bash
python inference_coco29_onnx.py
```

### Compare methods:
```bash
python compare_methods.py
```

### Use in your own code:
```python
from preprocessing_mmpose import preprocess_image_mmpose, decode_simcc, transform_keypoints_inverse
import onnxruntime as ort

# Load model
session = ort.InferenceSession('model.onnx')

# Preprocess
input_tensor, warp_mat = preprocess_image_mmpose(image, bbox, (192, 256))

# Inference
simcc_x, simcc_y = session.run(None, {'input': input_tensor})

# Decode
keypoints_model, scores = decode_simcc(simcc_x, simcc_y)
keypoints = transform_keypoints_inverse(keypoints_model, warp_mat)
```

## Key Takeaways

1. **Always match preprocessing exactly** when converting models - even small differences compound
2. **Affine transformations preserve geometry** better than simple crop+resize
3. **Aspect ratio matters** - especially for pose estimation where proportions are critical
4. **Test thoroughly** - Compare outputs numerically, not just visually
5. **Document your pipeline** - Make it reproducible

## References

- MMPose preprocessing: `mmpose/datasets/transforms/topdown_transforms.py`
- Bbox conversion: `mmpose/structures/bbox/transforms.py`
- Data preprocessor: `mmpose/models/data_preprocessors/data_preprocessor.py`