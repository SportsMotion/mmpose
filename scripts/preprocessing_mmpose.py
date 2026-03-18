"""
MMPose-compatible preprocessing for ONNX inference.

This module implements the exact preprocessing pipeline used by MMPose
for RTMPose inference, including aspect ratio fixing and affine transformation.
"""

import cv2
import numpy as np


def bbox_xyxy2cs(bbox, aspect_ratio, padding=1.25, pixel_std=200.0):
    """Convert bbox from xyxy format to center-scale format.

    This matches MMPose's bbox_xyxy2cs() implementation.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        aspect_ratio: Target aspect ratio (width / height), e.g., 192/256 = 0.75
        padding: Padding factor (1.25 for training, 1.0 for inference)
        pixel_std: Standard pixel value for scaling (default 200.0)

    Returns:
        center: (x, y) center of bbox
        scale: (w, h) scale of bbox in units of pixel_std
    """
    x1, y1, x2, y2 = bbox

    # Get bbox dimensions
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Center
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    # Fix aspect ratio
    if bbox_w > bbox_h * aspect_ratio:
        bbox_h = bbox_w / aspect_ratio
    else:
        bbox_w = bbox_h * aspect_ratio

    # Apply padding
    bbox_w *= padding
    bbox_h *= padding

    # Scale (normalize by pixel_std)
    scale = np.array([bbox_w / pixel_std, bbox_h / pixel_std], dtype=np.float32)

    return center, scale


def get_warp_matrix(center, scale, rot, output_size, pixel_std=200.0):
    """Get affine transformation matrix.

    This matches MMPose's get_warp_matrix() implementation.

    Args:
        center: (x, y) center point
        scale: (w, h) scale in units of pixel_std
        rot: Rotation angle in degrees
        output_size: (width, height) output size
        pixel_std: Standard pixel value (default 200.0)

    Returns:
        warp_mat: 2x3 affine transformation matrix
    """
    # Scale to pixel coordinates
    scale_tmp = scale * pixel_std

    # Get matrix components
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # Rotation
    rot_rad = np.deg2rad(rot)
    sn = np.sin(rot_rad)
    cs = np.cos(rot_rad)

    # Source points (3 points defining the original box)
    src_dir = np.array([0, src_w * -0.5], dtype=np.float32)  # Direction vector
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    # Rotate source direction
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center  # Center point
    src[1, :] = center + src_dir  # Top-center point
    src[2, :] = center + np.array([src_dir[1], -src_dir[0]])  # Right-center point (perpendicular)

    # Destination points (after transformation)
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]  # Center of output
    dst[1, :] = [dst_w * 0.5, dst_h * 0.5] + dst_dir  # Top-center
    dst[2, :] = [dst_w * 0.5, dst_h * 0.5] + np.array([dst_dir[1], -dst_dir[0]])

    # Get affine transformation matrix
    warp_mat = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))

    return warp_mat


def preprocess_image_mmpose(image, bbox, input_size=(192, 256), padding=1.0):
    """Preprocess image using MMPose pipeline.

    This exactly matches the preprocessing done by MMPose for RTMPose inference.

    Args:
        image: Input image in BGR format (H, W, 3)
        bbox: Bounding box in [x1, y1, x2, y2] format
        input_size: Target size (width, height) for model input
        padding: Padding factor (1.25 for training, 1.0 for inference)

    Returns:
        input_tensor: Preprocessed image tensor (1, 3, H, W)
        warp_mat: Affine transformation matrix for inverse transform
    """
    # Convert bbox to center-scale format with aspect ratio fixing
    aspect_ratio = input_size[0] / input_size[1]  # width / height
    center, scale = bbox_xyxy2cs(bbox, aspect_ratio, padding=padding)

    # Get affine transformation matrix
    rot = 0  # No rotation for inference
    warp_mat = get_warp_matrix(center, scale, rot, input_size)

    # Apply affine transformation
    transformed = cv2.warpAffine(
        image,
        warp_mat,
        input_size,
        flags=cv2.INTER_LINEAR
    )

    # Convert BGR to RGB
    rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)

    # Normalize using ImageNet statistics
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    normalized = (rgb.astype(np.float32) - mean) / std

    # Convert to CHW format and add batch dimension
    input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

    return input_tensor.astype(np.float32), warp_mat


def transform_keypoints_inverse(keypoints, warp_mat):
    """Transform keypoints from model space back to image space.

    Args:
        keypoints: (N, 2) keypoints in model output space
        warp_mat: 2x3 affine transformation matrix from preprocessing

    Returns:
        keypoints_original: (N, 2) keypoints in original image space
    """
    # Get inverse transformation matrix
    # Add third row [0, 0, 1] to make it 3x3
    warp_mat_3x3 = np.vstack([warp_mat, [0, 0, 1]])
    inv_warp_mat = np.linalg.inv(warp_mat_3x3)[:2, :]  # Take first 2 rows

    # Transform keypoints (OpenCV format)
    # Add ones for homogeneous coordinates
    keypoints_homo = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    keypoints_original = keypoints_homo @ inv_warp_mat.T

    return keypoints_original


def decode_simcc(simcc_x, simcc_y):
    """Decode SimCC outputs to keypoint coordinates in model space.

    Args:
        simcc_x: X-axis classification scores (1, K, W*2)
        simcc_y: Y-axis classification scores (1, K, H*2)

    Returns:
        keypoints: (K, 2) keypoints in model output space [0, W] x [0, H]
        scores: (K,) keypoint confidence scores
    """
    # Get the predicted positions (argmax)
    x_locs = np.argmax(simcc_x[0], axis=1)  # (K,)
    y_locs = np.argmax(simcc_y[0], axis=1)  # (K,)

    # Get confidence scores (max values)
    x_scores = np.max(simcc_x[0], axis=1)  # (K,)
    y_scores = np.max(simcc_y[0], axis=1)  # (K,)
    scores = (x_scores + y_scores) / 2.0  # Average of x and y scores

    # Convert from SimCC coordinate space to model output space
    # SimCC uses 2x resolution, so divide by 2
    x_coords = x_locs / 2.0  # Now in range [0, input_width]
    y_coords = y_locs / 2.0  # Now in range [0, input_height]

    # Stack into keypoints array
    keypoints = np.stack([x_coords, y_coords], axis=1)  # (K, 2)

    return keypoints, scores


# Example usage
if __name__ == '__main__':
    """Example demonstrating the complete pipeline."""
    import onnxruntime as ort

    # Load model
    onnx_model_path = '/home/marco/Desktop/phoenix/models/rtmPose-l_coco29/rtmpose_coco29.onnx'
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    # Load test image and bbox
    video_path = '/home/marco/Desktop/phoenix/emx_run/input/videos/cam03.mp4'
    bboxes_path = '/home/marco/Desktop/phoenix/emx_tracking/realtime_tracking/experiments/yolo_detections.npy'

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()

    bboxes_data = np.load(bboxes_path)
    bbox = bboxes_data[50][:4]

    print("Testing MMPose-compatible preprocessing:")
    print(f"Bbox: {bbox}")

    # Preprocess
    input_tensor, warp_mat = preprocess_image_mmpose(frame, bbox, input_size=(192, 256))
    print(f"Input tensor shape: {input_tensor.shape}")

    # Run inference
    simcc_x, simcc_y = session.run(None, {input_name: input_tensor})

    # Decode keypoints (in model space)
    keypoints_model, scores = decode_simcc(simcc_x, simcc_y)
    print(f"Keypoints in model space shape: {keypoints_model.shape}")

    # Transform back to original image space
    keypoints_original = transform_keypoints_inverse(keypoints_model, warp_mat)
    print(f"Keypoints in original space shape: {keypoints_original.shape}")
    print(f"Sample keypoints (first 3):")
    print(keypoints_original[:3])