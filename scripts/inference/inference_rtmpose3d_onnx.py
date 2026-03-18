"""
RTMPose 3D ONNX Inference Script with Pre-computed Bounding Boxes

This script performs 3D whole-body pose estimation using the exported ONNX model.
It loads pre-computed person bounding boxes from a .npy file and estimates 3D poses.

Requirements:
- onnxruntime or onnxruntime-gpu
- opencv-python
- numpy
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import onnxruntime as ort


def preprocess_image(img, bbox, input_size=(288, 384)):
    """
    Crop and preprocess image based on bounding box.

    Args:
        img: Input image (BGR format)
        bbox: Bounding box [x1, y1, x2, y2]
        input_size: Target input size (width, height)

    Returns:
        preprocessed: Preprocessed image tensor (1, 3, H, W)
        center: Center point of bbox
        scale: Scale factor for bbox
    """
    x1, y1, x2, y2 = bbox

    # Get bbox center and size
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Add padding (1.25x bbox size)
    scale = max(bbox_w, bbox_h) * 1.25

    # Crop and resize region
    crop_size = int(scale)
    x1_crop = int(center_x - crop_size / 2)
    y1_crop = int(center_y - crop_size / 2)
    x2_crop = int(center_x + crop_size / 2)
    y2_crop = int(center_y + crop_size / 2)

    # Handle boundary cases
    img_h, img_w = img.shape[:2]
    pad_left = max(0, -x1_crop)
    pad_top = max(0, -y1_crop)
    pad_right = max(0, x2_crop - img_w)
    pad_bottom = max(0, y2_crop - img_h)

    # Adjust crop coordinates
    x1_crop = max(0, x1_crop)
    y1_crop = max(0, y1_crop)
    x2_crop = min(img_w, x2_crop)
    y2_crop = min(img_h, y2_crop)

    # Crop image
    crop_img = img[y1_crop:y2_crop, x1_crop:x2_crop]

    # Pad if necessary
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop_img = cv2.copyMakeBorder(
            crop_img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    # Resize to input size (width, height)
    resized = cv2.resize(crop_img, input_size, interpolation=cv2.INTER_LINEAR)

    # Normalize (ImageNet mean/std)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # BGR to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize
    normalized = (resized.astype(np.float32) - mean) / std

    # HWC to CHW
    normalized = normalized.transpose(2, 0, 1)

    # Add batch dimension
    preprocessed = normalized[np.newaxis, ...]

    return preprocessed, (center_x, center_y), scale


def decode_simcc_3d(simcc_x, simcc_y, simcc_z, split_ratio=2.0):
    """
    Decode SimCC3D outputs to 3D keypoint coordinates.

    Args:
        simcc_x: X-axis classification scores (N, K, W*split_ratio)
        simcc_y: Y-axis classification scores (N, K, H*split_ratio)
        simcc_z: Z-axis classification scores (N, K, D*split_ratio)
        split_ratio: SimCC split ratio (default: 2.0)

    Returns:
        keypoints_3d: 3D keypoint coordinates (N, K, 3)
        scores: Keypoint confidence scores (N, K)
    """
    # Get argmax for each axis
    x_coords = np.argmax(simcc_x, axis=2).astype(np.float32) / split_ratio
    y_coords = np.argmax(simcc_y, axis=2).astype(np.float32) / split_ratio
    z_coords = np.argmax(simcc_z, axis=2).astype(np.float32) / split_ratio

    # Get confidence scores (max probability)
    x_scores = np.max(simcc_x, axis=2)
    y_scores = np.max(simcc_y, axis=2)
    z_scores = np.max(simcc_z, axis=2)

    # Average scores across all axes
    scores = (x_scores + y_scores + z_scores) / 3.0

    # Stack coordinates
    keypoints_3d = np.stack([x_coords, y_coords, z_coords], axis=-1)

    return keypoints_3d, scores


def convert_to_metric_scale(keypoints_3d, camera_intrinsics, bbox, input_size=(288, 384),
                           scale_method='root_depth', root_depth=2000.0,
                           bone_length=None, bone_indices=None):
    """
    Convert normalized 3D keypoints to metric scale using camera intrinsics.

    Args:
        keypoints_3d: Normalized 3D keypoints (N, K, 3) - before coordinate transformation
        camera_intrinsics: Camera intrinsic matrix K (3, 3)
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        bbox: Bounding box [x1, y1, x2, y2]
        input_size: Model input size (width, height)
        scale_method: Scaling method ('root_depth' or 'bone_length')
        root_depth: Depth of root point in mm (for 'root_depth' method)
        bone_length: Known bone length in mm (for 'bone_length' method)
        bone_indices: Tuple of (start_idx, end_idx) for bone (for 'bone_length' method)

    Returns:
        keypoints_3d_metric: 3D keypoints in metric coordinates (N, K, 3) in mm
    """
    # Extract intrinsic parameters
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Get bbox properties for scaling back to original image space
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_scale = max(bbox_w, bbox_h) * 1.25

    # Model outputs are in input_size space (288x384)
    # Scale back to original image coordinates
    input_w, input_h = input_size
    scale_x = bbox_scale / input_w
    scale_y = bbox_scale / input_h

    # Convert from model output space to original image space
    x_img = keypoints_3d[..., 0] * scale_x + (center_x - bbox_scale / 2)
    y_img = keypoints_3d[..., 1] * scale_y + (center_y - bbox_scale / 2)
    z_relative = keypoints_3d[..., 2]  # Relative depth from model

    if scale_method == 'root_depth':
        # Method 1: Assume root point (hip midpoint) is at a known depth
        # Root indices: 11 (left_hip), 12 (right_hip) for COCO format
        root_indices = [11, 12]

        # Calculate root point in image space
        if keypoints_3d.shape[1] > max(root_indices):
            root_x = (x_img[..., root_indices[0]] + x_img[..., root_indices[1]]) / 2
            root_y = (y_img[..., root_indices[0]] + y_img[..., root_indices[1]]) / 2
            root_z_rel = (z_relative[..., root_indices[0]] + z_relative[..., root_indices[1]]) / 2
        else:
            # Fallback to first keypoint if hips not available
            root_x = x_img[..., 0]
            root_y = y_img[..., 0]
            root_z_rel = z_relative[..., 0]

        # Absolute depth for root
        Z_root = root_depth  # mm

        # Back-project to 3D metric coordinates
        # X = (x - cx) * Z / fx
        # Y = (y - cy) * Z / fy
        X_metric = (x_img - cx) * Z_root / fx
        Y_metric = (y_img - cy) * Z_root / fy

        # For Z: scale the relative depth based on the root depth
        # Estimate scale factor from bbox size and focal length
        # Approximate: depth_scale = Z_root / fx * input_size_scale
        depth_scale = Z_root / fx * scale_x
        Z_metric = z_relative * depth_scale + (Z_root - root_z_rel * depth_scale)

    elif scale_method == 'bone_length':
        # Method 2: Normalize by a known bone length
        if bone_indices is None or bone_length is None:
            raise ValueError("bone_indices and bone_length must be provided for 'bone_length' method")

        start_idx, end_idx = bone_indices

        # Calculate bone vector in normalized space
        bone_vec = keypoints_3d[:, end_idx] - keypoints_3d[:, start_idx]
        bone_len_normalized = np.linalg.norm(bone_vec, axis=-1, keepdims=True)

        # Scale factor: known_length / normalized_length
        scale_factor = bone_length / bone_len_normalized

        # Apply scale to all keypoints
        # Note: This gives relative metric coordinates, not absolute camera space
        X_metric = x_img * scale_factor[..., 0:1]
        Y_metric = y_img * scale_factor[..., 0:1]
        Z_metric = z_relative * scale_factor[..., 0:1]

    else:
        raise ValueError(f"Unknown scale_method: {scale_method}")

    # Stack metric coordinates
    keypoints_3d_metric = np.stack([X_metric, Y_metric, Z_metric], axis=-1)

    return keypoints_3d_metric


def process_frame_onnx(frame, bbox, session, input_size=(288, 384),
                       kpt_thr=0.3, disable_rebase=False,
                       camera_intrinsics=None, convert_to_metric=False,
                       root_depth=2000.0):
    """
    Process a single frame for 3D pose estimation using ONNX model.

    Args:
        frame: Input image (BGR format)
        bbox: Bounding box [x1, y1, x2, y2]
        session: ONNX Runtime session
        input_size: Model input size (width, height)
        kpt_thr: Keypoint score threshold
        disable_rebase: Whether to disable rebasing 3D pose to ground
        camera_intrinsics: Camera intrinsic matrix K (3, 3), optional
        convert_to_metric: Whether to convert to metric scale (requires camera_intrinsics)
        root_depth: Root depth in mm for metric conversion (default: 2000mm = 2m)

    Returns:
        keypoints_3d: 3D keypoint coordinates (N, K, 3)
        scores: Keypoint confidence scores (N, K)
    """
    # Preprocess image
    input_tensor, center, scale = preprocess_image(frame, bbox, input_size)

    # Run ONNX inference
    outputs = session.run(None, {'input': input_tensor})
    simcc_x, simcc_y, simcc_z = outputs

    # Decode SimCC3D outputs (in normalized space)
    keypoints_3d, scores = decode_simcc_3d(simcc_x, simcc_y, simcc_z)

    # Convert to metric scale if requested
    if convert_to_metric and camera_intrinsics is not None:
        keypoints_3d_metric = convert_to_metric_scale(
            keypoints_3d,
            camera_intrinsics,
            bbox,
            input_size=input_size,
            scale_method='root_depth',
            root_depth=root_depth
        )
        # Use metric coordinates
        keypoints_3d = keypoints_3d_metric

    # Transform coordinates: convert to 3D visualization space
    # Original: X, Y, Z -> Transform: -X, -Z, -Y
    keypoints_3d = -keypoints_3d[..., [0, 2, 1]]

    # Rebase height (make the lowest keypoint touch the ground)
    if not disable_rebase:
        keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)

    return keypoints_3d, scores


def visualize_3d_skeleton(keypoints_3d, scores, skeleton_links, img_shape,
                          kpt_thr=0.3, axis_limit=400):
    """
    Simple 3D skeleton visualization.

    Args:
        keypoints_3d: 3D keypoint coordinates (K, 3)
        scores: Keypoint confidence scores (K,)
        skeleton_links: List of [start_idx, end_idx] for bone connections
        img_shape: Output image shape (H, W, 3)
        kpt_thr: Keypoint score threshold
        axis_limit: 3D axis limit for visualization

    Returns:
        vis_img: Visualization image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot keypoints
    valid_mask = scores > kpt_thr
    x, y, z = keypoints_3d[valid_mask, 0], keypoints_3d[valid_mask, 1], keypoints_3d[valid_mask, 2]
    ax.scatter(x, y, z, c='red', marker='o', s=50)

    # Plot skeleton
    for link in skeleton_links:
        start_idx, end_idx = link
        if start_idx >= len(scores) or end_idx >= len(scores):
            continue
        if scores[start_idx] > kpt_thr and scores[end_idx] > kpt_thr:
            xs = [keypoints_3d[start_idx, 0], keypoints_3d[end_idx, 0]]
            ys = [keypoints_3d[start_idx, 1], keypoints_3d[end_idx, 1]]
            zs = [keypoints_3d[start_idx, 2], keypoints_3d[end_idx, 2]]
            ax.plot(xs, ys, zs, 'b-', linewidth=2)

    # Set axis properties
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(0, axis_limit * 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=15, azim=70)

    # Convert plot to image
    fig.canvas.draw()
    vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Convert RGB to BGR for OpenCV
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Resize to match original image shape
    vis_img = cv2.resize(vis_img, (img_shape[1], img_shape[0]))

    return vis_img


def main():
    # =====================================================================
    # CONFIGURATION - Modify these parameters for your inference
    # =====================================================================

    # Paths
    onnx_model_path = '/home/marco/Desktop/phoenix/models/rtmPose3d/rtmw3d-x.onnx'  # Path to exported ONNX model
    video_path = '/home/marco/Desktop/phoenix/emx_run/input/videos/cam03.mp4'
    bboxes_path = '/home/marco/Desktop/phoenix/emx_tracking/realtime_tracking/experiments/yolo_detections.npy'
    output_path = 'output_rtmpose3d_onnx.mp4'

    # Settings
    input_size = (288, 384)  # (width, height) - must match model
    kpt_thr = 0.3  # Keypoint score threshold for visualization
    disable_rebase = False  # Set to True to disable rebasing 3D pose to ground level
    max_frames = 100  # Maximum number of frames to process (set to None for all frames)
    use_gpu = True  # Use GPU acceleration if available

    # Metric conversion settings (optional)
    convert_to_metric = False  # Set to True to convert normalized keypoints to metric scale
    camera_intrinsics = None  # Camera intrinsic matrix K (3x3), required if convert_to_metric=True
    root_depth = 2000.0  # Root depth in mm (default: 2000mm = 2m from camera)

    # Example camera intrinsics (uncomment and modify with your camera parameters):
    # camera_intrinsics = np.array([
    #     [1000.0, 0.0, 960.0],   # fx, 0, cx
    #     [0.0, 1000.0, 540.0],   # 0, fy, cy
    #     [0.0, 0.0, 1.0]         # 0, 0, 1
    # ], dtype=np.float32)

    # Skeleton connections for RTMW3D (133 keypoints - showing main body connections)
    skeleton_links = [
        # Body (COCO17)
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head
        [0, 5], [0, 6],  # Shoulders to nose
        [5, 7], [7, 9],  # Left arm
        [6, 8], [8, 10],  # Right arm
        [5, 11], [6, 12], [11, 12],  # Torso
        [11, 13], [13, 15],  # Left leg
        [12, 14], [14, 16],  # Right leg
    ]

    # =====================================================================
    # END CONFIGURATION
    # =====================================================================

    # Check if files exist
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(bboxes_path):
        raise FileNotFoundError(f"Bounding boxes file not found: {bboxes_path}")

    print("=" * 60)
    print("RTMPose 3D ONNX Inference with Pre-computed Bounding Boxes")
    print("=" * 60)

    # Load ONNX model
    print("\n[1/4] Loading ONNX model...")
    print(f"  Model: {onnx_model_path}")

    # Setup execution providers
    providers = ['CPUExecutionProvider']
    if use_gpu:
        providers.insert(0, 'CUDAExecutionProvider')

    session = ort.InferenceSession(onnx_model_path, providers=providers)

    # Print model info
    print(f"  Execution providers: {session.get_providers()}")
    print(f"  Input: {session.get_inputs()[0].name} - {session.get_inputs()[0].shape}")
    print(f"  Outputs: {[out.name + ' - ' + str(out.shape) for out in session.get_outputs()]}")

    # Load pre-computed bounding boxes
    print("\n[2/4] Loading pre-computed bounding boxes...")
    bboxes_data = np.load(bboxes_path)  # Shape: (nframes, 5) - [x1, y1, x2, y2, score]
    print(f"  Loaded {len(bboxes_data)} bounding boxes")
    print(f"  Shape: {bboxes_data.shape}")

    # Open video
    print("\n[3/4] Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {width}x{height} @ {fps} FPS")
    print(f"  Total frames: {total_frames}")

    # Determine how many frames to process
    frames_to_process = min(len(bboxes_data), total_frames)
    if max_frames is not None:
        frames_to_process = min(frames_to_process, max_frames)

    print(f"  Will process {frames_to_process} frames")

    # Print metric conversion info
    if convert_to_metric and camera_intrinsics is not None:
        print(f"\n[Metric Conversion] Enabled")
        print(f"  Camera intrinsics:")
        print(f"    fx={camera_intrinsics[0, 0]:.2f}, fy={camera_intrinsics[1, 1]:.2f}")
        print(f"    cx={camera_intrinsics[0, 2]:.2f}, cy={camera_intrinsics[1, 2]:.2f}")
        print(f"  Root depth: {root_depth:.2f} mm")
        print(f"  Output coordinates will be in metric scale (mm)")
    else:
        print(f"\n[Metric Conversion] Disabled (using normalized coordinates)")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    # Process video frame by frame
    print("\n[4/4] Processing video frames...")
    frame_idx = 0
    pbar = tqdm(total=frames_to_process, desc="Processing frames")

    while cap.isOpened() and frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        # Get bounding box for this frame
        bbox_data = bboxes_data[frame_idx]  # [x1, y1, x2, y2, score]
        bbox = bbox_data[:4].astype(np.float32)
        bbox_score = bbox_data[4]

        # Run pose estimation if bbox is valid
        if bbox_score > 0 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            try:
                # Run ONNX inference
                keypoints_3d, scores = process_frame_onnx(
                    frame,
                    bbox,
                    session,
                    input_size=input_size,
                    kpt_thr=kpt_thr,
                    disable_rebase=disable_rebase,
                    camera_intrinsics=camera_intrinsics,
                    convert_to_metric=convert_to_metric,
                    root_depth=root_depth
                )

                # Visualize (simple 3D plot)
                vis_frame = visualize_3d_skeleton(
                    keypoints_3d[0],  # First person
                    scores[0],
                    skeleton_links,
                    frame.shape,
                    kpt_thr=kpt_thr
                )
            except Exception as e:
                print(f"\n  Error processing frame {frame_idx}: {e}")
                vis_frame = np.ones_like(frame) * 255
        else:
            # No valid bbox - show blank white frame
            vis_frame = np.ones_like(frame) * 255

        # Initialize video writer on first frame
        if out is None:
            output_height, output_width = vis_frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            print(f"  Output video dimensions: {output_width}x{output_height}")

        # Write frame to output video
        out.write(vis_frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"\nProcessed {frame_idx} frames")
    print(f"Output saved to: {output_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()