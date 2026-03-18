"""
RTMPose 3D Inference Script with Pre-computed Bounding Boxes

This script performs 3D whole-body pose estimation using RTMPose 3D models.
It loads pre-computed person bounding boxes from a .npy file and estimates 3D poses.

Requirements:
- mmpose
- mmcv
- opencv-python
"""

import os

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

# Import RTMPose3D custom modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'projects', 'rtmpose3d'))
from rtmpose3d import *  # noqa


def process_frame(frame, bbox, pose_estimator, visualizer, disable_rebase=False,
                  kpt_thr=0.3, show=False):
    """
    Process a single frame for 3D pose estimation using pre-computed bbox.

    Args:
        frame: Input image (BGR format)
        bbox: Bounding box (1, 4) array [x1, y1, x2, y2]
        pose_estimator: 3D pose estimator model
        visualizer: Visualizer for rendering results
        disable_rebase: Whether to disable rebasing 3D pose to ground
        kpt_thr: Keypoint score threshold for visualization
        show: Whether to show visualization window

    Returns:
        Visualization frame with 3D skeleton only (no video overlay)
    """
    # Estimate 3D poses
    pose_results = inference_topdown(pose_estimator, frame, bbox)

    if len(pose_results) == 0:
        # Return blank frame if no pose detected
        return np.ones_like(frame) * 255

    # Post-process results
    for idx, pose_result in enumerate(pose_results):
        pred_instances = pose_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores

        # Squeeze extra dimensions if present
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_results[idx].if i .keypoint_scores = keypoint_scores

        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        # Transform coordinates: convert to 3D visualization space
        # Original: X, Y, Z -> Transform: -X, -Z, -Y
        keypoints = -keypoints[..., [0, 2, 1]]

        # Rebase height (make the lowest keypoint touch the ground)
        if not disable_rebase:
            keypoints[..., 2] -= np.min(keypoints[..., 2], axis=-1, keepdims=True)

        pose_results[idx].pred_instances.keypoints = keypoints

    # Merge results for visualization
    pred_3d_data_samples = merge_data_samples(pose_results)

    # Visualize - only 3D skeleton, no 2D overlay
    if visualizer is not None:
        # Create a blank white canvas for visualization
        blank_canvas = np.ones_like(frame) * 255
        blank_canvas_rgb = mmcv.bgr2rgb(blank_canvas)

        visualizer.add_datasample(
            'result',
            blank_canvas_rgb,
            data_sample=pred_3d_data_samples,
            det_data_sample=pred_3d_data_samples,
            draw_gt=False,
            draw_2d=False,  # Don't draw 2D overlay
            dataset_2d=pose_estimator.dataset_meta['dataset_name'],
            dataset_3d=pose_estimator.dataset_meta['dataset_name'],
            show=show,
            draw_bbox=False,  # Don't draw bounding box
            kpt_thr=kpt_thr,
            convert_keypoint=False,
            axis_limit=400,
            axis_azimuth=70,
            axis_elev=15,
            num_instances=len(pose_results),
            wait_time=0.001)

        # Get visualization result (just 3D skeleton)
        frame_vis = visualizer.get_image()
        return mmcv.rgb2bgr(frame_vis)

    return np.ones_like(frame) * 255


def main():
    # =====================================================================
    # CONFIGURATION - Modify these parameters for your inference
    # =====================================================================

    # Paths
    pose_checkpoint = '/home/marco/Downloads/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth'  # e.g., 'rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth'
    pose_config = 'projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py'  # Changed from -l to -x
    video_path = '/home/marco/Desktop/phoenix/emx_run/input/videos/cam03.mp4'
    bboxes_path = '/home/marco/Desktop/phoenix/emx_tracking/realtime_tracking/experiments/yolo_detections.npy'
    output_path = 'output_rtmpose3d.mp4'

    # Settings
    device = 'cuda:0'
    kpt_thr = 0.3  # Keypoint score threshold for visualization
    disable_rebase = False  # Set to True to disable rebasing 3D pose to ground level
    radius = 3  # Keypoint radius for visualization
    thickness = 2  # Link thickness for visualization
    max_frames = 100  # Maximum number of frames to process (set to None for all frames)

    # =====================================================================
    # END CONFIGURATION
    # =====================================================================

    # Check if files exist
    if not os.path.exists(pose_config):
        raise FileNotFoundError(f"Pose config not found: {pose_config}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(bboxes_path):
        raise FileNotFoundError(f"Bounding boxes file not found: {bboxes_path}")

    if pose_checkpoint is None:
        print("\nWARNING: No pose checkpoint specified!")
        print("Please download RTMW3D model from:")
        print("  RTMW3D-L: https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth")
        print("  RTMW3D-X: https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth")
        print("\nSet pose_checkpoint variable in main()")
        return

    if not os.path.exists(pose_checkpoint):
        raise FileNotFoundError(f"Pose checkpoint not found: {pose_checkpoint}")

    print("=" * 60)
    print("RTMPose 3D Video Inference with Pre-computed Bounding Boxes")
    print("=" * 60)

    # Load pre-computed bounding boxes
    print("\n[1/4] Loading pre-computed bounding boxes...")
    bboxes_data = np.load(bboxes_path)  # Shape: (nframes, 5) - [x1, y1, x2, y2, score]
    print(f"  Loaded {len(bboxes_data)} bounding boxes")
    print(f"  Shape: {bboxes_data.shape}")

    # Open video
    print("\n[2/4] Opening video...")
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

    if len(bboxes_data) != total_frames:
        print(f"  Note: Bounding boxes count ({len(bboxes_data)}) != video frames ({total_frames})")

    # Initialize pose model
    print("\n[3/4] Loading RTMPose 3D model...")
    print(f"  Config: {pose_config}")
    print(f"  Checkpoint: {pose_checkpoint}")
    pose_model = init_model(pose_config, pose_checkpoint, device=device)
    print("  Pose model loaded")

    # Configure visualizer
    pose_model.cfg.model.test_cfg.mode = 'vis'
    pose_model.cfg.visualizer.radius = radius
    pose_model.cfg.visualizer.line_width = thickness

    # Get dataset metadata for visualization
    det_kpt_color = pose_model.dataset_meta.get('keypoint_colors', None)
    det_skeleton = pose_model.dataset_meta.get('skeleton_links', None)
    det_link_color = pose_model.dataset_meta.get('skeleton_link_colors', None)

    pose_model.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_model.cfg.visualizer.det_dataset_skeleton = det_skeleton
    pose_model.cfg.visualizer.det_dataset_link_color = det_link_color
    pose_model.cfg.visualizer.skeleton = det_skeleton
    pose_model.cfg.visualizer.link_color = det_link_color
    pose_model.cfg.visualizer.kpt_color = det_kpt_color

    visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)

    # Setup video writer (will be initialized after first frame)
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
        bbox = bbox_data[:4].astype(np.float32).reshape(1, 4)  # (1, 4)
        bbox_score = bbox_data[4]

        # Run pose estimation if bbox is valid
        if bbox_score > 0 and bbox[0, 2] > bbox[0, 0] and bbox[0, 3] > bbox[0, 1]:
            # Run inference - returns just 3D skeleton visualization
            frame = process_frame(
                frame,
                bbox,
                pose_model,
                visualizer,
                disable_rebase=disable_rebase,
                kpt_thr=kpt_thr,
                show=False
            )
        else:
            # No valid bbox - show blank white frame
            frame = np.ones_like(frame) * 255

        # Initialize video writer on first frame (after we know the actual output size)
        if out is None:
            output_height, output_width = frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            print(f"  Output video dimensions: {output_width}x{output_height}")

        # Write frame to output video
        out.write(frame)

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