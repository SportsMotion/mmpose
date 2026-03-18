"""Video inference script for COCO29 RTMPose model with pre-computed bounding boxes.

This script loads pre-computed person bounding boxes from a .npy file and runs
pose estimation on each frame of a video, then saves the output with visualized
keypoints and bounding boxes.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from mmpose.apis import inference_topdown, init_model


def visualize_keypoints(image, all_keypoints, all_scores, kpt_thr=0.3):
    """Visualize predicted keypoints on image for multiple people.

    Args:
        image: Input image (H, W, 3)
        all_keypoints: All keypoints from all detected people (N, 29, 2)
        all_scores: All keypoint scores (N, 29)
        kpt_thr: Keypoint score threshold

    Returns:
        Image with visualized keypoints
    """
    img_vis = image.copy()

    # Define colors for different keypoint groups
    # Body (0-16): Red
    # Feet (17-22): Green
    # Hands (23-28): Blue
    colors = []
    for i in range(29):
        if i <= 16:  # Body keypoints
            colors.append((0, 0, 255))  # Red
        elif i <= 22:  # Feet keypoints
            colors.append((0, 255, 0))  # Green
        else:  # Hand keypoints
            colors.append((255, 0, 0))  # Blue

    # Draw skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (0, 5), (0, 6), (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]

    # Process each detected person
    for person_idx in range(len(all_keypoints)):
        keypoints = all_keypoints[person_idx]  # (29, 2)
        scores = all_scores[person_idx]  # (29,)

        # Draw skeleton first (so keypoints are on top)
        for start_idx, end_idx in skeleton:
            if scores[start_idx] > kpt_thr and scores[end_idx] > kpt_thr:
                start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(img_vis, start_pt, end_pt, (255, 255, 0), 2)

        # Draw keypoints
        for idx, (kpt, score) in enumerate(zip(keypoints, scores)):
            if score > kpt_thr:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(img_vis, (x, y), 5, colors[idx], -1)
                cv2.circle(img_vis, (x, y), 6, (255, 255, 255), 1)

    return img_vis

you
def main():
    # ======================== CONFIGURATION ========================
    # Paths
    pose_checkpoint = '/home/marco/Desktop/phoenix/models/rtmPose-l_coco29/rtmpose_coco29.pth'
    pose_config = 'configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py'
    video_path = '/home/marco/Desktop/phoenix/emx_run/input/videos/cam03.mp4'  # UPDATE THIS PATH
    bboxes_path = '/home/marco/Desktop/phoenix/emx_tracking/realtime_tracking/experiments/yolo_detections.npy'  # UPDATE THIS PATH
    output_path = 'output_video.mp4'

    # Settings
    device = 'cuda:0'
    kpt_thr = 0.3  # Keypoint score threshold for visualization

    # ===============================================================

    # Check if files exist
    if not os.path.exists(pose_checkpoint):
        raise FileNotFoundError(f"Pose checkpoint not found: {pose_checkpoint}")
    if not os.path.exists(pose_config):
        raise FileNotFoundError(f"Pose config not found: {pose_config}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(bboxes_path):
        raise FileNotFoundError(f"Bounding boxes file not found: {bboxes_path}")

    print("=" * 60)
    print("RTMPose COCO29 Video Inference")
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

    # Check if bboxes match frame count
    if len(bboxes_data) != total_frames:
        print(f"  Warning: Bounding boxes count ({len(bboxes_data)}) != video frames ({total_frames})")
        print(f"  Will process minimum of both: {min(len(bboxes_data), total_frames)} frames")

    # Initialize pose model
    print("\n[3/4] Loading pose model (RTMPose COCO29)...")
    print(f"  Config: {pose_config}")
    print(f"  Checkpoint: {pose_checkpoint}")
    pose_model = init_model(pose_config, pose_checkpoint, device=device)
    print("  Pose model loaded")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frame by frame
    print("\n[4/4] Processing video frames...")
    frame_idx = 0
    pbar = tqdm(total=min(len(bboxes_data), total_frames), desc="Processing frames")

    while cap.isOpened() and frame_idx < len(bboxes_data):
        ret, frame = cap.read()
        if not ret:
            break

        # Get bounding box for this frame
        bbox_data = bboxes_data[frame_idx]  # [x1, y1, x2, y2, score]
        bbox = bbox_data[:4].astype(np.float32).reshape(1, 4)  # (1, 4)
        bbox_score = bbox_data[4]

        # Run pose estimation if bbox is valid
        if bbox_score > 0 and bbox[0, 2] > bbox[0, 0] and bbox[0, 3] > bbox[0, 1]:
            # Run inference
            pose_results = inference_topdown(pose_model, frame, bboxes=bbox)

            # Extract keypoints and scores
            if len(pose_results) > 0:
                keypoints = pose_results[0].pred_instances.keypoints[0]  # (29, 2)
                scores = pose_results[0].pred_instances.keypoint_scores[0]  # (29,)

                # Visualize keypoints
                frame = visualize_keypoints(
                    frame,
                    keypoints.reshape(1, 29, 2),
                    scores.reshape(1, 29),
                    kpt_thr=kpt_thr
                )

            # Draw bounding box
            x1, y1, x2, y2 = bbox[0].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Score: {bbox_score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Write frame to output video
        out.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"\nProcessed {frame_idx} frames")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()