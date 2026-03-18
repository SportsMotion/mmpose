"""Example: Convert your multi-camera calibration data to COCO29 3D format.

This example shows how to load your actual calibration data and convert it
to the format required for RTMPose 3D training.

Adapt this script to match your data structure.
"""

import numpy as np
import json
from pathlib import Path
from convert_multicam_to_coco29_3d import convert_multicam_to_coco


def load_calibration_from_json(calib_file):
    """Load camera calibration from your calibration file.

    Args:
        calib_file: Path to calibration JSON/file

    Returns:
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix (world to camera)
        T: (3,) translation vector (world to camera)
        distortion: Distortion coefficients (optional)
    """
    # Example for OpenCV calibration format
    with open(calib_file, 'r') as f:
        calib = json.load(f)

    # Extract intrinsic matrix
    K = np.array([
        [calib['fx'], 0, calib['cx']],
        [0, calib['fy'], calib['cy']],
        [0, 0, 1]
    ])

    # Extract extrinsic matrix (rotation + translation)
    # If you have R_world_to_cam and T_world_to_cam directly:
    R = np.array(calib['R']).reshape(3, 3)
    T = np.array(calib['T'])

    # OR if you have camera position and orientation:
    # camera_pos = np.array(calib['position'])  # Camera position in world
    # camera_rot = np.array(calib['rotation'])  # Camera rotation (e.g., rodrigues or euler)
    # R = compute_rotation_matrix(camera_rot)
    # T = -R @ camera_pos  # Transform to world-to-camera

    return K, R, T


def example_usage():
    """Example showing how to convert your multi-camera data."""

    # ==================== YOUR DATA PATHS ====================

    # Path to your 3D keypoints in world space
    # Expected shape: (num_frames, 29, 3)
    keypoints_3d_path = 'path/to/your/keypoints_3d_world.npy'

    # Camera calibration directory
    calib_dir = Path('path/to/your/calibrations')

    # Image directory
    image_dir = Path('path/to/your/images')

    # Output path
    output_json = 'data/coco29_3d/annotations/train.json'

    # ==================== LOAD YOUR DATA ====================

    print("Loading 3D keypoints...")
    keypoints_3d_world = np.load(keypoints_3d_path)  # (num_frames, 29, 3)
    num_frames = len(keypoints_3d_world)
    print(f"Loaded {num_frames} frames with {keypoints_3d_world.shape[1]} keypoints")

    # ==================== LOAD CALIBRATIONS ====================

    print("\nLoading camera calibrations...")
    camera_calibrations = {}

    # Example: If you have calibrations named cam01.json, cam02.json, etc.
    camera_ids = ['cam01', 'cam02', 'cam03', 'cam04']

    for cam_id in camera_ids:
        calib_file = calib_dir / f'{cam_id}.json'

        if not calib_file.exists():
            print(f"Warning: Calibration not found for {cam_id}, skipping...")
            continue

        # Load calibration
        K, R, T = load_calibration_from_json(calib_file)

        # Get image dimensions (from first image or from calibration)
        sample_image_path = image_dir / cam_id / 'frame_000000.jpg'
        if sample_image_path.exists():
            import cv2
            img = cv2.imread(str(sample_image_path))
            height, width = img.shape[:2]
        else:
            # Default dimensions if images not available yet
            width, height = 1920, 1080

        camera_calibrations[cam_id] = {
            'K': K,
            'R': R,
            'T': T,
            'width': width,
            'height': height
        }

        print(f"  Loaded {cam_id}: {width}x{height}")

    # ==================== BUILD IMAGE PATHS ====================

    print("\nBuilding image paths...")
    image_paths = {}

    for cam_id in camera_calibrations.keys():
        cam_image_dir = image_dir / cam_id

        # Get all images for this camera
        images = sorted(cam_image_dir.glob('*.jpg'))

        if len(images) == 0:
            print(f"Warning: No images found for {cam_id}")
            # Generate paths anyway (images might be processed later)
            image_paths[cam_id] = [
                f'{cam_id}/frame_{i:06d}.jpg' for i in range(num_frames)
            ]
        else:
            # Use relative paths from data root
            image_paths[cam_id] = [
                str(img.relative_to(image_dir.parent)) for img in images[:num_frames]
            ]

        print(f"  {cam_id}: {len(image_paths[cam_id])} image paths")

    # ==================== DEFINE KEYPOINT STRUCTURE ====================

    # Your 29 keypoint names (adjust to match your skeleton)
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'left_big_toe', 'left_small_toe', 'left_heel',
        'right_big_toe', 'right_small_toe', 'right_heel',
        'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
        'right_thumb'
    ]

    # Define skeleton connections
    skeleton_links = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head
        [0, 5], [0, 6],  # Shoulders
        [5, 7], [7, 9],  # Left arm
        [6, 8], [8, 10],  # Right arm
        [5, 11], [6, 12], [11, 12],  # Torso
        [11, 13], [13, 15],  # Left leg
        [12, 14], [14, 16],  # Right leg
        [15, 17], [15, 18], [15, 19],  # Left foot
        [16, 20], [16, 21], [16, 22],  # Right foot
        [9, 23], [9, 24], [9, 25], [9, 26], [9, 27],  # Left hand
        [10, 28],  # Right hand
    ]

    # ==================== RUN CONVERSION ====================

    print("\n" + "=" * 60)
    print("Starting conversion...")
    print("=" * 60)

    convert_multicam_to_coco(
        keypoints_3d_world=keypoints_3d_world,
        camera_calibrations=camera_calibrations,
        image_paths=image_paths,
        output_path=output_json,
        keypoint_names=keypoint_names,
        skeleton_links=skeleton_links
    )

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nAnnotations saved to: {output_json}")
    print("\nNext steps:")
    print("1. Verify annotations by visualizing a few samples")
    print("2. Split into train/val sets if not done already")
    print("3. Start training with the config file")


if __name__ == '__main__':
    example_usage()