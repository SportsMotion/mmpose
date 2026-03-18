"""Convert multi-camera 3D keypoints to COCO29 3D training format.

This script converts 3D keypoints from world/calibration space to camera space
and generates COCO-format annotations for RTMPose 3D training.

Input:
    - 3D keypoints in world space (N, 29, 3)
    - Camera calibrations (intrinsics + extrinsics) for each view
    - Images from multiple cameras

Output:
    - COCO-format JSON with 3D annotations
    - One entry per camera view per frame

Usage:
    python tools/convert_multicam_to_coco29_3d.py
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


def world_to_camera(points_3d_world, R, T):
    """Transform 3D points from world space to camera space.

    Args:
        points_3d_world: (N, 3) array of 3D points in world coordinates
        R: (3, 3) rotation matrix from world to camera
        T: (3,) translation vector from world to camera

    Returns:
        points_3d_camera: (N, 3) array of 3D points in camera coordinates
    """
    # Apply rotation and translation: X_cam = R @ X_world + T
    points_3d_camera = (R @ points_3d_world.T).T + T
    return points_3d_camera


def project_to_2d(points_3d_camera, K):
    """Project 3D points in camera space to 2D image coordinates.

    Args:
        points_3d_camera: (N, 3) array of 3D points in camera space
        K: (3, 3) camera intrinsic matrix
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
        depth: (N,) array of depth values (Z coordinates)
    """
    # Project: [u, v, 1] = K @ [X, Y, Z]
    points_2d_homogeneous = (K @ points_3d_camera.T).T

    # Normalize by depth
    depth = points_2d_homogeneous[:, 2]
    points_2d = points_2d_homogeneous[:, :2] / depth[:, np.newaxis]

    return points_2d, depth


def is_visible(points_2d, depth, image_width, image_height, min_depth=0.1):
    """Check if keypoints are visible in the image.

    Args:
        points_2d: (N, 2) array of 2D pixel coordinates
        depth: (N,) array of depth values
        image_width: Image width in pixels
        image_height: Image height in pixels
        min_depth: Minimum valid depth

    Returns:
        visibility: (N,) array of visibility flags (0=not visible, 1=visible)
    """
    visibility = np.ones(len(points_2d), dtype=int)

    # Check if within image bounds
    x, y = points_2d[:, 0], points_2d[:, 1]
    visibility[(x < 0) | (x >= image_width)] = 0
    visibility[(y < 0) | (y >= image_height)] = 0

    # Check if depth is valid (in front of camera)
    visibility[depth < min_depth] = 0

    # Set visibility to 2 for visible points (COCO convention: 2=labeled and visible)
    visibility[visibility == 1] = 2

    return visibility


def compute_bbox_from_keypoints(points_2d, visibility, padding=20):
    """Compute bounding box from visible keypoints.

    Args:
        points_2d: (N, 2) array of 2D keypoints
        visibility: (N,) array of visibility flags
        padding: Extra padding around keypoints

    Returns:
        bbox: [x, y, width, height] in COCO format
    """
    visible_points = points_2d[visibility > 0]

    if len(visible_points) == 0:
        return None

    x_min = max(0, visible_points[:, 0].min() - padding)
    y_min = max(0, visible_points[:, 1].min() - padding)
    x_max = visible_points[:, 0].max() + padding
    y_max = visible_points[:, 1].max() + padding

    width = x_max - x_min
    height = y_max - y_min

    return [float(x_min), float(y_min), float(width), float(height)]


def create_coco_annotation(
    annotation_id: int,
    image_id: int,
    keypoints_2d: np.ndarray,
    keypoints_3d_camera: np.ndarray,
    visibility: np.ndarray,
    bbox: List[float],
    camera_intrinsics: Dict[str, float]
) -> Dict:
    """Create a COCO-format annotation with 3D extensions.

    Args:
        annotation_id: Unique annotation ID
        image_id: Associated image ID
        keypoints_2d: (29, 2) array of 2D keypoints
        keypoints_3d_camera: (29, 3) array of 3D keypoints in camera space
        visibility: (29,) array of visibility flags
        bbox: [x, y, width, height]
        camera_intrinsics: Dict with fx, fy, cx, cy

    Returns:
        annotation: COCO-format annotation dict
    """
    # Flatten keypoints for COCO format: [x1, y1, v1, x2, y2, v2, ...]
    keypoints_flat = []
    for i in range(len(keypoints_2d)):
        keypoints_flat.extend([
            float(keypoints_2d[i, 0]),
            float(keypoints_2d[i, 1]),
            int(visibility[i])
        ])

    # Flatten 3D keypoints: [X1, Y1, Z1, v1, X2, Y2, Z2, v2, ...]
    keypoints_3d_flat = []
    for i in range(len(keypoints_3d_camera)):
        keypoints_3d_flat.extend([
            float(keypoints_3d_camera[i, 0]),
            float(keypoints_3d_camera[i, 1]),
            float(keypoints_3d_camera[i, 2]),
            int(visibility[i])
        ])

    num_visible = int(np.sum(visibility > 0))

    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,  # Person category
        "bbox": bbox,
        "area": float(bbox[2] * bbox[3]),
        "keypoints": keypoints_flat,
        "num_keypoints": num_visible,
        "keypoints_3d": keypoints_3d_flat,
        "camera_intrinsics": camera_intrinsics,
        "iscrowd": 0
    }

    return annotation


def convert_multicam_to_coco(
    keypoints_3d_world: np.ndarray,
    camera_calibrations: Dict[str, Dict],
    image_paths: Dict[str, List[str]],
    output_path: str,
    keypoint_names: List[str],
    skeleton_links: List[List[int]]
):
    """Convert multi-camera 3D keypoints to COCO format.

    Args:
        keypoints_3d_world: (num_frames, 29, 3) array of 3D keypoints in world space
        camera_calibrations: Dict mapping camera_id to calibration params
            {
                'cam01': {
                    'K': (3, 3) intrinsic matrix,
                    'R': (3, 3) rotation matrix (world to camera),
                    'T': (3,) translation vector (world to camera),
                    'width': image width,
                    'height': image height
                },
                ...
            }
        image_paths: Dict mapping camera_id to list of image paths
            {
                'cam01': ['frame_000.jpg', 'frame_001.jpg', ...],
                ...
            }
        output_path: Path to save COCO JSON file
        keypoint_names: List of 29 keypoint names
        skeleton_links: List of [start_idx, end_idx] for skeleton connections
    """
    print("=" * 60)
    print("Converting Multi-Camera 3D Data to COCO Format")
    print("=" * 60)

    num_frames = len(keypoints_3d_world)
    print(f"\nTotal frames: {num_frames}")
    print(f"Cameras: {list(camera_calibrations.keys())}")
    print(f"Keypoints per frame: {keypoints_3d_world.shape[1]}")

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "COCO29 3D Pose Dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": keypoint_names,
                "skeleton": skeleton_links
            }
        ]
    }

    image_id = 0
    annotation_id = 0

    # Process each frame
    for frame_idx in range(num_frames):
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}/{num_frames}...")

        kpts_world = keypoints_3d_world[frame_idx]  # (29, 3)

        # Process each camera view
        for cam_id, calib in camera_calibrations.items():
            # Get camera parameters
            K = calib['K']
            R = calib['R']
            T = calib['T']
            img_width = calib['width']
            img_height = calib['height']

            # Transform world to camera space
            kpts_camera = world_to_camera(kpts_world, R, T)

            # Project to 2D
            kpts_2d, depth = project_to_2d(kpts_camera, K)

            # Check visibility
            visibility = is_visible(kpts_2d, depth, img_width, img_height)

            # Skip if no visible keypoints
            if np.sum(visibility > 0) == 0:
                continue

            # Compute bounding box
            bbox = compute_bbox_from_keypoints(kpts_2d, visibility)
            if bbox is None:
                continue

            # Get image path
            if cam_id in image_paths and frame_idx < len(image_paths[cam_id]):
                image_path = image_paths[cam_id][frame_idx]
            else:
                image_path = f"{cam_id}/frame_{frame_idx:06d}.jpg"

            # Create image entry
            image_entry = {
                "id": image_id,
                "file_name": image_path,
                "width": img_width,
                "height": img_height,
                "camera_id": cam_id,
                "frame_idx": frame_idx
            }
            coco_data["images"].append(image_entry)

            # Create annotation
            camera_intrinsics = {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2])
            }

            annotation = create_coco_annotation(
                annotation_id=annotation_id,
                image_id=image_id,
                keypoints_2d=kpts_2d,
                keypoints_3d_camera=kpts_camera,
                visibility=visibility,
                bbox=bbox,
                camera_intrinsics=camera_intrinsics
            )
            coco_data["annotations"].append(annotation)

            image_id += 1
            annotation_id += 1

    # Save to JSON
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✓ Conversion complete!")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Output: {output_path}")


def main():
    # ======================== CONFIGURATION ========================

    # Example: Load your 3D keypoints (shape: num_frames, 29, 3)
    # Replace this with your actual data loading
    keypoints_3d_world = np.load('path/to/keypoints_3d_world.npy')  # (N, 29, 3)

    # Define camera calibrations
    camera_calibrations = {
        'cam01': {
            'K': np.array([
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0]
            ]),
            'R': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]),  # Replace with actual rotation
            'T': np.array([0.0, 0.0, 2000.0]),  # Replace with actual translation
            'width': 1920,
            'height': 1080
        },
        'cam02': {
            'K': np.array([
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0]
            ]),
            'R': np.array([
                [0.866, 0.0, -0.5],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.866]
            ]),  # Example: 30 degree rotation
            'T': np.array([1000.0, 0.0, 1732.0]),
            'width': 1920,
            'height': 1080
        },
        # Add more cameras as needed
    }

    # Define image paths for each camera
    image_paths = {
        'cam01': [f'cam01/frame_{i:06d}.jpg' for i in range(len(keypoints_3d_world))],
        'cam02': [f'cam02/frame_{i:06d}.jpg' for i in range(len(keypoints_3d_world))],
    }

    # Define your 29 keypoint names (COCO29 format)
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
        [0, 5], [0, 6], [5, 7], [7, 9],  # Left arm
        [6, 8], [8, 10],  # Right arm
        [5, 11], [6, 12], [11, 12],  # Torso
        [11, 13], [13, 15],  # Left leg
        [12, 14], [14, 16],  # Right leg
        [15, 17], [15, 18], [15, 19],  # Left foot
        [16, 20], [16, 21], [16, 22],  # Right foot
    ]

    # Output path
    output_path = 'data/coco29_3d/annotations/train.json'

    # ===============================================================

    # Run conversion
    convert_multicam_to_coco(
        keypoints_3d_world=keypoints_3d_world,
        camera_calibrations=camera_calibrations,
        image_paths=image_paths,
        output_path=output_path,
        keypoint_names=keypoint_names,
        skeleton_links=skeleton_links
    )

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("""
1. Verify the generated annotations:
   - Check a few samples manually
   - Visualize projected 2D keypoints on images

2. Create train/val split:
   - Split annotations into train.json and val.json

3. Create dataset class (see tools/create_coco29_3d_dataset.py)

4. Update config file:
   - Set num_keypoints=29
   - Point to your annotations

5. Start training!
""")


if __name__ == '__main__':
    main()