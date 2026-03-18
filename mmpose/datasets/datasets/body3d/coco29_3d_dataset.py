"""COCO29 3D pose dataset for RTMPose 3D training."""

from mmpose.datasets.datasets.body3d import Body3DMocapDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class Coco293DDataset(Body3DMocapDataset):
    """COCO29 3D whole-body pose dataset.

    This dataset contains 29 keypoints (COCO17 body + feet + hands) with
    3D annotations in camera coordinate space.

    Annotation format:
        {
            "keypoints": [x1, y1, v1, x2, y2, v2, ...],  # 2D keypoints (29 × 3)
            "keypoints_3d": [X1, Y1, Z1, v1, ...],  # 3D keypoints (29 × 4)
            "bbox": [x, y, w, h],
            "camera_intrinsics": {
                "fx": focal_length_x,
                "fy": focal_length_y,
                "cx": principal_point_x,
                "cy": principal_point_y
            }
        }

    Args:
        ann_file (str): Annotation file path. The annotation format is COCO
            with 3D extensions.
        camera_param_file (str, optional): Camera parameter file path. Not
            needed if camera_intrinsics are in annotations.
        data_mode (str): Specifies the mode of data (topdown or bottomup).
            Default: 'topdown'
        seq_len (int): Number of frames in a sequence. Default: 1
        causal (bool): If True, only use past frames. Default: True
        pad_video_seq (bool): Whether to pad sequences. Default: False
        camera_param (dict, optional): Default camera parameters.
    """

    METAINFO: dict = dict(
        dataset_name='coco29_3d',
        paper_info=dict(
            author='Your Organization',
            title='COCO29 3D Dataset',
            year='2024',
        ),
        num_keypoints=29,
        keypoint_id2name={
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle',
            17: 'left_big_toe',
            18: 'left_small_toe',
            19: 'left_heel',
            20: 'right_big_toe',
            21: 'right_small_toe',
            22: 'right_heel',
            23: 'left_thumb',
            24: 'left_index',
            25: 'left_middle',
            26: 'left_ring',
            27: 'left_pinky',
            28: 'right_thumb',
        },
        keypoint_name2id={
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16,
            'left_big_toe': 17,
            'left_small_toe': 18,
            'left_heel': 19,
            'right_big_toe': 20,
            'right_small_toe': 21,
            'right_heel': 22,
            'left_thumb': 23,
            'left_index': 24,
            'left_middle': 25,
            'left_ring': 26,
            'left_pinky': 27,
            'right_thumb': 28,
        },
        # Skeleton connections for visualization
        skeleton_links=[
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [0, 5], [0, 6],  # Shoulders to nose
            [5, 7], [7, 9],  # Left arm
            [6, 8], [8, 10],  # Right arm
            [5, 11], [6, 12], [11, 12],  # Torso
            [11, 13], [13, 15],  # Left leg
            [12, 14], [14, 16],  # Right leg
            [15, 17], [15, 18], [15, 19],  # Left foot
            [16, 20], [16, 21], [16, 22],  # Right foot
            [9, 23], [9, 24], [9, 25], [9, 26], [9, 27],  # Left hand
            [10, 28],  # Right hand thumb (add more if you have all 5 fingers)
        ],
        # Joint angles for bone loss
        joint_angles=[],
        # Joints for MPJPE evaluation
        joint_weights=[1.0] * 29,
        # Keypoints for PCK evaluation
        sigmas=[0.026] * 29,
    )

    def __init__(self,
                 ann_file: str = '',
                 camera_param_file: str = '',
                 data_mode: str = 'topdown',
                 seq_len: int = 1,
                 causal: bool = True,
                 pad_video_seq: bool = False,
                 camera_param: dict = None,
                 **kwargs):

        # Default camera parameters if not provided
        if camera_param is None:
            camera_param = dict(
                f=[1000.0, 1000.0],  # Default focal length
                c=[960.0, 540.0],    # Default principal point
            )

        super().__init__(
            ann_file=ann_file,
            camera_param_file=camera_param_file,
            data_mode=data_mode,
            seq_len=seq_len,
            causal=causal,
            pad_video_seq=pad_video_seq,
            camera_param=camera_param,
            **kwargs)

    def _load_annotations(self):
        """Load annotations from COCO format with 3D extensions."""
        annotations = super()._load_annotations()

        # Parse camera intrinsics from annotations if available
        for ann in annotations:
            if 'camera_intrinsics' in ann:
                cam_params = ann['camera_intrinsics']
                # Store camera parameters for this sample
                ann['camera_param'] = dict(
                    f=[cam_params['fx'], cam_params['fy']],
                    c=[cam_params['cx'], cam_params['cy']]
                )

        return annotations