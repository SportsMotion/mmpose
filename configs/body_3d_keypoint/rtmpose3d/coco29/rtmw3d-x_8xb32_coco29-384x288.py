"""RTMPose 3D (RTMW3D-X) config for COCO29 3D dataset.

This config finetunes the pretrained RTMW3D-X model on COCO29 3D data.
"""

_base_ = ['../../../_base_/default_runtime.py']

custom_imports = dict(imports=['rtmpose3d'], allow_failed_imports=False)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Runtime
max_epochs = 100
stage2_num_epochs = 10
base_lr = 5e-5  # Lower learning rate for finetuning
num_keypoints = 29  # Changed from 133 to 29

train_cfg = dict(max_epochs=max_epochs, val_interval=5)
randomness = dict(seed=2024)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# Learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# Codec settings for COCO29 3D
codec = dict(
    type='SimCC3DLabel',
    input_size=(288, 384, 288),
    sigma=(6., 6.93, 6.),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
    root_index=(11, 12))  # Hip keypoints for root normalization

# Pretrained backbone path (load from RTMW3D pretrained)
backbone_path = 'https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth'  # noqa

# Model settings
model = dict(
    type='TopdownPoseEstimator3D',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.33,
        widen_factor=1.25,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=backbone_path)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    head=dict(
        type='RTMW3DHead',
        in_channels=1280,
        out_channels=num_keypoints,  # Changed to 29
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=[
            dict(
                type='KLDiscretLossWithWeight',
                use_target_weight=True,
                beta=10.,
                label_softmax=True),
            dict(
                type='BoneLoss',
                joint_parents=[
                    # Define parent joints for bone loss
                    # Format: parent_joint_id for each joint
                    -1,  # 0: nose (no parent)
                    0,   # 1: left_eye -> nose
                    0,   # 2: right_eye -> nose
                    1,   # 3: left_ear -> left_eye
                    2,   # 4: right_ear -> right_eye
                    0,   # 5: left_shoulder -> nose
                    0,   # 6: right_shoulder -> nose
                    5,   # 7: left_elbow -> left_shoulder
                    6,   # 8: right_elbow -> right_shoulder
                    7,   # 9: left_wrist -> left_elbow
                    8,   # 10: right_wrist -> right_elbow
                    5,   # 11: left_hip -> left_shoulder
                    6,   # 12: right_hip -> right_shoulder
                    11,  # 13: left_knee -> left_hip
                    12,  # 14: right_knee -> right_hip
                    13,  # 15: left_ankle -> left_knee
                    14,  # 16: right_ankle -> right_knee
                    15,  # 17: left_big_toe -> left_ankle
                    15,  # 18: left_small_toe -> left_ankle
                    15,  # 19: left_heel -> left_ankle
                    16,  # 20: right_big_toe -> right_ankle
                    16,  # 21: right_small_toe -> right_ankle
                    16,  # 22: right_heel -> right_ankle
                    9,   # 23: left_thumb -> left_wrist
                    9,   # 24: left_index -> left_wrist
                    9,   # 25: left_middle -> left_wrist
                    9,   # 26: left_ring -> left_wrist
                    9,   # 27: left_pinky -> left_wrist
                    10,  # 28: right_thumb -> right_wrist
                ],
                use_target_weight=True,
                loss_weight=2.0)
        ],
        decoder=codec),
    test_cfg=dict(flip_test=False))

# Base dataset settings
data_mode = 'topdown'
backend_args = dict(backend='local')

# Pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='Coco293DDataset',
        data_root='data/coco29_3d/',
        ann_file='annotations/train.json',
        data_prefix=dict(img=''),
        seq_len=1,
        causal=True,
        data_mode=data_mode,
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='Coco293DDataset',
        data_root='data/coco29_3d/',
        ann_file='annotations/val.json',
        data_prefix=dict(img=''),
        seq_len=1,
        causal=True,
        data_mode=data_mode,
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

# Evaluators
val_evaluator = [
    dict(type='SimpleMPJPE', mode='mpjpe'),
    dict(type='SimpleMPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator

# Hooks
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=3))

# Load pretrained weights for finetuning
load_from = 'https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth'  # noqa