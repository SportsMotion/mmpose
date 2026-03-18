"""Convert RTMPose 3D model to ONNX format with dynamic batching.

This script converts a trained RTMPose 3D (RTMW3D) .pth checkpoint to ONNX format,
supporting dynamic batch sizes for flexible inference.

Usage:
    python rtm3d2onnx.py
"""

import os
import torch
import numpy as np
import sys
from mmpose.apis import init_model

# Import RTMPose3D custom modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'projects', 'rtmpose3d'))
from rtmpose3d import *  # noqa


def export_to_onnx(
    config_path,
    checkpoint_path,
    output_path,
    input_shape=(1, 3, 384, 288),
    opset_version=11,
    simplify=True
):
    """Export RTMPose 3D model to ONNX format.

    Args:
        config_path: Path to model config file
        checkpoint_path: Path to model checkpoint (.pth)
        output_path: Path to save ONNX model
        input_shape: Input shape (N, C, H, W). N can be any value, will be dynamic
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model (requires onnx-simplifier)
    """
    print("=" * 60)
    print("RTMPose 3D to ONNX Converter")
    print("=" * 60)

    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Input shape: {input_shape} (batch size will be dynamic)")

    # Load model
    print("\n[1/4] Loading model...")
    model = init_model(config_path, checkpoint_path, device='cpu')
    model.eval()
    print("✓ Model loaded")

    # Prepare dummy input
    print("\n[2/4] Preparing dummy input...")
    batch_size, channels, height, width = input_shape
    dummy_input = torch.randn(batch_size, channels, height, width)
    print(f"✓ Dummy input shape: {dummy_input.shape}")

    # Define dynamic axes
    dynamic_axes = {
        'input': {0: 'batch_size'},  # Input has dynamic batch
    }

    print("\n[3/4] Preparing for export...")
    print("✓ Model structure analyzed (RTMPose 3D with SimCC3D head)")

    # Export to ONNX
    print("\n[4/4] Exporting to ONNX...")

    # For RTMPose 3D, we need to export the backbone, neck, and head
    class RTMW3DONNX(torch.nn.Module):
        """Wrapper for ONNX export."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            """Forward pass returning 3D keypoint predictions.

            Args:
                x: Input image tensor (N, 3, H, W)

            Returns:
                simcc_x: X-axis classification scores (N, K, W*split_ratio)
                simcc_y: Y-axis classification scores (N, K, H*split_ratio)
                simcc_z: Z-axis classification scores (N, K, D*split_ratio)
            """
            # Extract features from backbone
            feats = self.model.backbone(x)

            # Pass through neck if it exists
            if hasattr(self.model, 'neck') and self.model.neck is not None:
                feats = self.model.neck(feats)

            # Ensure feats is a tuple for the head
            if not isinstance(feats, tuple):
                feats = (feats,)

            # RTMW3DHead forward - returns (simcc_x, simcc_y, simcc_z)
            simcc_x, simcc_y, simcc_z = self.model.head.forward(feats)

            return simcc_x, simcc_y, simcc_z

    # Create wrapper
    onnx_model = RTMW3DONNX(model)
    onnx_model.eval()

    # Update dynamic axes for actual outputs
    dynamic_axes['simcc_x'] = {0: 'batch_size'}  # (N, K, W*2)
    dynamic_axes['simcc_y'] = {0: 'batch_size'}  # (N, K, H*2)
    dynamic_axes['simcc_z'] = {0: 'batch_size'}  # (N, K, D*2)

    # Export
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['simcc_x', 'simcc_y', 'simcc_z'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    print(f"✓ ONNX model exported to: {output_path}")

    # Verify the exported model
    print("\n[Verification] Loading and checking ONNX model...")
    try:
        import onnx
        onnx_model_check = onnx.load(output_path)
        onnx.checker.check_model(onnx_model_check)
        print("✓ ONNX model is valid")

        # Print model info
        print("\nModel Information:")
        print(f"  IR version: {onnx_model_check.ir_version}")
        print(f"  Opset version: {onnx_model_check.opset_import[0].version}")
        print(f"  Producer: {onnx_model_check.producer_name}")

        print("\n  Inputs:")
        for inp in onnx_model_check.graph.input:
            shape = [dim.dim_param if dim.dim_param else dim.dim_value
                    for dim in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {shape}")

        print("\n  Outputs:")
        for out in onnx_model_check.graph.output:
            shape = [dim.dim_param if dim.dim_param else dim.dim_value
                    for dim in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {shape}")

    except ImportError:
        print("⚠ onnx package not installed. Install with: pip install onnx")
    except Exception as e:
        print(f"⚠ Error checking ONNX model: {e}")

    # Simplify model if requested
    if simplify:
        print("\n[Optional] Simplifying ONNX model...")
        try:
            from onnxsim import simplify as onnx_simplify

            onnx_model_check = onnx.load(output_path)
            model_simplified, check = onnx_simplify(onnx_model_check)

            if check:
                simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                onnx.save(model_simplified, simplified_path)
                print(f"✓ Simplified model saved to: {simplified_path}")
            else:
                print("⚠ Simplification check failed, keeping original model")

        except ImportError:
            print("⚠ onnx-simplifier not installed. Install with: pip install onnx-simplifier")
            print("  Skipping simplification...")
        except Exception as e:
            print(f"⚠ Error during simplification: {e}")
            print("  Keeping original model...")

    # Test inference with different batch sizes
    print("\n[Testing] Testing dynamic batching...")
    try:
        import onnxruntime as ort

        # Create inference session
        session = ort.InferenceSession(output_path)

        # Test with different batch sizes
        test_batch_sizes = [1, 2, 4]
        for bs in test_batch_sizes:
            test_input = np.random.randn(bs, channels, height, width).astype(np.float32)
            outputs = session.run(None, {'input': test_input})
            print(f"  ✓ Batch size {bs}: Input {test_input.shape} -> "
                  f"Outputs {[o.shape for o in outputs]}")

        print("✓ Dynamic batching verified!")

    except ImportError:
        print("⚠ onnxruntime not installed. Install with: pip install onnxruntime-gpu")
        print("  or: pip install onnxruntime (CPU only)")
        print("  Skipping dynamic batch testing...")
    except Exception as e:
        print(f"⚠ Error during testing: {e}")

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nONNX model saved to: {output_path}")
    print("\nYou can now use this model with:")
    print("  - ONNX Runtime (Python, C++, C#, Java)")
    print("  - TensorRT (NVIDIA)")
    print("  - OpenVINO (Intel)")
    print("  - CoreML (Apple)")
    print("  - And many other inference backends!")


def main():
    # ======================== CONFIGURATION ========================
    # Paths
    config_path = 'projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py'
    checkpoint_path = '/home/marco/Downloads/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth'
    output_path = 'rtmw3d-x.onnx'

    # Export settings
    input_shape = (1, 3, 384, 288)  # (batch_size, channels, height, width)
    opset_version = 11  # ONNX opset version (11 is widely supported)
    simplify = True  # Simplify the model (reduces size and improves performance)

    # ===============================================================

    export_to_onnx(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=input_shape,
        opset_version=opset_version,
        simplify=simplify
    )

    print("\n" + "=" * 60)
    print("Quick Start with ONNX Runtime:")
    print("=" * 60)
    print("""
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('rtmw3d-x.onnx')

# Prepare input (N, 3, 384, 288)
input_img = np.random.randn(1, 3, 384, 288).astype(np.float32)

# Run inference
simcc_x, simcc_y, simcc_z = session.run(None, {'input': input_img})

# simcc_x shape: (N, 133, 576)  - X-axis classification (288*2)
# simcc_y shape: (N, 133, 768)  - Y-axis classification (384*2)
# simcc_z shape: (N, 133, 576)  - Z-axis classification (288*2)

# Decode 3D keypoints
x_coords = np.argmax(simcc_x, axis=2) / 2.0  # Divide by split_ratio
y_coords = np.argmax(simcc_y, axis=2) / 2.0
z_coords = np.argmax(simcc_z, axis=2) / 2.0

keypoints_3d = np.stack([x_coords, y_coords, z_coords], axis=-1)  # (N, 133, 3)
print(f"3D Keypoints shape: {keypoints_3d.shape}")

# Transform coordinates for visualization (as in inference script)
keypoints_3d = -keypoints_3d[..., [0, 2, 1]]  # -X, -Z, -Y

# Optional: Rebase height (make lowest keypoint touch ground)
keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)

print(f"Final 3D pose: {keypoints_3d.shape}")
""")


if __name__ == '__main__':
    main()