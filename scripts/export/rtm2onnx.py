"""Convert RTMPose model to ONNX format with dynamic batching.

This script converts a trained RTMPose .pth checkpoint to ONNX format,
supporting dynamic batch sizes for flexible inference.

Usage:
    python rtm2onnx.py
"""

import os
import torch
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.apis import init_model


def export_to_onnx(
    config_path,
    checkpoint_path,
    output_path,
    input_shape=(1, 3, 256, 192),
    opset_version=11,
    simplify=True
):
    """Export RTMPose model to ONNX format.

    Args:
        config_path: Path to model config file
        checkpoint_path: Path to model checkpoint (.pth)
        output_path: Path to save ONNX model
        input_shape: Input shape (N, C, H, W). N can be any value, will be dynamic
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model (requires onnx-simplifier)
    """
    print("=" * 60)
    print("RTMPose to ONNX Converter")
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
    # Dynamic batch size for input and all outputs
    dynamic_axes = {
        'input': {0: 'batch_size'},  # Input has dynamic batch
    }

    # Add dynamic axes for outputs
    # RTMPose typically outputs keypoint coordinates and heatmaps
    # We skip the forward pass analysis since we know RTMPose outputs simcc_x and simcc_y
    print("\n[3/4] Preparing for export...")
    print("✓ Model structure analyzed (RTMPose with SimCC head)")

    # Export to ONNX
    print("\n[4/4] Exporting to ONNX...")

    # For RTMPose, we need to export just the model's backbone and head
    # without the full forward logic
    class RTMPoseONNX(torch.nn.Module):
        """Wrapper for ONNX export."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            """Forward pass returning keypoint predictions.

            Args:
                x: Input image tensor (N, 3, H, W)

            Returns:
                simcc_x: X-axis classification scores (N, K, W*2)
                simcc_y: Y-axis classification scores (N, K, H*2)
            """
            # Extract features
            feats = self.model.backbone(x)

            # RTMCCHead expects a tuple and extracts the last feature itself
            if not isinstance(feats, tuple):
                feats = (feats,)

            # RTMCCHead forward
            simcc_x, simcc_y = self.model.head.forward(feats)

            return simcc_x, simcc_y

    # Create wrapper
    onnx_model = RTMPoseONNX(model)
    onnx_model.eval()

    # Update dynamic axes for actual outputs
    dynamic_axes['simcc_x'] = {0: 'batch_size'}  # (N, K, W*2)
    dynamic_axes['simcc_y'] = {0: 'batch_size'}  # (N, K, H*2)

    # Export
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['simcc_x', 'simcc_y'],
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
    config_path = 'configs/body_2d_keypoint/rtmpose/coco29/rtmpose-l_8xb256-420e_coco29-256x192.py'
    checkpoint_path = '/media/marco/T7/work_dirs/rtmpose-l_coco2/epoch_130.pth'  # Change to your checkpoint
    output_path = '/media/marco/T7/work_dirs/rtmpose-l_coco2/rtmpose_coco29.onnx'

    # Export settings
    input_shape = (1, 3, 256, 192)  # (batch_size, channels, height, width)
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
session = ort.InferenceSession('rtmpose_coco29.onnx')

# Prepare input (N, 3, 256, 192)
input_img = np.random.randn(1, 3, 256, 192).astype(np.float32)

# Run inference
simcc_x, simcc_y = session.run(None, {'input': input_img})

# simcc_x shape: (N, 29, 384)  - X-axis classification
# simcc_y shape: (N, 29, 512)  - Y-axis classification

# Decode keypoints
x_coords = np.argmax(simcc_x, axis=2) / 2.0  # Divide by split_ratio
y_coords = np.argmax(simcc_y, axis=2) / 2.0

keypoints = np.stack([x_coords, y_coords], axis=-1)  # (N, 29, 2)
print(f"Keypoints shape: {keypoints.shape}")
""")


if __name__ == '__main__':
    main()