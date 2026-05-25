#!/usr/bin/env python3
"""
Quick test script to verify DepthStudentDetector loads and runs
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

from modalities.depth_student import DepthStudentDetector
from core.entities import VideoContext


def test_model_loading():
    """Test that the model loads without errors"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)

    try:
        detector = DepthStudentDetector()
        print("✓ Model loaded successfully!")
        print(f"  Device: {detector.device}")
        print(f"  Model type: {type(detector.model)}")
        return detector
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None


def test_inference(detector):
    """Test inference with dummy depth map"""
    print("\n" + "="*60)
    print("TEST 2: Inference on Dummy Depth Map")
    print("="*60)

    try:
        # Create a dummy depth map
        depth_map = torch.rand(224, 224)  # Random depth values in [0, 1]

        # Create VideoContext with dummy data
        context = VideoContext(
            video_path="dummy_video.mp4",
            raw_frames=torch.rand(1, 3, 224, 224),  # RGB frames (not used)
            fps=30
        )
        context.depth_maps = [depth_map]

        # Run inference
        preprocessed = detector.preprocess(context)
        raw_output = detector.forward(preprocessed)
        result = detector.postprocess(raw_output)

        print("✓ Inference successful!")
        print(f"  Score: {result.score:.4f}")
        print(f"  Metadata: {result.metadata}")

        return result

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_real_depth_map():
    """Test inference with a real depth map image"""
    print("\n" + "="*60)
    print("TEST 3: Inference on Real Depth Map Image")
    print("="*60)

    # Create a realistic depth map (e.g., from a file or generated)
    # For now, we'll create a synthetic one with some structure
    try:
        # Create a more realistic depth map (e.g., face-like shape)
        depth_map = np.zeros((224, 224), dtype=np.float32)

        # Add some structure (circles, gradients, etc.)
        for i in range(224):
            for j in range(224):
                # Distance from center
                dist = np.sqrt((i - 112) ** 2 + (j - 112) ** 2)
                depth_map[i, j] = np.exp(-dist / 50)  # Gaussian

        depth_map = torch.from_numpy(depth_map).float()

        context = VideoContext(
            video_path="dummy_video.mp4",
            raw_frames=torch.rand(1, 3, 224, 224),
            fps=30
        )
        context.depth_maps = [depth_map]

        detector = DepthStudentDetector()
        preprocessed = detector.preprocess(context)
        raw_output = detector.forward(preprocessed)
        result = detector.postprocess(raw_output)

        print("✓ Real depth map inference successful!")
        print(f"  Score: {result.score:.4f}")
        print(f"  Class probabilities: {result.metadata.get('class_probs', {})}")

        return result

    except Exception as e:
        print(f"✗ Real depth map inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "#"*60)
    print("# Testing DepthStudentDetector (EfficientNet-B0)")
    print("#"*60)

    # Test 1: Loading
    detector = test_model_loading()
    if detector is None:
        print("\n✗ Cannot proceed without loading model")
        return

    # Test 2: Dummy inference
    result = test_inference(detector)

    # Test 3: Real depth map
    result = test_real_depth_map()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
