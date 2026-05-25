#!/usr/bin/env python3
"""
Compare SIDA Depth model vs EfficientNet-B0 Student model

Usage:
    python scripts/compare_depth_models.py \
        --dataset /mnt/tank/scratch/dstoronkin/depth \
        --output results.json \
        --num-samples 100
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modalities.depth import SIDADepthDetector
from modalities.depth_student import DepthStudentDetector
from data.video_loader import SimpleVideoLoader
from core.entities import VideoContext


class ModelComparator:
    """Compare two depth detection models"""

    def __init__(self, dataset_path, num_samples=None):
        self.dataset_path = dataset_path
        self.num_samples = num_samples

        self.loader = SimpleVideoLoader()

        # Load models
        print("\nLoading SIDA Depth model...")
        try:
            self.sida_model = SIDADepthDetector()
            self.sida_available = True
        except Exception as e:
            print(f"Warning: Could not load SIDA model: {e}")
            self.sida_available = False

        print("\nLoading EfficientNet Student model...")
        try:
            self.student_model = DepthStudentDetector()
            self.student_available = True
        except Exception as e:
            print(f"Warning: Could not load Student model: {e}")
            self.student_available = False

        if not (self.sida_available or self.student_available):
            raise RuntimeError("Could not load either model!")

        self.results = defaultdict(list)
        self.timings = {"sida": [], "student": []}
        self.memory_usage = {"sida": [], "student": []}

    def load_test_data(self):
        """Load test videos from dataset"""
        dataset_path = Path(self.dataset_path)

        # Categories: real, fake (with subdirs like background_aug, dlc, etc.)
        test_files = []

        half = self.num_samples // 2 if self.num_samples else 1000

        # Real videos
        real_dir = dataset_path / "real"
        real_files = []
        if real_dir.exists():
            for video_file in sorted(real_dir.glob("*.mp4"))[:half]:
                real_files.append((str(video_file), "real"))

        # Fake videos (collect from all subdirs, then trim)
        fake_dir = dataset_path / "fake"
        fake_files = []
        if fake_dir.exists():
            for subdir in sorted(fake_dir.iterdir()):
                if subdir.is_dir():
                    for video_file in sorted(subdir.glob("*.mp4")):
                        fake_files.append((str(video_file), "fake"))
        fake_files = fake_files[:half]

        test_files = real_files + fake_files
        print(f"\nLoaded {len(test_files)} test files ({len(real_files)} real, {len(fake_files)} fake)")
        if self.num_samples:
            test_files = test_files[:self.num_samples]

        return test_files

    def infer_with_model(self, model, video_path):
        """Run inference with a model"""
        try:
            # Load video/depth map
            context = self.loader.load(video_path)

            # Check if this is a depth map or regular video
            # Depth maps should be single channel, convert to have depth_maps
            if context.raw_frames.shape[1] == 1:  # Single channel
                context.depth_maps = [context.raw_frames[0]]
            else:
                # Assume it's already depth or create dummy
                context.depth_maps = [context.raw_frames[0].mean(dim=0)]

            # Infer
            start_time = time.time()
            preprocessed = model.preprocess(context)
            raw_output = model.forward(preprocessed)
            result = model.postprocess(raw_output)
            print(f"    Текст модели: {result.metadata.get('reasoning', '')[:150]}")
            end_time = time.time()

            latency = (end_time - start_time) * 1000  # ms

            # Get GPU memory if available
            gpu_mem = 0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

            return result.score, latency, gpu_mem

        except Exception as e:
            print(f"Error inferring {video_path}: {e}")
            return None, None, None

    def compare(self):
        """Run comparison on test dataset"""
        test_files = self.load_test_data()

        print("\n" + "="*70)
        print("STARTING COMPARISON TEST")
        print("="*70)

        y_true = []
        y_pred_sida = []
        y_pred_student = []

        for idx, (video_path, ground_truth) in enumerate(test_files):
            print(f"\n[{idx+1}/{len(test_files)}] Processing: {Path(video_path).name}")

            # Convert ground truth to binary (0=real, 1=fake)
            true_label = 0 if ground_truth == "real" else 1
            y_true.append(true_label)

            # SIDA inference
            if self.sida_available:
                score, latency, gpu_mem = self.infer_with_model(self.sida_model, video_path)
                if score is not None:
                    self.timings["sida"].append(latency)
                    self.memory_usage["sida"].append(gpu_mem)
                    pred = 1 if score > 0.5 else 0  # Binary classification
                    y_pred_sida.append(pred)
                    print(f"  SIDA:    score={score:.4f}, latency={latency:.1f}ms, gpu={gpu_mem:.2f}GB")
                else:
                    y_pred_sida.append(-1)  # Error

            # Student inference
            if self.student_available:
                score, latency, gpu_mem = self.infer_with_model(self.student_model, video_path)
                if score is not None:
                    self.timings["student"].append(latency)
                    self.memory_usage["student"].append(gpu_mem)
                    pred = 1 if score > 0.5 else 0
                    y_pred_student.append(pred)
                    print(f"  Student: score={score:.4f}, latency={latency:.1f}ms, gpu={gpu_mem:.2f}GB")
                else:
                    y_pred_student.append(-1)

        # Compute metrics
        metrics = self._compute_metrics(y_true, y_pred_sida, y_pred_student)
        return metrics

    def _compute_metrics(self, y_true, y_pred_sida, y_pred_student):
        """Compute accuracy, precision, recall, F1 for both models"""
        metrics = {}

        if self.sida_available and len([y for y in y_pred_sida if y != -1]) > 0:
            y_valid = [y for y, pred in zip(y_true, y_pred_sida) if pred != -1]
            y_pred_valid = [pred for pred in y_pred_sida if pred != -1]

            metrics["sida"] = {
                "accuracy": accuracy_score(y_valid, y_pred_valid),
                "precision": precision_score(y_valid, y_pred_valid, zero_division=0),
                "recall": recall_score(y_valid, y_pred_valid, zero_division=0),
                "f1": f1_score(y_valid, y_pred_valid, zero_division=0),
                "avg_latency_ms": np.mean(self.timings["sida"]) if self.timings["sida"] else None,
                "avg_gpu_memory_gb": np.mean(self.memory_usage["sida"]) if self.memory_usage["sida"] else None,
                "fps": 1000 / np.mean(self.timings["sida"]) if self.timings["sida"] else None,
            }

        if self.student_available and len([y for y in y_pred_student if y != -1]) > 0:
            y_valid = [y for y, pred in zip(y_true, y_pred_student) if pred != -1]
            y_pred_valid = [pred for pred in y_pred_student if pred != -1]

            metrics["student"] = {
                "accuracy": accuracy_score(y_valid, y_pred_valid),
                "precision": precision_score(y_valid, y_pred_valid, zero_division=0),
                "recall": recall_score(y_valid, y_pred_valid, zero_division=0),
                "f1": f1_score(y_valid, y_pred_valid, zero_division=0),
                "avg_latency_ms": np.mean(self.timings["student"]) if self.timings["student"] else None,
                "avg_gpu_memory_gb": np.mean(self.memory_usage["student"]) if self.memory_usage["student"] else None,
                "fps": 1000 / np.mean(self.timings["student"]) if self.timings["student"] else None,
            }

        return metrics

    def print_results(self, metrics):
        """Print formatted comparison results"""
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)

        if "sida" in metrics:
            print("\nSIDA Depth Model:")
            for key, val in metrics["sida"].items():
                if val is not None:
                    if "accuracy" in key or "precision" in key or "recall" in key or "f1" in key:
                        print(f"  {key:25s} {val:.4f}")
                    else:
                        print(f"  {key:25s} {val:.2f}")

        if "student" in metrics:
            print("\nEfficientNet-B0 Student Model:")
            for key, val in metrics["student"].items():
                if val is not None:
                    if "accuracy" in key or "precision" in key or "recall" in key or "f1" in key:
                        print(f"  {key:25s} {val:.4f}")
                    else:
                        print(f"  {key:25s} {val:.2f}")

        # Comparison
        if "sida" in metrics and "student" in metrics:
            print("\nComparison (Student vs SIDA):")
            sida_fps = metrics["sida"].get("fps", 0) or 0
            student_fps = metrics["student"].get("fps", 0) or 0
            if sida_fps > 0:
                speedup = student_fps / sida_fps
                print(f"  Speedup: {speedup:.2f}x faster")

            sida_acc = metrics["sida"].get("accuracy", 0) or 0
            student_acc = metrics["student"].get("accuracy", 0) or 0
            acc_diff = student_acc - sida_acc
            print(f"  Accuracy diff: {acc_diff:+.4f}")

        print("="*70 + "\n")

    def save_results(self, metrics, output_path):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare depth detection models")
    parser.add_argument("--dataset", type=str, default="/mnt/tank/scratch/dstoronkin/depth",
                        help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="comparison_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to test (None=all)")

    args = parser.parse_args()

    # Check dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path not found: {args.dataset}")
        return

    # Create comparator and run
    comparator = ModelComparator(args.dataset, num_samples=args.num_samples)
    metrics = comparator.compare()

    # Print and save results
    comparator.print_results(metrics)
    comparator.save_results(metrics, args.output)


if __name__ == "__main__":
    main()
