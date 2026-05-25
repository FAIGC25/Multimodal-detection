import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.models import efficientnet_b0
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SIDA')))

from core.interfaces import BaseModality
from core.entities import VideoContext, DetectionResult


class DepthStudentDetector(BaseModality):
    """EfficientNet-B0 based distilled depth detector (3-class classification)"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Try local path first, then fall back to server path
            local_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../SIDA/ck/efficientnet_student_depth.pth')
            )
            server_path = "/mnt/tank/scratch/dstoronkin/SIDA/efficientnet_student_depth.pth"

            if os.path.exists(local_path):
                model_path = local_path
            elif os.path.exists(server_path):
                model_path = server_path
            else:
                raise FileNotFoundError(
                    f"EfficientNet student model not found at:\n"
                    f"  - {local_path}\n"
                    f"  - {server_path}\n"
                    f"Please specify model_path or move model to one of these locations."
                )

        super().__init__()
        print(f"[depth_student] Инициализация EfficientNet-B0 (дистиллированная модель). Путь к весам: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load EfficientNet-B0 backbone
        self.model = efficientnet_b0(weights=None)  # No ImageNet pretraining

        # Replace classifier for 3-class classification
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 3)  # [Real, Full Synthetic, Tampered]

        # Load distilled weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[depth_student] Веса загружены успешно")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @property
    def name(self) -> str:
        return "depth_student"

    def preprocess(self, context: VideoContext) -> dict:
        if not context.depth_maps or len(context.depth_maps) == 0:
            raise ValueError(f"Карты глубины не найдены! {self.name} не может обработать видео.")

        # Extract first depth map
        depth_map = context.depth_maps[0]

        # Convert tensor to numpy if needed
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        # Handle different depth map shapes
        if len(depth_map.shape) == 3 and depth_map.shape[0] == 1:
            depth_map = np.squeeze(depth_map, axis=0)  # (H, W)
        elif len(depth_map.shape) == 3 and depth_map.shape[-1] == 1:
            depth_map = np.squeeze(depth_map, axis=-1)  # (H, W)

        # Normalize to [0, 255]
        if depth_map.max() <= 1.0:
            depth_map = (depth_map * 255).astype(np.uint8)
        else:
            depth_map = depth_map.astype(np.uint8)

        # Convert depth map (grayscale) to RGB for model input
        image_rgb = Image.fromarray(depth_map).convert('RGB')

        # Apply transforms
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        print(f"[depth_student] Препроцессинг: depth_map shape={depth_map.shape}, image_tensor shape={image_tensor.shape}")

        return {"image": image_tensor}

    def forward(self, preprocessed_data: dict) -> dict:
        with torch.no_grad():
            logits = self.model(preprocessed_data["image"])

        print(f"[depth_student] Инференс: logits shape={logits.shape}, values={logits}")

        return {"logits": logits}

    def postprocess(self, raw_output: dict) -> DetectionResult:
        logits = raw_output["logits"][0]  # First (only) sample in batch
        probs = F.softmax(logits, dim=0)

        # Class mapping: 0=Real, 1=Full Synthetic, 2=Tampered
        real_prob = probs[0].item()
        synthetic_prob = probs[1].item()
        tampered_prob = probs[2].item()

        # Combine synthetic + tampered as "fake"
        fake_probability = synthetic_prob + tampered_prob

        print(f"[depth_student] Постпроцессинг: Real={real_prob:.4f}, Synthetic={synthetic_prob:.4f}, Tampered={tampered_prob:.4f}")
        print(f"[depth_student] Итоговая вероятность фейка: {fake_probability:.4f}")

        return DetectionResult(
            score=fake_probability,
            metadata={
                "reasoning": f"Real:{real_prob:.4f} | Synthetic:{synthetic_prob:.4f} | Tampered:{tampered_prob:.4f}",
                "type": "depth_student",
                "confidence": fake_probability,
                "class_probs": {
                    "real": float(real_prob),
                    "full_synthetic": float(synthetic_prob),
                    "tampered": float(tampered_prob)
                }
            }
        )
