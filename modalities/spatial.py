import torch
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from core.interfaces import BaseModality
from core.entities import VideoContext, DetectionResult

class SpatialHiddenCLIP(BaseModality):
    def __init__(self):
        super().__init__()
        print("[spatial] Инициализация: загрузка весов HiddenCLIP (TorchScript)...")
        # Скачиваем с Hugging Face напрямую в кэш. 
        model_path = hf_hub_download(repo_id="yermandy/deepfake-detection", filename="model.torchscript")
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # Загружаем модель JIT
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Трансформации для CLIP (Resize + Normalize из пайплайна DeepfakeBench)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    @property
    def name(self) -> str:
        return "spatial"

    def preprocess(self, context: VideoContext) -> torch.Tensor:
        print(f"[{self.name}] Подготовка данных (Resize -> Normalize)...")
        if not context.face_tracks:
            raise ValueError(f"Лица не найдены! {self.name} модальность не может обработать видео.")
        
        # Берем кропы первого попавшегося лица
        face_frames = context.face_tracks[0].frames
        
        # Применяем трансформации (ожидается B, C, H, W или T, C, H, W)
        processed_frames = self.transform(face_frames)
        return processed_frames.to(self.device)

    def forward(self, preprocessed_data: torch.Tensor) -> torch.Tensor:
        print(f"[{self.name}] Реальный инференс модели HiddenCLIP...")
        with torch.no_grad():
            # Инференс: получаем логиты
            logits = self.model(preprocessed_data)
            
            # Переводим логиты в вероятность для КАЖДОГО кадра
            probs = torch.sigmoid(logits)
            
        # Возвращаем вероятности по всем кадрам, чтобы postprocess мог посчитать разброс
        return probs

    def postprocess(self, raw_output: torch.Tensor) -> DetectionResult:
        print(f"[{self.name}] Упаковка результатов...")

        score = raw_output.mean().item()
        std_dev = raw_output.std().item() if raw_output.numel() > 1 else 0.0
            
        return DetectionResult(
            score=score,
            metadata={
                "source": "huggingface:yermandy/deepfake-detection",
                "variance": float(std_dev)
            }
        )
