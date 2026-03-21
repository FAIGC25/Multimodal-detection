import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from core.interfaces import BaseModality
from core.entities import VideoContext, DetectionResult

# Заглушка для темпоральной модальности как примера 
class TemporalAnalyzer(BaseModality):
    def __init__(self):
        super().__init__()
        print("[temporal] Инициализация: загрузка Spatial Feature Extractor (MobileNetV3) и LSTM...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 1. Feature Extractor 
        # Мы используем MobileNetV3 для быстрого извлечения признаков каждого кадра
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.feature_extractor = mobilenet_v3_small(weights=weights).features
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Замораживаем веса 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Трансформации, нужные для MobileNet
        self.transform = weights.transforms()

        # 2. LSTM
        self.lstm = nn.LSTM(input_size=576, hidden_size=64, batch_first=True)
        self.lstm.to(self.device)
        
        # 3. Классификатор 
        self.fc = nn.Linear(64, 1)
        self.fc.to(self.device)
        self.sigmoid = nn.Sigmoid()

    @property
    def name(self) -> str:
        return "temporal"

    def preprocess(self, context: VideoContext) -> torch.Tensor:
        print(f"[{self.name}] Подготовка темпоральных данных (Извлечение визуальных признаков)...")
        if not context.face_tracks:
            raise ValueError(f"Лица не найдены! {self.name} модальность не имеет данных.")
        
        frames = context.face_tracks[0].frames
        
        # Применяем стандартную трансформацию (ожидает [C, H, W] или батч)
        processed_frames = self.transform(frames).to(self.device)
        
        # Извлекаем признаки из каждого кадра: (T, C, H, W) -> (T, 576, X, Y)
        with torch.no_grad():
            features = self.feature_extractor(processed_frames) 
            features = features.mean(dim=[2, 3])

        return features.unsqueeze(0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        print(f"[{self.name}] Инференс темпоральной модели (анализ динамики кадров)...")
        
        # features: (1, T, 576) -> out: (1, T, 64)
        out, (hn, cn) = self.lstm(features)
        
        # Берем выход последнего скрытого состояния
        last_hidden = out[:, -1, :] # (1, 64)
        
        # Выдаем единый скор (0-1)
        score = self.sigmoid(self.fc(last_hidden))
        return score.squeeze() 

    def postprocess(self, raw_output: torch.Tensor) -> DetectionResult:
        print(f"[{self.name}] Упаковка результатов...")
        
        # Заглушка 
        if raw_output.dim() == 0:
            score = raw_output.item()
        else:
            score = raw_output[0].item()
            
        return DetectionResult(
            score=score,
            metadata={"artifacts_found": "проанализировано LSTM"}
        )
