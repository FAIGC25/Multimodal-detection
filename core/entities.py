import torch
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class FaceTrack:
    """Сущность для хранения трека лица на протяжении видео"""
    track_id: int
    frames: torch.Tensor  # (T, C, H, W) - кропы лица
    # Здесь могут быть дополнительные метаданные 

@dataclass
class VideoContext:
    """Единый контекст, который передается по пайплайну"""
    video_path: str
    raw_frames: torch.Tensor  # (T, C, H, W) полный кадр
    fps: float
    face_tracks: List[FaceTrack] = field(default_factory=list)
    depth_maps: List[Any] = field(default_factory=list)

@dataclass
class DetectionResult:
    """Стандартизированный выход любой модальности или агрегатора"""
    score: float       
    metadata: Dict[str, Any] = field(default_factory=dict)
