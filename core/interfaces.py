import abc
from typing import Dict, Any
import torch.nn as nn
from .entities import VideoContext, DetectionResult

class BaseVideoLoader(abc.ABC):
    @abc.abstractmethod
    def load(self, video_path: str) -> VideoContext:
        pass

class BaseFaceTracker(abc.ABC):
    @abc.abstractmethod
    def process(self, context: VideoContext) -> VideoContext:
        pass

class BaseModality(nn.Module, abc.ABC):
    """Базовый класс для всех модальностей"""
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Название модальности (используется для агрегации)"""
        pass

    @abc.abstractmethod
    def preprocess(self, context: VideoContext) -> Any:
        """Подготовка данных (например, ресайз и нормализация кропов лица)"""
        pass

    @abc.abstractmethod
    def forward(self, preprocessed_data: Any) -> Any:
        """Инференс модели (использование весов, например, HiddenCLIP)"""
        pass

    @abc.abstractmethod
    def postprocess(self, raw_output: Any) -> DetectionResult:
        """Упаковка сырого выхода модели в стандартизированный DetectionResult"""
        pass

    def process(self, context: VideoContext) -> DetectionResult:
        """Весь жизненный цикл обработки для конкретной модальности"""
        data = self.preprocess(context)
        out = self.forward(data)
        return self.postprocess(out)

class BaseAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(self, results: Dict[str, DetectionResult]) -> DetectionResult:
        pass
