from typing import List, Dict
from core.entities import VideoContext, DetectionResult
from core.interfaces import BaseVideoLoader, BaseFaceTracker, BaseModality, BaseAggregator

class MultimodalDeepfakeSystem:
    """
    Главный оркестратор пайплайна:
    1. Загрузка видео -> 2. Трекинг лица (общий этап) -> 3. Модальности независимо -> 4. Агрегация
    """
    def __init__(
        self,
        video_loader: BaseVideoLoader,
        face_tracker: BaseFaceTracker,
        modalities: List[BaseModality],
        aggregator: BaseAggregator
    ):
        self.video_loader = video_loader
        self.face_tracker = face_tracker
        # Реестр модальностей (Словарь модальностей по их именам/ключам)
        self.modalities: Dict[str, BaseModality] = {m.name: m for m in modalities}
        self.aggregator = aggregator

    def predict(self, video_path: str) -> DetectionResult:
        # Этап 1: Чтение видео один раз
        context = self.video_loader.load(video_path)
        
        # Этап 2: Выделение общего признака (координаты лица, кропы)
        context = self.face_tracker.process(context)
        
        # Этап 3: Каждая модальность берет из контекста что ей нужно
        results = {}
        for name, modality in self.modalities.items():
            results[name] = modality.process(context)
            
        # Этап 4: Агрегируем скоры
        final_result = self.aggregator.aggregate(results)
        return final_result
