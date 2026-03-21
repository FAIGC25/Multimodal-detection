import os
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.video_loader import SimpleVideoLoader
from data.face_tracker import DummyFaceTracker
from modalities.spatial import SpatialHiddenCLIP
from modalities.temporal import TemporalAnalyzer
from pipeline.aggregator import WeightedAggregator
from pipeline.system import MultimodalDeepfakeSystem

def main():
    # 1. Инициализация компонентов инфраструктуры
    loader = SimpleVideoLoader()
    tracker = DummyFaceTracker()
    
    # 2. Регистрируем модальности. Spatial (HiddenCLIP) + Temporal
    modalities = [
        SpatialHiddenCLIP(),
        TemporalAnalyzer(),
    ]
    
    # 3. Инициализируем агрегатор.
    # Веса по умолчанию (0.6 Spatial, 0.4 Temporal)
    aggregator = WeightedAggregator(weights={"spatial": 0.6, "temporal": 0.4})
    
    # 4. Сборка системы (Оркестратора)
    system = MultimodalDeepfakeSystem(
        video_loader=loader,
        face_tracker=tracker,
        modalities=modalities,
        aggregator=aggregator
    )
    
    # --- Запуск ---
    video_path = "path/to/your/video.mp4"
    print(f"\nЗапуск пайплайна Multimodal Deepfake Detection для видео: {video_path}\n")
    
    if not os.path.exists(video_path):
        print(f"Ошибка: Видео файл не найден по пути: {video_path}")
        return

    result = system.predict(video_path)
    
    print("\n=== Итоговый результат ===")
    print(f"Вероятность фейка (Score): {result.score:.4f}")
    print(f"Метаданные (Metadata):     {result.metadata}")
    print("==========================")

if __name__ == "__main__":
    main()
