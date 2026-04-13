import os
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import transformers
from transformers import AutoConfig, AutoModelForCausalLM

orig_register = AutoConfig.register
orig_model_register = AutoModelForCausalLM.register

def patched_register(model_type, config_class, exist_ok=False, *args, **kwargs):
    try:
        orig_register(model_type, config_class, exist_ok=True, *args, **kwargs)
    except TypeError:
        pass
    except ValueError:
        pass

def patched_model_register(config_class, model_class, exist_ok=False, *args, **kwargs):
    try:
        orig_model_register(config_class, model_class, exist_ok=True, *args, **kwargs)
    except (TypeError, ValueError):
        pass

AutoConfig.register = patched_register
AutoModelForCausalLM.register = patched_model_register

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.video_loader import SimpleVideoLoader
from data.face_tracker import DummyFaceTracker
from modalities.spatial import SpatialSIDADetector
from modalities.depth import SIDADepthDetector
# from modalities.temporal import TemporalAnalyzer
from pipeline.aggregator import WeightedAggregator
from pipeline.system import MultimodalDeepfakeSystem

def main():
    # 1. Инициализация компонентов инфраструктуры
    loader = SimpleVideoLoader()
    tracker = DummyFaceTracker()
    
    # 2. Регистрируем модальности. Spatial + Depth
    modalities = [
        SpatialSIDADetector(),
        SIDADepthDetector(),
    ]
    
    # 3. Инициализируем агрегатор.
    aggregator = WeightedAggregator(weights={"spatial_sida": 0.5, "sida_depth": 0.5})
    
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
