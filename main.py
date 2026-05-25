import os
import sys
import ssl
import time
import argparse

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
from modalities.depth_student import DepthStudentDetector
from pipeline.aggregator import WeightedAggregator
from pipeline.system import MultimodalDeepfakeSystem

def main():
    parser = argparse.ArgumentParser(description="Multimodal Deepfake Detection")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--depth", default=None, help="Path to depth video")
    args = parser.parse_args()

    video_path = args.video
    depth_video_path = args.depth

    if not os.path.exists(video_path):
        print(f"Ошибка: Видео файл не найден по пути: {video_path}")
        return

    # 1. Инициализация компонентов инфраструктуры
    loader = SimpleVideoLoader()
    tracker = DummyFaceTracker()

    # 2. Регистрируем модальности. Spatial + Depth (student)
    modalities = [
        SpatialSIDADetector(),
        DepthStudentDetector(),
    ]

    # 3. Инициализируем агрегатор.
    aggregator = WeightedAggregator(weights={"spatial_sida": 0.5, "depth_student": 0.5})

    # 4. Сборка системы (Оркестратора)
    system = MultimodalDeepfakeSystem(
        video_loader=loader,
        face_tracker=tracker,
        modalities=modalities,
        aggregator=aggregator
    )

    print(f"\nЗапуск пайплайна Multimodal Deepfake Detection для видео: {video_path}\n")

    print("\n--- Начало инференса ---")
    start_time = time.perf_counter()

    result = system.predict(video_path, depth_video_path=depth_video_path)

    end_time = time.perf_counter()
    inference_time = end_time - start_time
    print(f"--- Конец инференса. Время выполнения: {inference_time:.2f} секунд ---")

    print("\n=== Итоговый результат ===")
    print(f"Вероятность фейка (Score): {result.score:.4f}")
    print(f"Метаданные (Metadata):     {result.metadata}")
    print("==========================")

if __name__ == "__main__":
    main()
