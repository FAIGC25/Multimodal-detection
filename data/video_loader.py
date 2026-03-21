import torch
import cv2
import numpy as np
from core.interfaces import BaseVideoLoader
from core.entities import VideoContext

class SimpleVideoLoader(BaseVideoLoader):
    def __init__(self, max_frames: int = 16):
        """
        :param max_frames: Максимальное количество кадров, которое будет извлечено из видео
                           (для предотвращения нехватки памяти). Кадры выбираются равномерно.
        """
        self.max_frames = max_frames

    def load(self, video_path: str) -> VideoContext:
        print(f"[VideoLoader] Чтение видео {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Индексы кадров, которые мы хотим извлечь
        if total_frames > self.max_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        else:
            frame_indices = np.arange(total_frames)
            
        frames_list = []
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if current_frame in frame_indices:
                # Конвертируем из BGR (OpenCV) в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
                
            current_frame += 1
            
        cap.release()
        
        if not frames_list:
            raise ValueError(f"Видео {video_path} не содержит кадров или повреждено.")
            
        # Преобразуем список кадров (H, W, C) numpy массива в PyTorch Tensor
        # И меняем размерность на (T, C, H, W)
        frames_tensor = torch.from_numpy(np.array(frames_list))  # (T, H, W, C)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)        # (T, C, H, W)
        
        # Нормализуем к [0, 1] для удобства
        frames_tensor = frames_tensor.float() / 255.0
        
        print(f"[VideoLoader] Извлечено {len(frames_tensor)} кадров размерности {frames_tensor.shape[2]}x{frames_tensor.shape[3]}")

        return VideoContext(
            video_path=video_path,
            raw_frames=frames_tensor,
            fps=fps
        )

