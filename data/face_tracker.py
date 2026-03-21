import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from core.interfaces import BaseFaceTracker
from core.entities import VideoContext, FaceTrack
import torchvision.transforms.functional as F

class DummyFaceTracker(BaseFaceTracker):
    """Этап детекции и трекинга лиц. Добавляет FaceTrack в видео контекст"""
    def __init__(self):
        # Инициализируем детектор MTCNN.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
        self.crop_size = 224

    def process(self, context: VideoContext) -> VideoContext:
        print("[FaceTracker] Поиск и кропинг лиц на кадрах (MTCNN)...")
        
        frames_tensor = context.raw_frames # (T, C, H, W) в диапазоне [0, 1]
        T = frames_tensor.shape[0]
        
        face_crops = []
        valid_frames = [] # Индексы кадров, на которых нашли лицо

        for i in range(T):
            # Конвертируем тензор обратно в PIL Image
            # frames_tensor[i] имеет размерность (C, H, W) и значения [0, 1]
            frame_uint8 = (frames_tensor[i] * 255).byte()
            frame_np = frame_uint8.permute(1, 2, 0).numpy() # (H, W, C)
            img_pil = Image.fromarray(frame_np)

            # Детектируем лица и сразу кропаем (возвращает тензор [-1, 1] размера (3, 160, 160) по умолчанию)
            # Но мы хотим достать оригинальный кроп, поэтому можно использовать возвращаемые bbox
            boxes, probs = self.mtcnn.detect(img_pil)
            
            if boxes is not None and len(boxes) > 0:
                # Берем самую первую (уже отсортировано по размеру или вероятности благодаря select_largest)
                box = boxes[0]
                # Добавляем небольшой отступ (margin) к лицу
                x1, y1, x2, y2 = [int(v) for v in box]
                
                # Защита от выхода за границы кадра
                h_img, w_img = img_pil.size[1], img_pil.size[0]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                
                # Кропаем оригинальный кадр как тензор
                crop = frames_tensor[i, :, y1:y2, x1:x2]
                
                # Приводим к единому размеру (например, 224x224)
                # Используем билинейную интерполяцию для сглаживания
                crop_resized = F.resize(crop, [self.crop_size, self.crop_size], antialias=True)
                
                face_crops.append(crop_resized)
                valid_frames.append(i)
        
        if not face_crops:
            print("[FaceTracker] ВНИМАНИЕ: На видео не найдено ни одного лица!")
            return context
            
        print(f"[FaceTracker] Лицо найдено на {len(face_crops)}/{T} кадров.")
        
        # Собираем кропы в единый тензор (T_new, C, H, W)
        faces_tensor = torch.stack(face_crops)
        
        # Эмуляция трекинга (пока просто считаем, что это всегда один и тот же человек - track_id=1)
        track = FaceTrack(track_id=1, frames=faces_tensor)
        context.face_tracks.append(track)
        
        return context

