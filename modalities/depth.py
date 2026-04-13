import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Добавляем путь к репозиторию SIDA
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SIDA')))
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from core.interfaces import BaseModality
from core.entities import VideoContext, DetectionResult

class SIDADepthDetector(BaseModality):
    def __init__(self, model_path: str = "../../Clean_SIDA_Depth"):
        super().__init__()
        print("[depth] Инициализация SIDA (Depth) без SAM в 8-bit...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=512, padding_side="right", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        
        kwargs = {
            "torch_dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
        }
        
        self.model = SIDAForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        self.model.eval()
        self.clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    @property
    def name(self) -> str:
        return "sida_depth"

    def preprocess(self, context: VideoContext) -> dict:
        if not context.depth_maps or len(context.depth_maps) == 0:
            raise ValueError(f"Карты глубины не найдены! {self.name} не может обработать видео.")
            
        depth_map = context.depth_maps[0] 
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()
        
        if len(depth_map.shape) == 3 and depth_map.shape[0] == 1:
            depth_map = np.squeeze(depth_map, axis=0) # (H, W)
        elif len(depth_map.shape) == 3 and depth_map.shape[-1] == 1:
            depth_map = np.squeeze(depth_map, axis=-1) # (H, W)

        depth_map = (depth_map * 255).astype(np.uint8) if depth_map.max() <= 1.0 else depth_map.astype(np.uint8)
        image_rgb = Image.fromarray(depth_map).convert('RGB')

        image_clip = self.clip_image_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0].unsqueeze(0).half().to(self.device)

        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
        prompt += "Analyze this depth map. Is it a deepfake?"
        
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        
        input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)

        return {"image_clip": image_clip, "input_ids": input_ids, "conv": conv}

    def forward(self, preprocessed_data: dict) -> dict:
        stop_str = preprocessed_data["conv"].sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, preprocessed_data["input_ids"])

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs=preprocessed_data["input_ids"],
                images=preprocessed_data["image_clip"],
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        return {"output_ids": output_ids}

    def postprocess(self, raw_output: dict) -> DetectionResult:
        output_ids = raw_output["output_ids"][0]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        is_fake = "tampered" in text_output.lower() or "fake" in text_output.lower()
        score = 0.95 if is_fake else 0.05
        
        return DetectionResult(
            score=score, 
            confidence=abs(score - 0.5) * 2,
            metadata={"reasoning": text_output, "type": "sida_depth"}
        )