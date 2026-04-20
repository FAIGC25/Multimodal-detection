import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SIDA')))
from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from core.interfaces import BaseModality
from core.entities import VideoContext, DetectionResult

def preprocess_sam(x, img_size=1024):
    """Normalize pixel values and pad to a square input."""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(x.device, dtype=x.dtype)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(x.device, dtype=x.dtype)
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

class SpatialSIDADetector(BaseModality):
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../SIDA/ck/SIDA-7B-description'))
            
        super().__init__()
        print(f"[spatial] Инициализация SIDA (RGB) в 8-bit... Путь к весам: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vis_save_path = "./sida_masks_output"
        os.makedirs(self.vis_save_path, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=512, padding_side="right", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        cls_token_idx = self.tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

        kwargs = {
            "torch_dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model"]
            )
        }
        
        self.model = SIDAForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, vision_tower="openai/clip-vit-large-patch14",
            seg_token_idx=seg_token_idx, cls_token_idx=cls_token_idx, **kwargs
        )
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if hasattr(self.model, "get_model") and hasattr(self.model.get_model(), "initialize_vision_modules"):
            try:
                self.model.get_model().initialize_vision_modules(self.model.get_model().config)
                vision_tower = self.model.get_model().get_vision_tower()
                vision_tower.to(dtype=torch.float16, device=self.device)
            except AttributeError:
                pass
                
        self.model.eval()

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(1024)

    @property
    def name(self) -> str:
        return "spatial_sida"

    def preprocess(self, context: VideoContext) -> dict:
        if not context.face_tracks or len(context.face_tracks[0].frames) == 0:
            raise ValueError(f"Лица не найдены! {self.name} модальность не может обработать видео.")
            
        frame_tensor = context.face_tracks[0].frames[0]
        if isinstance(frame_tensor, torch.Tensor):
            image_np = frame_tensor.cpu().numpy()
            if image_np.shape[0] == 3: # C, H, W
                image_np = np.transpose(image_np, (1, 2, 0))
            image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)
        else:
            image_np = frame_tensor

        image_clip = self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).half().to(self.device)
        
        image_sam = self.transform.apply_image(image_np)
        resize_list = [image_sam.shape[:2]]
        original_size_list = [image_np.shape[:2]]
        image_sam = torch.from_numpy(image_sam).permute(2, 0, 1).contiguous().unsqueeze(0).half().to(self.device)
        image_sam = preprocess_sam(image_sam)

        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
        prompt += "Please answer begin with [CLS] for classification, if the image is tampered, output mask the tampered region and explain why."
        
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        
        input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)

        return {
            "image_clip": image_clip, "image_sam": image_sam, "input_ids": input_ids,
            "resize_list": resize_list, "original_size_list": original_size_list, "raw_image": image_np
        }

    def forward(self, preprocessed_data: dict) -> dict:
        with torch.no_grad():
            output_ids, pred_masks = self.model.evaluate(
                preprocessed_data["image_clip"], preprocessed_data["image_sam"],
                preprocessed_data["input_ids"], preprocessed_data["resize_list"],
                preprocessed_data["original_size_list"], max_new_tokens=512, tokenizer=self.tokenizer
            )
        return {"output_ids": output_ids, "pred_masks": pred_masks, "raw_image": preprocessed_data["raw_image"]}

    def postprocess(self, raw_output: dict) -> DetectionResult:
        output_ids = raw_output["output_ids"][0][raw_output["output_ids"][0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False).replace("\n", "").strip()
        
        is_fake = "tampered" in text_output.lower() or "fake" in text_output.lower()
        score = 0.95 if is_fake else 0.05
        
        mask_paths = []
        for i, pred_mask in enumerate(raw_output["pred_masks"]):
            if pred_mask.shape[0] == 0: continue
            pred_mask = (pred_mask.detach().cpu().numpy()[0] > 0)
            
            save_path = os.path.join(self.vis_save_path, f"mask_{np.random.randint(10000)}.jpg")
            save_img = raw_output["raw_image"].copy()
            save_img[pred_mask] = save_img[pred_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            cv2.imwrite(save_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
            mask_paths.append(save_path)

        return DetectionResult(
            score=score, 
            metadata={"reasoning": text_output, "mask_paths": mask_paths, "type": "spatial_sida", "confidence": abs(score - 0.5) * 2}
        )
