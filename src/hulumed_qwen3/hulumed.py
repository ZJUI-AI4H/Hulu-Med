import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from tqdm import tqdm
import os

from models.hulumed import load_pretrained_model
from models.hulumed.mm_utils import load_images, get_model_name_from_path
from models.hulumed.model.processor import HulumedProcessor

class HuluMedModel:
    """
    一个用于加载和运行 HuluMed / OmniV-Med 模型的类，
    其接口仿照 MedGemma 模板。
    """
    def __init__(self, model_path, args):
        """
        初始化模型、分词器和处理器。

        :param model_path: 模型的路径。
        :param args: 包含生成参数的对象 (例如 temperature, top_p, max_new_tokens)。
        """
        super().__init__()
        
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, image_processor, _ = load_pretrained_model(
            model_path, 
            None, 
            model_name,
            device_map="cuda:0"
        )
        self.processor = HulumedProcessor(image_processor, self.tokenizer)
        self.model.config.use_token_compression = False
        
        self.model.eval()

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.repetition_penalty = args.repetition_penalty

    def process_messages(self, messages):
        """
        处理输入的 messages 字典，将其转换为模型所需的格式。
        
        :param messages: 包含 prompt 和 image/images 的字典。
        :return: 准备好输入给 model.generate 的字典。
        """
        prompt = messages.get("prompt", "")
        
        conversation = [{"role": "user", "content": []}]
        
        loaded_images = None
        modal_type = "text"

        image_paths_or_pil = messages.get("images") or ([messages["image"]] if "image" in messages else [])
        merge_size = 1
        if image_paths_or_pil:
            modal_type = "image"
            loaded_images = load_images(image_paths_or_pil)
            if len(loaded_images) > 5:
                merge_size = 2
                modal_type = 'video'
            else:
                merge_size = 1
            for _ in loaded_images:
                conversation[0]["content"].append({"type": "image"})
        
        conversation[0]["content"].append({"type": "text", "text": prompt})
        
        inputs = self.processor(
            images=[loaded_images] if loaded_images is not None else None,
            text=conversation,
            merge_size= merge_size,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            
        return inputs, modal_type


    def generate_output(self, messages):
        """
        为单条消息生成回复。

        :param messages: 包含 prompt 和 image/images 的字典。
        :return: 模型生成的文本字符串。
        """
        llm_inputs, modal_type = self.process_messages(messages)
        
        do_sample = True if self.temperature > 0 else False
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **llm_inputs,
                do_sample=False,
                temperature=self.temperature if do_sample else 0, 
                modals=[modal_type],
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_outputs(self, messages_list):
        """
        为多条消息批量生成回复。
        
        :param messages_list: 包含多个 messages 字典的列表。
        :return: 一个包含所有生成结果的列表。
        """
        res = []
        for messages in tqdm(messages_list, desc="Generating Outputs"):
            result = self.generate_output(messages)
            print(result)
            res.append(result)
        return res
