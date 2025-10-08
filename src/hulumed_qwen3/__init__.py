import os
import copy
import math
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .model.processor import HulumedProcessor
from .mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, STREAM_START_TOKEN, STREAM_END_TOKEN


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def model_init(model_path=None, max_visual_tokens=None, **kwargs):
    model_path = "HuluMed-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    if max_visual_tokens is not None:
        image_processor.max_tokens = max_visual_tokens

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    processor = HulumedProcessor(image_processor, tokenizer)
    return model, processor


def mm_infer(data_dict, model, tokenizer, modal='video', **kwargs):
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, data_dict["input_ids"])

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 1.0)
    top_p = kwargs.get('top_p', 0.9 if do_sample else 1.0)
    top_k = kwargs.get('top_k', 20 if do_sample else 50)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    torch_dtype = model.config.torch_dtype if hasattr(model.config, "torch_dtype") else torch.float16

    data_dict["modals"] = [modal]
    data_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
    if "pixel_values" in data_dict:
        data_dict["pixel_values"] = data_dict["pixel_values"].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(
            **data_dict,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs
