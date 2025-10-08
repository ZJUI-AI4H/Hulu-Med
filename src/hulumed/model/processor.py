# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import math
import warnings
from typing import List, Union, Dict, Optional

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from hulumed.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from hulumed.mm_utils import load_video, load_images


DEFAULT_CHAT_TEMPLATE = """
{%- set identifier = 'im' %}
{% for message in messages %}
    {% if message['role'] == 'stream' %}
        {% set identifier = 'stream' %}
    {% else %}
        {% set identifier = 'im' %}
    {% endif %}
    {{- '<|' + identifier + '_start|>' + message['role'] + '\n' -}}
    {% if message['content'] is string %}
        {{- message['content'] + '<|' + identifier + '_end|>\n' -}}
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if 'time' in content %}
                    {{- 'Time ' + content['time'] | round(1) | string + 's: ' -}}
                {% endif %}
"""
DEFAULT_CHAT_TEMPLATE += """
                {{- '%s\n' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
            {% elif content['type'] == 'video' or 'video' in content or 'video_url' in content %}
                {% for i in range(content['num_frames']) %}
                    {% if 'timestamps' in content %}
                        {{- 'Time ' + content['timestamps'][i] | round(1) | string + 's:' -}}
                    {% endif %}
                    {% if i < content['num_frames'] - 1 %}
"""
DEFAULT_CHAT_TEMPLATE += """
                        {{- '%s,' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
                    {% else %}
"""
DEFAULT_CHAT_TEMPLATE += """
                        {{- '%s\n' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
                    {% endif %}
                {% endfor %}
            {% elif content['type'] == 'text' or 'text' in content %}
                {{- content['text'] -}}
            {% endif %}
        {% endfor %}
        {{- '<|' + identifier + '_end|>\n' -}}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' -}}
{% endif %}
"""


class HulumedProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class HulumedProcessor(ProcessorMixin):
    r"""
    Modified from Qwen2VLProcessor
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if chat_template is None:
            chat_template = DEFAULT_CHAT_TEMPLATE
        tokenizer.chat_template = chat_template
        self.chat_template = chat_template
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.generation_prompt = self._infer_generation_prompt()
        self.generation_prompt_ids = self.tokenizer.encode(self.generation_prompt, return_tensors="pt")
        self.generation_prompt_length = len(self.generation_prompt_ids[0])
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_generation_prompt(self):
        return self.generation_prompt

    def get_generation_prompt_ids(self):
        return self.generation_prompt_ids

    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    def load_images(self, *args, **kwargs):
        return load_images(*args, **kwargs)

    def _infer_generation_prompt(self):
        pseudo_message = [{"role": "user", "content": ""}]
        instruction = self.tokenizer.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=True)
        conversation = self.tokenizer.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=False)
        return instruction.replace(conversation, "")

    def _process_text_with_label(
        self,
        text: List[Dict],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        assert kwargs.pop("return_tensors", "pt") == "pt", "Only PyTorch tensors are supported when return_labels=True."
        assert isinstance(text[0], dict), "When return_labels=True, text must be a list of messages."

        input_ids_list = []
        targets_list = []
        sample_types_list = []
        image_idx = 0

        for message_idx, message in enumerate(text):
            prompt = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
            prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
            prompt = []
            for chunk_idx in range(len(prompt_chunks) - 1):
                prompt.append(prompt_chunks[chunk_idx])
                thw = grid_sizes[image_idx]
                prompt.append(DEFAULT_IMAGE_TOKEN * thw.prod().long())
                image_idx += 1
            prompt.append(prompt_chunks[-1])
            prompt = "".join(prompt)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            input_ids_list.append(input_ids)

            targets = torch.full_like(input_ids, IGNORE_INDEX)
            sample_types = torch.full_like(input_ids, IGNORE_INDEX)
            if message["role"] == "assistant":
                targets[self.generation_prompt_length:-1] = input_ids[self.generation_prompt_length:-1].clone()
            elif message["role"] == "stream":
                diff = torch.diff((input_ids == self.image_token_id).float())
                image_end_indices = torch.nonzero(diff < 0)[:, 0]
                targets[image_end_indices + 1] = input_ids[image_end_indices + 1]
                sample_types = targets.clone()
                sample_types[torch.logical_and(sample_types > 0, sample_types != self.eos_token_id)] = 0
                targets[-2] = input_ids[-2]

            targets_list.append(targets)
            sample_types_list.append(sample_types)
        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        targets = torch.cat(targets_list)
        sample_types = torch.cat(sample_types_list)
        types, counts = torch.unique(sample_types[sample_types > -1], return_counts=True)

        if len(types) > 0:
            target_num_samples = counts.amin()

            for type_id, type_count in zip(types, counts):
                if type_count > target_num_samples:
                    indices = torch.nonzero(sample_types == type_id)[:, 0]
                    random_selector = torch.randperm(indices.size(0))[:-target_num_samples]
                    targets[indices[random_selector]] = IGNORE_INDEX
                    sample_types[indices[random_selector]] = -1

        text_inputs = {
            "input_ids": torch.cat(input_ids_list),
            "labels": targets,
        }

        return text_inputs

    def _process_text_without_label(
        self,
        text: Union[List[str], List[Dict]],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        if isinstance(text[0], dict):
            warnings.warn("Input text is a list of messages. Automatically convert it to a string with 'apply_chat_template' with generation prompt.")
            text = [self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)]

        image_idx = 0
        for i in range(len(text)):
            while DEFAULT_IMAGE_TOKEN in text[i]:
                thw = grid_sizes[image_idx]
                text[i] = text[i].replace(DEFAULT_IMAGE_TOKEN, "<placeholder>" * thw.prod().long(), 1)
                image_idx += 1
            text[i] = text[i].replace("<placeholder>", DEFAULT_IMAGE_TOKEN)
        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        text_inputs = self.tokenizer(text, **kwargs)
        return text_inputs

    def process_text(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]],
        image_inputs: Dict[str, torch.Tensor] = {},
        return_labels: bool = False,
        **kwargs,
    ):
        kwargs.pop("padding", None)
        kwargs.pop("padding_side", None)

        if not isinstance(text, (list, tuple)):
            text = [text]
        assert len(text), "At least one text must be provided."

        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                warnings.warn(f"Grid size {grid_size} is not divisible by merge size. Some undesired errors may occur.")
            if grid_size[0] == 1:
                grid_sizes.append(grid_size[1:] / merge_size)
            elif grid_size[0] > 1:
                grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])

        if return_labels:
            return self._process_text_with_label(text, grid_sizes, **kwargs)
        return self._process_text_without_label(text, grid_sizes, **kwargs)

    def process_images(
        self,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        **kwargs,
    ):
        if images is None:
            return {}
        image_inputs = self.image_processor(images=images, merge_size=merge_size, **kwargs)
        return image_inputs

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]] = None,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        return_labels: bool = False,
        **kwargs: Unpack[HulumedProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            HulumedProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        output_kwargs["text_kwargs"].pop("padding")
        output_kwargs["text_kwargs"].pop("padding_side")

        image_inputs = self.process_images(images, merge_size, **output_kwargs["images_kwargs"])
        text_inputs = self.process_text(text, image_inputs, return_labels, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
