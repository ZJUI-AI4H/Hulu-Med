#    Copyright 2024 Alibaba DAMO Academy
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
import os
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from transformers import TRANSFORMERS_CACHE


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'mm_projector.bin')):
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class SimSpatialConv(nn.Module):

    def __init__(self, mm_hidden_size, hidden_size, downsample=(2, 2), padding=1, depth=1, mlp_depth=2):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = mm_hidden_size
        self.output_hidden_size = output_hidden_size = hidden_size
        self.downsample = downsample
        self.padding = padding
        self.sampler = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder_hidden_size,
                out_channels=4 * self.encoder_hidden_size,
                kernel_size=self.downsample,
                stride=self.downsample,
                padding=self.padding,
                bias=True
            ),
            nn.SiLU(),
        )
        self.readout = build_mlp(mlp_depth, 4 * self.encoder_hidden_size, self.output_hidden_size)

    def forward(self, x):
        hw = int(x.size(1) ** 0.5)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.sampler(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x

    def cal_proj_size(self, input_size):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        height = math.ceil((input_size[0] + self.padding) / self.downsample[0])
        width  = math.ceil((input_size[1] + self.padding) / self.downsample[1])
        return height * width


class MlpGeluProjector(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, projector_type):
        super().__init__()

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))

        self.readout = build_mlp(mlp_depth, mm_hidden_size, hidden_size)

    def forward(self, x):
        x = self.readout(x)
        return x

    def cal_proj_size(self, input_size):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        height = input_size[0]
        width  = input_size[1]
        return height * width


def build_vision_projector(config, mm_hidden_size, delay_load=False, **kwargs):
    # hulumed projector only support image-wise operation now, i.e., prohibit the temporal aggregation
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    hidden_size = config.hidden_size

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(mm_hidden_size, hidden_size)
    elif  projector_type == "simp_spatial_conv":
        return SimSpatialConv(mm_hidden_size, hidden_size)
    elif projector_type.startswith("mlp"):
        return MlpGeluProjector(mm_hidden_size, hidden_size, projector_type)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
