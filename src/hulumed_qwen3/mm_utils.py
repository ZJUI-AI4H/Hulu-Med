import ast
import os
import re
import math
import base64
import traceback
from io import BytesIO
from typing import Optional

import torch
import torchvision.transforms.functional as VF
import numpy as np
from transformers import StoppingCriteria

import cv2
import imageio
import ffmpeg
from PIL import Image
from decord import VideoReader, cpu

from .constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MODAL_INDEX_MAP, DEFAULT_IMAGE_TOKEN

import nibabel as nib

def slice_nifti_to_images(nifti_path, num_slices=180, axis='axial'):
    try:
        
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        if axis not in axis_map:
            raise ValueError("Axis arugment must be one of 'axial', 'coronal', 'sagittal'")
        
        slice_axis = axis_map[axis]

        nifti_file = nib.load(nifti_path)
        image_data = nifti_file.get_fdata()

        num_total_slices = image_data.shape[slice_axis]
        
        sampled_indices = np.linspace(0, num_total_slices - 1, num_slices, dtype=int)
        
        images = []
        for slice_index in sampled_indices:
            if slice_axis == 0:
                slice_2d = image_data[slice_index, :, :]
            elif slice_axis == 1:
                slice_2d = image_data[:, slice_index, :]
            else:
                slice_2d = image_data[:, :, slice_index]
            if slice_2d.max() > slice_2d.min():
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255.0
            slice_2d = slice_2d.astype(np.uint8)
            pil_image = Image.fromarray(slice_2d).convert('RGB')
            images.append(pil_image)
            
        return images

    except ImportError:
        raise ImportError("please run 'pip install nibabel'")
    except Exception as e:
        print(f" NIfTI  {nifti_path} error: {e}")
        return [] 
def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def grid_divide(image, cell_size):
    grid = []
    width, height = image.size
    for i in range(0, height, cell_size):
        row = []
        for j in range(0, width, cell_size):
            box = (j, i, j + cell_size, i + cell_size)
            row.append(image.crop(box))
        grid.append(row)

    return grid
def load_images(image_path, nii_slice_axis='axial', nii_num_slices=180):
    images = []

    def safe_open(f):
        try:
            with Image.open(f).convert('RGB') as img:
                return img
        except Exception:
            pass

    if isinstance(image_path, str) and (image_path.endswith('.nii') or image_path.endswith('.nii.gz')):
        print(f"Detected NIfTI file, slicing into {nii_num_slices} slices along '{nii_slice_axis}' axis...")
        images = slice_nifti_to_images(image_path, num_slices=nii_num_slices, axis=nii_slice_axis)

    elif isinstance(image_path, str) and os.path.isfile(image_path):
        img = safe_open(image_path)
        if img is not None:
            images.append(img)

    elif isinstance(image_path, str) and os.path.isdir(image_path):
        for f in sorted(os.listdir(image_path)):
            full_path = os.path.join(image_path, f)
            if os.path.isfile(full_path):
                img = safe_open(full_path)
                if img is not None:
                    images.append(img)

    elif isinstance(image_path, list) and isinstance(image_path[0], str):
        for f in image_path:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                print(f"Detected NIfTI file, slicing into {nii_num_slices} slices along '{nii_slice_axis}' axis...")
                images.extend(slice_nifti_to_images(f, num_slices=nii_num_slices, axis=nii_slice_axis))
            else:
                img = safe_open(f)
                if img is not None:
                    images.append(img)

    elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
        images = [img.convert('RGB') for img in image_path]

    elif isinstance(image_path, Image.Image):
        images = [image_path.convert('RGB')]

    else:
        if isinstance(image_path, str):
             raise ValueError(f"Unsupported image path or file type: {image_path}")
        else:
             raise ValueError(f"Unsupported image path type: {type(image_path)}")

    return images


def process_pad_image(image, padding_value=(0, 0, 0)):
    image = expand2square(image, padding_value)

    return [image]


def find_closest_aspect_ratio(src_ratio, tgt_ratios, ori_size, tgt_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = ori_size[0] * ori_size[1]
    for ratio in tgt_ratios:
        tgt_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(src_ratio - tgt_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * tgt_size[0] * tgt_size[1] * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def process_dynamic_image(image, image_size=384, use_thumbnail=True):
    min_num = 1
    max_num = 12

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    ori_size = image.size
    aspect_ratio = ori_size[0] / ori_size[1]

    tgt_ratios = []
    for n in range(min_num, max_num + 1):
        tgt_ratios.extend([(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num])
    tgt_ratios = set(tgt_ratios)
    tgt_ratios = sorted(tgt_ratios, key=lambda x: x[0] * x[1])

    tgt_ratio = find_closest_aspect_ratio(aspect_ratio, tgt_ratios, ori_size, image_size)

    tgt_width = image_size[0] * tgt_ratio[0]
    tgt_height = image_size[1] * tgt_ratio[1]
    resized_img = image.resize((tgt_width, tgt_height))

    image_grid = grid_divide(resized_img, image_size[0])

    if use_thumbnail:
        thumbnail_img = image.resize((image_size[0], image_size[1]))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_highres_image(image_path, image_size=384, use_thumbnail=True, padding_value=(0, 0, 0)):
    grid_width = [1, 2, 3]
    grid_width_real = [x * image_size for x in grid_width]

    longest_side = max(image.size)
    fit_grid_width_real = [x for x in grid_width_real if x >= longest_side]
    if len(fit_grid_width_real) == 0:
        select_size = max(grid_width_real)
    else:
        select_size = min(fit_grid_width_real)

    image_padded = expand2square(image, padding_value)
    image_padded = image_padded.resize((select_size, select_size))
    image_grid = grid_divide(image_padded, image_size)

    if use_thumbnail:
        thumbnail_img = image.resize((image_size, image_size))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def select_best_resolution(original_size, possible_resolutions):
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def process_anyres_image(image, image_size=384, use_thumbnail=True, padding_value=(0, 0, 0)):
    possible_grids = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    possible_resolutions = [(x * image_size, y * image_size) for x, y in possible_grids]

    best_resolution = select_best_resolution(image.size, possible_resolutions)

    nw, nh = best_resolution
    ow, oh = image.size

    scale_factor = min(nw / ow, nh / oh)
    new_size = (int(ow * scale_factor), int(oh * scale_factor))

    image_padded = Image.new("RGB", (nw, nh), padding_value)
    image_padded.paste(image.resize(new_size), ((nw - new_size[0]) // 2, (nh - new_size[1]) // 2))

    image_grid = grid_divide(image_padded, image_size)

    if use_thumbnail:
        thumbnail_img = image.resize((image_size, image_size))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_adares_image(image_path, image_size=384, use_thumbnail=True):
    min_num = 1
    max_num = 12

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    ori_size = image.size
    aspect_ratio = ori_size[0] / ori_size[1]

    tgt_ratios = []
    for n in range(min_num, max_num + 1):
        tgt_ratios.extend([(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num])
    tgt_ratios = set(tgt_ratios)
    possible_resolutions = [(x * image_size[0], y * image_size[1]) for x, y in tgt_ratios]

    best_resolution = select_best_resolution(ori_size, possible_resolutions)

    resized_img = image.resize((best_resolution[0], best_resolution[1]))

    image_grid = grid_divide(resized_img, image_size[0])

    if use_thumbnail:
        thumbnail_img = image.resize((image_size[0], image_size[1]))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_images(image_path, processor, aspect_ratio='anyres', image_size=384, use_thumbnail=True):
    images = load_images(image_path)

    padding_value = tuple(int(x*255) for x in processor.image_mean)

    image_grids = []
    for image in images:
        if aspect_ratio == 'pad':
            image_grid = process_pad_image(image, padding_value=padding_value)
        elif aspect_ratio == 'dynamic':
            image_grid = process_dynamic_image(image, image_size=image_size, use_thumbnail=use_thumbnail)
        elif aspect_ratio == 'highres':
            image_grid = process_highres_image(image, image_size=image_size, use_thumbnail=use_thumbnail, padding_value=padding_value)
        elif aspect_ratio == 'anyres':
            image_grid = process_anyres_image(image, image_size=image_size, use_thumbnail=use_thumbnail, padding_value=padding_value)
        elif aspect_ratio == 'adares':
            image_grid = process_adares_image(image, image_size=image_size, use_thumbnail=use_thumbnail)
        else:
            image_grid = [image]

        image_grid = [processor.preprocess(image_row, return_tensors='pt', num_images=len(images)) for image_row in image_grid]
        image_grids.append(image_grid)

    return image_grids


def frame_sample(duration, mode='uniform', num_frames=None, vid_fps=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        if duration <= num_frames:
            return np.arange(duration).astype(int)
        return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert vid_fps is not None, "FPS must be provided for FPS sampling."
        fps = fps if fps is not None else NUM_FRAMES_PER_SECOND
        segment_len = min(vid_fps // fps, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def load_video_from_ids(video_path, s=None, e=None, fps=None, max_frames=None, temporal_factor=1):
    if s is not None and e is not None:
        s = s if s >= 0. else 0.
        e = e if e >= 0. else 0.
        if s > e:
            s, e = e, s
        elif s == e:
            e = s + 1

    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))

        vid_fps = 1
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)

        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=64)
        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    f_start = 0                       if s is None else max(int(s * vid_fps) - 1, 0)
    f_end   = num_frames_of_video - 1 if e is None else min(int(e * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    duration = len(frame_indices)
    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    if fps is not None and duration / vid_fps < max_frames:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', vid_fps=vid_fps, fps=fps)]
    else:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=max_frames)]

    if os.path.isdir(video_path):
        frames = []
        for frame_idx in sampled_frame_indices:
            filepath = os.path.join(video_path, frame_files[frame_idx])
            try:
                with Image.open(filepath).convert('RGB') as img:
                    frames.append(img)
            except Exception as e:
                print(f"Error with {filepath} with {e}")
                pass
    elif video_path.endswith('.gif'):
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
    else:
        frames = vreader.get_batch(sampled_frame_indices).asnumpy()

    timestamps = [x / vid_fps for x in sampled_frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    return frames, timestamps


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    temporal_factor: int = 1
):
    if isinstance(video_path, list):
        video_path = video_path[0]
    if start_time is not None and end_time is not None and end_time - start_time < 1:
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if os.path.isdir(video_path):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if video_path.endswith('.gif'):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not exist: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps

    start_frame = 0
    end_frame = total_frames
    if start_time is not None:
        start_frame = int(start_time * video_fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
    if end_time is not None:
        end_frame = int(end_time * video_fps)
        end_frame = max(start_frame, min(end_frame, total_frames))

    frame_indices = list(range(start_frame, end_frame))
    if fps is not None:
        target_frame_rate = fps
        frame_step = max(1, int(video_fps / target_frame_rate))
        frame_indices = frame_indices[::frame_step]

    if max_frames is not None and len(frame_indices) > max_frames:
        frame_indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)

    frames = vr.get_batch(frame_indices).asnumpy()

    if size is not None:
        h, w = frames.shape[1], frames.shape[2]
        scale_factor = size / min(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        new_h = new_h // size_divisible * size_divisible
        new_w = new_w // size_divisible * size_divisible

        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (new_w, new_h))
            resized_frames.append(resized_frame)
        frames = np.array(resized_frames)

    timestamps = [i / video_fps for i in frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad_length, axis=0)])
        timestamps.extend([timestamps[-1] + 1 / video_fps] * pad_length)

    return frames, timestamps


def process_video(video_path, processor, s=None, e=None, aspect_ratio='avt', num_frames=None):
    fps = 1 if num_frames is None else None
    frames, timestamps = load_video(video_path, s, e, fps=fps, max_frames=num_frames)

    assert len(frames) == len(timestamps), "Number of frames and timestamps must match."

    if aspect_ratio == 'pad':
        frames = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in frames]

    if aspect_ratio == 'avt':
        frames = [processor.preprocess(frame, return_tensors='pt', image_num=len(frames)) for frame in frames]
        grid_frames = [frames]
    else:
        frames = processor.preprocess(frames, return_tensors='pt', image_num=len(frames))
        grid_frames = [[frames]]

    return grid_frames, timestamps


def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token_index is None:
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for idx, chunk in enumerate(prompt.split(multimodal_token))]

        input_ids = []
        for i in range(1, 2 * len(prompt_chunks)):
            if i % 2 == 1:
                input_ids.extend(prompt_chunks[i // 2])
            else:
                input_ids.append(multimodal_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
