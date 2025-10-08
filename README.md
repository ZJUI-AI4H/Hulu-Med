# Hulu-Med: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/xxxx.xxxxx)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/ZJU-AI4H/Hulu-Med)
[![ModelScope](https://img.shields.io/badge/ModelScope-Models-blue)](https://modelscope.cn/models/Med-Team/Hulu-Med)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[📄 Paper](https://arxiv.org/abs/xxxx.xxxxx) | [🤗 HuggingFace Models](https://huggingface.co/ZJU-AI4H/Hulu-Med) | [🔮 ModelScope Models](https://modelscope.cn/models/Med-Team/Hulu-Med) | [📊 Demo](#demo)

</div>

## 🔥 News

- **[2025-10-08]** Hulu-Med models and inference code released!

## 📖 Overview

**Hulu-Med** is a transparent medical vision-language model that unifies understanding across diverse modalities including **medical text, 2D/3D images, and videos**. Built with a focus on transparency and accessibility, Hulu-Med achieves state-of-the-art performance on 30 medical benchmarks while being trained entirely on public data.

### Key Features

- 🌟 **Holistic Multimodal Understanding**: Seamlessly processes medical text, 2D images, 3D volumes, and surgical videos
- 🔓 **Fully Transparent**: Complete open-source pipeline including data curation, training code, and model weights
- 📊 **State-of-the-Art Performance**: Outperforms leading open-source models and competes with proprietary systems
- ⚡ **Efficient Training**: Only 4,000-40,000 GPU hours required for 7B-32B variants
- 🗂️ **Comprehensive Coverage**: Trained on 16.7M samples spanning 12 anatomical systems and 14 imaging modalities

## 🏆 Performance Highlights

Hulu-Med achieves **state-of-the-art** results across diverse medical tasks:

| Task Category | Representative Benchmark | Hulu-Med-7B | Previous SOTA |
|--------------|-------------------------|-------------|---------------|
| 2D Medical VQA | OmniMedVQA | **84.2** | 82.9 |
| Medical Report Generation | MIMIC-CXR (RaTEScore) | **57.0** | 51.3 |
| 3D Medical Understanding | M3D (Open-ended Recall) | **98.5** | 98.4 |
| Surgical Video Analysis | Cholec80-VQA | **96.9** | 95.7 |
| Medical Text Reasoning | MedXpertQA | **19.6** | 16.5 |
| Multilingual Medical QA | MMedBench (Avg) | **71.38** | 67.75 |

## 🚀 Model Zoo

We provide three model variants with different parameter scales:

| Model | Parameters | LLM Base | Training Cost | HuggingFace | ModelScope |
|-------|-----------|----------|---------------|-------------|------------|
| **Hulu-Med-7B** | 7B | Qwen2.5-7B | ~4,000 GPU hours | [🤗 Link](https://huggingface.co/ZJU-AI4H/Hulu-Med-7B) | [🔮 Link](https://modelscope.cn/models/Med-Team/Hulu-Med-7B) |
| **Hulu-Med-14B** | 14B | Qwen3-14B | ~8,000 GPU hours | [🤗 Link](https://huggingface.co/ZJU-AI4H/Hulu-Med-14B) | [🔮 Link](https://modelscope.cn/models/Med-Team/Hulu-Med-14B) |
| **Hulu-Med-32B** | 32B | Qwen2.5-32B | ~40,000 GPU hours | [🤗 Link](https://huggingface.co/ZJU-AI4H/Hulu-Med-32B) | [🔮 Link](https://modelscope.cn/models/Med-Team/Hulu-Med-32B) |

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/your-org/Hulu-Med.git
cd Hulu-Med

# Create conda environment
conda create -n hulumed python=3.10
conda activate hulumed

# Install dependencies
pip install -r requirements.txt
