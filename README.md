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

<div align="center">
<img src="https://github.com/user-attachments/files/22758795/Github_fig.pdf" width="100%">
</div>


### Key Features

- 🌟 **Holistic Multimodal Understanding**: Seamlessly processes medical text, 2D images, 3D volumes, and surgical videos
- 🔓 **Fully Transparent**: Complete open-source pipeline including data curation, training code, and model weights
- 📊 **State-of-the-Art Performance**: Outperforms leading open-source models and competes with proprietary systems
- ⚡ **Efficient Training**: Only 4,000-40,000 GPU hours required for 7B-32B variants
- 🗂️ **Comprehensive Coverage**: Trained on 16.7M samples spanning 12 anatomical systems and 14 imaging modalities

### Comprehensive Data Coverage

Our training corpus encompasses:

- **12 Major Anatomical Systems**: Multi-System, Skin/Integumentary, Respiratory, Cellular/Tissue Level, Digestive, Nervous, Cardiovascular, Musculoskeletal, Reproductive, Urinary, Whole Body, Endocrine, Immune/Lymphatic, and Hematologic systems
- **14 Medical Imaging Modalities**: CT, MRI, X-Ray, Ultrasound, PET, OCT, Endoscopy, Microscopy, Histopathology, Fundus, Dermoscopy, Angiography, Digital Photograph, and Medical Chart
- **Diverse Downstream Tasks**: Medical Dialogue, Anomaly Detection, Prognosis Prediction, Treatment Planning, Surgical Skill Assessment, Education, Medical Report Generation, Surgical Phase Recognition, Medical Computation, and more

## 🏆 Performance Highlights

### Medical Multimodal Benchmarks

**Table 1**: Performance comparison on medical multimodal benchmarks. For the 'Medical VLM < 10B' subgroup, **bold** and <u>underline</u> scores indicate the best and second-best methods, respectively.

| Models | Multi-modality Benchmarks |  | Specific-modality Benchmarks |  |  | Reasoning Benchmark | Knowledge-based Benchmark |
|--------|---------|---------|---------|-------|---------|---------|------------|
|  | OM.VQA | PMCVQA | VQA-RAD | SLAKE | PathVQA | MedXQA | MMMU-Med |

**Proprietary Models**
| GPT-4.1 | 75.5 | 55.2 | 65.0 | 72.2 | 55.5 | 45.2 | 75.2 |
| GPT-4o | 67.5 | 49.7 | 61.0 | 71.2 | 55.5 | 44.3 | 62.8 |
| Claude Sonnet 4 | 65.5 | 54.4 | 67.6 | 70.6 | 54.2 | 43.3 | 74.6 |
| Gemini-2.5-Flash | 71.0 | 55.4 | 68.5 | 75.8 | 55.4 | 52.8 | 76.9 |

**General-purpose Multimodal VLMs (Models < 10B)**
| Qwen2.5VL-7B | 63.6 | 51.9 | 63.2 | 66.8 | 44.1 | 20.1 | 50.6 |
| Janus-Pro-7B | 59.6 | 50.1 | 49.7 | 55.2 | 35.4 | 18.4 | 36.1 |
| InternVL2.5-8B | 81.3 | 51.3 | 59.4 | 69.0 | 42.1 | 21.7 | 53.5 |
| InternVL3-8B | 79.1 | 53.8 | 65.4 | 72.8 | 48.6 | 22.4 | 59.2 |

**General-purpose Multimodal VLMs (Models > 10B)**
| Llama3.2-11B | 43.8 | 48.1 | 58.8 | 65.8 | 32.9 | 20.1 | 51.0 |
| InternVL3-14B | 78.9 | 54.1 | 66.3 | 72.8 | 48.0 | 23.1 | 63.1 |
| Qwen2.5V-32B | 68.2 | 54.5 | 71.8 | 71.2 | 41.9 | 25.2 | 59.6 |
| InternVL2.5-38B | 79.9 | 57.2 | 61.4 | 70.3 | 46.9 | 24.4 | 61.6 |
| InternVL3-38B | 79.8 | 56.6 | 65.4 | 72.7 | 51.0 | 25.2 | 65.2 |

**Medical Multimodal VLMs (Models < 10B)**
| LLaVA-Med-7B | 34.8 | 22.7 | 46.6 | 51.9 | 35.2 | 20.8 | 28.1 |
| MedGemma-4B-IT | 70.7 | 49.2 | 72.3 | 78.2 | 48.1 | 25.4 | 43.2 |
| HuatuoGPT-V-7B | 74.3 | 53.1 | 67.6 | 68.1 | 44.8 | 23.2 | 49.8 |
| Lingshu-7B | 82.9 | 56.3 | 67.9 | 83.1 | 61.9 | 26.7 | - |
| **Hulu-Med-7B** | **84.2** | **66.8** | **78.0** | **86.8** | **65.6** | **29.0** | **51.4** |

**Medical Multimodal VLMs (Models > 10B)**
| HealthGPT-14B | 75.2 | 56.4 | 65.0 | 66.1 | 56.7 | 24.7 | 49.6 |
| HuatuoGPT-V-34B | 74.0 | 56.6 | 61.4 | 69.5 | 44.4 | 22.1 | 51.8 |
| Lingshu-32B | 83.4 | 57.9 | 76.7 | 86.7 | 65.5 | 30.9 | - |
| **Hulu-Med-14B** | **85.1** | **68.9** | **76.1** | **86.5** | **64.4** | **30.0** | **54.8** |
| **Hulu-Med-32B** | **84.6** | **69.4** | **81.4** | **85.7** | **67.3** | **34.0** | **60.4** |

### Medical Text Benchmarks

**Table 2**: Performance comparison on medical text benchmarks. Within each open-source subgroup, **bold** and <u>underline</u> scores indicate the best and second-best methods, respectively.

| Models | Complex Reasoning Benchmarks |  |  |  | Text Understanding | Medical Exam Benchmarks |  |  |
|--------|---------------|---------|-----------|-------|------------|---------|-------|---------|
|  | MMLU-Pro-Med | MedXQA | Medbullets | SGPQA | PubMedQA | MedMCQA | MedQA | MMLU-Med |

**Proprietary Models**
| GPT-4.1 | 78.0 | 30.9 | 77.0 | 49.9 | 75.6 | 77.7 | 89.1 | 89.6 |
| o3-mini | 78.1 | 35.4 | 83.7 | 50.1 | 73.6 | 60.6 | 74.5 | 87.0 |
| GPT-4o | 75.6 | 25.9 | 76.3 | 45.9 | 71.8 | 76.9 | 89.2 | 88.2 |
| Claude Sonnet 4 | 79.5 | 33.6 | 80.2 | 56.3 | 78.6 | 79.3 | 92.1 | 91.3 |
| Gemini-2.5-Flash | 70.0 | 35.6 | 77.6 | 53.3 | 73.8 | 73.6 | 91.2 | 84.2 |
| Deepseek-V3 | 74.6 | 20.0 | 48.4 | 32.1 | 77.7 | 88.0 | 51.0 | 86.5 |

**General-purpose Multimodal VLMs (Models < 10B)**
| Qwen2.5VL-7B | 50.5 | 12.8 | 42.1 | 26.3 | 76.4 | 52.6 | 57.3 | 73.4 |
| Janus-Pro-7B | 20.2 | 10.0 | 30.2 | 14.8 | 72.0 | 37.5 | 37.4 | 46.4 |
| InternVL2.5-8B | 50.6 | 11.6 | 42.4 | 26.1 | 76.4 | 52.4 | 53.7 | 74.2 |
| InternVL3-8B | 57.9 | 13.1 | 48.5 | 31.2 | 75.4 | 57.7 | 62.1 | 77.5 |

**General-purpose Multimodal VLMs (Models > 10B)**
| Qwen2.5VL-32B | 66.5 | 15.6 | 54.2 | 37.6 | 68.4 | 63.0 | 71.6 | 83.2 |
| InternVL3-14B | 65.4 | 14.1 | 49.5 | 37.9 | 77.2 | 62.0 | 70.1 | 81.7 |
| InternVL2.5-38B | 71.5 | 14.7 | 55.0 | 39.9 | 74.2 | 65.9 | 74.4 | 84.6 |
| InternVL3-38B | 72.1 | 16.0 | 54.6 | 42.5 | 73.2 | 64.9 | 73.5 | 83.8 |

**Medical Multimodal VLMs (Models < 10B)**
| LLaVA-Med-7B | 16.6 | 9.9 | 34.4 | 16.1 | 26.4 | 39.4 | 42.0 | 50.6 |
| MedGemma-4B-IT | 38.6 | 12.8 | 45.6 | 21.6 | 72.2 | 52.2 | 56.2 | 66.7 |
| HealthGPT-M3 | 38.3 | 11.5 | 41.4 | 18.9 | 57.8 | 54.2 | 55.0 | 72.5 |
| HuatuoGPT-V-7B | 44.6 | 10.1 | 40.9 | 21.9 | 72.8 | 51.2 | 52.9 | 69.3 |
| Lingshu-7B | 50.4 | 16.5 | 56.2 | 26.3 | 76.6 | 55.9 | 63.3 | 74.5 |
| **Hulu-Med-7B** | **60.6** | **19.6** | **61.5** | **31.1** | **77.4** | **67.6** | **73.5** | **79.5** |

**Medical Multimodal VLMs (Models > 10B)**
| HealthGPT-14B | 63.4 | 11.3 | 39.8 | 25.7 | 68.0 | 63.4 | 66.2 | 80.2 |
| Lingshu-32B | 70.2 | 22.7 | 65.4 | 41.1 | 77.8 | 66.1 | 74.7 | 84.7 |
| HuatuoGPT-V-34B | 51.8 | 11.4 | 42.7 | 26.5 | 72.2 | 54.7 | 58.8 | 74.7 |
| **Hulu-Med-14B** | **68.0** | **23.2** | **68.5** | **37.7** | **79.8** | **70.4** | **78.1** | **83.3** |
| **Hulu-Med-32B** | **72.9** | **24.2** | **68.8** | **41.8** | **80.8** | **72.8** | **80.4** | **85.6** |

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
