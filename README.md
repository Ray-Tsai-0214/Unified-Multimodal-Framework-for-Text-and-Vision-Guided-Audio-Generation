# 🎵 Unified Multimodal Framework for Text and Vision Guided Audio Generation

> **A cutting-edge research project extending Make-An-Audio with ImageBind integration for multimodal audio synthesis**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📝 Overview

This project presents a novel framework that extends the Make-An-Audio system to support multimodal conditioning by unifying text and image representations for audio generation. While existing text-to-audio generation systems have achieved remarkable progress, they are limited by the inherent ambiguity of text-only conditioning. Visual information provides complementary contextual cues that can disambiguate and enrich the generation process.

### 🎯 Key Innovation

Our approach leverages **ImageBind's unified multimodal embedding space** to enable joint text-image guided audio generation, addressing the fundamental limitation of text-only audio synthesis systems.

## 🏗️ Architecture

### Core Components

1. **ImageBind Integration**: Unified multimodal embedding space across six modalities
2. **Make-An-Audio Foundation**: Base text-to-audio generation framework
3. **Automated Data Pipeline**: n8n workflow for multimodal dataset creation
4. **Dual Experimental Design**: Comparative analysis of integration strategies

### 🔬 Experimental Approaches

#### Experiment A: Direct Replacement Strategy
- **Approach**: Direct substitution of CLAP with ImageBind-based multimodal encoder
- **Architecture**: ImageBind Encoder → Fusion Module → Projection Layer → U-Net
- **Training**: Two-stage process (projection training + end-to-end fine-tuning)

#### Experiment B: CLAP Alignment Strategy  
- **Approach**: Learnable transformation to maintain CLAP compatibility
- **Architecture**: ImageBind Encoder → Fusion Module → Token Sequence → CLAP Supervision
- **Advantage**: Stable convergence through existing representation space alignment

## 🚀 Key Features

- **🔄 Multimodal Conditioning**: Simultaneous text and image input processing
- **🤖 Automated Data Generation**: Intelligent n8n workflow for dataset expansion
- **📊 Comprehensive Analysis**: Detailed experimental comparison and failure analysis
- **🎨 Creative Applications**: Film production, gaming, VR, and accessibility tools

## 📊 Results & Insights

### Performance Comparison

| Metric | Experiment A | Experiment B |
|--------|--------------|--------------|
| Initial Convergence | Poor | Good |
| Training Stability | Unstable | Stable |
| Validation Performance | Very Poor | Limited |
| Overfitting Severity | Severe | Moderate |
| Implementation Complexity | High | Low |
| Deployment Feasibility | Difficult | Feasible |

### Key Findings

1. **Alignment-based approaches** outperform direct replacement strategies
2. **Feature space compatibility** is more critical than architectural sophistication  
3. **Simple averaging fusion** performs better than complex mechanisms with limited data
4. **CLAP supervision** enables stable learning through established representation spaces

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Models**: ImageBind, Make-An-Audio, CLAP, BigVGAN
- **Audio Processing**: Mel-spectrogram, Latent Diffusion Models
- **Automation**: n8n workflow, GPT-4.1 for semantic decomposition
- **Image Generation**: FLUX.1-schnell
- **Evaluation**: FAD, IS, CLAP Score, Cosine Similarity

## 📁 Project Structure

```
multimodal-audio-generation/
├── 📂 Exp_A/                          # Direct Replacement Experiment
│   ├── 📂 ImageBind/                  # ImageBind integration
│   ├── 📂 Make-An-Audio/              # Base audio generation framework
│   ├── 📂 ldm/                       # Latent diffusion models
│   ├── 📂 configs/                   # Training configurations
│   ├── 📂 scripts/                   # Inference and evaluation scripts
│   └── 📂 processed/                 # Processed audio data
├── 📂 Exp_B/                          # CLAP Alignment Experiment  
│   └── 📂 Adapter/                   # Learnable transformation modules
├── 📂 n8n_workflow/                   # Automated data generation
├── 📄 Unified_Multimodal_Framework_for_Text_and_Vision_Guided_Audio_Generation.pdf
└── 📄 README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (RTX 4090 recommended)
- 16GB+ GPU memory

### Installation

#### 1. Environment Setup for ImageBind

```bash
cd Exp_A/ImageBind
conda create --name imagebind python=3.10 -y
conda activate imagebind
pip install .
pip install pandas numpy==1.24.3 matplotlib
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Environment Setup for Make-An-Audio

```bash
cd ../
conda create -n multimodal-audio python=3.8
conda activate multimodal-audio
python -m pip install pip==23.3.2
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Download Pre-trained Models

Download the following checkpoints and place them in `./useful_ckpts/`:
- `maa1_full.ckpt` - Make-An-Audio checkpoint
- `BigVGAN vocoder` - Neural vocoder
- `CLAP_weights_2022.pth` - CLAP encoder weights

### 🎯 Usage

#### Generate Embeddings
```bash
cd Exp_A/ImageBind
python mlsp.py
```

#### Process Audio Data
```bash
python preprocess/mel_spec.py --tsv_path "data/data_with_embeddings.tsv" --num_gpus 1 --max_duration 10
```

#### Train Model (Experiment A)
```bash
python main.py --base configs/train/mlsp_embedding_stage1.yaml -t --gpus 1
```

#### Generate Audio
```bash
python gen_wavs_by_tsv.py \
    --tsv_path test.tsv \
    --save_dir generated_audio/ \
    --ddim_steps 100
```

## 🎨 Applications

### 🎬 Film & Media Production
- **Context-aware sound design**: Generate audio matching specific visual scenes
- **Efficient post-production**: Reduce manual sound design time from 4-8 hours to 1-2 hours
- **Creative exploration**: Rapid prototyping of soundscapes

### 🎮 Interactive Gaming & VR
- **Dynamic audio generation**: Real-time audio synthesis for procedural environments
- **Contextual soundscapes**: Audio that adapts to visual environments
- **Resource optimization**: Reduce large audio asset libraries

### ♿ Assistive Technology
- **Enhanced accessibility**: Generate contextual ambient sounds for visually impaired users
- **Audio descriptions**: Complement narrative with appropriate environmental audio
- **Cultural immersion**: Create authentic soundscapes for educational content

## 📈 Automated Data Generation Pipeline

Our innovative **n8n workflow** automatically creates complementary multimodal training pairs:

### Pipeline Stages

1. **🧠 Semantic Decomposition**: GPT-4.1 splits audio captions into visual and textual components
2. **🖼️ Image Generation**: FLUX.1-schnell creates corresponding visual context
3. **✅ Quality Control**: Ensures zero overlap between components
4. **💾 Automated Storage**: Systematic organization in Google Drive

### Benefits
- **📈 Scalability**: Transform hundreds into thousands of training pairs
- **⏰ Efficiency**: 24/7 automated operation
- **💰 Cost-effective**: Reduce manual annotation costs significantly
- **🎯 Consistency**: Maintain systematic quality standards

## 📊 Evaluation Metrics

- **🎵 Audio Quality**: FAD (Fréchet Audio Distance), IS (Inception Score)
- **🔗 Multimodal Alignment**: CLAP Score, Cosine Similarity
- **📈 Training Dynamics**: Loss convergence, validation performance
- **👥 Human Evaluation**: Context appropriateness, perceived quality

## 🔧 Future Work

### 🎯 Research Directions

1. **📊 Scalable Training**: Progressive learning strategies for larger datasets
2. **📏 Evaluation Framework**: Standardized multimodal audio generation metrics
3. **🏗️ Architecture Innovation**: Hierarchical fusion mechanisms
4. **🎬 Real-world Deployment**: Professional tool integration studies

### 🚀 Technical Improvements

- **🎥 Temporal Conditioning**: Extend to video-guided audio generation
- **⚖️ Adaptive Weighting**: Dynamic modal influence adjustment
- **🌍 Cross-modal Understanding**: Deeper semantic alignment analysis

## 📚 Academic Paper

**Title**: "Unified Multimodal Framework for Text and Vision Guided Audio Generation: Extending Make-An-Audio with ImageBind Integration"

**Authors**: Ray Tsai, YuChi Chen, Yi-Chuan Huang (National Yang Ming Chiao Tung University)

**Abstract**: This research extends Make-An-Audio to support multimodal conditioning through ImageBind integration, enabling complementary text-image guided audio generation and addressing the inherent limitations of text-only conditioning systems.

[📄 Read Full Paper](./Unified_Multimodal_Framework_for_Text_and_Vision_Guided_Audio_Generation.pdf)

## 🤝 Acknowledgments

This implementation builds upon several outstanding open-source projects:
- [Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio) - Base audio generation framework
- [ImageBind](https://github.com/facebookresearch/ImageBind) - Multimodal representation learning
- [CLAP](https://github.com/LAION-AI/CLAP) - Audio-text alignment
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Latent diffusion models

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This technology is intended for research and creative applications only. Any organization or individual is prohibited from using this technology to generate audio content without proper consent, including but not limited to speeches of public figures, celebrities, or copyrighted material.

## 🏆 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{tsai2024unified,
  title={Unified Multimodal Framework for Text and Vision Guided Audio Generation: Extending Make-An-Audio with ImageBind Integration},
  author={Tsai, Ray and Chen, YuChi and Huang, Yi-Chuan},
  journal={National Yang Ming Chiao Tung University},
  year={2024}
}
```

---

<div align="center">

**🎵 Bridging Vision, Language, and Sound through AI 🎵**

*Developed with ❤️ at National Yang Ming Chiao Tung University*

</div>