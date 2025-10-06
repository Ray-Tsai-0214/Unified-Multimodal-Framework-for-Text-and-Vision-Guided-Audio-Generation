# 📖 Usage Guide

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with 16GB+ VRAM (RTX 4090 recommended)
- Python 3.8+ and 3.10+ (dual environments required)
- CUDA 12.1+
- 50GB+ free disk space

## 📋 Step-by-Step Setup

### 1. Environment Preparation

#### ImageBind Environment
```bash
cd Exp_A/ImageBind
conda create --name imagebind python=3.10 -y
conda activate imagebind
pip install .
pip install pandas numpy==1.24.3 matplotlib
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Make-An-Audio Environment
```bash
cd ../
conda create -n multimodal-audio python=3.8 -y
conda activate multimodal-audio
python -m pip install pip==23.3.2
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download Required Models

Create directory structure:
```bash
mkdir -p useful_ckpts/bigvgan
mkdir -p useful_ckpts/CLAP
```

Download the following models:
- **Make-An-Audio checkpoint**: `maa1_full.ckpt` → `./useful_ckpts/`
- **BigVGAN vocoder**: 
  - `args.yml` → `./useful_ckpts/bigvgan/`
  - `best_netG.pt` → `./useful_ckpts/bigvgan/`
- **CLAP weights**: 
  - `CLAP_weights_2022.pth` → `./useful_ckpts/CLAP/`
  - `config.yml` → `./useful_ckpts/CLAP/`

## 🎯 Usage Examples

### Generate Multimodal Embeddings

```bash
conda activate imagebind
cd Exp_A/ImageBind
python mlsp.py
```

**Important**: Delete all TSV files except the final one before proceeding!

### Process Audio Data

```bash
conda activate multimodal-audio
cd ../
python preprocess/mel_spec.py \
    --tsv_path "data/data_with_embeddings.tsv" \
    --num_gpus 1 \
    --max_duration 10
```

### Train Model (Experiment A)

**Stage 1: Projection Training**
```bash
python main.py \
    --base configs/train/mlsp_embedding_stage1.yaml \
    -t --gpus 1
```

Monitor training progress:
```bash
tensorboard --logdir logs
```

### Generate Audio

```bash
python gen_wavs_by_tsv.py \
    --tsv_path test.tsv \
    --save_dir generated_audio/ \
    --ddim_steps 100
```

### Single Audio Generation

```bash
python gen_wav.py \
    --prompt "a bird chirps in the forest" \
    --ddim_steps 100 \
    --duration 10 \
    --scale 3 \
    --n_samples 1 \
    --save_name "bird_forest"
```

## 🔧 Advanced Usage

### Custom Dataset Preparation

1. **Create TSV file** with columns: `name`, `dataset`, `audio_path`, `caption`, `mel_path`

Example TSV format:
```tsv
name	dataset	audio_path	caption	mel_path
sample_001	custom	audio/sample_001.wav	dog barking loudly	processed/sample_001_mel.npy
sample_002	custom	audio/sample_002.wav	rain on metal roof	processed/sample_002_mel.npy
```

2. **Generate mel-spectrograms**:
```bash
python preprocess/mel_spec.py \
    --tsv_path your_dataset.tsv \
    --num_gpus 1 \
    --max_duration 10
```

### Experiment B: CLAP Alignment

```bash
cd Exp_B/Adapter
python train.py
```

Monitor training:
```bash
# View loss curves
python -c "
import matplotlib.pyplot as plt
import torch
# Load and plot training metrics
"
```

### Audio-to-Audio Generation

```bash
python scripts/audio2audio.py \
    --prompt "birds chirping peacefully" \
    --strength 0.3 \
    --init-audio sample_input.wav \
    --ckpt useful_ckpts/maa1_full.ckpt \
    --vocoder_ckpt useful_ckpts/bigvgan \
    --config configs/text_to_audio/txt2audio_args.yaml \
    --outdir audio2audio_samples
```

## 📊 Evaluation

### Audio Quality Metrics

```bash
# Install audioldm_eval
git clone https://github.com/haoheliu/audioldm_eval.git

# Calculate FAD, IS, KL metrics
python scripts/test.py \
    --pred_wavsdir generated_audio/ \
    --gt_wavsdir ground_truth_audio/
```

### CLAP Score Evaluation

```bash
python wav_evaluation/cal_clap_score.py \
    --tsv_path generated_audio/result.tsv
```

## 🐛 Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`

**Solution**: Update torchvision
```bash
pip uninstall torchvision -y
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 2. CUDA Out of Memory

**Solutions**:
- Reduce batch size in config files
- Use gradient checkpointing
- Clear GPU cache: `torch.cuda.empty_cache()`

#### 3. TSV File Issues

**Requirements**:
- Only one TSV file should exist in the data directory
- Ensure all paths in TSV file are absolute
- Check file encoding (UTF-8 recommended)

#### 4. Audio Generation Quality Issues

**Check**:
- Model checkpoint integrity
- Input text/image quality
- Mel-spectrogram preprocessing
- Vocoder configuration

### Performance Optimization

#### GPU Memory Optimization
```python
# Add to training script
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

#### Training Speed Up
```bash
# Use mixed precision training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 📁 File Organization

### Input Data Structure
```
data/
├── audio/
│   ├── sample_001.wav
│   └── sample_002.wav
├── images/
│   ├── sample_001.jpg
│   └── sample_002.jpg
└── metadata.tsv
```

### Output Structure
```
generated_audio/
├── audio/
│   ├── generated_001.wav
│   └── generated_002.wav
├── logs/
│   └── generation_log.txt
└── result.tsv
```

## 🎛️ Configuration Options

### Training Parameters

```yaml
# configs/train/custom.yaml
model:
  params:
    learning_rate: 1e-4
    batch_size: 4
    num_epochs: 50
    
data:
  params:
    max_duration: 10
    sample_rate: 16000
    n_mels: 80
    
generation:
  params:
    ddim_steps: 100
    guidance_scale: 3.0
    num_samples: 1
```

### Generation Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `ddim_steps` | Denoising steps | 100-200 |
| `guidance_scale` | Text guidance strength | 2.0-5.0 |
| `duration` | Audio length (seconds) | 5-10 |
| `temperature` | Sampling randomness | 0.8-1.2 |

## 📞 Support

For technical issues:
1. Check the troubleshooting section above
2. Verify environment setup
3. Ensure all dependencies are correctly installed
4. Check GPU memory and CUDA compatibility

For research inquiries:
- Email: allcare.c@nycu.edu.tw
- Institution: National Yang Ming Chiao Tung University