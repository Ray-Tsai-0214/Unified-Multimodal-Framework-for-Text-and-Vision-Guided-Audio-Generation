# Multimodal-Make-An-Audio

This project is built upon [Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio), with the original codebase protected under the MIT License, copyright owned by Text-to-Audio.  
Modifications and additional content in this project are created by YuChi-Chen/PingJuei-Tsai/NYCU, also released under the MIT License.  
The original GitHub repository can be found [here](https://github.com/Text-to-Audio/Make-An-Audio).

## Adjustment

### Add new file

- `ldm/modules/encoders/embedding_encoder.py`
- `configs/train/mlsp_embedding_stage1.yaml.yaml`
- `configs/train/mlsp_embedding_stage2.yaml.yaml`

### Add new things in the existed file

- `ldm/modules/encoders/modules.py`
- `ldm\data\joinaudiodataset_624.py`

### Change function in the existed file

- `ldm/models/diffusion/ddpm_audio.py`
- `gen_wav.py`

### Training Techniques

- Add data augmentation（`ldm/data/joinaudiodataset_624.py`  `__getitem__`）
- 

## Quick Started

### Prepare paired data (audio / image / caption)

### Generate Embeddings

- Create the environment
  ```
  cd ImageBind
  conda create --name imagebind python=3.10 -y
  conda activate imagebind
  pip install .
  pip install pandas numpy==1.24.3 matplotlib
  pip uninstall torch torchvision torchaudio -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```  

  If you encounter the error `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'` during execution, please refer to the following solution: [报错：ModuleNotFoundError: No module named ‘torchvision.transforms.functional_tensor‘ 解决办法](https://blog.csdn.net/lanxing147/article/details/136625264)

- Run `mlsp.py`
- **Delete all other tsv file and remain the last one, make sure there's only one tsv file!!!**

### Make-An-Audio

- Download checkpoints
  ```
  maa1_full.ckpt and put it into ./useful_ckpts  
  BigVGAN vocoder and put it into ./useful_ckpts  
  CLAP_weights_2022.pth and put it into ./useful_ckpts/CLAP
  ```

- Create the environment
  ```
  cd ..
  conda deactivate
  conda create -n mlsp_final python=3.8
  conda activate mlsp_final
  python -m pip install pip==23.3.2
  pip install -r requirements.txt

  pip uninstall torch torchvision torchaudio -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- Generate mel-spectrogram from audio: `python preprocess/mel_spec.py --tsv_path "data/data_with_embeddings.tsv" --num_gpus 1 --max_duration 10`

- Train Make-An-Audio:
  - `python main.py --base configs\train\mlsp_embedding_stage1.yaml -t --gpus 1`

- View the training result
  - `tensorboard --logdir logs`
  - TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)

### Inference

```
python gen_wavs_by_tsv.py \
    --tsv_path test.tsv \
    --save_dir generated_audio/ \
    --ddim_steps 100
```

## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[CLAP](https://github.com/LAION-AI/CLAP),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ImageBind](https://github.com/facebookresearch/ImageBind), and [Make-An-Audio (ICML'23)](https://github.com/Text-to-Audio/Make-An-Audio).

# Disclaimer

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.



<!-- 

## Quick Started
We provide an example of how you can generate high-fidelity samples using Make-An-Audio.

To try on your own dataset, simply clone this repo in your local machine provided with NVIDIA GPU + CUDA cuDNN and follow the below instructions.


### Support Datasets and Pretrained Models

Simply run following command to download the weights from [Google drive](https://drive.google.com/drive/folders/1zZTI3-nHrUIywKFqwxlFO6PjB66JA8jI?usp=drive_link).
Download CLAP weights from [Hugging Face](https://huggingface.co/microsoft/msclap/blob/main/CLAP_weights_2022.pth).

```
Download:
    maa1_full.ckpt and put it into ./useful_ckpts  
    BigVGAN vocoder and put it into ./useful_ckpts  
    CLAP_weights_2022.pth and put it into ./useful_ckpts/CLAP
```
The directory structure should be:
```
useful_ckpts/
├── bigvgan
│   ├── args.yml
│   └── best_netG.pt
├── CLAP
│   ├── config.yml
│   └── CLAP_weights_2022.pth
└── maa1_full.ckpt
```


### Dependencies
See requirements in `requirement.txt`:

## Inference with pretrained model
```bash
python gen_wav.py --prompt "a bird chirps" --ddim_steps 100 --duration 10 --scale 3 --n_samples 1 --save_name "results"
```
# Train
## dataset preparation
We can't provide the dataset download link for copyright issues. We provide the process code to generate melspec.  
Before training, we need to construct the dataset information into a tsv file, which includes name (id for each audio), dataset (which dataset the audio belongs to), audio_path (the path of .wav file),caption (the caption of the audio) ,mel_path (the processed melspec file path of each audio). We provide a tsv file of audiocaps test set: ./data/audiocaps_test.tsv as a sample.
### generate the melspec file of audio
Assume you have already got a tsv file to link each caption to its audio_path, which mean the tsv_file have "name","audio_path","dataset" and "caption" columns in it.
To get the melspec of audio, run the following command, which will save mels in ./processed
```bash
python preprocess/mel_spec.py --tsv_path tmp.tsv --num_gpus 1 --max_duration 10
```
## Train variational autoencoder
Assume we have processed several datasets, and save the .tsv files in data/*.tsv . Replace **data.params.spec_dir_path** with the **data**(the directory that contain tsvs) in the config file. Then we can train VAE with the following command. If you don't have 8 gpus in your machine, you can replace --gpus 0,1,...,gpu_nums
```bash
python main.py --base configs/train/vae.yaml -t --gpus 0,1,2,3,4,5,6,7
```
The training result will be save in ./logs/
## train latent diffsuion
After Trainning VAE, replace model.params.first_stage_config.params.ckpt_path with your trained VAE checkpoint path in the config file.
Run the following command to train Diffusion model
```bash
python main.py --base configs/train/diffusion.yaml -t  --gpus 0,1,2,3,4,5,6,7
```
The training result will be save in ./logs/
# Evaluation
## generate audiocaps samples
```bash
python gen_wavs_by_tsv.py --tsv_path data/audiocaps_test.tsv --save_dir audiocaps_gen
```

## calculate FD,FAD,IS,KL
install [audioldm_eval](https://github.com/haoheliu/audioldm_eval) by
```bash
git clone git@github.com:haoheliu/audioldm_eval.git
```
Then test with:
```bash
python scripts/test.py --pred_wavsdir {the directory that saves the audios you generated} --gt_wavsdir {the directory that saves audiocaps test set waves}
```
## calculate Clap_score
```bash
python wav_evaluation/cal_clap_score.py --tsv_path {the directory that saves the audios you generated}/result.tsv
```
# X-To-Audio
## Audio2Audio
```bash
python scripts/audio2audio.py  --prompt "a bird chirping"  --strength 0.3 --init-audio sample.wav --ckpt useful_ckpts/maa1_full.ckpt --vocoder_ckpt useful_ckpts/bigvgan --config configs/text_to_audio/txt2audio_args.yaml --outdir audio2audio_samples
```

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[CLAP](https://github.com/LAION-AI/CLAP),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion),
as described in our code.

## Citations ##
If you find this code useful in your research, please consider citing:
```bibtex
@article{huang2023make,
  title={Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models},
  author={Huang, Rongjie and Huang, Jiawei and Yang, Dongchao and Ren, Yi and Liu, Luping and Li, Mingze and Ye, Zhenhui and Liu, Jinglin and Yin, Xiang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.12661},
  year={2023}
}
```

# Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws. -->
