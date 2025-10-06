import torch
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from vocoder.bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
import argparse
import os

def generate_audio(embedding_path, model_config, model_ckpt, vocoder_path, output_path):
    # Load diffusion model
    config = OmegaConf.load(model_config)
    model = instantiate_from_config(config.model)
    
    # Load checkpoint
    ckpt = torch.load(model_ckpt, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    
    # Load vocoder
    vocoder = VocoderBigVGAN(vocoder_path)
    
    # Load embedding
    embedding = np.load(embedding_path)
    embedding = torch.from_numpy(embedding).cuda()
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    
    with torch.no_grad():
        # Generate mel spectrogram
        sampler = DDIMSampler(model)
        
        with model.ema_scope():
            c = model.get_learned_conditioning(embedding)
            uc = model.get_learned_conditioning(torch.zeros_like(embedding))
            
            shape = [4, 78, 78]  # latent shape
            samples, _ = sampler.sample(
                S=200,
                conditioning=c,
                batch_size=1,
                shape=shape,
                verbose=True,
                unconditional_guidance_scale=5.0,
                unconditional_conditioning=uc
            )            
            # Decode to mel spectrogram
            mel = model.decode_first_stage(samples)
            mel = mel.squeeze(0).cpu().numpy()
            
            # Convert to audio
            wav = vocoder.vocode(mel)
            
            # Save audio
            sf.write(output_path, wav, 16000)
            print(f"Generated audio saved to {output_path}")
            
            return wav, mel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", type=str, required=True, help="Path to embedding .npy file")
    parser.add_argument("--config", type=str, default="configs/train/mlsp_embedding_stage2.yaml")
    parser.add_argument("--ckpt", type=str, default="logs/2025-05-31T19-16-27_mlsp_embedding_stage2/checkpoints/last.ckpt")
    parser.add_argument("--vocoder", type=str, default="useful_ckpts/bigvgan")
    parser.add_argument("--output", type=str, default="output.wav")
    
    args = parser.parse_args()
    
    generate_audio(
        args.embedding,
        args.config,
        args.ckpt,
        args.vocoder,
        args.output
    )