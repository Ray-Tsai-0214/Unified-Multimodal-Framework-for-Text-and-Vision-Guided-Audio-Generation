import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import argparse
import os

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--scale", type=float, default=5.0)
    opt = parser.parse_args()

    # Load model
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)
    sampler = DDIMSampler(model)
    
    os.makedirs(opt.outdir, exist_ok=True)
    
    # Load embedding
    embedding = torch.from_numpy(np.load(opt.embedding_path)).cuda()
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)    
    with torch.no_grad():
        with model.ema_scope():
            # Get conditioning
            c = model.get_learned_conditioning(embedding)
            uc = model.get_learned_conditioning(torch.zeros_like(embedding))
            
            # Sample
            shape = [4, 32, 32]  # Adjust based on your model
            samples, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=opt.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc
            )
            
            # Decode to mel spectrogram
            mel_specs = model.decode_first_stage(samples)
            
            # Save results
            for i, mel in enumerate(mel_specs):
                mel_path = os.path.join(opt.outdir, f"mel_{i:04d}.npy")
                np.save(mel_path, mel.cpu().numpy())
                print(f"Saved mel spectrogram to {mel_path}")

if __name__ == "__main__":
    main()