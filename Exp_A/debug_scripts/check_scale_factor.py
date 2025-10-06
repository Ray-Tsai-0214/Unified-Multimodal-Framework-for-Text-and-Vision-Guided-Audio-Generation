import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def check_scale_factor():
    """檢查 VAE 的 scale_factor 問題"""
    
    # 1. 加載配置
    config_path = "configs/train/mlsp_embedding_stage1.yaml"
    config = OmegaConf.load(config_path)
    
    # 2. 加載模型
    model = instantiate_from_config(config.model)
    
    # 3. 加載 checkpoint
    ckpt_path = "useful_ckpts/maa1_full-001.ckpt"
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 4. 檢查 scale_factor
    if 'scale_factor' in checkpoint['state_dict']:
        pretrained_scale_factor = checkpoint['state_dict']['scale_factor'].item()
        print(f"Pretrained scale_factor: {pretrained_scale_factor}")
    else:
        print("No scale_factor in checkpoint!")
    
    # 5. 加載一些數據來計算新的 scale_factor
    from ldm.data.joinaudiodataset_624 import JoinSpecsTrainWithEmbedding
    dataset_config = {
        'spec_dir_path': 'data',
        'embedding_dir_path': 'data/embeddings',
        'spec_crop_len': 624,
        'mel_num': 80,
        'drop': 0.0
    }
    
    dataset = JoinSpecsTrainWithEmbedding({'specs_dataset_cfg': dataset_config})
    
    # 6. 計算前幾個樣本的統計
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    z_values = []
    for i in range(min(10, len(dataset))):
        data = dataset[i]
        x = torch.FloatTensor(data['image']).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 編碼
            encoder_posterior = model.encode_first_stage(x)
            if hasattr(encoder_posterior, 'sample'):
                z = encoder_posterior.sample()
            else:
                z = encoder_posterior
            z_values.append(z.cpu())
    
    # 7. 計算統計
    all_z = torch.cat(z_values, dim=0)
    z_mean = all_z.mean().item()
    z_std = all_z.std().item()
    z_min = all_z.min().item()
    z_max = all_z.max().item()
    
    print(f"\n您的數據的潛在空間統計：")
    print(f"Mean: {z_mean:.4f}")
    print(f"Std: {z_std:.4f}")
    print(f"Min: {z_min:.4f}")
    print(f"Max: {z_max:.4f}")
    print(f"建議的 scale_factor: {1.0 / z_std:.4f}")
    
    # 8. 測試編碼-解碼
    print("\n測試編碼-解碼循環：")
    x = torch.FloatTensor(dataset[0]['image']).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 原始
        print(f"Input shape: {x.shape}")
        print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        
        # 編碼
        z = model.get_first_stage_encoding(model.encode_first_stage(x))
        print(f"Encoded z shape: {z.shape}")
        print(f"Encoded z range: [{z.min():.4f}, {z.max():.4f}]")
        
        # 解碼
        x_rec = model.decode_first_stage(z)
        print(f"Reconstructed shape: {x_rec.shape}")
        print(f"Reconstructed range: [{x_rec.min():.4f}, {x_rec.max():.4f}]")
        
        # 計算重構誤差
        mse = torch.mean((x - x_rec) ** 2).item()
        print(f"Reconstruction MSE: {mse:.6f}")

if __name__ == "__main__":
    check_scale_factor()
