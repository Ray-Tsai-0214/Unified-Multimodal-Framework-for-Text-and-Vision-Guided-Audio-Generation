import torch

class DebugLogger:
    """在訓練中添加調試日誌"""
    
    @staticmethod
    def log_vae_stats(model, batch, name=""):
        """記錄 VAE 編碼解碼的統計信息"""
        x = batch['image']
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = x.unsqueeze(0) if len(x.shape) == 3 else x
        x = x.to(model.device)
        
        with torch.no_grad():
            # 編碼
            z = model.get_first_stage_encoding(model.encode_first_stage(x))
            # 解碼
            x_rec = model.decode_first_stage(z)
            
            print(f"\n=== VAE Stats {name} ===")
            print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Latent range: [{z.min():.4f}, {z.max():.4f}]")
            print(f"Recon range: [{x_rec.min():.4f}, {x_rec.max():.4f}]")
            print(f"Scale factor: {model.scale_factor}")
            print(f"Recon error: {torch.mean((x - x_rec)**2):.6f}")
    
    @staticmethod
    def log_condition_stats(c, name="condition"):
        """記錄條件的統計信息"""
        if isinstance(c, torch.Tensor):
            print(f"\n=== {name} Stats ===")
            print(f"Shape: {c.shape}")
            print(f"Range: [{c.min():.4f}, {c.max():.4f}]")
            print(f"Mean: {c.mean():.4f}, Std: {c.std():.4f}")
            # 檢查是否有 NaN 或 Inf
            if torch.isnan(c).any():
                print("WARNING: Contains NaN values!")
            if torch.isinf(c).any():
                print("WARNING: Contains Inf values!")
