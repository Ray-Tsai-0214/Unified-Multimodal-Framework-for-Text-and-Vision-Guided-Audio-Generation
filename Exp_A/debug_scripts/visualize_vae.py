import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_vae_process(model, batch, save_path="vae_process.png"):
    """可視化 VAE 編碼解碼過程"""
    
    x = batch['image']
    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)
    x = x.unsqueeze(0) if len(x.shape) == 3 else x
    x = x.to(model.device)
    
    with torch.no_grad():
        # 1. 原始輸入
        input_spec = x[0, 0].cpu().numpy()
        
        # 2. 編碼到潛在空間
        encoder_posterior = model.encode_first_stage(x)
        z_no_scale = encoder_posterior.sample() if hasattr(encoder_posterior, 'sample') else encoder_posterior
        z_with_scale = model.get_first_stage_encoding(encoder_posterior)
        
        # 3. 解碼（使用正確的 scale）
        x_rec = model.decode_first_stage(z_with_scale)
        rec_spec = x_rec[0, 0].cpu().numpy()
        
        # 4. 解碼（不使用 scale）- 錯誤的方式
        x_rec_wrong = model.first_stage_model.decode(z_no_scale)
        rec_wrong = x_rec_wrong[0, 0].cpu().numpy()
        
        # 5. 加噪後解碼（模擬 diffusion_row）
        noise = torch.randn_like(z_with_scale)
        z_noisy = z_with_scale + 0.5 * noise  # 中等程度的噪聲
        x_noisy = model.decode_first_stage(z_noisy)
        noisy_spec = x_noisy[0, 0].cpu().numpy()
    
    # 創建可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 繪製頻譜圖
    specs = [
        (input_spec, "Original Input", axes[0, 0]),
        (rec_spec, "Correct Reconstruction", axes[0, 1]),
        (rec_wrong, "Wrong Reconstruction (no scale)", axes[1, 0]),
        (noisy_spec, "Noisy (like diffusion_row)", axes[1, 1])
    ]
    
    for spec, title, ax in specs:
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # 打印統計信息
    print(f"Input range: [{input_spec.min():.4f}, {input_spec.max():.4f}]")
    print(f"Correct recon range: [{rec_spec.min():.4f}, {rec_spec.max():.4f}]")
    print(f"Wrong recon range: [{rec_wrong.min():.4f}, {rec_wrong.max():.4f}]")
    print(f"Scale factor: {model.scale_factor}")
    
    return {
        'input': input_spec,
        'correct_rec': rec_spec,
        'wrong_rec': rec_wrong,
        'noisy': noisy_spec
    }
