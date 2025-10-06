import torch
import torch.nn as nn
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# === 定義 Adapter ===
class ImageBindAdapter(nn.Module):
    def __init__(self, in_dim=1024, out_len=77, out_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_len = out_len

    def forward(self, x):
        x = self.mlp(x)  # [B, 1024]
        x = x.unsqueeze(1).repeat(1, self.out_len, 1)  # -> [B, 77, 1024]
        return x

# 載入 ImageBind 與 Adapter（只需初始化一次）
imagebind_model = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)
adapter = ImageBindAdapter().to(device)

# === 封裝函數 ===
def get_adapter_conditioning(text_prompt: str, image_path: str) -> torch.Tensor:
    """
    給定 text 和 image，回傳經 ImageBind + Adapter 處理後的 [1, 77, 1024] 向量。
    """
    # 處理輸入
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([text_prompt], device),
        ModalityType.VISION: data.load_and_transform_vision_data([image_path], device),
    }

    with torch.no_grad():
        embeddings = imagebind_model(inputs)
        text_emb = embeddings[ModalityType.TEXT]     # [1, 1024]
        image_emb = embeddings[ModalityType.VISION]  # [1, 1024]
        fused = (text_emb + image_emb) / 2           # [1, 1024]
        conditioning = adapter(fused)                # [1, 77, 1024]

    return conditioning
