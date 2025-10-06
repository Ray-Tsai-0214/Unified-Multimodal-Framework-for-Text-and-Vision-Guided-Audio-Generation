import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from ldm.modules.encoders.modules import FrozenCLAPEmbedder
from blip import blip
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib.pyplot as plt  # ✅ 新增




class ImageBindAdapter(nn.Module):
    def __init__(self, input_dim=1024, token_dim=1024, num_tokens=77, num_layers=4, num_heads=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.token_query = nn.Parameter(torch.randn(1, num_tokens, token_dim))  # learnable queries
        self.project = nn.Linear(input_dim, token_dim)  # project ImageBind fusion

        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, 1024]
        x = self.project(x)  # [B, 1024]
        x = x.unsqueeze(1)   # [B, 1, 1024]

        token_q = self.token_query.expand(x.size(0), -1, -1)  # [B, 77, 1024]
        out = self.transformer(token_q, src_key_padding_mask=None)  # self-attention only on token_q
        return out  # [B, 77, 1024]

# === Dataset 定義 ===
class MultiModalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.fnames = []
        self.missing = []

        for f in os.listdir(root_dir):
            if f.endswith(".png"):
                base = f[:-4]
                txt_path = os.path.join(root_dir, base + ".txt")
                if os.path.exists(txt_path):
                    self.fnames.append(base)
                else:
                    self.missing.append(base)

        print(f"[INFO] Total usable samples: {len(self.fnames)}")
        print(f"[WARNING] Missing .txt for {len(self.missing)} samples")

    def __len__(self):
        return len(self.fnames)
    
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.root_dir, fname + ".png")
        txt_path = os.path.join(self.root_dir, fname + ".txt")
        ocr_path = os.path.join(self.root_dir, fname + "_o.txt")  # 👈 BLIP caption

        image = Image.open(img_path).convert("RGB")

        # === 載入兩種 caption 並結合 ===
        text = open(txt_path).read().strip()
        blip_caption = open(ocr_path).read().strip()
        full_text = text + " and " + blip_caption

        if self.transform:
            image = self.transform(image)

        return image, text, full_text, img_path


# === 初始化 ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入模型
imagebind = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)
clap_encoder = FrozenCLAPEmbedder(device=device).eval().to(device)
adapter = ImageBindAdapter().to(device)
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-5)

# === transform 設定 ===export CC=/usr/bin/gcc-11

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])



dataset = MultiModalDataset("MLSP_train_data", transform=image_transform)

total_len = len(dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len  # 確保總數正確

train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)



@torch.no_grad()
def evaluate(model, dataloader, clap_encoder):
    model.eval()
    total_loss = 0
    count = 0

    for image_tensor, text, full_text, img_path in dataloader:
        image_tensor = image_tensor.to(device)
        texts = list(text)
        paths = list(img_path)

        vision_input = imagebind_data.load_and_transform_vision_data(paths, device)
        text_input = imagebind_data.load_and_transform_text(texts, device)

        inputs = {
            ModalityType.TEXT: text_input,
            ModalityType.VISION: vision_input,
        }

        embeddings = imagebind(inputs)
        fused = (embeddings[ModalityType.TEXT] + embeddings[ModalityType.VISION]) / 2

        pred_tokens = model(fused)

        gt_tokens = clap_encoder.encode(full_text)  # [1, 77, 1024]
        gt_tokens = gt_tokens.expand(pred_tokens.size(0), -1, -1)

        mse = F.mse_loss(pred_tokens, gt_tokens)
        cos = 1 - F.cosine_similarity(pred_tokens, gt_tokens, dim=-1).mean()
        loss = 2 * mse + cos

        total_loss += loss.item()
        count += 1
        
        

    return total_loss / count





# === 訓練迴圈 ===




train_losses = []
val_losses = []

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


for epoch in range(20):
    running_loss = 0
    for image_tensor, text, full_text, img_path in tqdm(train_loader, desc=f"Epoch {epoch}"):
        image_tensor = image_tensor.to(device)

        #  text + image path
        texts = list(text)
        paths = list(img_path)
        #breakpoint()

        vision_input = imagebind_data.load_and_transform_vision_data(paths, device)
        text_input = imagebind_data.load_and_transform_text(texts, device)

        inputs = {
            ModalityType.TEXT: text_input,
            ModalityType.VISION: vision_input,
        }

        with torch.no_grad():
            embeddings = imagebind(inputs)
            fused = (embeddings[ModalityType.TEXT] + embeddings[ModalityType.VISION]) / 2  # [B, 1024]

        pred_tokens = adapter(fused)  # [B, 77, 1024]
        #breakpoint()
        
        # === Ground truth from CLAP ===
        with torch.no_grad():
            gt_tokens = clap_encoder.encode(full_text)  # [1, 77, 1024]
            gt_tokens = gt_tokens.expand(pred_tokens.size(0), -1, -1)

        mse = F.mse_loss(pred_tokens, gt_tokens)
        cos = 1 - F.cosine_similarity(pred_tokens, gt_tokens, dim=-1).mean()
        loss = 2 * mse + cos

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    


    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
    val_loss = evaluate(adapter, val_loader, clap_encoder)
    print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}")
    
    
    train_losses.append(loss.item())
    val_loss = evaluate(adapter, val_loader, clap_encoder)
    val_losses.append(val_loss)
    scheduler.step()
    
    
    
    
    
    
test_loss = evaluate(adapter, test_loader, clap_encoder)
print(f"[Test Loss]: {test_loss:.4f}")

test_loss = evaluate(adapter, test_loader, clap_encoder)
print(f"[Test Loss]: {test_loss:.4f}")
torch.save(adapter.state_dict(), "adapter_token77.pth")

# === 畫圖並儲存 Loss Curve ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Adapter Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()

# === 儲存 Adapter 模型 ===
torch.save(adapter.state_dict(), "adapter_token77.pth")
