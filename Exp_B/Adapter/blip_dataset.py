import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def blip(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_dir = "MLSP_train_data"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()

    for fname in os.listdir(image_dir):
        if fname.endswith(".png"):
            img_path = os.path.join(image_dir, fname)
            base_name = os.path.splitext(fname)[0]  # 例如 "1.png" → "1"
            out_path = os.path.join(image_dir, f"{base_name}_o.txt")

            image = Image.open(img_path).convert("RGB")
            caption = blip(image, processor, model)

            with open(out_path, "w") as f:
                f.write(caption)

            print(f"[Saved] {out_path} ← {caption}")
