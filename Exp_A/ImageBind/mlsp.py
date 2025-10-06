import os
import argparse
import pandas as pd
import torch
import logging
import random
import shutil
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# ============ Argument Parser ============

def get_parser():
    parser = argparse.ArgumentParser(description="ImageBind Multimodal Embedding Pipeline")
    parser.add_argument("--image_dir", type=str, default="../data/audiocaps_raw_audio")
    parser.add_argument("--audio_dir", type=str, default="../data/audiocaps_raw_audio")
    parser.add_argument("--text_dir", type=str, default="../data/audiocaps_raw_audio")
    parser.add_argument("--tsv_path", type=str, default="../data/data.tsv")
    parser.add_argument("--embedding_dir", type=str, default="../data/embeddings")
    parser.add_argument("--output_tsv_path", type=str, default="../data/data_with_embeddings.tsv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_check", action="store_true", help="進行隨機抽樣 embedding 檢查")
    return parser

# ============ 掃描資料夾並產生TSV ============

def scan_and_generate_tsv(image_dir, audio_dir, text_dir, output_tsv):
    logging.info("開始掃描資料夾")
    image_names = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png')))
    audio_names = set(os.path.splitext(f)[0] for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3')))
    text_names = set(os.path.splitext(f)[0] for f in os.listdir(text_dir) if f.endswith('.txt'))
    common_names = image_names & audio_names & text_names
    logging.info("找到交集資料數量: %d", len(common_names))

    data_rows = []
    for name in tqdm(sorted(common_names), desc="檢查檔案存在性"):
        image_path = os.path.join(image_dir, name + ".png")
        audio_path = os.path.join(audio_dir, name + ".wav")
        caption_path = os.path.join(text_dir, name + ".txt")
        if os.path.exists(image_path) and os.path.exists(audio_path) and os.path.exists(caption_path):
            data_rows.append({
                "name": name,
                "image_path": image_path,
                "audio_path": audio_path,
                "caption_path": caption_path
            })

    df = pd.DataFrame(data_rows)
    df.to_csv(output_tsv, sep="\t", index=False)
    logging.info("✅ TSV 檔案已建立，共 %d 筆資料", len(df))

# ============ 批次生成 embedding ============

def batch_inference(model, df, embedding_save_dir, batch_size, device, output_tsv_path):
    os.makedirs(embedding_save_dir, exist_ok=True)
    total = len(df)
    embedding_paths = {}

    for batch_start in tqdm(range(0, total, batch_size), desc="進行 Batch 推論"):
        batch_df = df.iloc[batch_start: batch_start+batch_size]
        image_paths = batch_df["image_path"].tolist()
        captions = [open(p, encoding="utf-8").read().strip() for p in batch_df["caption_path"].tolist()]
        names = batch_df["name"].tolist()

        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(captions, device),
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        for i, name in enumerate(names):
            text_embed = embeddings[ModalityType.TEXT][i].cpu()
            image_embed = embeddings[ModalityType.VISION][i].cpu()
            combined_embed = torch.cat([image_embed, text_embed])
            save_path = os.path.join(embedding_save_dir, f"{name}_embed.pt")
            torch.save(combined_embed, save_path)
            embedding_paths[name] = save_path

    df["embedding_path"] = df["name"].map(embedding_paths)
    df.to_csv(output_tsv_path, sep="\t", index=False)
    logging.info("✅ 已更新 TSV 並儲存至: %s", output_tsv_path)

# ============ 修正路徑 ============

def normalize_path(path):
    path = path.replace("\\", "/")
    path = path.lstrip("./")
    path = path.lstrip("../")
    return path

def fix_paths(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    path_columns = ["image_path", "audio_path", "caption_path", "embedding_path"]
    for col in path_columns:
        df[col] = df[col].apply(normalize_path)
    df.to_csv(tsv_path, sep="\t", index=False)
    logging.info("✅ 路徑修正完成！")

# ============ 隨機抽樣檢查 ============

def sample_embedding_check(embedding_dir, sample_num=3):
    all_files = [f for f in os.listdir(embedding_dir) if f.endswith(".pt")]
    if len(all_files) < sample_num:
        raise ValueError("Embedding 檔案數量不足")
    sample_files = random.sample(all_files, sample_num)
    for file in sample_files:
        path = os.path.join(embedding_dir, file)
        embedding = torch.load(path)
        embedding_numpy = embedding.numpy()
        print(f"檔案: {file}")
        print("Embedding shape:", embedding.shape)
        print("Embedding (前10維):", embedding_numpy[:10])
        print("-" * 50)

# ============ 主流程 ============

def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    if not os.path.exists(args.tsv_path):
        logging.info("未發現TSV，開始掃描生成")
        scan_and_generate_tsv(args.image_dir, args.audio_dir, args.text_dir, args.tsv_path)

    df = pd.read_csv(args.tsv_path, sep="\t")
    if len(df) == 0:
        logging.error("無有效資料，程式終止")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(device)
    logging.info("模型已載入並移至 %s", device)

    batch_inference(model, df, args.embedding_dir, args.batch_size, device, args.output_tsv_path)
    fix_paths(args.output_tsv_path)

    if args.sample_check:
        sample_embedding_check(args.embedding_dir)

    # ✅ 這裡新增自動刪除 embedding_dir
    if os.path.exists(args.embedding_dir):
        logging.info("刪除 embedding 目錄: %s", args.embedding_dir)
        shutil.rmtree(args.embedding_dir)
        logging.info("✅ Embedding 目錄刪除完成")

# ============ 程式入口 ============

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(args=[])  # for jupyter/debug 用
    main(args)
