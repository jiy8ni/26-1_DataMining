"""
Build raw semantic embeddings (one row per unique resolved_url) from the
crawled page text and images produced by crawl_pages.py.

Text  : BAAI/bge-m3 (multilingual, strong Korean)        -> 1024-d
Image : openai/clip-vit-base-patch32 (image encoder)     -> 512-d
        multiple images per page are mean-pooled.

Output (keyed by resolved_url):
    data/processed/item_text_emb.parquet   resolved_url, txt_0 .. txt_1023
    data/processed/item_image_emb.parquet  resolved_url, img_0 .. img_511, img_missing

These RAW embeddings carry no label information, so they are computed once over
all items (fold-independent). Fold-aware PCA reduction happens later in data.py.

Usage:
    python src/build_embeddings.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES_PARQUET = REPO_ROOT / "data" / "raw" / "crawl" / "pages.parquet"
TEXT_EMB_PATH = REPO_ROOT / "data" / "processed" / "item_text_emb.parquet"
IMAGE_EMB_PATH = REPO_ROOT / "data" / "processed" / "item_image_emb.parquet"

TEXT_MODEL = "BAAI/bge-m3"
TEXT_DIM = 1024
CLIP_MODEL = "openai/clip-vit-base-patch32"
IMAGE_DIM = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_text_embeddings(pages: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    print(f"[text] loading {TEXT_MODEL} on {DEVICE} ...")
    model = SentenceTransformer(TEXT_MODEL, device=DEVICE)

    texts = pages["page_text"].fillna("").astype(str).tolist()
    # bge-m3 truncates internally to its max sequence length (8192 tokens).
    emb = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    cols = [f"txt_{i}" for i in range(emb.shape[1])]
    out = pd.DataFrame(emb, columns=cols)
    out.insert(0, "resolved_url", pages["resolved_url"].values)
    return out


def build_image_embeddings(pages: pd.DataFrame) -> pd.DataFrame:
    from transformers import CLIPModel, CLIPProcessor

    print(f"[image] loading {CLIP_MODEL} on {DEVICE} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    vecs = np.zeros((len(pages), IMAGE_DIM), dtype=np.float32)
    missing = np.ones(len(pages), dtype=np.int8)

    for row_i, (_, row) in enumerate(pages.iterrows()):
        files = row["image_files"]
        if files is None or len(files) == 0:
            continue
        imgs = []
        for rel in files:
            try:
                imgs.append(Image.open(REPO_ROOT / rel).convert("RGB"))
            except Exception:
                continue
        if not imgs:
            continue
        with torch.no_grad():
            inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
            feats = model.get_image_features(**inputs)  # (n_img, 512)
            # transformers >=5 returns a model-output object whose projected
            # 512-d image embedding is in .pooler_output (older versions
            # returned the tensor directly).
            if not torch.is_tensor(feats):
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1)
            pooled = feats.mean(dim=0)                    # mean-pool over page images
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vecs[row_i] = pooled.cpu().numpy()
        missing[row_i] = 0
        if (row_i + 1) % 25 == 0:
            print(f"    [image] {row_i + 1}/{len(pages)}")

    cols = [f"img_{i}" for i in range(IMAGE_DIM)]
    out = pd.DataFrame(vecs, columns=cols)
    out.insert(0, "resolved_url", pages["resolved_url"].values)
    out["img_missing"] = missing
    return out


def main() -> None:
    if not PAGES_PARQUET.exists():
        raise FileNotFoundError(
            f"{PAGES_PARQUET} not found. Run crawl_pages.py first."
        )
    pages = pd.read_parquet(PAGES_PARQUET)
    # only embed successfully-fetched pages; others get no embedding row and
    # will be median/zero-filled at training time.
    pages = pages[pages["fetch_status"] == 200].reset_index(drop=True)
    print(f"pages to embed: {len(pages)}")

    # --- incremental: embed only resolved_urls not already in the parquet ---
    def _missing(path) -> pd.DataFrame:
        if not path.exists():
            return pages
        done = set(pd.read_parquet(path, columns=["resolved_url"])["resolved_url"])
        return pages[~pages["resolved_url"].isin(done)].reset_index(drop=True)

    def _append(path, new_df, label):
        if path.exists():
            old = pd.read_parquet(path)
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df.to_parquet(path, index=False)
        print(f"[{label}] saved {new_df.shape} -> {path.relative_to(REPO_ROOT)}")
        return new_df

    text_todo = _missing(TEXT_EMB_PATH)
    if len(text_todo) == 0:
        print("[text] all pages already embedded (skip)")
    else:
        print(f"[text] embedding {len(text_todo)} new pages")
        _append(TEXT_EMB_PATH, build_text_embeddings(text_todo), "text")

    image_todo = _missing(IMAGE_EMB_PATH)
    if len(image_todo) == 0:
        print("[image] all pages already embedded (skip)")
    else:
        print(f"[image] embedding {len(image_todo)} new pages")
        image_df = _append(IMAGE_EMB_PATH, build_image_embeddings(image_todo), "image")
        n_with_img = int((image_df["img_missing"] == 0).sum())
        print(f"[image] pages with >=1 usable image: {n_with_img}/{len(image_df)}")


if __name__ == "__main__":
    main()
