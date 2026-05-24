"""старая версия: раздельные эмбеддинги через MiniLM + CLIP
строит 4 индекса: ocr, caption, keywords, image
заменен на bge-m3 с 3 индексами (C_all, D_search_queries, ru_search_queries)
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

VQA_FILE = Path("data/processed/vqa_annotations_v2.jsonl")
OUTPUT_DIR = Path("data/processed")

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def load_vqa_data(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def extract_ocr_texts(records):
    texts = []
    for r in records:
        ocr = r.get("ocr_text", "").strip()
        ocr_norm = r.get("ocr_normalized", "").strip()
        text = ocr_norm if ocr_norm else ocr
        if len(text) > 2:
            texts.append(text)
        else:
            texts.append("")
    return texts


def extract_caption_texts(records):
    texts = []
    for r in records:
        parts = []
        caption = r.get("caption", "")
        if caption:
            parts.append(caption)

        idea = r.get("main_idea", "")
        if idea and "one sentence" not in idea.lower():
            parts.append(idea)

        text = " ".join(parts).strip()
        if not text:
            text = r.get("raw_response", r.get("filename", ""))[:200]
        texts.append(text)
    return texts


def extract_keyword_texts(records):
    texts = []
    for r in records:
        parts = []

        objects = r.get("objects", [])
        if isinstance(objects, list) and objects:
            clean = []
            for o in objects:
                s = str(o)
                if s not in ("key", "objects", "max 5"):
                    clean.append(s)
            if clean:
                parts.append(", ".join(clean[:8]))

        objects_det = r.get("objects_detailed", [])
        if isinstance(objects_det, list) and objects_det:
            existing = set()
            for o in objects:
                existing.add(str(o).lower())
            new_objs = []
            for o in objects_det:
                if str(o).lower() not in existing:
                    new_objs.append(str(o))
                if len(new_objs) >= 5:
                    break
            if new_objs:
                parts.append(", ".join(new_objs))

        tone = r.get("tone", "")
        if tone:
            if "/" in tone and len(tone) > 20:
                tone = tone.split("/")[0].strip()
            parts.append(tone)

        if parts:
            texts.append(". ".join(parts))
        else:
            texts.append("")
    return texts


def generate_text_embeddings(texts, model):
    empty_mask = []
    safe_texts = []
    for t in texts:
        is_empty = t.strip() == ""
        empty_mask.append(is_empty)
        safe_texts.append(t if not is_empty else "empty")

    embeddings = model.encode(
        safe_texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    for i, is_empty in enumerate(empty_mask):
        if is_empty:
            embeddings[i] = np.zeros(embeddings.shape[1])

    return embeddings.astype(np.float32)


def generate_image_embeddings(records):
    existing_path = OUTPUT_DIR / "image_embeddings.npy"
    if existing_path.exists():
        emb = np.load(existing_path)
        if len(emb) == len(records):
            return emb

    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    embeddings = []
    errors = 0

    for record in tqdm(records, desc="clip"):
        img_path = Path(record.get("source_path", ""))
        if not img_path.exists():
            embeddings.append(np.zeros(512, dtype=np.float32))
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")  # type: ignore

            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                img_features = model.visual_projection(vision_out.pooler_output)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            embeddings.append(img_features.numpy().flatten().astype(np.float32))
        except Exception:
            embeddings.append(np.zeros(512, dtype=np.float32))
            errors += 1

    return np.stack(embeddings)


def build_faiss_index(embeddings, index_path):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # type: ignore

    faiss.write_index(index, str(index_path))
    return index


def save_metadata(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            meta = {
                "filename": r.get("filename", ""),
                "source_path": r.get("source_path", ""),
                "caption": r.get("caption", ""),
                "ocr_text": r.get("ocr_text", "")[:200],
                "objects": r.get("objects", []),
                "tone": r.get("tone", ""),
                "main_idea": r.get("main_idea", ""),
                "source_type": r.get("source_type", ""),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


records = load_vqa_data(VQA_FILE)

ocr_texts = extract_ocr_texts(records)
caption_texts = extract_caption_texts(records)
keyword_texts = extract_keyword_texts(records)

from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer(TEXT_MODEL_NAME)

emb_ocr = generate_text_embeddings(ocr_texts, text_model)
np.save(OUTPUT_DIR / "emb_ocr.npy", emb_ocr)

emb_caption = generate_text_embeddings(caption_texts, text_model)
np.save(OUTPUT_DIR / "emb_caption.npy", emb_caption)

emb_keywords = generate_text_embeddings(keyword_texts, text_model)
np.save(OUTPUT_DIR / "emb_keywords.npy", emb_keywords)

emb_image = generate_image_embeddings(records)
np.save(OUTPUT_DIR / "emb_image.npy", emb_image)

build_faiss_index(emb_ocr, OUTPUT_DIR / "faiss_ocr.index")
build_faiss_index(emb_caption, OUTPUT_DIR / "faiss_caption.index")
build_faiss_index(emb_keywords, OUTPUT_DIR / "faiss_keywords.index")
build_faiss_index(emb_image, OUTPUT_DIR / "faiss_image.index")

save_metadata(records, OUTPUT_DIR / "index_metadata.jsonl")

print(f"готово ocr={emb_ocr.shape} caption={emb_caption.shape} kw={emb_keywords.shape} img={emb_image.shape}")
