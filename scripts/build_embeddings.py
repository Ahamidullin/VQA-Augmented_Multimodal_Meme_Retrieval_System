"""
build_embeddings.py

Объединяет OCR + VQA аннотации, генерирует текстовые и картиночные
эмбеддинги, строит FAISS индексы для мультимодального поиска мемов.
"""

import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Конфигурация
VQA_FILE = Path("data/processed/vqa_annotations.jsonl")
OUTPUT_DIR = Path("data/processed")

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

MAX_IMAGE_SIZE = 224  # CLIP input size


def load_vqa_data(path):
    """Загружает VQA аннотации."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    log.info(f"Загружено {len(records)} записей из {path}")
    return records


def build_combined_text(record):
    """Объединяет VQA + OCR в единый текст для эмбеддинга."""
    parts = []

    caption = record.get("caption", "")
    if caption:
        parts.append(caption)

    ocr = record.get("ocr_text", "").strip()
    if ocr and len(ocr) > 2:
        parts.append(f"Text on image: {ocr[:300]}")

    objects = record.get("objects", [])
    tone = record.get("tone", "")
    meta_parts = []
    if isinstance(objects, list) and objects:
        meta_parts.append(f"Objects: {', '.join(str(o) for o in objects[:8])}")
    if tone:
        # Берём только первый тон, если модель вернула несколько через /
        tone_clean = tone.split("/")[0].strip() if "/" in tone and len(tone) > 20 else tone
        meta_parts.append(f"Tone: {tone_clean}")
    if meta_parts:
        parts.append(". ".join(meta_parts))

    idea = record.get("main_idea", "")
    if idea:
        parts.append(idea)

    combined = "\n".join(parts)
    if not combined.strip():
        # Fallback: raw response или filename
        combined = record.get("raw_response", record.get("filename", "meme"))[:200]

    return combined


def generate_text_embeddings(texts):
    """Генерирует текстовые эмбеддинги через sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    log.info(f"Загрузка модели {TEXT_MODEL_NAME}...")
    model = SentenceTransformer(TEXT_MODEL_NAME)

    log.info(f"Генерация текстовых эмбеддингов для {len(texts)} записей...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    log.info(f"Текстовые эмбеддинги: {embeddings.shape}")
    return embeddings.astype(np.float32)


def generate_image_embeddings(records):
    """Генерирует картиночные эмбеддинги через CLIP."""
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    log.info(f"Загрузка модели {CLIP_MODEL_NAME}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    log.info("CLIP загружен (CPU)")

    embeddings = []
    valid_count = 0
    errors = 0

    for i, record in enumerate(tqdm(records, desc="CLIP embeddings")):
        img_path = Path(record.get("source_path", ""))
        if not img_path.exists():
            embeddings.append(np.zeros(512, dtype=np.float32))
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")

            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                img_features = model.visual_projection(vision_out.pooler_output)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            embeddings.append(img_features.numpy().flatten().astype(np.float32))
            valid_count += 1
        except Exception as e:
            embeddings.append(np.zeros(512, dtype=np.float32))
            errors += 1
            if errors <= 3:
                log.warning(f"CLIP error [{img_path.name}]: {e}")

    result = np.stack(embeddings)
    log.info(f"Картиночные эмбеддинги: {result.shape}, валидных: {valid_count}, ошибок: {errors}")
    return result


def build_faiss_index(embeddings, index_path):
    """Строит FAISS индекс (Inner Product для нормализованных векторов)."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    log.info(f"FAISS индекс сохранён: {index_path} ({index.ntotal} векторов, dim={dim})")
    return index


def save_metadata(records, path):
    """Сохраняет метаданные индекса (маппинг idx -> filename, caption)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            meta = {
                "filename": r.get("filename", ""),
                "source_path": r.get("source_path", ""),
                "caption": r.get("caption", ""),
                "ocr_text": r.get("ocr_text", "")[:200],
                "tone": r.get("tone", ""),
                "source_type": r.get("source_type", ""),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    log.info(f"Метаданные: {path} ({len(records)} записей)")


def main():
    log.info("Запуск построения эмбеддингов")

    # 1. Загрузка
    records = load_vqa_data(VQA_FILE)

    # 2. Объединение OCR + VQA в текст
    log.info("Формирование объединённых текстов...")
    texts = [build_combined_text(r) for r in records]

    # Показать примеры
    for i in range(min(3, len(texts))):
        log.info(f"Пример {i+1}: {texts[i][:120]}...")

    # 3. Текстовые эмбеддинги
    text_emb = generate_text_embeddings(texts)
    np.save(OUTPUT_DIR / "text_embeddings.npy", text_emb)
    log.info(f"Сохранено: text_embeddings.npy")

    # 4. Картиночные эмбеддинги (CLIP)
    image_emb = generate_image_embeddings(records)
    np.save(OUTPUT_DIR / "image_embeddings.npy", image_emb)
    log.info(f"Сохранено: image_embeddings.npy")

    # 5. FAISS индексы
    build_faiss_index(text_emb, OUTPUT_DIR / "faiss_text.index")
    build_faiss_index(image_emb, OUTPUT_DIR / "faiss_image.index")

    # 6. Метаданные
    save_metadata(records, OUTPUT_DIR / "index_metadata.jsonl")

    log.info("Готово!")


if __name__ == "__main__":
    main()
