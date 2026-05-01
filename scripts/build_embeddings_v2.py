"""
build_embeddings_v2.py

Строит РАЗДЕЛЬНЫЕ эмбеддинги для каждого типа данных:
  - emb_ocr.npy         — OCR-текст с картинки
  - emb_caption.npy     — caption + main_idea (семантическое описание)
  - emb_keywords.npy    — objects + tone (ключевые слова)
  - emb_image.npy       — CLIP-эмбеддинг картинки (не пересчитываем если есть)

+ FAISS-индексы для каждого
+ обновлённые метаданные
"""

import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Конфиг ---
VQA_FILE = Path("data/processed/vqa_annotations_v2.jsonl")  # v2 — полные аннотации
OUTPUT_DIR = Path("data/processed")

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def load_vqa_data(path):
    """загружает vqa аннотации"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    log.info(f"загружено {len(records)} записей из {path}")
    return records


# ──────────────────────────────────────────────
# Подготовка текстов для каждого типа эмбеддинга
# ──────────────────────────────────────────────

def extract_ocr_texts(records):
    """OCR-текст с картинки (как есть, без обработки)"""
    texts = []
    for r in records:
        ocr = r.get("ocr_text", "").strip()
        # Если есть нормализованная версия из VQA — берём её
        ocr_norm = r.get("ocr_normalized", "").strip()
        text = ocr_norm if ocr_norm else ocr
        texts.append(text if len(text) > 2 else "")
    non_empty = sum(1 for t in texts if t)
    log.info(f"OCR тексты: {non_empty}/{len(texts)} непустых")
    return texts


def extract_caption_texts(records):
    """Caption + main_idea — семантическое описание мема"""
    texts = []
    for r in records:
        parts = []
        caption = r.get("caption", "")
        if caption:
            parts.append(caption)

        idea = r.get("main_idea", "")
        # Фильтруем шаблонные ответы (когда модель вернула промпт)
        if idea and "one sentence" not in idea.lower():
            parts.append(idea)

        text = " ".join(parts).strip()
        if not text:
            # fallback на raw_response
            text = r.get("raw_response", r.get("filename", ""))[:200]
        texts.append(text)

    non_empty = sum(1 for t in texts if t)
    log.info(f"Caption тексты: {non_empty}/{len(texts)} непустых")
    return texts


def extract_keyword_texts(records):
    """Objects + tone — ключевые слова для поиска"""
    texts = []
    for r in records:
        parts = []

        # objects из v1
        objects = r.get("objects", [])
        if isinstance(objects, list) and objects:
            # Фильтруем шаблонные (когда модель вернула промпт)
            clean = [str(o) for o in objects if str(o) not in ("key", "objects", "max 5")]
            if clean:
                parts.append(", ".join(clean[:8]))

        # objects_detailed из v2 (более подробные)
        objects_det = r.get("objects_detailed", [])
        if isinstance(objects_det, list) and objects_det:
            # Берём уникальные, не повторяя v1
            existing = set(str(o).lower() for o in objects)
            new_objs = [str(o) for o in objects_det
                        if str(o).lower() not in existing][:5]
            if new_objs:
                parts.append(", ".join(new_objs))

        tone = r.get("tone", "")
        if tone:
            # Берём первый тон, если модель вернула весь список
            tone_clean = tone.split("/")[0].strip() if "/" in tone and len(tone) > 20 else tone
            parts.append(tone_clean)

        texts.append(". ".join(parts) if parts else "")

    non_empty = sum(1 for t in texts if t)
    log.info(f"Keyword тексты: {non_empty}/{len(texts)} непустых")
    return texts


# ──────────────────────────────────────────────
# Генерация эмбеддингов
# ──────────────────────────────────────────────

def generate_text_embeddings(texts, model):
    """текстовые эмбеддинги через sentence-transformers"""
    # Пустые строки заменяем на placeholder (чтобы модель не падала)
    # Результат для пустых потом зануляем
    empty_mask = [t.strip() == "" for t in texts]
    safe_texts = [t if t.strip() else "empty" for t in texts]

    log.info(f"генерация эмбеддингов для {len(safe_texts)} записей...")
    embeddings = model.encode(
        safe_texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    # Зануляем эмбеддинги для пустых текстов
    for i, is_empty in enumerate(empty_mask):
        if is_empty:
            embeddings[i] = np.zeros(embeddings.shape[1])

    return embeddings.astype(np.float32)


def generate_image_embeddings(records):
    """картиночные эмбеддинги через CLIP"""
    # Если файл уже существует, не пересчитываем (экономим ~20 мин)
    existing_path = OUTPUT_DIR / "image_embeddings.npy"
    if existing_path.exists():
        emb = np.load(existing_path)
        if len(emb) == len(records):
            log.info(f"image_embeddings.npy уже существует ({emb.shape}), пропускаем CLIP")
            return emb

    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    log.info(f"загрузка модели {CLIP_MODEL_NAME}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    embeddings = []
    valid_count = 0
    errors = 0

    for record in tqdm(records, desc="CLIP embeddings"):
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
    log.info(f"картиночные эмбеддинги: {result.shape}, валидных: {valid_count}, ошибок: {errors}")
    return result


def build_faiss_index(embeddings, index_path):
    """строит faiss индекс (inner product для нормализованных векторов)"""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    log.info(f"FAISS: {index_path} ({index.ntotal} векторов, dim={dim})")
    return index


def save_metadata(records, path):
    """метаданные индекса — расширенные"""
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
    log.info(f"метаданные: {path} ({len(records)} записей)")


def main():
    log.info("=" * 60)
    log.info("build_embeddings_v2: раздельные эмбеддинги")
    log.info("=" * 60)

    # Загрузка данных (v2 — полные аннотации)
    records = load_vqa_data(VQA_FILE)

    # Извлечение текстов для каждого типа
    ocr_texts = extract_ocr_texts(records)
    caption_texts = extract_caption_texts(records)
    keyword_texts = extract_keyword_texts(records)

    # Примеры
    for i in range(min(3, len(records))):
        log.info(f"--- пример {i+1} ({records[i].get('filename', '')}) ---")
        log.info(f"  OCR:      {ocr_texts[i][:80]}")
        log.info(f"  Caption:  {caption_texts[i][:80]}")
        log.info(f"  Keywords: {keyword_texts[i][:80]}")

    # Загрузка текстовой модели (одна на все текстовые эмбеддинги)
    from sentence_transformers import SentenceTransformer
    log.info(f"загрузка модели {TEXT_MODEL_NAME}...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME)

    # 1. OCR эмбеддинги
    log.info("── emb_ocr ──")
    emb_ocr = generate_text_embeddings(ocr_texts, text_model)
    np.save(OUTPUT_DIR / "emb_ocr.npy", emb_ocr)
    log.info(f"сохранено: emb_ocr.npy {emb_ocr.shape}")

    # 2. Caption эмбеддинги
    log.info("── emb_caption ──")
    emb_caption = generate_text_embeddings(caption_texts, text_model)
    np.save(OUTPUT_DIR / "emb_caption.npy", emb_caption)
    log.info(f"сохранено: emb_caption.npy {emb_caption.shape}")

    # 3. Keywords эмбеддинги
    log.info("── emb_keywords ──")
    emb_keywords = generate_text_embeddings(keyword_texts, text_model)
    np.save(OUTPUT_DIR / "emb_keywords.npy", emb_keywords)
    log.info(f"сохранено: emb_keywords.npy {emb_keywords.shape}")

    # 4. Image эмбеддинги (переиспользуем если есть)
    log.info("── emb_image ──")
    emb_image = generate_image_embeddings(records)
    np.save(OUTPUT_DIR / "emb_image.npy", emb_image)
    log.info(f"сохранено: emb_image.npy {emb_image.shape}")

    # FAISS индексы
    log.info("── FAISS индексы ──")
    build_faiss_index(emb_ocr, OUTPUT_DIR / "faiss_ocr.index")
    build_faiss_index(emb_caption, OUTPUT_DIR / "faiss_caption.index")
    build_faiss_index(emb_keywords, OUTPUT_DIR / "faiss_keywords.index")
    build_faiss_index(emb_image, OUTPUT_DIR / "faiss_image.index")

    # Метаданные
    save_metadata(records, OUTPUT_DIR / "index_metadata.jsonl")

    log.info("=" * 60)
    log.info("ГОТОВО! Файлы:")
    log.info(f"  emb_ocr.npy       {emb_ocr.shape}")
    log.info(f"  emb_caption.npy   {emb_caption.shape}")
    log.info(f"  emb_keywords.npy  {emb_keywords.shape}")
    log.info(f"  emb_image.npy     {emb_image.shape}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
