
"""
  Создаёт эмбеддинги и FAISS индекс по VQA-ответам.
  VQA-ответы склеиваются в один текст для каждого мема.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Пути
VQA_FILE = Path("data/processed/vqa_annotations_v2.jsonl")
VQA_BACKUP = Path("data/processed/vqa_annotations_v2_backup.jsonl")
OUTPUT_EMB = Path("data/processed/emb_vqa.npy")
OUTPUT_INDEX = Path("data/processed/faiss_vqa.index")

MODEL_NAME = "BAAI/bge-m3"


def extract_vqa_text(record):
    """Извлекает vqa ответы и склеивает в один текст"""
    vqa = record.get("vqa", [])

    if not vqa:
        return ""

    answers = []

    # VQA может быть списком словарей или списком строк
    if isinstance(vqa, list):
        for item in vqa:
            if isinstance(item, dict):
                # {"q": "...", "a": "..."} или {"question": "...", "answer": "..."}
                ans = item.get("a") or item.get("answer") or item.get("A") or ""
                if ans:
                    answers.append(str(ans))
            elif isinstance(item, str):
                answers.append(item)
    elif isinstance(vqa, dict):
        # {"Q1": "...", "Q2": "...", ...}
        for key, val in vqa.items():
            if val:
                answers.append(str(val))

    return " ".join(answers)


def main():
    # Выбираем файл (основной или backup)
    if VQA_FILE.exists() and VQA_FILE.stat().st_size > 100000:
        src = VQA_FILE
    elif VQA_BACKUP.exists():
        src = VQA_BACKUP
        print(f"Используем backup: {VQA_BACKUP}")
    else:
        print("ERROR: VQA файл не найден")
        return

    
    
    with open(src) as f:
        records = [json.loads(line) for line in f]
    print(f"Записей: {len(records)}")

    # Извлекаем VQA тексты
    vqa_texts = []
    empty_count = 0
    for r in records:
        text = extract_vqa_text(r)
        if not text:
            # Fallback на caption + main_idea
            text = f"{r.get('caption', '')} {r.get('main_idea', '')}".strip()
            empty_count += 1
        vqa_texts.append(text)

    print(f"С VQA: {len(records) - empty_count}, без VQA (fallback): {empty_count}")
    print(f"Пример VQA текста: {vqa_texts[0][:200]}")

    # Загружаем модель
    model = SentenceTransformer(MODEL_NAME)

    # Кодируем
    embeddings = model.encode(
        vqa_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"Shape: {embeddings.shape}")

    # Сохраняем эмбеддинги
    np.save(OUTPUT_EMB, embeddings.astype(np.float32))
    print(f"Сохранено: {OUTPUT_EMB}")

    # Создаём FAISS индекс
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(OUTPUT_INDEX))
    print(f"Сохранено: {OUTPUT_INDEX}")



if __name__ == "__main__":
    main()