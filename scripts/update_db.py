"""
Автообновление БД мемов.
1. Сканирует data/raw/new/ на новые картинки
2. Аннотирует через GPT (10 полей)
3. Переводит search_queries на RU
4. Encode через bge-m3
5. Добавляет в FAISS индексы
6. Перемещает обработанные в data/raw/processed_new/

Запуск: OMP_NUM_THREADS=1 .venv/bin/python scripts/update_db.py
"""

import json
import os
import re
import shutil
import base64
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── конфиг ──
NEW_DIR = Path("data/raw/new")
DONE_DIR = Path("data/raw/processed_new")
VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
EXP_DIR = Path("data/experiments")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://ai.redivo.ru/v1"
)

PROMPT = """Analyze this meme. Return JSON with these fields:

1. "caption" - literal visual description (2-3 sentences)
2. "ocr_text" - exact text from image, empty string if none
3. "meme_template" - template name if recognized, else ""
4. "objects" - list of 5-10 key visual elements
5. "tone" - one of: "ironic", "wholesome", "absurd", "dark", "relatable", "aggressive", "neutral"
6. "main_idea" - the joke or point (1 sentence)
7. "search_queries" - 3-5 short search phrases
8. "tags" - 5-10 keyword tags
9. "emotions" - list of 2-4 emotions
10. "vqa" - list of 3 {"q": "...", "a": "..."} pairs

Return ONLY valid JSON, no markdown."""


def encode_image(path, max_size=512):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def annotate_meme(img_path):
    b64 = encode_image(img_path)
    resp = client.chat.completions.create(
        model="gpt-5.4-mini-fast",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        temperature=0.2,
        max_tokens=800,
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def translate_queries(queries_en):
    if not queries_en:
        return []
    prompt = f"""Translate these English meme search queries to Russian.
Keep them short (3-7 words each). Return ONLY the Russian queries, one per line.

{chr(10).join(queries_en)}"""

    resp = client.chat.completions.create(
        model="gpt-5.4-mini-fast",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    text = resp.choices[0].message.content.strip()
    return [q.strip().strip('"\'') for q in text.split("\n") if q.strip()]


def main():
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    # найти новые картинки
    new_images = [f for f in NEW_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    if not new_images:
        print("Нет новых мемов в data/raw/new/")
        return

    print(f"Найдено {len(new_images)} новых мемов")

    # загрузка существующих данных
    records = []
    with open(VQA_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    existing_files = {r["filename"] for r in records}

    # загрузка модели
    print("Загрузка bge-m3...")
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")

    # загрузка индексов
    idx_all = faiss.read_index(str(EXP_DIR / "faiss_C_all.index"))
    idx_sq_en = faiss.read_index(str(EXP_DIR / "faiss_D_search_queries.index"))
    idx_sq_ru = faiss.read_index(str(EXP_DIR / "faiss_ru_search_queries.index"))

    added = 0
    for img_path in new_images:
        if img_path.name in existing_files:
            print(f"  [пропуск] {img_path.name} уже в БД")
            shutil.move(str(img_path), str(DONE_DIR / img_path.name))
            continue

        print(f"  [{added+1}/{len(new_images)}] {img_path.name}")

        try:
            # 1. аннотация
            ann = annotate_meme(img_path)
            ann["filename"] = img_path.name
            ann["source_path"] = str(DONE_DIR / img_path.name)
            ann["source"] = "user_upload"
            ann["is_nsfw"] = False

            # 2. перевод search_queries
            ann["search_queries_ru"] = translate_queries(ann.get("search_queries", []))

            # 3. encode и добавление в индексы
            # C (мега-текст)
            all_text = ". ".join([
                ann.get("caption", ""),
                ann.get("main_idea", ""),
                ann.get("ocr_text", ""),
                " ".join(ann.get("objects", [])),
                ann.get("tone", ""),
                ann.get("meme_template", ""),
                " ".join(ann.get("search_queries", [])),
                " ".join(ann.get("tags", [])),
                " ".join(ann.get("emotions", [])),
            ])
            emb_all = model.encode(all_text, normalize_embeddings=True).reshape(1, -1).astype(np.float32)
            idx_all.add(emb_all)

            # EN search_queries
            sq_en = ", ".join(ann.get("search_queries", []))
            emb_sq_en = model.encode(sq_en if sq_en else "empty", normalize_embeddings=True).reshape(1, -1).astype(np.float32)
            idx_sq_en.add(emb_sq_en)

            # RU search_queries
            sq_ru = ", ".join(ann.get("search_queries_ru", []))
            emb_sq_ru = model.encode(sq_ru if sq_ru else "empty", normalize_embeddings=True).reshape(1, -1).astype(np.float32)
            idx_sq_ru.add(emb_sq_ru)

            # 4. сохранить запись
            records.append(ann)
            added += 1

            # 5. переместить файл
            shutil.move(str(img_path), str(DONE_DIR / img_path.name))

        except Exception as e:
            print(f"    ошибка: {e}")

    if added == 0:
        print("Ничего не добавлено")
        return

    # сохранить обновлённые данные
    with open(VQA_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    faiss.write_index(idx_all, str(EXP_DIR / "faiss_C_all.index"))
    faiss.write_index(idx_sq_en, str(EXP_DIR / "faiss_D_search_queries.index"))
    faiss.write_index(idx_sq_ru, str(EXP_DIR / "faiss_ru_search_queries.index"))

    print(f"\n✅ Добавлено {added} мемов. Всего: {len(records)}")
    print("Индексы обновлены. Перезапусти бота для применения.")


if __name__ == "__main__":
    main()
