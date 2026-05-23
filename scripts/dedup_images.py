"""
dedup_images.py

Дедупликация мемов по perceptual hash 
Сравнивает все картинки кросс источников
Помечает дубликаты: is_duplicate=True в vqa_annotations_v3.jsonl
Оригинал - первый по порядку остаётся, дубли помечаются

"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import imagehash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
PHASH_THRESHOLD = 0  # hamming distance <= 0 = дубль
HASH_SIZE = 16       # 16x16=256 бит (default 8x8=64 бит, слишком грубо для мемов)


def compute_phash(img_path):
    """Считает phash картинки, возвращает None если не удалось"""
    try:
        img = Image.open(img_path).convert("RGB")
        return imagehash.phash(img, hash_size=HASH_SIZE)
    except Exception:
        return None



# загрузка записей
with open(VQA_FILE) as f:
    records = [json.loads(line) for line in f if line.strip()]

log.info(f"Всего записей: {len(records)}")

# считаем phash для каждой картинки
hashes = []
for r in tqdm(records, desc="Computing phash"):
    img_path = Path(r.get("source_path", ""))
    if img_path.exists():
        h = compute_phash(img_path)
    else:
        h = None
    hashes.append(h)

no_hash = sum(1 for h in hashes if h is None)
log.info(f"Хеши посчитаны. Без хеша (файл не найден): {no_hash}")

# поиск дубликатов
# для каждой картинки проверяем все предыдущие
# первая встреченная = оригинал, остальные = дубли
duplicates = set()
originals = []  # список (index, hash) оригиналов

for i in tqdm(range(len(records)), desc="Finding duplicates"):
    if hashes[i] is None:
        continue
    if records[i].get("is_nsfw"):
        continue  # NSFW уже отфильтрованы

    is_dup = False
    for orig_idx, orig_hash in originals:
        if abs(hashes[i] - orig_hash) <= PHASH_THRESHOLD:
            duplicates.add(i)
            is_dup = True
            break

    if not is_dup:
        originals.append((i, hashes[i]))

log.info(f"Найдено дубликатов: {len(duplicates)}")
log.info(f"Уникальных: {len(originals)}")

# примеры дублей
shown = 0
for i in range(len(records)):
    if i in duplicates and shown < 30:
        # найти оригинал
        for orig_idx, orig_hash in originals:
            if abs(hashes[i] - orig_hash) <= PHASH_THRESHOLD:
                log.info(
                    f"Дубль: {records[i]['filename']} "
                    f"оригинал: {records[orig_idx]['filename']} "
                    f"(distance={abs(hashes[i] - orig_hash)})"
                )
                shown += 1
                break

# помечаем
for i, r in enumerate(records):
    if i in duplicates:
        r["is_duplicate"] = True
    else:
        r["is_duplicate"] = False

# сохраняем
with open(VQA_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

log.info(f"Сохранено. Дубликаты помечены is_duplicate=True.")
log.info(f"Чистых (не NSFW, не дубль): "
            f"{sum(1 for r in records if not r.get('is_nsfw') and not r.get('is_duplicate'))}")



