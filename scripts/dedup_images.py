"""дедупликация мемов по perceptual hash
сравнивает все картинки кросс-источников
помечает дубликаты is_duplicate=True в vqa_annotations_v3.jsonl
оригинал - первый по порядку, остальные помечаются
"""

import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import imagehash

VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
PHASH_THRESHOLD = 0 # дубль если все биты совпалают
HASH_SIZE = 16 # сжимаем в 16 на 16 для того чтобы можно было распохнать текст


def compute_phash(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        return imagehash.phash(img, hash_size=HASH_SIZE)
    except Exception:
        return None


records = []
with open(VQA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

hashes = []
for r in tqdm(records, desc="phash"):
    img_path = Path(r.get("source_path", ""))
    if img_path.exists():
        h = compute_phash(img_path)
    else:
        h = None
    hashes.append(h)

duplicates = set()
originals = []

for i in tqdm(range(len(records)), desc="dedup"):
    if hashes[i] is None:
        continue
    if records[i].get("is_nsfw"):
        continue

    is_dup = False
    for orig_idx, orig_hash in originals:
        if abs(hashes[i] - orig_hash) <= PHASH_THRESHOLD:
            duplicates.add(i)
            is_dup = True
            break

    if not is_dup:
        originals.append((i, hashes[i]))

for i, r in enumerate(records):
    if i in duplicates:
        r["is_duplicate"] = True
    else:
        r["is_duplicate"] = False

with open(VQA_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

clean = 0
for r in records:
    if not r.get("is_nsfw") and not r.get("is_duplicate"):
        clean += 1

print(f"дубликатов {len(duplicates)} уникальных {len(originals)} чистых {clean}")
