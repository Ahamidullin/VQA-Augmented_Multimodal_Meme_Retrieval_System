"""Просмотр ПАР дубликатов — открывает дубль и оригинал рядом"""
import json, subprocess, imagehash
from pathlib import Path
from PIL import Image

VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
HASH_SIZE = 16

with open(VQA_FILE) as f:
    records = [json.loads(l) for l in f if l.strip()]

# пересчитать хеши

hashes = []
for r in records:
    p = Path(r.get("source_path", ""))
    try:
        h = imagehash.phash(Image.open(p).convert("RGB"), hash_size=HASH_SIZE)
    except:
        h = None
    hashes.append(h)

# найти пары
originals = []  # (index, hash)
pairs = []      # (dup_index, orig_index, distance)

for i in range(len(records)):
    if hashes[i] is None:
        continue
    found = False
    for orig_idx, orig_hash in originals:
        dist = abs(hashes[i] - orig_hash)
        if dist == 0:
            pairs.append((i, orig_idx, dist))
            found = True
            break
    if not found:
        originals.append((i, hashes[i]))

print(f"Найдено пар: {len(pairs)}")

# показать и открыть первые N пар
N = 10
for pair_num, (dup_i, orig_i, dist) in enumerate(pairs[:N]):
    dup = records[dup_i]
    orig = records[orig_i]
    print(f"\nПара {pair_num+1} (distance={dist})")
    print(f"  дубль:    {dup['filename']} — {dup.get('caption','')[:60]}")
    print(f"  оригинал: {orig['filename']} — {orig.get('caption','')[:60]}")
    
    # открыть обе картинки
    paths = []
    for r in [dup, orig]:
        p = Path(r.get("source_path", ""))
        if p.exists():
            paths.append(str(p))
    if paths:
        subprocess.run(["open"] + paths)
    
    
