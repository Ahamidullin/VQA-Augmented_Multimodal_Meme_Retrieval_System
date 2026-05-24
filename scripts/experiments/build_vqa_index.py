
"""
создаем эмбеддинги и faiss индекс по vqa ответам
vqa ответы склеиваются в один текст для каждого мема
"""

import json
import sys
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


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

    if isinstance(vqa, list):
        for item in vqa:
            if isinstance(item, dict):
                ans = item.get("a") or item.get("answer") or item.get("A") or ""
                if ans:
                    answers.append(str(ans))
            elif isinstance(item, str):
                answers.append(item)
    elif isinstance(vqa, dict):
        for key, val in vqa.items():
            if val:
                answers.append(str(val))

    return " ".join(answers)



if VQA_FILE.exists() and VQA_FILE.stat().st_size > 100000:
    src = VQA_FILE
elif VQA_BACKUP.exists():
    src = VQA_BACKUP
else:
    sys.exit(0)



with open(src) as f:
    records = [json.loads(line) for line in f]


vqa_texts = []
empty_count = 0
for r in records:
    text = extract_vqa_text(r)
    if not text:
        text = f"{r.get('caption', '')} {r.get('main_idea', '')}".strip()
        empty_count += 1
    vqa_texts.append(text)

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    vqa_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save(OUTPUT_EMB, embeddings.astype(np.float32))

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))  # type: ignore
faiss.write_index(index, str(OUTPUT_INDEX))
print(f"save {OUTPUT_INDEX}")



