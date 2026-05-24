"""перевод search_queries на русский через gpt-5.4-mini-fast
берет en search_queries из vqa_annotations_v3.jsonl
переводит каждый мем параллельно (10 воркеров)
сохраняет результат в поле search_queries_ru
чекпоинт каждые 500 переводов, поддерживает resume
"""

import json
import os
import threading
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
WORKERS = 10

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://ai.redivo.ru/v1"
)

records = []
with open(VQA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

lock = threading.Lock()
updated = 0


def translate_queries(queries_en):
    prompt = f"""Translate these English meme search queries to Russian.
Keep them short (3-7 words each). Return ONLY the Russian queries, one per line.

{chr(10).join(queries_en)}"""

    resp = client.chat.completions.create(
        model="gpt-5.4-mini-fast",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    text = (resp.choices[0].message.content or "").strip()
    result = []
    for q in text.split("\n"):
        q = q.strip().strip("\"'")
        if q:
            result.append(q)
    return result


def process(idx):
    r = records[idx]
    if r.get("search_queries_ru"):
        return None
    queries_en = r.get("search_queries", [])
    if not queries_en:
        r["search_queries_ru"] = []
        return None
    try:
        r["search_queries_ru"] = translate_queries(queries_en)
        return idx
    except Exception:
        r["search_queries_ru"] = []
        return None


to_process = []
for i, r in enumerate(records):
    if not r.get("search_queries_ru") and r.get("search_queries"):
        to_process.append(i)

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {}
    for i in to_process:
        fut = pool.submit(process, i)
        futures[fut] = i
    for f in tqdm(as_completed(futures), total=len(futures), desc="translate"):
        result = f.result()
        if result is not None:
            updated += 1
        if updated % 500 == 0 and updated > 0:
            with lock:
                with open(VQA_FILE, "w", encoding="utf-8") as out:
                    for rec in records:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

with open(VQA_FILE, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"переведено {updated}")
