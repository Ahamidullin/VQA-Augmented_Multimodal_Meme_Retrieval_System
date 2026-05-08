"""
Генерация русских search_queries для каждого мема.
Берёт EN search_queries из vqa_annotations_v3.jsonl,
переводит через OpenAI API, сохраняет в vqa_annotations_v3.jsonl (поле search_queries_ru).
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

# загрузка
records = []
with open(VQA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# считаем сколько уже переведено
already = sum(1 for r in records if r.get("search_queries_ru"))
print(f"Уже переведено: {already}, осталось: {len(records) - already}")

lock = threading.Lock()
updated = 0


def translate_queries(queries_en):
    """Переводит список EN search_queries на RU"""
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
    except Exception as e:
        r["search_queries_ru"] = []
        return None


# параллельный перевод
to_process = [i for i, r in enumerate(records) if not r.get("search_queries_ru") and r.get("search_queries")]
print(f"К обработке: {len(to_process)}")

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(process, i): i for i in to_process}
    for f in tqdm(as_completed(futures), total=len(futures), desc="Перевод"):
        result = f.result()
        if result is not None:
            updated += 1
        # сохранение каждые 500
        if updated % 500 == 0 and updated > 0:
            with lock:
                with open(VQA_FILE, "w", encoding="utf-8") as out:
                    for rec in records:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

# финальное сохранение
with open(VQA_FILE, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\nПереведено {updated} записей")
print(f"Пример: {records[0].get('search_queries_ru', [])}")
