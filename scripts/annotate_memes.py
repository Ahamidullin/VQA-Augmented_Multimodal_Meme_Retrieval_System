"""
аннотация мемов через gpt-5.4-mini-fast
один проход: описание + метаданные + vqa
вход: data/processed/final_dataset_text.csv
выход: data/processed/vqa_annotations_v3.jsonl
поддерживает resume
"""

import os
import csv
import json
import time
import base64
import sys
import threading
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

INPUT_CSV = Path("data/processed/final_dataset_text.csv")
OUTPUT_JSONL = Path("data/processed/vqa_annotations_v3.jsonl")
MODEL = "gpt-5.4-mini-fast"
MAX_IMAGE_SIZE = 512

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://ai.redivo.ru/v1"
)

PROMPT = """Analyze this meme. Return JSON with these fields:

1. "caption" - literal visual description: who/what is shown, setting, expressions (2-3 sentences, no interpretation)
2. "ocr_text" - exact text from the image, preserve original language and punctuation. Empty string "" if no text.
3. "meme_template" - known template name if recognized (e.g. "Drake", "Distracted Boyfriend", "Two Buttons"), else ""
4. "objects" - list of 5-10 key visual elements: ["person", "cat", "phone", ...]
5. "tone" - one of: "ironic", "wholesome", "absurd", "dark", "relatable", "aggressive", "neutral"
6. "main_idea" - the joke or point being made, why it's funny (1 sentence, interpretation allowed)
7. "search_queries" - 3-5 short phrases someone would type to find this meme
8. "tags" - 5-10 keyword tags for search
9. "emotions" - list of emotions conveyed: ["surprise", "anger", ...]
10. "vqa" - answer these 6 questions:
    Q1: What text is on the meme?
    Q2: What is literally depicted?
    Q3: What is happening?
    Q4: What is the joke or message?
    Q5: Who is the target audience?
    Q6: One sentence describing this meme for search

Return JSON only. No markdown, no explanation."""


def image_to_base64_resized(image_path, max_size=512):
    """читаем как байты и переводим в base64 (текст)"""
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')

    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)  # type: ignore

    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)

    base64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_string}"


def query_gpt(image_path, ocr_text="", num_attempt=3):
    """отправляет картинку в gpt api, возвращает сырой текст ответа"""

    img_url = image_to_base64_resized(image_path, MAX_IMAGE_SIZE)
    current_promt = PROMPT
    if ocr_text and ocr_text.strip():
        current_promt += f'\nOCR text detected externally: "{ocr_text.strip()[:200]}"'

    for attempt in range(num_attempt):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": current_promt},
                        {"type": "image_url", "image_url": {
                            "url": img_url,
                            "detail": "low"
                        }}
                    ]
                }],
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e):
                sys.exit(1)
            if attempt < num_attempt - 1:
                time.sleep(2)
    return None


def parse_json_response(text):
    """делаем json из ответа модели"""
    if not text:
        return None
    text = text.strip()

    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]

    text = text.strip()

    # находим json 
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None


def load_already_processed(output_path):
    """загружает имена уже обработанных файлов для resume"""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj.get("filename", ""))
            except json.JSONDecodeError:
                continue
    return done


memes = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    for meme in csv.DictReader(f):
        if Path(meme["source_path"]).exists():
            memes.append(meme)

done = load_already_processed(OUTPUT_JSONL)

remaining = []
for meme in memes:
    if meme["filename"] not in done:
        remaining.append(meme)

if not remaining:
    sys.exit(0)

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
stats = {"success": 0, "failed": 0, "skipped": 0}
lock = threading.Lock()
WORKERS = 5


def process_meme(meme):
    gpt_response = query_gpt(meme["source_path"], meme.get("ocr_text", ""))
    if gpt_response is None:
        return None, "skipped"
    parsed = parse_json_response(gpt_response)
    if parsed is None:
        return None, "failed"
    result = {
        "filename": meme["filename"],
        "source_path": meme["source_path"],
        "source_type": meme.get("source_type", ""),
        "confidence": float(meme.get("confidence", 0)),
        **parsed
    }
    return result, "success"


with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {}
        for meme in remaining:
            fut = executor.submit(process_meme, meme)
            futures[fut] = meme
        for future in tqdm(as_completed(futures), total=len(remaining), desc="annotate"):
            result, status = future.result()
            with lock:
                stats[status] += 1
                if result is not None:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

print(f"готово {stats}")
