"""vqa аннотации мемов через qwen3.5:9b (ollama)
один проход: описание + обогащение через локальную модель
вход: data/processed/final_dataset_text.csv
выход: data/processed/vqa_annotations_v2.jsonl
поддерживает resume, исторически предшествует gpt-аннотации
"""

import csv
import json
import sys
import time
import base64
import random
import requests
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

INPUT_CSV = Path("data/processed/final_dataset_text.csv")
OUTPUT_JSONL = Path("data/processed/vqa_annotations_v2.jsonl")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"
TARGET_COUNT = 10000
MAX_IMAGE_SIZE = 512

PROMPT = """Analyze this meme. Return JSON with these fields:

1. "caption" - literal visual description: who/what is shown, setting, expressions (1-2 sentences, no interpretation)
2. "ocr_text" - exact text from the image, preserve original language and punctuation. Empty string "" if no text.
3. "meme_template" - known template name if recognized (e.g. "Drake", "Distracted Boyfriend", "Two Buttons"), else ""
4. "objects" - list of key visual elements: ["person", "cat", "phone", ...]
5. "tone" - one of: "ironic", "wholesome", "absurd", "dark", "relatable", "aggressive", "neutral"
6. "main_idea" - the joke or point being made, why it's funny (1 sentence, interpretation allowed)
7. "ocr_normalized" - cleaned OCR: fix repeated chars (AAAA->A), remove noise. Empty string if no text.
8. "vqa" - answer these 6 questions:
   Q1: What text is on the meme?
   Q2: What is literally depicted?
   Q3: What is happening?
   Q4: What is the joke or message?
   Q5: Who is the target audience?
   Q6: One sentence describing this meme for search

Return JSON only. No markdown, no explanation. /no_think"""


def image_to_base64_resized(image_path, max_size=512):
    """resize и encode в base64"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)  # type: ignore
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(image_path, ocr_text="", max_retries=3):
    """отправляет картинку в ollama с retry"""
    img_b64 = image_to_base64_resized(image_path, MAX_IMAGE_SIZE)

    full_prompt = PROMPT
    if ocr_text and ocr_text.strip():
        full_prompt += f'\nOCR text detected by external OCR: "{ocr_text.strip()[:200]}"'

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": full_prompt,
            "images": [img_b64]
        }],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2000,
        }
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
            if raw and raw.strip():
                return raw
            if attempt < max_retries - 1:
                time.sleep(1)
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def parse_json_response(raw_text):
    """парсит json из ответа модели"""
    if not raw_text:
        return None
    text = raw_text.strip()
    # убираем markdown
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
    # убираем <think>
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            # пробуем закрыть обрезанный json
            fragment = text[start:]
            for closer in ['"}', '"]', '"]}', '"]}']:
                try:
                    return json.loads(fragment + closer)
                except json.JSONDecodeError:
                    continue
    return None


def load_already_processed(output_path):
    """загружает имена уже обработанных файлов"""
    done = set()
    if output_path.exists():
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


def select_images(rows, target=10000):
    telegram = []
    bing = []
    for r in rows:
        if r.get("source_type") == "telegram":
            telegram.append(r)
        elif r.get("source_type") == "bing":
            bing.append(r)

    random.seed(42)
    random.shuffle(bing)

    selected = telegram.copy()
    remaining = target - len(selected)
    if remaining > 0:
        selected.extend(bing[:remaining])
    return selected


if not INPUT_CSV.exists():
    sys.exit(1)

    # загружаем все строки
rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if Path(row.get("source_path", "")).exists():
            rows.append(row)

selected = select_images(rows, TARGET_COUNT)
done = load_already_processed(OUTPUT_JSONL)

remaining = []
for r in selected:
    if r.get("filename", "") not in done:
        remaining.append(r)

if not remaining:
    sys.exit(0)

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
stats = {"success": 0, "failed_parse": 0, "empty": 0, "skipped": 0}

with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
    for row in tqdm(remaining, desc="vqa"):
        filename = row["filename"]
        source_path = row["source_path"]
        ocr_text = row.get("ocr_text", "")
        img_path = Path(source_path)

        if not img_path.exists():
            stats["skipped"] += 1
            continue

        raw_response = query_ollama(str(img_path), ocr_text)

        if raw_response is None or raw_response.strip() == "":
            stats["empty"] += 1
            continue

        parsed = parse_json_response(raw_response)

        if parsed is None:
            stats["failed_parse"] += 1
            parsed = {"raw_response": raw_response[:500]}

        result = {
            "filename": filename,
            "source_path": source_path,
            "ocr_text_external": ocr_text,
            "confidence": float(row.get("confidence", 0)),
            "source_type": row.get("source_type", ""),
            **parsed
        }

        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        f_out.flush()
        stats["success"] += 1

print(f"готово success={stats['success']} failed={stats['failed_parse']} empty={stats['empty']}")
