"""
generate_vqa.py

Полный пайплайн VQA-аннотаций мемов через Qwen3.5:9b (ollama).
Один проход: базовое описание + обогащение.

Вход:  data/processed/final_dataset_text.csv
Выход: data/processed/vqa_annotations_v2.jsonl

Поддерживает resume (безопасно перезапускать).
"""

import csv
import json
import time
import base64
import random
import logging
import requests
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vqa_generation.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# конфиг
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
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
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
        "messages": [
            {
                "role": "user",
                "content": full_prompt,
                "images": [img_b64]
            }
        ],
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
            data = resp.json()
            raw = data.get("message", {}).get("content", "")
            if raw and raw.strip():
                return raw
            if attempt < max_retries - 1:
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed (attempt {attempt+1}): {e}")
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
            for closer in ['"}', '"]', '"]}', '"}]}']:
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
    """выбираем: весь telegram, потом bing. hf пропускаем"""
    telegram = [r for r in rows if r.get("source_type") == "telegram"]
    bing = [r for r in rows if r.get("source_type") == "bing"]

    random.seed(42)
    random.shuffle(bing)

    selected = telegram.copy()
    remaining = target - len(selected)
    if remaining > 0:
        selected.extend(bing[:remaining])

    log.info(f"Selected {len(selected)} images: "
             f"{len(telegram)} telegram + {min(remaining, len(bing))} bing")
    return selected


def main():
    log.info(f"Запуск VQA генерации | Model: {MODEL}")

    if not INPUT_CSV.exists():
        log.error(f"Input CSV not found: {INPUT_CSV}")
        return

    # загружаем все строки
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if Path(row.get("source_path", "")).exists():
                rows.append(row)

    log.info(f"Images with existing files: {len(rows)}")

    # выбираем подмножество
    selected = select_images(rows, TARGET_COUNT)

    # resume
    done = load_already_processed(OUTPUT_JSONL)
    log.info(f"Already processed: {len(done)}")

    remaining = [r for r in selected if r.get("filename", "") not in done]
    log.info(f"Remaining: {len(remaining)}")

    if not remaining:
        log.info("All done!")
        return

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed_parse": 0, "empty": 0, "skipped": 0}
    start_time = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
        for row in tqdm(remaining, desc="VQA"):
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
                log.warning(f"Parse fail: {filename}: {raw_response[:80]}")
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

    elapsed = time.time() - start_time
    total = stats["success"] + stats["failed_parse"] + stats["empty"]
    log.info(f"DONE in {elapsed/3600:.1f} hours")
    log.info(f"Success: {stats['success']} | Parse fail: {stats['failed_parse']} | Empty: {stats['empty']}")
    if total > 0:
        log.info(f"Avg: {elapsed/total:.1f}s/image")


if __name__ == "__main__":
    main()
