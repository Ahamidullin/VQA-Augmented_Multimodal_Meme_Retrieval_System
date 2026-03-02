"""
generate_vqa.py

генерация vqa-аннотаций мемов через qwen vl (ollama chat api)
приоритет: telegram > bing
поддерживает resume, retry, resize
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

# конфиг
INPUT_CSV = Path("data/processed/final_dataset_text.csv")
OUTPUT_JSONL = Path("data/processed/vqa_annotations.jsonl")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:3b"

TARGET_COUNT = 10000
MAX_IMAGE_SIZE = 512

PROMPT = """Look at this meme image. Return ONLY a JSON object:
{"caption": "1-2 sentence neutral description",
"objects": ["key", "objects", "max 5"],
"tone": "humor/sarcasm/critique/support/neutral/absurd",
"main_idea": "one sentence: the main message"}
JSON only. No markdown. No explanation."""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vqa_generation.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)


def load_already_processed(output_path):
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
        # fallback без resize
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(image_path, ocr_text="", max_retries=2):
    """отправляет картинку в ollama с retry"""
    img_b64 = image_to_base64_resized(image_path, MAX_IMAGE_SIZE)

    full_prompt = PROMPT
    if ocr_text and ocr_text.strip():
        full_prompt += f'\nOCR text: "{ocr_text.strip()[:200]}"'

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
            "temperature": 0.3,
            "num_predict": 256,
        }
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            raw_response = data.get("message", {}).get("content", "")
            if raw_response and raw_response.strip():
                return raw_response
            # пустой ответ, повторяем
            if attempt < max_retries - 1:
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def parse_json_response(raw_text):
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
            for closer in ['"}', '"]', '"]}']:
                try:
                    return json.loads(fragment + closer)
                except json.JSONDecodeError:
                    continue
    return None


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
    log.info("запуск генерации vqa-аннотаций")
    log.info(f"Model: {MODEL}")
    log.info(f"Target: {TARGET_COUNT} images")
    log.info(f"Image resize: {MAX_IMAGE_SIZE}px")

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

    log.info(f"Total images with existing files: {len(rows)}")

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
        for row in tqdm(remaining, desc="VQA Generation"):
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
                "ocr_text": ocr_text,
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
    log.info(f"Success: {stats['success']} | Parse fail (raw saved): {stats['failed_parse']} | Empty: {stats['empty']}")
    if total > 0:
        log.info(f"Avg time: {elapsed/total:.1f}s/image")


if __name__ == "__main__":
    main()
