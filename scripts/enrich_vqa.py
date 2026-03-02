"""
enrich_vqa.py

Дополняет существующие VQA-аннотации расширенными полями:
- ocr_normalized: очищенный OCR текст
- objects_detailed: расширенный список объектов (5-15)
- relations: отношения [(subj, rel, obj), ...]
- required_context: нужен ли внешний контекст
- vqa: блок вопрос-ответ (6 пар)

Читает vqa_annotations.jsonl, отправляет картинку + caption,
сохраняет результат в vqa_annotations_v2.jsonl.
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
        logging.FileHandler("vqa_enrich.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# Конфигурация
INPUT_JSONL = Path("data/processed/vqa_annotations.jsonl")
OUTPUT_JSONL = Path("data/processed/vqa_annotations_v2.jsonl")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:3b"
MAX_IMAGE_SIZE = 512

ENRICH_PROMPT = """Context about this meme:
Caption: "{caption}"
OCR: "{ocr_text}"
Tone: "{tone}"

Return a JSON with these fields:
1. "ocr_normalized" - clean the OCR text: fix repeated characters like AAAA->A, remove noise. Empty string if no text.
2. "objects_detailed" - list 5-15 visible objects/elements in the image
3. "relations" - up to 5 triples like ["cat", "sitting on", "table"]
4. "required_context" - {{"is_required": bool, "what": "what knowledge is needed to understand this meme"}}
5. "vqa" - answer these 6 questions about the meme:
   Q1: What text is on the meme?
   Q2: What is literally depicted?
   Q3: What is happening?
   Q4: What is the joke or message?
   Q5: Who is the target audience?
   Q6: One sentence describing this meme for search

Format: {{"ocr_normalized": "...", "objects_detailed": [...], "relations": [[...]], "required_context": {{...}}, "vqa": [{{"q": "...", "a": "..."}}, ...]}}
JSON only. No markdown."""


def image_to_base64_resized(image_path, max_size=512):
    """Resize и encode в base64."""
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


def query_ollama(image_path, prompt_text, max_retries=2):
    """Отправляет картинку + промпт в Ollama."""
    img_b64 = image_to_base64_resized(image_path, MAX_IMAGE_SIZE)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [img_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 800,
        }
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
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
    """Парсит JSON из ответа модели."""
    if not raw_text:
        return None
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]
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
            fragment = text[start:]
            for closer in ['"}', '"]', '"]}', '"}]}'  , '"}]}']:
                try:
                    return json.loads(fragment + closer)
                except json.JSONDecodeError:
                    continue
    return None


def load_existing_records(path):
    """Загружает существующие аннотации."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_already_enriched(path):
    """Загружает уже обработанные файлы для resume."""
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        done.add(obj.get("filename", ""))
                    except json.JSONDecodeError:
                        continue
    return done


def main():
    log.info("Запуск обогащения VQA-аннотаций")
    log.info(f"Модель: {MODEL}")

    records = load_existing_records(INPUT_JSONL)
    log.info(f"Загружено {len(records)} базовых аннотаций")

    done = load_already_enriched(OUTPUT_JSONL)
    log.info(f"Уже обработано: {len(done)}")

    remaining = [r for r in records if r.get("filename", "") not in done]
    log.info(f"Нужно обработать: {len(remaining)}")

    if not remaining:
        log.info("Всё готово!")
        return

    random.seed(42)
    random.shuffle(remaining)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0, "no_image": 0}
    start_time = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
        for record in tqdm(remaining, desc="Enriching VQA"):
            filename = record.get("filename", "")
            source_path = record.get("source_path", "")
            img_path = Path(source_path)

            if not img_path.exists():
                # Сохраняем оригинал без обогащения
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                stats["no_image"] += 1
                continue

            # Формируем промпт с контекстом
            prompt = ENRICH_PROMPT.format(
                caption=record.get("caption", "no caption"),
                ocr_text=record.get("ocr_text", "")[:200],
                tone=record.get("tone", "unknown"),
            )

            raw_response = query_ollama(str(img_path), prompt)
            enriched = parse_json_response(raw_response)

            if enriched:
                # Мержим новые поля в существующую запись
                merged = {**record}
                merged["ocr_normalized"] = enriched.get("ocr_normalized", "")
                merged["objects_detailed"] = enriched.get("objects_detailed", record.get("objects", []))
                merged["relations"] = enriched.get("relations", [])
                merged["required_context"] = enriched.get("required_context", {"is_required": False, "what": ""})
                merged["vqa"] = enriched.get("vqa", [])

                f_out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                f_out.flush()
                stats["success"] += 1
            else:
                # Сохраняем оригинал
                log.warning(f"Parse fail: {filename}")
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                stats["failed"] += 1

    elapsed = time.time() - start_time
    total = stats["success"] + stats["failed"] + stats["no_image"]
    log.info(f"Завершено за {elapsed/3600:.1f} часов")
    log.info(f"Успешно: {stats['success']}, ошибок: {stats['failed']}, без картинки: {stats['no_image']}")
    if total > 0:
        log.info(f"Среднее: {elapsed/total:.1f} с/мем")


if __name__ == "__main__":
    main()
