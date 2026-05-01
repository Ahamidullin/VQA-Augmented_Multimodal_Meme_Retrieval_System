"""
compare_models.py — A/B тест: Qwen3-VL-8B vs Qwen3-VL-2B

Берёт 200 СЛУЧАЙНЫХ картинок из final_dataset_text.csv,
прогоняет через ОБЕ модели (8B и 2B), сохраняет результаты.
"""

import csv
import json
import time
import base64
import random
import logging
import requests
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
INPUT_CSV = Path("data/processed/final_dataset_text.csv")
OUTPUT_8B = Path("data/processed/comparison_8b.jsonl")
OUTPUT_2B = Path("data/processed/comparison_2b.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"
SAMPLE_SIZE = 200  # сколько картинок сравнивать

PROMPT = """Look at this meme image. Return ONLY a JSON object (no extra text, no markdown):
{"caption": "1-2 sentence neutral description of what is depicted",
"objects": ["list", "of", "key", "objects", "max 7"],
"tone": "one of: humor, sarcasm, critique, support, neutral, absurd",
"main_idea": "one sentence: the main message or joke"}
Respond with valid JSON only. No explanation."""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_response(raw_text):
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
            pass
    return None


def query_ollama(image_path, model, ocr_text=""):
    img_b64 = image_to_base64(image_path)
    full_prompt = PROMPT
    if ocr_text and ocr_text.strip():
        full_prompt += f'\nOCR text already extracted: "{ocr_text.strip()}"'

    payload = {
        "model": model,
        "prompt": full_prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        }
    }

    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - start
        return data.get("response", ""), elapsed
    except Exception as e:
        log.error(f"Request failed for {model}: {e}")
        return None, time.time() - start


def process_batch(rows, model, output_path):
    """Process a batch of images with one model."""
    results = []
    times = []

    with open(output_path, "w", encoding="utf-8") as f_out:
        for row in tqdm(rows, desc=f"{model}"):
            filename = row["filename"]
            source_path = row["source_path"]
            ocr_text = row.get("ocr_text", "")

            img_path = Path(source_path)
            if not img_path.exists():
                continue

            raw_resp, elapsed = query_ollama(str(img_path), model, ocr_text)
            times.append(elapsed)

            parsed = parse_json_response(raw_resp) if raw_resp else None

            result = {
                "filename": filename,
                "source_path": source_path,
                "ocr_text": ocr_text,
                "model": model,
                "time_seconds": round(elapsed, 2),
                "parse_success": parsed is not None,
            }

            if parsed:
                result.update(parsed)
            else:
                result["raw_response"] = (raw_resp or "")[:300]

            results.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

    return results, times


def main():
    random.seed(42)

    # 1. Загружаем CSV
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Только файлы, которые существуют
            if Path(row.get("source_path", "")).exists():
                rows.append(row)

    log.info(f"Total images with existing files: {len(rows)}")

    # 2. Случайная выборка
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))
    log.info(f"Sample size: {len(sample)}")

    # Показать распределение по source_type
    from collections import Counter
    sources = Counter(r.get("source_type", "?") for r in sample)
    log.info(f"Sample sources: {dict(sources)}")

    # 3. 8B прогон
    log.info("=" * 40)
    log.info("Running 8B model...")
    results_8b, times_8b = process_batch(sample, "qwen3-vl:8b", OUTPUT_8B)
    success_8b = sum(1 for r in results_8b if r["parse_success"])
    avg_8b = sum(times_8b) / len(times_8b) if times_8b else 0
    log.info(f"8B done: {success_8b}/{len(results_8b)} parsed ({success_8b/len(results_8b)*100:.1f}%), avg {avg_8b:.2f}s")

    # 4. 2B прогон
    log.info("=" * 40)
    log.info("Running 2B model...")
    results_2b, times_2b = process_batch(sample, "qwen3-vl:2b", OUTPUT_2B)
    success_2b = sum(1 for r in results_2b if r["parse_success"])
    avg_2b = sum(times_2b) / len(times_2b) if times_2b else 0
    log.info(f"2B done: {success_2b}/{len(results_2b)} parsed ({success_2b/len(results_2b)*100:.1f}%), avg {avg_2b:.2f}s")

    # 5. Summary
    log.info("=" * 40)
    log.info("SUMMARY")
    log.info(f"{'Metric':<25} {'8B':<15} {'2B':<15}")
    log.info(f"{'JSON Success Rate':<25} {success_8b/len(results_8b)*100:.1f}%{'':>10} {success_2b/len(results_2b)*100:.1f}%")
    log.info(f"{'Avg time (s/img)':<25} {avg_8b:.2f}{'':>12} {avg_2b:.2f}")
    log.info(f"{'ETA 10k images (hours)':<25} {10000*avg_8b/3600:.1f}{'':>12} {10000*avg_2b/3600:.1f}")


if __name__ == "__main__":
    main()
