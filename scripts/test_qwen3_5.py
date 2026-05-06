"""
test_qwen35.py

Тест Qwen3.5:9b vs Qwen2.5-VL:3b на 100 валидационных мемах.
Сравнивает качество описаний.
"""

import json
import base64
import time
import requests
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NEW = "qwen3.5:9b"
MAX_IMAGE_SIZE = 512

PROMPT = """Analyze this meme. Return JSON with these fields:

1. "caption" - literal visual description: who/what is shown, setting, expressions (1-2 sentences, no interpretation)
2. "ocr_text" - exact text from the image, preserve original language and punctuation. Empty string "" if no text.
3. "meme_template" - known template name if recognized (e.g. "Drake", "Distracted Boyfriend", "Two Buttons"), else ""
4. "objects" - list of key visual elements: ["person", "cat", "phone", ...]
5. "tone" - one of: "ironic", "wholesome", "absurd", "dark", "relatable", "aggressive", "neutral"
6. "main_idea" - the joke or point being made, why it's funny (1 sentence, interpretation allowed)

Return JSON only. No markdown, no explanation. /no_think"""


def image_to_base64_resized(image_path, max_size=512):
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


def query_model(image_path, model=MODEL_NEW):
    img_b64 = image_to_base64_resized(image_path, MAX_IMAGE_SIZE)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2000}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
        # Убираем think tags
        if "<think>" in raw:
            think_end = raw.find("</think>")
            if think_end != -1:
                raw = raw[think_end + len("</think>"):]
        # Парсим JSON
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1]
        if "```" in text:
            text = text.split("```")[0]
        return json.loads(text.strip())
    except Exception as e:
        return {"error": str(e), "raw": raw[:200] if 'raw' in dir() else ""}


def main():
    # Загружаем 100 валидационных мемов
    with open("eval/validation_set/queries_v3.json") as f:
        queries = json.load(f)

    with open("data/processed/vqa_annotations_v2.jsonl") as f:
        all_recs = {r["filename"]: r for r in (json.loads(l) for l in f)}

    output_path = Path("eval/results/qwen35_vs_25_comparison.json")
    results = []

    # Загружаем прогресс
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done = {r["filename"] for r in results}
    else:
        done = set()

    for q in tqdm(queries, desc="Qwen3.5:9b"):
        fn = q["filename"]
        if fn in done:
            continue

        # Найти путь к картинке
        rec = all_recs.get(fn, {})
        sp = rec.get("source_path", "")
        if not sp or not Path(sp).exists():
            for p in Path("data/raw").rglob(fn):
                sp = str(p)
                break

        if not sp or not Path(sp).exists():
            continue

        new_ann = query_model(sp, MODEL_NEW)
        time.sleep(0.3)

        results.append({
            "filename": fn,
            "old_caption": rec.get("caption", ""),
            "old_ocr": rec.get("ocr_text", ""),
            "old_main_idea": rec.get("main_idea", ""),
            "new": new_ann
        })

        # Сохраняем каждые 10
        if len(results) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Статистика
    ok = sum(1 for r in results if "error" not in r.get("new", {}))
    print(f"\nГотово: {ok}/{len(results)} успешных")
    print(f"Сохранено: {output_path}")

    # Пример сравнения
    for r in results[:3]:
        print(f"\n--- {r['filename']} ---")
        print(f"OLD caption: {r['old_caption']}")
        print(f"NEW caption: {r['new'].get('caption', 'ERROR')}")
        print(f"OLD main_idea: {r['old_main_idea']}")
        print(f"NEW main_idea: {r['new'].get('main_idea', 'ERROR')}")


if __name__ == "__main__":
    main()
