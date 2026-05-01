"""
Генерация русских поисковых запросов для языкового эксперимента.
Берёт EN-описания мемов и генерирует короткие RU-запросы через Qwen.
"""

import json
import requests
from tqdm import tqdm
from pathlib import Path

QUERIES_FILE = Path("eval/language_experiment_queries.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-vl:2b"  # лёгкая модель, для перевода хватит


def translate_query(caption_en, main_idea_en, query_en):
    """Генерирует короткий русский поисковый запрос для мема"""
    prompt = f"""You are a translator. Given an English meme description, write a SHORT Russian search query (3-7 words) that a Russian-speaking user would type to find this meme.

English description: {caption_en}
Main idea: {main_idea_en}
English query: {query_en}

Write ONLY the Russian query, nothing else. No quotes, no explanation. Example format:
мем про кота и программиста"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 50}
        }, timeout=30)
        
        text = resp.json().get("response", "").strip()
        # Берём только первую строку
        text = text.split("\n")[0].strip().strip('"\'')
        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""


def main():
    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    updated = 0
    for q in tqdm(queries, desc="Generating RU queries"):
        if q.get("query_ru"):  # уже есть
            continue
        
        ru = translate_query(q["caption"], q["main_idea"], q["query_en"])
        q["query_ru"] = ru
        updated += 1

    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"\nОбновлено {updated} запросов")
    print("\nПримеры:")
    for q in queries[:10]:
        print(f'  EN: "{q["query_en"]}"')
        print(f'  RU: "{q["query_ru"]}"')
        print()


if __name__ == "__main__":
    main()
