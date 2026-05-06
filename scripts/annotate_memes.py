"""
annotate_memes.py

Аннотация мемов через GPT-4o-mini API.
Один проход: описание + метаданные + VQA.

Вход:  data/processed/final_dataset_text.csv
Выход: data/processed/vqa_annotations_v3.jsonl

Поддерживает resume (безопасно перезапускать).
"""

import csv
import json
import time
import base64
import logging
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("annotation.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# конфиг
INPUT_CSV = Path("data/processed/final_dataset_text.csv")
OUTPUT_JSONL = Path("data/processed/vqa_annotations_v3.jsonl")
MODEL = "gpt-4o-mini"
MAX_IMAGE_SIZE = 512

client = OpenAI()

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
    """resize и encode в base64"""
    # TODO: реализовать
    pass


def query_gpt(image_path, ocr_text=""):
    """отправляет картинку в GPT API"""
    # TODO: реализовать
    pass


def parse_json_response(raw_text):
    """парсит json из ответа модели"""
    # TODO: реализовать
    pass


def load_already_processed(output_path):
    """загружает имена уже обработанных файлов для resume"""
    # TODO: реализовать
    pass


def main():
    log.info(f"Запуск аннотации | Model: {MODEL}")

    # TODO: загрузить CSV, отфильтровать существующие файлы
    # TODO: загрузить уже обработанные (resume)
    # TODO: цикл по мемам с tqdm
    # TODO: сохранять результат в JSONL


if __name__ == "__main__":
    main()
