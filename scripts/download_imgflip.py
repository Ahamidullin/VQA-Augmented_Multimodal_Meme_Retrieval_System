"""
Скачивание чистых мемов из ImgFlip575K.
Фильтрует по стоп-словам, скачивает картинки, создаёт metadata.csv.

Запуск:
    python scripts/download_imgflip.py
"""

import json
import csv
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# === Настройки ===
DATASET_DIR = Path('data/raw/imgflip575k/dataset/memes')
OUTPUT_DIR = Path('data/raw/imgflip_clean')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

MAX_TOTAL = 5000        # сколько всего мемов скачать
MAX_PER_TEMPLATE = 80   # макс мемов с одного шаблона
WORKERS = 8             # параллельные загрузки

# === Загружаем стоп-слова ===
bad_words = []
with open('configs/bad_words.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            bad_words.append(line.lower())
print(f'Загружено {len(bad_words)} стоп-слов')


def is_clean(meme: dict) -> bool:
    """Проверяет что мем не содержит стоп-слов."""
    texts = meme.get('boxes', [])
    title = meme.get('metadata', {}).get('title', '')
    all_text = ' '.join(texts + [title]).lower()
    for word in bad_words:
        if word in all_text:
            return False
    return True


def download_image(url: str, save_path: Path) -> bool:
    """Скачивает одну картинку."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and 'image' in resp.headers.get('content-type', ''):
            save_path.write_bytes(resp.content)
            return True
    except:
        pass
    return False


def main():
    # Собираем все JSON-файлы шаблонов
    json_files = sorted(DATASET_DIR.glob('*.json'))
    print(f'Найдено {len(json_files)} шаблонов')

    # Собираем чистые мемы из всех шаблонов
    all_memes = []  # (template_name, meme_dict)

    for jf in json_files:
        template_name = jf.stem
        with open(jf, 'r') as f:
            memes = json.load(f)

        clean = [m for m in memes if is_clean(m)]
        # Берём не больше MAX_PER_TEMPLATE
        selected = clean[:MAX_PER_TEMPLATE]
        for m in selected:
            all_memes.append((template_name, m))

        if len(all_memes) >= MAX_TOTAL:
            all_memes = all_memes[:MAX_TOTAL]
            break

    print(f'Отобрано {len(all_memes)} чистых мемов для скачивания')

    # Скачиваем параллельно
    metadata_rows = []
    downloaded = 0
    failed = 0

    def process(idx, template_name, meme):
        url = meme['url']
        ext = url.split('.')[-1].split('?')[0]
        if ext not in ('jpg', 'jpeg', 'png', 'gif', 'webp'):
            ext = 'jpg'
        filename = f'imgflip_{idx:05d}.{ext}'
        save_path = IMAGES_DIR / filename
        ok = download_image(url, save_path)
        if ok:
            boxes_text = ' | '.join(meme.get('boxes', []))
            title = meme.get('metadata', {}).get('title', '')
            return {
                'image_name': filename,
                'template': template_name,
                'title': title,
                'text': boxes_text,
                'source_url': url,
                'source': 'imgflip'
            }
        return None

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {}
        for idx, (tname, meme) in enumerate(all_memes):
            fut = executor.submit(process, idx, tname, meme)
            futures[fut] = idx

        for fut in tqdm(as_completed(futures), total=len(futures), desc='Скачиваем'):
            result = fut.result()
            if result:
                metadata_rows.append(result)
                downloaded += 1
            else:
                failed += 1

    # Сортируем по имени файла
    metadata_rows.sort(key=lambda x: x['image_name'])

    # Сохраняем metadata.csv
    csv_path = OUTPUT_DIR / 'metadata.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'template', 'title', 'text', 'source_url', 'source'])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f'\n=== Готово ===')
    print(f'Скачано: {downloaded}')
    print(f'Не удалось: {failed}')
    print(f'Метаданные: {csv_path}')
    print(f'Картинки: {IMAGES_DIR}')


if __name__ == '__main__':
    main()
