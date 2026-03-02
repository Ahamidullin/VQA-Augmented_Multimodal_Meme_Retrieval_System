"""скачивание мемов с huggingface datasets.
скачивает картинки, фильтрует по стоп-словам, сохраняет metadata.csv.

перед запуском: pip install datasets pillow

запуск:python scripts/download_hf_memes.py
"""

import csv
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


OUTPUT_DIR = Path('data/raw/hf_memes')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

MAX_TOTAL = 8000

# стоп слова
bad_words = []
bad_words_path = Path('configs/bad_words.txt')
if bad_words_path.exists():
    with open(bad_words_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                bad_words.append(line.lower())
print(f'Загружено {len(bad_words)} стоп-слов')


def is_clean(text: str) -> bool:
    """проверка что текст без стоп-слов"""
    text_lower = text.lower()
    for word in bad_words:
        if word in text_lower:
            return False
    return True


def save_image(img, save_path: Path) -> bool:
    """сохранение pil image в jpeg"""
    try:
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(save_path, 'JPEG', quality=90)
        return True
    except Exception as e:

        return False


def safe_iter(ds):
    """итерация по датасету с пропуском битых элементов"""
    if hasattr(ds, '__iter__'):
        iterator = iter(ds)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break
            except Exception as e:
                continue
    else:

        for item in ds:
            yield item


def download_memes_dataset():
    """not-lain/meme-dataset, маленький датасет ~300 картинок"""
    print('\nnot-lain/meme-dataset')
    results = []

    try:

        ds = load_dataset('not-lain/meme-dataset', split='train')
    except Exception as e:
        print(f'  Не удалось загрузить: {e}')
        return results

    for idx, item in enumerate(tqdm(safe_iter(ds), desc='not-lain')):
        text = ''
        for field in ['text', 'caption', 'title', 'label', 'description']:
            if field in item and isinstance(item[field], str):
                text += ' ' + item[field]

        if text.strip() and not is_clean(text):
            continue

        img = item.get('image')
        if img is None:
            continue

        filename = f'hf_notlain_{idx:05d}.jpg'
        save_path = IMAGES_DIR / filename
        if save_image(img, save_path):
            results.append({
                'image_name': filename,
                'title': text.strip(),
                'text_on_image': '',
                'tags': '',
                'source': 'not-lain/meme-dataset',
            })

    print(f'  Сохранено: {len(results)}')
    return results


def download_mimic_memes(limit=5000):
    """Aakash941/MIMIC-Meme-Dataset, 5k+ мемов"""
    print('\nAakash941/MIMIC-Meme-Dataset')
    results = []

    try:

        ds = load_dataset('Aakash941/MIMIC-Meme-Dataset', split='train')
    except Exception as e:
        print(f'  Не удалось загрузить (попробуем stream): {e}')
        try:
            ds = load_dataset('Aakash941/MIMIC-Meme-Dataset', split='train', streaming=True)
        except Exception as e2:
             print(f'  Не удалось загрузить вообще: {e2}')
             return results

    for idx, item in enumerate(tqdm(safe_iter(ds), desc='MIMIC')):
        if len(results) >= limit:
            break
            
        text = ''
        for field in ['text', 'caption', 'title', 'label', 'description', 'OCR_text']:
            if field in item and isinstance(item[field], str):
                text += ' ' + item[field]

        if text.strip() and not is_clean(text):
            continue

        img = item.get('image')
        if img is None:
            continue

        filename = f'hf_mimic_{idx:05d}.jpg'
        save_path = IMAGES_DIR / filename
        if save_image(img, save_path):
            results.append({
                'image_name': filename,
                'title': text.strip(),
                'text_on_image': '',
                'tags': '',
                'source': 'Aakash941/MIMIC-Meme-Dataset',
            })

    print(f'  Сохранено: {len(results)}')
    return results


def download_harpreetsahota_memes():
    """harpreetsahota/memes-dataset"""
    print('\nharpreetsahota/memes-dataset')
    results = []

    try:

        ds = load_dataset('harpreetsahota/memes-dataset', split='train')
    except Exception as e:
        print(f'  Не удалось загрузить: {e}')

        try:
            ds = load_dataset('harpreetsahota/memes-dataset', split='train', streaming=True)
        except:
            return results

    for idx, item in enumerate(tqdm(safe_iter(ds), desc='Harpreet')):
        text = ''
        for field in ['text', 'caption', 'title', 'label', 'description']:
            if field in item and isinstance(item[field], str):
                text += ' ' + item[field]

        if text.strip() and not is_clean(text):
            continue

        img = item.get('image')
        if img is None:
            continue

        filename = f'hf_harpreet_{idx:05d}.jpg'
        save_path = IMAGES_DIR / filename
        if save_image(img, save_path):
            results.append({
                'image_name': filename,
                'title': text.strip(),
                'text_on_image': '',
                'tags': '',
                'source': 'harpreetsahota/memes-dataset',
            })
            
        if len(results) >= 2000:
            break

    print(f'  Сохранено: {len(results)}')
    return results


def main():
    print('HuggingFace Meme Downloader')


    csv_path = OUTPUT_DIR / 'metadata.csv'
    fieldnames = ['image_name', 'title', 'text_on_image', 'tags', 'source']
    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def save_results(new_results):
        if not new_results:
            return
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(new_results)
        print(f"  Saved {len(new_results)} records to metadata.csv")

    all_results = []

    # not-lain
    res = download_memes_dataset()
    save_results(res)
    all_results.extend(res)
    
    # mimic
    if len(all_results) < MAX_TOTAL:
        needed = MAX_TOTAL - len(all_results)
        res = download_mimic_memes(limit=needed)
        save_results(res)
        all_results.extend(res)

    # harpreet
    if len(all_results) < MAX_TOTAL:
        res = download_harpreetsahota_memes()
        save_results(res)
        all_results.extend(res)

    print(f'\nготово')
    print(f'Всего сохранено: {len(all_results)}')
    print(f'Метаданные: {csv_path}')
    print(f'Картинки: {IMAGES_DIR}')

    from collections import Counter
    src_counts = Counter(r['source'] for r in all_results)
    print(f'\nпо источникам:')
    for src, count in src_counts.most_common():
        print(f'  {src}: {count}')


if __name__ == '__main__':
    main()
