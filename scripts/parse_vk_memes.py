"""
Парсер мемов из VK пабликов.
Скачивает картинки из открытых пабликов с мемами, фильтрует по стоп-словам.

Настройка:
    1. Получи сервисный ключ VK API: https://vk.com/apps?act=manage
    2. Добавь в .env файл: VK_SERVICE_TOKEN=ваш_ключ
    3. Запусти: python scripts/parse_vk_memes.py

Или передай ключ напрямую:
    python scripts/parse_vk_memes.py --token ВАШ_КЛЮЧ
"""

import os
import csv
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# === Настройки ===
OUTPUT_DIR = Path('data/raw/vk_memes')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

VK_API_VERSION = '5.199'
VK_API_BASE = 'https://api.vk.com/method'

# Паблики с мемами (domain или числовой id)
# Подобраны максимально безобидные, без политики
GROUPS = [
    'memes',              # Мемы
    'memesrussia',        # Мемы (Россия)
    'memes_prog',         # Мемы для программистов
    'ithumor',            # IT-юмор
    'science_humor',      # Научный юмор
    'catsmemes',          # Мемы с котиками
    'animalsmems',        # Мемы с животными
    'dayvinchik',         # Дайвинчик
    'jumoreski',          # Юморески
    'lepramemes',         # Мемы
]

MAX_PER_GROUP = 300      # макс постов с каждого паблика
MAX_TOTAL = 3000         # всего мемов
COUNT_PER_REQUEST = 100  # постов за запрос (макс VK = 100)


def load_bad_words():
    """Загружает стоп-слова из файла."""
    bad_words = []
    path = Path('configs/bad_words.txt')
    if path.exists():
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    bad_words.append(line.lower())
    # Добавляем русские стоп-слова
    russian_bad = [
        'блять', 'бляд', 'хуй', 'хуя', 'хуе', 'пизд', 'ебат', 'ебан',
        'ебну', 'сука', 'мудак', 'мудач', 'пидор', 'пидар', 'залуп',
        'шлюх', 'дроч', 'нахуй', 'нахер', 'похуй', 'заеб', 'отъеб',
        'выеб', 'долбоёб', 'долбоеб', 'ёбан', 'уёб', 'трахн',
        'путин', 'навальн', 'политик', 'война', 'укра', 'зеленск',
        'байден', 'трамп', 'сво '
    ]
    bad_words.extend(russian_bad)
    print(f'Загружено {len(bad_words)} стоп-слов (EN + RU)')
    return bad_words


def is_clean(text: str, bad_words: list) -> bool:
    """Проверяет что текст не содержит стоп-слов."""
    if not text:
        return True
    text_lower = text.lower()
    for word in bad_words:
        if word in text_lower:
            return False
    return True


def get_wall_posts(token: str, domain: str, count: int = 100, offset: int = 0):
    """Получает посты со стены паблика."""
    resp = requests.get(f'{VK_API_BASE}/wall.get', params={
        'domain': domain,
        'count': min(count, 100),
        'offset': offset,
        'access_token': token,
        'v': VK_API_VERSION,
    })
    data = resp.json()
    if 'error' in data:
        print(f'  Ошибка VK API для {domain}: {data["error"]["error_msg"]}')
        return []
    return data.get('response', {}).get('items', [])


def get_best_photo_url(photo: dict) -> str:
    """Выбирает URL фото максимального размера."""
    sizes = photo.get('sizes', [])
    if not sizes:
        return ''
    # Приоритет размеров: w > z > y > x > m > s
    priority = {'w': 6, 'z': 5, 'y': 4, 'x': 3, 'm': 2, 's': 1}
    best = max(sizes, key=lambda s: priority.get(s.get('type', ''), 0))
    return best.get('url', '')


def download_image(url: str, save_path: Path) -> bool:
    """Скачивает картинку."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and 'image' in resp.headers.get('content-type', ''):
            save_path.write_bytes(resp.content)
            return True
    except:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description='Парсер мемов из VK')
    parser.add_argument('--token', type=str, default=None, help='VK сервисный ключ')
    args = parser.parse_args()

    # Получаем токен
    token = args.token or os.environ.get('VK_SERVICE_TOKEN')
    if not token:
        # Пробуем из .env
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('VK_SERVICE_TOKEN='):
                        token = line.strip().split('=', 1)[1]
                        break

    if not token:
        print('❌ Нужен VK сервисный ключ!')
        print('   python scripts/parse_vk_memes.py --token ВАШ_КЛЮЧ')
        print('   или добавь VK_SERVICE_TOKEN=... в .env')
        return

    bad_words = load_bad_words()
    metadata_rows = []
    total_downloaded = 0
    idx = 0

    for group in GROUPS:
        if total_downloaded >= MAX_TOTAL:
            break

        print(f'\n📥 Парсим паблик: {group}')
        group_downloaded = 0
        offset = 0

        while group_downloaded < MAX_PER_GROUP and total_downloaded < MAX_TOTAL:
            posts = get_wall_posts(token, group, COUNT_PER_REQUEST, offset)
            if not posts:
                break

            for post in posts:
                if total_downloaded >= MAX_TOTAL or group_downloaded >= MAX_PER_GROUP:
                    break

                text = post.get('text', '')

                # Проверяем текст поста
                if not is_clean(text, bad_words):
                    continue

                # Ищем фото в аттачментах
                attachments = post.get('attachments', [])
                for att in attachments:
                    if att.get('type') != 'photo':
                        continue

                    photo = att.get('photo', {})
                    url = get_best_photo_url(photo)
                    if not url:
                        continue

                    # Скачиваем
                    ext = 'jpg'
                    filename = f'vk_{idx:05d}.{ext}'
                    save_path = IMAGES_DIR / filename

                    if download_image(url, save_path):
                        metadata_rows.append({
                            'image_name': filename,
                            'text': text[:500] if text else '',
                            'source': f'vk_{group}',
                            'language': 'ru',
                            'source_url': f'https://vk.com/wall{post.get("owner_id")}_{post.get("id")}',
                        })
                        total_downloaded += 1
                        group_downloaded += 1
                        idx += 1

                        if total_downloaded % 50 == 0:
                            print(f'  Скачано: {total_downloaded}/{MAX_TOTAL}')

                    if total_downloaded >= MAX_TOTAL or group_downloaded >= MAX_PER_GROUP:
                        break

            offset += COUNT_PER_REQUEST
            time.sleep(0.35)  # VK rate limit: 3 запроса/сек

        print(f'  ✅ {group}: скачано {group_downloaded} мемов')

    # Сохраняем metadata.csv
    csv_path = OUTPUT_DIR / 'metadata.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'text', 'source', 'language', 'source_url'])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f'\n=== Готово ===')
    print(f'Скачано: {total_downloaded} мемов')
    print(f'Метаданные: {csv_path}')
    print(f'Картинки: {IMAGES_DIR}')


if __name__ == '__main__':
    main()
