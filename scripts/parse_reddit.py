"""
Скачивание мемов с Reddit через JSON-эндпоинты (без API ключей).
Фильтрует по стоп-словам, скачивает только картинки с i.redd.it.

Запуск:
    python scripts/parse_reddit.py

Настройки в configs/config.yaml и configs/bad_words.txt
"""

import json
import csv
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# === Настройки ===
OUTPUT_DIR = Path('data/raw/reddit_memes')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

# Сабреддиты для парсинга (добрые/нейтральные)
SUBREDDITS = [
    'memes',
    'wholesomememes',
    'me_irl',
    'dankmemes',
    'AdviceAnimals',
    'MemeEconomy',
    'BikiniBottomTwitter',
    'PrequelMemes',
    'lotrmemes',
    'programmerhumor',
    'historymemes',
    'sciencememes',
    'mathmemes',
    'antimeme',
]

# Сортировки для каждого сабреддита
SORT_MODES = ['top', 'hot']
TIME_FILTERS = ['all', 'year', 'month']  # для top
POSTS_PER_REQUEST = 100  # Reddit отдаёт максимум 100
MAX_PAGES = 5            # страниц на сортировку
MAX_TOTAL = 8000         # общий лимит
WORKERS = 4              # параллельные загрузки (не больше, Reddit банит)
DELAY_BETWEEN_REQUESTS = 2  # секунд между запросами к Reddit

HEADERS = {
    'User-Agent': 'MemeCollector/1.0 (academic research project)'
}

# === Загружаем стоп-слова ===
bad_words = []
bad_words_path = Path('configs/bad_words.txt')
if bad_words_path.exists():
    with open(bad_words_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                bad_words.append(line.lower())
print(f'Загружено {len(bad_words)} стоп-слов')


def is_clean(title: str) -> bool:
    """Проверяет что заголовок не содержит стоп-слов."""
    title_lower = title.lower()
    for word in bad_words:
        if word in title_lower:
            return False
    return True


def is_image_url(url: str) -> bool:
    """Проверяет что URL — это прямая ссылка на картинку."""
    return url.startswith('https://i.redd.it/') and any(
        url.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.webp')
    )


def fetch_reddit_json(url: str) -> dict | None:
    """Запрашивает JSON с Reddit."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print(f'  Rate limited, ждём 60 сек...')
            time.sleep(60)
            return fetch_reddit_json(url)  # retry
        else:
            print(f'  HTTP {resp.status_code} для {url}')
    except Exception as e:
        print(f'  Ошибка: {e}')
    return None


def collect_posts_from_subreddit(subreddit: str) -> list[dict]:
    """Собирает посты из одного сабреддита."""
    posts = []
    seen_ids = set()

    for sort in SORT_MODES:
        if sort == 'top':
            # Для top перебираем разные time_filter
            for tf in TIME_FILTERS:
                after = None
                for page in range(MAX_PAGES):
                    url = f'https://www.reddit.com/r/{subreddit}/{sort}.json?t={tf}&limit={POSTS_PER_REQUEST}'
                    if after:
                        url += f'&after={after}'

                    data = fetch_reddit_json(url)
                    time.sleep(DELAY_BETWEEN_REQUESTS)

                    if not data or 'data' not in data:
                        break

                    children = data['data'].get('children', [])
                    if not children:
                        break

                    for child in children:
                        post = child.get('data', {})
                        post_id = post.get('id', '')

                        # Пропускаем дубликаты, NSFW, не-картинки
                        if post_id in seen_ids:
                            continue
                        if post.get('over_18', False):
                            continue
                        url_img = post.get('url', '')
                        if not is_image_url(url_img):
                            continue
                        title = post.get('title', '')
                        if not is_clean(title):
                            continue

                        seen_ids.add(post_id)
                        posts.append({
                            'id': post_id,
                            'title': title,
                            'url': url_img,
                            'subreddit': subreddit,
                            'score': post.get('score', 0),
                            'permalink': f"https://reddit.com{post.get('permalink', '')}",
                        })

                    after = data['data'].get('after')
                    if not after:
                        break

        elif sort == 'hot':
            after = None
            for page in range(MAX_PAGES):
                url = f'https://www.reddit.com/r/{subreddit}/{sort}.json?limit={POSTS_PER_REQUEST}'
                if after:
                    url += f'&after={after}'

                data = fetch_reddit_json(url)
                time.sleep(DELAY_BETWEEN_REQUESTS)

                if not data or 'data' not in data:
                    break

                children = data['data'].get('children', [])
                if not children:
                    break

                for child in children:
                    post = child.get('data', {})
                    post_id = post.get('id', '')

                    if post_id in seen_ids:
                        continue
                    if post.get('over_18', False):
                        continue
                    url_img = post.get('url', '')
                    if not is_image_url(url_img):
                        continue
                    title = post.get('title', '')
                    if not is_clean(title):
                        continue

                    seen_ids.add(post_id)
                    posts.append({
                        'id': post_id,
                        'title': title,
                        'url': url_img,
                        'subreddit': subreddit,
                        'score': post.get('score', 0),
                        'permalink': f"https://reddit.com{post.get('permalink', '')}",
                    })

                after = data['data'].get('after')
                if not after:
                    break

    return posts


def download_image(url: str, save_path: Path) -> bool:
    """Скачивает одну картинку."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and 'image' in resp.headers.get('content-type', ''):
            save_path.write_bytes(resp.content)
            return True
    except:
        pass
    return False


def main():

    # === Шаг 1: Сбор ссылок ===
    all_posts = []
    seen_urls = set()

    for subreddit in SUBREDDITS:
        print(f'\n--- r/{subreddit} ---')
        posts = collect_posts_from_subreddit(subreddit)

        # Убираем дубликаты по URL
        new_posts = []
        for p in posts:
            if p['url'] not in seen_urls:
                seen_urls.add(p['url'])
                new_posts.append(p)

        all_posts.extend(new_posts)
        print(f'  Собрано: {len(new_posts)} (всего: {len(all_posts)})')

        if len(all_posts) >= MAX_TOTAL:
            all_posts = all_posts[:MAX_TOTAL]
            print(f'\nДостигнут лимит {MAX_TOTAL}')
            break

    print(f'\nИтого собрано ссылок: {len(all_posts)}')

    # === Шаг 2: Скачивание картинок ===
    metadata_rows = []
    downloaded = 0
    failed = 0

    def process(idx, post):
        ext = post['url'].split('.')[-1].split('?')[0]
        if ext not in ('jpg', 'jpeg', 'png', 'webp'):
            ext = 'jpg'
        filename = f"reddit_{idx:05d}.{ext}"
        save_path = IMAGES_DIR / filename
        ok = download_image(post['url'], save_path)
        if ok:
            return {
                'image_name': filename,
                'title': post['title'],
                'subreddit': post['subreddit'],
                'score': post['score'],
                'source_url': post['url'],
                'permalink': post['permalink'],
                'source': 'reddit',
            }
        return None

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {}
        for idx, post in enumerate(all_posts):
            fut = executor.submit(process, idx, post)
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

    # === Шаг 3: Сохраняем metadata.csv ===
    csv_path = OUTPUT_DIR / 'metadata.csv'
    fieldnames = ['image_name', 'title', 'subreddit', 'score', 'source_url', 'permalink', 'source']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f'Ссылок собрано:  {len(all_posts)}')
    print(f'Скачано:         {downloaded}')
    print(f'Не удалось:      {failed}')
    print(f'Метаданные:      {csv_path}')
    print(f'Картинки:        {IMAGES_DIR}')

    # Статистика по сабреддитам
    print(f'\nПо сабреддитам:')
    from collections import Counter
    sub_counts = Counter(r['subreddit'] for r in metadata_rows)
    for sub, count in sub_counts.most_common():
        print(f'  r/{sub}: {count}')


if __name__ == '__main__':
    main()