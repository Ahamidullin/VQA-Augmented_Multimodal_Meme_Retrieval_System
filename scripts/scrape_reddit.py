"""парсер мемов с reddit
скачивает картинки из мем-сабреддитов через json api (без ключа)
фильтрует по score, размеру, дедуплицирует по phash
кладёт в data/raw/new/ для подхвата auto_update_loop()

"""

import argparse
import uuid
import time
import requests
import imagehash
from pathlib import Path
from PIL import Image
from io import BytesIO

OUTPUT_DIR = Path("data/raw/new")
SEEN_FILE = Path("data/raw/.reddit_seen.txt")

DEFAULT_SUBREDDITS = [
    "memes", "dankmemes", "me_irl", "shitposting",
    "wholesomememes", "ProgrammerHumor", "HistoryMemes",
    "animemebank", "2meirl4meirl",
]

HEADERS = {
    "User-Agent": "MemeSearchBot/1.0 (coursework project; contact: none)",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MIN_SIDE_PX = 200
MAX_SIDE_PX = 3000
PHASH_SIZE = 16
PHASH_THRESHOLD = 8
REQUEST_DELAY = 1.5  # reddit rate limit 1 req/sec


def load_seen():
    if SEEN_FILE.exists():
        return set(SEEN_FILE.read_text().strip().split("\n"))
    return set()


def save_seen(seen):
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text("\n".join(seen))


def fetch_posts(subreddit, sort="hot", limit=50, after=None):
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": min(limit, 100)}
    if after:
        params["after"] = after
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        posts = data["data"]["children"]
        next_after = data["data"].get("after")
        return posts, next_after
    except Exception as e:
        print(f"  ошибка запроса r/{subreddit}: {e}")
        return [], None


def download_image(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        content = resp.content
        if len(content) < 5000:
            return None
        img = Image.open(BytesIO(content))
        img.verify()
        img = Image.open(BytesIO(content))
        w, h = img.size
        if not (MIN_SIDE_PX <= min(w, h) and max(w, h) <= MAX_SIDE_PX):
            return None
        return content, img
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="scrape memes from reddit")
    parser.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS)
    parser.add_argument("--limit", type=int, default=25, help="posts per subreddit")
    parser.add_argument("--min-score", type=int, default=100, help="minimum upvote score")
    parser.add_argument("--sort", choices=["hot", "top", "new"], default="hot")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    seen_hashes = []
    total_saved = 0
    total_skipped = 0

    for sub in args.subreddits:
        print(f"r/{sub} ({args.sort}, limit={args.limit}, min_score={args.min_score})")
        posts, _ = fetch_posts(sub, sort=args.sort, limit=args.limit)
        time.sleep(REQUEST_DELAY)

        sub_saved = 0
        for post in posts:
            d = post["data"]
            post_id = d.get("id", "")
            url = d.get("url", "")
            score = d.get("score", 0)

            # уже видели
            if post_id in seen:
                total_skipped += 1
                continue

            # фильтр по score
            if score < args.min_score:
                continue

            # только картинки
            if not any(url.lower().endswith(ext) for ext in IMAGE_EXTS):
                # i.redd.it без расширения
                if "i.redd.it" in url and not any(url.endswith(e) for e in IMAGE_EXTS):
                    url += ".jpg"
                elif "i.imgur.com" in url and not any(url.endswith(e) for e in IMAGE_EXTS):
                    url += ".jpg"
                else:
                    continue

            result = download_image(url)
            if result is None:
                seen.add(post_id)
                continue

            content, img = result

            # phash дедуп
            try:
                ph = imagehash.phash(img.convert("RGB"), hash_size=PHASH_SIZE)
                is_dup = any(abs(ph - h) <= PHASH_THRESHOLD for h in seen_hashes)
                if is_dup:
                    seen.add(post_id)
                    continue
                seen_hashes.append(ph)
            except Exception:
                pass

            # сохраняем
            ext = url.split(".")[-1].split("?")[0].lower()
            if ext not in {"jpg", "jpeg", "png", "gif", "webp"}:
                ext = "jpg"
            filename = f"{uuid.uuid4().hex[:12]}.{ext}"
            dest = OUTPUT_DIR / filename
            dest.write_bytes(content)

            seen.add(post_id)
            sub_saved += 1
            total_saved += 1

        print(f"  сохранено: {sub_saved}")
        time.sleep(REQUEST_DELAY)

    save_seen(seen)
    print(f"\nитого: сохранено={total_saved} пропущено={total_skipped}")
    print(f"файлы в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
