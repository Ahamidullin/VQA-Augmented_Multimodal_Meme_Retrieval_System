"""
Fast Bing Meme Scraper
- Uses 'requests' with proper User-Agent to avoid 403 blocks (Reddit, etc.)
- Parses Bing Images query results
- Fast timeouts to skip locking urls
- Dedup & Size filter included

Usage:
    pip install requests pillow imagehash
    python scripts/scrape_bing_fast.py
"""

import os
import re
import csv
import uuid
import shutil
import logging
import requests
import imagehash
from pathlib import Path
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# Конфигурация
OUTPUT_DIR = Path("data/raw/bing_memes")
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

LIMIT_PER_QUERY = 120                    # Target images per query
MAX_WORKERS = 10                         # Parallel downloads
MIN_SIDE_PX = 200
MAX_SIDE_PX = 2000
PHASH_THRESHOLD = 8
ADULT_FILTER = True

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}

QUERIES = [
    # English
    "drake meme template", "distracted boyfriend meme", "woman yelling at cat meme",
    "expanding brain meme", "change my mind meme", "this is fine meme",
    "surprised pikachu meme", "stonks meme", "uno reverse card meme",
    "galaxy brain meme", "always has been meme", "is this a pigeon meme",
    "trade offer meme", "grim reaper knocking door meme", "two buttons meme",
    "patrick star meme", "spongebob meme", "homer simpson meme",
    "wojak meme", "pepe frog meme", "doge meme", "cheems meme",
    "trollface meme", "chad yes meme", "funny meme", "dank meme",
    "wholesome meme", "dark humor meme", "relatable meme", "deep fried meme",
    "surreal meme", "anti meme", "cursed image meme", "shitpost meme",
    "bonehurtingjuice meme", "okbuddyretard meme", "programming meme",
    "math meme", "science meme", "school meme", "college meme",
    "monday meme", "work meme", "gaming meme", "cat meme funny",
    "dog meme funny", "anime meme", "movie meme", "marvel meme",
    "star wars meme", "history meme", "food meme", "sleep meme",
    "introvert meme", "dating meme", "gen z meme", "best memes 2024",
    "viral meme",
    # Russian
    "мем смешной", "русский мем", "мем шаблон", "мемы 2024",
    "мемы про школу", "мемы про программирование", "мемы про котов",
    "мемы жиза", "мемы про работу", "мем ну ты и", "мем типичный",
]

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scrape_fast.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)




def get_bing_image_links(query, limit=100):
    """Parses Bing Image Search (async api) to get direct image links."""
    links = set()
    page_counter = 0
    while len(links) < limit:
        url = f"https://www.bing.com/images/async?q={query}&first={page_counter}&count=35&adlt={'on' if ADULT_FILTER else 'off'}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            
            # Regex to extract 'murl' (media url)
            # murl matches: murl&quot;:&quot;(.*?)&quot;
            new_links = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
            
            if not new_links:
                break
                
            for link in new_links:
                if len(links) >= limit:
                    break
                links.add(link)
                
            page_counter += 35
            
        except Exception as e:
            log.warning(f"Error fetching bing page for '{query}': {e}")
            break
            
    log.info(f"Found {len(links)} links for '{query}'")
    return list(links)


def download_image(url, query_tag, seen_hashes):
    """Downloads, verifies, and saves a single image."""
    try:
        # Download with timeout and headers to pass blocks
        resp = requests.get(url, headers=HEADERS, timeout=4)
        if resp.status_code != 200:
            return None

        # Helper to read content
        content = resp.content
        if len(content) < 1000:  # Skip tiny files/errors
            return None

        # Verify image
        try:
            img = Image.open(BytesIO(content))
            img.verify()
            # Re-open for processing since verify closes it
            img = Image.open(BytesIO(content))
        except Exception:
            return None

        # Check dimensions
        w, h = img.size
        min_side = min(w, h)
        if not (MIN_SIDE_PX <= min_side <= MAX_SIDE_PX):
            return None
            
        # Deduplicate
        try:
             phash_val = imagehash.phash(img.convert("RGB"))
        except Exception:
            return None # hashing failed

        # Thread-safe check? (We run queries sequentially, but downloads in parallel)
        # For simplicity, we'll allow slight race conditions in hashes or lock.
        # Since we append to list, it's atomic in CPython, but 'in' check isn't.
        # Let's just check. Minimal duplicates are fine.
        is_dup = False
        for h_exist in seen_hashes:
            if abs(phash_val - h_exist) <= PHASH_THRESHOLD:
                is_dup = True
                break
        
        if is_dup:
            return None
            
        # Add to seen (approximate due to threads)
        seen_hashes.append(phash_val)

        # Save
        file_id = uuid.uuid4().hex[:12]
        # Guess extension from header or PIL
        ext = img.format.lower() if img.format else "jpg"
        if ext == "jpeg": ext = "jpg"
        new_name = f"{file_id}.{ext}"
        dest_path = IMAGES_DIR / new_name
        
        # Save real bytes
        with open(dest_path, "wb") as f:
            f.write(content)
            
        return {
            "id": file_id,
            "filename": new_name,
            "original_name": url.split("/")[-1][:50], # truncated original name
            "query_tag": query_tag,
            "width": w,
            "height": h,
            "phash": str(phash_val),
        }

    except Exception as e:
        # log.debug(f"Failed {url}: {e}")
        return None


def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IMAGES_DIR.exists():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
    # metadata CSV init
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "filename", "original_name", "query_tag", "width", "height", "phash"])
            writer.writeheader()

    seen_hashes = [] # In-memory dedup across queries
    
    total_saved = 0

    # Sequential queries, threaded downloads
    for i, query in enumerate(QUERIES, 1):
        log.info(f"[{i}/{len(QUERIES)}] Querying: '{query}'")
        links = get_bing_image_links(query, limit=LIMIT_PER_QUERY)
        
        if not links:
            continue
            
        success_count = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all downloads
            futures = [executor.submit(download_image, url, query, seen_hashes) for url in links]
            
            for future in futures:
                res = future.result()
                if res:
                    success_count += 1
                    total_saved += 1
                    # Append metadata immediately
                    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["id", "filename", "original_name", "query_tag", "width", "height", "phash"])
                        writer.writerow(res)
                        
        log.info(f"  Saved {success_count} images for '{query}'. Total so far: {total_saved}")
        
    log.info(f"Done! Total saved: {total_saved}")

if __name__ == "__main__":
    main()
