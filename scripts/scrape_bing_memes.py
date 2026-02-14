"""
Meme Scraper via Bing Image Downloader
=======================================
- Downloads memes by diverse search queries (EN + RU)
- Filters by image size (200-2000px min side)
- Deduplicates via perceptual hash (pHash)
- Saves metadata.csv with tags, dimensions, phash
- Consolidates everything into a single flat folder with UUIDs

Usage:
    pip install bing-image-downloader ImageHash Pillow
    python scripts/scrape_bing_memes.py
"""

import os
import csv
import uuid
import shutil
import hashlib
import logging
from pathlib import Path
from PIL import Image
import imagehash
from bing_image_downloader import downloader

# ============================================================
# CONFIG
# ============================================================
# Use project structure
OUTPUT_DIR = Path("data/raw/bing_memes")
IMAGES_DIR = OUTPUT_DIR / "images"
TEMP_DIR = Path("data/raw/bing_temp")         # bing downloader saves here first
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

LIMIT_PER_QUERY = 150                    # limit per query (bing gives ~150 real max usually)
MIN_SIDE_PX = 200                        # minimum dimension
MAX_SIDE_PX = 2000                       # maximum dimension
PHASH_THRESHOLD = 8                      # hamming distance threshold for dedup (8 is standard for strict dedup)
ADULT_FILTER = True                      # bing adult filter on

# ============================================================
# SEARCH QUERIES
# ============================================================
QUERIES = [
    # === English: Popular templates ===
    "drake meme template",
    "distracted boyfriend meme",
    "woman yelling at cat meme",
    "expanding brain meme",
    "change my mind meme",
    "this is fine meme",
    "surprised pikachu meme",
    "stonks meme",
    "uno reverse card meme",
    "galaxy brain meme",
    "always has been meme",
    "is this a pigeon meme",
    "trade offer meme",
    "grim reaper knocking door meme",
    "two buttons meme",
    "patrick star meme",
    "spongebob meme",
    "homer simpson meme",
    "wojak meme",
    "pepe frog meme",
    "doge meme",
    "cheems meme",
    "trollface meme",
    "chad yes meme",

    # === English: Genre / style ===
    "funny meme",
    "dank meme",
    "wholesome meme",
    "dark humor meme",
    "relatable meme",
    "deep fried meme",
    "surreal meme",
    "anti meme",
    "cursed image meme",
    "shitpost meme",
    "bonehurtingjuice meme",
    "okbuddyretard meme",

    # === English: Topic-based ===
    "programming meme",
    "math meme",
    "science meme",
    "school meme",
    "college meme",
    "monday meme",
    "work meme",
    "gaming meme",
    "cat meme funny",
    "dog meme funny",
    "anime meme",
    "movie meme",
    "marvel meme",
    "star wars meme",
    "history meme",
    "food meme",
    "sleep meme",
    "introvert meme",
    "dating meme",
    "gen z meme",

    # === English: Temporal ===
    "best memes 2024",
    "best memes 2023",
    "viral meme",
    "new memes trending",

    # === Russian ===
    "мем смешной",
    "русский мем",
    "мем шаблон",
    "мемы 2024",
    "мемы про школу",
    "мемы про программирование",
    "мемы про котов",
    "мемы жиза",
    "мемы про работу",
    "мем ну ты и",
    "мем типичный",
]

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scrape_memes.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# HELPERS
# ============================================================
def is_valid_image(path: Path) -> bool:
    """Check if file is a valid image we can open."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def passes_size_filter(path: Path) -> bool:
    """Check if image dimensions are within allowed range."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            min_side = min(w, h)
            return MIN_SIDE_PX <= min_side <= MAX_SIDE_PX
    except Exception:
        return False


def compute_phash(path: Path) -> imagehash.ImageHash | None:
    """Compute perceptual hash for an image."""
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image."""
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


def is_duplicate(phash_val: imagehash.ImageHash, seen_hashes: list) -> bool:
    """Check if phash is within PHASH_THRESHOLD of any seen hash."""
    for existing in seen_hashes:
        if abs(phash_val - existing) <= PHASH_THRESHOLD:
            return True
    return False


# ============================================================
# MAIN PIPELINE
# ============================================================
def download_all_queries():
    """Step 1: Download images for all queries into temp dir."""
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting download of {len(QUERIES)} queries...")
    for i, query in enumerate(QUERIES, 1):
        log.info(f"[{i}/{len(QUERIES)}] Downloading: '{query}' (limit={LIMIT_PER_QUERY})")
        try:
            # bing_image_downloader creates a subdir inside output_dir with the query name
            downloader.download(
                query,
                limit=LIMIT_PER_QUERY,
                output_dir=str(TEMP_DIR),
                adult_filter_off=not ADULT_FILTER,
                force_replace=False,
                timeout=30,
                verbose=False
            )
            log.info(f"  Done: '{query}'")
        except Exception as e:
            log.warning(f"  Failed query '{query}': {e}")


def collect_and_filter():
    """Step 2: Collect all downloaded images, filter, deduplicate, save."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IMAGES_DIR.exists():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Starting collection and filtering...")
    
    seen_hashes: list[imagehash.ImageHash] = []
    stats = {
        "total_found": 0,
        "invalid": 0,
        "size_filtered": 0,
        "duplicates": 0,
        "kept": 0,
    }

    # Open metadata CSV
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "id", "filename", "original_name", "query_tag", "width", "height", "phash"
        ])
        writer.writeheader()

        # Walk through all downloaded folders
        # TEMP_DIR contains folders named after queries
        if not TEMP_DIR.exists():
            log.warning("Temp directory not found, nothing to process.")
            return

        for query_dir in sorted(TEMP_DIR.iterdir()):
            if not query_dir.is_dir():
                continue

            query_tag = query_dir.name  # folder name = search query
            # Bing downloader often adds extra extensions or weird filenames, handle gracefully
            image_files = [
                f for f in query_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
            ]

            log.info(f"Processing '{query_tag}': found {len(image_files)} potential images")

            for img_path in image_files:
                stats["total_found"] += 1

                # 1. Validate image
                if not is_valid_image(img_path):
                    stats["invalid"] += 1
                    continue

                # 2. Size filter
                if not passes_size_filter(img_path):
                    stats["size_filtered"] += 1
                    continue

                # 3. Compute phash & dedup
                phash_val = compute_phash(img_path)
                if phash_val is None:
                    stats["invalid"] += 1
                    continue

                if is_duplicate(phash_val, seen_hashes):
                    stats["duplicates"] += 1
                    continue

                seen_hashes.append(phash_val)

                # 4. Copy to output with UUID name
                file_id = uuid.uuid4().hex[:12]
                ext = img_path.suffix.lower()
                new_name = f"{file_id}{ext}"
                dest = IMAGES_DIR / new_name # Save to images subdir

                try:
                    shutil.copy2(img_path, dest)
                except Exception as e:
                    log.error(f"Failed to copy {img_path}: {e}")
                    continue

                w, h = get_image_dimensions(dest)

                writer.writerow({
                    "id": file_id,
                    "filename": new_name,
                    "original_name": img_path.name,
                    "query_tag": query_tag,
                    "width": w,
                    "height": h,
                    "phash": str(phash_val),
                })

                stats["kept"] += 1

                if stats["kept"] % 100 == 0:
                    log.info(f"  ... kept {stats['kept']} images so far")

    log.info("=" * 60)
    log.info("SCRAPING COMPLETE — STATS:")
    log.info(f"  Total found:     {stats['total_found']}")
    log.info(f"  Invalid:         {stats['invalid']}")
    log.info(f"  Size filtered:   {stats['size_filtered']}")
    log.info(f"  Duplicates:      {stats['duplicates']}")
    log.info(f"  KEPT (unique):   {stats['kept']}")
    log.info(f"  Output dir:      {IMAGES_DIR.resolve()}")
    log.info(f"  Metadata:        {METADATA_FILE.resolve()}")
    log.info("=" * 60)


def cleanup_temp():
    """Step 3: Remove temp download directory automatically (since we have copies)."""
    if TEMP_DIR.exists():
        log.info(f"Removing temp directory: {TEMP_DIR}")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            log.warning(f"Could not remove temp dir: {e}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("LOGGING TO: scrape_memes.log")
    log.info(f"Starting meme scraper: {len(QUERIES)} queries x {LIMIT_PER_QUERY} limit")
    log.info(f"Size filter: {MIN_SIDE_PX}-{MAX_SIDE_PX}px | pHash threshold: {PHASH_THRESHOLD}")

    # Step 1: Download
    download_all_queries()

    # Step 2: Filter + Dedup + Save
    collect_and_filter()

    # Step 3: Cleanup
    cleanup_temp()

    log.info("All done!")
    print("Done! Check scrape_memes.log for details.")
