"""
Telegram Sticker Pack Downloader (Structure by Folders)
=======================================================
- Uses Telegram Bot API
- Downloads static stickers from public sticker packs
- Converts .webp → .png (RGBA)
- Saves to: data/raw/telegram_stickers/<PackName>/<uuid>.png
- Saves metadata.csv (id, filename, pack_name, emoji, dimensions)

Usage:
    pip install requests Pillow
    python scripts/scrape_telegram_stickers.py
"""

import os
import csv
import uuid
import time
import logging
import requests
import shutil
from pathlib import Path
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = "8539149130:AAFNOPLb1zED6lIhsNGq8gmmGdVfXl-XBXU"

OUTPUT_DIR = Path("data/raw/telegram_stickers")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

# Delay between requests to avoid rate limiting
REQUEST_DELAY = 0.2

# ============================================================
# STICKER PACKS
# ============================================================
# Source list + User provided + Extra popular ones
RAW_PACK_LIST = [
    # === User Provided ===
    "sahasraraopens_by_fStikBot",
    "fruits_eating_fruits",
    "LolAnimals4",
    "vryadly1",
    "lydvok",
    "tytvsechannel2",
    "hamstersad",
    "tuti_patuti",
    "QZTOGTWNPN_by_stikeri_stikeri_bot",
    "minionsareperfect",
    "set_1895_by_makestick3_bot",
    "Formidable_Brown_Ox_by_fStikBot",
    "phichit2020",
    "ProhorOneLove_by_fStikBot",
    "userpack7230517_by_stickrubot",
    "epwpew",
    "chauvez",
    "kapsoajshajsosbvx_by_fStikBot",
    "beliuchel",
    "SRMTSQDBJI_by_stikeri_stikeri_bot",
    "normtipDOGGY",
    "kitikinianiania",
    "komustickers",
    "Drill_trap",
    "luntikpidoras",
    "cringebymasha_by_fStikBot",
    "Manahontana",
    "ckdesiwd_by_stickrubot",
    "pk_1926380_by_Ctikerubot",
    "nagievpapa_by_fStikBot",
    "trychatgpt_ru",
    "yes_okda",
    "FeelFuckmemes_by_fStikBot",
    "zxcpudgetruededinsaidkanekiken",
    "tupayaa_by_fStikBot",
    "WISTICKERScomCLOWN",
    "vwd88",
    "smjksakqk_by_fStikBot",
    "shlrona",
    "SkaldDealSex_by_fStikBot",
    "bananafoncrushminion",
    "nskcho_by_fStikBot",
    "stinkySQUAD",
    "Klubni4ka_by_fStikBot",
    "cursedemoticon2",
    "mrrzzzmssk",
    "babypigschyz",
    "HFVNZZAJIF_by_stikeri_stikeri_bot",

    # === Default/Popular ===
    "PepeRus", "PepeTheF", "peabornt", "PepesetNew", "Pepe_the_Frog_Pack", "FrogPepe1",
    "maboroshi", "MemeManpack", "Memespack1", "MemeCats", "memesrussia", "FunnyMemes2020",
    "dank_meme_stickers", "CatMemes", "SadCat", "PopCat", "CatVibing", "CATPACKS",
    "WojakStickers", "Wojak_Pack", "ChadWojak", "Doomer_pack", "DogeStickers",
    "CheemsStickers", "DogeePack", "AmogusStickers", "AmongUsPack", "russiamemes",
    "zhizamemes", "memiRUS", "StickersMemeRu", "russkie_memy", "memes_ru_pack",
    "AnimeMemes1", "AnimeStickersMeme", "StonksPack", "CursedEmojis", "GigaChadStickers",
    "SkullEmoji", "Bruhstickers", "PhilosophyMemes", "TrashTaste", "NPC_meme",
    "RickRollPack", "SigmaGrindset", "BasedStickers", "SussyBaka", "MemeDogs",
    "ShrekMemes", "MonkeyMemes", "SkeletonMemes",
    
    # === Classics (VK style imports) ===
    "Senya_vk", "Diggy_vk", "Persik_vk", "Spotty_vk", "Nichosi_vk",
    
    # === More Random ===
    "arcane_jinz_vi", "breaking_bad_stickers", "spongebob_memes", "shrek_is_love",
    "postirony_pack", "yoba_face", "k_on_stickers", "evangelion_memes",
    "jojo_memes", "berserk_memes", "gachimuchi_stickers"
]

# Dedup list
STICKER_PACKS = sorted(list(set(RAW_PACK_LIST)))

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scrape_stickers.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# TELEGRAM BOT API HELPERS
# ============================================================
def get_sticker_set(name: str) -> dict | None:
    """Fetch sticker set info via Bot API."""
    try:
        resp = requests.get(
            f"{BASE_URL}/getStickerSet",
            params={"name": name},
            timeout=10,
        )
        data = resp.json()
        if data.get("ok"):
            return data["result"]
        else:
            # log.warning(f"  API error for '{name}': {data.get('description', 'unknown')}")
            return None
    except Exception as e:
        log.warning(f"  Request failed for '{name}': {e}")
        return None


def get_file_path(file_id: str) -> str | None:
    """Get file path on Telegram servers."""
    try:
        resp = requests.get(
            f"{BASE_URL}/getFile",
            params={"file_id": file_id},
            timeout=10,
        )
        data = resp.json()
        if data.get("ok"):
            return data["result"]["file_path"]
        return None
    except Exception:
        return None


def download_file(file_path: str, dest: Path) -> bool:
    """Download file from Telegram servers."""
    try:
        url = f"{FILE_URL}/{file_path}"
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            return True
        return False
    except Exception:
        return False


# ============================================================
# MAIN
# ============================================================
def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    # Remove old flat 'images' folder if it exists, to avoid confusion
    flat_images_dir = OUTPUT_DIR / "images"
    if flat_images_dir.exists():
        log.info(f"Removing old flat directory: {flat_images_dir}...")
        try:
            shutil.rmtree(flat_images_dir)
        except Exception as e:
            log.warning(f"Could not remove old dir: {e}")

    stats = {
        "packs_processed": 0,
        "packs_failed": 0,
        "stickers_total": 0,
        "stickers_animated_skipped": 0,
        "stickers_download_failed": 0,
        "stickers_downloaded": 0,
    }

    # Open CSV in append mode if exists, else write header
    # We overwrite metadata completely since we are restructuring folders
    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "id", "filename", "pack_name", "pack_short_name",
            "emoji", "width", "height", "sticker_type",
        ])
        writer.writeheader()

        for i, pack_name in enumerate(STICKER_PACKS, 1):
            log.info(f"[{i}/{len(STICKER_PACKS)}] Fetching pack: {pack_name}")

            sticker_set = get_sticker_set(pack_name)
            if sticker_set is None:
                stats["packs_failed"] += 1
                continue

            stats["packs_processed"] += 1
            pack_title = sticker_set.get("title", pack_name)
            stickers = sticker_set.get("stickers", [])
            sticker_type = sticker_set.get("sticker_type", "regular")
            
            # Create folder for pack
            pack_folder_name = "".join(x for x in pack_name if x.isalnum() or x in "_-")
            pack_dir = OUTPUT_DIR / pack_folder_name
            pack_dir.mkdir(exist_ok=True)

            log.info(f"  Pack '{pack_title}': {len(stickers)} stickers -> {pack_dir.name}/")

            for sticker in stickers:
                stats["stickers_total"] += 1

                # Skip animated and video stickers
                is_animated = sticker.get("is_animated", False)
                is_video = sticker.get("is_video", False)

                if is_animated or is_video:
                    stats["stickers_animated_skipped"] += 1
                    continue

                file_id = sticker.get("file_id")
                emoji = sticker.get("emoji", "")

                if not file_id:
                    continue

                # Get file path
                tg_file_path = get_file_path(file_id)
                if not tg_file_path:
                    stats["stickers_download_failed"] += 1
                    time.sleep(REQUEST_DELAY)
                    continue

                # Download
                uid = uuid.uuid4().hex[:12]
                temp_path = pack_dir / f"{uid}.webp"

                if not download_file(tg_file_path, temp_path):
                    stats["stickers_download_failed"] += 1
                    continue

                # Convert webp → png
                png_path = pack_dir / f"{uid}.png"
                try:
                    with Image.open(temp_path) as img:
                        img = img.convert("RGBA")
                        w, h = img.size
                        img.save(png_path, "PNG")
                    # Clean up webp
                    temp_path.unlink(missing_ok=True)
                except Exception as e:
                    temp_path.unlink(missing_ok=True)
                    stats["stickers_download_failed"] += 1
                    continue

                # Metadata - store relative path in filename
                rel_path = f"{pack_folder_name}/{uid}.png"
                
                writer.writerow({
                    "id": uid,
                    "filename": rel_path,
                    "pack_name": pack_title,
                    "pack_short_name": pack_name,
                    "emoji": emoji,
                    "width": w,
                    "height": h,
                    "sticker_type": sticker_type,
                })

                stats["stickers_downloaded"] += 1
                time.sleep(REQUEST_DELAY)

            log.info(f"  Total downloaded so far: {stats['stickers_downloaded']}")

    log.info("=" * 60)
    log.info("STICKER DOWNLOAD COMPLETE — STATS:")
    log.info(f"  Packs processed:      {stats['packs_processed']}")
    log.info(f"  Packs failed:         {stats['packs_failed']}")
    log.info(f"  Stickers total:       {stats['stickers_total']}")
    log.info(f"  Animated (skipped):   {stats['stickers_animated_skipped']}")
    log.info(f"  DOWNLOADED:           {stats['stickers_downloaded']}")
    log.info(f"  Output:               {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
