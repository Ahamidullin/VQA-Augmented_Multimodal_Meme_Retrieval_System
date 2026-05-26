"""скачивание стикеров из telegram через bot api
конвертирует webp -> png, сохраняет в data/raw/telegram_stickers/
пропускает анимированные и видео стикеры
"""

import csv
import uuid
import time
import requests
import shutil
from pathlib import Path
from PIL import Image

BOT_TOKEN = "8539149130:AAFNOPLb1zED6lIhsNGq8gmmGdVfXl-XBXU"

OUTPUT_DIR = Path("data/raw/telegram_stickers")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

REQUEST_DELAY = 0.2

RAW_PACK_LIST = [
    "sahasraraopens_by_fStikBot", "fruits_eating_fruits", "LolAnimals4", "vryadly1",
    "lydvok", "tytvsechannel2", "hamstersad", "tuti_patuti",
    "QZTOGTWNPN_by_stikeri_stikeri_bot", "minionsareperfect", "set_1895_by_makestick3_bot",
    "Formidable_Brown_Ox_by_fStikBot", "phichit2020", "ProhorOneLove_by_fStikBot",
    "userpack7230517_by_stickrubot", "epwpew", "chauvez",
    "kapsoajshajsosbvx_by_fStikBot", "beliuchel", "SRMTSQDBJI_by_stikeri_stikeri_bot",
    "normtipDOGGY", "kitikinianiania", "komustickers", "Drill_trap", "luntikpidoras",
    "cringebymasha_by_fStikBot", "Manahontana", "ckdesiwd_by_stickrubot",
    "pk_1926380_by_Ctikerubot", "nagievpapa_by_fStikBot", "trychatgpt_ru", "yes_okda",
    "FeelFuckmemes_by_fStikBot", "zxcpudgetruededinsaidkanekiken", "tupayaa_by_fStikBot",
    "WISTICKERScomCLOWN", "vwd88", "smjksakqk_by_fStikBot", "shlrona",
    "SkaldDealSex_by_fStikBot", "bananafoncrushminion", "nskcho_by_fStikBot",
    "stinkySQUAD", "Klubni4ka_by_fStikBot", "cursedemoticon2", "mrrzzzmssk",
    "babypigschyz", "HFVNZZAJIF_by_stikeri_stikeri_bot",
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
    "Senya_vk", "Diggy_vk", "Persik_vk", "Spotty_vk", "Nichosi_vk",
    "arcane_jinz_vi", "breaking_bad_stickers", "spongebob_memes", "shrek_is_love",
    "postirony_pack", "yoba_face", "k_on_stickers", "evangelion_memes",
    "jojo_memes", "berserk_memes", "gachimuchi_stickers"
]

STICKER_PACKS = sorted(list(set(RAW_PACK_LIST)))


def get_sticker_set(name):
    try:
        resp = requests.get(f"{BASE_URL}/getStickerSet", params={"name": name}, timeout=10)
        data = resp.json()
        if data.get("ok"):
            return data["result"]
        return None
    except Exception:
        return None


def get_file_path(file_id):
    try:
        resp = requests.get(f"{BASE_URL}/getFile", params={"file_id": file_id}, timeout=10)
        data = resp.json()
        if data.get("ok"):
            return data["result"]["file_path"]
        return None
    except Exception:
        return None


def download_file(file_path, dest):
    try:
        url = f"{FILE_URL}/{file_path}"
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            return True
        return False
    except Exception:
        return False


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

flat_images_dir = OUTPUT_DIR / "images"
if flat_images_dir.exists():
    try:
        shutil.rmtree(flat_images_dir)
    except Exception:
        pass

stats = {
    "packs_ok": 0,
    "packs_failed": 0,
    "total": 0,
    "skipped_animated": 0,
    "failed": 0,
    "downloaded": 0,
}

fieldnames = ["id", "filename", "pack_name", "pack_short_name", "emoji", "width", "height", "sticker_type"]

with open(METADATA_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, pack_name in enumerate(STICKER_PACKS, 1):
        sticker_set = get_sticker_set(pack_name)
        if sticker_set is None:
            stats["packs_failed"] += 1
            continue

        stats["packs_ok"] += 1
        pack_title = sticker_set.get("title", pack_name)
        stickers = sticker_set.get("stickers", [])
        sticker_type = sticker_set.get("sticker_type", "regular")

        pack_folder_name = ""
        for x in pack_name:
            if x.isalnum() or x in "_-":
                pack_folder_name += x
        pack_dir = OUTPUT_DIR / pack_folder_name
        pack_dir.mkdir(exist_ok=True)

        for sticker in stickers:
            stats["total"] += 1
            if sticker.get("is_animated") or sticker.get("is_video"):
                stats["skipped_animated"] += 1
                continue
            file_id = sticker.get("file_id")
            emoji = sticker.get("emoji", "")
            if not file_id:
                continue
            tg_file_path = get_file_path(file_id)
            if not tg_file_path:
                stats["failed"] += 1
                time.sleep(REQUEST_DELAY)
                continue
            uid = uuid.uuid4().hex[:12]
            temp_path = pack_dir / f"{uid}.webp"
            if not download_file(tg_file_path, temp_path):
                stats["failed"] += 1
                continue
            png_path = pack_dir / f"{uid}.png"
            try:
                with Image.open(temp_path) as img:
                    img = img.convert("RGBA")
                    w, h = img.size
                    img.save(png_path, "PNG")
                temp_path.unlink(missing_ok=True)
            except Exception:
                temp_path.unlink(missing_ok=True)
                stats["failed"] += 1
                continue
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
            stats["downloaded"] += 1
            time.sleep(REQUEST_DELAY)

        print(f"{i}/{len(STICKER_PACKS)} {pack_name} downloaded={stats['downloaded']}")

print(f"готово пакеты ok={stats['packs_ok']} failed={stats['packs_failed']} стикеры={stats['downloaded']}/{stats['total']}")
