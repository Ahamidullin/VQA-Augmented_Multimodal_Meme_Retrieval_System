"""
Телеграм-бот для поиска мемов.
Pipeline: bge-m3 encode → FAISS (C + EN_sq + RU_sq) → RRF → top-5
С пагинацией и автообновлением БД.
"""

import os
import json
import re
import logging
import asyncio
import base64
import shutil
import numpy as np
import imagehash
import faiss
from pathlib import Path
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#  конфиг 
BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
EXP_DIR = Path("data/experiments")
NEW_DIR = Path("data/raw/new")
DONE_DIR = Path("data/raw/processed_new")
PAGE_SIZE = 5
MAX_PAGES = 5
UPDATE_INTERVAL = 300  # секунд (5 мин)
MAX_DB_SIZE = 20000    # лимит мемов, после которого включается ротация
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
BAD_WORDS_FILE = Path("configs/bad_words.txt")
PHASH_THRESHOLD = 3
PHASH_SIZE = 16
NSFW_WHITELIST_FILE = Path("configs/nsfw_whitelist.txt")
SCRAPE_INTERVAL = 1800  # секунд (30 мин) — как часто парсить reddit
REDDIT_SUBREDDITS = [
    "memes", "dankmemes", "me_irl", "shitposting",
    "wholesomememes", "ProgrammerHumor", "HistoryMemes",
    "animemebank", "2meirl4meirl",
]
REDDIT_SEEN_FILE = Path("data/raw/.reddit_seen.txt")
REDDIT_MIN_SCORE = 100
REDDIT_LIMIT = 25


def load_word_list(path):
    """Загружает список слов из файла, игнорирует комменты и пустые строки."""
    words = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip().lower().replace("ё", "е")
                if line and not line.startswith("#"):
                    words.add(line)
    return words


def load_bad_words():
    whitelist = load_word_list(NSFW_WHITELIST_FILE)
    bad = load_word_list(BAD_WORDS_FILE)
    return bad - whitelist, whitelist


def check_nsfw(text, bad_words_set, whitelist):
    """Returns matched bad word or None if clean."""
    if not text:
        return None
    words = re.findall(r"[a-zа-яе0-9]+", text.lower().replace("ё", "е"))
    for word in words:
        if word in whitelist:
            continue
        if word in bad_words_set:
            return word
        for bw in bad_words_set:
            if len(bw) >= 3 and word.startswith(bw) and word != bw:
                return f"{word}~{bw}"
    return None


def compute_phash(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        return imagehash.phash(img, hash_size=PHASH_SIZE)
    except Exception:
        return None


bad_words_set, nsfw_whitelist = load_bad_words()
log.info(f"Загружено {len(bad_words_set)} стоп-слов, {len(nsfw_whitelist)} в whitelist")

gpt_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://ai.redivo.ru/v1"
)

# загрузка данных 
records = []
with open(VQA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if not r.get("is_nsfw"):
                records.append(r)

log.info(f"Загружено {len(records)} мемов")

# загрузка индексов 
idx_all = faiss.read_index(str(EXP_DIR / "faiss_C_all.index"))
idx_sq_en = faiss.read_index(str(EXP_DIR / "faiss_D_search_queries.index"))
idx_sq_ru = faiss.read_index(str(EXP_DIR / "faiss_ru_search_queries.index"))

# эмбеддинги из индекса C для MMR
emb_all = idx_all.reconstruct_n(0, idx_all.ntotal)
log.info(f"Эмбеддинги для MMR: {emb_all.shape}")

# загрузка модели
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
log.info("Модель загружена")

# хранилище сессий
user_sessions = {}  # user_id -> {query, fused, page, lang}


# поисковые функции
def detect_lang(text):
    return "ru" if re.search("[а-яА-ЯёЁ]", text) else "en"


def rrf_fusion(index_results, weights, k=60):
    scores = {}
    for (idxs, _), w in zip(index_results, weights):
        for rank, idx in enumerate(idxs):
            if idx < 0:
                continue
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: -x[1])]


def mmr_diversify(candidates, all_embs, threshold=0.80):
    """MMR: убирает из candidates результаты, слишком похожие на уже выбранные.
    all_embs — numpy массив всех эмбеддингов из индекса C."""
    if not candidates:
        return candidates
    selected = [candidates[0]]
    sel_vecs = [all_embs[candidates[0]]]

    for idx in candidates[1:]:
        vec = all_embs[idx]
        # cosine similarity с каждым уже выбранным
        max_sim = max(float(np.dot(vec, sv)) for sv in sel_vecs)
        if max_sim < threshold:
            selected.append(idx)
            sel_vecs.append(vec)

    return selected


def search_full(query):
    """Возвращает полный ранжированный список (до PAGE_SIZE * MAX_PAGES)"""
    lang = detect_lang(query)
    q_emb = model.encode(query, normalize_embeddings=True).reshape(1, -1).astype(np.float32)

    s1, i1 = idx_all.search(q_emb, 100)
    s2, i2 = idx_sq_en.search(q_emb, 100)

    if lang == "ru":
        s3, i3 = idx_sq_ru.search(q_emb, 100)
        results = [(i1[0], s1[0]), (i2[0], s2[0]), (i3[0], s3[0])]
        weights = [0.4, 0.3, 0.3]
    else:
        results = [(i1[0], s1[0]), (i2[0], s2[0])]
        weights = [0.6, 0.4]

    fused = rrf_fusion(results, weights)
    fused = [idx for idx in fused if not records[idx].get("is_duplicate")]
    fused = mmr_diversify(fused, emb_all)[:PAGE_SIZE * MAX_PAGES]
    return fused, lang


async def send_page(message_or_callback, user_id):
    """Отправляет текущую страницу мемов"""
    session = user_sessions.get(user_id)
    if not session:
        return

    page = session["page"]
    fused = session["fused"]
    lang = session["lang"]
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    page_indices = fused[start:end]

    if not page_indices:
        target = message_or_callback if isinstance(message_or_callback, Message) else message_or_callback.message
        await target.answer("Больше результатов нет 😕")
        return

    target = message_or_callback if isinstance(message_or_callback, Message) else message_or_callback.message

    sent = 0
    for i, idx in enumerate(page_indices):
        r = records[idx]
        img_path = Path(r.get("source_path", ""))
        if img_path.exists():
            try:
                photo = FSInputFile(str(img_path))
                num = start + i + 1
                caption = f"#{num} | {r.get('main_idea', '')[:200]}"
                await target.answer_photo(photo=photo, caption=caption)
                sent += 1
            except Exception as e:
                log.warning(f"Не удалось отправить {r.get('filename')}: {e}")

    # кнопка "показать ещё"
    has_more = end < len(fused)
    if has_more and sent > 0:
        kb = InlineKeyboardBuilder()
        kb.button(text="✅ Нашёл!", callback_data="found")
        kb.button(text="🔍 Показать ещё", callback_data="more")
        kb.adjust(2)
        lang_label = "🇷🇺" if lang == "ru" else "🇺🇸"
        await target.answer(
            f"{lang_label} Показано {sent} мемов. Нашёл нужный?",
            reply_markup=kb.as_markup()
        )
    elif sent > 0:
        await target.answer(f"Это все результаты ({end} мемов).")


#  бот 
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "🔍 **Meme Search Bot**\n\n"
        "Отправь текстовый запрос — найду подходящие мемы!\n"
        "Поддерживаю EN и RU запросы.\n\n"
        "Примеры:\n"
        "• `cat programming meme`\n"
        "• `мем про дедлайн`\n"
        "• `distracted boyfriend template`",
        parse_mode="Markdown"
    )


@dp.message(F.text)
async def handle_query(message: Message):
    query = (message.text or "").strip()
    if not query or len(query) < 2:
        await message.answer("Запрос слишком короткий.")
        return

    await message.answer(f"🔍 Ищу: _{query}_...", parse_mode="Markdown")

    try:
        fused, lang = search_full(query)
    except Exception as e:
        log.error(f"Ошибка поиска: {e}")
        await message.answer("❌ Ошибка при поиске.")
        return

    user_sessions[message.from_user.id] = {  # type: ignore[union-attr]
        "query": query,
        "fused": fused,
        "page": 0,
        "lang": lang,
    }

    await send_page(message, message.from_user.id)  # type: ignore[union-attr]


@dp.callback_query(F.data == "more")
async def show_more(callback: CallbackQuery):
    user_id = callback.from_user.id
    session = user_sessions.get(user_id)
    if not session:
        await callback.answer("Сессия истекла. Отправь запрос заново.")
        return

    session["page"] += 1
    await callback.answer()
    await send_page(callback, user_id)


@dp.callback_query(F.data == "found")
async def found_it(callback: CallbackQuery):
    await callback.answer("Отлично! 🎉")
    await callback.message.answer("Рад помочь! Отправь новый запрос когда понадобится 🔍")  # type: ignore[union-attr]
    user_sessions.pop(callback.from_user.id, None)  # type: ignore[union-attr]


@dp.message(F.photo)
async def handle_photo(message: Message):
    """Поиск по картинке или гибрид """
    user_text = (message.caption or "").strip()
    mode = "hybrid" if user_text else "image"
    await message.answer("Анализирую картинку" if mode == "image" else "Гибридный поиск: картинка + текст")

    # скачать фото
    photo = message.photo[-1]  # type: ignore[index]
    file = await bot.get_file(photo.file_id)
    img_bytes = await bot.download_file(file.file_path)  # type: ignore[arg-type]

    # phash входной картинки (для исключения дублей из выдачи)
    img_bytes.seek(0)  # type: ignore[union-attr]
    try:
        query_img = Image.open(img_bytes).convert("RGB")  # type: ignore[arg-type]
        query_hash = imagehash.phash(query_img, hash_size=16)
    except Exception:
        query_hash = None

    # GPT Vision -> описание
    img_bytes.seek(0)  # type: ignore[union-attr]
    b64 = base64.b64encode(img_bytes.read()).decode()  # type: ignore[union-attr]
    try:
        resp = gpt_client.chat.completions.create(
            model="gpt-5.4-mini-fast",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this meme in 2-3 sentences for search. Include: what's shown, the joke/message, any text on image, emotions. Be specific."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}],
            temperature=0.3, max_tokens=200,
        )
        img_caption = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.error(f"GPT Vision ошибка: {e}")
        await message.answer("❌ Не удалось проанализировать картинку.")
        return

    log.info(f"Image caption: {img_caption}")

    try:
        if mode == "hybrid":
            # гибрид: RRF(text_results, image_results)
            fused_text, lang = search_full(user_text)
            fused_image, _ = search_full(img_caption)

            # RRF fusion двух списков
            scores = {}
            for rank, idx in enumerate(fused_text):
                scores[idx] = scores.get(idx, 0.0) + 0.6 / (60 + rank + 1)
            for rank, idx in enumerate(fused_image):
                scores[idx] = scores.get(idx, 0.0) + 0.4 / (60 + rank + 1)
            fused = [idx for idx, _ in sorted(scores.items(), key=lambda x: -x[1])]
            fused = [idx for idx in fused if not records[idx].get("is_duplicate")]
            fused = mmr_diversify(fused, emb_all)[:PAGE_SIZE * MAX_PAGES]
            log.info(f"Hybrid search: text='{user_text}', image caption='{img_caption[:80]}'")
        else:
            # только картинка
            fused, lang = search_full(img_caption)
    except Exception as e:
        log.error(f"Ошибка поиска: {e}")
        await message.answer("❌ Ошибка при поиске.")
        return

    # исключить входную картинку из результатов (phash match)
    if query_hash is not None:
        filtered = []
        for idx in fused:
            r = records[idx]
            try:
                rh = imagehash.phash(Image.open(r.get("source_path", "")).convert("RGB"), hash_size=16)
                if rh - query_hash > 3:
                    filtered.append(idx)
            except Exception:
                filtered.append(idx)
        fused = filtered

    user_sessions[message.from_user.id] = {  # type: ignore[union-attr]
        "query": user_text or img_caption,
        "fused": fused,
        "page": 0,
        "lang": lang if mode == "hybrid" else detect_lang(img_caption),
    }

    await send_page(message, message.from_user.id)  # type: ignore[union-attr]
ANNOTATE_PROMPT = """Analyze this meme. Return JSON with: "caption", "ocr_text", "meme_template", "objects", "tone", "main_idea", "search_queries", "tags", "emotions", "vqa" (3 q/a pairs). Return ONLY valid JSON."""


def rotate_db():
    """Ротация БД: пропорциональное удаление старых мемов по кластерам тегов.
    Сохраняет баланс категорий (предложено научным руководителем)."""
    global records, emb_all
    to_remove = len(records) - MAX_DB_SIZE
    if to_remove <= 0:
        return

    # кластеризация по первому тегу
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, r in enumerate(records):
        tags = r.get("tags", []) or ["other"]
        tag = tags[0] if isinstance(tags, list) else "other"
        clusters[tag].append(i)

    # пропорциональное удаление (старые первыми — они в начале списка)
    remove_ids = set()
    for tag, indices in clusters.items():
        n = max(1, int(to_remove * len(indices) / len(records)))
        remove_ids.update(indices[:n])

    log.info(f"Ротация: удаляем {len(remove_ids)} мемов из {len(clusters)} кластеров")

    # пересобрать records и индексы
    new_records = [r for i, r in enumerate(records) if i not in remove_ids]
    records.clear()
    records.extend(new_records)

    # пересобрать FAISS индексы
    idx_all.reset()
    idx_sq_en.reset()
    idx_sq_ru.reset()

    for r in records:
        all_text = ". ".join([
            r.get("caption", ""), r.get("main_idea", ""),
            r.get("ocr_text", ""), " ".join(r.get("objects", []) or []),
            r.get("tone", ""), r.get("meme_template", ""),
            " ".join(r.get("search_queries", []) or []),
            " ".join(r.get("tags", []) or []), " ".join(r.get("emotions", []) or []),
        ])
        emb = model.encode(all_text, normalize_embeddings=True).reshape(1, -1).astype(np.float32)  # type: ignore
        idx_all.add(emb)  # type: ignore

        sq_en = ", ".join(r.get("search_queries", []) or []) or "empty"
        idx_sq_en.add(model.encode(sq_en, normalize_embeddings=True).reshape(1, -1).astype(np.float32))  # type: ignore

        sq_ru = ", ".join(r.get("search_queries_ru", []) or []) or "empty"
        idx_sq_ru.add(model.encode(sq_ru, normalize_embeddings=True).reshape(1, -1).astype(np.float32))  # type: ignore

    emb_all = idx_all.reconstruct_n(0, idx_all.ntotal)

    # сохранить
    with open(VQA_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    faiss.write_index(idx_all, str(EXP_DIR / "faiss_C_all.index"))
    faiss.write_index(idx_sq_en, str(EXP_DIR / "faiss_D_search_queries.index"))
    faiss.write_index(idx_sq_ru, str(EXP_DIR / "faiss_ru_search_queries.index"))

    log.info(f"Ротация завершена. Осталось: {len(records)} мемов")


def encode_image_b64(path, max_size=512):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def annotate_new_meme(img_path):
    b64 = encode_image_b64(img_path)
    resp = gpt_client.chat.completions.create(
        model="gpt-5.4-mini-fast",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": ANNOTATE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        temperature=0.2, max_tokens=800,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def translate_sq(queries_en):
    if not queries_en:
        return []
    resp = gpt_client.chat.completions.create(
        model="gpt-5.4-mini-fast",
        messages=[{"role": "user", "content": f"Translate to Russian, one per line:\n" + "\n".join(queries_en)}],
        temperature=0.3, max_tokens=200,
    )
    return [q.strip().strip('"\'') for q in (resp.choices[0].message.content or "").strip().split("\n") if q.strip()]


async def auto_update_loop():
    """Фоновая задача: проверяет data/raw/new/ каждые 5 мин
    Для каждого нового мема:
    1. phash дедуп против существующих
    2. GPT аннотация
    3. NSFW/profanity проверка по словарю
    4. Перевод search_queries на RU
    5. bge-m3 кодирование + добавление в 3 FAISS индекса
    """
    NEW_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    existing = {r["filename"] for r in records}

    # собрать phash всех существующих мемов для дедупа
    existing_hashes = []
    log.info("Вычисляю phash существующих мемов для дедупа...")
    for r in records:
        h = compute_phash(r.get("source_path", ""))
        if h is not None:
            existing_hashes.append(h)
    log.info(f"phash готово: {len(existing_hashes)} хешей")

    while True:
        await asyncio.sleep(UPDATE_INTERVAL)
        try:
            new_imgs = [f for f in NEW_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS and f.name not in existing]
            if not new_imgs:
                continue

            log.info(f"Автообновление: {len(new_imgs)} новых мемов")
            added = 0
            skipped_dup = 0
            skipped_nsfw = 0

            for img_path in new_imgs:
                try:
                    # 1. phash дедуп против существующей БД
                    new_hash = compute_phash(img_path)
                    if new_hash is not None:
                        is_dup = any(abs(new_hash - h) <= PHASH_THRESHOLD for h in existing_hashes)
                        if is_dup:
                            skipped_dup += 1
                            shutil.move(str(img_path), str(DONE_DIR / img_path.name))
                            log.info(f"  ~ {img_path.name} (дубль, пропуск)")
                            continue

                    # 2. GPT аннотация
                    ann = annotate_new_meme(img_path)
                    ann["filename"] = img_path.name
                    ann["source_path"] = str(DONE_DIR / img_path.name)
                    ann["source"] = "auto_update"

                    # 3. NSFW/profanity проверка
                    all_text_for_nsfw = " ".join(str(v) for v in ann.values() if isinstance(v, str))
                    nsfw_match = check_nsfw(all_text_for_nsfw, bad_words_set, nsfw_whitelist)
                    if nsfw_match:
                        ann["is_nsfw"] = True
                        ann["nsfw_reason"] = nsfw_match
                        skipped_nsfw += 1
                        shutil.move(str(img_path), str(DONE_DIR / img_path.name))
                        # сохраняем в jsonl но НЕ добавляем в индексы
                        records.append(ann)
                        existing.add(img_path.name)
                        log.info(f"  ! {img_path.name} NSFW ({nsfw_match}), пропуск индексов")
                        continue

                    ann["is_nsfw"] = False
                    ann["nsfw_reason"] = ""

                    # 4. Перевод
                    ann["search_queries_ru"] = translate_sq(ann.get("search_queries", []))

                    # 5. Encode + добавить в индексы
                    def _s(v):
                        """str или list → str"""
                        return " ".join(v) if isinstance(v, list) else str(v or "")

                    all_text = ". ".join([
                        _s(ann.get("caption")), _s(ann.get("main_idea")),
                        _s(ann.get("ocr_text")), _s(ann.get("objects")),
                        _s(ann.get("tone")), _s(ann.get("meme_template")),
                        _s(ann.get("search_queries")),
                        _s(ann.get("tags")), _s(ann.get("emotions")),
                    ])
                    emb = model.encode(all_text, normalize_embeddings=True).reshape(1, -1).astype(np.float32)  # type: ignore
                    idx_all.add(emb)  # type: ignore

                    sq_en = _s(ann.get("search_queries")) or "empty"
                    idx_sq_en.add(model.encode(sq_en, normalize_embeddings=True).reshape(1, -1).astype(np.float32))  # type: ignore

                    sq_ru = _s(ann.get("search_queries_ru")) or "empty"
                    idx_sq_ru.add(model.encode(sq_ru, normalize_embeddings=True).reshape(1, -1).astype(np.float32))  # type: ignore

                    records.append(ann)
                    existing.add(img_path.name)
                    if new_hash is not None:
                        existing_hashes.append(new_hash)
                    shutil.move(str(img_path), str(DONE_DIR / img_path.name))
                    added += 1
                    log.info(f"  + {img_path.name}")

                except Exception as e:
                    log.error(f"  ошибка {img_path.name}: {e}")

            # сохранить на диск
            with open(VQA_FILE, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            faiss.write_index(idx_all, str(EXP_DIR / "faiss_C_all.index"))
            faiss.write_index(idx_sq_en, str(EXP_DIR / "faiss_D_search_queries.index"))
            faiss.write_index(idx_sq_ru, str(EXP_DIR / "faiss_ru_search_queries.index"))

            # обновить emb_all для MMR
            global emb_all
            emb_all = idx_all.reconstruct_n(0, idx_all.ntotal)

            log.info(f"Автообновление: +{added} дубли={skipped_dup} nsfw={skipped_nsfw} всего={len(records)}")

            # ротация по кластерам (если БД превысила лимит)
            if len(records) > MAX_DB_SIZE:
                log.info(f"Ротация: {len(records)} > {MAX_DB_SIZE}, удаляем старые")
                rotate_db()
            
        except Exception as e:
            log.error(f"Ошибка автообновления: {e}")


async def reddit_scrape_loop():
    """Фоновый парсер: каждые SCRAPE_INTERVAL сек скачивает новые мемы с Reddit."""
    import requests as req
    import uuid
    HEADERS = {"User-Agent": "MemeSearchBot/1.0 (coursework project)"}
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    await asyncio.sleep(10)  # дать боту стартануть
    while True:
        try:
            # загрузить seen
            seen = set()
            if REDDIT_SEEN_FILE.exists():
                seen = set(REDDIT_SEEN_FILE.read_text().strip().split("\n"))

            total_saved = 0
            for sub in REDDIT_SUBREDDITS:
                try:
                    url = f"https://www.reddit.com/r/{sub}/hot.json"
                    resp = await asyncio.to_thread(
                        req.get, url,
                        headers=HEADERS,
                        params={"limit": REDDIT_LIMIT},
                        timeout=15
                    )
                    resp.raise_for_status()
                    posts = resp.json()["data"]["children"]
                except Exception as e:
                    log.warning(f"reddit r/{sub}: {e}")
                    await asyncio.sleep(2)
                    continue

                for post in posts:
                    d = post["data"]
                    post_id = d.get("id", "")
                    img_url = d.get("url", "")
                    score = d.get("score", 0)

                    if post_id in seen or score < REDDIT_MIN_SCORE:
                        continue

                    if not any(img_url.lower().endswith(e) for e in IMG_EXTS):
                        if "i.redd.it" in img_url or "i.imgur.com" in img_url:
                            img_url += ".jpg"
                        else:
                            seen.add(post_id)
                            continue

                    try:
                        r = await asyncio.to_thread(
                            req.get, img_url, headers=HEADERS, timeout=10
                        )
                        if r.status_code != 200 or len(r.content) < 5000:
                            seen.add(post_id)
                            continue
                        img = Image.open(BytesIO(r.content))
                        img.verify()
                        img = Image.open(BytesIO(r.content))
                        w, h = img.size
                        if not (200 <= min(w, h) and max(w, h) <= 3000):
                            seen.add(post_id)
                            continue
                    except Exception:
                        seen.add(post_id)
                        continue

                    ext = img_url.split(".")[-1].split("?")[0].lower()
                    if ext not in {"jpg", "jpeg", "png", "gif", "webp"}:
                        ext = "jpg"
                    fname = f"{uuid.uuid4().hex[:12]}.{ext}"
                    dest = NEW_DIR / fname
                    dest.write_bytes(r.content)
                    seen.add(post_id)
                    total_saved += 1

                await asyncio.sleep(2)  # rate limit

            # сохранить seen
            REDDIT_SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            REDDIT_SEEN_FILE.write_text("\n".join(seen))
            log.info(f"🔄 Reddit scrape: скачано {total_saved} новых мемов")
        except Exception as e:
            log.error(f"reddit_scrape_loop ошибка: {e}")

        await asyncio.sleep(SCRAPE_INTERVAL)


async def main():
    log.info("Бот запущен")
    asyncio.create_task(auto_update_loop())
    asyncio.create_task(reddit_scrape_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
