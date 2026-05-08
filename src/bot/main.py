"""
Телеграм-бот для поиска мемов.
Pipeline: bge-m3 encode → FAISS (C + EN_sq + RU_sq) → RRF → top-5
С пагинацией: "Показать ещё" если не нашёл нужное.
"""

import os
import json
import re
import logging
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── конфиг ──
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
VQA_FILE = Path("data/processed/vqa_annotations_v3.jsonl")
EXP_DIR = Path("data/experiments")
PAGE_SIZE = 5
MAX_PAGES = 5  # максимум 25 мемов

# ── загрузка данных ──
records = []
with open(VQA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if not r.get("is_nsfw"):
                records.append(r)

log.info(f"Загружено {len(records)} мемов")

# ── загрузка индексов ──
idx_all = faiss.read_index(str(EXP_DIR / "faiss_C_all.index"))
idx_sq_en = faiss.read_index(str(EXP_DIR / "faiss_D_search_queries.index"))
idx_sq_ru = faiss.read_index(str(EXP_DIR / "faiss_ru_search_queries.index"))

# ── загрузка модели ──
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
log.info("Модель загружена")

# ── хранилище сессий ──
user_sessions = {}  # user_id → {query, fused, page, lang}


# ── поисковые функции ──
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

    fused = rrf_fusion(results, weights)[:PAGE_SIZE * MAX_PAGES]
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
            f"{lang_label} Показано {end}/{len(fused)}. Нашёл нужный мем?",
            reply_markup=kb.as_markup()
        )
    elif sent > 0:
        await target.answer(f"Это все результаты ({end} мемов).")


# ── бот ──
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
    query = message.text.strip()
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

    user_sessions[message.from_user.id] = {
        "query": query,
        "fused": fused,
        "page": 0,
        "lang": lang,
    }

    await send_page(message, message.from_user.id)


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
    await callback.message.answer("Рад помочь! Отправь новый запрос когда понадобится 🔍")
    user_sessions.pop(callback.from_user.id, None)


async def main():
    log.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
