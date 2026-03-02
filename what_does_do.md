18888 сырых изображений
         ↓
   EasyOCR + PaddleOCR → OCR текст с картинок
         ↓
   Cleaning/filtering → 16 080 чистых записей
         ↓
   Отбираем 9998 мемов (сначала Telegram, затем Bing)
         ↓
[Stage 1] generate_vqa.py (Qwen2.5-VL-3B)
   → vqa_annotations.jsonl (9998 записей)
   Поля: caption, objects, tone, main_idea
         ↓
[Stage 2] enrich_vqa.py (тот же LLM, 22.9 часа)
   → vqa_annotations_v2.jsonl (9998 записей)
   Добавляет: ocr_normalized, objects_detailed,
              relations, required_context, vqa (6 Q&A)
         ↓
[Stage 3] build_embeddings.py (SentenceTransformer + CLIP)
   Берёт v2, объединяет caption + ocr + objects + vqa ответы
   → text_embeddings.npy  (9998 × 384)   ← sentence-transformer
   → image_embeddings.npy (9998 × 512)   ← CLIP ViT-B/32
   → faiss_text.index    (индекс для текстового поиска)
   → faiss_image.index   (индекс для поиска по картинке)
   → index_metadata.jsonl (маппинг: №строки → имя файла + caption)

Итого каждый мем сейчас имеет: OCR текст + VQA описание + 6 Q&A + relations + два вектора (текст и картинка) в FAISS-индексе.

----

Qwen (один раз)  →  тексты описаний сохранены
                          ↓
      all-MiniLM  →  text FAISS index  (поиск по тексту)
      CLIP        →  image FAISS index (поиск по картинке)
Модели в проекте
1. Qwen2.5-VL-3B (через Ollama) — описание мемов

Смотрит на картинку → генерирует JSON с caption, objects, tone, 

vqa
 и т.д.
Запускался 2 раза: 

generate_vqa.py
 (Stage 1) + 

enrich_vqa.py
 (Stage 2)
Больше не нужен — аннотации уже готовы
2. all-MiniLM-L6-v2 (SentenceTransformer) — текстовые эмбеддинги

Маленькая быстрая модель, только текст
Берёт готовый текст (caption + OCR + vqa ответы) и превращает в вектор 384d
Используется при поиске по тексту: твой запрос → вектор → ищем в FAISS
3. CLIP ViT-B/32 (OpenAI, через transformers) — картиночные эмбеддинги

Понимает и картинки, и текст в одном пространстве
Превращает каждый мем → вектор 512d
Используется при поиске по картинке: твоя картинка → вектор → ищем в FAISS