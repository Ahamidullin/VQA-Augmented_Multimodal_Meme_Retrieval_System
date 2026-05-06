Сейчас Предложение
build_embeddings.py + build_embeddings_v2.py + build_index.py → один scripts/build_index.py
vqa_annotations.jsonl + vqa_annotations_v2.jsonl Оставить только _v2 (полная версия)

---
scripts/: scrape_bing_fast, scrape_telegram_stickers, download_hf_memes,
          run_ocr, run_paddle_ocr, clean_easy_ocr, merge_and_clean,
          generate_vqa, enrich_vqa, nsfw_filter, generate_ru_queries
data/:    emb_caption/ocr/keywords/image/caption_ru.npy + faiss_*.index,
          index_metadata.jsonl, vqa_annotations_v2.jsonl, ru_translations.json
eval/:    language_experiment_queries.json, final_pipeline_results.json, evaluate.py
notebooks/: language_experiment, build_embeddings_bge, search_pipeline,
            ru_index_after_marshrutization_exp

----

1. Что перезапустить после нового набора запросов
Индексы и эмбеддинги мемов НЕ меняются — меняются только запросы. Порядок:

1. Подготовить новый JSON (5 EN + 5 RU на мем)
   → eval/language_experiment_queries_v3.json
1. Перезапустить search_pipeline.ipynb:
   - Ячейка 4 (Pre-encode запросов) — encode новых 500 EN + 500 RU
   - Ячейка 5 (Evaluate) — финальная таблица
1. Перезапустить ru_index.ipynb:
   - Только ячейки 5-7 (evaluate с RU индексом)
Всё. Индексы, BM25, переводы — не трогаем.
