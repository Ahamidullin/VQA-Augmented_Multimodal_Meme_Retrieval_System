# Meme Retrieval System

**A VQA-Augmented Multimodal Meme Retrieval System**

Курсовой проект — мультимодальная система поиска мемов с VQA-аннотациями.

## Описание

Интеллектуальный сервис для поиска мемов по текстовому описанию, референсной картинке или их комбинации. Система автоматически анализирует содержание изображений, извлекает семантическое описание и индексирует мемы для точного поиска.

## Архитектура

```
Telegram Bot (пользовательский интерфейс)
        │
        ▼
   Backend API (FastAPI)
        │
        ├── CLIP encoder (текст → эмбеддинг, картинка → эмбеддинг)
        ├── Vector DB (Qdrant/FAISS) — быстрый поиск top-K кандидатов
        ├── BM25 index — текстовый поиск по описаниям и OCR-тексту
        ├── Cross-encoder reranker — переранжирование top-K
        └── VLM (BLIP-2/LLaVA) — генерация описаний к результатам (опционально, GPU)
```

## Режимы поиска

1. **Текстовый** — пользователь описывает мем текстом
2. **По картинке** — пользователь отправляет референсное изображение
3. **Гибридный** — текст + картинка одновременно

## Стек технологий

- **Python 3.11+**
- **CLIP** (OpenAI) — мультимодальные эмбеддинги
- **BLIP-2 / LLaVA** — VQA-аннотации и описания
- **PaddleOCR / EasyOCR** — извлечение текста с картинок
- **FAISS / Qdrant** — векторная база данных
- **FastAPI** — backend API
- **aiogram** — Telegram bot
- **rank-bm25** — текстовый поиск

## Структура проекта

```
meme-retrieval/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/                    # Документация и отчёт
│   └── report_stage1.pdf
├── data/                    # Данные (не в git, только .gitkeep)
│   ├── raw/                 # Сырые скачанные мемы
│   ├── processed/           # Очищенные мемы
│   └── annotations/         # Описания, OCR-тексты, теги
├── notebooks/               # Jupyter notebooks для экспериментов
├── scripts/                 # Скрипты парсинга, аннотации, оценки
│   ├── parse_reddit.py
│   ├── parse_imgflip.py
│   ├── run_ocr.py
│   ├── generate_vqa.py
│   └── build_index.py
├── src/                     # Основной код
│   ├── api/                 # FastAPI backend
│   ├── bot/                 # Telegram bot
│   ├── search/              # Логика поиска (CLIP, BM25, rerank)
│   └── models/              # Обёртки моделей
├── eval/                    # Валидация и метрики
│   ├── validation_set/      # 300 размеченных запросов
│   └── evaluate.py
└── configs/                 # Конфиги
    └── config.yaml
```

## Этапы разработки


## Метрики качества

- **Hit@K** — попал ли релевантный мем в top-K
- **Recall@K** — доля найденных релевантных мемов в top-K
- **MRR@10** — Mean Reciprocal Rank на top-10

## Установка

```bash
git clone https://github.com/<username>/meme-retrieval.git
cd meme-retrieval
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

