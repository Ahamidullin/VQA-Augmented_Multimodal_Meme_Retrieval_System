"""
Clean EasyOCR results.
Filters metadata_ocr.csv based on bad words, text length, and confidence.
Saves filtered list to data/processed/metadata_clean_step1.csv.
"""

import csv
import logging
from pathlib import Path
from thefuzz import fuzz

# Конфигурация
INPUT_FILE = Path("data/processed/metadata_ocr.csv")
OUTPUT_FILE = Path("data/processed/metadata_clean_step1.csv")
BAD_WORDS_FILE = Path("configs/bad_words.txt")

MIN_CONFIDENCE = 0.4
MIN_LENGTH = 3
MAX_LENGTH = 400
FUZZY_THRESHOLD = 85  # Similarity score (0-100)

# Логгер
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def load_bad_words(path):
    if not path.exists():
        log.warning(f"Bad words file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

def contains_bad_word(text, bad_words):
    text_lower = text.lower()
    
    # 1. Exact match (fast)
    for word in bad_words:
        if word in text_lower:
            return True, word

    # 2. Fuzzy match (slower but catches typos)
    # We check each word against bad words list
    text_words = text_lower.split()
    for t_word in text_words:
        for b_word in bad_words:
            ratio = fuzz.ratio(t_word, b_word)
            if ratio >= FUZZY_THRESHOLD:
                return True, b_word
                
    return False, None

def main():
    log.info("Loading bad words...")
    bad_words = load_bad_words(BAD_WORDS_FILE)
    log.info(f"Loaded {len(bad_words)} bad words.")

    stats = {
        "total": 0,
        "kept": 0,
        "removed_confidence": 0,
        "removed_length": 0,
        "removed_bad_word": 0,
    }

    if not INPUT_FILE.exists():
        log.error(f"Input file not found: {INPUT_FILE}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            stats["total"] += 1
            text = row.get("ocr_text", "")
            try:
                conf = float(row.get("confidence", 0.0))
            except ValueError:
                conf = 0.0

            # 1. Confidence filter
            # Logic: If text is heavily confident but contains bad words -> remove.
            # If confidence is low, we treat it as "no text" (safe to keep image, but ignore text).
            
            # 2. Length filter & Empty text handling
            if len(text) < MIN_LENGTH:
                # Too short or empty -> Keep image (might be visual meme)
                # But verify it's not a short bad word (e.g. "sex")
                if len(text) > 0:
                     is_bad, _ = contains_bad_word(text, bad_words)
                     if is_bad:
                         stats["removed_bad_word"] += 1
                         continue
                
                # If safe short text or empty -> KEEP
                writer.writerow(row)
                stats["kept"] += 1
                continue
            
            if len(text) > MAX_LENGTH:
                stats["removed_length"] += 1
                continue

            # 3. Bad words filter (fuzzy check) for valid text
            text_lower = text.lower()
            is_bad = False

            
            # Simple direct check
            for bad in bad_words:
                if bad in text_lower:
                    is_bad = True
                    break
            
            # If still clean, try fuzzy
            if not is_bad:
                # Tokenize text
                tokens = text_lower.split()
                for token in tokens:
                   for bad in bad_words:
                       if fuzz.ratio(token, bad) > FUZZY_THRESHOLD:
                           is_bad = True
                           break
                   if is_bad: break
            
            if is_bad:
                stats["removed_bad_word"] += 1
                continue

            writer.writerow(row)
            stats["kept"] += 1

    log.info("Очистка завершена")
    log.info(f"Обработано: {stats['total']}, сохранено: {stats['kept']}")
    log.info(f"Удалено: conf={stats['removed_confidence']}, length={stats['removed_length']}, bad_word={stats['removed_bad_word']}")
    log.info(f"Результат: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
