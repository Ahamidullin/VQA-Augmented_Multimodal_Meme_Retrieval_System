"""
Merge PaddleOCR results and filter bad content.
This script consolidates scattered 'ocr_paddle.csv' files into a single clean dataset.
"""

import csv
import logging
from pathlib import Path
from tqdm import tqdm
from thefuzz import fuzz


ROOT_DIR = Path("data") # Where to search for ocr_paddle.csv
OUTPUT_FILE = Path("data/processed/final_dataset_text.csv")
BAD_WORDS_FILE = Path("configs/bad_words.txt")

# Filter settings
MIN_CONFIDENCE = 0.6 # для пустых где нет текста или 60
MIN_LENGTH = 3
FUZZY_THRESHOLD = 85

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def load_bad_words(path):
    if not path.exists():
        log.warning(f"Bad words file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]

def collect_all_paddle_files(root_dir):

    paddle_files = []
    for file in root_dir.rglob("ocr_paddle.csv"):
        paddle_files.append(file)
    
    return paddle_files

def is_bad_text(text, bad_words):
    """
    TODO: Проверь текст на наличие плохих слов.
    1. Если текст пустой -> False (не плохой, просто пустой).
    2. Если точное совпадение слова из bad_words в тексте -> True.
    3. Если fuzz.ratio(слово, bad_word) > FUZZY_THRESHOLD -> True.
    
    Верни (is_bad: bool, reason: str).
    Reason - это найденное плохое слово (для логов).
    """
    if not text:
        return False, None
        
    text_lower = text.lower()
    
    # можно  fuzz.ratio(слово, bad_word) > FUZZY_THRESHOLD  но дольше пока так
    
    for bw in bad_words:
        if bw in text_lower:
            return True, bw

    return False, None



def main():
    bad_words = load_bad_words(BAD_WORDS_FILE)
    log.info(f"Loaded {len(bad_words)} bad words.")

    paddle_files = collect_all_paddle_files(ROOT_DIR)
    log.info(f"Found {len(paddle_files)} files to merge.")

    # Prepare output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "kept": 0, "removed": 0}

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["filename", "source_path", "ocr_text", "confidence", "source_type"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for csv_file in tqdm(paddle_files, desc="Merging files"):
            # Determine source type based on path (e.g. 'bing', 'telegram', 'hf')
            source_type = "unknown"
            if "bing" in str(csv_file): source_type = "bing"
            elif "telegram" in str(csv_file): source_type = "telegram"
            elif "hf" in str(csv_file): source_type = "huggingface"

            # Clean processing logic
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        stats['total'] += 1
                        
                        text = row.get("ocr_text", "")
                        if text is None: text = ""
                        
                        try:
                            conf = float(row.get("confidence", 0.0))
                        except (ValueError, TypeError):
                            conf = 0.0

                        # Filter 1: Confidence logic
                        # If text exists (len > 2) but confidence low -> remove
                        # If confidence is high or confidence is 0 (no text) -> keep
                        if len(text) > 2 and 0 < conf < MIN_CONFIDENCE:
                            stats['removed'] += 1
                            continue
                            
                        # Filter 2: Bad words
                        if text:
                            is_bad, bad_word = is_bad_text(text, bad_words)
                            if is_bad:
                                # log.info(f"Removed bad word '{bad_word}'")
                                stats['removed'] += 1
                                continue
                        
                        # Keep it!
                        stats["kept"] += 1
                        
                        # Prepare row for final csv
                        row_out = {
                            "filename": row.get("filename"),
                            "source_path": str(csv_file.parent / row.get("filename", "")),
                            "ocr_text": text,
                            "confidence": conf,
                            "source_type": source_type
                        }
                        writer.writerow(row_out)
                        
            except Exception as e:
                log.error(f"Error reading {csv_file}: {e}")

    log.info(f"Merge complete. Total: {stats['total']}, Kept: {stats['kept']}, Removed: {stats['removed']}")
    log.info(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
