import os
import csv
import easyocr
import logging
from pathlib import Path
from tqdm import tqdm

# Список папок, где искать картинки
SOURCE_DIRS = [
    Path("data/raw/bing_memes/images"),
    Path("data/raw/telegram_stickers"),
    Path("data/raw/hf_memes/images"),
]
# Куда сохранять результат
OUTPUT_FILE = Path("data/processed/metadata_ocr.csv")

# Логгер 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def get_all_image_paths(source_dirs):
    """
    Функция должна вернуть список (list) путей (Path) ко всем картинкам.
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    
    for directory in source_dirs:
        if not directory.exists():
            continue

        for file in directory.rglob("*"):   
            if file.is_file() and file.suffix.lower() in extensions:
                images.append(file)

                 
    return images

def run_ocr():
    # 1. Инициализация модели
    log.info("Loading EasyOCR model")

    reader = easyocr.Reader(['ru', 'en'], gpu=False)

    # 2. Поиск картинок
    log.info("Scanning directories")
    all_images = get_all_image_paths(SOURCE_DIRS)
    log.info(f"Found {len(all_images)} images total.")

    # Создаем папку для output файла
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 3  CSV на запись
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        # Заголовки таблицы
        fieldnames = ["filename", "source_path", "ocr_text", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 4. Проходимся по всем картинкам
        for img_path in tqdm(all_images, desc="Running OCR"):
            try:
                # - detail=1 (чтобы получить уверенность)
                # - paragraph=True (чтобы объединить слова в предложения)
                
                result =  reader.readtext(str(img_path), detail = 1, paragraph=False) 
                
                    
                if result:
                    texts = [item[1] for item in result]
                    confs = [item[2] for item in result]
                    
                    final_text = " ".join(texts)
                    confidence = sum(confs) / len(confs)
                else:
                    final_text = ""
                    confidence = 0.0

                 
                writer.writerow({
                        "filename": img_path.name,
                        "source_path": str(img_path),
                        "ocr_text": final_text,
                        "confidence": confidence
                })

            except Exception as e:
                log.error(f"Error processing {img_path}: {e}")
                continue

if __name__ == "__main__":
    run_ocr()