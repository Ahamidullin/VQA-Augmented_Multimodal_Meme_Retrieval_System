"""paddle ocr для мемов
читает список картинок из metadata_clean_step1.csv
результаты сохраняются в ocr_paddle.csv рядом с картинками
поддерживает paddleocr v3 (dict формат) и старый формат
"""

import csv
from pathlib import Path
from tqdm import tqdm
import cv2

from paddleocr import PaddleOCR

INPUT_CSV = Path("data/processed/metadata_clean_step1.csv")


def get_all_image_paths(csv_path):
    images = []
    if not csv_path.exists():
        return images
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_str = row.get("source_path")
            if not path_str:
                continue
            path = Path(path_str)
                # абсолютный путь
            if path.exists():
                images.append(path)
            else:
                    # проверяем относительный путь
                rel_path = Path.cwd() / path_str
                if rel_path.exists():
                    images.append(rel_path)

    return images


def save_ocr_result(img_path, text, conf):
    """сохраняет результат в ocr_paddle.csv рядом с картинкой"""
    csv_path = img_path.parent / "ocr_paddle.csv"
    file_exists = csv_path.exists()
    fieldnames = ["filename", "ocr_text", "confidence"]
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "filename": img_path.name,
                "ocr_text": text,
                "confidence": round(conf, 4)
            })
    except Exception:
        pass


ocr = PaddleOCR(use_angle_cls=True, lang="ru")
all_images = get_all_image_paths(INPUT_CSV)

for img_path in tqdm(all_images, desc="ocr"):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

            # запускаем ocr
        result = ocr.ocr(img)

        final_text = ""
        confidence = 0.0

            # парсим результат
        if result:
                # paddleocr v3 возвращает список с dict
                
                # получаем первый элемент
            data = result[0] if isinstance(result, list) else result

            if isinstance(data, dict) and "rec_texts" in data:
                texts = data.get("rec_texts", [])
                confs = data.get("rec_scores", [])
                valid_texts = []
                valid_confs = []
                for t, c in zip(texts, confs):
                    if t and str(t).strip():
                        valid_texts.append(str(t))
                        valid_confs.append(float(c))
                if valid_texts:
                    final_text = " ".join(valid_texts)
                    confidence = sum(valid_confs) / len(valid_confs)
            else:
                blocks = []
                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], list):
                        blocks = result[0]
                    else:
                        blocks = result
                texts = []
                confs = []
                for line in blocks:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        content = line[1]
                        if isinstance(content, (list, tuple)) and len(content) >= 2:
                            texts.append(str(content[0]))
                            confs.append(float(content[1]))
                if texts:
                    final_text = " ".join(texts)
                    confidence = sum(confs) / len(confs)

        if final_text.strip():
            save_ocr_result(img_path, final_text, confidence)

    except Exception:
        continue
