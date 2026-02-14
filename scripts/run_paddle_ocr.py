"""
PaddleOCR runner.
Pro-quality OCR for memes (supports rotated text, complex backgrounds).
Results are saved to 'ocr_paddle.csv' next to the images.
"""

import csv
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# Import PaddleOCR 
from paddleocr import PaddleOCR

# === CONFIG ===
INPUT_CSV = Path("data/processed/metadata_clean_step1.csv")

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("ppocr").setLevel(logging.WARNING)

def get_all_image_paths(csv_path):
    images = []
    
    if not csv_path.exists():
        log.error(f"Input CSV not found: {csv_path}")
        return []
        
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_str = row.get("source_path")
            if path_str:
                path = Path(path_str)
                # Check absolute path
                if path.exists():
                    images.append(path)
                else:
                    # Check relative to current working dir
                    # (in case CSV was generated with different root)
                    rel_path = Path.cwd() / path_str
                    if rel_path.exists():
                        images.append(rel_path)

    return images

def save_ocr_result(img_path, text, conf):
    """
    Save result to ocr_paddle.csv (next to image).
    """
    csv_path = img_path.parent / "ocr_paddle.csv"
    file_exists = csv_path.exists()
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ["filename", "ocr_text", "confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow({
                "filename": img_path.name,
                "ocr_text": text,
                "confidence": round(conf, 4)
            })
    except Exception as e:
        log.error(f"Failed to write CSV for {img_path}: {e}")

def run_paddle():
    # 1. Init PaddleOCR
    # lang='ru' enables Cyrillic + English
    log.info("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='ru')

    # 2. Scanning
    log.info(f"Reading image list from {INPUT_CSV}...")
    all_images = get_all_image_paths(INPUT_CSV)
    log.info(f"Found {len(all_images)} images to process.")

    # 3. Processing
    for img_path in tqdm(all_images, desc="Running PaddleOCR"):
        try:
            # Fix: Read image using CV2 to support text-only paths or weird formats (like .webp)
            # PaddleOCR works best with numpy arrays
            img = cv2.imread(str(img_path))
            
            if img is None:
                # Fallback or skip
                # log.warning(f"CV2 failed to read: {img_path}")
                continue

            # Run OCR on the image array
            # Removing cls=True as it causes error in new version
            result = ocr.ocr(img)
            
            final_text = ""
            confidence = 0.0
            
            # Robust parsing of the result
            # Robust parsing of the result
            if result:
                # PaddleOCR v3 returns a list with a dict inside
                # Structure: [{'rec_texts': [...], 'rec_scores': [...]}]
                
                # Get the first element (it might be a list containing the dict)
                data = result[0] if isinstance(result, list) else result
                
                # Check for v3 dictionary format
                if isinstance(data, dict) and 'rec_texts' in data:
                    texts = data.get('rec_texts', [])
                    confs = data.get('rec_scores', [])
                    
                    # Filter out empty strings if any
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
                    # Fallback for older versions: [[box, [text, conf]], ...]
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
            
            # Save result
            if final_text.strip():
                save_ocr_result(img_path, final_text, confidence)
            else:
                # Save empty result to indicate processing was done
                # Optional: save_ocr_result(img_path, "", 0.0)
                pass

        except Exception as e:
            # Log error but keep going
            log.error(f"Error processing {img_path}: {e}")
            continue

if __name__ == "__main__":
    run_paddle()
