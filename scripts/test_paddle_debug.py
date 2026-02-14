"""
Test PaddleOCR on a single image.
"""
from paddleocr import PaddleOCR
import cv2
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Test image (known meme with text)
IMG_PATH = "/Users/amirhamidullin/PycharmProjects/coursework3/data/raw/kaggle_russian_memes/1e64b3e5-0911-46b6-9702-40b5896bafba.jpg" 

def main():
    log.info("Testing PaddleOCR on one image...")
    
    # Init (same as in script)
    # Removing show_log=True as it's deprecated and causes error
    ocr = PaddleOCR(use_angle_cls=True, lang='ru')
    
    # Read
    img = cv2.imread(IMG_PATH)
    if img is None:
        log.error(f"Failed to read image: {IMG_PATH}")
        return

    # Run
    log.info("Running OCR...")
    # Removing cls=True as it causes error in new version
    result = ocr.ocr(img)
    
    log.info(f"RAW RESULT TYPE: {type(result)}")
    log.info(f"RAW RESULT: {result}")

    # Try parsing new format
    if result:
        # Paddle v3 returns a list containing a dict
        data = result[0] if isinstance(result, list) else result
        
        if isinstance(data, dict):
            texts = data.get('rec_texts', [])
            scores = data.get('rec_scores', [])
            
            log.info(f"Docs found texts: {texts}")
            log.info(f"Docs found scores: {scores}")
            
            final_text = " ".join(texts)
            if scores:
                 avg_conf = sum(scores) / len(scores)
                 log.info(f"Final text: '{final_text}' (conf: {avg_conf:.2f})")
            else:
                 log.info("No text found.")
        else:
             log.info("Unknown format (not a dict)")
    
if __name__ == "__main__":
    main()
