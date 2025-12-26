import cv2
from paddleocr import PaddleOCR
import easyocr
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from PIL import Image

############################################
# PADDLE OCR
############################################

_paddle = PaddleOCR(use_angle_cls=True, lang='en')

def ocr_paddle(image_path):
    result = _paddle.ocr(image_path, cls=True)

    words = []
    text = []

    for line in result:
        for box, (txt, conf) in line:
            x1, y1 = box[0]
            x3, y3 = box[2]
            words.append({
                "text": txt,
                "bbox": [x1, y1, x3-x1, y3-y1],
                "conf": conf
            })
            text.append(txt)

    return words, "\n".join(text)

############################################
# SURYA OCR
############################################

_det_proc, _det_model = segformer.load_processor(), segformer.load_model()
_rec_model, _rec_proc = load_model(), load_processor()

def ocr_surya(image_path):
    image = Image.open(image_path)
    preds = run_ocr(
        [image],
        [["en"]],
        _det_model,
        _det_proc,
        _rec_model,
        _rec_proc
    )

    words = []
    text = []

    for res in preds:
        for line in res.text_lines:
            words.append({
                "text": line.text,
                "bbox": line.bbox,
                "conf": line.confidence
            })
            text.append(line.text)

    return words, "\n".join(text)

############################################
# EASY OCR 
############################################

_easy = easyocr.Reader(['en'], gpu=False)

def ocr_easy(image_path):
    result = _easy.readtext(image_path)

    words = []
    text = []

    for bbox, txt, conf in result:
        x1, y1 = bbox[0]
        x3, y3 = bbox[2]
        words.append({
            "text": txt,
            "bbox": [x1, y1, x3-x1, y3-y1],
            "conf": conf
        })
        text.append(txt)

    return words, "\n".join(text)
