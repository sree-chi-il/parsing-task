import json
import os
from datetime import datetime
from ocr_engines import (
    ocr_tesseract,
    ocr_paddle,
    ocr_surya,
    ocr_easy
)
from semantics import parse_ordinance_semantics

ENGINES = {
    "tesseract_max": ocr_tesseract,
    "paddleocr": ocr_paddle,
    "surya": ocr_surya,
    "easyocr": ocr_easy
}

def run_all_pipelines(image_path, out_dir="outputs"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    os.makedirs(out_dir, exist_ok=True)

    for name, engine in ENGINES.items():
        print(f"\n=== Running {name.upper()} ===")

        words, text = engine(image_path)

        # Confidence handling (robust)
        valid_confs = [w["conf"] for w in words if w.get("conf", -1) >= 0]
        avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0

        semantics = parse_ordinance_semantics(text)

        output = {
            "engine": name,
            "processed_date": datetime.utcnow().isoformat() + "Z",
            "image": os.path.basename(image_path),
            "full_transcription": text,
            "word_count": len(words),
            "avg_confidence": round(avg_conf, 3),
            "words": words,
            "ordinance_summary": semantics
        }

        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Wrote {out_path}")
        print(f"Words: {len(words)}")
        print(f"Avg conf: {avg_conf:.3f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pipeline_compare.py <scan.png>")
        sys.exit(1)

    run_all_pipelines(sys.argv[1])
