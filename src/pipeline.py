import cv2
import pytesseract
import json
import sys
import os
import re
from pytesseract import Output
from datetime import datetime

load_dotenv('.env.local')


# Configure Tesseract path from environment variable
tesseract_path = os.getenv("TESSERACT_PATH")
if tesseract_path and os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print(f"Warning: Tesseract not found at {tesseract_path}. Relying on system PATH.")

#

# repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# tesseract_path = os.path.join(repo_root, "tesseract", "tesseract.exe")
# if os.path.exists(tesseract_path):
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
# else:
#     print(f"Warning: Local Tesseract not found at {tesseract_path}. Relying on system PATH.")

# Regex patterns
ORDINANCE_RE = re.compile(r"ORDINANCE\s+NO\.?\s*(\d+)", re.IGNORECASE)
SECTION_RE = re.compile(r"SECTION\s+(\d+)[\.:]?", re.IGNORECASE)
ZONING_MAP_RE = re.compile(r"ZONING\s+MAP\s*(\d+)?", re.IGNORECASE)
ZONING_LABEL_RE = re.compile(r"\b[A-Z]-?\d\b", re.IGNORECASE)


################################################################################
# 1. Layout Analysis
################################################################################

def analyze_layout(image_path):
    """
    Two-stage dilation:
        1. Merge text lines horizontally
        2. Merge lines into larger blocks
    Zones >20% of the page area are graphical assets (maps, photos).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Stage 1 – join text lines
    kernel_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    line_dil = cv2.dilate(thresh, kernel_lines, iterations=2)

    # Stage 2 – join lines into blocks
    kernel_blocks = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 20))
    block_dil = cv2.dilate(line_dil, kernel_blocks, iterations=1)

    contours, _ = cv2.findContours(
        block_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blocks = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 8000:
            continue

        blk_type = "graphical_asset" if area > 0.20 * H * W else "text_block"

        blocks.append({
            "type": blk_type,
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": int(area)
        })

    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return blocks


################################################################################
# 2. OCR Extraction
################################################################################

def extract_text(image_path, blocks):
    """
    OCR pipeline:
    - text_block → word-level OCR with bounding boxes & confidence
    - graphical_asset → sparse OCR for map captions
    """
    img = cv2.imread(image_path)
    enriched = []
    full_text_parts = []

    for blk in blocks:
        x, y, w, h = blk["bbox"]
        roi = img[y:y+h, x:x+w]
        entry = dict(blk)

        if blk["type"] == "text_block":
            data = pytesseract.image_to_data(
                roi, output_type=Output.DICT, config="--psm 6"
            )

            words = []
            texts = []

            for i, t in enumerate(data["text"]):
                t = t.strip()
                if not t:
                    continue

                words.append({
                    "text": t,
                    "bbox": [
                        int(data["left"][i]),
                        int(data["top"][i]),
                        int(data["width"][i]),
                        int(data["height"][i])
                    ],
                    "conf": float(data["conf"][i])
                    if data["conf"][i] != "-1" else -1.0
                })
                texts.append(t)

            entry["ocr_words"] = words
            entry["text"] = " ".join(texts)

            if texts:
                full_text_parts.append(entry["text"])

        else:
            # Sparse OCR for map graphics / illustrations
            sparse = pytesseract.image_to_string(
                roi, config="--psm 11"
            ).strip()

            entry["description_ocr"] = sparse
            entry["text"] = "[GRAPHICAL_ASSET]"

            if sparse:
                full_text_parts.append(sparse)

        enriched.append(entry)

    return enriched, "\n\n".join(full_text_parts)

################################################################################
# 3. Semantic Extraction
################################################################################

def parse_ordinance_semantics(full_text):
    upper = full_text.upper()

    return {
        "ordinance_ids": ORDINANCE_RE.findall(upper),
        "sections": SECTION_RE.findall(upper),
        "zoning_map_refs": ZONING_MAP_RE.findall(upper),
        "zoning_labels": ZONING_LABEL_RE.findall(upper),
        "penalties_mentioned": bool(
            re.search(r"PENALTY|FINE|MISDEMEANOR|IMPRISON", upper)
        ),
        "topics": list({
            t for t in [
                "Zoning" if "ZONING" in upper else None,
                "Ordinance" if "ORDINANCE" in upper else None
            ] if t
        })
    }


################################################################################
# 4. Metadata from Filename
################################################################################

def extract_metadata_from_filename(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]

    meta = {
        "source_file": base,
        "publication_name": None,
        "issue_date": None,
        "page_number": None
    }

    # Detect date (e.g. jan-05-1959)
    dat = re.search(r"([a-zA-Z]+)[-_](\d{1,2})[-_](\d{4})", name)
    if dat:
        month, day, year = dat.groups()
        try:
            dt = datetime.strptime(f"{month} {day} {year}", "%b %d %Y")
        except:
            try:
                dt = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
            except:
                dt = None
        if dt:
            meta["issue_date"] = dt.date().isoformat()

    # Page number p-XX
    p = re.search(r"p-(\d+)", name, re.IGNORECASE)
    if p:
        meta["page_number"] = p.group(1)

    # Publication name = prefix before the date
    pub = re.match(r"(.+?)-[a-zA-Z]+-\d{1,2}-\d{4}", name)
    if pub:
        meta["publication_name"] = pub.group(1)

    return meta


################################################################################
# 5. Pipeline Runner
################################################################################

def run_pipeline(scan_path, out_json, out_md):
    print(f"[Pipeline] Running on {scan_path}")

    # Extract metadata
    meta = extract_metadata_from_filename(scan_path)

    # 1. Layout analysis
    blocks = analyze_layout(scan_path)

    # 2. OCR
    enriched_blocks, full_text = extract_text(scan_path, blocks)

    # 3. Semantics
    semantics = parse_ordinance_semantics(full_text)

    # 4. Build JSON output
    output = {
        "metadata": {
            **meta,
            "processed_date": datetime.utcnow().isoformat() + "Z",
            "source_path": os.path.abspath(scan_path)
        },
        "full_transcription": full_text,
        "layout_structure": enriched_blocks,
        "ordinance_summary": semantics
    }

    # Write JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Write Markdown report
    with open(out_md, "w", encoding="utf-8") as r:
        r.write(f"# Report for {os.path.basename(scan_path)}\n\n")
        r.write(f"**Ordinances:** {semantics['ordinance_ids']}\n\n")
        r.write(f"**Sections:** {semantics['sections']}\n\n")
        r.write(f"**Zoning Codes:** {semantics['zoning_labels']}\n\n")
        r.write(f"**Penalties Mentioned:** {semantics['penalties_mentioned']}\n\n")
        r.write("## Transcript Preview (first 2000 chars)\n\n")
        r.write(full_text[:2000] + "\n")

    print(f"[Pipeline] Wrote {out_json} and {out_md}")


################################################################################
# Entry Point
################################################################################

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pipeline.py <scan.png> <output.json> <report.md>")
        sys.exit(1)

    run_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])
