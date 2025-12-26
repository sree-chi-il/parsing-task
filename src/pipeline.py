import cv2
import pytesseract
import json
import sys
import os
import re
from pytesseract import Output
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('.env.local')


# Configure Tesseract path from environment variable
tesseract_path = "C:\\Users\\sreec\\Research\\Housing Economics Research\\tesseract.exe"
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
    print("Step 1 started")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, thresh = cv2.threshold(
        gray_small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
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
        # scale back to original image coordinates
        scale = 2.0
        x = int(x * scale)
        y = int(y * scale)
        w = int(w * scale)
        h = int(h * scale)
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

    print("Step 1 complete")

    return blocks


################################################################################
# 2. OCR Extraction
################################################################################

def extract_text(image_path, blocks):
    """
    Single-pass OCR:
    - Run Tesseract ONCE on full image
    - Assign OCR words to layout blocks by bounding box overlap
    """

    print("Step 2 started (Single-pass OCR)")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # 🚨 DO NOT RESIZE — full resolution is CRITICAL
    img_ocr = img

    config = (
        "--oem 1 "                     # LSTM only (BEST)
        "--psm 4 "                     # Fully automatic layout
        "-c preserve_interword_spaces=1 "
    )


    # Resize for faster OCR (may reduce accuracy)
    # img = cv2.imread(image_path)
    # if img is None:
    #     raise FileNotFoundError(image_path)

    # H, W = img.shape[:2]
    # scale = 0.5
    # img_ocr = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


    # ONE Tesseract call for entire page
    data = pytesseract.image_to_data(
        img_ocr,
        output_type=Output.DICT,
        # config="--psm 4"  # old Multi-column layout (best for newspapers)
        config=config
    )


    # Build word list once
    all_words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue

        all_words.append({
            "text": text,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "w": int(data["width"][i]),
            "h": int(data["height"][i]),
            "conf": float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
        })

        # all_words.append({
        #     "text": text,
        #     "x": int(data["left"][i] / scale),
        #     "y": int(data["top"][i] / scale),
        #     "w": int(data["width"][i] / scale),
        #     "h": int(data["height"][i] / scale),
        #     "conf": float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
        # })

    enriched_blocks = []
    full_text_parts = []

    # Assign words → blocks
    for blk in blocks:
        bx, by, bw, bh = blk["bbox"]
        bx2, by2 = bx + bw, by + bh
        entry = dict(blk)

        block_words = []
        block_texts = []

        if blk["type"] == "text_block":
            for w in all_words:
                wx, wy = w["x"], w["y"]
                wx2, wy2 = wx + w["w"], wy + w["h"]

                # Check if word center is inside the block
                cx = (wx + wx2) // 2
                cy = (wy + wy2) // 2

                if bx <= cx <= bx2 and by <= cy <= by2:
                    block_words.append({
                        "text": w["text"],
                        "bbox": [w["x"] - bx, w["y"] - by, w["w"], w["h"]],
                        "conf": w["conf"]
                    })
                    block_texts.append(w["text"])

            entry["ocr_words"] = block_words
            entry["text"] = " ".join(block_texts)

            if block_texts:
                full_text_parts.append(entry["text"])

        else:
            # Graphical assets: DO NOT re-OCR
            entry["text"] = "[GRAPHICAL_ASSET]"

        enriched_blocks.append(entry)

    print("Step 2 complete")

    return enriched_blocks, "\n\n".join(full_text_parts)

################################################################################
# 3. Semantic Extraction
################################################################################

def parse_ordinance_semantics(full_text):

    print("Step 3 started")

    upper = full_text.upper()

    print("Step 3 complete")

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

    print("Step 4 started")

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

    print("Step 4 complete")
    
    return meta


################################################################################
# 5. Pipeline Runner
################################################################################

def run_pipeline(scan_path, out_json, out_md):
    print("Step 5 started")

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
    
    print("Step 5 complete")
    
    print(f"[Pipeline] Wrote {out_json} and {out_md}")


################################################################################
# Entry Point
################################################################################

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pipeline.py <scan.png> <output.json> <report.md>")
        sys.exit(1)

    run_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])