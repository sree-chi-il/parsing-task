import os
import subprocess
import glob
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCAN_DIR = "scans"
WORK_DIR = "work"
OUT_DIR = "outputs"
STRUCT_DIR = os.path.join(OUT_DIR, "structured")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

def run_pipeline():
    # 1. Create all necessary directories
    for d in [WORK_DIR, STRUCT_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

    # 2. Find files (Checks for both .png and .jpg to be safe)
    scans = glob.glob(os.path.join(SCAN_DIR, "*.png")) + \
            glob.glob(os.path.join(SCAN_DIR, "*.jpg"))
    
    if not scans:
        print(f"[Error] No scans found in '{SCAN_DIR}/'. Checked *.png and *.jpg.")
        return

    print(f"Found {len(scans)} scans. Starting full pipeline...\n")

    # 3. Process each file
    for scan_path in scans:
        filename = os.path.basename(scan_path)
        stem = os.path.splitext(filename)[0]
        
        print(f"==> Processing {filename}")
        
        # Define paths
        clean_img_path = os.path.join(WORK_DIR, f"{stem}_clean.png")
        json_path = os.path.join(STRUCT_DIR, f"{stem}.json")
        report_path = os.path.join(REPORT_DIR, f"{stem}.md")

        # Step A: Preprocess (Cleaning)
        try:
            print(f"   [1/2] Preprocessing...")
            # Using sys.executable ensures we use the same Python environment
            subprocess.run([sys.executable, "src/preprocess.py", scan_path, clean_img_path], check=True)
        except subprocess.CalledProcessError:
            print(f"   ❌ Preprocessing failed for {filename}. Skipping.")
            continue
        
        # Step B: Pipeline (OCR & Extraction)
        try:
            print(f"   [2/2] Running Extraction Pipeline...")
            subprocess.run([sys.executable, "src/pipeline.py", clean_img_path, json_path, report_path], check=True)
            print(f"   ✔ Success! Output: {json_path}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Extraction failed for {filename}.")
        
    print("\nPipeline finished. Check the 'outputs/' folder.")

if __name__ == "__main__":
    run_pipeline()