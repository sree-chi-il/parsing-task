import os
import subprocess
import glob

# Configuration
SCAN_DIR = "scans"
WORK_DIR = "work"
OUT_DIR = "outputs"
STRUCT_DIR = os.path.join(OUT_DIR, "structured")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

def run():
    # 1. Create Directories
    for d in [WORK_DIR, STRUCT_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

    # 2. Find all PNG files (Change *.png to *.jpg if using JPGs)
    scans = glob.glob(os.path.join(SCAN_DIR, "*.png"))
    
    if not scans:
        print("No scans found! Check if files are .png or .jpg")
        return

    # 3. Process each file
    for scan_path in scans:
        filename = os.path.basename(scan_path)
        stem = os.path.splitext(filename)[0]
        
        print(f"Processing {filename}...")
        
        # Define paths
        clean_img_path = os.path.join(WORK_DIR, f"{stem}_clean.png")
        json_path = os.path.join(STRUCT_DIR, f"{stem}.json")
        report_path = os.path.join(REPORT_DIR, f"{stem}.md")

        # Step A: Preprocess
        subprocess.run(["python", "src/preprocess.py", scan_path, clean_img_path], check=True)
        
        # Step B: Pipeline
        subprocess.run(["python", "src/pipeline.py", clean_img_path, json_path, report_path], check=True)
        
    print("Done! Check outputs/ folder.")

if __name__ == "__main__":
    run()