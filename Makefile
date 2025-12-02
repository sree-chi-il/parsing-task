PYTHON = python3
PIP = pip3

SCAN_DIR = scans
WORK_DIR = work
OUT_DIR = outputs
STRUCT_DIR = $(OUT_DIR)/structured
REPORT_DIR = $(OUT_DIR)/reports

SCANS := $(wildcard $(SCAN_DIR)/*.png)
STRUCT_TARGETS := $(patsubst $(SCAN_DIR)/%.png, $(STRUCT_DIR)/%.json, $(SCANS))

.PHONY: all setup clean pipeline dirs

all: pipeline

dirs:
	@mkdir -p $(WORK_DIR)
	@mkdir -p $(STRUCT_DIR)
	@mkdir -p $(REPORT_DIR)

setup:
	@echo "[Setup] Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Make sure Tesseract OCR is installed on your system."

pipeline: dirs $(STRUCT_TARGETS)
	@echo "[Pipeline] Complete."

$(STRUCT_DIR)/%.json: $(SCAN_DIR)/%.png
	@echo "[Process] $<"
	$(eval STEM := $(basename $(notdir $<)))

	$(PYTHON) src/preprocess.py "$<" "$(WORK_DIR)/$(STEM)_clean.png"

	$(PYTHON) src/pipeline.py \
		"$(WORK_DIR)/$(STEM)_clean.png" \
		"$@" \
		"$(REPORT_DIR)/$(STEM).md"

clean:
	rm -rf $(WORK_DIR) $(OUT_DIR)
