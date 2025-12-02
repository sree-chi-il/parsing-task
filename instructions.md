# Bakersfield Ordinance Parsing

## Overview
You are asked to design and implement a small but **robust** pipeline that ingests a handful of Bakersfield *California* newspaper scans containing municipal ordinance content and produces structured artifacts ready for downstream consumers (LLM workflows, search pipelines, or analytics jobs). Assume you are working with fragile historical pages: expect skew, bleed-through, nonstandard fonts, and torn edges. The goal is to move from raw PNGs to structured, metadata-rich representations (think JSON/CSV/markdown bundles) that capture layout, sections, and ordinance-specific context with enough fidelity for automated ingestion.

## Provided Materials
Each PNG is a single newspaper page that includes ordinance material. The input scans live under `scans/`:
- `scans/bakersfield-californian-aug-31-1937-p-14.png`
- `scans/bakersfield-californian-jun-22-1964-p-30.png`
- `scans/bakersfield-californian-jan-05-1959-p-30.png`

Feel free to add subfolders under `scans/` (e.g., `scans/processed/`) if your workflow needs them, but keep the original PNGs unmodified for reproducibility.

## Core Task
Build an automated pipeline (language and tooling are your choice) that:
1. Preprocesses the provided scans (deskewing, denoising, segmentation, etc.).
2. Runs OCR/layout analysis to capture text, bounding boxes, reading order, and any ordinance-related zones (titles, ordinance body, penalties, approvals, etc.).
3. Normalizes and enriches the extracted data into an explicit schema that downstream consumers—whether LLM-based tooling, search indexes, or analytics scripts—can ingest without extra hand-cleaning. Treat the page as multimodal—capture text blocks, photos/illustrations, captions, and any visual separators so the resulting structure reflects the full layout. At minimum, include:
   - Source metadata (`publication_name`, `issue_date`, `page_number`, `source_file`).
   - Full plain-text transcription.
   - Structured layout (ordered blocks with coordinates, role labels, modality tags, and confidence).
   - Non-text asset descriptors (e.g., image bounding boxes, captions, inferred subject matter) when present.
   - Ordinance-level summary objects (e.g., ordinance title, scope, key obligations, penalties) derived from the structured text.
4. Writes machine-readable outputs (JSON preferred) to `outputs/structured/` and any human-readable summaries to `outputs/reports/`.

## Execution Requirements
- Provide a top-level `Makefile` target named `pipeline`. After installing your documented dependencies, a reviewer should be able to run `make pipeline` from this directory and see the full end-to-end sequence (from raw PNGs to populated `outputs/` artifacts) without manual intervention. This target must automatically enumerate every PNG under `scans/` and run them through the exact same pipeline.
- Clearly document additional targets (`make setup`, `make test`, etc.) if you add them.
- Log intermediate artifacts (e.g., preprocessed images, OCR dumps) either under `work/` or `outputs/intermediate/` so reviewers can inspect failure points.
- Keep the repo self-contained: note any large external downloads or API keys that are required, and provide shims/fail-fast messaging when something is missing.

## Makefile Primer
If you have not authored a Makefile before, think of it as a declarative command runner:
- Each **target** (e.g., `pipeline`) is a name you can invoke via `make pipeline`. Targets expand into the shell commands you list underneath them.
- You can define helper targets such as `setup` (install deps), `preprocess` (prepare scans), or `validate` (schema checks) and have `pipeline` depend on or call them in order.
- Use `.PHONY` at the top to mark targets that should always re-run.
- Targets can be used as documentation: add short `@echo` lines to explain what is happening, and exit non-zero on failure so CI/reviewers see issues immediately.
- If you prefer Python/Node/etc., simply have the target run your script, e.g. `pipeline: ; python -m pipeline.run`.

At the end, a reviewer should only need `make pipeline` (plus any documented environment setup) to process **all three** Bakersfield scans end-to-end through your designed pipeline.

## Deliverables & Evaluation
1. **Code + Makefile**: Reproducible, readable, and modular pipeline code. Decompose stages (preprocess, OCR, postprocess) so we can swap components later.
2. **Structured Outputs**: At least one JSON file per scan that adheres to your schema and captures both raw text and higher-level ordinance structure, plus metadata for any non-text elements (images, illustrations, captions, separators) detected on the page.
3. **Documentation**: Extend `instructions.md` (this file) or add `NOTES.md` with:
   - Architecture overview (data flow diagram or narrative).
   - Dependency/setup instructions.
   - Schema description with sample snippet.
   - Known limitations or follow-up ideas.
4. **Quality hooks** (optional but encouraged): quick validation scripts, schema checks, or heuristics that flag OCR uncertainty.

We will evaluate submissions on reproducibility, clarity of the schema, how easily an LLM could consume the resulting data, and the depth of reasoning around edge cases (mixed fonts, multi-column layouts, footnotes, etc.).

## Minimum Submission Checklist
Your submission is considered complete only if **all** of the following are true:
1. Running `make pipeline` from a fresh checkout processes every PNG under `scans/` and populates `outputs/structured/` and `outputs/reports/` without manual steps.
2. The repository includes all scripts/notebooks/config required to reproduce the run (no hidden local assets).
3. Structured outputs exist for each scan and conform to the documented schema (validate them if possible).
4. Documentation clearly states prerequisites, setup commands, pipeline stages, schema description, and how to inspect results or rerun failed stages.
5. Any external services (APIs, models) are described with instructions for authentication or offline alternatives; the Makefile should fail fast with a meaningful message if credentials are missing.
