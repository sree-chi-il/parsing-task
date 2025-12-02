import json
import sys
from jsonschema import validate, ValidationError

SCHEMA = {
    "type": "object",
    "required": [
        "metadata",
        "full_transcription",
        "layout_structure",
        "ordinance_summary"
    ]
}

def validate_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        validate(instance=data, schema=SCHEMA)
        print(f"{path}: OK")
    except ValidationError as e:
        print(f"{path}: INVALID — {e.message}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_schema.py <file.json> ...")
        sys.exit(1)

    for file in sys.argv[1:]:
        validate_file(file)
