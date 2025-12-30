import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _build_text_block(row: Dict[str, str]) -> str:
    city = row.get("City", "Not Known")
    aspect = row.get("Aspect", "Not Known")
    subaspect = row.get("Sub-Aspect", "Not Known")
    problem_ar = row.get("Problem", "Not Known")
    cause_en = row.get("Caused By", "Not Known")
    cause_ar = row.get("Caused By (Arabic)", "Not Known")
    expanded_to = row.get("Expanded to", "Not Known")
    time = row.get("Time", "Not Known")
    scale = row.get("Scale", "Not Known")
    value = row.get("Value", "Not Known")
    solutions = row.get("Suggested Solutions", "Not Known")
    score = row.get("SCORE %", "Not Known")
    final_score = row.get("FINAL SCORE", "Not Known")

    return (
        f"[City] {city} | المدينة: {city}\n"
        f"[Aspect] {aspect} | المحور: {aspect}\n"
        f"[Sub-Aspect] {subaspect} | الجانب الفرعي: {subaspect}\n"
        f"[Problem] المشكلة: {problem_ar}\n"
        f"[Caused By] السبب: {cause_en}\n"
        f"[Caused By (Arabic)] {cause_ar}\n"
        f"[Expanded To] يتوسع إلى: {expanded_to}\n"
        f"[Time] الزمن: {time}\n"
        f"[Scale] النطاق: {scale}\n"
        f"[Value] القيمة: {value}\n"
        f"[Suggested Solutions] الحلول المقترحة: {solutions}\n"
        f"[Score %] {score} | [Final Score] {final_score}\n"
        f"-----------------------------"
    )


def json_to_bilingual_text(
    json_path: str,
    output_path: str,
    metadata_output_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Convert cleaned JSON records into bilingual deterministic text blocks.
    Each record becomes one text block combining Arabic and English fields.
    The text blocks are written to a .txt file and returned alongside metadata.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, str]] = []
    for sheet_name, rows in data.items():
        for index, row in enumerate(rows):
            text_block = _build_text_block(row)

            records.append(
                {
                    "id": f"{sheet_name}_{index}",
                    "sheet": sheet_name,
                    "index": index,
                    "city": row.get("City", "Not Known"),
                    "aspect": row.get("Aspect", "Not Known"),
                    "subaspect": row.get("Sub-Aspect", "Not Known"),
                    "text": text_block,
                }
            )

    # Write all text blocks into one readable .txt file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record["text"] + "\n\n")

    if metadata_output_path:
        metadata_path = Path(metadata_output_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as meta_file:
            json.dump(records, meta_file, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(records)} records into text blocks.")
    print(f"📄 Text output saved at: {output_path}")
    if metadata_output_path:
        print(f"🗂️ Metadata output saved at: {metadata_output_path}")

    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert cleaned JSON records into deterministic bilingual text blocks."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=str(Path(__file__).parent / "jobar_cleaned_final.json"),
        help="Path to the cleaned JSON file.",
    )
    parser.add_argument(
        "--text-output",
        type=str,
        default=str(Path(__file__).parent / "jobar_cleaned_final.txt"),
        help="Where to write the concatenated text blocks.",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default=str(Path(__file__).parent / "jobar_cleaned_final_blocks.json"),
        help="Where to write the JSON metadata (id, sheet, index, text).",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    json_to_bilingual_text(
        json_path=args.json_path,
        output_path=args.text_output,
        metadata_output_path=args.metadata_output,
    )


if __name__ == "__main__":
    main()