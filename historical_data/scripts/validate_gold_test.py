"""Independently validate packaged Gold Test, source annotations, and split."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "DISASTER_TYPE", "WEATHER_CAUSE", "LOCATION", "TIME", "DAMAGE",
    "CASUALTY", "AFFECTED_POPULATION", "RESPONSE_ACTION",
    "RESPONSE_AGENCY", "WARNING_LEVEL",
}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def validate_rows(rows: list[dict], expected: set[str], name: str) -> dict:
    errors: list[str] = []
    entities = 0
    for i, row in enumerate(rows, 1):
        if set(row) != {"document_id", "text", "label"}:
            errors.append(f"{name} row {i}: wrong keys")
        if not isinstance(row.get("document_id"), str) or not row["document_id"].strip():
            errors.append(f"{name} row {i}: empty/invalid document_id")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{name} row {i}: empty/invalid text")
            continue
        labels = row.get("label")
        if not isinstance(labels, list):
            errors.append(f"{name} row {i}: label is not a list")
            continue
        seen: set[tuple] = set()
        spans: list[tuple[int, int]] = []
        for entity in labels:
            entities += 1
            if not isinstance(entity, list) or len(entity) != 3:
                errors.append(f"{name} row {i}: entity must be [start,end,label]")
                continue
            start, end, label = entity
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                errors.append(f"{name} row {i}: invalid offsets {entity}")
                continue
            if label not in expected:
                errors.append(f"{name} row {i}: unknown label {label}")
            if text[start:end].strip() != text[start:end]:
                errors.append(f"{name} row {i}: span has leading/trailing whitespace")
            key = (start, end, label)
            if key in seen:
                errors.append(f"{name} row {i}: duplicate span {key}")
            seen.add(key)
            spans.append((start, end))
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if right[0] < left[1]:
                errors.append(f"{name} row {i}: overlapping/nested spans {left} and {right}")
    return {"records": len(rows), "entities": entities, "errors": errors}


def main() -> None:
    gold_path = ROOT / "data/gold_test_final.jsonl"
    a1_path = ROOT / "data/annotations/annotator_1_original.jsonl"
    a2_path = ROOT / "data/annotations/annotator_2_original.jsonl"
    train_path = ROOT / "data/train_candidate_clean.jsonl"
    gold = read(gold_path)
    a1, a2, train = read(a1_path), read(a2_path), read(train_path)
    annotation_1_validation = validate_rows(a1, LABELS, "annotator_1")
    annotation_2_validation = validate_rows(a2, LABELS, "annotator_2")
    iaa_validation_path = ROOT / "statistics/iaa_recomputed/validation_report.json"
    iaa_validation = json.loads(iaa_validation_path.read_text(encoding="utf-8")) if iaa_validation_path.exists() else {}
    result = {
        "gold_test": validate_rows(gold, LABELS, "gold_test"),
        "annotator_1": annotation_1_validation,
        "annotator_2": annotation_2_validation,
        "alignment": {
            "a1_a2_same_record_count": len(a1) == len(a2) == len(gold),
            "final_aligned_with_both": len(a1) == len(a2) == len(gold) and all(
                (x["document_id"], x["text"]) == (y["document_id"], y["text"]) == (z["document_id"], z["text"])
                for x, y, z in zip(a1, a2, gold)
            ),
        },
        "leakage": {
            "remaining_normalized_sentence_text_overlap_count": len(
                {norm(x["text"]) for x in gold} & {norm(x["text"]) for x in train}
            ),
            "document_id_overlap_count": len(
                {x["document_id"] for x in gold} & {x["document_id"] for x in train}
            ),
        },
        "known_source_annotation_observations": [
            error for error in annotation_1_validation["errors"]
            if "leading/trailing whitespace" in error
        ],
        "reported_unaligned_token_boundaries": iaa_validation.get("annotation_checks", {}).get("unaligned_token_boundaries", {}),
        "source_annotation_errors_after_known_observation_exclusion": {
            "annotator_1": [error for error in annotation_1_validation["errors"] if "leading/trailing whitespace" not in error],
            "annotator_2": annotation_2_validation["errors"],
        },
    }
    result["passed"] = (
        result["gold_test"]["records"] == 1000
        and not result["gold_test"]["errors"]
        and not result["source_annotation_errors_after_known_observation_exclusion"]["annotator_1"]
        and not result["source_annotation_errors_after_known_observation_exclusion"]["annotator_2"]
        and result["alignment"]["a1_a2_same_record_count"]
        and result["alignment"]["final_aligned_with_both"]
        and result["leakage"]["remaining_normalized_sentence_text_overlap_count"] == 0
        and result["leakage"]["document_id_overlap_count"] == 0
    )
    path = ROOT / "statistics/validation_results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

