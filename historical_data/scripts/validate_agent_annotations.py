"""Validate AI training labels, provenance, coverage, leakage, and source integrity."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def norm(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def validate():
    source = read(ROOT / "data/train_candidate_clean.jsonl")
    annotated = read(ROOT / "data/train_agent_annotated.jsonl")
    pending = read(ROOT / "data/train_agent_pending.jsonl")
    provenance = read(ROOT / "annotation_work/provenance.jsonl")
    gold = read(ROOT / "data/gold_test_final.jsonl")
    labels = set(json.loads((ROOT / "data/annotations/annotation_schema.json").read_text(encoding="utf-8"))["labels"])
    errors = []
    acceptance_path = ROOT / "annotation_work/review_acceptances.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.exists() else {"records": []}
    accepted = {r["source_row"]: r["annotation_sha256"] for r in acceptance["records"]}
    if not annotated:
        errors.append("No annotated records")
    if len(annotated) != len(provenance):
        errors.append("Output/provenance lengths differ")
    processed = set()
    totals = Counter()
    for output_number, (row, meta) in enumerate(zip(annotated, provenance), 1):
        number = meta.get("source_row")
        if type(number) is not int or not 1 <= number <= len(source) or number in processed:
            errors.append(f"Invalid or repeated source row: {number}")
            continue
        processed.add(number)
        original = source[number - 1]
        if set(row) != {"document_id", "text", "label"}:
            errors.append(f"Wrong output keys at source row {number}")
        if (row["document_id"], row["text"]) != (original["document_id"], original["text"]):
            errors.append(f"Source text/document modified: {number}")
        if meta["output_row"] != output_number or meta["sentence_id"] != original["sentence_id"]:
            errors.append(f"Wrong provenance mapping: {number}")
        if meta["text_sha256"] != hashlib.sha256(row["text"].encode("utf-8")).hexdigest():
            errors.append(f"Wrong text hash: {number}")
        expected_status = "user_reported_reviewed_accepted" if number in accepted else "agent_annotated_not_human_reviewed"
        if meta["review_status"] != expected_status:
            errors.append(f"Wrong annotation provenance status: {number}")
        if number in accepted:
            digest = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if digest != accepted[number]:
                errors.append(f"User-accepted annotation changed: {number}")
        if not isinstance(row["label"], list):
            errors.append(f"Invalid label list: {number}")
            continue
        end_previous = 0
        seen = set()
        for entity in row["label"]:
            if not isinstance(entity, list) or len(entity) != 3:
                errors.append(f"Invalid entity shape: {number}")
                continue
            start, end, label = entity
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row["text"]):
                errors.append(f"Out-of-range offset: {number}, {entity}")
                continue
            surface = row["text"][start:end]
            if label not in labels or surface != surface.strip():
                errors.append(f"Invalid label/whitespace: {number}, {entity}")
            if start < end_previous or tuple(entity) in seen:
                errors.append(f"Unsorted, duplicate, or overlapping entity: {number}, {entity}")
            end_previous = end
            seen.add(tuple(entity))
            totals[label] += 1
    expected_pending = [row for number, row in enumerate(source, 1) if number not in processed]
    expected_new = [row for row, meta in zip(annotated, provenance) if meta["source_row"] not in accepted]
    if read(ROOT / "data/train_agent_new_since_review.jsonl") != expected_new:
        errors.append("New-since-review export does not match unreviewed rows in combined output")
    if not set(accepted).issubset(processed):
        errors.append("Previously user-accepted rows are missing")
    if pending != expected_pending:
        errors.append("Pending rows do not exactly match unprocessed source records in original order")
    sentence_overlap = {norm(r["text"]) for r in annotated} & {norm(r["text"]) for r in gold}
    document_overlap = {r["document_id"] for r in annotated} & {r["document_id"] for r in gold}
    if sentence_overlap or document_overlap:
        errors.append("Train/test overlap detected")
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    changed = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            changed.append(entry["path"])
    if changed:
        errors.append(f"Original package files changed: {changed}")
    progress = json.loads((ROOT / "annotation_work/progress.json").read_text(encoding="utf-8"))
    expected_counts = {"annotated_rows": len(annotated), "pending_rows": len(pending),
                       "entity_count": sum(totals.values()), "explicitly_annotated_empty_rows": sum(not r["label"] for r in annotated)}
    if any(progress[key] != value for key, value in expected_counts.items()) or progress["label_counts"] != dict(totals):
        errors.append("Progress statistics disagree with output")
    if progress["user_reported_reviewed_rows"] != len(accepted) or progress["new_rows_since_user_review"] != len(expected_new):
        errors.append("Review-status statistics disagree with acceptance records")
    report = dict(expected_counts, source_rows=len(source), user_reported_reviewed_rows=len(accepted),
                  new_rows_since_user_review=len(annotated) - len(accepted), checked_original_files=len(manifest["files"]),
                  original_file_hash_mismatches=changed, normalized_test_sentence_overlap=len(sentence_overlap),
                  test_document_overlap=len(document_overlap), errors=errors, passed=not errors,
                  scope="Structural, provenance, and leakage checks only; semantic correctness still requires review.")
    return report


if __name__ == "__main__":
    result = validate()
    (ROOT / "annotation_work/validation_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
