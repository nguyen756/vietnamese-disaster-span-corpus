"""Compile explicit agent-selected surface spans into resumable NER JSONL.

This script does not predict entities. Decisions are authored by the agent after
reading each sentence. Only exact sentence duplicates inside the same article
inherit a decision. Never infer an empty label list for an unprocessed sentence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "annotation_work"
SOURCE = ROOT / "data/train_candidate_clean.jsonl"
LABELS = {
    "D": "DISASTER_TYPE", "W": "WEATHER_CAUSE", "L": "LOCATION",
    "T": "TIME", "M": "DAMAGE", "C": "CASUALTY",
    "P": "AFFECTED_POPULATION", "A": "RESPONSE_ACTION",
    "G": "RESPONSE_AGENCY", "V": "WARNING_LEVEL",
}


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve(text, selections):
    spans = []
    for selection in selections:
        code, surface, *occurrence = selection
        if code not in LABELS or not surface or surface != surface.strip():
            raise ValueError(f"Invalid label/surface: {selection!r}")
        starts = []
        position = 0
        while (position := text.find(surface, position)) >= 0:
            starts.append(position)
            position += len(surface)
        if not starts:
            raise ValueError(f"Surface absent: {selection!r}")
        if not occurrence and len(starts) != 1:
            raise ValueError(f"Repeated surface requires occurrence index or 'all': {selection!r}")
        if occurrence:
            if len(occurrence) != 1:
                raise ValueError(f"Invalid occurrence: {selection!r}")
            if occurrence[0] != "all":
                index = occurrence[0]
                if type(index) is not int or not 1 <= index <= len(starts):
                    raise ValueError(f"Invalid occurrence: {selection!r}")
                starts = [starts[index - 1]]
        spans.extend([start, start + len(surface), LABELS[code]] for start in starts)
    spans.sort()
    if len({tuple(s) for s in spans}) != len(spans):
        raise ValueError("Duplicate spans")
    for left, right in zip(spans, spans[1:]):
        if right[0] < left[1]:
            raise ValueError(f"Overlapping spans: {left!r} and {right!r}")
    return spans


def main():
    WORK.mkdir(exist_ok=True)
    (WORK / "decisions").mkdir(exist_ok=True)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    lock = WORK / "source_lock.json"
    lock_data = json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else {
        "source": "data/train_candidate_clean.jsonl", "sha256": source_hash,
    }
    if lock.exists():
        if lock_data["sha256"] != source_hash:
            raise ValueError("Source changed; row decisions must be reconciled before continuing")
    for relative in ["documentation/annotation_guideline.md", "data/annotations/annotation_schema.json", "data/gold_test_final.jsonl"]:
        reference_hash = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if relative in lock_data and lock_data[relative] != reference_hash:
            raise ValueError(f"Reference changed: {relative}; reconcile previous decisions")
        lock_data[relative] = reference_hash
    lock.write_text(json.dumps(lock_data, indent=2) + "\n", encoding="utf-8")
    rows = read_jsonl(SOURCE)
    acceptance_path = WORK / "review_acceptances.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.exists() else {"records": []}
    accepted = {r["source_row"]: r["annotation_sha256"] for r in acceptance["records"]}
    quality_path = WORK / "source_quality_notes.json"
    quality_notes = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    run_path = WORK / "active_run.json"
    last_source_row = json.loads(run_path.read_text(encoding="utf-8"))["last_source_row"] if run_path.exists() else len(rows)
    decisions, errors = {}, []
    for path in sorted((WORK / "decisions").glob("*.jsonl")):
        for record in read_jsonl(path):
            number, selections, *note = record
            if number in decisions or type(number) is not int or not 1 <= number <= len(rows):
                raise ValueError(f"Duplicate or invalid row number {number} in {path.name}")
            try:
                spans = resolve(rows[number - 1]["text"], selections)
                decisions[number] = {"label": spans, "review_note": note[0] if note else "", "decision_file": path.name}
            except (ValueError, TypeError) as error:
                errors.append(f"row {number}: {error}")
    if errors:
        raise ValueError("\n".join(errors))

    by_text = {}
    for number, decision in decisions.items():
        row = rows[number - 1]
        key = (row["document_id"], row["text"])
        if key in by_text and decisions[by_text[key]]["label"] != decision["label"]:
            raise ValueError(f"Conflicting decisions for identical sentence in same article: {number}, {by_text[key]}")
        by_text.setdefault(key, number)

    annotated, pending, provenance, review = [], [], [], []
    for number, row in enumerate(rows, 1):
        donor = number if number in decisions else by_text.get((row["document_id"], row["text"]))
        if number > last_source_row:
            donor = None
        if donor is None:
            pending.append(row)
            continue
        decision = decisions[donor]
        review_note = " ".join(note for note in [decision["review_note"], quality_notes.get(row["document_id"], "")] if note)
        output = {"document_id": row["document_id"], "text": row["text"], "label": decision["label"]}
        if number in accepted:
            digest = hashlib.sha256(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if digest != accepted[number]:
                raise ValueError(f"Previously user-accepted annotation changed at source row {number}")
        annotated.append(output)
        entry = {
            "output_row": len(annotated), "source_row": number,
            "sentence_id": row["sentence_id"], "document_id": row["document_id"],
            "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            "method": "direct_agent_annotation" if donor == number else "exact_same_article_duplicate",
            "decision_source_row": donor, "decision_file": decision["decision_file"],
            "review_status": "user_reported_reviewed_accepted" if number in accepted else "agent_annotated_not_human_reviewed",
            "review_note": review_note,
        }
        provenance.append(entry)
        if review_note and number not in accepted:
            review.append({"source_row": number, "sentence_id": row["sentence_id"], "text": row["text"],
                           "entities": json.dumps([[row["text"][s:e], label] for s, e, label in decision["label"]], ensure_ascii=False),
                           "review_note": review_note})

    if not set(accepted).issubset({p["source_row"] for p in provenance}):
        raise ValueError("Previously user-accepted rows are missing from compiled output")
    write_jsonl(ROOT / "data/train_agent_annotated.jsonl", annotated)
    write_jsonl(ROOT / "data/train_agent_new_since_review.jsonl", [row for row, meta in zip(annotated, provenance) if meta["source_row"] not in accepted])
    write_jsonl(ROOT / "data/train_agent_pending.jsonl", pending)
    write_jsonl(WORK / "provenance.jsonl", provenance)
    with (WORK / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_row", "sentence_id", "text", "entities", "review_note"])
        writer.writeheader()
        writer.writerows(review)
    counts = Counter(entity[2] for row in annotated for entity in row["label"])
    processed_numbers = {p["source_row"] for p in provenance}
    source_docs = Counter(r["document_id"] for r in rows)
    annotated_docs = Counter(r["document_id"] for r in annotated)
    summary = {
        "source_sha256": source_hash, "source_rows": len(rows),
        "annotated_rows": len(annotated), "direct_decisions": len(decisions),
        "same_article_exact_duplicates_propagated": len(annotated) - len(decisions),
        "pending_rows": len(pending), "annotated_documents": len({r["document_id"] for r in annotated}),
        "fully_annotated_documents": sum(count == source_docs[doc] for doc, count in annotated_docs.items()),
        "partially_annotated_documents": sum(count < source_docs[doc] for doc, count in annotated_docs.items()),
        "entity_count": sum(counts.values()), "label_counts": dict(sorted(counts.items())),
        "explicitly_annotated_empty_rows": sum(not r["label"] for r in annotated),
        "flagged_rows": len(review), "complete": not pending,
        "user_reported_reviewed_rows": len(accepted),
        "new_rows_since_user_review": len(annotated) - len(accepted),
        "first_pending_source_row": next((i for i in range(1, len(rows) + 1) if i not in processed_numbers), None),
        "annotation_type": "AI-generated training annotations; user-reported review recorded for accepted rows only",
        "test_use": "Guideline/schema and eight reviewed examples inspected for conventions; no test labels copied or used as training examples",
    }
    (WORK / "progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    next_batch = []
    for number, row in enumerate(rows, 1):
        if number not in processed_numbers:
            next_batch.append({"source_row": number, "sentence_id": row["sentence_id"],
                               "document_id": row["document_id"], "text": row["text"]})
            if len(next_batch) == 50:
                break
    write_jsonl(WORK / "next_batch.jsonl", next_batch)
    write_preview(annotated, provenance, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def write_preview(annotated, provenance, summary):
    template = Path(__file__).with_name("annotation_preview_template.html").read_text(encoding="utf-8")
    payload = json.dumps({"summary": summary, "rows": [dict(row, **meta) for row, meta in zip(annotated, provenance)]}, ensure_ascii=False).replace("<", "\\u003c")
    (WORK / "review.html").write_text(template.replace("/*ANNOTATION_DATA*/null", payload), encoding="utf-8")


if __name__ == "__main__":
    main()
