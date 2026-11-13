"""Freeze the selected 12,680-row subset and preserve the deferred 1,563 rows.

Never changes canonical labels, review acceptance, source files or old runs.
The existing 803-row development split is retained, including its review status.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from train_crf_pilot import connected_groups, decode, encode, normalize

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
CUTOFF = 12680


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"Existing output differs: {path}"
    else:
        path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    release = PACKAGE / "final_work_12680_v1"
    deferred = PACKAGE / "deferred_annotation_1563"
    if (release / "split_manifest.json").exists() or (deferred / "manifest.json").exists():
        raise FileExistsError("Completed snapshots exist; use a new version.")
    if (release.exists() or deferred.exists()) and not args.resume_incomplete:
        raise FileExistsError("Preserve existing snapshots. Use a new version instead of overwriting.")
    relative_inputs = [
        "data/train_agent_annotated.jsonl", "data/train_candidate_clean.jsonl",
        "annotation_work/provenance.jsonl", "annotation_work/review_acceptances.json",
        "annotation_work/source_quality_notes.json", "data/gold_test_final.jsonl",
        "documentation/annotation_guideline.md", "data/annotations/annotation_schema.json",
        "experiments/crf_pilot_0001/validation.jsonl",
    ]
    input_hashes = {name: sha(ROOT / name) for name in relative_inputs}
    annotations = read(ROOT / relative_inputs[0])
    source = read(ROOT / relative_inputs[1])
    provenance = read(ROOT / relative_inputs[2])
    gold = read(ROOT / "data/gold_test_final.jsonl")
    validation = read(ROOT / "experiments/crf_pilot_0001/validation.jsonl")
    quality = json.loads((ROOT / "annotation_work/source_quality_notes.json").read_text(encoding="utf-8"))
    assert len(annotations) == len(source) == len(provenance) == 14243
    assert [m["source_row"] for m in provenance] == list(range(1, 14244))
    assert all((a["document_id"], a["text"]) == (s["document_id"], s["text"])
               for a, s in zip(annotations, source))
    selected, queued = annotations[:CUTOFF], annotations[CUTOFF:]
    assert len(selected) == 12680 and len(queued) == 1563
    items = [dict(a, source_row=m["source_row"], review_status=m["review_status"],
                  review_note=m["review_note"]) for a, m in zip(selected, provenance)]
    # Compute article connections without labels; keep connected new articles out
    # of training even when their shared text is absent from the old dev export.
    eligible = [r for r in items if r["document_id"] not in quality]
    groups = connected_groups(eligible)
    validation_ids = {r["source_row"] for r in validation}
    assert len(validation_ids) == len(validation) == 803
    assert all(r == items[r["source_row"] - 1] for r in validation), "Frozen dev references changed"
    held_groups = {groups[r["document_id"]] for r in validation}
    training, excluded, seen = [], [], set()
    for row in items:
        n = row["source_row"]
        reason = None
        if row["document_id"] in quality:
            reason = "documented_broken_regulation_fragments"
        elif n in validation_ids:
            continue
        elif groups[row["document_id"]] in held_groups:
            reason = "article_component_reserved_for_development"
        else:
            tokens, tags, bad = encode(row)
            if bad:
                reason = "span_boundary_not_representable_by_existing_text_only_tokenizer"
            else:
                assert decode(tokens, tags) == row["label"]
                key = (row["document_id"], normalize(row["text"]),
                       tuple(tuple(s) for s in row["label"]))
                if key in seen:
                    reason = "identical_within_article_text_and_labels"
                else:
                    seen.add(key)
        if reason:
            excluded.append(dict(row, exclusion_reason=reason))
        else:
            training.append(row)
    assert len(training) + len(validation) + len(excluded) == CUTOFF
    assert not {r["source_row"] for r in training} & validation_ids
    assert not {r["document_id"] for r in training} & {r["document_id"] for r in validation}
    assert not {normalize(r["text"]) for r in training} & {normalize(r["text"]) for r in validation}
    assert not {groups[r["document_id"]] for r in training} & held_groups
    assert not {normalize(r["text"]) for r in selected} & {normalize(r["text"]) for r in gold}
    assert not {r["document_id"] for r in selected} & {r["document_id"] for r in gold}
    assert all(not encode(r)[2] and decode(*encode(r)[:2]) == r["label"] for r in validation)
    decisions_by_file = []
    decision_lookup = {}
    for path in sorted((ROOT / "annotation_work/decisions").glob("*.jsonl")):
        for record in read(path):
            if record[0] > CUTOFF:
                decisions_by_file.append({"decision_file": path.name, "record": record})
                assert (path.name, record[0]) not in decision_lookup
                decision_lookup[path.name, record[0]] = record
    decisions = [decision_lookup[m["decision_file"], m["source_row"]] for m in provenance[CUTOFF:]]
    assert len(decisions) == len({r[0] for r in decisions}) == 1563
    assert {r[0] for r in decisions} == set(range(CUTOFF + 1, 14244))
    release.mkdir(exist_ok=True)
    deferred.mkdir(exist_ok=True)
    write_rows(release / "annotations_done_12680.jsonl", selected)
    write_rows(release / "provenance_done_12680.jsonl", provenance[:CUTOFF])
    write_rows(release / "train.jsonl", training)
    # Preserve the exact frozen development bytes as well as its records.
    shutil.copyfile(ROOT / "experiments/crf_pilot_0001/validation.jsonl", release / "validation.jsonl")
    write_rows(release / "excluded_from_training.jsonl", excluded)
    write_rows(deferred / "annotations_needing_repair.jsonl", queued)
    write_rows(deferred / "source_rows_12681_14243.jsonl", source[CUTOFF:])
    write_rows(deferred / "provenance_original.jsonl", provenance[CUTOFF:])
    write_rows(deferred / "original_decisions.jsonl", sorted(decisions, key=lambda r: r[0]))
    write_rows(deferred / "all_decisions_with_filenames.jsonl", decisions_by_file)
    shutil.copyfile(ROOT / "documentation/annotation_guideline.md", deferred / "annotation_guideline.md")
    shutil.copyfile(ROOT / "data/annotations/annotation_schema.json", deferred / "annotation_schema.json")
    label_counts = lambda rows: dict(sorted(Counter(s[2] for r in rows for s in r["label"]).items()))
    manifest = {
        "created_date": "2026-11-06", "release": release.name,
        "selection": "Original source rows 1-12680, selected by user for current experiments; not a new human-review claim",
        "selected_rows": len(selected), "selected_entity_spans": sum(len(r["label"]) for r in selected),
        "selected_articles": len({r["document_id"] for r in selected}),
        "deferred_rows": len(queued), "deferred_first_source_row": CUTOFF + 1,
        "deferred_last_source_row": len(source), "training_rows": len(training),
        "validation_rows": len(validation), "excluded_rows": len(excluded),
        "exclusion_counts": dict(Counter(r["exclusion_reason"] for r in excluded)),
        "training_articles": len({r["document_id"] for r in training}),
        "validation_articles": len({r["document_id"] for r in validation}),
        "selected_label_counts": label_counts(selected),
        "train_label_counts": label_counts(training), "validation_label_counts": label_counts(validation),
        "selected_recorded_accepted_rows": sum(m["review_status"] == "user_reported_reviewed_accepted" for m in provenance[:CUTOFF]),
        "validation_recorded_accepted_rows": sum(r["review_status"] == "user_reported_reviewed_accepted" for r in validation),
        "gold_test_rows": len(gold), "gold_test_used_for_training_or_scoring": False,
        "split_policy": "Retain exact old 803-row dev; exclude all connected article components from training",
        "checks": {"deferred_rows_absent_from_training_and_validation": True,
                   "all_included_spans_round_trip": True, "train_dev_document_overlap": 0,
                   "train_dev_normalized_text_overlap": 0, "selected_gold_document_overlap": 0,
                   "selected_gold_normalized_text_overlap": 0},
        "source_inputs_relative_to_final_v2": input_hashes,
        "limitations": ["Known semantic errors in selected rows remain unchanged under the user's scope decision",
                        "Most selected annotations and 583/803 development references lack recorded human acceptance",
                        "Exact duplicate/article checks do not establish absence of near-duplicate or event overlap"],
    }
    manifest["files"] = {p.name: sha(p) for p in sorted(release.glob("*.jsonl"))}
    write_json(release / "split_manifest.json", manifest)
    queue_manifest = {
        "status": "Deferred by user; not input to current experiments",
        "source_rows": [12681, 14243], "rows": len(queued),
        "entity_spans_in_unrepaired_labels": sum(len(r["label"]) for r in queued),
        "reason": "Keyword-derived candidate labels; missing LOCATION/TIME and other semantic issues",
        "decision_records_including_duplicate_files": len(decisions_by_file),
        "active_decisions_selected_by_provenance": len(decisions),
        "duplicate_file_note": "batch_0111/0112 duplicate source rows in batch_0109/0110. Preserved for later reconciliation; do not blindly recompile all decision files.",
        "canonical_inputs_unchanged": True,
        "source_inputs_relative_to_final_v2": input_hashes,
        "files": {p.name: sha(p) for p in sorted(deferred.iterdir()) if p.is_file()},
    }
    write_json(deferred / "manifest.json", queue_manifest)
    assert input_hashes == {name: sha(ROOT / name) for name in relative_inputs}
    print(json.dumps({k: manifest[k] for k in ["selected_rows", "selected_entity_spans", "training_rows",
                     "validation_rows", "excluded_rows", "exclusion_counts", "deferred_rows"]}, indent=2))


if __name__ == "__main__":
    main()
