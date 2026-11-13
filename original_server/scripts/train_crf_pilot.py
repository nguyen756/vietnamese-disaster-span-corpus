"""Reproducible NER feasibility experiment; never trains on or evaluates Gold Test.

Original annotations stay immutable. Split by article components connected by
normalized duplicate sentences, remove within-article repetitions, and exclude
the documented hard-linebreak regulation from this sentence-level experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def tokenize(text):
    """Text-only Unicode word/punctuation tokens, with generic camel-case splits.

    No gold entity boundaries, labels or normalization are used to create tokens.
    Return original Unicode code-point offsets, including punctuation tokens.
    """
    result = []
    for match in re.finditer(r"\w+|[^\w\s]", text):
        start, end = match.span()
        breaks = [start] + [i for i in range(start + 1, end)
                            if text[i - 1].islower() and text[i].isupper()] + [end]
        result.extend((a, b, text[a:b]) for a, b in zip(breaks, breaks[1:]))
    return result


def encode(row):
    tokens = tokenize(row["text"])
    starts = {a: i for i, (a, _, _) in enumerate(tokens)}
    ends = {b: i for i, (_, b, _) in enumerate(tokens)}
    tags, unaligned = ["O"] * len(tokens), []
    for start, end, label in row["label"]:
        if start not in starts or end not in ends:
            unaligned.append([start, end, label])
            continue
        first, last = starts[start], ends[end]
        assert all(tag == "O" for tag in tags[first:last + 1]), "Overlapping input annotations"
        tags[first] = "B-" + label
        tags[first + 1:last + 1] = ["I-" + label] * (last - first)
    return tokens, tags, unaligned


def decode(tokens, tags):
    """Repair an orphan I as B; reconstruct end-exclusive source offsets."""
    assert len(tokens) == len(tags)
    spans, current = [], None
    for (start, end, _), tag in zip(tokens, tags):
        if tag == "O":
            if current:
                spans.append(current)
                current = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or current is None or current[2] != label:
            if current:
                spans.append(current)
            current = [start, end, label]
        else:
            current[1] = end
    if current:
        spans.append(current)
    return spans


def connected_groups(rows):
    """Article components prevent identical normalized text crossing the split."""
    parent = {row["document_id"]: row["document_id"] for row in rows}

    def find(doc):
        while parent[doc] != doc:
            parent[doc] = parent[parent[doc]]
            doc = parent[doc]
        return doc

    seen = {}
    for row in rows:
        doc, key = row["document_id"], normalize(row["text"])
        if key in seen:
            left, right = sorted([find(doc), find(seen[key])])
            parent[right] = left
        seen.setdefault(key, doc)
    return {doc: find(doc) for doc in parent}


def shape(word):
    value = "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c for c in word)
    return re.sub(r"(.)\1+", r"\1", value)


def features(tokens):
    words = [token[2] for token in tokens]
    result = []
    for i, word in enumerate(words):
        lower = word.lower()
        feat = {"bias": 1.0, "word": lower, "shape": shape(word),
                "title": word.istitle(), "upper": word.isupper(),
                "digit": word.isdigit(), "has_digit": any(c.isdigit() for c in word)}
        for size in (1, 2, 3, 4):
            feat[f"prefix{size}"] = lower[:size]
            feat[f"suffix{size}"] = lower[-size:]
        for offset in (-2, -1, 1, 2):
            position = i + offset
            prefix = f"{offset}:"
            if 0 <= position < len(words):
                neighbor = words[position]
                feat.update({prefix + "word": neighbor.lower(), prefix + "title": neighbor.istitle(),
                             prefix + "digit": neighbor.isdigit(), prefix + "shape": shape(neighbor)})
            else:
                feat[prefix + "boundary"] = True
        result.append(feat)
    return result


def score(rows, predictions, labels):
    """Entity identity requires matching start, end AND class; O never scores."""
    assert len(rows) == len(predictions), "Every reference sentence needs a prediction"
    counts = {label: Counter() for label in labels}
    for row, predicted in zip(rows, predictions):
        gold = {tuple(span) for span in row["label"]}
        pred = {tuple(span) for span in predicted}
        for span in gold & pred:
            counts[span[2]]["tp"] += 1
        for span in pred - gold:
            counts[span[2]]["fp"] += 1
        for span in gold - pred:
            counts[span[2]]["fn"] += 1

    def metrics(tp=0, fp=0, fn=0):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": precision, "recall": recall,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "support": tp + fn, "predicted": tp + fp, "tp": tp, "fp": fp, "fn": fn}

    per_label = {label: metrics(**counts[label]) for label in labels}
    total = sum(counts.values(), Counter())
    return {"micro": metrics(**total), "macro_f1": sum(m["f1"] for m in per_label.values()) / len(labels),
            "per_label": per_label, "sentences": len(rows),
            "exact_sentence_accuracy": sum(row["label"] == predicted for row, predicted in zip(rows, predictions)) / len(rows) if rows else 0.0}


def error_analysis(rows, predictions):
    errors, categories = [], Counter()
    for row, pred_list in zip(rows, predictions):
        gold, predicted = {tuple(s) for s in row["label"]}, {tuple(s) for s in pred_list}
        if gold == predicted:
            continue
        missed, extra = sorted(gold - predicted), sorted(predicted - gold)
        details = []
        for start, end, label in missed:
            if any(a == start and b == end for a, b, _ in extra):
                category = "wrong_label_same_boundary"
            elif any(other == label and a < end and start < b for a, b, other in extra):
                category = "same_label_boundary_error"
            else:
                category = "missed_entity"
            categories[category] += 1
            details.append({"span": [start, end, label], "surface": row["text"][start:end], "category": category})
        errors.append({"source_row": row["source_row"], "document_id": row["document_id"],
                       "text": row["text"], "gold": row["label"], "predicted": pred_list,
                       "missing": details,
                       "extra": [{"span": list(s), "surface": row["text"][s[0]:s[1]]} for s in extra]})
    return errors, dict(categories)


def subset(rows, groups, target, seed):
    if target == "all":
        return rows
    ordered = sorted({groups[row["document_id"]] for row in rows})
    random.Random(seed).shuffle(ordered)
    selected, count = set(), 0
    sizes = Counter(groups[row["document_id"]] for row in rows)
    for group in ordered:
        selected.add(group)
        count += sizes[group]
        if count >= int(target):
            break
    return [row for row in rows if groups[row["document_id"]] in selected]


def main():
    import sklearn_crfsuite
    from sklearn.model_selection import GroupShuffleSplit

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/crf_pilot_0001")
    parser.add_argument("--sizes", nargs="+", default=["500", "1000", "all"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("Use a new output directory to preserve an existing experiment")
    protected = ["data/train_agent_annotated.jsonl", "data/gold_test_final.jsonl",
                 "annotation_work/provenance.jsonl", "documentation/annotation_guideline.md"]
    before = {name: sha(ROOT / name) for name in protected}
    data = read_jsonl(ROOT / protected[0])
    provenance = read_jsonl(ROOT / "annotation_work/provenance.jsonl")
    assert len(data) == len(provenance)
    excluded_docs = json.loads((ROOT / "annotation_work/source_quality_notes.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "data/annotations/annotation_schema.json").read_text(encoding="utf-8"))["labels"]
    rows, excluded, duplicate_rows, seen = [], [], [], set()
    for row, meta in zip(data, provenance):
        assert row["document_id"] == meta["document_id"]
        item = dict(row, source_row=meta["source_row"], review_status=meta["review_status"], review_note=meta["review_note"])
        if row["document_id"] in excluded_docs:
            excluded.append(dict(item, exclusion_reason=excluded_docs[row["document_id"]]))
            continue
        key = (row["document_id"], normalize(row["text"]), tuple(tuple(s) for s in row["label"]))
        if key in seen:
            duplicate_rows.append(item)
            continue
        seen.add(key)
        rows.append(item)
    groups = connected_groups(rows)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(rows, groups=[groups[r["document_id"]] for r in rows]))
    train, validation = [rows[i] for i in train_idx], [rows[i] for i in val_idx]
    train_texts, val_texts = {normalize(r["text"]) for r in train}, {normalize(r["text"]) for r in validation}
    assert not train_texts & val_texts
    assert not {r["document_id"] for r in train} & {r["document_id"] for r in validation}
    encoded = {r["source_row"]: encode(r) for r in rows}
    unaligned = [dict(r, unaligned_spans=encoded[r["source_row"]][2]) for r in rows if encoded[r["source_row"]][2]]
    train_bad = {r["source_row"] for r in unaligned}
    train = [r for r in train if r["source_row"] not in train_bad]
    for row in rows:
        tokens, tags, bad = encoded[row["source_row"]]
        if not bad:
            assert decode(tokens, tags) == row["label"], f"BIO round trip failed at {row['source_row']}"
    x = {r["source_row"]: features(encoded[r["source_row"]][0]) for r in rows}
    for name, contents in [("train", train), ("validation", validation), ("excluded_source_fragments", excluded),
                           ("removed_within_article_duplicates", duplicate_rows), ("unaligned_rows", unaligned)]:
        write_jsonl(output / (name + ".jsonl"), contents)
    split = {"seed": args.seed, "validation_fraction_of_article_components": 0.2,
             "original_rows": len(data), "excluded_fragment_rows": len(excluded),
             "within_article_duplicates_removed": len(duplicate_rows), "usable_unique_rows": len(rows),
             "article_components": len(set(groups.values())), "unaligned_rows": len(unaligned),
             "unaligned_train_rows_excluded": sum(r["source_row"] in train_bad for r in [rows[i] for i in train_idx]),
             "training_rows": len(train), "validation_rows": len(validation),
             "training_articles": len({r["document_id"] for r in train}),
             "validation_articles": len({r["document_id"] for r in validation}),
             "validation_reviewed_rows": sum(r["review_status"] == "user_reported_reviewed_accepted" for r in validation),
             "normalized_text_overlap": 0, "article_overlap": 0,
             "gold_test_used": False, "document_groups": groups,
             "train_label_counts": dict(Counter(s[2] for r in train for s in r["label"])),
             "validation_label_counts": dict(Counter(s[2] for r in validation for s in r["label"]))}
    write_json(output / "split_manifest.json", split)
    params = {"algorithm": "lbfgs", "c1": 0.1, "c2": 0.1, "max_iterations": args.iterations,
              "all_possible_transitions": True}
    result = {"experiment": "CRF feasibility pilot", "metric": "Exact Unicode character-span and class micro-F1; excludes O",
              "input_sha256": before, "split": split, "parameters": params,
              "versions": {name: importlib.metadata.version(name) for name in ["sklearn-crfsuite", "python-crfsuite", "scikit-learn", "numpy"]},
              "python": platform.python_version(), "learning_curve": [],
              "outside_only_baseline": score(validation, [[] for _ in validation], labels)}
    print(json.dumps({k: v for k, v in split.items() if k != "document_groups"}), flush=True)
    for size in args.sizes:
        selected = subset(train, groups, size, args.seed)
        name = f"crf_{size}"
        print(f"Training {name}: {len(selected)} rows from {len({r['document_id'] for r in selected})} articles", flush=True)
        model = sklearn_crfsuite.CRF(**params, model_filename=str(output / (name + ".crfsuite")), keep_tempfiles=True)
        start = time.perf_counter()
        model.fit([x[r["source_row"]] for r in selected], [encoded[r["source_row"]][1] for r in selected])
        seconds = time.perf_counter() - start
        def predict(items):
            tags = model.predict([x[r["source_row"]] for r in items])
            return [decode(encoded[r["source_row"]][0], y) for r, y in zip(items, tags)]
        predictions = predict(validation)
        train_predictions = predict(selected)
        errors, categories = error_analysis(validation, predictions)
        report = {"requested_size": size, "training_rows": len(selected),
                  "training_articles": len({r["document_id"] for r in selected}),
                  "training_entity_counts": dict(Counter(s[2] for r in selected for s in r["label"])),
                  "train_seconds": seconds, "training": score(selected, train_predictions, labels),
                  "validation": score(validation, predictions, labels), "error_categories": categories,
                  "model_file": name + ".crfsuite"}
        for status, key in [("user_reported_reviewed_accepted", "reviewed_validation"),
                            ("agent_annotated_not_human_reviewed", "unreviewed_validation")]:
            subset_pairs = [(r, pred) for r, pred in zip(validation, predictions) if r["review_status"] == status]
            report[key] = score([r for r, _ in subset_pairs], [pred for _, pred in subset_pairs], labels)
        result["learning_curve"].append(report)
        write_json(output / (name + "_metrics.json"), report)
        write_jsonl(output / (name + "_errors.jsonl"), errors)
        write_jsonl(output / (name + "_predictions.jsonl"),
                    [dict(r, predicted_label=pred) for r, pred in zip(validation, predictions)])
        write_jsonl(output / (name + "_training_source_rows.jsonl"), [{"source_row": r["source_row"]} for r in selected])
        print(json.dumps({"model": name, "seconds": round(seconds, 2), "train_f1": report["training"]["micro"]["f1"],
                          "validation": report["validation"]["micro"]}), flush=True)
    assert before == {name: sha(ROOT / name) for name in protected}, "Protected input changed during experiment"
    result["protected_inputs_unchanged"] = True
    write_json(output / "results.json", result)
    with (output / "per_label.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "precision", "recall", "f1", "support", "predicted", "tp", "fp", "fn"])
        writer.writeheader()
        for label, metrics in result["learning_curve"][-1]["validation"]["per_label"].items():
            writer.writerow(dict(label=label, **metrics))
    print(f"Complete. Results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
