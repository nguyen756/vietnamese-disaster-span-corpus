#!/usr/bin/env python3
"""Split sentence records by document into a reproducible gold-test holdout."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_records(input_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: List[Tuple[str, Dict[str, Any]]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            document_id = record.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(
                    f"Missing document_id on line {line_number} of {input_path}"
                )
            records.append((line, record))
    return records


def select_test_documents(
    document_counts: Counter[str],
    target_sentences: int,
    seed: int,
) -> Tuple[set[str], int]:
    if target_sentences <= 0:
        raise ValueError("target_sentences must be greater than zero")
    if not document_counts:
        raise ValueError("Input contains no sentence records")

    rng = random.Random(seed)
    documents = list(document_counts.items())
    rng.shuffle(documents)

    search_limit = target_sentences + max(document_counts.values())
    reachable: Dict[int, Tuple[str, ...]] = {0: ()}
    for document_id, sentence_count in documents:
        additions: Dict[int, Tuple[str, ...]] = {}
        for current_count, selected_ids in reachable.items():
            new_count = current_count + sentence_count
            if new_count <= search_limit and new_count not in reachable:
                additions[new_count] = selected_ids + (document_id,)
        reachable.update(additions)

    best_count = min(
        reachable,
        key=lambda count: (
            abs(count - target_sentences),
            count > target_sentences,
        ),
    )
    return set(reachable[best_count]), best_count


def write_jsonl(path: Path, lines: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible sentence-level holdout by document_id."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="raw/scoped_sentences.jsonl",
        help="Input sentence JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="raw",
        help="Directory for gold_test_set.jsonl, train_candidate.jsonl, and manifest",
    )
    parser.add_argument(
        "--target-sentences",
        type=int,
        default=1000,
        help="Target number of test sentences; complete documents are kept together",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to choose among document subsets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_paths = {
        "gold_test_set.jsonl": output_dir / "gold_test_set.jsonl",
        "train_candidate.jsonl": output_dir / "train_candidate.jsonl",
        "split_manifest.json": output_dir / "split_manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output files: "
            + ", ".join(str(path) for path in existing)
        )

    records = load_records(input_path)
    document_counts: Counter[str] = Counter(
        record["document_id"] for _, record in records
    )
    test_document_ids, test_sentence_count = select_test_documents(
        document_counts,
        args.target_sentences,
        args.seed,
    )

    gold_lines = [
        line for line, record in records
        if record["document_id"] in test_document_ids
    ]
    train_lines = [
        line for line, record in records
        if record["document_id"] not in test_document_ids
    ]
    actual_test_count = len(gold_lines)
    if actual_test_count != test_sentence_count:
        raise RuntimeError("Selected document sentence count did not validate")

    train_document_ids = set(document_counts) - test_document_ids
    if test_document_ids & train_document_ids:
        raise RuntimeError("Document leakage detected between test and train")
    if len(gold_lines) + len(train_lines) != len(records):
        raise RuntimeError("Split did not preserve all input records")

    selected_document_counts = Counter(
        record["document_id"]
        for _, record in records
        if record["document_id"] in test_document_ids
    )
    metadata_by_document: Dict[str, Dict[str, str]] = {}
    for _, record in records:
        document_id = record["document_id"]
        if document_id in test_document_ids and document_id not in metadata_by_document:
            metadata_by_document[document_id] = {
                "source_name": str(record.get("source_name", "") or ""),
                "title": str(record.get("title", "") or ""),
            }

    package_root = Path(__file__).resolve().parent.parent
    try:
        manifest_input = input_path.relative_to(package_root).as_posix()
    except ValueError:
        manifest_input = input_path.name
    manifest = {
        "input_file": manifest_input,
        "split_unit": "document_id",
        "random_seed": args.seed,
        "target_test_sentences": args.target_sentences,
        "actual_test_sentences": actual_test_count,
        "difference_from_target": actual_test_count - args.target_sentences,
        "total_sentences": len(records),
        "total_documents": len(document_counts),
        "test_documents": len(test_document_ids),
        "train_documents": len(train_document_ids),
        "test_document_sentence_counts": [
            {
                "document_id": document_id,
                "sentence_count": selected_document_counts[document_id],
                **metadata_by_document[document_id],
            }
            for document_id in sorted(test_document_ids)
        ],
        "outputs": {
            "gold_test_set": output_paths["gold_test_set.jsonl"].name,
            "train_candidate": output_paths["train_candidate.jsonl"].name,
        },
        "annotation_note": (
            "matched_categories and scope_reasons are automatic filtering "
            "metadata, not gold NER labels."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_directory:
        temporary_dir = Path(temporary_directory)
        temporary_paths = {
            name: temporary_dir / name for name in output_paths
        }
        write_jsonl(temporary_paths["gold_test_set.jsonl"], gold_lines)
        write_jsonl(temporary_paths["train_candidate.jsonl"], train_lines)
        temporary_paths["split_manifest.json"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output files: "
                + ", ".join(str(path) for path in existing)
            )
        for name, output_path in output_paths.items():
            temporary_paths[name].replace(output_path)

    print(f"Gold test: {actual_test_count} sentences from {len(test_document_ids)} documents")
    print(f"Train candidate: {len(train_lines)} sentences from {len(train_document_ids)} documents")
    print(f"Target difference: {actual_test_count - args.target_sentences}")
    for path in output_paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
