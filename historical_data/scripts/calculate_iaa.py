#!/usr/bin/env python3
"""Validate two Doccano exports and calculate inter-annotator agreement."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


LABELS: Tuple[str, ...] = (
    "DISASTER_TYPE",
    "WEATHER_CAUSE",
    "LOCATION",
    "TIME",
    "DAMAGE",
    "CASUALTY",
    "AFFECTED_POPULATION",
    "RESPONSE_ACTION",
    "RESPONSE_AGENCY",
    "WARNING_LEVEL",
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
DISAGREEMENT_CATEGORIES: Tuple[str, ...] = (
    "LABEL_DISAGREEMENT",
    "BOUNDARY_DISAGREEMENT",
    "BOUNDARY_AND_LABEL_DISAGREEMENT",
    "ANNOTATOR_1_ONLY",
    "ANNOTATOR_2_ONLY",
    "NO_ENTITY_AGREEMENT",
)
EntityTuple = Tuple[int, int, str]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Validate two Doccano JSONL annotations and calculate IAA."
    )
    parser.add_argument(
        "annotator_1",
        nargs="?",
        default=str(project_dir / "Cohen" / "gold_test_1.jsonl"),
        help="Annotator 1 Doccano JSONL file",
    )
    parser.add_argument(
        "annotator_2",
        nargs="?",
        default=str(project_dir / "Cohen" / "gold_test_2.jsonl"),
        help="Annotator 2 Doccano JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "Cohen" / "IAA_results"),
        help="Directory for generated reports (default: Cohen/IAA_results)",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent / candidate
    return candidate.resolve()


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(
                    {"line": line_number, "error": "Blank lines are not valid records."}
                )
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line": line_number,
                        "error": f"Invalid JSON: {exc.msg}",
                        "column": exc.colno,
                    }
                )
                continue
            if not isinstance(record, dict):
                errors.append(
                    {"line": line_number, "error": "Record must be a JSON object."}
                )
                continue
            records.append(record)
    return records, errors


def get_document_id(record: Dict[str, Any]) -> Any:
    document_id = record.get("document_id")
    if document_id is not None:
        return document_id
    return record.get("id")


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Split Unicode word runs and punctuation, also splitting lower-to-upper joins."""
    tokens: List[Tuple[str, int, int]] = []
    for match in TOKEN_PATTERN.finditer(text):
        value = match.group()
        start = match.start()
        if value.isalnum() or "_" in value:
            boundaries = [0]
            boundaries.extend(
                index
                for index in range(1, len(value))
                if (
                    value[index - 1].islower() and value[index].isupper()
                )
                or (
                    value[index - 1].isdigit() and value[index].isalpha()
                )
            )
            boundaries.append(len(value))
            for left, right in zip(boundaries, boundaries[1:]):
                tokens.append(
                    (value[left:right], start + left, start + right)
                )
        else:
            tokens.append((value, match.start(), match.end()))
    return tokens


def validate_annotations(
    records: Sequence[Dict[str, Any]], annotator: str
) -> Tuple[List[Dict[str, Any]], int]:
    errors: List[Dict[str, Any]] = []
    unaligned_entity_count = 0

    for record_index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str):
            errors.append(
                {
                    "annotator": annotator,
                    "record_index": record_index,
                    "document_id": get_document_id(record),
                    "error": "The text field must be a string.",
                }
            )
            continue

        labels = record.get("label")
        if not isinstance(labels, list):
            errors.append(
                {
                    "annotator": annotator,
                    "record_index": record_index,
                    "document_id": get_document_id(record),
                    "error": "The label field must be a list.",
                }
            )
            continue

        tokens = tokenize(text)
        token_starts = {start for _, start, _ in tokens}
        token_ends = {end for _, _, end in tokens}
        valid_entities: List[Tuple[int, int, int]] = []
        seen: Set[EntityTuple] = set()

        for entity_index, entity in enumerate(labels):
            location = {
                "annotator": annotator,
                "record_index": record_index,
                "document_id": get_document_id(record),
                "entity_index": entity_index,
            }
            if not isinstance(entity, list) or len(entity) != 3:
                errors.append(
                    {**location, "error": "Entity must be [start, end, label]."}
                )
                continue

            start, end, label = entity
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
            ):
                errors.append(
                    {
                        **location,
                        "error": "Entity offsets must be integers (not booleans).",
                        "entity": entity,
                    }
                )
                continue

            if not isinstance(label, str) or label not in LABELS:
                errors.append(
                    {
                        **location,
                        "error": "Unknown entity label.",
                        "entity": entity,
                        "allowed_labels": list(LABELS),
                    }
                )

            if not (0 <= start < end <= len(text)):
                errors.append(
                    {
                        **location,
                        "error": "Entity offsets are outside the text or have an empty span.",
                        "entity": entity,
                        "text_length": len(text),
                    }
                )
                continue

            extracted = text[start:end]
            if not extracted:
                errors.append(
                    {**location, "error": "Entity span extracts an empty substring."}
                )
                continue

            if isinstance(label, str):
                entity_key = (start, end, label)
                if entity_key in seen:
                    errors.append(
                        {
                            **location,
                            "error": "Duplicated entity annotation.",
                            "entity": entity_key,
                        }
                    )
                seen.add(entity_key)
            valid_entities.append((start, end, entity_index))
            if start not in token_starts or end not in token_ends:
                unaligned_entity_count += 1

        valid_entities.sort()
        for left_index, left in enumerate(valid_entities):
            for right in valid_entities[left_index + 1 :]:
                if right[0] < left[1]:
                    errors.append(
                        {
                            "annotator": annotator,
                            "record_index": record_index,
                            "document_id": get_document_id(record),
                            "error": "Overlapping or nested entities.",
                            "entity_indices": [left[2], right[2]],
                            "first_span": [left[0], left[1]],
                            "second_span": [right[0], right[1]],
                            "first_text": text[left[0] : left[1]],
                            "second_text": text[right[0] : right[1]],
                        }
                    )

        token_entity_counts = [0] * len(tokens)
        for start, end, _ in valid_entities:
            for token_index, (_, token_start, token_end) in enumerate(tokens):
                if token_start < end and start < token_end:
                    token_entity_counts[token_index] += 1
        collisions = [
            index for index, count in enumerate(token_entity_counts) if count > 1
        ]
        if collisions:
            errors.append(
                {
                    "annotator": annotator,
                    "record_index": record_index,
                    "document_id": get_document_id(record),
                    "error": (
                        "Multiple annotations map to the same token; BIO tags "
                        "cannot represent this without ambiguity."
                    ),
                    "token_indices": collisions,
                }
            )

    return errors, unaligned_entity_count


def validate_datasets(
    records_1: Sequence[Dict[str, Any]],
    records_2: Sequence[Dict[str, Any]],
    load_errors_1: Sequence[Dict[str, Any]],
    load_errors_2: Sequence[Dict[str, Any]],
    path_1: Path,
    path_2: Path,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    errors.extend(
        {"annotator": "annotator_1", **error} for error in load_errors_1
    )
    errors.extend(
        {"annotator": "annotator_2", **error} for error in load_errors_2
    )
    if len(records_1) != len(records_2):
        errors.append(
            {
                "error": "Record counts do not match.",
                "annotator_1_count": len(records_1),
                "annotator_2_count": len(records_2),
            }
        )

    aligned_count = 0
    text_mismatch_count = 0
    id_mismatch_count = 0
    for index, (record_1, record_2) in enumerate(zip(records_1, records_2)):
        text_1 = record_1.get("text")
        text_2 = record_2.get("text")
        id_1 = get_document_id(record_1)
        id_2 = get_document_id(record_2)
        if isinstance(text_1, str) and isinstance(text_2, str) and text_1 == text_2:
            aligned_count += 1
        else:
            text_mismatch_count += 1
            errors.append(
                {
                    "error": "Sentence text mismatch at the same record position.",
                    "record_index": index,
                    "annotator_1_document_id": id_1,
                    "annotator_2_document_id": id_2,
                    "annotator_1_text": text_1,
                    "annotator_2_text": text_2,
                }
            )
        if id_1 != id_2 and (id_1 is not None or id_2 is not None):
            id_mismatch_count += 1
            errors.append(
                {
                    "error": "Document ID mismatch at the same record position.",
                    "record_index": index,
                    "annotator_1_document_id": id_1,
                    "annotator_2_document_id": id_2,
                }
            )

    annotation_errors_1, unaligned_1 = validate_annotations(
        records_1, "annotator_1"
    )
    annotation_errors_2, unaligned_2 = validate_annotations(
        records_2, "annotator_2"
    )
    errors.extend(annotation_errors_1)
    errors.extend(annotation_errors_2)

    return {
        "valid": not errors,
        "input_files": {
            "annotator_1": str(path_1),
            "annotator_2": str(path_2),
        },
        "schema_labels": list(LABELS),
        "record_counts": {
            "annotator_1": len(records_1),
            "annotator_2": len(records_2),
            "equal": len(records_1) == len(records_2),
        },
        "alignment": {
            "positionally_aligned_text_records": aligned_count,
            "text_mismatches": text_mismatch_count,
            "document_id_mismatches": id_mismatch_count,
            "order_is_aligned": (
                len(records_1) == len(records_2)
                and text_mismatch_count == 0
                and id_mismatch_count == 0
            ),
        },
        "annotation_checks": {
            "doccano_format": "JSONL objects with text and label=[start,end,label].",
            "record_load_errors": len(load_errors_1) + len(load_errors_2),
            "annotation_errors": len(annotation_errors_1) + len(annotation_errors_2),
            "unaligned_token_boundaries": {
                "annotator_1": unaligned_1,
                "annotator_2": unaligned_2,
                "policy": (
                    "Reported as a tokenizer limitation, not rejected: BIO assignment "
                    "uses every token intersecting a character span."
                ),
            },
            "semantic_correctness": (
                "Not machine-verifiable; the extracted substring is text[start:end]."
            ),
        },
        "errors": errors,
    }


def annotation_set(record: Dict[str, Any]) -> Set[EntityTuple]:
    return {
        (entity[0], entity[1], entity[2])
        for entity in record.get("label", [])
    }


def entity_object(text: str, entity: Optional[EntityTuple]) -> Optional[Dict[str, Any]]:
    if entity is None:
        return None
    start, end, label = entity
    return {
        "start": start,
        "end": end,
        "label": label,
        "text": text[start:end],
    }


def categorize_overlap(entity_1: EntityTuple, entity_2: EntityTuple) -> str:
    start_1, end_1, label_1 = entity_1
    start_2, end_2, label_2 = entity_2
    if start_1 == start_2 and end_1 == end_2:
        return "LABEL_DISAGREEMENT"
    if label_1 == label_2:
        return "BOUNDARY_DISAGREEMENT"
    return "BOUNDARY_AND_LABEL_DISAGREEMENT"


def analyze_sentence_disagreements(
    record_1: Dict[str, Any],
    record_2: Dict[str, Any],
) -> List[Dict[str, Any]]:
    text = record_1.get("text", "")
    entities_1 = annotation_set(record_1)
    entities_2 = annotation_set(record_2)
    exact_matches = entities_1 & entities_2
    remaining_1 = sorted(entities_1 - exact_matches)
    remaining_2 = sorted(entities_2 - exact_matches)
    disagreements: List[Dict[str, Any]] = []
    matched_1: Set[EntityTuple] = set()
    matched_2: Set[EntityTuple] = set()

    for entity_1 in remaining_1:
        for entity_2 in remaining_2:
            if max(entity_1[0], entity_2[0]) < min(entity_1[1], entity_2[1]):
                matched_1.add(entity_1)
                matched_2.add(entity_2)
                disagreements.append(
                    {
                        "category": categorize_overlap(entity_1, entity_2),
                        "annotator_1_entity": entity_object(text, entity_1),
                        "annotator_2_entity": entity_object(text, entity_2),
                    }
                )

    for entity in remaining_1:
        if entity not in matched_1:
            disagreements.append(
                {
                    "category": "ANNOTATOR_1_ONLY",
                    "annotator_1_entity": entity_object(text, entity),
                    "annotator_2_entity": None,
                }
            )
    for entity in remaining_2:
        if entity not in matched_2:
            disagreements.append(
                {
                    "category": "ANNOTATOR_2_ONLY",
                    "annotator_1_entity": None,
                    "annotator_2_entity": entity_object(text, entity),
                }
            )
    return disagreements


def bio_tags(record: Dict[str, Any]) -> List[str]:
    tokens = tokenize(record["text"])
    tags = ["O"] * len(tokens)
    for start, end, label in sorted(annotation_set(record)):
        matching = [
            index
            for index, (_, token_start, token_end) in enumerate(tokens)
            if token_start < end and start < token_end
        ]
        for position, token_index in enumerate(matching):
            tags[token_index] = ("B-" if position == 0 else "I-") + label
    return tags


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def flatten_disagreement(
    document_id: Any,
    text: str,
    disagreement: Dict[str, Any],
) -> Dict[str, Any]:
    entity_1 = disagreement["annotator_1_entity"] or {}
    entity_2 = disagreement["annotator_2_entity"] or {}
    return {
        "document_id": document_id,
        "text": text,
        "annotator_1_entity_text": entity_1.get("text"),
        "annotator_1_start": entity_1.get("start"),
        "annotator_1_end": entity_1.get("end"),
        "annotator_1_label": entity_1.get("label"),
        "annotator_2_entity_text": entity_2.get("text"),
        "annotator_2_start": entity_2.get("start"),
        "annotator_2_end": entity_2.get("end"),
        "annotator_2_label": entity_2.get("label"),
        "category": disagreement["category"],
    }


def calculate_metrics(
    records_1: Sequence[Dict[str, Any]],
    records_2: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    all_entities_1: List[EntityTuple] = []
    all_entities_2: List[EntityTuple] = []
    class_entities_1: Dict[str, List[EntityTuple]] = {
        label: [] for label in LABELS
    }
    class_entities_2: Dict[str, List[EntityTuple]] = {
        label: [] for label in LABELS
    }
    class_true_positives: Counter[str] = Counter()
    exact_matches: Set[Tuple[int, EntityTuple]] = set()
    disagreement_rows: List[Dict[str, Any]] = []
    adjudication_records: List[Dict[str, Any]] = []
    categories = Counter({category: 0 for category in DISAGREEMENT_CATEGORIES})
    exact_sentence_count = 0
    empty_both_count = 0
    token_tags_1: List[str] = []
    token_tags_2: List[str] = []

    for record_index, (record_1, record_2) in enumerate(zip(records_1, records_2)):
        text = record_1["text"]
        row_entities_1 = sorted(annotation_set(record_1))
        row_entities_2 = sorted(annotation_set(record_2))
        all_entities_1.extend(row_entities_1)
        all_entities_2.extend(row_entities_2)
        for entity in row_entities_1:
            class_entities_1[entity[2]].append(entity)
        for entity in row_entities_2:
            class_entities_2[entity[2]].append(entity)

        sentence_matches = set(row_entities_1) & set(row_entities_2)
        exact_matches.update((record_index, entity) for entity in sentence_matches)
        class_true_positives.update(entity[2] for entity in sentence_matches)
        identical = set(row_entities_1) == set(row_entities_2)
        exact_sentence_count += int(identical)
        empty_both = not row_entities_1 and not row_entities_2
        empty_both_count += int(empty_both)

        disagreements = analyze_sentence_disagreements(record_1, record_2)
        for disagreement in disagreements:
            categories[disagreement["category"]] += 1
            disagreement_rows.append(
                flatten_disagreement(
                    get_document_id(record_1), text, disagreement
                )
            )
        if empty_both:
            categories["NO_ENTITY_AGREEMENT"] += 1
            disagreement_rows.append(
                flatten_disagreement(
                    get_document_id(record_1),
                    text,
                    {
                        "category": "NO_ENTITY_AGREEMENT",
                        "annotator_1_entity": None,
                        "annotator_2_entity": None,
                    },
                )
            )

        if identical:
            status = "AGREED"
            adjudicated = record_1.get("label", [])
        else:
            status = "PENDING"
            adjudicated = None
        adjudication_records.append(
            {
                "document_id": get_document_id(record_1),
                "text": text,
                "annotator_1_label": record_1.get("label", []),
                "annotator_2_label": record_2.get("label", []),
                "disagreements": disagreements,
                "adjudication_status": status,
                "adjudicated_label": adjudicated,
            }
        )

        token_tags_1.extend(bio_tags(record_1))
        token_tags_2.extend(bio_tags(record_2))

    tp = len(exact_matches)
    fp = len(all_entities_1) - tp
    fn = len(all_entities_2) - tp
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    try:
        kappa_value = float(cohen_kappa_score(token_tags_1, token_tags_2))
        token_kappa: Optional[float] = (
            kappa_value if math.isfinite(kappa_value) else None
        )
    except ValueError:
        token_kappa = None

    per_class: List[Dict[str, Any]] = []
    for label in LABELS:
        entities_1 = class_entities_1[label]
        entities_2 = class_entities_2[label]
        true_positives = class_true_positives[label]
        false_positives = len(entities_1) - true_positives
        false_negatives = len(entities_2) - true_positives
        class_precision = safe_ratio(true_positives, len(entities_1))
        class_recall = safe_ratio(true_positives, len(entities_2))
        per_class.append(
            {
                "label": label,
                "annotator_1_entities": len(entities_1),
                "annotator_2_entities": len(entities_2),
                "exact_matches": true_positives,
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "precision": class_precision,
                "recall": class_recall,
                "f1": f1_score(class_precision, class_recall),
                "reverse_precision": class_recall,
                "reverse_recall": class_precision,
                "reverse_f1": f1_score(class_recall, class_precision),
            }
        )

    macro_f1 = float(np.mean([row["f1"] for row in per_class])) if per_class else 0.0
    entity_metrics = {
        "annotator_1_entities": len(all_entities_1),
        "annotator_2_entities": len(all_entities_2),
        "exact_matches": tp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "reverse_precision": recall,
        "reverse_recall": precision,
        "reverse_f1": f1_score(recall, precision),
        "macro_f1": macro_f1,
        "macro_f1_policy": (
            "Unweighted mean across every schema class; zero-division classes "
            "receive F1=0 and remain in the macro average."
        ),
    }
    tokenization_description = (
        "Unicode regex tokens: contiguous Python Unicode \\w+ runs plus each "
        "non-whitespace punctuation character; whitespace is omitted. Within a "
        "word run, lowercase-to-uppercase and digit-to-letter transitions start "
        "new tokens to separate concatenated Vietnamese text and date prefixes. "
        "Each entity assigns B-LABEL to the first intersecting token and "
        "I-LABEL to subsequent intersecting tokens; all other tokens are O."
    )
    return {
        "token_level": {
            "cohen_kappa": token_kappa,
            "evaluated_tokens": len(token_tags_1),
            "tokenization": tokenization_description,
            "interpretation_note": (
                "Supplementary measure; O-tag prevalence and tokenization "
                "choices can affect interpretation."
            ),
        },
        "entity_level": entity_metrics,
        "sentence_level": {
            "total_sentences": len(records_1),
            "sentences_with_identical_annotation_sets": exact_sentence_count,
            "sentences_with_at_least_one_disagreement": (
                len(records_1) - exact_sentence_count
            ),
            "exact_agreement_rate": safe_ratio(exact_sentence_count, len(records_1)),
            "sentences_with_no_entities_from_either_annotator": empty_both_count,
        },
        "entity_agreement_by_class": per_class,
        "disagreement_category_counts": dict(categories),
        "disagreement_details": disagreement_rows,
        "adjudication_records": adjudication_records,
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(
        path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )


def save_charts(output_dir: Path, metrics: Dict[str, Any]) -> List[str]:
    class_rows = metrics["entity_agreement_by_class"]
    labels = [row["label"] for row in class_rows]
    x = np.arange(len(labels))
    width = 0.38

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        x - width / 2,
        [row["annotator_1_entities"] for row in class_rows],
        width,
        label="Annotator 1",
    )
    axis.bar(
        x + width / 2,
        [row["annotator_2_entities"] for row in class_rows],
        width,
        label="Annotator 2",
    )
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Số thực thể")
    axis.set_title("Số thực thể theo lớp")
    axis.legend()
    figure.tight_layout()
    counts_chart = "entity_counts_by_class.png"
    figure.savefig(output_dir / counts_chart, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(labels, [row["f1"] for row in class_rows], color="#4472C4")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1 exact-match")
    axis.set_title("Entity-level F1 theo lớp")
    figure.tight_layout()
    f1_chart = "entity_f1_by_class.png"
    figure.savefig(output_dir / f1_chart, dpi=160)
    plt.close(figure)

    categories = list(metrics["disagreement_category_counts"])
    counts = [metrics["disagreement_category_counts"][name] for name in categories]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(categories, counts, color="#ED7D31")
    axis.set_xticks(np.arange(len(categories)), categories, rotation=35, ha="right")
    axis.set_ylabel("Số trường hợp")
    axis.set_title("Phân loại bất đồng (NO_ENTITY_AGREEMENT là trạng thái đồng thuận)")
    figure.tight_layout()
    disagreement_chart = "disagreement_categories.png"
    figure.savefig(output_dir / disagreement_chart, dpi=160)
    plt.close(figure)
    return [counts_chart, f1_chart, disagreement_chart]


def export_results(
    output_dir: Path,
    path_1: Path,
    path_2: Path,
    validation: Dict[str, Any],
    metrics: Dict[str, Any],
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parent.parent
    validation["input_files"] = {
        "annotator_1": path_1.resolve().relative_to(package_root).as_posix(),
        "annotator_2": path_2.resolve().relative_to(package_root).as_posix(),
    }
    write_json(output_dir / "validation_report.json", validation)

    entity = metrics["entity_level"]
    sentence = metrics["sentence_level"]
    token = metrics["token_level"]
    summary_rows = [
        {"metric": "total_sentences", "value": sentence["total_sentences"]},
        {"metric": "evaluated_tokens", "value": token["evaluated_tokens"]},
        {"metric": "token_level_cohen_kappa", "value": token["cohen_kappa"]},
        {"metric": "entity_annotator_1_count", "value": entity["annotator_1_entities"]},
        {"metric": "entity_annotator_2_count", "value": entity["annotator_2_entities"]},
        {"metric": "entity_exact_matches", "value": entity["exact_matches"]},
        {"metric": "entity_tp", "value": entity["tp"]},
        {"metric": "entity_fp", "value": entity["fp"]},
        {"metric": "entity_fn", "value": entity["fn"]},
        {"metric": "entity_precision_annotator_1_vs_2", "value": entity["precision"]},
        {"metric": "entity_recall_annotator_1_vs_2", "value": entity["recall"]},
        {"metric": "entity_f1", "value": entity["f1"]},
        {"metric": "entity_reverse_precision", "value": entity["reverse_precision"]},
        {"metric": "entity_reverse_recall", "value": entity["reverse_recall"]},
        {"metric": "entity_reverse_f1", "value": entity["reverse_f1"]},
        {"metric": "entity_macro_f1", "value": entity["macro_f1"]},
        {
            "metric": "sentences_with_identical_annotation_sets",
            "value": sentence["sentences_with_identical_annotation_sets"],
        },
        {
            "metric": "sentences_with_at_least_one_disagreement",
            "value": sentence["sentences_with_at_least_one_disagreement"],
        },
        {"metric": "sentence_exact_agreement_rate", "value": sentence["exact_agreement_rate"]},
        {
            "metric": "sentences_with_no_entities_from_either_annotator",
            "value": sentence["sentences_with_no_entities_from_either_annotator"],
        },
    ]
    write_csv(
        output_dir / "IAA_summary.csv",
        summary_rows,
        ("metric", "value"),
    )
    write_csv(
        output_dir / "entity_agreement_by_class.csv",
        metrics["entity_agreement_by_class"],
        (
            "label",
            "annotator_1_entities",
            "annotator_2_entities",
            "exact_matches",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "reverse_precision",
            "reverse_recall",
            "reverse_f1",
        ),
    )
    category_rows = [
        {"category": category, "count": count}
        for category, count in metrics["disagreement_category_counts"].items()
    ]
    write_csv(
        output_dir / "disagreement_summary.csv",
        category_rows,
        ("category", "count"),
    )
    write_csv(
        output_dir / "disagreement_details.csv",
        metrics["disagreement_details"],
        (
            "document_id",
            "text",
            "annotator_1_entity_text",
            "annotator_1_start",
            "annotator_1_end",
            "annotator_1_label",
            "annotator_2_entity_text",
            "annotator_2_start",
            "annotator_2_end",
            "annotator_2_label",
            "category",
        ),
    )
    write_jsonl(
        output_dir / "adjudication_review.jsonl",
        metrics["adjudication_records"],
    )
    review_rows = []
    for record in metrics["adjudication_records"]:
        review_rows.append(
            {
                "document_id": record["document_id"],
                "text": record["text"],
                "annotator_1_label": json.dumps(
                    record["annotator_1_label"], ensure_ascii=False
                ),
                "annotator_2_label": json.dumps(
                    record["annotator_2_label"], ensure_ascii=False
                ),
                "disagreements": json.dumps(
                    record["disagreements"], ensure_ascii=False
                ),
                "adjudication_status": record["adjudication_status"],
                "adjudicated_label": (
                    json.dumps(record["adjudicated_label"], ensure_ascii=False)
                    if record["adjudicated_label"] is not None
                    else ""
                ),
            }
        )
    write_csv(
        output_dir / "adjudication_review.csv",
        review_rows,
        (
            "document_id",
            "text",
            "annotator_1_label",
            "annotator_2_label",
            "disagreements",
            "adjudication_status",
            "adjudicated_label",
        ),
    )

    chart_files = save_charts(output_dir, metrics)
    package_root = Path(__file__).resolve().parent.parent
    def package_relative(path: Path) -> str:
        try:
            return path.resolve().relative_to(package_root).as_posix()
        except ValueError:
            return path.name

    report = {
        "input_files": {
            "annotator_1": package_relative(path_1),
            "annotator_2": package_relative(path_2),
        },
        "schema_labels": list(LABELS),
        "dataset_validation": {
            "valid": validation["valid"],
            "records_annotator_1": validation["record_counts"]["annotator_1"],
            "records_annotator_2": validation["record_counts"]["annotator_2"],
            "aligned_text_records": validation["alignment"][
                "positionally_aligned_text_records"
            ],
            "error_count": len(validation["errors"]),
        },
        **{
            key: value
            for key, value in metrics.items()
            if key not in {"disagreement_details", "adjudication_records"}
        },
        "outputs": [
            "IAA_report.json",
            "IAA_summary.csv",
            "entity_agreement_by_class.csv",
            "disagreement_summary.csv",
            "disagreement_details.csv",
            "adjudication_review.jsonl",
            "adjudication_review.csv",
            "validation_report.json",
            *chart_files,
        ],
    }
    write_json(output_dir / "IAA_report.json", report)
    return report["outputs"]


def main() -> int:
    args = parse_args()
    path_1 = resolve_path(args.annotator_1)
    path_2 = resolve_path(args.annotator_2)
    output_dir = resolve_path(args.output_dir)
    if output_dir in {path_1, path_2, path_1.parent, path_2.parent}:
        print(
            "Error: output directory cannot be an input file or its parent directory.",
            file=sys.stderr,
        )
        return 2
    for input_path in (path_1, path_2):
        if not input_path.is_file():
            print(f"Error: input file not found: {input_path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records_1, load_errors_1 = load_jsonl(path_1)
    records_2, load_errors_2 = load_jsonl(path_2)
    validation = validate_datasets(
        records_1,
        records_2,
        load_errors_1,
        load_errors_2,
        path_1,
        path_2,
    )
    package_root = Path(__file__).resolve().parent.parent
    validation["input_files"] = {
        "annotator_1": path_1.resolve().relative_to(package_root).as_posix(),
        "annotator_2": path_2.resolve().relative_to(package_root).as_posix(),
    }
    write_json(output_dir / "validation_report.json", validation)
    if not validation["valid"]:
        print(
            f"Validation failed ({len(validation['errors'])} errors). "
            f"Metrics were not calculated. See {output_dir / 'validation_report.json'}",
            file=sys.stderr,
        )
        return 2

    metrics = calculate_metrics(records_1, records_2)
    outputs = export_results(output_dir, path_1, path_2, validation, metrics)
    print(
        json.dumps(
            {
                "status": "success",
                "sentences": metrics["sentence_level"]["total_sentences"],
                "evaluated_tokens": metrics["token_level"]["evaluated_tokens"],
                "token_cohen_kappa": metrics["token_level"]["cohen_kappa"],
                "entity_metrics": metrics["entity_level"],
                "sentence_level": metrics["sentence_level"],
                "disagreement_categories": metrics["disagreement_category_counts"],
                "output_directory": str(output_dir),
                "generated_files": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
