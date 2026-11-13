"""Remove exact normalized sentence-text overlaps from training data.

Normalization is Unicode NFC plus leading/trailing trim and whitespace-run
collapse. Punctuation and sentence content are otherwise left unchanged.
Original records and sentence strings are preserved in both outputs.
"""
from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/intermediate/train_candidate.jsonl"
GOLD = ROOT / "data/gold_test_final.jsonl"
CLEAN = ROOT / "data/train_candidate_clean.jsonl"
EXCLUDED = ROOT / "data/excluded_train_overlap.jsonl"
STATS = ROOT / "statistics/overlap_statistics.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    train, gold = read_jsonl(TRAIN), read_jsonl(GOLD)
    gold_refs: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(gold, 1):
        gold_refs[normalized_text(row["text"])].append(
            {"record_number": index, "document_id": row["document_id"]}
        )
    kept, excluded = [], []
    for row in train:
        refs = gold_refs.get(normalized_text(row["text"]))
        if refs:
            excluded.append({
                "record": row,
                "reason": "exact_normalized_sentence_text_matches_gold_test",
                "matching_gold_test": refs,
            })
        else:
            kept.append(row)
    write_jsonl(CLEAN, kept)
    write_jsonl(EXCLUDED, excluded)
    remaining = {normalized_text(row["text"]) for row in kept} & set(gold_refs)
    stats = {
        "normalization": "Unicode NFC; trim and collapse whitespace runs; punctuation unchanged",
        "gold_test_rows": len(gold),
        "train_rows_before": len(train),
        "train_rows_after": len(kept),
        "excluded_train_rows": len(excluded),
        "distinct_overlapping_texts_removed": len({normalized_text(x["record"]["text"]) for x in excluded}),
        "remaining_exact_normalized_text_overlaps": len(remaining),
        "document_id_overlap_after_cleaning": len({r["document_id"] for r in kept} & {r["document_id"] for r in gold}),
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
