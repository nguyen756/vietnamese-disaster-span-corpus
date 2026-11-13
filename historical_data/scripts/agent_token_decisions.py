"""Render indexed source tokens and compile agent-selected token spans.

No labels are predicted here: every interval and label is selected by the agent.
Token intervals are inclusive and zero-based. Output uses exact source text,
then the existing compiler creates Unicode character offsets as usual.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "annotation_work"


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def tokens(text):
    return list(re.finditer(r"\S+", text))


def selection(text, code, first, last):
    parts = tokens(text)
    if not 0 <= first <= last < len(parts):
        raise ValueError(f"Token interval out of range: {first}-{last}")
    start, end = parts[first].start(), parts[last].end()
    punctuation = '.,;:!?…|*"“”‘’'
    while start < end and text[start] in punctuation:
        start += 1
    while start < end and text[end - 1] in punctuation:
        end -= 1
    previous = None
    while previous != (start, end):
        previous = (start, end)
        while start < end and text[start] in punctuation:
            start += 1
        while start < end and text[end - 1] in punctuation:
            end -= 1
        for left, right in [("(", ")"), ("[", "]")]:
            if start < end and text[start] == left and text[end - 1] == right:
                start += 1
                end -= 1
            elif start < end and text[start] == left and text[start:end].count(left) > text[start:end].count(right):
                start += 1
            elif start < end and text[end - 1] == right and text[start:end].count(right) > text[start:end].count(left):
                end -= 1
    surface = text[start:end]
    if not surface or surface != surface.strip():
        raise ValueError(f"Invalid surface selected at tokens {first}-{last}")
    positions, cursor = [], 0
    while (cursor := text.find(surface, cursor)) >= 0:
        positions.append(cursor)
        cursor += len(surface)
    if start not in positions:
        raise ValueError("Selected surface cannot be represented by an occurrence index")
    return [code, surface] if len(positions) == 1 else [code, surface, positions.index(start) + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["show", "compile"])
    parser.add_argument("first", nargs="?", type=int)
    parser.add_argument("last", nargs="?", type=int)
    args = parser.parse_args()
    source = ROOT / "data/train_candidate_clean.jsonl"
    lock = json.loads((WORK / "source_lock.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != lock["sha256"]:
        raise ValueError("Training source changed")
    rows = read(source)
    if args.mode == "show":
        for number in range(args.first, args.last + 1):
            row = rows[number - 1]
            print(str(number) + "|" + " ".join(str(i) + ":" + match.group() for i, match in enumerate(tokens(row["text"]))))
        return
    output = WORK / "decisions"
    for path in sorted((WORK / "token_decisions").glob("batch_*.jsonl")):
        result = []
        for number, spans, *note in read(path):
            if isinstance(spans, list):
                result.append([number, spans] + note)
                continue
            entities = []
            for span in spans.split():
                match = re.fullmatch(r"([DWLTMCPAGV])(\d+)(?:-(\d+))?", span)
                if not match:
                    raise ValueError(f"Invalid decision {span!r} at row {number}")
                code, first, last = match.groups()
                entities.append(selection(rows[number - 1]["text"], code, int(first), int(last or first)))
            result.append([number, entities] + note)
        target = output / path.name
        if target.exists() and not any(r[0] >= 1032 for r in read(target)):
            raise ValueError(f"Refuse to overwrite a previous-run decision file: {target.name}")
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result), encoding="utf-8")
    print("Compiled explicit token-span decisions into canonical surface selections.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
