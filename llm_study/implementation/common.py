"""Standard-library utilities; immutable inputs, explicit outputs, no model imports."""
from pathlib import Path
import hashlib
import json
import math
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'supplied' if (ROOT / 'supplied').is_dir() else ROOT
sys.path.insert(0, str(INPUT / 'tools'))
from study import compile_response, validate_annotation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def exact_rows(actual, expected):
    indexed = {r['id']: r for r in actual}
    if len(indexed) != len(actual) or set(indexed) != {r['id'] for r in expected}:
        raise ValueError('Require each expected ID exactly once; missing/extra/duplicate rows rejected')
    for row in expected:
        candidate = indexed[row['id']]
        if candidate.get('text') != row['text'] or candidate.get('text_sha256', text_sha(candidate['text'])) != text_sha(row['text']):
            raise ValueError('Changed text/hash: ' + row['id'])
    return [indexed[r['id']] for r in expected]


def utc_timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Missing UTC timestamp')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError('Timestamp must be explicitly UTC')
    return dt


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


LABELS = read(INPUT / 'config/labels.json')
RUNS = ['M1-P1', 'M1-P2', 'M2-P1', 'M2-P2']


def validate_spans(text, spans):
    if not isinstance(spans, list):
        raise ValueError('Null/unstarted is not an empty span list')
    for s in spans:
        if (not isinstance(s,list) or len(s)!=3 or type(s[0]) is not int or type(s[1]) is not int
                or not 0<=s[0]<s[1]<=len(text) or s[2] not in LABELS):
            raise ValueError('Invalid Unicode span')
    ordered=sorted(spans)
    if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Overlapping/duplicate spans')


def lock_inputs(out, config=None):
    config = Path(config) if config else INPUT / 'config/study.json'
    paths = [INPUT / n for n in read(INPUT / 'MANIFEST.json')['files']]
    files = {p.relative_to(INPUT).as_posix(): sha(p) for p in paths}
    value = {'locked_at_utc': now(), 'input_files': files,
             'execution_config_sha256': sha(config), 'config': read(config),
             'implementation_files': {p.name: sha(p) for p in sorted((ROOT / 'implementation').glob('*')) if p.is_file()},
             'primary_run': 'M1-P2', 'references_not_for_payloads': True}
    write(out, value)
    return value


def verify_lock(path, config=None):
    value = read(path)
    for name, expected in value['input_files'].items():
        target = (INPUT / name).resolve()
        if not target.is_relative_to(INPUT.resolve()) or sha(target) != expected:
            raise ValueError('Locked input changed: ' + name)
    for name, expected in value['implementation_files'].items():
        if sha(ROOT / 'implementation' / name) != expected:
            raise ValueError('Locked implementation changed: ' + name)
    if config and sha(config) != value['execution_config_sha256']:
        raise ValueError('Execution config differs from pre-scoring lock')
    return value
