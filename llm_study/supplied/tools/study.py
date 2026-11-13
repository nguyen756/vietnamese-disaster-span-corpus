"""Offline preparation utilities only; no provider, network, model or training calls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = set(json.loads((ROOT / 'config/labels.json').read_text(encoding='utf-8')))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def dump_rows(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def occurrences(text, substring):
    return [i for i in range(len(text)) if text.startswith(substring, i)] if substring else []


def compile_response(text, raw):
    """Resolve exact strings; no guessing, normalization, JSON repair or silent drops."""
    rejected, accepted = [], []
    if not isinstance(raw, str):
        return {'label': [], 'rejections': [{'item_index': None, 'reason': 'response_not_string'}],
                'whole_response_invalid': True, 'structural_rejection': True}
    try:
        values = json.loads(raw)
    except (ValueError, TypeError):
        values = None
    if not isinstance(values, list):
        return {'label': [], 'rejections': [{'item_index': None, 'reason': 'expected_json_array'}],
                'whole_response_invalid': True, 'structural_rejection': True}
    for index, item in enumerate(values):
        reason = None
        if not isinstance(item, list) or len(item) != 3:
            reason = 'expected_three_element_array'
        else:
            string, occurrence, label = item
            if not isinstance(string, str) or not string:
                reason = 'empty_or_nonstring_span'
            elif type(occurrence) is not int or occurrence < 0:
                reason = 'invalid_occurrence'
            elif not isinstance(label, str) or label not in LABELS:
                reason = 'unknown_category'
            else:
                starts = occurrences(text, string)
                if not starts:
                    reason = 'string_absent'
                elif occurrence >= len(starts):
                    reason = 'occurrence_out_of_range'
                else:
                    start = starts[occurrence]
                    span = [start, start + len(string), label]
                    if any(start < b and span[1] > a for a, b, _ in accepted):
                        reason = 'overlap_or_duplicate'
                    else:
                        accepted.append(span)
        if reason:
            rejected.append({'item_index': index, 'item': item, 'reason': reason})
    return {'label': sorted(accepted), 'rejections': rejected,
            'whole_response_invalid': False, 'structural_rejection': bool(rejected)}


def validate_annotation(row):
    """Validate a COMPLETED human row; unknown/null never passes as an empty set."""
    if row.get('status') != 'completed' or not isinstance(row.get('label'), list):
        raise ValueError('Not a completed annotation')
    text = row['text']
    if sha(text.encode('utf-8')) != row['text_sha256']:
        raise ValueError('Text hash mismatch')
    spans = row['label']
    for span in spans:
        if not isinstance(span, list) or len(span) != 3:
            raise ValueError('Invalid triple')
        a, b, label = span
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError('Invalid code-point offsets')
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError('Invalid category')
    ordered = sorted(spans)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Overlapping or duplicate spans')
    if not isinstance(row.get('reviewer_id'), str) or not row['reviewer_id'].strip():
        raise ValueError('Missing reviewer pseudonym')
    if not row.get('completed_at_utc'):
        raise ValueError('Missing completion timestamp')
    seconds = row.get('active_seconds')
    if seconds is not None and (type(seconds) not in (int, float) or seconds < 0):
        raise ValueError('Invalid measured time')


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError('Unsafe manifest path')
        data = target.read_bytes()
        if len(data) != info['bytes'] or sha(data) != info['sha256']:
            raise ValueError('Manifest mismatch: ' + name)
    inputs = load_rows(ROOT / 'blind/gold_inputs.jsonl')
    assert len(inputs) == 1000 and len({r['id'] for r in inputs}) == 1000
    for row in inputs:
        assert set(row) == {'id', 'text', 'text_sha256'}
        assert sha(row['text'].encode('utf-8')) == row['text_sha256']
    sample = load_rows(ROOT / 'human/E4/annotator_A.jsonl')
    assert len(sample) == 300 and len({r['id'] for r in sample}) == 300
    assert sample == load_rows(ROOT / 'human/E4/annotator_B.jsonl')
    assert all(r['label'] is None and r['status'] == 'not_started' for r in sample)
    assert len(load_rows(ROOT / 'human/E5/gold_signoff.jsonl')) == 960
    examples = json.loads((ROOT / 'prompts/dev_examples.json').read_text(encoding='utf-8'))
    assert len(examples) == 10
    gold_keys = {' '.join(r['text'].split()).casefold() for r in inputs}
    for example in examples:
        assert ' '.join(example['sentence'].split()).casefold() not in gold_keys
        assert not compile_response(example['sentence'], json.dumps(example['annotation']))['structural_rejection']
    config = json.loads((ROOT / 'config/study.json').read_text(encoding='utf-8'))
    assert config['execution']['mode'] == 'prepare_only'
    assert config['execution']['allow_network'] is False
    print(json.dumps({'status': 'pass', 'manifest_files': len(manifest['files']),
                      'blind_gold_rows': 1000, 'e4_rows': 300, 'e5_representatives': 960,
                      'dev_examples': 10, 'model_calls': 0}))


def requests(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = load_rows(ROOT / 'blind/gold_inputs.jsonl')
    for model in ['M1', 'M2']:
        for prompt in ['P1', 'P2']:
            system = (ROOT / 'prompts' / (prompt + '.txt')).read_text(encoding='utf-8')
            run = model + '-' + prompt
            records = [{'custom_id': run + ':' + row['id'], 'run_id': run,
                        'input_id': row['id'], 'text_sha256': row['text_sha256'],
                        'prompt_sha256': sha(system.encode('utf-8')),
                        'messages': [{'role': 'system', 'content': system},
                                     {'role': 'user', 'content': json.dumps({'sentence': row['text']}, ensure_ascii=False)}]}
                       for row in inputs]
            dump_rows(output / (run + '.jsonl'), records)
    print(json.dumps({'status': 'prepared_only', 'request_records': 4000, 'model_calls': 0,
                      'output': str(output)}))


def compile_file(source, output):
    """Offline response import contract: one {id,response} per input; no guessed omissions."""
    inputs = load_rows(ROOT / 'blind/gold_inputs.jsonl')
    responses = load_rows(source)
    by_id = {row['id']: row for row in responses}
    if len(by_id) != len(responses) or set(by_id) != {row['id'] for row in inputs}:
        raise ValueError('Require exactly one explicit response record per input')
    result = []
    for row in inputs:
        source_row = by_id[row['id']]
        if 'response' not in source_row:
            raise ValueError('Missing response field; record actual failure explicitly')
        compiled = compile_response(row['text'], source_row['response'])
        result.append({**row, **compiled, 'response_sha256': sha(json.dumps(source_row['response'], ensure_ascii=False).encode('utf-8'))})
    dump_rows(output, result)
    print(json.dumps({'compiled_rows': len(result), 'model_calls': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify')
    request_parser = commands.add_parser('requests')
    request_parser.add_argument('--out', type=Path, required=True)
    compile_parser = commands.add_parser('compile')
    compile_parser.add_argument('--responses', type=Path, required=True)
    compile_parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
    elif args.command == 'requests':
        requests(args.out)
    else:
        compile_file(args.responses, args.out)


if __name__ == '__main__':
    main()
