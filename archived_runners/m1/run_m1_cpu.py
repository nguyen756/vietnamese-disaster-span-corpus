"""One explicitly selected 1,000-sentence M1 run, with full console/file reporting."""
import argparse
import csv
import datetime
import fcntl
import json
import os
import resource
import sys
import time
from pathlib import Path

import pilot_cpu as core
import m1_reporting as reporting

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'gold_cpu_v1'
RUN_IDS = ['M1-P1', 'M1-P2']
SETTINGS = {k: v for k, v in core.SETTINGS.items() if k not in ['prompt_id', 'requests']}
SETTINGS.update(run_ids=RUN_IDS, sentences_per_run=1000, primary_run='M1-P2')


def cpu_environment():
    # Set before importing the model backend; CPU-only even from a direct CLI.
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='12', OPENBLAS_NUM_THREADS='1',
                      HF_HOME=str(core.ASSETS / 'hf_cache'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')


def load_inputs(metadata_path=None):
    for name, expected in core.read(ROOT / 'MANIFEST.json')['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or core.file_hash(path) != expected:
            raise ValueError('Original package manifest mismatch: ' + name)
    rows = [json.loads(s) for s in (ROOT / 'blind/gold_inputs.jsonl').read_text().splitlines() if s.strip()]
    if len(rows) != 1000 or len({r['id'] for r in rows}) != 1000:
        raise ValueError('Require exactly 1,000 distinct frozen Gold IDs')
    for i, row in enumerate(rows):
        if set(row) != {'id', 'text', 'text_sha256'}:
            raise ValueError('Blind inputs must contain ID, text and text hash only')
        if core.hashlib.sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Input text hash differs: ' + row['id'])
        # Input file order is known; original source row and document are not guessed.
        row.update(input_row_index_1based=i+1, split='gold_v1', document_id=None,
                   original_row_index=None, source_index_base=None)
    if metadata_path:
        attach_metadata(rows, Path(metadata_path))
    prompts = {p: (ROOT / f'prompts/{p}.txt').read_text() for p in ['P1', 'P2']}
    return rows, prompts


def attach_metadata(rows, path):
    allowed = {'id', 'text_sha256', 'document_id', 'split', 'original_row_index', 'source_index_base'}
    values = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    by_id = {v['id']: v for v in values}
    if len(by_id) != len(values) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Metadata must map every input ID once')
    for row in rows:
        value = by_id[row['id']]
        if set(value)-allowed:
            raise ValueError('Metadata has unsupported fields; annotations are forbidden')
        if value.get('text_sha256') != row['text_sha256']:
            raise ValueError('Metadata text hash differs: ' + row['id'])
        if value.get('document_id') is not None and not str(value['document_id']).strip():
            raise ValueError('Document ID cannot be blank')
        index = value.get('original_row_index')
        if index is not None and (type(index) is not int or value.get('source_index_base') not in [0, 1]
                                  or index < value['source_index_base']):
            raise ValueError('Source row index requires an explicit 0/1 index base')
        if value.get('split', 'gold_v1') != 'gold_v1':
            raise ValueError('Metadata cannot change the frozen split')
        row.update({k: v for k, v in value.items() if k not in ['id', 'text_sha256']})


def make_lock(metadata_path=None):
    lock = core.runtime_lock()
    lock.update(settings=SETTINGS,
                prompt_sha256={p: core.file_hash(ROOT / f'prompts/{p}.txt') for p in ['P1','P2']},
                input_sha256=core.file_hash(ROOT / 'blind/gold_inputs.jsonl'),
                run_mode='main; generation requires explicit --run and --run-id',
                metadata=dict(path=str(Path(metadata_path).resolve()) if metadata_path else None,
                              sha256=core.file_hash(metadata_path) if metadata_path else None,
                              missing_values_are_explicit_null=True),
                reporting_protocol=dict(version='m1-cpu-full-report-v1', reference_labels_read=False,
                                        max_attempts=1, output_repair=False, truncated_accepts_no_spans=True,
                                        offset_unit='Unicode code points; 0-based, end-exclusive',
                                        item_rejection_denominator='all parsed top-level JSON-array items',
                                        missing_document_metadata_blocks_document_bootstrap=True,
                                        full_console_output_default=True,
                                        raw_token_ids_saved_in_claims_and_results=True,
                                        latency='inference_seconds includes prefill and token generation',
                                        cpu_time='sum of CPU time across process threads',
                                        cost='unknown; not inferred from local execution',
                                        metrics='P/R/F1/CI require separate reference-based evaluation'))
    lock['pilot_backend_code_sha256'] = lock.pop('code_sha256')
    lock['pilot_test_sha256'] = lock.pop('test_sha256')
    lock['pilot_amendment_sha256'] = lock.pop('amendment_sha256')
    lock['pilot_launcher_sha256'] = lock.pop('launcher_sha256')
    lock.pop('gold_generation_allowed')
    lock['main_files_sha256'] = {name: core.file_hash(HERE / name) for name in
                                ['run_m1_cpu.py', 'm1_reporting.py', 'test_m1_main.py',
                                 'launch_m1_cpu.py', 'CPU_MAIN_PROTOCOL.md', 'README_M1_MAIN.md']}
    pilot = HERE / 'pilot_cpu_20_P2'
    evidence = core.read(pilot / 'FINAL_VERIFICATION.json')
    if evidence.get('real_requests') != 20:
        raise ValueError('Expected the preserved 20-Dev P2 pilot evidence')
    result_files = sorted(pilot.glob('result_*.json'))
    if len(result_files) != 20:
        raise ValueError('Expected 20 immutable pilot result files')
    lock['pilot_evidence'] = dict(P2_completed=20, P1_completed=0,
                                 result_files_sha256={p.name:core.file_hash(p) for p in result_files},
                                 verification_sha256=core.file_hash(pilot / 'FINAL_VERIFICATION.json'),
                                 runtime_lock_sha256=core.file_hash(pilot / 'RUNTIME_LOCK.json'))
    return lock


def row_fields(row):
    return {k: row.get(k) for k in ['id','text','text_sha256','split','document_id',
                                   'input_row_index_1based','original_row_index','source_index_base']}


def rss_kib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    return None


def execute_one(index, row, prompt, backend, out, run_id, lock_sha, session_id):
    claim_path = out / f'claim_{index:04d}.json'
    result_path = out / f'result_{index:04d}.json'
    started = time.monotonic()
    start_utc = core.utc()
    request = backend.prepare(row, prompt)
    identity = dict(run_id=run_id, id=row['id'], text_sha256=row['text_sha256'],
                    runtime_lock_sha256=lock_sha, prompt_sha256=core.hashlib.sha256(prompt.encode()).hexdigest(),
                    request=request)
    sha = core.digest(identity)
    if result_path.exists():
        previous = core.read(result_path)
        claim = core.read(claim_path) if claim_path.exists() else {}
        if previous.get('request_sha256') != sha or claim.get('request_sha256') != sha:
            raise ValueError('Previous request identity differs; do not replay: ' + row['id'])
        claim_identity = {k:v for k,v in claim.items() if k not in
                          ['request_sha256','started_at_utc','attempt','session_id']}
        if core.digest(claim_identity) != sha or previous.get('id') != row['id']:
            raise ValueError('Saved request/ID is corrupt; do not replay')
        if previous.get('outcome') == 'runtime_failed' or not previous.get('record_complete'):
            raise ValueError('Prior failed/unresolved attempt blocks retry')
        return previous, True
    if claim_path.exists():
        raise ValueError('Unresolved claim blocks retry: ' + str(claim_path))
    core.save(claim_path, dict(identity, request_sha256=sha, started_at_utc=start_utc,
                              attempt=1, session_id=session_id))
    record = dict(row_fields(row), run_id=run_id, phase='main_gold', attempt=1,
                  session_id=session_id, started_at_utc=start_utc,
                  prompt_sha256=identity['prompt_sha256'], runtime_lock_sha256=lock_sha,
                  request_sha256=sha, cached_input_tokens=None, local_inference_cost_usd=None,
                  claim_path=str(claim_path), result_path=str(result_path))
    try:
        value = backend.generate(request)
        record.update(value)
        record['inference_seconds'] = value['elapsed_seconds']
        truncated = value['finish_reason'] == 'length'
        compiled = core.classify(row['text'], value['response'], truncated)
        record.update(compiled)
        record.update(reporting.item_accounting(row['text'], value['response'], compiled, truncated))
        record['input_tokens'] = request['input_tokens']
        record['total_tokens'] = record['input_tokens']+record['output_tokens']
        record['output_tokens_per_inference_second'] = (record['output_tokens']/record['inference_seconds']
                                                       if record['inference_seconds'] else None)
        record['current_process_rss_kib'] = rss_kib()
        record['record_complete'] = True
    except Exception as error:
        record.update(outcome='runtime_failed', error=dict(type=type(error).__name__, message=str(error)),
                      label=[], rejections=[], structural_rejection=True, record_complete=True)
        record.update(finished_at_utc=core.utc(), end_to_end_seconds=time.monotonic()-started)
        core.save(result_path, record)
        raise
    record.update(finished_at_utc=core.utc(), end_to_end_seconds=time.monotonic()-started)
    core.save(result_path, record)
    return record, False


def collect(rows, out):
    records = {}
    for i, row in enumerate(rows):
        result, claim = out / f'result_{i:04d}.json', out / f'claim_{i:04d}.json'
        if not result.exists() and not claim.exists():
            continue
        try:
            if result.exists():
                value = core.read(result)
                if value.get('id') != row['id']:
                    raise ValueError('Saved ID differs from frozen row')
                value = dict(value, response_record_sha256=core.file_hash(result))
            else:
                pending = core.read(claim)
                value = dict(row_fields(row), outcome='unresolved_claim', record_complete=False,
                             run_id=pending.get('run_id'), request_sha256=pending.get('request_sha256'),
                             runtime_lock_sha256=pending.get('runtime_lock_sha256'),
                             session_id=pending.get('session_id'),
                             started_at_utc=pending.get('started_at_utc'),
                             claim_path=str(claim), result_path=str(result),
                             label=[], rejections=[], structural_rejection=True)
        except (OSError, ValueError) as error:
            value = dict(id=row['id'], outcome='record_invalid', record_complete=False,
                         label=[], rejections=[], structural_rejection=True, error=str(error))
        records[row['id']] = value
    return records


def export(rows, records, out, run_id, summary):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = out / 'exports' / stamp
    target.mkdir(parents=True)
    predictions = []
    for row in rows:
        value = records.get(row['id'])
        predictions.append(dict(row_fields(row), label=value.get('label', []) if value else [],
                                outcome=value.get('outcome') if value else 'not_run',
                                structural_rejection=value.get('structural_rejection', False) if value else False,
                                rejections=value.get('rejections', []) if value else [],
                                request_sha256=value.get('request_sha256') if value else None,
                                response_record_sha256=value.get('response_record_sha256') if value else None))
    with (target / f'{run_id}.jsonl').open('x', encoding='utf-8') as f:
        for value in predictions:
            f.write(json.dumps(value, ensure_ascii=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    core.save(target / 'STATUS.json', summary)
    # Machine-readable per-sentence accounting; unrun/failed rows remain present.
    fields = ['run_id','id','outcome','document_id','input_row_index_1based','original_row_index',
              'input_tokens','output_tokens','total_tokens','parsed_items','accepted_spans','rejected_items',
              'rejected_item_rate_pct','inference_seconds','cpu_process_seconds','end_to_end_seconds',
              'finish_reason','peak_process_rss_kib','request_sha256','response_record_sha256']
    with (target / 'requests_summary.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            value = dict(row, **{k:v for k,v in records.get(row['id'], {}).items() if k not in row})
            value.update(run_id=run_id, outcome=records.get(row['id'], {}).get('outcome', 'not_run'))
            writer.writerow({k:value.get(k) for k in fields})
        f.flush(); os.fsync(f.fileno())
    core.save(target / 'EXPORT_MANIFEST.json',
              {q.name: core.file_hash(q) for q in target.iterdir() if q.is_file()})
    print('EXPORT_DIRECTORY '+str(target), flush=True)
    return target


def run(rows, prompts, run_id, metadata_path, compact=False):
    cpu_environment()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    session_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    started = time.monotonic()
    out = OUTPUT / run_id
    out.mkdir(exist_ok=True)
    with (OUTPUT / '.process.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = make_lock(metadata_path)
        lock_path = OUTPUT / 'RUNTIME_LOCK.json'
        if lock_path.exists():
            if core.read(lock_path) != identity:
                raise ValueError('Main runtime lock differs; no inference performed')
        else:
            core.save(lock_path, identity)
        lock_sha = core.file_hash(lock_path)
        reporting.print_header(identity, lock_sha, run_id, out)
        session = dict(session_id=session_id, started_at_utc=core.utc(), new_requests=0,
                       skipped_completed_requests=0, runtime_lock_sha256=lock_sha,
                       model_load_seconds=None, preflight_seconds=None, error=None)
        try:
            # Stop on unresolved or failed prior work before loading a model.
            prior = collect(rows, out)
            blocked = [v['id'] for v in prior.values() if v.get('outcome') in
                       ['unresolved_claim','record_invalid','runtime_failed']]
            if blocked:
                raise ValueError('Prior unresolved/failed rows block automatic resume: '+str(blocked))
            begin = time.monotonic(); backend = core.CPUModel()
            session['model_load_seconds'] = time.monotonic()-begin
            begin = time.monotonic()
            prompt = prompts[run_id.split('-')[1]]
            sizes = []
            for row in rows:
                try:
                    sizes.append(backend.prepare(row, prompt)['input_tokens'])
                except Exception as error:
                    raise ValueError(f"Preflight failed at {row['id']}: {error}") from error
            session['preflight_seconds'] = time.monotonic()-begin
            print(f"PREFLIGHT PASS ids={len(rows)} input_tokens_min={min(sizes)} "
                  f"input_tokens_max={max(sizes)} no_truncation=true HF_GGUF_tokens_match=true "
                  f"document_ids_known={sum(r['document_id'] is not None for r in rows)}/{len(rows)}", flush=True)
            for i, row in enumerate(rows):
                try:
                    value, resumed = execute_one(i, row, prompt, backend, out, run_id, lock_sha, session_id)
                except BaseException:
                    failed = collect(rows, out)
                    value = failed.get(row['id'])
                    if value:
                        value = dict(value, run_id=run_id)
                        reporting.print_record(value, i+1, len(rows),
                                               reporting.summarize(rows, failed, run_id), detailed=not compact)
                    raise
                session['skipped_completed_requests' if resumed else 'new_requests'] += 1
                value = dict(value, response_record_sha256=core.file_hash(out / f'result_{i:04d}.json'))
                prior[row['id']] = value
                summary = reporting.summarize(rows, prior, run_id)
                display = dict(value, session_elapsed_seconds=round(time.monotonic()-started, 3),
                               estimated_remaining_seconds=round(summary['latency_seconds']['mean']*(len(rows)-i-1), 1)
                               if summary['latency_seconds']['mean'] is not None else None)
                reporting.print_record(display, i+1, len(rows), summary, resumed=resumed, detailed=not compact)
        except BaseException as error:
            session['error'] = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            records = collect(rows, out)
            session['new_requests'] = sum(v.get('session_id') == session_id for v in records.values())
            session.update(finished_at_utc=core.utc(), wall_seconds=time.monotonic()-started,
                           peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            core.save(out / f'session_{session_id}.json', session)
            summary = reporting.summarize(rows, records, run_id, session)
            reporting.print_summary(summary)
            export(rows, records, out, run_id, summary)
    print('RUN STOPPED: '+run_id+'; no other run is automatically started.', flush=True)


def preview():
    rows, prompt = core.inputs()
    source = HERE / 'pilot_cpu_20_P2'
    row = rows[3]
    value = core.read(source / 'result_0003.json')
    value = dict(row, **{k:v for k,v in value.items() if k not in row})
    value.update(reporting.item_accounting(row['text'], value['response'], value))
    value.update(inference_seconds=value['elapsed_seconds'], total_tokens=value['input_tokens']+value['output_tokens'],
                 record_complete=True, current_process_rss_kib=value['peak_process_rss_kib'],
                 document_id=None, original_row_index=None, input_row_index_1based=4, split='development',
                 output_tokens_per_inference_second=value['output_tokens']/value['elapsed_seconds'],
                 claim_path=str(source / 'claim_0003.json'), result_path=str(source / 'result_0003.json'),
                 response_record_sha256=core.file_hash(source / 'result_0003.json'))
    claim = core.read(source / 'claim_0003.json')
    value.update(started_at_utc=claim['started_at_utc'],
                 prompt_sha256=core.hashlib.sha256(prompt.encode()).hexdigest(),
                 runtime_lock_sha256=core.file_hash(source / 'RUNTIME_LOCK.json'))
    summary = reporting.summarize(rows, {row['id']:value}, 'M1-DEV-P2')
    print('FORMAT PREVIEW FROM ONE EXISTING DEV RECORD. No new generation; not a Gold result.', flush=True)
    reporting.print_record(value, 4, 20, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', choices=RUN_IDS, default='M1-P2')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--export', action='store_true')
    mode.add_argument('--preview', action='store_true')
    parser.add_argument('--metadata', type=Path, help='Optional label-free source mapping JSONL, frozen before main run')
    parser.add_argument('--compact', action='store_true', help='Hide raw text/response/span details on console only')
    args = parser.parse_args()
    if args.preview:
        preview(); return
    if args.metadata is None and (OUTPUT / 'RUNTIME_LOCK.json').exists():
        saved = core.read(OUTPUT / 'RUNTIME_LOCK.json')['metadata']['path']
        args.metadata = Path(saved) if saved else None
    rows, prompts = load_inputs(args.metadata)
    if args.run:
        run(rows, prompts, args.run_id, args.metadata, args.compact)
    elif args.status or args.export:
        out = OUTPUT / args.run_id
        records = collect(rows, out)
        summary = reporting.summarize(rows, records, args.run_id)
        reporting.print_summary(summary)
        if args.export:
            export(rows, records, out, args.run_id, summary)
    else:
        print(json.dumps(dict(offline_check='PASS', sentences=len(rows), run_ids=RUN_IDS,
                              prompt_hashes={k:core.hashlib.sha256(v.encode()).hexdigest() for k,v in prompts.items()},
                              full_console_output_default=True, model_calls=0,
                              document_ids_known=sum(r['document_id'] is not None for r in rows),
                              original_row_indices_known=sum(r['original_row_index'] is not None for r in rows),
                              next_step='Use --preview to inspect output format; --run starts the selected run only'),
                         ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
