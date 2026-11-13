"""Exactly 20 frozen Dev/P2 requests; no Gold generation or automatic retries."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / 'e1_m1_assets_20261113'
OUTPUT = ROOT / 'm1_server/pilot_cpu_20_P2'
sys.path.insert(0, str(ROOT / 'compiler/tools'))
from study import compile_response

SETTINGS = dict(model='Qwen/Qwen3-8B', quantization='GGUF Q4_K_M',
                backend='llama-cpp-python', device='CPU', n_gpu_layers=0,
                n_ctx=16384, n_threads=12, n_threads_batch=12, n_batch=512,
                max_new_tokens=1024, temperature=0.0, top_k=1, top_p=1.0,
                min_p=0.0, repeat_penalty=1.0, seed=20261112,
                enable_thinking=False, prompt_id='P2', requests=20)
MODEL_SHA = 'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    """Exclusive durable record; never overwrite a claim or an inference result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inputs():
    manifest = read(ROOT / 'MANIFEST.json')
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != expected:
            raise ValueError('Package manifest mismatch: ' + name)
    rows = [json.loads(line) for line in
            (ROOT / 'pilot/dev_inputs_20.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
        raise ValueError('Expected exactly 20 distinct frozen Dev IDs')
    for row in rows:
        if set(row) != {'id', 'text', 'text_sha256'}:
            raise ValueError('Unexpected fields in blind Dev input')
        if hashlib.sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Text hash mismatch')
    return rows, (ROOT / 'prompts/P2.txt').read_text(encoding='utf-8')


def messages(row, prompt):
    return [{'role': 'system', 'content': prompt},
            {'role': 'user', 'content': json.dumps({'sentence': row['text']},
                                                  ensure_ascii=False)}]


def classify(text, response, truncated=False):
    compiled = compile_response(text, response)
    if truncated:
        # Keep the attempted compilation separately; never accept capped output.
        compiled['label'] = []
        compiled['structural_rejection'] = True
        return dict(compiled, outcome='truncated')
    return dict(compiled, outcome='malformed' if compiled['whole_response_invalid']
                else 'partially_rejected' if compiled['structural_rejection'] else 'valid')


def runtime_lock():
    import llama_cpp
    source = read(ASSETS / 'model_source.json')
    model = ASSETS / 'model/Qwen3-8B-Q4_K_M.gguf'
    if source['sha256'] != MODEL_SHA or file_hash(model) != MODEL_SHA:
        raise ValueError('Official GGUF hash mismatch')
    files = list((ASSETS / 'tokenizer').iterdir())
    cpu = next((s.split(':', 1)[1].strip() for s in
                Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')), 'unknown')
    return dict(settings=SETTINGS, model_source=source,
                tokenizer_files={p.name: file_hash(p) for p in files if p.is_file()},
                package_manifest_sha256=file_hash(ROOT / 'MANIFEST.json'),
                prompt_sha256=file_hash(ROOT / 'prompts/P2.txt'),
                input_sha256=file_hash(ROOT / 'pilot/dev_inputs_20.jsonl'),
                compiler_sha256=file_hash(ROOT / 'compiler/tools/study.py'),
                code_sha256=file_hash(__file__),
                test_sha256=file_hash(Path(__file__).with_name('test_pilot_cpu.py')),
                amendment_sha256=file_hash(Path(__file__).with_name('CPU_PILOT_AMENDMENT.md')),
                launcher_sha256=file_hash(Path(__file__).with_name('launch_cpu_pilot.py')),
                llama_cpu_build_info=llama_cpp.llama_print_system_info().decode(),
                llama_shared_libraries={str(p.relative_to(Path(llama_cpp.__file__).parent)): file_hash(p)
                                        for p in Path(llama_cpp.__file__).parent.rglob('*.so')},
                thread_environment={n: os.environ.get(n) for n in
                                    ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
                versions={n: importlib.metadata.version(n) for n in
                          ['llama-cpp-python', 'transformers', 'tokenizers', 'numpy']},
                python=sys.version, executable=sys.executable,
                platform=platform.platform(), cpu=cpu, logical_cpus=os.cpu_count(),
                local_inference_cost_usd=None, gold_generation_allowed=False)


class CPUModel:
    def __init__(self):
        # These environment variables are set by the launcher before imports.
        from transformers import AutoTokenizer
        from llama_cpp import Llama
        self.tokenizer = AutoTokenizer.from_pretrained(
            ASSETS / 'tokenizer', local_files_only=True, trust_remote_code=False)
        self.model = Llama(model_path=str(ASSETS / 'model/Qwen3-8B-Q4_K_M.gguf'),
                           n_ctx=SETTINGS['n_ctx'], n_threads=SETTINGS['n_threads'],
                           n_threads_batch=SETTINGS['n_threads_batch'],
                           n_batch=SETTINGS['n_batch'], n_gpu_layers=0,
                           seed=SETTINGS['seed'], logits_all=False, verbose=False)
        eos = read(ASSETS / 'tokenizer/generation_config.json')['eos_token_id']
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.eos.update([self.tokenizer.eos_token_id, self.model.token_eos()])

    def prepare(self, row, prompt):
        msg = messages(row, prompt)
        rendered = self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        expected = self.tokenizer(rendered, add_special_tokens=False)['input_ids']
        tokens = self.model.tokenize(rendered.encode(), add_bos=False, special=True)
        if tokens != expected:
            raise ValueError('GGUF tokenization differs from official HF tokenizer')
        if len(tokens) + SETTINGS['max_new_tokens'] > SETTINGS['n_ctx']:
            raise ValueError('Context overflow; truncation is disabled')
        return dict(messages=msg, rendered_prompt=rendered, input_token_ids=tokens,
                    input_tokens=len(tokens), official_tokenizer_match=True)

    def generate(self, request):
        tokens = []
        start = time.monotonic()
        process_start = time.process_time()
        stopped = False
        for token in self.model.generate(request['input_token_ids'],
                                         temp=0.0, top_k=1, top_p=1.0, min_p=0.0,
                                         repeat_penalty=1.0, reset=True):
            tokens.append(int(token))
            if token in self.eos:
                stopped = True
                break
            if len(tokens) >= SETTINGS['max_new_tokens']:
                break
        elapsed = time.monotonic() - start
        return dict(response=self.tokenizer.decode(tokens, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False),
                    output_token_ids=tokens, output_tokens=len(tokens),
                    input_tokens=request['input_tokens'], stopped_on_eos=stopped,
                    finish_reason='eos' if stopped else 'length',
                    elapsed_seconds=elapsed, cpu_process_seconds=time.process_time()-process_start,
                    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def execute_one(number, row, prompt, backend, out):
    claim = out / f'claim_{number:04d}.json'
    result_path = out / f'result_{number:04d}.json'
    request = backend.prepare(row, prompt)
    identity = dict(id=row['id'], text_sha256=row['text_sha256'], run_id='M1-DEV-P2',
                    settings_sha256=digest(SETTINGS), request=request)
    request_sha = digest(identity)
    if result_path.exists():
        value = read(result_path)
        if not claim.exists() or value['request_sha256'] != request_sha:
            raise ValueError('Completed request identity differs')
        if value['outcome'] == 'runtime_failed':
            raise ValueError('Prior failure blocks automatic retry')
        return value
    if claim.exists():
        raise ValueError('Unresolved prior claim; no retry: ' + str(claim))
    save(claim, dict(identity, request_sha256=request_sha, started_at_utc=utc(), attempt=1))
    try:
        value = backend.generate(request)
        value.update(classify(row['text'], value['response'], value['finish_reason'] == 'length'))
    except Exception as error:
        value = dict(outcome='runtime_failed', error_type=type(error).__name__,
                     error=str(error), label=[], rejections=[], structural_rejection=True)
        save(result_path, dict(value, id=row['id'], request_sha256=request_sha,
                               finished_at_utc=utc(), attempt=1, run_id='M1-DEV-P2'))
        raise
    value.update(id=row['id'], run_id='M1-DEV-P2', text_sha256=row['text_sha256'],
                 request_sha256=request_sha, finished_at_utc=utc(), attempt=1)
    save(result_path, value)
    return value


def summarize(out, rows, started=None):
    records = []
    predictions = []
    rejected = Counter()
    for number, row in enumerate(rows):
        path = out / f'result_{number:04d}.json'
        value = read(path) if path.exists() else None
        if value:
            records.append(value)
            rejected.update(r['reason'] for r in value['rejections'])
        predictions.append(dict(row, label=value['label'] if value else [],
                                outcome=value['outcome'] if value else 'not_run',
                                structural_rejection=value['structural_rejection'] if value else False,
                                rejections=value['rejections'] if value else [],
                                request_sha256=value['request_sha256'] if value else None,
                                response_record_sha256=file_hash(path) if value else None))
    latencies = [v['elapsed_seconds'] for v in records if 'elapsed_seconds' in v]
    report = dict(timestamp_utc=utc(), configured_requests=20, completed=len(records),
                  complete=len(records) == 20, run_id='M1-DEV-P2', settings=SETTINGS,
                  outcomes=dict(Counter(v['outcome'] for v in predictions)),
                  rejected_items_by_reason=dict(rejected),
                  accepted_spans=sum(len(v['label']) for v in predictions),
                  partially_rejected_with_no_spans=sum(v['outcome']=='partially_rejected' and
                                                       not v['label'] for v in predictions),
                  input_tokens=sum(v.get('input_tokens', 0) for v in records),
                  output_tokens=sum(v.get('output_tokens', 0) for v in records),
                  inference_seconds=sum(latencies),
                  mean_latency_seconds=statistics.mean(latencies) if latencies else None,
                  median_latency_seconds=statistics.median(latencies) if latencies else None,
                  total_process_seconds=time.monotonic()-started if started else None,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  local_inference_cost_usd=None, reference_labels_read=False,
                  gold_requests=0, semantic_quality_evaluated=False)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    save(out / f'STATUS_{stamp}.json', report)
    prediction_path = out / f'predictions_{stamp}.jsonl'
    with prediction_path.open('x', encoding='utf-8') as f:
        for value in predictions:
            f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    rows, prompt = inputs()
    if args.status:
        summarize(OUTPUT, rows)
        return
    if not args.run:
        print('OFFLINE PASS: exactly 20 frozen Dev/P2 rows; no Gold execution exists.')
        return
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.process.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = runtime_lock()
        lock_path = OUTPUT / 'RUNTIME_LOCK.json'
        if lock_path.exists():
            if read(lock_path) != identity:
                raise ValueError('Runtime lock differs; no inference performed')
        else:
            save(lock_path, identity)
        print('Runtime/model SHA-256 lock sealed. Loading CPU model...', flush=True)
        backend = CPUModel()
        print('Model loaded; validating exact tokenization and context for all 20 Dev rows...', flush=True)
        prepared = [backend.prepare(row, prompt) for row in rows]
        print('Preflight PASS: HF/GGUF tokens match; input token range',
              min(r['input_tokens'] for r in prepared), max(r['input_tokens'] for r in prepared), flush=True)
        try:
            for number, row in enumerate(rows):
                value = execute_one(number, row, prompt, backend, OUTPUT)
                print(f"{number+1}/20 M1-DEV-P2 {row['id']}: {value['outcome']} "
                      f"input={value['input_tokens']} output={value['output_tokens']} "
                      f"seconds={value['elapsed_seconds']:.2f}", flush=True)
        finally:
            summarize(OUTPUT, rows, started)
        print('DONE: 20 Dev only. No Gold run scheduled.', flush=True)


if __name__ == '__main__':
    main()
