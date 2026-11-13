"""Shared paths, immutable experiment configuration and atomic artifacts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

EXP = Path(__file__).resolve().parent
PACKAGE = EXP.parent.parent
sys.path.insert(0, str(PACKAGE / 'scripts'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from train_crf_pilot import read_jsonl, write_jsonl, score, decode, encode, tokenize, error_analysis
from train_phobert_pilot import collate, bucket_batches

ARCHITECTURES = ('phobert', 'bilstm', 'bilstm_crf')
SEEDS = (42, 999, 2026)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def log_event(run, event, **details):
    value = dict(time_utc=datetime.now(timezone.utc).isoformat(), event=event, **details)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    if run is not None:
        with (Path(run) / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    return value


def protocol():
    return read_json(EXP / 'protocol.json')


def source_hashes():
    return {p.name: sha(p) for p in sorted(EXP.glob('*.py'))}


def verify_sources():
    expected = read_json(EXP / 'SOURCE_MANIFEST.json')['files']
    assert all(sha(EXP / name) == digest for name, digest in expected.items()), 'Experiment source/configuration changed'
    frozen = read_json(PACKAGE / 'MANIFEST.json')['files']
    assert all(sha(PACKAGE / name) == digest for name, digest in frozen.items()), 'Original package changed'
    return expected


def versions():
    return {name: importlib.metadata.version(name) for name in
            ['torch', 'transformers', 'tokenizers', 'numpy', 'safetensors', 'py-vncorenlp', 'pyjnius', 'pytorch-crf']}


def setup_java():
    runtimes = sorted((EXP / 'tools').glob('jdk-*-jre'))
    assert runtimes, 'Project-local Java runtime is missing'
    runtime = runtimes[-1]
    os.environ['JAVA_HOME'] = str(runtime)
    os.environ['PATH'] = str(runtime / 'bin') + os.pathsep + os.environ['PATH']
    assets = read_json(EXP / 'tools/vncorenlp_assets.json')
    for name, info in assets['files'].items():
        assert sha(EXP / 'tools/vncorenlp' / name) == info['sha256'], f'VnCoreNLP asset changed: {name}'
    return assets


def segmenter():
    setup_java()
    from py_vncorenlp import VnCoreNLP
    cwd = Path.cwd()
    try:
        model = VnCoreNLP(annotators=['wseg'], save_dir=str(EXP / 'tools/vncorenlp'))
    finally:
        os.chdir(cwd)  # The Python wrapper changes the process working directory.
    return model


def run_path(architecture, seed):
    assert architecture in ARCHITECTURES and seed in SEEDS
    return EXP / 'runs' / f'{architecture}_seed{seed}'


def load_cache():
    manifest = read_json(EXP / 'cache/manifest.json')
    current, previous = compatible_plans()
    assert manifest['protocol_sha256'] in {sha(EXP / 'protocol.json'), sha(legacy_revision() / 'protocol.json')}, 'Cache belongs to an incompatible protocol'
    assert manifest['preprocessing_code_sha256'] == sha(EXP / 'preprocessing.py')
    assert manifest['versions'] == {n: versions()[n] for n in ['tokenizers', 'py-vncorenlp', 'pyjnius']}
    for name, digest in manifest['inputs'].items():
        assert sha(PACKAGE / name) == digest, f'Preprocessing input changed: {name}'
    for name, digest in manifest['assets'].items():
        assert sha(EXP / 'tools/vncorenlp' / name) == digest
    for name, digest in manifest['files'].items():
        assert sha(EXP / 'cache' / name) == digest, f'Cache changed: {name}'
    return manifest, read_jsonl(EXP / 'cache/train.jsonl'), read_jsonl(EXP / 'cache/development.jsonl')


def legacy_revision():
    """Pinned prior implementation, retained for verifying already completed runs."""
    return EXP / 'revisions/all_epoch_checkpoints_v1'


def compatible_plans():
    current = protocol()
    previous = read_json(legacy_revision() / 'protocol.json')
    # Storage is the only protocol change; training, model selection and preprocessing stay fixed.
    without_storage = lambda plan: {k: v for k, v in plan.items() if k != 'checkpoint_policy'}
    assert without_storage(current) == without_storage(previous), 'Legacy compatibility only permits checkpoint-storage changes'
    for name in ['model.py', 'preprocessing.py']:
        assert sha(EXP / name) == sha(legacy_revision() / name), f'Numerical/preprocessing code changed: {name}'
    return current, previous


def validate_run_spec(spec):
    current, previous = compatible_plans()
    if spec.get('checkpoint_policy') == current['checkpoint_policy']:
        plan, codes, retention = current, source_hashes(), 'best_only'
    elif spec.get('checkpoint_policy') == previous['checkpoint_policy']:
        plan = previous
        codes = {p.name: sha(p) for p in sorted(legacy_revision().glob('*.py'))}
        retention = 'all_epochs'
    else:
        raise AssertionError('Unknown checkpoint-storage policy')
    assert all(spec.get(k) == v for k, v in plan.items()), 'Training protocol differs'
    assert spec['code_sha256'] == codes, 'Run source hashes do not match its pinned revision'
    assert spec['versions'] == versions(), 'Run software versions differ'
    assert spec['cache_manifest_sha256'] == sha(EXP / 'cache/manifest.json'), 'Run cache differs'
    return retention


def retained_checkpoints(run, result):
    """Validate actual retained checkpoint inventory without requiring 20 files in new runs."""
    retention = validate_run_spec(result['configuration'])
    if retention == 'all_epochs':
        paths = [run / f'checkpoints/epoch_{epoch:02d}.pt' for epoch in range(1, 21)]
        assert all(path.is_file() for path in paths), 'Legacy epoch checkpoint missing'
    else:
        path = run / 'best_checkpoint.pt'
        assert path.is_file() and not path.is_symlink(), 'Best checkpoint missing'
        assert not list((run / 'checkpoints').glob('epoch_*.pt')), 'New run unexpectedly retained epoch checkpoints'
        assert result['retained_checkpoint_epoch'] == result['best_development_epoch']
        assert sha(path) == result['best_checkpoint_sha256'], 'Best checkpoint hash differs'
        paths = [path]
    return retention, paths
