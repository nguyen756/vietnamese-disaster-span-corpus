"""Start only the 20-Dev CPU pilot in its own persistent tmux session."""
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = ROOT.parent / 'e1_m1_env_20261113/bin/python'
SESSION = 'e1_m1_cpu_dev20_20261113'


def inside():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='12', OPENBLAS_NUM_THREADS='1',
               HF_HOME=str(ROOT.parent / 'e1_m1_assets_20261113/hf_cache'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    log = ROOT / 'server_logs/m1_cpu_dev20_P2.log'
    with log.open('a', encoding='utf-8') as f:
        child = subprocess.Popen([str(PYTHON), '-u', str(HERE / 'pilot_cpu.py'), '--run'],
                                 cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
        code = child.wait()
        message = f'\nPILOT_EXIT_CODE={code}; Dev-only runner stopped. No Gold scheduled.\n'
        print(message, flush=True); f.write(message); f.flush(); os.fsync(f.fileno())
        return code


def main():
    if '--inside' in sys.argv:
        return inside()
    check = subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True)
    if check.returncode == 0:
        raise SystemExit('Session already exists; inspect it instead of launching twice: ' + SESSION)
    command = shlex.join([str(PYTHON), '-u', str(Path(__file__).resolve()), '--inside'])
    subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, '-c', str(ROOT), command], check=True)
    subprocess.run(['tmux', 'set-option', '-t', SESSION, 'remain-on-exit', 'on'], check=True)
    print('Started tmux session: ' + SESSION)
    print('View: tmux attach -t ' + SESSION)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
