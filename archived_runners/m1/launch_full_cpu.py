"""Start the authorized two-Gold-run CPU M1 continuation once in tmux."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUTPUT=HERE/'full_cpu_e1_v1'
PYTHON=ROOT.parent/'e1_m1_env_20261113/bin/python'
SESSION='e1_m1_cpu_full_20261113'


def main():
    if '--inside' not in sys.argv:
        check=subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True)
        if check.returncode==0:raise SystemExit('Session already exists; inspect '+SESSION+'; no second launch.')
        cmd=shlex.join([str(PYTHON),'-B','-u',str(Path(__file__).resolve()),'--inside'])
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(ROOT),cmd],check=True)
        subprocess.run(['tmux','set-option','-t',SESSION,'remain-on-exit','on'],check=True)
        print('Started tmux '+SESSION+'; attach: tmux attach -t '+SESSION)
        return
    os.umask(0o077)
    env=dict(os.environ)
    for n in ['OPENAI_API_KEY','TOGETHER_API_KEY','HF_TOKEN']:env.pop(n,None)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='12',OPENBLAS_NUM_THREADS='1',
               HF_HOME=str(ROOT.parent/'e1_m1_assets_20261113/hf_cache'),HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    def run(name,command):
        meta={'argv':command,'cwd':str(ROOT),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'API_calls':0}
        with (OUTPUT/f'logs/{name}_{stamp}.command.json').open('x') as f:json.dump(meta,f,indent=2)
        with (OUTPUT/f'logs/{name}_{stamp}.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=proc.wait()
        with (OUTPUT/f'logs/{name}_{stamp}.exit.json').open('x') as f:
            json.dump({'exit_code':code,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
        return code
    code=run('full_cpu',[str(PYTHON),'-B','-u',str(HERE/'full_cpu_driver.py'),'--run'])
    print('GENERATION PROCESS EXIT '+str(code)+'; preserving ledger; no automatic restart.',flush=True)
    package_code=run('package',[str(PYTHON),'-B','-u',str(HERE/'package_full_cpu.py')])
    print('PACKAGE PROCESS EXIT '+str(package_code),flush=True)
    raise SystemExit(code or package_code)


if __name__=='__main__':main()
