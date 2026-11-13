"""Continue the tested CPU M1: reuse Dev/P2, run Dev/P1, seal, run both Gold jobs.

One authoritative filesystem ledger; no API, retry, reference scoring or reset.
"""
import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / 'full_cpu_e1_v1'
sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='12', OPENBLAS_NUM_THREADS='1',
                  HF_HOME=str(ROOT.parent/'e1_m1_assets_20261113/hf_cache'),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
import pilot_cpu as core
import m1_reporting as reporting

BASE_SETTINGS = {k:v for k,v in core.SETTINGS.items() if k not in ['prompt_id','requests']}
GOOD = {'valid','partially_rejected','malformed','truncated'}
CODE = ['full_cpu_driver.py','test_full_cpu.py','launch_full_cpu.py','package_full_cpu.py','CPU_FULL_AMENDMENT.md']


def rows(path, expected=None):
    data=[json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if expected is not None and len(data)!=expected:
        raise ValueError('Incorrect blind row count: '+str(path))
    if len({r['id'] for r in data})!=len(data):raise ValueError('Duplicate blind ID')
    for r in data:
        if set(r)!={'id','text','text_sha256'} or hashlib.sha256(r['text'].encode()).hexdigest()!=r['text_sha256']:
            raise ValueError('Blind fields/text hash changed')
    return data


def verify_manifest():
    for name, digest in core.read(ROOT/'MANIFEST.json')['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or core.file_hash(p)!=digest:
            raise ValueError('Supplied manifest mismatch: '+name)


def prompt(pid):
    if pid not in ['P1','P2']:raise ValueError('Unexpected prompt')
    return (ROOT/f'prompts/{pid}.txt').read_text(encoding='utf-8')


def stable_code():
    return {name:core.file_hash(HERE/name) for name in CODE}


def runtime_identity():
    actual=core.runtime_lock()
    old=core.read(HERE/'pilot_cpu_20_P2/RUNTIME_LOCK.json')
    if actual!=old:
        fields=[k for k in set(actual)|set(old) if actual.get(k)!=old.get(k)]
        raise ValueError('Tested pilot runtime changed: '+str(fields))
    if old['gold_generation_allowed'] is not False:raise ValueError('Original Dev-only lock changed')
    return dict(tested_pilot_runtime=actual, tested_pilot_lock_sha256=core.file_hash(HERE/'pilot_cpu_20_P2/RUNTIME_LOCK.json'),
                settings=BASE_SETTINGS, code_sha256=stable_code(),
                prompt_sha256={p:core.file_hash(ROOT/f'prompts/{p}.txt') for p in ['P1','P2']},
                input_sha256={'development':core.file_hash(ROOT/'pilot/dev_inputs_20.jsonl'),
                              'gold_blind':core.file_hash(ROOT/'blind/gold_inputs.jsonl')},
                tokenizer_provenance=core.read(core.ASSETS/'tokenizer_source.json'),
                official_chat_template_sha256=hashlib.sha256(core.read(core.ASSETS/'tokenizer/tokenizer_config.json')['chat_template'].encode()).hexdigest(),
                compiler_sha256=core.file_hash(ROOT/'compiler/tools/study.py'),
                offline_tests_log_sha256=core.file_hash(OUTPUT/'logs/offline_tests.log'),
                CPU_quantization_amendment=True, originally_proposed_BF16_GPU_experiment=False,
                reference_labels_read=False, semantic_repair=False, retries=0, local_cost_usd=None,
                creator_names=[])


def lock_once(path, identity):
    if path.exists():
        if core.read(path)!=identity:raise ValueError('Sealed lock differs: '+str(path))
    else:core.save(path, identity)
    return core.file_hash(path)


def verify_reused_p2(backend):
    data=rows(ROOT/'pilot/dev_inputs_20.jsonl',20)
    old=HERE/'pilot_cpu_20_P2'
    records=[]
    for i,row in enumerate(data):
        claim=core.read(old/f'claim_{i:04d}.json'); result=core.read(old/f'result_{i:04d}.json')
        req=backend.prepare(row,prompt('P2'))
        identity=dict(id=row['id'],text_sha256=row['text_sha256'],run_id='M1-DEV-P2',
                      settings_sha256=core.digest(core.SETTINGS),request=req)
        digest=core.digest(identity)
        if claim['request_sha256']!=digest or result['request_sha256']!=digest or result['id']!=row['id']:
            raise ValueError('Reused P2 request identity differs')
        saved={k:v for k,v in claim.items() if k not in ['request_sha256','started_at_utc','attempt']}
        if core.digest(saved)!=digest:raise ValueError('Reused P2 claim is corrupt')
        if result['outcome'] not in GOOD:raise ValueError('Reused P2 is incomplete/failed')
        if backend.tokenizer.decode(result['output_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=result['response']:
            raise ValueError('Reused P2 decoded text differs')
        if len(result['output_token_ids'])!=result['output_tokens']:raise ValueError('Reused P2 token count differs')
        compiled=core.classify(row['text'],result['response'],result['finish_reason']=='length')
        if any(result.get(k)!=v for k,v in compiled.items()):raise ValueError('Reused P2 compiler differs')
        records.append(dict(index=i,id=row['id'],claim_sha256=core.file_hash(old/f'claim_{i:04d}.json'),
                            result_sha256=core.file_hash(old/f'result_{i:04d}.json')))
    report=dict(real_completed_requests=20,reused_without_generation=True,new_P2_Dev_generations=0,
                exact_tokenization_and_raw_decode_verified=True,strict_compiler_recomputed_equal=True,
                records=records,old_lock_gold_generation_allowed=False)
    lock_once(OUTPUT/'evidence/reused_Dev_P2.json',report)
    return report


def request_identity(row,pid,run_id,backend,lock_sha):
    req=backend.prepare(row,prompt(pid))
    return dict(run_id=run_id,source=row,runtime_lock_sha256=lock_sha,
                settings_sha256=core.digest(BASE_SETTINGS),prompt_sha256=core.file_hash(ROOT/f'prompts/{pid}.txt'),
                request=req)


def execute_one(index,row,pid,run_id,backend,lock_sha,session_id):
    directory=OUTPUT/'ledger'/run_id; directory.mkdir(parents=True,exist_ok=True)
    claim_path=directory/f'claim_{index:04d}.json'; result_path=directory/f'result_{index:04d}.json'
    identity=request_identity(row,pid,run_id,backend,lock_sha); digest=core.digest(identity)
    if result_path.exists():
        saved=core.read(result_path); claim=core.read(claim_path) if claim_path.exists() else {}
        identity_saved={k:v for k,v in claim.items() if k not in ['request_sha256','started_at_utc','session_id','attempt']}
        if core.digest(identity_saved)!=digest or claim.get('request_sha256')!=digest or saved.get('request_sha256')!=digest:
            raise ValueError('Completed request identity differs; no generation')
        if saved.get('id')!=row['id'] or saved.get('run_id')!=run_id or saved.get('text_sha256')!=row['text_sha256']:
            raise ValueError('Completed result source differs; no generation')
        if saved.get('outcome') not in GOOD or saved.get('record_complete') is not True:
            raise ValueError('Prior failed/unresolved attempt blocks generation')
        if core.digest(saved.get('generation'))!=saved.get('generation_sha256'):
            raise ValueError('Completed raw generation integrity differs')
        raw=saved['generation']; compiled=core.classify(row['text'],raw['response'],raw['finish_reason']=='length')
        if any(saved.get(k)!=v for k,v in compiled.items()):raise ValueError('Saved compilation differs')
        return saved,True
    if claim_path.exists():raise ValueError('Unresolved claim; never repeat: '+str(claim_path))
    core.save(claim_path,dict(identity,request_sha256=digest,started_at_utc=core.utc(),session_id=session_id,attempt=1))
    start=time.monotonic()
    record=dict(**row,run_id=run_id,phase='development' if run_id=='M1-DEV-P1' else 'gold',
                session_id=session_id,request_sha256=digest,request_record_sha256=core.file_hash(claim_path),
                runtime_lock_sha256=lock_sha,attempt=1,local_inference_cost_usd=None,cached_input_tokens=None)
    try:
        generated=backend.generate(identity['request'])
        record.update(generation=generated,generation_sha256=core.digest(generated),**generated)
        record.update(core.classify(row['text'],generated['response'],generated['finish_reason']=='length'))
        record.update(reporting.item_accounting(row['text'],generated['response'],record,generated['finish_reason']=='length'))
        record.update(record_complete=True,inference_seconds=generated['elapsed_seconds'],
                      total_tokens=generated['input_tokens']+generated['output_tokens'])
    except BaseException as error:
        record.update(outcome='runtime_failed',error={'type':type(error).__name__,'message':str(error)},
                      label=[],rejections=[],structural_rejection=True,record_complete=False,
                      generation=record.get('generation'),
                      usage_status='unknown unless preserved in generation; no retry permitted')
        record.update(finished_at_utc=core.utc(),end_to_end_seconds=time.monotonic()-start)
        core.save(result_path,record)
        raise
    record.update(finished_at_utc=core.utc(),end_to_end_seconds=time.monotonic()-start)
    core.save(result_path,record)
    return record,False


def inspect_records(data,run_id):
    values={}
    for i,row in enumerate(data):
        directory=OUTPUT/'ledger'/run_id
        p=directory/f'result_{i:04d}.json'; c=directory/f'claim_{i:04d}.json'
        if not p.exists() and not c.exists():continue
        try:
            if p.exists():
                v=core.read(p)
                if v.get('id')!=row['id'] or v.get('text_sha256')!=row['text_sha256'] or v.get('run_id')!=run_id or not c.is_file():
                    raise ValueError('Result source or claim differs')
                v=dict(v,response_record_sha256=core.file_hash(p),request_record_sha256=core.file_hash(c))
            else:v=dict(id=row['id'],outcome='unresolved_claim',record_complete=False,
                        structural_rejection=True,label=[],rejections=[],request_record_sha256=core.file_hash(c))
        except (OSError,ValueError,KeyError) as error:
            v=dict(id=row['id'],outcome='record_invalid',record_complete=False,structural_rejection=True,
                   label=[],rejections=[],error=str(error))
        values[row['id']]=v
    return values


def export_run(data,run_id,directory):
    records=inspect_records(data,run_id); predictions=[]
    for row in data:
        v=records.get(row['id'],{})
        predictions.append(dict(**row,label=v.get('label',[]),outcome=v.get('outcome','not_run'),
                                structural_rejection=v.get('structural_rejection',False),rejections=v.get('rejections',[]),
                                request_sha256=v.get('request_sha256'),request_record_sha256=v.get('request_record_sha256'),
                                response_record_sha256=v.get('response_record_sha256')))
    path=directory/f'{run_id}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        for v in predictions:f.write(json.dumps(v,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    returned=[v for v in records.values() if v.get('outcome') in GOOD and v.get('record_complete') is True]
    latencies=[v['elapsed_seconds'] for v in returned]
    return dict(expected_rows=len(data),prediction_rows=len(predictions),attempts=len(records),
                completed_generations=len(returned),complete=len(returned)==len(data),
                outcomes=dict(Counter(v['outcome'] for v in predictions)),
                input_tokens=sum(v['input_tokens'] for v in returned),output_tokens=sum(v['output_tokens'] for v in returned),
                inference_seconds=sum(latencies),cpu_process_seconds=sum(v['cpu_process_seconds'] for v in returned),
                latency_mean_seconds=statistics.mean(latencies) if latencies else None,
                latency_median_seconds=statistics.median(latencies) if latencies else None,
                peak_process_rss_kib=max((v.get('peak_process_rss_kib',0) for v in returned),default=None),
                accepted_spans=sum(len(v['label']) for v in predictions),
                local_inference_cost_usd=None,cached_input_tokens=None,reference_labels_read=False,
                prediction_sha256=core.file_hash(path))


def export_all(reason):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=OUTPUT/'exports'/stamp; directory.mkdir(parents=True)
    summary={'timestamp_utc':core.utc(),'reason':reason,'runs':{},'local_inference_cost_usd':None,
             'reference_labels_read':False,'Gold_scoring_performed':False,'API_calls':0,'creator_names':[]}
    dev=rows(ROOT/'pilot/dev_inputs_20.jsonl',20)
    summary['Dev_P1']=export_run(dev,'M1-DEV-P1',directory/'development')
    gold=rows(ROOT/'blind/gold_inputs.jsonl',1000)
    for run in ['M1-P1','M1-P2']:
        summary['runs'][run]=export_run(gold,run,directory/'predictions')
    summary['complete']=all(v['complete'] for v in summary['runs'].values())
    summary['Dev_P2_reused']=len(list((HERE/'pilot_cpu_20_P2').glob('result_*.json')))==20
    core.save(directory/'STATUS.json',summary)
    # Derived current view only. Immutable export snapshots and raw ledger remain.
    for run in ['M1-P1','M1-P2']:
        target=OUTPUT/'predictions'/f'{run}.jsonl';temp=target.with_suffix('.jsonl.tmp')
        with temp.open('wb') as f:
            f.write((directory/f'predictions/{run}.jsonl').read_bytes());f.flush();os.fsync(f.fileno())
        os.replace(temp,target)
    current=OUTPUT/'STATUS.json';temp=current.with_suffix('.json.tmp')
    with temp.open('w') as f:
        json.dump(summary,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,current)
    print(json.dumps({'export':str(directory),'complete':summary['complete'],
                      'completed':{r:v['completed_generations'] for r,v in summary['runs'].items()}},indent=2),flush=True)
    return summary


def preexisting_failures():
    for c in (OUTPUT/'ledger').glob('*/claim_*.json'):
        p=c.with_name(c.name.replace('claim_','result_',1))
        if not p.is_file():raise ValueError('Unresolved claim blocks all automatic resume: '+str(c))
        v=core.read(p)
        if v.get('outcome') not in GOOD or v.get('record_complete') is not True:
            raise ValueError('Failed/incomplete ledger attempt blocks automatic resume: '+str(p))


def run_all():
    started=time.monotonic(); process=time.process_time()
    session_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session={'session_id':session_id,'started_at_utc':core.utc(),'new_requests':{},'skipped_completed':{},'error':None}
    with (OUTPUT/'.process.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify_manifest(); preexisting_failures()
            identity=runtime_identity()
            dev_lock=dict(**identity,phase='development',run_ids=['M1-DEV-P1'],gold_generation_allowed=False)
            dev_sha=lock_once(OUTPUT/'DEV_P1_RUNTIME_LOCK.json',dev_lock)
            begin=time.monotonic();backend=core.CPUModel()
            session['model_load_seconds']=time.monotonic()-begin
            verify_reused_p2(backend)
            dev=rows(ROOT/'pilot/dev_inputs_20.jsonl',20)
            for row in dev:backend.prepare(row,prompt('P1'))
            for i,row in enumerate(dev):
                v,resumed=execute_one(i,row,'P1','M1-DEV-P1',backend,dev_sha,session_id)
                key='skipped_completed' if resumed else 'new_requests';session[key]['M1-DEV-P1']=session[key].get('M1-DEV-P1',0)+1
                print(f"M1-DEV-P1 {i+1}/20 {row['id']} {v['outcome']} tokens={v['output_tokens']} seconds={v['elapsed_seconds']:.2f} resumed={resumed}",flush=True)
            p1=inspect_records(dev,'M1-DEV-P1')
            if len(p1)!=20 or any(v['outcome'] not in GOOD for v in p1.values()):raise ValueError('Dev/P1 incomplete')
            full=dict(**identity,phase='gold',gold_generation_allowed=True,run_ids=['M1-P1','M1-P2'],
                      sentences_per_run=1000,primary_run='M1-P2',
                      Dev_P1_lock_sha256=dev_sha,
                      Dev_P1_results_sha256={f'result_{i:04d}.json':core.file_hash(OUTPUT/f'ledger/M1-DEV-P1/result_{i:04d}.json') for i in range(20)},
                      Dev_P2_reuse_verification_sha256=core.file_hash(OUTPUT/'evidence/reused_Dev_P2.json'))
            full_sha=lock_once(OUTPUT/'RUNTIME_LOCK.json',full)
            print('FULL CPU LOCK SEALED '+full_sha+'; 20 Dev/P1 complete, 20 Dev/P2 reused.',flush=True)
            gold=rows(ROOT/'blind/gold_inputs.jsonl',1000)
            # Tokenization/context validation for both fixed prompts before any Gold generation.
            begin=time.monotonic();counts={}
            for pid in ['P1','P2']:
                sizes=[backend.prepare(row,prompt(pid))['input_tokens'] for row in gold]
                counts[pid]={'minimum':min(sizes),'maximum':max(sizes),'rows':len(sizes),'official_tokenizer_match':True,'truncation':False}
            lock_once(OUTPUT/'evidence/Gold_tokenization_preflight.json',counts)
            session['gold_tokenization_preflight_seconds']=time.monotonic()-begin
            export_all('Gold runtime lock sealed; preflight passed')
            for run_id in ['M1-P1','M1-P2']:
                pid=run_id.split('-')[1]
                for i,row in enumerate(gold):
                    v,resumed=execute_one(i,row,pid,run_id,backend,full_sha,session_id)
                    key='skipped_completed' if resumed else 'new_requests';session[key][run_id]=session[key].get(run_id,0)+1
                    print(f"{run_id} {i+1}/1000 {row['id']} {v['outcome']} input={v['input_tokens']} output={v['output_tokens']} seconds={v['elapsed_seconds']:.2f} resumed={resumed}",flush=True)
                    if (i+1)%50==0:export_all('periodic checkpoint; immutable raw ledger retained')
                export_all(run_id+' completed')
        except BaseException as error:
            session['error']={'type':type(error).__name__,'message':str(error)}
            raise
        finally:
            session.update(finished_at_utc=core.utc(),wall_seconds=time.monotonic()-started,
                           cpu_process_seconds=time.process_time()-process,
                           peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            core.save(OUTPUT/f'session_{session_id}.json',session)
            export_all('session stopped; inspect complete flag and errors')


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',action='store_true');group.add_argument('--export',action='store_true');group.add_argument('--status',action='store_true')
    args=p.parse_args()
    if args.run:run_all()
    elif args.export:
        with (OUTPUT/'.process.lock').open('a') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);export_all('explicit offline export')
    else:
        if (OUTPUT/'STATUS.json').exists():print(json.dumps(core.read(OUTPUT/'STATUS.json'),indent=2))
        else:print('No Gold export yet. Inspect immutable ledger/log progress; no inference started by --status.')


if __name__=='__main__':main()
