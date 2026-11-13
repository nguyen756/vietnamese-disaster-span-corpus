"""Luna M2: one Dev parameter canary, then two locked 1,000-row Gold runs."""
import argparse,datetime,hashlib,json,os,platform,sys,zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path
import api_transport as api
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compiler/tools'))
from study import compile_response

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s)for s in p.read_text(encoding='utf-8').splitlines()if s.strip()]
def hashed(value):return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def prepare():
    manifest=read(ROOT/'MANIFEST.json')
    for name,expected in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT)or sha(path.read_bytes())!=expected:raise ValueError('Input/code lock mismatch: '+name)
    gold=rows(ROOT/'blind/gold_inputs.jsonl');dev=rows(ROOT/'pilot/dev_inputs_20.jsonl')
    if len(gold)!=1000 or len({r['id']for r in gold})!=1000:raise ValueError('Expected exactly 1,000 distinct Gold inputs')
    for row in gold+dev:
        if set(row)!={'id','text','text_sha256'}or sha(row['text'].encode())!=row['text_sha256']:raise ValueError('Blind payload/hash violation')
    prompts={p:(ROOT/'prompts'/f'{p}.txt').read_text(encoding='utf-8')for p in ['P1','P2']}
    config=read(ROOT/'m2_luna/config.json')
    jobs=[dict(**dev[0],run_id='M2-DEV-CANARY',prompt_id='P2',phase='development')]
    jobs += [dict(**r,run_id=run,prompt_id=run.split('-')[1],phase='gold')for run in config['run_ids']for r in gold]
    assert len(jobs)==2001
    return jobs,prompts,config

def body(job,prompts,config):
    return dict(model=config['model'],instructions=prompts[job['prompt_id']],input=json.dumps({'sentence':job['text']},ensure_ascii=False),max_output_tokens=config['max_output_tokens'],reasoning={'effort':'none'},temperature=config['temperature'],top_p=config['top_p'],service_tier='default',tools=[],store=False,truncation='disabled')

def cost(usage):
    if not isinstance(usage,dict):return None
    d=usage.get('input_tokens_details')or{}
    values=[usage.get('input_tokens'),usage.get('output_tokens'),d.get('cached_tokens'),d.get('cache_write_tokens')]
    if any(type(x)is not int or x<0 for x in values):return None
    i,o,c,w=values
    if c+w>i:return None
    return str((Decimal(i-c-w)*Decimal('.10')+Decimal(c)*Decimal('.01')+Decimal(w)*Decimal('.125')+Decimal(o)*Decimal('.50'))/Decimal(1000000))

def decode(job,response,config):
    raw=response.get('data')
    result={k:v for k,v in response.items()if k!='data'}
    result.update(id=job['id'],run_id=job['run_id'],phase=job['phase'],text_sha256=job['text_sha256'],raw_provider_response=raw,label=[],rejections=[],structural_rejection=True,outcome='transport_failed',usage=None,calculated_cost_usd=None)
    if not isinstance(raw,dict):return result
    result['usage']=raw.get('usage');result['calculated_cost_usd']=cost(raw.get('usage'))
    if raw.get('model')!=config['model']:result['outcome']='model_mismatch';return result
    if raw.get('status')!='completed':result['outcome']='incomplete_or_failed';return result
    # Requested values must not be silently changed by an API alias/provider.
    for field in ['temperature','top_p']:
        if raw.get(field)!=config[field]:result['outcome']='sampling_setting_unverified';return result
    text=''.join(c.get('text','')for o in raw.get('output',[])if o.get('type')=='message'for c in o.get('content',[])if c.get('type')=='output_text')
    result['response']=text
    compiled=compile_response(job['text'],text);result.update(compiled)
    result['outcome']='malformed'if compiled['whole_response_invalid']else 'partially_rejected'if compiled['structural_rejection']else 'valid'
    return result

def fatal(result):
    return result['calculated_cost_usd']is None or result['outcome']in ['transport_failed','model_mismatch','incomplete_or_failed','sampling_setting_unverified']

def execute(jobs,prompts,config,key,transport=api.http,output=None):
    out=output or ROOT/'m2_luna/results';out.mkdir(parents=True,exist_ok=True)
    guard=out/'.running'
    with guard.open('x')as f:f.write(api.utc());f.flush();os.fsync(f.fileno())
    try:
        identity={'manifest_sha256':sha((ROOT/'MANIFEST.json').read_bytes()),'jobs_sha256':hashed(jobs),'config_sha256':hashed(config),'prompts_sha256':hashed(prompts)}
        lock=out/'RUN_LOCK.json'
        if lock.exists():
            if read(lock)!=identity:raise ValueError('Run lock differs; no calls made')
        else:
            save(lock,identity)
            save(out/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'device':'remote API; local CPU client','started_at_utc':api.utc(),'billing_receipt':None})
        reserved=Decimal(config['prior_completed_pilot_reservations_usd'])+sum(Decimal(read(p)['reserved_usd'])for p in out.glob('claim_*.json'))
        for number,job in enumerate(jobs):
            req=body(job,prompts,config);identity_sha=hashed({'job':job,'body':req})
            claim=out/f'claim_{number:04d}.json';result_path=out/f'result_{number:04d}.json'
            if result_path.exists():
                previous=read(result_path)
                if not claim.exists()or previous['request_sha256']!=identity_sha:raise ValueError('Prior request differs')
                if fatal(previous):raise ValueError('Prior failed/unresolved attempt blocks resume; preserve it, no retry')
                continue
            if claim.exists():raise ValueError('Prior interrupted request unresolved; preserve claim, no retry')
            provision=Decimal(config['reservation_per_request_usd'])
            if reserved+provision>Decimal(config['reservation_ceiling_usd']):raise ValueError('Reservation guard reached')
            save(claim,{'job':job,'request':req,'request_sha256':identity_sha,'started_at_utc':api.utc(),'reserved_usd':str(provision),'attempt':1})
            reserved+=provision
            result=decode(job,transport('POST','/responses',key,req),config)
            result.update(request_sha256=identity_sha,finished_at_utc=api.utc(),attempt=1)
            save(result_path,result)
            print(f"{number+1}/{len(jobs)} {job['run_id']} {job['id']}: {result['outcome']} cost={result['calculated_cost_usd']}",flush=True)
            if fatal(result):print('STOP: inspect saved status/error/usage; no retries or silent parameter fallback.');break
    finally:guard.unlink()

def export(jobs):
    out=ROOT/'m2_luna/results';out.mkdir(exist_ok=True)
    actual={}
    for path in out.glob('result_*.json'):
        value=read(path);actual[(value['run_id'],value['id'])]=value
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    exports=out/('export_'+stamp);exports.mkdir()
    info={}
    for run in ['M2-P1','M2-P2']:
        values=[]
        for job in jobs:
            if job['run_id']!=run:continue
            result=actual.get((run,job['id']))
            values.append({'id':job['id'],'text':job['text'],'text_sha256':job['text_sha256'],'label':result['label']if result else [],'outcome':result['outcome']if result else 'not_run','structural_rejection':result['structural_rejection']if result else False,'rejections':result['rejections']if result else [],'record_sha256':hashed(result)if result else None})
        with (exports/(run+'.jsonl')).open('x',encoding='utf-8')as f:
            for value in values:f.write(json.dumps(value,ensure_ascii=False)+'\n')
        info[run]={'rows':len(values),'outcomes':dict(Counter(v['outcome']for v in values)),'complete':all(v['outcome']!='not_run'for v in values)}
    records=list(actual.values());known=[Decimal(r['calculated_cost_usd'])for r in records if r['calculated_cost_usd']is not None]
    save(exports/'STATUS.json',{'runs':info,'attempts_including_Dev_canary':len(records),'known_calculated_cost_usd':str(sum(known)),'unknown_cost_attempts':len(records)-len(known),'actual_billing_receipt':None,'measured_request_seconds':sum(r['elapsed_seconds']for r in records),'full_E1_grid_complete':False,'Gold_scores_computed':False,'reference_labels_read':False})
    archive=ROOT/f'E1_M2_return_{stamp}.zip'
    paths=[p for p in out.rglob('*')if p.is_file()and p.name!='.running']+[ROOT/n for n in read(ROOT/'MANIFEST.json')['files']]+[ROOT/'MANIFEST.json']
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p.read_bytes())for p in paths}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED)as z:
        for p in paths:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('RETURN_MANIFEST.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(archive)as z:
        assert z.testzip()is None
        assert all(sha(z.read(n))==h for n,h in hashes.items())
    print('RETURN ZIP: '+str(archive));print('SHA256: '+sha(archive.read_bytes()))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--export',action='store_true');parser.add_argument('--env-key',action='store_true');args=parser.parse_args()
    jobs,prompts,config=prepare()
    if args.run:
        if datetime.datetime.now(datetime.timezone.utc)>=api.EXPIRES:raise ValueError('Reverify dated pricing before launching')
        key=api.key_from_terminal(args.env_key)
        try:execute(jobs,prompts,config,key)
        finally:export(jobs)
    elif args.export:export(jobs)
    else:print('OFFLINE PASS: 1 Dev parameter canary + 2,000 blind Gold jobs; zero API calls.')

if __name__=='__main__':
    try:main()
    except (ValueError,FileExistsError,EOFError,KeyboardInterrupt)as e:print('STOP: '+str(e));raise SystemExit(1)
