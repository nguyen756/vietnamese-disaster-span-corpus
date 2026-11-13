"""Validate and export actual CPU M1 progress; excludes weights/envs/credentials."""
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUTPUT=HERE/'full_cpu_e1_v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p,value):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def package():
    sys.dont_write_bytecode=True
    import full_cpu_driver as driver
    with (OUTPUT/'.process.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        summary=driver.export_all('offline package of actual immutable ledger')
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        before=read(OUTPUT/'evidence/preserved_before.json')
        changed=[name for name,meta in before.items() if not Path(name).is_file() or Path(name).stat().st_size!=meta['bytes'] or sha(name)!=meta['sha256']]
        if changed:raise ValueError('Preserved original file changed; package stopped: '+str(changed))
        validation={'prior_files_verified_unchanged':len(before),'preserved_pilot_gold_generation_allowed':False,
                    'M2_regenerated':False,'source_text_and_unicode_offsets_verified':True,
                    'request_and_response_hashes_verified':True,'strict_compiler_recomputed_equal':True,
                    'completed_gold':{},'created_at_utc':driver.core.utc()}
        for run in ['M1-DEV-P1','M1-P1','M1-P2']:
            data=driver.rows(ROOT/('pilot/dev_inputs_20.jsonl' if run=='M1-DEV-P1' else 'blind/gold_inputs.jsonl'),20 if run=='M1-DEV-P1' else 1000)
            completed=0
            for i,row in enumerate(data):
                p=OUTPUT/f'ledger/{run}/result_{i:04d}.json';c=p.with_name(f'claim_{i:04d}.json')
                if not p.exists():continue
                r=read(p)
                if r.get('outcome') not in driver.GOOD:continue
                claim=read(c);identity={k:v for k,v in claim.items() if k not in ['request_sha256','started_at_utc','session_id','attempt']}
                assert driver.core.digest(identity)==r['request_sha256']==claim['request_sha256']
                assert identity['source']==row and r['id']==row['id'] and r['text']==row['text'] and r['text_sha256']==row['text_sha256']
                assert sha(c)==r['request_record_sha256']
                assert driver.core.digest(r['generation'])==r['generation_sha256']
                for k,v in r['generation'].items():assert r[k]==v
                compiled=driver.core.classify(row['text'],r['response'],r['finish_reason']=='length')
                assert all(r[k]==v for k,v in compiled.items())
                assert len(r['output_token_ids'])==r['output_tokens'] and r['output_tokens']<=1024
                assert r['finish_reason'] in ['eos','length']
                assert all(0<=a<b<=len(row['text']) for a,b,_ in r['label'])
                completed+=1
            validation['completed_gold' if run!='M1-DEV-P1' else 'completed_development']=dict(validation.get('completed_gold' if run!='M1-DEV-P1' else 'completed_development',{}),**{run:completed})
        full=summary['complete'] and summary['Dev_P1']['complete'] and (OUTPUT/'RUNTIME_LOCK.json').is_file()
        validation['complete']=full
        validation_path=OUTPUT/f'evidence/final_validation_{stamp}.json';save(validation_path,validation)
        # Optional offline raw token decoding: no inference, no weights loaded.
        os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(driver.core.ASSETS/'tokenizer',local_files_only=True,trust_remote_code=False)
        for p in (OUTPUT/'ledger').glob('*/result_*.json'):
            r=read(p)
            if r.get('outcome') in driver.GOOD:
                assert tokenizer.decode(r['output_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['response']
        unresolved={'status':'complete CPU M1 generation; not reference-scored' if full else 'partial/stopped CPU M1; incomplete Gold outputs',
                    'complete_gold_generation':full,'original_BF16_GPU_experiment_executed':False,
                    'CPU_quantization_amendment':True,'local_inference_cost_usd':None,
                    'cached_input_tokens':None,'reference_labels_read':False,'Gold_scores_computed':False,
                    'checks':[{'item':'original_BF16_GPU_protocol','status':'amended','detail':'Actual authorized and tested CPU Q4_K_M configuration was used.'},
                              {'item':'local_monetary_cost','status':'unknown','detail':'No electricity/invoice measurement was supplied; no zero dollar cost is invented.'},
                              {'item':'cached_input_tokens','status':'not_instrumented','detail':'Full logical input and output token counts are retained; backend prefix cache use is not measured.'},
                              {'item':'document_source_mapping','status':'not_supplied','detail':'Blind inputs contain ID/text/hash only; source/document identifiers and original source row indices are not guessed.'},
                              {'item':'reference_accuracy_and_E2','status':'not_executed','detail':'No Gold labels, scoring, voting/gating, human records, IAA or paper changes are created here.'}],
                    'run_status':summary['runs'],'creator_names':[],
                    'code_weights':'private; licenses undecided; upstream Qwen license applies',
                    'annotations_documentation_license':'CC BY-NC 4.0'}
        if not full:unresolved['checks'].append({'item':'Gold_generation','status':'incomplete','detail':'Unrun, unresolved and failed rows remain explicit; no automatic retry.'})
        unresolved_path=OUTPUT/f'evidence/unresolved_checks_{stamp}.json';save(unresolved_path,unresolved)
        target=ROOT/('E1_M1_return_20261113.zip' if full else f'E1_M1_partial_return_{stamp}.zip')
        if target.exists():raise FileExistsError('Preserve existing return archive: '+str(target))
        payload={}
        # Include only allowlisted source/evidence paths, never environments/weights.
        for path in [*driver.CODE,'pilot_cpu.py','test_pilot_cpu.py','launch_cpu_pilot.py','CPU_PILOT_AMENDMENT.md','m1_reporting.py','run_m1_cpu.py','test_m1_main.py','CPU_MAIN_PROTOCOL.md']:
            payload['implementation/'+path]=HERE/path
        for base,alias in [(ROOT/'compiler','compiler'),(ROOT/'prompts','prompts'),(ROOT/'pilot','pilot'),
                           (HERE/'pilot_cpu_20_P2','development/reused_P2'),
                           (OUTPUT/'ledger','ledger'),(OUTPUT/'evidence','evidence'),(OUTPUT/'logs','logs')]:
            for p in base.rglob('*'):
                if p.is_file() and '__pycache__'not in p.parts and p.suffix!='.pyc' and p.name!='.process.lock' and not (alias=='logs' and p.name.startswith('package_') and p.suffix=='.log'):
                    payload[alias+'/'+str(p.relative_to(base))]=p
        payload['blind/gold_inputs.jsonl']=ROOT/'blind/gold_inputs.jsonl'
        payload['INPUT_MANIFEST.json']=ROOT/'MANIFEST.json'
        payload['PROTOCOL_AMENDMENT.md']=ROOT/'PROTOCOL_AMENDMENT.md'
        payload['CPU_FULL_AMENDMENT.md']=HERE/'CPU_FULL_AMENDMENT.md'
        payload['unresolved_checks.json']=unresolved_path
        payload['VALIDATION.json']=validation_path
        payload['STATUS.json']=OUTPUT/'STATUS.json'
        for run in ['M1-P1','M1-P2']:payload[f'predictions/{run}.jsonl']=OUTPUT/f'predictions/{run}.jsonl'
        for name in ['DEV_P1_RUNTIME_LOCK.json','RUNTIME_LOCK.json']:
            p=OUTPUT/name
            if p.exists():payload[name]=p
        for p in OUTPUT.glob('session_*.json'):payload['sessions/'+p.name]=p
        for name in ['model_source.json','tokenizer_source.json','requirements-installed.txt']:
            payload['asset_provenance/'+name]=driver.core.ASSETS/name
        for p in (driver.core.ASSETS/'tokenizer').iterdir():
            if p.is_file():payload['tokenizer/'+p.name]=p
        assert not any(p.suffix.lower() in {'.gguf','.safetensors','.bin','.so'} for p in payload.values())
        manifest={'complete':full,'files':{n:{'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in sorted(payload.items())}}
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for n,p in sorted(payload.items()):z.write(p,n)
            z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
            assert set(z.namelist())==set(payload)|{'RETURN_MANIFEST.json'}
            for n,meta in manifest['files'].items():
                b=z.read(n);assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256']
        receipt={'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'CRC_passed':True,
                 'manifest_files_verified':len(payload),'complete':full,'weights_environments_credentials_excluded':True}
        save(target.with_suffix('.verification.json'),receipt)
        with Path(str(target)+'.sha256').open('x') as f:f.write(receipt['sha256']+'  '+target.name+'\n')
        print(json.dumps(receipt,indent=2),flush=True)
        return receipt


if __name__=='__main__':package()
