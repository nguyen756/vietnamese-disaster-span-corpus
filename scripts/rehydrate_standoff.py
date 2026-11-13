"""Exact text recovery helper; source snapshots are not distributed."""
from pathlib import Path
import argparse,hashlib,json,os

def read(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--dataset',type=Path,required=True)
    a.add_argument('--documents',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--report',type=Path,required=True)
    a.add_argument('--extra',action='store_true',help='Recover the additional E6 configuration')
    args=a.parse_args()
    assert not args.output.resolve().is_relative_to(args.dataset.resolve()),'Never write recovered text inside public payload'
    docs={r['document_id']:r for r in read(args.documents)}
    metadata='additional/e6_documents.jsonl' if args.extra else 'metadata/documents.jsonl'
    spec={r['document_id']:r for r in read(args.dataset/metadata)}
    labels=json.loads((args.dataset/'schema.json').read_text(encoding='utf-8'))['labels']
    result={};failures=[];good_docs=set();rules={}
    for split in (['e6_test'] if args.extra else ['train','validation','test']):
        output=[]
        folder='additional' if args.extra else 'data'
        for r in read(args.dataset/f'{folder}/{split}.jsonl'):
            try:
                d=docs[r['document_id']];m=spec[r['document_id']];body=d[m['source_text_field']]
                assert digest(body)==m['source_text_sha256'] and len(body)==m['source_text_length'],'document hash/length mismatch'
                fragment=body[r['source_start']:r['source_end']]
                assert digest(fragment)==r['source_fragment_sha256'],'fragment hash mismatch'
                assert r['reconstruction'] in ['exact','collapse_whitespace'],'unknown reconstruction rule'
                text=fragment if r['reconstruction']=='exact' else ' '.join(fragment.split())
                assert digest(text)==r['sentence_sha256'] and len(text)==r['sentence_length'],'sentence hash/length mismatch'
                last=0;seen=set()
                for e in sorted(r['entities'],key=lambda e:e['start']):
                    t=(e['start'],e['end'],e['label'])
                    assert type(t[0])==int and type(t[1])==int and 0<=last<=t[0]<t[1]<=len(text) and t[2] in labels and t not in seen,'invalid span'
                    last=t[1];seen.add(t)
                output.append({k:r[k] for k in ['id','document_id','entities','source_url','source_name','annotation_method']}|{'text':text})
                good_docs.add(r['document_id']);rules[r['reconstruction']]=rules.get(r['reconstruction'],0)+1
            except (KeyError,AssertionError,TypeError) as e:
                failures.append(dict(split=split,id=r['id'],document_id=r['document_id'],reason=str(e)))
        result[split]=output
    report=dict(complete=not failures,splits={s:dict(sentences=len(rr),spans=sum(len(r['entities']) for r in rr)) for s,rr in result.items()},
        reconstructed_documents=len(good_docs),reconstruction_rules=rules,failures=failures,
        raw_source_text_redistributed=False,live_recovery_test=False)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if failures:
        raise SystemExit(f'{len(failures)} reconstruction failures. No recovered split files written; see report.')
    args.output.mkdir(parents=True,exist_ok=True)
    for split,rr in result.items():
        path=args.output/f'{split}.jsonl'
        temporary=path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr),encoding='utf-8')
        os.replace(temporary,path)
    print(json.dumps({k:v for k,v in report.items() if k!='failures'},indent=2))
if __name__=='__main__':main()
