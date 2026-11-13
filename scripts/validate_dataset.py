"""Validate the stand-off release without obtaining source text or network access."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json

def rows(p):return [json.loads(s)for s in Path(p).read_text(encoding='utf-8-sig').splitlines()if s.strip()]
def validate(dataset,verify_manifest=True):
    dataset=Path(dataset);labels=json.loads((dataset/'schema.json').read_text(encoding='utf-8'))['labels']
    ids=set();core_docs={};core_hashes={};statistics={}
    for split,path,metadata in [('train','data/train.jsonl','metadata/documents.jsonl'),('validation','data/validation.jsonl','metadata/documents.jsonl'),('test','data/test.jsonl','metadata/documents.jsonl'),('e6_test','additional/e6_test.jsonl','additional/e6_documents.jsonl')]:
        documents={r['document_id']:r for r in rows(dataset/metadata)}
        data=rows(dataset/path);hashes=set();docs=set();spans=0
        for row in data:
            assert not set(row)&{'text','sentence','body','title','sentence_text'},'Source text included'
            assert row['id']not in ids,'Duplicate ID';ids.add(row['id'])
            assert len(row['sentence_sha256'])==64
            assert type(row['sentence_length'])is int and row['sentence_length']>0
            assert row['document_id']in documents
            assert 0<=row['source_start']<row['source_end']<=documents[row['document_id']]['source_text_length']
            assert row['reconstruction']in ['exact','collapse_whitespace']
            last=0;seen=set()
            for entity in sorted(row['entities'],key=lambda e:(e['start'],e['end'],e['label'])):
                a,b,c=entity['start'],entity['end'],entity['label']
                assert type(a)is int and type(b)is int and 0<=last<=a<b<=row['sentence_length']
                assert c in labels and (a,b,c)not in seen
                last=b;seen.add((a,b,c));spans+=1
            hashes.add(row['sentence_sha256']);docs.add(row['document_id'])
        statistics[split]={'rows':len(data),'spans':spans,'documents':len(docs)}
        if split!='e6_test':core_docs[split]=docs;core_hashes[split]=hashes
    assert statistics=={'train':{'rows':9686,'spans':42439,'documents':526},'validation':{'rows':803,'spans':5229,'documents':19},'test':{'rows':1000,'spans':5026,'documents':40},'e6_test':{'rows':300,'spans':1270,'documents':28}},statistics
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not core_docs[a]&core_docs[b]and not core_hashes[a]&core_hashes[b]
    for filename,split,count in [('A1_original.jsonl','test',1000),('A2_original.jsonl','test',1000),('e4_consistency_reference.jsonl','train',300)]:
        base={r['id']:r for r in rows(dataset/f'data/{split}.jsonl')};ref=rows(dataset/'additional'/filename)
        assert len(ref)==count
        for row in ref:
            assert row['id']in base and row['document_id']==base[row['id']]['document_id']
            assert row['sentence_sha256']==base[row['id']]['sentence_sha256']
            assert not set(row)&{'text','body','sentence','title'}
            assert all(0<=e['start']<e['end']<=row['sentence_length']and e['label']in labels for e in row['entities'])
    verified=0
    if verify_manifest:
        manifest=json.loads((dataset/'MANIFEST.json').read_text(encoding='utf-8'))
        for name,value in manifest.items():
            path=(dataset/name).resolve();assert path.is_relative_to(dataset.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==value['sha256'];verified+=1
    return {'passed':True,'splits':statistics,'core_rows':11489,'core_spans':52694,'additional_E6_rows':300,'source_text_included':False,'manifest_files_verified':verified,'semantic_accuracy_verified':False,'live_source_recovery_verified':False}

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--dataset',type=Path,required=True);a.add_argument('--skip-manifest',action='store_true');args=a.parse_args()
    print(json.dumps(validate(args.dataset,not args.skip_manifest),indent=2))
