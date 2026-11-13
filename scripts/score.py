"""Strict/overlap entity scores with explicit IDs and maximum-intersection matching."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'llm_study/implementation'))
from analysis import general_score
def read(p):return [json.loads(s)for s in Path(p).read_text(encoding='utf-8-sig').splitlines()if s.strip()]
def spans(row):
    for name in ['predicted_label','label','entities']:
        if name in row:
            return [[e['start'],e['end'],e['label']]if isinstance(e,dict)else e for e in row[name]]
    raise ValueError('Missing span list')
def evaluate(reference,predictions):
    if len({r['id']for r in reference})!=len(reference):raise ValueError('Duplicate reference ID')
    found={r['id']:r for r in predictions}
    if len(found)!=len(predictions)or set(found)!={r['id']for r in reference}:raise ValueError('Require exact reference/prediction IDs')
    converted=[];output=[]
    from common import LABELS
    for row in reference:
        pred=found[row['id']]
        if 'text'in row and 'text'in pred and row['text']!=pred['text']:raise ValueError('Changed text')
        rh=row.get('sentence_sha256',row.get('text_sha256'));ph=pred.get('sentence_sha256',pred.get('text_sha256'))
        if rh and ph and rh!=ph:raise ValueError('Changed sentence hash')
        if rh and 'text'in pred and hashlib.sha256(pred['text'].encode()).hexdigest()!=rh:raise ValueError('Text/hash mismatch')
        truth=spans(row);guess=spans(pred)
        length=row.get('sentence_length',len(row['text'])if 'text'in row else None)
        for values in [truth,guess]:
            if len(set(map(tuple,values)))!=len(values):raise ValueError('Duplicate spans')
            for start,end,label in values:
                if type(start)is not int or type(end)is not int or not 0<=start<end or (length is not None and end>length)or label not in LABELS:raise ValueError('Invalid span')
        converted.append(dict(row,label=truth));output.append(guess)
    return {mode:general_score(converted,output,mode)for mode in ['strict','overlap']}
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--reference',type=Path,required=True);a.add_argument('--predictions',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    result=evaluate(read(args.reference),read(args.predictions));args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v['micro']for k,v in result.items()},indent=2))
