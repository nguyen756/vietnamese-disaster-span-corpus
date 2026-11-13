"""Recompute main observed scores from public stand-off labels/predictions, offline."""
from pathlib import Path
import argparse,json,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'llm_study/implementation'))
from score import evaluate
from analysis import gates
def rows(p):return [json.loads(s)for s in p.read_text(encoding='utf-8').splitlines()if s.strip()]
def labels(rr):return [dict(r,label=[[e['start'],e['end'],e['label']]for e in r['entities']])for r in rr]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--oracle',action='store_true');a=p.parse_args()
    gold=rows(a.dataset/'data/test.jsonl');pred=ROOT/'results/predictions';result={};grid={}
    for run in ['M1-P1','M1-P2','M2-P1','M2-P2']:
        grid[run]=rows(pred/(run+'.jsonl'));result[run]=evaluate(gold,grid[run])
    result['original_human_agreement']=evaluate(rows(a.dataset/'additional/A1_original.jsonl'),rows(a.dataset/'additional/A2_original.jsonl'))
    e4=rows(a.dataset/'additional/e4_consistency_reference.jsonl');train={r['id']:r for r in rows(a.dataset/'data/train.jsonl')}
    result['E4_consistency']=evaluate(e4,[train[r['id']]for r in e4])
    result['supervised']={}
    e6=rows(a.dataset/'additional/e6_test.jsonl')
    for head in ['bilstm','bilstm_crf']:
        result['supervised'][head]={}
        for split,ref in [('Gold',gold),('E6',e6)]:
            per_seed={str(seed):evaluate(ref,rows(pred/f'{head}_seed{seed}_{split}.jsonl'))for seed in [42,999,2026]}
            result['supervised'][head][split]={'per_seed':per_seed,
                'strict_mean_F1':statistics.mean(v['strict']['micro']['f1']for v in per_seed.values()),
                'strict_sample_SD':statistics.stdev(v['strict']['micro']['f1']for v in per_seed.values()),
                'overlap_mean_F1':statistics.mean(v['overlap']['micro']['f1']for v in per_seed.values())}
    if a.oracle:
        tokens=json.loads((ROOT/'results/SOURCE_TOKEN_COUNTS.json').read_text(encoding='utf-8'))['sentence_tokens_by_id']
        result['E2_oracle']=gates(labels(gold),grid,tokens)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'LLM_strict_F1_percent':{run:100*result[run]['strict']['micro']['f1']for run in grid},
        'original_human_strict_F1_percent':100*result['original_human_agreement']['strict']['micro']['f1'],
        'E4_consistency_F1_percent':100*result['E4_consistency']['strict']['micro']['f1'],
        'supervised_F1_percent':{h:{s:100*v['strict_mean_F1']for s,v in splits.items()}for h,splits in result['supervised'].items()},'training_or_API_calls':False},indent=2))
if __name__=='__main__':main()
