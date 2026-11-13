"""Offline FP32 inference (fine-tuned weights not distributed); no training, checkpoint selection or upload."""
from pathlib import Path
import argparse,hashlib,json,os,sys,zipfile

HERE=Path(__file__).resolve().parent
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run',choices=[f'{h}_seed{s}' for h in ['bilstm','bilstm_crf'] for s in [42,999,2026]],required=True)
    w=a.add_mutually_exclusive_group(required=True)
    w.add_argument('--weights',type=Path)
    w.add_argument('--weights-archive',type=Path)
    inp=a.add_mutually_exclusive_group(required=True)
    inp.add_argument('--prepared',type=Path,help='Original-format text-only prepared cache; no Java required')
    inp.add_argument('--input',type=Path,help='New JSONL records with text and optional id; requires exact VnCoreNLP assets')
    a.add_argument('--vncorenlp',type=Path)
    a.add_argument('--ordered',action='store_true',help='Keep given batches; required for exact eight-row fixture')
    a.add_argument('--batch-size',type=int,default=8)
    a.add_argument('--threads',type=int,default=6)
    a.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    if args.batch_size<1 or args.threads<1:a.error('Positive batch size and thread count required')
    for key in ['weights','weights_archive','prepared','input','vncorenlp','output']:
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve())
    if args.output.exists():raise FileExistsError('Preserve existing predictions; choose a new output file')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    # Local development-only dependency location; normal installs use requirements.txt.
    devdeps=HERE.parent.parent/'.paper_tools/release_20261111/python'
    if devdeps.exists():sys.path.insert(0,str(devdeps))
    import torch
    from transformers import RobertaConfig,RobertaModel
    from safetensors.torch import load
    from runtime.model import NERModel
    from runtime.batching import collate,bucket_batches
    from runtime.source_units import decode
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True,warn_only=True)
    info=json.loads((HERE/'checkpoint_map.json').read_text(encoding='utf-8'))['models'][args.run]
    if args.weights_archive:
        with zipfile.ZipFile(args.weights_archive) as z:blob=z.read(info['weight_file'])
    else:blob=args.weights.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=info['weight_sha256']:raise ValueError('Weight SHA-256 mismatch')
    state=load(blob);del blob
    config=RobertaConfig.from_json_file(str(HERE/f'configs/{args.run}/encoder_config/config.json'))
    # Match returned training.py build_model exactly: SDPA and no encoder pooler.
    config._attn_implementation='sdpa'
    model=NERModel(RobertaModel(config,add_pooling_layer=False),info['bio_labels'],info['architecture'],hidden=128,dropout=.2)
    model.load_state_dict(state,strict=True);del state
    model.to(args.device).eval()
    if args.prepared:
        prepared=read(args.prepared)
        # Do not permit labels in a supplied cache to influence inference.
        for item in prepared:item['tags']=[]
        rows=[dict(source_row=x['source_row']) for x in prepared]
    else:
        if not args.vncorenlp:a.error('--input requires --vncorenlp; no alternate segmenter fallback')
        assets=json.loads((HERE/'assets/vncorenlp_assets.json').read_text(encoding='utf-8'))['files']
        for name,meta in assets.items():
            if hashlib.sha256((args.vncorenlp/name).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError(f'VnCoreNLP asset mismatch: {name}')
        import py_vncorenlp
        from tokenizers import Tokenizer
        from runtime.alignment import prepare_text
        cwd=Path.cwd()
        try:segmenter=py_vncorenlp.VnCoreNLP(annotators=['wseg'],save_dir=str(args.vncorenlp.resolve()))
        finally:os.chdir(cwd)  # Official wrapper can change process working directory.
        tokenizer=Tokenizer.from_file(str(HERE/'assets/tokenizer/tokenizer.json'))
        rows=read(args.input)
        for i,r in enumerate(rows):r.setdefault('source_row',r.get('id',i))
        prepared=[prepare_text(r,segmenter,tokenizer) for r in rows]
    if not prepared:raise ValueError('Empty input')
    batches=([list(range(i,min(i+args.batch_size,len(prepared)))) for i in range(0,len(prepared),args.batch_size)]
             if args.ordered else bucket_batches(prepared,args.batch_size,0))
    result=[None]*len(prepared);labels=info['bio_labels']
    with torch.inference_mode():
        for indices in batches:
            batch=collate([prepared[i] for i in indices],{t:i for i,t in enumerate(labels)})
            batch={k:v.to(args.device) if isinstance(v,torch.Tensor) and k!='lengths' else v for k,v in batch.items()}
            seqs=model.tag_sequences(model(batch),batch)
            for index,sequence in zip(indices,seqs):
                tags=[labels[t] for t in sequence]
                result[index]=dict(source_row=rows[index]['source_row'],predicted_label=decode(prepared[index]['units'],tags),raw_bio_tags=tags)
                if 'id' in rows[index]:result[index]['id']=rows[index]['id']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    report=dict(run=args.run,rows=len(result),device=args.device,threads=args.threads,batch_size=args.batch_size,ordered=args.ordered,
                weight_sha256=info['weight_sha256'],training=False,upload=False,torch=torch.__version__)
    args.output.with_suffix('.run.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
