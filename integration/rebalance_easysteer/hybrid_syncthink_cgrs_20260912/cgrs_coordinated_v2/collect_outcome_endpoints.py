"""Unsteered, same-path teacher-forced endpoint states and raw max probabilities."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['engineering','full'],required=True);p.add_argument('--engineering-result',type=Path);a=p.parse_args()
    home=Path(__file__).parent;rel=read(home/'release.json')
    for n,h in rel['source_sha256'].items():assert sha(home/n)==h,n
    for n,h in rel['artifact_sha256'].items():assert sha(home/n)==h,n
    assets=rel['assets']
    for n,m in assets['model_files'].items():assert sha(Path(assets['model_path'])/n)==m['sha256']
    if a.phase=='full':
        gate=read(a.engineering_result/'complete.json');assert gate['passed'] and gate['release_sha256']==sha(home/'release.json')
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    a.output.mkdir(parents=True,exist_ok=False)
    def timeout(*_):raise TimeoutError('Fixed1800s collection ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from author_labels import has_lexicon_hit
    started=time.monotonic()
    try:
        torch.manual_seed(42)
        tok=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(assets['model_path'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        boundaries={i for s,i in tok.get_vocab().items() if 'ĊĊ' in s}
        rows=read(home/('engineering.json' if a.phase=='engineering' else 'rows.json'))
        selected=[];capture={}
        def hook(_,inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            capture['h']=h[0,selected].detach().float().cpu().numpy()
        handle=model.model.layers[20].register_forward_hook(hook)
        def forward(row):
            nonlocal selected
            ids=row['token_ids'];prompt=row['prompt_token_ids'];n=len(ids);spans=[];first=0
            for stop in range(n+1):
                if stop<n and ids[stop] not in boundaries:continue
                if stop>first:spans.append((first,stop))
                first=stop+1
            assert spans
            selected=[len(prompt)+x for x,_ in spans]
            x=torch.tensor([prompt+ids],device='cuda')
            torch.cuda.synchronize();began=time.monotonic()
            with torch.inference_mode():
                hidden=model.model(input_ids=x,use_cache=False,return_dict=True).last_hidden_state
                ps=[]
                # Hidden at t-1 predicts the saved token at t; full-vocab maximum,
                # irrespective of which token was forced into the next input.
                for start in range(0,n,256):
                    z=model.lm_head(hidden[0,len(prompt)-1+start:len(prompt)-1+min(start+256,n)]).float()
                    prob=torch.exp(z.amax(-1)-torch.logsumexp(z,-1))
                    if start==0:torch.testing.assert_close(prob[:1],z[:1].softmax(-1).amax(-1),atol=2e-6,rtol=2e-5)
                    ps.append(prob.cpu().numpy())
                pmax=np.concatenate(ps)
            torch.cuda.synchronize();elapsed=time.monotonic()-began
            assert pmax.shape==(n,) and np.isfinite(pmax).all() and (pmax>0).all() and (pmax<=1+1e-5).all()
            features=capture.pop('h');assert features.shape==(len(spans),1536) and np.isfinite(features).all()
            steps=[];previous=None
            for start,stop in spans:
                c=float(pmax[start:stop].mean(dtype=np.float64));v=0. if previous is None else (c-previous)**2/4
                steps.append(dict(start=start,stop=stop,confidence=c,variance=v,lexical_hit=bool(has_lexicon_hit(tok.decode(ids[start:stop])))))
                previous=c
            return features,pmax,steps,elapsed
        manifest=[];seconds=0.
        for index,row in enumerate(rows):
            h,pmax,steps,elapsed=forward(row);seconds+=elapsed
            if a.phase=='engineering':
                h2,p2,s2,t2=forward(row);seconds+=t2
                assert np.array_equal(h,h2) and np.array_equal(pmax,p2) and steps==s2,'Repeat mismatch'
            name=f"{row['question']}_{row['side']}.npz";np.savez_compressed(a.output/name,features=h,max_probabilities=pmax)
            item=dict(question=row['question'],side=row['side'],kind=row['kind'],source=row['source'],problem_sha256=row['problem_sha256'],steps=steps,file=name,sha256=sha(a.output/name))
            manifest.append(item)
            with (a.output/'partial.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(item)+'\n')
            if (index+1)%20==0:print(json.dumps(dict(completed=index+1,rows=len(rows),seconds=seconds)),flush=True)
        handle.remove();save(a.output/'manifest.json',manifest)
        save(a.output/'complete.json',dict(passed=True,phase=a.phase,release_sha256=sha(home/'release.json'),rows=len(rows),forward_seconds=seconds,wall_seconds=time.monotonic()-started,torch=torch.__version__,new_generation_count=0,measurement='Current unsteered HF SDPA reconstruction, not historical or online vLLM probability equivalence'))
    except BaseException as e:
        save(a.output/'failure.json',dict(error=repr(e),wall_seconds=time.monotonic()-started));raise

if __name__=='__main__':main()
