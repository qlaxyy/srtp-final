"""Replay newly generated text in base model, then reuse unchanged v2 fitting."""
import argparse
import importlib.util
import signal
import sys
import time
from types import SimpleNamespace
from common import HOME, ROOT, read, save, sha, spans, verify

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=__import__('pathlib').Path)
    p.add_argument('--output', required=True, type=__import__('pathlib').Path)
    p.add_argument('--engineering', action='store_true')
    p.add_argument('--engineering-result', type=__import__('pathlib').Path)
    a=p.parse_args();verify()
    data=read(a.source);rows=data['records'];expected=read(HOME/'rows.json')
    assert len(rows)==500
    for r,e in zip(rows,expected):
        assert r['problem_sha256']==e['problem_sha256'] and r['prompt_token_ids']==e['prompt_token_ids']
    if not a.engineering:
        gate=read(a.engineering_result/'complete.json')
        assert gate['passed'] and gate['engineering'] and gate['source_sha256']==sha(a.source)
        assert gate['manifest_sha256']==sha(HOME/'manifest.json')
    import subprocess
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    a.output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    def timeout(*_):raise TimeoutError('Replay time budget exhausted; keep partial data')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600 if a.engineering else 2400)
    try:
        import numpy as np
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        sys.path.insert(0,str(ROOT/'integration/rebalance_easysteer/scripts'))
        import calibrate_auto as cal
        author=cal.load_module('iteration_author',ROOT/'sources/ReBalance/hidden_analysis_auto.py')
        assets=read(HOME/'assets.json');model_path=assets['model_path']
        for n,m in assets['model_files'].items():assert sha(__import__('pathlib').Path(model_path)/n)==m['sha256']
        tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        boundaries={i for text,i in tok.get_vocab().items() if 'ĊĊ' in text}
        assert tok.encode('</think>',add_special_tokens=False)==[151649]
        selected_rows=rows[:4] if a.engineering else rows
        if a.engineering:
            # Explicitly closed synthetic prefixes: engineering only, never fitted.
            selected_rows=[dict(r,token_ids=r['token_ids'][:512]+[151649]) for r in selected_rows]
        layouts=[spans(r['token_ids'],boundaries) for r in selected_rows]
        count=sum(len(s) for s,n in layouts)
        assert count>0
        model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).cuda().eval()
        layers=len(model.model.layers);width=model.config.hidden_size
        import shutil
        assert shutil.disk_usage(a.output).free>layers*count*width*4+1024**3
        maps={i:np.lib.format.open_memmap(a.output/f'layer_{i}.npy',mode='w+',dtype=np.float32,
               shape=(count,width)) for i in range(1,layers+1)}
        selected=[];capture={}
        def hook(i):
            def record(_,inputs,output):
                h=output[0] if isinstance(output,tuple) else output
                capture[i]=h[0,selected].detach().float().cpu().numpy()
            return record
        handles=[block.register_forward_hook(hook(i+1)) for i,block in enumerate(model.model.layers)]
        def replay(row,segments,n):
            nonlocal selected
            prompt=row['prompt_token_ids'];selected=[len(prompt)+start for start,stop in segments]
            with torch.inference_mode():
                result=model.model(torch.tensor([prompt+row['token_ids'][:n]],device='cuda'),
                    use_cache=False,return_dict=True)
                values=[]
                for start in range(0,n,128):
                    z=model.lm_head(result.last_hidden_state[0,len(prompt)-1+start:len(prompt)-1+min(n,start+128)]).float()
                    values.append(torch.exp(z.amax(-1)-torch.logsumexp(z,-1)).cpu().numpy())
                probs=np.concatenate(values) if values else np.empty(0)
            assert np.isfinite(probs).all() and (probs>0).all() and (probs<=1.00001).all()
            return {k:v.copy() for k,v in capture.items()},probs
        steps=[];offset=0
        for q,(row,(segments,n)) in enumerate(zip(selected_rows,layouts)):
            if not segments:continue
            h,probs=replay(row,segments,n)
            if a.engineering:
                h2,p2=replay(row,segments,n)
                assert np.array_equal(probs,p2) and all(np.array_equal(h[k],h2[k]) for k in h)
            for k,v in h.items():
                assert v.shape==(len(segments),width) and np.isfinite(v).all()
                maps[k][offset:offset+len(segments)]=v
            prev=None
            for start,stop in segments:
                c=float(probs[start:stop].mean(dtype=np.float64))
                steps.append(dict(question=q,start=start,stop=stop,confidence=c,
                    variance=0. if prev is None else (c-prev)**2/4,
                    lexical_hit=bool(author.has_lexicon_hit(tok.decode(row['token_ids'][start:stop])))))
                prev=c
            offset+=len(segments)
            with (a.output/'progress.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(question=q,steps=offset,seconds=time.monotonic()-began))+'\n')
        assert offset==count
        for handle in handles:handle.remove()
        for mm in maps.values():mm.flush()
        save(a.output/'steps.json',steps)
        c=np.array([s['confidence'] for s in steps]);v=np.array([s['variance'] for s in steps])
        cl,ch=np.quantile(c,[.25,.75]);vl,vh=np.quantile(v,[.25,.75])
        save(a.output/'protocol.json',dict(source_sha256=sha(a.source),model=model_path,
            confidence_quantiles=[cl,ch],variance_quantiles=[vl,vh],measurement='raw base model teacher forcing'))
        save(a.output/'collection.json',dict(layer_ids=list(maps),feature_dir=str(a.output.resolve()),
            layers=layers,width=width,steps=count,representation='raw decoder output before final norm'))
        del model
        torch.cuda.empty_cache()
        if not a.engineering:
            assert 0<=cl<ch<1 and 0<=vl<vh<=.25
            args=SimpleNamespace(output=a.output,model=__import__('pathlib').Path(model_path))
            cal.select(args);cal.fit(args)
        save(a.output/'complete.json',dict(passed=True,engineering=a.engineering,rows=len(selected_rows),
            steps=count,source_sha256=sha(a.source),manifest_sha256=sha(HOME/'manifest.json'),
            wall_seconds=time.monotonic()-began,new_answers=0,
            forward_count=len([s for s,n in layouts if s])*(2 if a.engineering else 1)))
    except BaseException as e:
        save(a.output/'failure.json',dict(error=repr(e),wall_seconds=time.monotonic()-began));raise

if __name__=='__main__':main()
