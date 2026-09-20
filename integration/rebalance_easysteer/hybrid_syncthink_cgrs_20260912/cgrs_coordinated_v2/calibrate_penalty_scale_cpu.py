"""Replay frozen greedy calibration prefixes, without model execution.

Double-precision analytical R reconstruction is not native GPU state replay.
"""
import hashlib
import json
import math
import tarfile
import unicodedata
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
from engineering import HERE, ROOT, save, sha
from audit_strength_coupling import coefficients
from policy import token_flags, next_opening, eligible, TRIGGERS


def vocabulary(path):
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['decoder']['type']=='ByteLevel'
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256))
    cs=bs.copy();extra=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+extra);extra+=1
    inverse=dict(zip(map(chr,cs),bs))
    raw={i:s for s,i in data['model']['vocab'].items()}
    pieces={i:bytes(inverse[c] for c in s).decode('utf-8',errors='replace') for i,s in raw.items()}
    for item in data['added_tokens']:pieces[item['id']]=item['content']
    boundaries={i for i,s in raw.items() if 'ĊĊ' in s}
    assert all(pieces[i]==s for i,s in TRIGGERS.items())
    return pieces,boundaries


def collect_updates(row,boundaries,native_fp32=False):
    total=0.0;count=0;previous=float('nan');updates=[]
    for pos,(token,logp) in enumerate(zip(row['token_ids'],row['logprobs'])):
        if token==151649:break
        if token not in boundaries:
            total=np.float32(total+np.float32(math.exp(logp))) if native_fp32 else total+math.exp(logp)
            count+=1
        elif count:
            mean=np.float32(total/np.float32(count)) if native_fp32 else total/count
            variance=(mean-previous)**2/4 if math.isfinite(previous) else 0.0
            if native_fp32 and math.isfinite(previous):
                diff=np.float32(mean-previous);variance=np.float32(np.float32(diff*diff)/np.float32(4))
            updates.append((pos,mean,variance));previous=mean;total=0.;count=0
    return updates


def replay(row,pieces,boundaries,updates,coefs,lower_bound,native_fp32=False):
    lookup={u[0]:(u[1],float(c)) for u,c in zip(updates,coefs)}
    opening=False;thinking=151648 in row['prompt_token_ids'];previous=float('nan');coef=-1.
    scales=[];positions=[]
    for pos,token in enumerate(row['token_ids']):
        if eligible(opening,thinking,coef,previous,'negative'):
            ratio=float(np.float32(coef)/np.float32(lower_bound)) if native_fp32 else coef/lower_bound
            scales.append(min(1.,max(0.,ratio)));positions.append(pos)
        if token==151649:break
        thinking=thinking or token==151648
        opening=next_opening(opening,pieces[token],token in boundaries,token) and thinking
        if pos in lookup:previous,coef=lookup[pos]
    return scales,positions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-fp32',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    out=args.output or HERE/('penalty_scale_native_cpu_20260916' if args.native_fp32 else 'penalty_scale_calibration_cpu_20260916');out.mkdir(exist_ok=False)
    root=Path('E:/srtp/srtp-final/.codex_work')
    tokenizer=root/'label_audit_30_20260910/tokenizer.json'
    assert sha(tokenizer)=='88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773'
    pieces,boundaries=vocabulary(tokenizer)
    source=ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py'
    native=None
    if args.native_fp32:
        import torch
        torch.set_num_threads(1)
        spec=importlib.util.spec_from_file_location('frozen_rebalance_cpu',source)
        native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
    inputs={str(tokenizer):sha(tokenizer),str(source):sha(source)};results={};ledger=[]
    for model in ('1.5B','7B'):
        if model=='1.5B':
            fit=root/'auto_code_v2_500_20260908/fit.json'
            archive=root/'own_calibration_500_20260908/calibration_assets.tar.gz'
            with tarfile.open(archive) as t:
                manifest=json.load(t.extractfile('manifest.json'));raw=t.extractfile('generations.jsonl').read()
            inputs[str(archive)]=sha(archive)
        else:
            folder=root/'qwen7b_validation/auto_code_v2_qwen7b_20260908'
            fit=folder/'fit.json';path=folder/'calibration/generations.jsonl'
            manifest=json.loads((folder/'calibration/manifest.json').read_text());raw=path.read_bytes()
            inputs[str(path)]=sha(path)
        f=json.loads(fit.read_text());inputs[str(fit)]=sha(fit)
        assert hashlib.sha256(raw).hexdigest()==f['calibration_source_sha256']
        assert manifest['temperature']==0 and manifest['count']==500
        rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==500
        updates=[collect_updates(r,boundaries,args.native_fp32) for r in rows];flat=[u for us in updates for u in us]
        c=np.array([u[1] for u in flat]);v=np.array([u[2] for u in flat])
        analytical=coefficients(c.astype(np.float64),v.astype(np.float64),f['parameters'],source)
        coefs=analytical
        if native:
            params=native.ReBalanceParams(boundary_token_ids=tuple(sorted(boundaries)),think_start_token_id=151648,think_end_token_id=151649,**f['parameters'])
            coefs=native.compute_rebalance_coefficient(torch.tensor(c,dtype=torch.float32,device='cpu'),torch.tensor(v,dtype=torch.float32,device='cpu'),params).numpy()
        offset=0;allscales=[];covered=0
        for row,us in zip(rows,updates):
            assert len(row['token_ids'])==len(row['logprobs'])
            bound=min(f['parameters']['low_val_1'],f['parameters']['low_val_2'])
            scales,positions=replay(row,pieces,boundaries,us,coefs[offset:offset+len(us)],bound,args.native_fp32);offset+=len(us)
            allscales.extend(scales);covered+=bool(scales)
            ledger.append(dict(model=model,train_index=row['train_index'],problem_sha256=hashlib.sha256(''.join(unicodedata.normalize('NFKC',row['problem']).split()).encode()).hexdigest(),purpose='existing calibration only, not validation',eligible_positions=positions,scale_sum=sum(scales),eligible_count=len(scales)))
        assert offset==len(coefs)
        results[model]=dict(questions=500,covered_questions=covered,eligible_slots=len(allscales),
            constant_scale=float(np.mean(allscales)),constant_penalty=float(np.mean(allscales))*math.log(2),
            scale_quantiles=list(map(float,np.quantile(allscales,[0,.25,.5,.75,1]))),
            lower_bound=bound,fit_sha256=sha(fit),generations_sha256=f['calibration_source_sha256'])
        if native:
            results[model]['native_vs_float64_sign_disagreements']=int(np.sum((coefs<0)!=(analytical<0)))
            results[model]['native_vs_float64_max_abs_difference']=float(np.max(np.abs(coefs-analytical)))
    save(out/'ledger.json',ledger)
    save(out/'audit.json',dict(results=results,input_sha256=inputs,source_sha256=sha(Path(__file__),True),
        scope=('Native CPU Torch '+torch.__version__+' FP32 controller, sequential FP32 accumulation from saved greedy logprobs.' if native else 'Exact original token positions with analytical float64 R coefficients, on unsteered greedy calibration trajectories.'),
        limitations=['Not online steered trajectory exposure or counterfactual token saving.',
                     'Saved logprobs are converted back to probabilities; original GPU softmax values/rounding are not replayed. Tokenizer flags still require installed tokenizer parity.',
                     'All errors and capped calibration outputs retained; no answer-based selection.',
                     'Whitespace opening slots counted individually as actual adapter would; new paths can have different dose.'],gpu_ready=False))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
