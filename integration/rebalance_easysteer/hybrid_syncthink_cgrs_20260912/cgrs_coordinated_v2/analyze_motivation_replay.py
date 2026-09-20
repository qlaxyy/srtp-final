"""Fixed-pair descriptive statistics of current-runtime forced replay."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    vocab=json.loads(a.tokenizer.read_text(encoding='utf8'))['model']['vocab']
    boundaries={i for s,i in vocab.items() if 'ĊĊ' in s};groups={};parents=[]
    for name in ['U','R']:
        rows=[json.loads(s) for s in (a.input/(name+'_partial.jsonl')).read_text().splitlines() if 'logmax' in json.loads(s)]
        assert len(rows)==286
        groups[name]={}
        for r in rows:
            ids=r['token_ids'];lp=np.asarray(r['logmax'],dtype=float)
            assert len(ids)==len(lp) and np.isfinite(lp).all() and (lp<=1e-5).all()
            stop=ids.index(151649);first=0;steps=[]
            for j in range(stop+1):
                if j<stop and ids[j] not in boundaries:continue
                if j>first:
                    segment=lp[first:j];steps.append((float(np.exp(segment.mean())),float(np.exp(segment).mean())))
                first=j+1
            assert steps
            item=dict(group=name,dataset_index=r['dataset_index'],pair_kind=r['pair_kind'],
                problem_sha256=r['problem_sha256'],thinking_tokens=stop,steps=len(steps),
                token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),modes={})
            for col,mode in enumerate(['geometric','arithmetic']):
                c=np.array(steps)[:,col];local=np.r_[0,np.diff(c)**2/4]
                item['modes'][mode]=dict(mean=float(c.mean()),minimum=float(c.min()),variance=float(c.var()),local_w2=float(local.mean()),sum_confidence=float(c.sum()))
            groups[name][r['dataset_index']]=item;parents.append(item)
    results={}
    for kind,n in [('over_proxy',275),('under_proxy',11)]:
        indices=sorted(i for i,r in groups['U'].items() if r['pair_kind']==kind);assert len(indices)==n
        for i in indices:assert groups['U'][i]['problem_sha256']==groups['R'][i]['problem_sha256']
        normal,proxy=('R','U') if kind=='over_proxy' else ('U','R')
        result=dict(n=n,normal=normal,proxy=proxy,modes={},length={})
        result['length']={g:float(np.mean([groups[g][i]['thinking_tokens'] for i in indices])) for g in ['U','R']}
        for mode in ['geometric','arithmetic']:
            metrics={}
            for metric in ['mean','minimum','variance','local_w2']:
                normal_values=np.array([groups[normal][i]['modes'][mode][metric] for i in indices])
                proxy_values=np.array([groups[proxy][i]['modes'][mode][metric] for i in indices])
                d=proxy_values-normal_values;rng=np.random.default_rng(20260918)
                boot=d[rng.integers(0,n,(20000,n))].mean(1)
                metrics[metric]=dict(normal=float(normal_values.mean()),proxy=float(proxy_values.mean()),
                    delta_proxy_minus_normal=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                    positive=int((d>0).sum()),negative=int((d<0).sum()),zero=int((d==0).sum()))
            metrics['pooled_step_confidence']={g:sum(groups[g][i]['modes'][mode]['sum_confidence'] for i in indices)/sum(groups[g][i]['steps'] for i in indices) for g in ['U','R']}
            result['modes'][mode]=metrics
        results[kind]=result
    output=dict(results=results,interpretation='Current-runtime original-controller reconstruction of frozen texts; historical probabilities unavailable. Outcome/length proxy groups, not semantic ground truth. R confidence is intervention-dependent. Length contrast selected by construction.',
        inputs={n:sha(a.input/n) for n in ['U_partial.jsonl','R_partial.jsonl','complete.json']},
        statistics='Equal-question pairs;20000 paired bootstrap,seed20260918; geometric primary, arithmetic sensitivity; all11 under pairs retained')
    (a.output/'result.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    (a.output/'parents.json').write_text(json.dumps(parents,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(11,7))
    for row,kind in enumerate(['over_proxy','under_proxy']):
        result=results[kind]
        for col,metric in enumerate(['mean','variance','length']):
            if metric=='length':values=[result['length'][result[g]] for g in ['normal','proxy']]
            else:values=[result['modes']['geometric'][metric][g] for g in ['normal','proxy']]
            ax=axs[row,col];bars=ax.bar(['Normal proxy','Over proxy' if row==0 else 'Under proxy'],values,color=['#3065cb','#ec8618'])
            ax.set_title(['Mean step confidence','Across-step variance','Thinking tokens'][col]);ax.set_ylim(0,1 if col==0 else max(values)*1.22)
            ax.bar_label(bars,labels=[f'{v:.4f}' if col<2 else f'{v:.1f}' for v in values],padding=4)
            if col==0:ax.set_ylabel(f'n = {result["n"]} question pairs')
    fig.suptitle('Frozen MATH-500 text replay: 1.5B\nCurrent-runtime reconstruction; not exact original Figure 2 reproduction')
    fig.tight_layout();fig.savefig(a.output/'paired_confidence.png',dpi=170);plt.close(fig)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
