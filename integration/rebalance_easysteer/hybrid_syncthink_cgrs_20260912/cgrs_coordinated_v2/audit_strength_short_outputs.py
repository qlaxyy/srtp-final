"""Read-only diagnosis; cannot attribute unchanged draws without eligible logits."""
import math
from pathlib import Path
import torch
from engineering import HERE,read,save,sha
from calibrate_penalty_scale_cpu import vocabulary
from policy import TRIGGERS


def main():
    torch.set_num_threads(1)
    out=HERE/'strength_short_audit_20260916';out.mkdir(exist_ok=False)
    root=Path('E:/srtp/B-strength-engineering_0916');manifest=read(root/'manifest.json');inputs={}
    arms={}
    for name in ('R','RC14default','RCscaled','RCconstant'):
        p=root/'completed'/name/'result.json';assert sha(p)==manifest[p.relative_to(root).as_posix()]
        inputs[str(p)]=sha(p);arms[name]=read(p)
    tokenizer=Path('E:/srtp/srtp-final/.codex_work/label_audit_30_20260910/tokenizer.json')
    pieces,_=vocabulary(tokenizer);inputs[str(tokenizer)]=sha(tokenizer);cases=[]
    for a,b,c,d in zip(*(arms[k]['records'] for k in ('R','RC14default','RCscaled','RCconstant'))):
        assert a['problem_sha256']==b['problem_sha256']==c['problem_sha256']==d['problem_sha256']
        assert a['token_ids']==c['token_ids']==d['token_ids']
        if a['token_ids']==b['token_ids']:continue
        i=next(i for i,(x,y) in enumerate(zip(a['token_ids'],b['token_ids'])) if x!=y)
        snippet=lambda ids,lo,hi:''.join(pieces[t] for t in ids[lo:hi])
        cases.append(dict(train_index=a['train_index'],problem_sha256=a['problem_sha256'],first_difference=i,
            R_token=a['token_ids'][i],RC14_token=b['token_ids'][i],R_is_trigger=a['token_ids'][i] in TRIGGERS,
            prefix=snippet(a['token_ids'],max(0,i-100),i),R_continuation=snippet(a['token_ids'],i,i+100),
            RC14_continuation=snippet(b['token_ids'],i,i+100)))
    old=Path('E:/srtp/B-same-logits8_0916');manifest=read(old/'manifest.json');captures=[]
    for p in sorted((old/'results').glob('capture_*.pt')):
        assert sha(p)==manifest[p.relative_to(old).as_posix()];inputs[str(p)]=sha(p)
        blob=torch.load(p,map_location='cpu',weights_only=True);idx=blob['slots']
        mask=blob['owner']['opening'][idx]&blob['owner']['thinking'][idx]&torch.isfinite(blob['prev_step_mean'][idx])&torch.isfinite(blob['coefs'][idx])&(blob['coefs'][idx]<0)&torch.tensor(blob['valid'])
        captures.append(dict(file=p.name,rows=len(idx),eligible_rows=int(mask.sum())))
    scale=read(HERE/'penalty_scale_native_cpu_v2_20260916/audit.json')['results']['1.5B']['constant_scale']
    save(out/'audit.json',dict(input_sha256=inputs,source_sha256=sha(Path(__file__),True),cases=cases,captures=captures,
        pairwise_trigger_vs_nontrigger_odds_multiplier_before_top_p=dict(temperature=.7,fixed=math.exp(-math.log(2)/.7),weak_constant=math.exp(-math.log(2)*scale/.7)),
        manual_case_notes={'1441':'Wait continuation repeats the same expansion already present; RC14 proceeds to combine like terms. Local repetition evidence only.',
                           '1483':'Wait continuation develops y>=2 and a new transformed problem; word alone is not proof of useless reflection. No final correctness claim from partial output.'},
        limitations=['Historical14 capture batches belong to earlier7B diagnostic, not current1.5B run; all have zero eligible rows.',
                     'Current engineering run saved aggregate logit-update counts but no eligible full-logit trace.',
                     'Identical sampled outputs do not imply identical distributions or disabled interventions.',
                     'Pairwise odds identity is before top-p; nucleus filtering and rounding prevent interpreting it as final token-probability multiplier.',
                     'Cannot decide short-horizon versus penalty-strength explanations from these records.'],
        conclusion='No demonstrated adaptive advantage. Proceed only to separately registered full-length four-arm screen, not another identical short engineering rerun.'))
    print('Audited',len(cases),'divergences;',len(captures),'historical snapshots,',sum(r['eligible_rows'] for r in captures),'eligible rows.')


if __name__=='__main__':main()
