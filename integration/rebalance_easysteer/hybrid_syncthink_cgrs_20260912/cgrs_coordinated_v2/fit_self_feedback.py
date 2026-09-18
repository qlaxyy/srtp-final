"""Fit original states using frozen outcome eligibility, with no test outcomes."""
import argparse,json
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    folder=HERE/'self_feedback_train500_20260918_run1';labels=read(__import__('pathlib').Path(a.labels))
    protocol=read(folder/'protocol.json');plan=read(folder/'plan.json')
    assert len(labels)==500
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    assert sha(cache/'steps.json')=='fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93'
    assert sha(cache/'layer_21.npy')=='66fdc33b5ca4d40d76243c1ff062dabdecbe542bcc02580ee0cb1ae2ea31ba66'
    steps=read(cache/'steps.json');x=np.load(cache/'layer_21.npy',mmap_mode='r')
    q=np.array([s['question'] for s in steps]);c=np.array([s['confidence'] for s in steps]);lex=np.array([s['lexical_hit'] for s in steps],bool)
    frozen=ROOT/'.codex_work/auto_code_v2_500_20260908';low,high=read(frozen/'protocol.json')['confidence_quantiles']
    over=lex|(c<low);under=~lex&(c>high)
    center=lambda mask:x[mask].mean(0,dtype=np.float64)
    original=torch.load(frozen/'auto_vector.pt',map_location='cpu',weights_only=True)
    assert torch.equal(torch.from_numpy((center(over)-center(under)).astype(np.float32)),original)
    chosen=[]
    for i,(r,row) in enumerate(zip(labels,plan['rows'])):
        assert r['dataset_index']==i and r['problem_sha256']==row['problem_sha256']
        u,b=r['U'],r['L27']
        if (u['correct'] and b['correct'] and u['closed'] and b['closed'] and
            b['thinking_tokens']<u['thinking_tokens'] and b['tokens']<u['tokens'] and i not in protocol['benchmark_overlap_excluded_indices']):chosen.append(i)
    mask=np.isin(q,chosen);o=over&mask;u=under&mask
    parents=[len(np.unique(q[m])) for m in [o,u]]
    out=__import__('pathlib').Path(a.output);out.mkdir(exist_ok=False)
    passed=min(parents)>=30 and o.any() and u.any()
    report=dict(eligible_indices=chosen,class_parent_counts=parents,class_step_counts=[int(o.sum()),int(u.sum())],
        support_gate_passed=bool(passed),label_sha256=sha(__import__('pathlib').Path(a.labels)),protocol_sha256=sha(folder/'protocol.json'))
    if passed:
        d=center(o)-center(u);assert np.isfinite(d).all() and np.linalg.norm(d)>0
        v=torch.from_numpy(d.astype(np.float32));v*= (original.double().norm()/v.double().norm()).float()
        torch.save(v,out/'norm_vector.pt')
        report.update(vector_sha256=sha(out/'norm_vector.pt'),cosine=float(np.dot(d,original.numpy())/(np.linalg.norm(d)*original.double().norm().item())),
            status='Training-derived candidate only; no compression efficacy established')
    save(out/'result.json',report);print(json.dumps(report))

if __name__=='__main__':main()
