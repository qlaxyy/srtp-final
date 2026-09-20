"""Descriptive direction checks; fixed calibration-question holdout, no GPU."""
import json, hashlib
from pathlib import Path
import numpy as np
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    x=np.load(cache/'layer_21.npy',mmap_mode='r');steps=read(cache/'steps.json')
    audit=read(HERE/'trajectory_length_labels_20260918_run1/result.json')
    lengths=np.array([r['thinking_tokens'] for r in audit['questions']])
    c=np.array([s['confidence'] for s in steps]);q=np.array([s['question'] for s in steps]);h=np.array([s['lexical_hit'] for s in steps],dtype=bool)
    lo,hi=np.quantile(c,[.25,.75]);over=h|(c<lo);under=~h&(c>hi)
    def center(m):return x[m].mean(0,dtype=np.float64)
    def cos(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    old=center(over)-center(under);new=center(over&(lengths[q]>lengths.mean()))-center(under&(lengths[q]<lengths.mean()))
    pure_conf=center(c<lo)-center(c>hi);pure_length=center(lengths[q]>lengths.mean())-center(lengths[q]<lengths.mean())
    rng=np.random.default_rng(42);order=rng.permutation(500);train,test=order[:400],order[400:]
    trainmask=np.isin(q,train);train_mean=lengths[train].mean()
    train_lo,train_hi=np.quantile(c[trainmask],[.25,.75])
    over_train=(h|(c<train_lo))&trainmask;under_train=(~h&(c>train_hi))&trainmask
    old_train=center(over_train)-center(under_train)
    new_train=center(over_train&(lengths[q]>train_mean))-center(under_train&(lengths[q]<train_mean))
    def summarize(v,first_only):
        feats=[];confs=[]
        for i in test:
            ix=np.flatnonzero(q==i)
            if first_only:ix=ix[:4]
            feats.append(x[ix].mean(0,dtype=np.float64));confs.append(c[ix].mean())
        z=np.array(feats)@v/np.linalg.norm(v);conf=np.array(confs);loglen=np.log1p(lengths[test])
        design=np.column_stack([np.ones(len(test)),conf])
        rz=z-design@np.linalg.lstsq(design,z,rcond=None)[0]
        rl=loglen-design@np.linalg.lstsq(design,loglen,rcond=None)[0]
        return dict(correlation_confidence=float(np.corrcoef(z,conf)[0,1]),
            correlation_log_length=float(np.corrcoef(z,loglen)[0,1]),
            partial_correlation_log_length_adjusting_confidence=float(np.corrcoef(rz,rl)[0,1]))
    result=dict(scope='CPU descriptive only, not efficacy; 400/100 split within original calibration, NOT new benchmark questions; no model inference',
        cosines=dict(old_vs_confidence_only=cos(old,pure_conf),new_vs_confidence_only=cos(new,pure_conf),
                     old_vs_length_only=cos(old,pure_length),new_vs_length_only=cos(new,pure_length)),
        calibration_holdout=dict(train_questions=train.tolist(),test_questions=test.tolist(),seed=42,train_mean=float(train_mean),
            train_confidence_quantiles=[float(train_lo),float(train_hi)],
            old_all_steps=summarize(old_train,False),new_all_steps=summarize(new_train,False),
            old_first4_steps=summarize(old_train,True),new_first4_steps=summarize(new_train,True)),
        limitations=['Final GPU vector uses all500; this split only audits the construction recipe.',
            'Original layer selection used calibration data; holdout is not fully independent of layer selection.',
            'Length may represent difficulty, position or looping. Linear confidence adjustment cannot certify semantic overthinking.',
            'No ground-truth labels of excessive or insufficient reasoning are available.'],
        inputs=dict(steps_sha256=sha(cache/'steps.json'),features_sha256=sha(cache/'layer_21.npy'),script_sha256=sha(Path(__file__))))
    save(HERE/'trajectory_length_vector_20260918_run2/geometry_audit.json',result)
    print(json.dumps(dict(cosines=result['cosines'],holdout={k:v for k,v in result['calibration_holdout'].items() if not k.endswith('questions')}),indent=2))

if __name__=='__main__':main()
