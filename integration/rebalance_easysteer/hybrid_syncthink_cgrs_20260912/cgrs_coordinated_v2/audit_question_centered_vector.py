"""Frozen-label question fixed-effect decomposition, CPU only.

Centers use only the union of the two original classes, not unused steps.
No semantic purity or compression is inferred from cross-question stability.
"""
import hashlib,json,tarfile
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha
from prepare_process_review import norm

def cosine(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

def main():
    out=HERE/'question_centered_vector_20260918_cpu';out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.pt binary -text\n*.npz binary -text\n',encoding='utf8')
    save(out/'protocol.json',dict(
        hypothesis='Remove additive between-question offsets from original OR-label mean difference. Not semantic relabeling, not equal-parent prototypes, not confidence residualization.',
        formula='m_q=(nO_q*muO_q+nU_q*muU_q)/(nO_q+nU_q); within=mean_O(h-m_q)-mean_U(h-m_q); old=within+between. Equivalently within=sum_q[nO_q*nU_q/(nO_q+nU_q)*(1/NO+1/NU)*(muO_q-muU_q)].',
        inclusion='All original over/under rows; unused steps excluded from centering; single-class parents cancel within exactly, not synthetically imputed.',
        validation='Five parent-disjoint folds: sort SHA256(qcenter-v1:normalized_problem_hash), round-robin. Recompute on train; test only alignment with held-out class differences. No benchmark files, threshold tuning or seed search.',
        admission='CPU candidate only if exact decomposition and finite vectors pass, effective parent count>=30, and all five train/heldout within-direction cosines>0. This is an engineering/stability screen, not statistical power or GPU efficacy authorization.',
        export='If passed: match frozen original norm; retain output layer20, raw OR labels, controller fit, L27 suppression and all generation settings. No runtime changes or generation in this script.',
        limitations='Removes additive question mean only. Label semantics, within-question position/confidence/content and difficulty-selection confounding remain. Fixed effects do not identify causal overthinking.',gpu_calls=0))
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer';frozen=ROOT/'.codex_work/auto_code_v2_500_20260908'
    assert sha(cache/'steps.json')=='fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93'
    assert sha(cache/'layer_21.npy')=='66fdc33b5ca4d40d76243c1ff062dabdecbe542bcc02580ee0cb1ae2ea31ba66'
    steps=read(cache/'steps.json');x=np.load(cache/'layer_21.npy',mmap_mode='r')
    c=np.array([r['confidence'] for r in steps]);lex=np.array([r['lexical_hit'] for r in steps],bool);q=np.array([r['question'] for r in steps]);low,high=read(frozen/'protocol.json')['confidence_quantiles']
    over=lex|(c<low);under=(~lex)&(c>high);assert not (over&under).any()
    ns=np.zeros((500,2),int);sums=np.zeros((500,2,1536),np.float64)
    for j in range(500):
        for k,mask in enumerate([over,under]):
            ix=(q==j)&mask;ns[j,k]=ix.sum();sums[j,k]=x[ix].sum(axis=0,dtype=np.float64)
    def decompose(js):
        n=ns[js];z=sums[js];N=n.sum(0);assert (N>0).all()
        means=np.divide(z,n[:,:,None],out=np.zeros_like(z),where=n[:,:,None]>0)
        pooled=np.divide(z.sum(1),n.sum(1)[:,None],out=np.zeros_like(z[:,0]),where=n.sum(1)[:,None]>0)
        old=z[:,0].sum(0)/N[0]-z[:,1].sum(0)/N[1]
        between=((n[:,0]/N[0]-n[:,1]/N[1])[:,None]*pooled).sum(0)
        within=((z[:,0]-n[:,0,None]*pooled)/N[0]-(z[:,1]-n[:,1,None]*pooled)/N[1]).sum(0)
        w=np.divide(n[:,0]*n[:,1],n.sum(1),out=np.zeros(len(js)),where=n.sum(1)>0)*(1/N[0]+1/N[1])
        analytic=(w[:,None]*(means[:,0]-means[:,1])).sum(0)
        assert np.allclose(old,within+between,atol=1e-10,rtol=1e-10)
        assert np.allclose(within,analytic,atol=1e-10,rtol=1e-10)
        return old,within,between,w,means[:,0]-means[:,1]
    old,within,between,w,diffs=decompose(np.arange(500))
    original=torch.load(frozen/'auto_vector.pt',map_location='cpu',weights_only=True)
    assert torch.equal(torch.from_numpy(old.astype(np.float32)),original)
    with tarfile.open(ROOT/'.codex_work/own_calibration_500_20260908/calibration_assets.tar.gz') as t:raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()=='4d0194b30bb97d1ef0779b13d24ba47ba0aa7ccf38803e6a68f5c233ceb756a3'
    gs=[json.loads(l) for l in raw.splitlines()]
    hashes=[hashlib.sha256(norm(g['problem']).encode()).hexdigest() for g in gs]
    order=sorted(range(500),key=lambda j:hashlib.sha256(('qcenter-v1:'+hashes[j]).encode()).hexdigest())
    folds=np.empty(500,int)
    for rank,j in enumerate(order):folds[j]=rank%5
    validation=[]
    for f in range(5):
        tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
        otr,vtr,_,_,_=decompose(tr);_,vte,_,wt,dt=decompose(te);active=wt>0
        validation.append(dict(fold=f,parents=len(te),both_class_parents=int(active.sum()),
            train_test_within_cosine=cosine(vtr,vte),train_original_test_within_cosine=cosine(otr,vte),
            within_positive_parent_fraction=float(np.mean((dt@vtr)[active]>0)),original_positive_parent_fraction=float(np.mean((dt@otr)[active]>0))))
    neff=float(w.sum()**2/(w@w));ok=neff>=30 and all(r['train_test_within_cosine']>0 for r in validation)
    assert np.isfinite(within).all() and np.linalg.norm(within)>0
    if ok:
        candidate=torch.from_numpy(within.astype(np.float32));candidate*= (original.double().norm()/candidate.double().norm()).to(candidate.dtype)
        assert abs(float(candidate.double().norm()/original.double().norm())-1)<1e-6
        torch.save(candidate,out/'norm_vector.pt')
    np.savez(out/'decomposition.npz',old=old,within=within,between=between,parent_weights=w,parent_class_differences=diffs,counts=ns,folds=folds)
    save(out/'registry.json',[dict(question=j,train_index=gs[j]['train_index'],problem_sha256=hashes[j],fold=int(folds[j]),over=int(ns[j,0]),under=int(ns[j,1]),within_weight=float(w[j])) for j in range(500)])
    result=dict(over=int(over.sum()),under=int(under.sum()),both_class_parents=int((w>0).sum()),effective_parents=neff,
        old_norm=float(np.linalg.norm(old)),within_norm=float(np.linalg.norm(within)),between_norm=float(np.linalg.norm(between)),
        old_within_cosine=cosine(old,within),within_between_cosine=cosine(within,between),
        between_projection_fraction_on_old=float(between@old/(old@old)),
        note='Projection fraction is an exact directional decomposition, not variance explained, semantic contamination percentage, or expected compression.',
        folds=validation,cpu_screen_pass=ok,gpu_run_started=False,
        original_vector_exact=True,decomposition_verified=True,
        candidate_sha256=sha(out/'norm_vector.pt') if ok else None,
        input_hashes=dict(steps=sha(cache/'steps.json'),features=sha(cache/'layer_21.npy'),original_vector=sha(frozen/'auto_vector.pt')))
    save(out/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
