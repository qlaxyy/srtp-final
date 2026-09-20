"""Test agreement of the two existing extraction cues, without chain-length filtering."""
import hashlib,io,json,tarfile
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    run='strict_and_vector_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n*.pt binary -text\n',encoding='utf8')
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer';frozen=ROOT/'.codex_work/auto_code_v2_500_20260908'
    assert sha(cache/'steps.json')=='fc1e677fa50b834eec62a4caebbc39d341f93716624d4deca9c427a5fa0c8f93'
    assert sha(cache/'layer_21.npy')=='66fdc33b5ca4d40d76243c1ff062dabdecbe542bcc02580ee0cb1ae2ea31ba66'
    steps=read(cache/'steps.json');x=np.load(cache/'layer_21.npy',mmap_mode='r')
    c=np.array([s['confidence'] for s in steps]);hit=np.array([s['lexical_hit'] for s in steps],bool);q=np.array([s['question'] for s in steps])
    lo,hi=read(frozen/'protocol.json')['confidence_quantiles'];over=hit|(c<lo);under=~hit&(c>hi);strict=hit&(c<lo)
    old=x[over].mean(0,dtype=np.float64)-x[under].mean(0,dtype=np.float64)
    original=torch.load(frozen/'auto_vector.pt',map_location='cpu',weights_only=True)
    assert torch.equal(torch.from_numpy(old.astype(np.float32)),original)
    new=x[strict].mean(0,dtype=np.float64)-x[under].mean(0,dtype=np.float64)
    assert strict.sum()==10636 and under.sum()==11309 and np.isfinite(new).all()
    vector=torch.from_numpy((new*np.linalg.norm(old)/np.linalg.norm(new)).astype(np.float32))
    assert abs(float(vector.double().norm()/original.double().norm())-1)<1e-6
    torch.save(vector,out/'norm_vector.pt')
    def counts(m):
        n=np.bincount(q[m],minlength=500);return dict(steps=int(m.sum()),questions=int((n>0).sum()),largest_question_fraction=float(n.max()/n.sum()))
    audit=dict(old_over=counts(over),new_over=counts(strict),under=counts(under),removed=int((over&~strict).sum()),
        raw_norm=float(np.linalg.norm(new)),deployed_norm=float(vector.double().norm()),original_norm=float(original.double().norm()),
        cosine_old=float(old@new/np.linalg.norm(old)/np.linalg.norm(new)),original_exact_reproduction=True,
        feature_sha256=sha(cache/'layer_21.npy'),steps_sha256=sha(cache/'steps.json'),vector_sha256=sha(out/'norm_vector.pt'),
        limitation='Cue agreement is weak-label precision hypothesis, not semantic overthinking ground truth. Useful corrections can satisfy both cues; recall may fall. No length or final correctness filter.')
    save(out/'cpu_checks.json',audit)
    prev=HERE/'trajectory_length_vector_20260918_run2';plan=read(prev/'plan.json');rel=read(prev/'release.json')
    plan.update(run_id=run,experiment_kind='strict_and_vector_v1',arms=[dict(name='AND_NORM_L27',vector=run+'/norm_vector.pt',suppression_table=run+'/opening.npz')],new_answers=500,
        change='Replace over=lexical_hit OR low confidence with AND. Under pool unchanged. Match original vector norm, retain original fit/controller/layer/L27. No trajectory-length filter.',
        secondary_arm=None,process_hard_stop_seconds=2100,hard_stop_seconds_per_arm=1800,
        engineering=dict(rows=8,arms=['L27_REFERENCE','AND_NORM_L27','AND_NORM_REPEAT'],max_tokens=512,gate='CPU exact old reconstruction; no divergence before first injection; repeated candidate tokens and controller history exact'),
        expected_cost='One 1.5B MATH500 arm, approx7-12 minutes plus <2 minute engineering and grading. No baseline regeneration or extra inference forwards.',
        supersedes=None,interpretation='Exploratory exposed benchmark, not independent confirmation. Only disagreement-based extraction filtering plus fixed norm matching. No posthoc threshold tuning.')
    save(out/'plan.json',plan)
    for n in ('opening.npz','historical_compact.json'):(out/n).write_bytes((prev/n).read_bytes())
    names=list(rel['source_sha256'])+['prepare_strict_and_vector.py'];files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    rel.update(status='Fixed AND extraction candidate, CPU passed; engineering then full500',plan_relative_path=run+'/plan.json',plan_sha256=sha(out/'plan.json'),artifact_root=run,
        artifact_sha256={n:sha(out/n) for n in ('norm_vector.pt','opening.npz','historical_compact.json','plan.json','cpu_checks.json')},source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    save(out/'release.json',rel)
    for n in ('release.json','plan.json','cpu_checks.json','norm_vector.pt','opening.npz','historical_compact.json'):files[run+'/'+n]=(out/n).read_bytes()
    package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with package.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as t:
            for n,b in files.items():
                info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(audit=audit,package=str(package),sha256=sha(package))))
if __name__=='__main__':main()
