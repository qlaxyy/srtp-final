"""Complete same-label calibration with the original LDA crossing formula."""
import copy,hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
import torch
from prepare_length_vector import HERE,ROOT,read,save,sha
from prepare_label_alignment import fit,load

def save_once(p,x):
    if p.exists():assert read(p)==x
    else:save(p,x)

def main():
    rel='trajectory_length_refit_20260918_run1';out=HERE/rel
    parent=HERE/'trajectory_length_vector_20260918_run2'
    # Recompute once into a CPU check folder to make the prior probe reproducible.
    cache=ROOT/'.codex_work/question_balanced_20260911/original_selected_layer'
    steps=read(cache/'steps.json');audit=read(HERE/'trajectory_length_labels_20260918_run1/result.json')
    c=np.array([s['confidence'] for s in steps]);h=np.array([s['lexical_hit'] for s in steps]);q=np.array([s['question'] for s in steps])
    lens=np.array([r['thinking_tokens'] for r in audit['questions']]);old=read(ROOT/'.codex_work/auto_code_v2_500_20260908/fit.json')['parameters']
    over=(h|(c<old['q25c']))&(lens[q]>lens.mean());under=(~h&(c>old['q75c']))&(lens[q]<lens.mean())
    runtime=load('length_refit_runtime',ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py')
    check=out/'cpu_reproduction'
    if check.exists():
        vector=torch.load(check/'auto_vector.pt',weights_only=True);info=read(check/'fit.json')
    else:vector,info,_=fit(np.load(cache/'layer_21.npy',mmap_mode='r'),over,under,old,check,runtime)
    assert torch.equal(vector,torch.load(out/'auto_vector.pt',weights_only=True))
    assert info['parameters']==read(out/'fit.json')['parameters']
    assert sha(out/'auto_vector.pt')==sha(parent/'auto_vector.pt')
    oldrelease=read(parent/'release.json');plan=read(parent/'plan.json')
    plan.update(experiment_kind='length_refit_v1',run_id=rel,new_answers=500,
        arms=[dict(name='LENGTH_REFIT_L27',vector=rel+'/auto_vector.pt',fit=rel+'/fit.json',suppression_table=rel+'/opening.npz')],
        change='Same raw length-conditioned vector; replace only low_val_2 by original LDA maximum-crossing fit on the same new classes. All other controller parameters exactly unchanged.',
        secondary_arm=None,hard_stop_seconds_per_arm=1200,process_hard_stop_seconds=1500,
        engineering=dict(rows=8,arms=['L27_REFERENCE','LENGTH_REFIT_L27','LENGTH_REFIT_REPEAT'],max_tokens=512,gate='exact CPU fit reproduction, finite curve grid; native repeat exact tokens and control history'),
        advancement='Same L27 compression/accuracy/cap rule. Later sequential exploratory candidate after first two full results; not independent confirmation. No GSM expansion in this batch.',
        expected_cost='Single full MATH500 after 8x3x512 engineering; about8-12min.',
        sequential_context='Raw-only and norm-matched replacements failed compression advancement. Refit is determined from frozen calibration data using existing formula, not an optimized benchmark threshold.')
    save_once(out/'plan.json',plan)
    if (out/'opening.npz').exists():assert sha(out/'opening.npz')==sha(parent/'opening.npz')
    else:(out/'opening.npz').write_bytes((parent/'opening.npz').read_bytes())
    refs=read(parent/'historical_compact.json')
    archive=HERE.parents[3]/'.codex_work/trajectory_length_vector_20260918_run2.completed.tar.gz'
    with tarfile.open(archive) as tf:
        base='trajectory_length_vector_20260918_run2/full/'
        for name in ('LENGTH_L27','LENGTH_NORM_L27'):
            d=json.load(tf.extractfile(base+name+'/result.json'))
            labels=[json.loads(s) for s in tf.extractfile(base+name+'/author_partial.jsonl').read().splitlines()]
            records=[dict(label,tokens=r['tokens'],thinking_tokens=r['thinking_tokens']) for r,label in zip(d['records'],labels)]
            assert len(records)==500
            refs['groups'][name]=dict(records=records,generation_seconds=d['generation_seconds'])
    refs.setdefault('source_sha256',{})[str(archive)]=sha(archive);save_once(out/'historical_compact.json',refs)
    release=copy.deepcopy(oldrelease)
    release.update(artifact_root=rel,plan_relative_path=rel+'/plan.json',plan_sha256=sha(out/'plan.json'),
        artifact_sha256={n:sha(out/n) for n in ('auto_vector.pt','fit.json','opening.npz','historical_compact.json','plan.json')})
    src=list(release['source_sha256'])+['prepare_length_refit.py']
    release['source_sha256']={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in src}
    save(out/'release.json',release)
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in src}
    for n in list(release['artifact_sha256'])+['release.json']:files[rel+'/'+n]=(out/n).read_bytes()
    archive=HERE.parents[3]/('.codex_work/'+rel+'.tar.gz')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tf:
            for n,b in files.items():
                i=tarfile.TarInfo(n);i.size=len(b);tf.addfile(i,io.BytesIO(b))
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),low_val_2=info['parameters']['low_val_2'])))

if __name__=='__main__':main()
