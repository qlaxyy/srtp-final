"""Freeze a direction x positive-control ablation, with two reused arms."""
import copy,hashlib,io,json,tarfile
from pathlib import Path
import torch
from prepare_length_vector import HERE,ROOT,read,sha,save
from nonpositive_adapter import cpu_checks

def main():
    rel='length_sign_ablation_20260918_run1';out=HERE/rel;out.mkdir(exist_ok=False)
    checks=cpu_checks();parent=HERE/'trajectory_length_refit_20260918_run1'
    previous=read(parent/'release.json');prior=read(parent/'plan.json')
    norm=HERE/'trajectory_length_vector_20260918_run2/norm_vector.pt'
    old=ROOT/'.codex_work/auto_code_v2_500_20260908/auto_vector.pt'
    nv=torch.load(norm,weights_only=True);ov=torch.load(old,weights_only=True)
    assert abs(float(nv.double().norm()/ov.double().norm())-1)<1e-6
    for n in ('opening.npz','historical_compact.json'):(out/n).write_bytes((parent/n).read_bytes())
    (out/'norm_vector.pt').write_bytes(norm.read_bytes())
    save(out/'cpu_checks.json',dict(checks=checks,matched_norm=True,
        original_vector_sha256=sha(old),new_vector_sha256=sha(norm),
        state_audit_sha256=sha(HERE/'length_state_alignment_20260918_run1/result.json')))
    plan=copy.deepcopy(prior)
    plan.update(experiment_kind='nonpositive_v1',run_id=rel,new_answers=1000,
        arms=[dict(name='OLD_NONPOS_L27',suppression_table=rel+'/opening.npz'),
              dict(name='LENGTH_NONPOS_L27',suppression_table=rel+'/opening.npz',vector=rel+'/norm_vector.pt')],
        change='Set only positive native R coefficients to zero before native scale-history recording; retain original raw confidence, variance, initial -1 and all negative coefficients. Compare original and length-filtered norm-matched directions.',
        engineering=dict(rows=8,arms=['OLD_SIGN_REFERENCE','OLD_SIGN_OFF','OLD_SIGN_SHADOW','OLD_NONPOS_L27','LENGTH_SIGN_REFERENCE','LENGTH_SIGN_OFF','LENGTH_SIGN_SHADOW','LENGTH_NONPOS_L27'],max_tokens=512,
            gate='both directions: off/shadow exact tokens and histories, actual positive removal, no divergence before first changed coefficient; reject preemption'),
        hard_stop_seconds_per_arm=1800,process_hard_stop_seconds=3600,
        factorial=dict(direction=['old','length_norm'],positive_branch=['original','zero'],
            reused=['L27_L27','LENGTH_NORM_L27'],new=['OLD_NONPOS_L27','LENGTH_NONPOS_L27'],
            interaction='(LENGTH_NONPOS - LENGTH_NORM) - (OLD_NONPOS - L27); mean token counts and accuracy percentage points. Negative token interaction means removing positive control helps new direction more.',
            limits='Historical reused controls and exposed benchmark; cannot establish original ReBalance x CGRS synergy.'),
        decision=dict(primary_reference='L27_L27',max_accuracy_loss_pp=2,
            advance='Each new arm must reduce both thinking and total mean tokens versus L27, not increase caps, lose at most2pp accuracy versus L27 and R. No automatic GSM in this batch.',
            length_specific='Also compare LENGTH_NONPOS with OLD_NONPOS; an improvement shared by both directions is not evidence for length filtering.',
            uncertainty='20000 paired bootstrap; nominal and descriptive, repeated benchmark exposure and sequential selection disclosed.',
            stop='Engineering mismatch, preemption, nonfinite coefficient, OOM, asset mismatch, or timeout: stop, preserve partials.'),
        advancement='Same frozen-L27 compression/accuracy/cap limits; no GSM expansion in this batch.',
        secondary_arm=None,supersedes=None,
        interpretation='Mechanism ablation of positive-control compatibility; exact-repeat CPU proxy is not semantic gold or causal evidence.',
        expected_cost='Two distinct full MATH500 arms on1.5B; about16-25min, plus8x8x512 engineering; no extra model forwards.')
    save(out/'plan.json',plan)
    release=copy.deepcopy(previous)
    src=list(release['source_sha256'])+['nonpositive_adapter.py','prepare_nonpositive.py']
    release.update(artifact_root=rel,plan_relative_path=rel+'/plan.json',plan_sha256=sha(out/'plan.json'),
        artifact_sha256={n:sha(out/n) for n in ('norm_vector.pt','opening.npz','historical_compact.json','plan.json','cpu_checks.json')},
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in src})
    save(out/'release.json',release)
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in src}
    for n in list(release['artifact_sha256'])+['release.json']:files[rel+'/'+n]=(out/n).read_bytes()
    archive=HERE.parents[3]/('.codex_work/'+rel+'.tar.gz')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tf:
            for n,b in files.items():
                entry=tarfile.TarInfo(n);entry.size=len(b);tf.addfile(entry,io.BytesIO(b))
    print(json.dumps(dict(archive=str(archive),sha256=sha(archive),checks=checks)))

if __name__=='__main__':main()
