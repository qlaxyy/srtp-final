"""Predeclared direction x positive-control interaction on complete paired data."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from prepare_length_vector import HERE,read,save,sha
from grade_label_alignment import compare,summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);a=ap.parse_args()
    folder=HERE/'length_sign_ablation_20260918_run1';refs=read(folder/'historical_compact.json')['groups']
    groups={k:v['records'] for k,v in refs.items()}
    with tarfile.open(a.archive) as tf:
        root='length_sign_ablation_20260918_run1/full/'
        for name in ('OLD_NONPOS_L27','LENGTH_NONPOS_L27'):
            decision=read(folder/(name+'_decision.json'));assert decision['archive_sha256']==sha(a.archive)
            data=json.load(tf.extractfile(root+name+'/result.json'))
            labels=[json.loads(s) for s in tf.extractfile(root+name+'/author_partial.jsonl').read().splitlines()]
            groups[name]=[dict(l,tokens=r['tokens'],thinking_tokens=r['thinking_tokens']) for l,r in zip(labels,data['records'])]
            assert summary(groups[name])==decision['summary']
    keys=['LENGTH_NONPOS_L27','LENGTH_NORM_L27','OLD_NONPOS_L27','L27_L27']
    for k in keys:
        assert len(groups[k])==500
        assert [r['problem_sha256'] for r in groups[k]]==[r['problem_sha256'] for r in groups[keys[0]]]
    rng=np.random.default_rng(20260917);interaction={}
    for metric in ('tokens','thinking_tokens','correct'):
        values=[np.array([float(r[metric]) for r in groups[k]]) for k in keys]
        delta=values[0]-values[1]-values[2]+values[3]
        if metric=='correct':delta*=100
        boots=[]
        for _ in range(100):boots.extend(delta[rng.integers(0,500,(200,500))].mean(1))
        interaction[metric]=dict(difference_in_differences=float(delta.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())
    result=dict(status='complete paired four-cell descriptive mechanism comparison',summaries={k:summary(groups[k]) for k in keys},
        interaction=interaction,
        length_vs_old_with_positive_removed=compare(groups['OLD_NONPOS_L27'],groups['LENGTH_NONPOS_L27']),
        removing_positive_old=compare(groups['L27_L27'],groups['OLD_NONPOS_L27']),
        removing_positive_length=compare(groups['LENGTH_NORM_L27'],groups['LENGTH_NONPOS_L27']),
        bootstrap_replicates=20000,analysis_seed=20260917,archive_sha256=sha(a.archive),script_sha256=sha(Path(__file__)),
        limitations=['Two reused historical arms, two current arms; single generation seed and exposed benchmark.',
            'Interaction is additive mean token count or accuracy percentage points, not ReBalance x CGRS synergy.',
            'Exact-repeat calibration audit does not provide gold overthinking labels or establish causality.'])
    save(folder/'interaction.json',result)
    print(json.dumps(dict(summaries=result['summaries'],interaction=interaction),indent=2))

if __name__=='__main__':main()
