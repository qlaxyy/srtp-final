"""Fixed full continuations of existing eight native snapshots; no vector fitting."""
import hashlib,io,json,tarfile
from prepare_length_vector import HERE,ROOT,read,save,sha

def main():
    run='counterfactual_labels_20260918_run1';out=HERE/run;out.mkdir(exist_ok=False)
    (out/'.gitattributes').write_text('*.json -text\n*.npz binary -text\n',encoding='utf8')
    parent=HERE/'counterfactual_fork_20260918_run1';rel=read(parent/'release.json');plan=read(parent/'plan.json')
    verified=read(parent/'verified_result.json');assert verified['local_checks_passed'] and verified['fork_gate']['passed']
    train=ROOT/'sources/ReBalance/Data/Math_Train/test.jsonl'
    assert sha(train)=='4bd7f5266f6c5c4fc729da2d9cbefbe2453cf8df7bd6ba93dc02b9e11512cefd'
    data=[json.loads(s) for s in train.read_text(encoding='utf8').splitlines()]
    for row in plan['rows']:
        gold=data[row['train_index']];assert gold['problem']==row['problem'];row['gold_row']=gold
    plan.update(run_id=run,phase='action_labels',arms=['apply_a','skip'],max_tokens=16000,max_new_continuation_tokens='16000 minus saved prefix length',
        purpose='Full paired continuations to audit whether one-action benefit provides usable extraction labels. No new vector or benchmark efficacy claim.',
        gate='Reuse passed no-op gate, require paired identities and same prior masks, only final injection differs; all full lengths <=16000.',
        label_rule='Both correct, stop and think closed: prefer arm only if both thinking and total are strictly lower. One correct with completed thinking: correctness preference separately. Caps, both wrong, mixed length objectives and ties are inconclusive.',
        extraction_rule='Never attach these L27 on-policy prefixes to original unsteered 500 feature rows. No vector fit from eight rows; action labels are not over/under gold labels.',
        remaining_limit='600 process seconds; retain partial outputs on failure; no automatic retries or parameter search',
        expected_cost='16 full continuations, <=254724 new tokens; about2-6 minutes, no duplicate apply_b arm',
        limitation='Eight previously exposed early negative-coefficient prefixes, same restarted seed42. No positive-action coverage, no independent efficacy test or expected causal effect estimate.')
    plan['engineering_evidence_sha256']=sha(parent/'verified_result.json')
    plan['grader_sha256']={n:sha(ROOT/'sources/ReBalance/utils'/n) for n in ('parser.py','grader.py')}
    save(out/'plan.json',plan);(out/'opening.npz').write_bytes((parent/'opening.npz').read_bytes())
    names=list(rel['sources'])+['grade_action_pilot.py'];files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in names}
    rel.update(sources={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},plan_sha256=sha(out/'plan.json'))
    save(out/'release.json',rel)
    for n in ('plan.json','release.json','opening.npz'):files[run+'/'+n]=(out/n).read_bytes()
    package=HERE.parents[3]/('.codex_work/'+run+'.tar.gz')
    with package.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as t:
            for n,b in files.items():
                info=tarfile.TarInfo(n);info.size=len(b);t.addfile(info,io.BytesIO(b))
    print(json.dumps(dict(package=str(package),sha256=sha(package))))

if __name__=='__main__':main()
