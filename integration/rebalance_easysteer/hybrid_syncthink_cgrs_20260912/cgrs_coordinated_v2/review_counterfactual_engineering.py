"""Local archive/identity/token/mask recount; no efficacy grading."""
import json,tarfile
from prepare_length_vector import HERE,read,save,sha

def main():
    work=HERE.parents[3]/'.codex_work';cap=work/'counterfactual_capture_20260918_completed.tar.gz';fork=work/'counterfactual_fork_20260918_run1.completed.tar.gz'
    assert sha(cap)=='322e0253becb90abab5ec65c6e202b3046f80a83f58b7b89c0324ccef97ebe45'
    assert sha(fork)=='8aa2d39f8c66d46d324a5b266f2a721b7af819b71000026d0b12209f02dbe145'
    with tarfile.open(cap) as t:
        prefix='counterfactual_capture_20260918_run2/'
        cg=json.load(t.extractfile(prefix+'complete.json'))
        a=json.load(t.extractfile(prefix+'reference_complete.json'));b=json.load(t.extractfile(prefix+'observer_complete.json'));assert a==b and len(a)==8
        fail=json.load(t.extractfile('counterfactual_capture_20260918_run1/failure.json'))
        cm=json.load(t.extractfile(prefix+'capture_manifest.json'))
        assert cg['release_sha256']==sha(HERE/'counterfactual_capture_20260918_run2/release.json')
    with tarfile.open(fork) as t:
        prefix='counterfactual_fork_20260918_run1/'
        fg=json.load(t.extractfile(prefix+'complete.json'))
        assert fg['release_sha256']==sha(HERE/'counterfactual_fork_20260918_run1/release.json')
        data={arm:json.load(t.extractfile(prefix+arm+'_complete.json')) for arm in ('apply_a','apply_b','skip')}
        masks={arm:json.load(t.extractfile(prefix+arm+'_prefill_masks.json')) for arm in data}
    assert data['apply_a']==data['apply_b'] and masks['apply_a']==masks['apply_b']
    registry=read(HERE/'counterfactual_fork_20260918_run1/plan.json')['rows'];checks=[]
    for i,row in enumerate(registry):
        for arm in data:
            r=data[arm][str(i)];assert r['train_index']==row['train_index'] and 0<len(r['token_ids'])<=128
            assert r['control']['tokens']==cm['rows'][i]['generated_tokens']+len(r['token_ids'])
        original=masks['apply_a'][i];skipped=masks['skip'][i]
        assert len(original)==len(skipped) and original[:-1]==skipped[:-1] and skipped[-1]==0
        checks.append(dict(train_index=row['train_index'],problem_sha256=row['problem_sha256'],prefix_sha256=cm['rows'][i]['prefix_sha256'],
            original_target_scale=original[-1],skip_target_scale=skipped[-1],
            generated_tokens={arm:len(data[arm][str(i)]['token_ids']) for arm in data},
            apply_skip_same_tokens=data['apply_a'][str(i)]['token_ids']==data['skip'][str(i)]['token_ids']))
    assert any(r['original_target_scale'] for r in checks)
    result=dict(capture_gate=cg,fork_gate=fg,local_checks_passed=True,rows=checks,
        capture_total_new_tokens=sum(len(r['token_ids']) for r in a.values())*2,
        fork_total_new_tokens=sum(len(r['token_ids']) for records in data.values() for r in records.values()),
        capture_failed_attempt_seconds=fail['wall_seconds'],failure_new_tokens=None,
        archives={cap.name:sha(cap),fork.name:sha(fork)},
        conclusion='Native snapshot capture and two same-partition no-op fork continuations passed on eight short training cases; skip mask changed at most target input. No causal outcome labels or vector efficacy yet.',
        limits='Not long-prefix replay validation, full numerical equality to uninterrupted parent, semantic label quality, independent benchmark or accuracy evidence. Request-local sample RNG restarts in both sibling forks.')
    save(HERE/'counterfactual_fork_20260918_run1/verified_result.json',result)
    print(json.dumps(dict(capture_tokens=result['capture_total_new_tokens'],fork_tokens=result['fork_total_new_tokens'],
        changed_targets=sum(r['original_target_scale']!=0 for r in checks),different_short_continuations=sum(not r['apply_skip_same_tokens'] for r in checks))))

if __name__=='__main__':main()
