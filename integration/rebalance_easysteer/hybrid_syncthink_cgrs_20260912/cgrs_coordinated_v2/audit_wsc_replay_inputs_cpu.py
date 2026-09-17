"""Verify fixed diagnostic payload against original calibration; no LLMs."""
import hashlib
import json
from pathlib import Path
import tarfile
from wsc_replay_diagnostic import boundary_rows,self_test


def audit(plan_path,archive):
    plan=json.loads(Path(plan_path).read_text(encoding='utf8'))
    with tarfile.open(archive) as t:raw=t.extractfile('generations.jsonl').read()
    assert hashlib.sha256(raw).hexdigest()==plan['original_calibration_sha256']
    original={r['train_index']:r for r in map(json.loads,raw.splitlines())}
    for case in plan['cases']:
        row=original[case['train_index']]
        assert case['prompt_token_ids']==row['prompt_token_ids']
        assert case['token_ids']==row['token_ids'][:len(case['token_ids'])]
        assert case['problem']==row['problem']
        assert case['boundaries']==boundary_rows(case['token_ids'],len(case['prompt_token_ids']),set(plan['boundary_ids']))
    return dict(original_prompts_and_prefixes_exact=True,cases=len(plan['cases']),
        input_tokens=sum(len(c['prompt_token_ids'])+len(c['token_ids']) for c in plan['cases']),
        generated_tokens=0,tests=self_test())


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--archive',required=True)
    a=p.parse_args();print(json.dumps(audit(a.plan,a.archive),indent=2))
