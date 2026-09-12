"""Prepare exact saved BCC prefixes locally. Never load a model or generate answers."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.root.resolve(); out = args.output.resolve()
    require(not out.exists(), 'Output already exists')
    cfg = root/'integration/rebalance_easysteer/configs'
    bcc = cfg/'bcc_v1_20260912'; old = cfg/'local_prepared_batch_20260912'
    manifest = read(bcc/'candidate_manifest.json')
    clarification = read(bcc/'content_clarification.json')
    require(clarification['user_selected_standard']=='core_derivation_valid_local_errors_retained'
            and clarification['core_valid_pairs']==10, 'Content clarification missing')
    prompt = read(old/'prompt_audit.json')
    base = root/'.codex_work/overnight_research_20260912/prepared_gpu_execution_20260912/unpacked/local_prepared_batch_20260912'
    receipt = read(base/'native_prompt_audit.json')
    require(receipt['prompt_audit_sha256'] == sha(old/'prompt_audit.json'), 'Native prompt receipt mismatch')
    records = []; sources = {str(old/'prompt_audit.json'):sha(old/'prompt_audit.json')}
    for batch, group in sorted(manifest['groups'].items()):
        rows = {}
        for arm in ('original_dynamic', batch):
            path = base/batch/'screen'/f'{arm}.json'; labels = path.with_name(arm+'.author.json')
            obj = read(path); lab = read(labels)
            require(lab['input_sha256'] == sha(path), 'Author label input mismatch')
            sources[str(path)] = sha(path); sources[str(labels)] = sha(labels)
            rows[arm] = {s['train_index']:r for s,r in zip(lab['records'],obj['rebalance_dynamic']['records'])}
        prompts = {r['train_index']:r for r in prompt['records'][batch+'/screen']}
        ordered = sorted(group['details'], key=lambda x: hashlib.sha256(
            f"BCC-v1|20260912|{batch}|{x['train_index']}".encode()).hexdigest())
        for pair in ordered:
            item = dict(pair_id=f"{batch}:{pair['train_index']}", batch=batch, train_index=pair['train_index'],
                        split=pair['split'], short_arm=pair['short_arm'], placebo_sign=pair.get('placebo_sign'),
                        stratum=[batch,pair['short_arm']], prefixes=[])
            for arm, positions in sorted(pair['positions'].items()):
                row = rows[arm][pair['train_index']]; ids = row['token_ids']; pp = prompts[pair['train_index']]
                a,b = positions
                require(pair['lcp_tokens'] <= a < b < row['thinking_tokens'], 'Invalid capture positions')
                prompt_ids = pp['prompt_token_ids']
                require(pp['prompt_text_sha256'] == pair['prompt_text_sha256'], 'Wrong prompt')
                # Extra 32 saved tokens only for the predefined causal-length controls.
                longer_stop = min(b+33,row['thinking_tokens'])
                item['prefixes'].append(dict(arm=arm, input_ids=prompt_ids+ids[:b+1],
                    longer_input_ids=prompt_ids+ids[:longer_stop],
                    positions=[len(prompt_ids)+a,len(prompt_ids)+b],
                    output_token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
            records.append(item)
    require(len(records)==76 and sum(r['split']=='fit' for r in records)==40, 'Wrong split')
    require(sum(len(p['input_ids']) for r in records for p in r['prefixes'])==66159, 'Wrong replay token count')
    plan0 = read(old/'sampled_confidence/screen/plan.json')
    fitpath = root/'.codex_work/question_balanced_20260911/original_selected_layer/fit.json'
    fit = read(fitpath)
    require(fit['vector_sha256']=='fb360600cad48aebf1242ae125f7ccb3860177ae542351377a6898eb3a840b93', 'Wrong parent vector')
    engineering = [r['pair_id'] for r in records if r['split']=='fit'][:2]
    out.mkdir(parents=True); save(out/'prefixes.json',records)
    save(out/'plan.json',dict(method='bcc-v1-fixed-controller', status='prepared_CPU_only',
        model=plan0['model'], model_files_sha256=plan0['model_files_sha256'],
        decoder_output_layer=20, hidden_size=1536, dtype='bfloat16', attention='sdpa',
        batch_size=1, timeout_seconds=600, new_answers=0, pairs=76, hidden_vectors=304,
        input_tokens=66159, engineering_pair_ids=engineering,
        max_absolute_difference=.125, max_relative_l2=.002,
        parent_fit_sha256=sha(fitpath), parent_vector_sha256=fit['vector_sha256'],
        parent_vector_norm=fit['vector_norm'], prefixes_sha256=sha(out/'prefixes.json'),
        manifest_sha256=sha(bcc/'candidate_manifest.json'), source_inputs_sha256=sources,
        runtime_source_sha256={name:sha(Path(__file__).with_name(name)) for name in
                              ('prepare_bcc.py','replay_bcc_prefixes.py','fit_bcc.py')},
        content_gate='passed_under_user_clarified_standard', content_clarification_sha256=sha(bcc/'content_clarification.json'),
        authorization='GPU requires current user authorization; preparation is not authorization'))
    print(json.dumps(dict(status='prepared_CPU_only',pairs=76,input_tokens=66159,engineering_pair_ids=engineering,plan_sha256=sha(out/'plan.json'))))


if __name__=='__main__':
    main()
