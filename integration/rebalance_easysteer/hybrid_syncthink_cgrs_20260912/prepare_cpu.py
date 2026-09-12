"""Read-only CPU audit and provisional B-line data registration. Never runs GPU.

The external A-line checkout is an input, not an output or import location.
Output must be new; derived reservations are NOT frozen until coordination.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
from types import SimpleNamespace
from reference import SyncConfig, sync_action, raw_statistics, cgrs_probability, token_interaction

ROOT = Path(__file__).resolve().parents[3]
NS = 'hybrid_syncthink_cgrs_20260912'


def sha(path, lf=False):
    b = Path(path).read_bytes()
    return hashlib.sha256(b.replace(b'\r\n', b'\n') if lf else b).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def phash(problem):
    # Conservative format normalization; preserve case, numbers and math signs.
    s = re.sub(r'\s+', '', unicodedata.normalize('NFKC', problem))
    return hashlib.sha256(s.encode()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def save(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def formula_checks():
    checks = {}
    x = [2., 1., 0.]
    assert sync_action(x, 2, 0, True)[0] is x
    assert sync_action(x, 2, 0, True, SyncConfig('shadow'))[0] is x
    assert sync_action(x, 2, 0, False, SyncConfig('enforce'))[0] is x
    checks['off_shadow_identity_and_answer_inactive'] = True
    a = raw_statistics(x, 2)
    b = raw_statistics([z+1000 for z in x], 2)
    assert abs(a['entropy']-b['entropy']) < 1e-12 and a['rank']==b['rank']
    assert [raw_statistics([0.,0.,0.], i)['rank'] for i in range(3)] == [0,1,2]
    checks['shift_invariance_and_rank_ties'] = True
    thresholds = [sync_action(x,2,100,True,SyncConfig('shadow',v))[1]['threshold'] for v in [0.,.4,.8,1.6]]
    assert thresholds == sorted(thresholds,reverse=True)
    checks['lambda_larger_is_stricter_on_same_prefix'] = thresholds
    _, cap_a = sync_action(x,2,63,True,SyncConfig('shadow'))
    _, cap_b = sync_action(x,2,15999,True,SyncConfig('shadow'))
    assert cap_a['threshold']==cap_b['threshold']
    checks['pacing_cap'] = cap_a['threshold']
    y,s = sync_action([0.,10.],1,0,True,SyncConfig('enforce'))
    assert y[1]==0. and y[0]==-math.inf and s['trigger']
    checks['forced_end_single_support'] = True
    c,p = cgrs_probability([math.log(1000)]*3,1000)
    assert c==0 and p==0
    c,p = cgrs_probability([0.]*3,1000)
    assert c==1 and p==1
    assert cgrs_probability([.5],1000,.1)[1] > cgrs_probability([.5],1000,.9)[1]
    checks['full_entropy_limits_and_delta_direction'] = True
    # Equal maximum probability cannot identify full entropy or transition rank.
    p1=[.5,.49,.01]; p2=[.5,.25,.25]
    h1=raw_statistics(list(map(math.log,p1)),1)
    h2=raw_statistics(list(map(math.log,p2)),1)
    assert abs(h1['max_probability']-h2['max_probability'])<1e-12 and h1['entropy']!=h2['entropy']
    checks['max_probability_not_sufficient'] = [h1,h2]
    # Masking a rival changes both normalized confidence and end rank.
    before=raw_statistics(list(map(math.log,[.6,.3,.1])),1)
    after=raw_statistics([-1000,math.log(.3),math.log(.1)],1)
    checks['post_mask_contamination_witness'] = dict(before=before,after=after)
    assert before['rank']==1 and after['rank']==0
    # Winning over BOTH singles need not constitute synergy.
    assert token_interaction(dict(U=100,R=80,S=90,RS=75))==5
    assert token_interaction(dict(U=100,R=80,S=90,RS=65))==-5
    checks['dominance_not_synergy'] = True
    for bad in ([float('nan'),0.],[float('inf'),0.]):
        try: raw_statistics(bad,1)
        except ValueError: pass
        else: raise AssertionError('Nonfinite logits accepted')
    checks['nonfinite_rejection'] = True
    return checks


def author_audit(author):
    source = author/'crgs.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    env={'argparse':argparse,'np':__import__('numpy'),'math':math}
    extract=ast.Module(body=[funcs['parse_args'],funcs['calculate_certrainty_score']],type_ignores=[])
    exec(compile(extract,str(source),'exec'),env)
    old=sys.argv
    try:
        sys.argv=['crgs.py']; args=env['parse_args']()
        attrs={n.attr for n in ast.walk(funcs['main']) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='args'}
        missing=sorted(attrs-set(vars(args)))
        sys.argv=['crgs.py','--policy','entropy']
        with contextlib.redirect_stderr(io.StringIO()):
            try:env['parse_args']()
            except SystemExit as e: rejected=e.code
            else:rejected=0
    finally:sys.argv=old
    fake={i:SimpleNamespace(logprob=math.log(.001)) for i in range(5)}
    c=float(env['calculate_certrainty_score']([fake,fake,fake]))
    assert c>.9 and missing==['points','rep','suppress_ratio'] and rejected==2
    # Test the real nested callback binding without executing GPU code.
    node=next(n for n in ast.walk(funcs['main']) if isinstance(n,ast.FunctionDef) and n.name=='probabilistic_suppress_processor')
    wrapper=ast.parse('def build():\n callbacks=[]\n for suppression_probability in (0.2,0.8):\n  pass\n return callbacks\n')
    wrapper.body[0].body[1].body=[node,ast.parse('callbacks.append(probabilistic_suppress_processor)').body[0]]
    ast.fix_missing_locations(wrapper)
    ns={'torch':SimpleNamespace(Tensor=object),'List':list}
    exec(compile(wrapper,'isolated_callback_binding','exec'),ns)
    callbacks=ns['build']()
    bound=[dict(zip(f.__code__.co_freevars,[c.cell_contents for c in f.__closure__]))['suppression_probability'] for f in callbacks]
    assert bound==[.8,.8]
    return dict(commit=subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip(),
                files={str(p.relative_to(author)):sha(p) for p in author.rglob('*') if p.is_file() and '.git' not in p.parts},
                missing_parser_attributes=missing,script_calls_missing_cgrs_py=not (author/'cgrs.py').exists(),
                policy_argument_exit_code=rejected,public_score_for_uniform_V1000_top5=c,
                paper_full_entropy_certainty=0,callback_probabilities_for_two_requests=bound,
                notes=['AST-extracted CPU functions only; no model/vLLM import or startup',
                       'last_token_strs read at line 162 before local assignment at 167',
                       'logprobs=5 differs from full vocabulary entropy; probe token 0 skipped',
                       'default think_ratio=0.6 introduces separate forced budget',
                       'probe outputs not included in reported generated response or main output budget'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--author-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.perf_counter()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source=args.source_root.resolve()
    train_path=source/'sources/ReBalance/Data/Math_Train/test.jsonl'; train=rows(train_path)
    plan_path=source/'integration/rebalance_easysteer/configs/local_prepared_batch_20260912/plan.json'
    plan=read(plan_path)
    assert sha(train_path,True)==plan['train_sha256']
    exclusions=dict(plan['exclusions'])
    for cand,stages in plan['splits'].items():
        for stage,ids in stages.items():exclusions[cand+'/'+stage]=ids
    forbidden=set().union(*(set(v) for v in exclusions.values()))
    seen={phash(train[i]['problem']) for i in forbidden}
    for name,digest in plan['test_prompt_source_sha256'].items():
        assert sha(source/name,True)==digest
        seen.update(phash(r['problem']) for r in rows(source/name))
    eligible=[]
    for i,r in enumerate(train):
        h=phash(r['problem'])
        if i not in forbidden and h not in seen:
            eligible.append((i,h));seen.add(h)
    ordered=sorted(eligible,key=lambda ih:hashlib.sha256((NS+'|'+ih[1]).encode()).hexdigest())
    assert len(ordered)>=272
    reservation=[]
    for j,(i,h) in enumerate(ordered[:272]):
        role='engineering' if j<8 else 'screening' if j<72 else 'confirmation'
        reservation.append(dict(train_index=i,problem_sha256=h,purpose=role,status='proposed_not_frozen'))
    tok_path=source/'.codex_work/label_audit_30_20260910/tokenizer.json'; tok=read(tok_path)
    vocab=tok['model']['vocab']; inverse={i:s for s,i in vocab.items()}
    inverse.update({r['id']:r['content'] for r in tok['added_tokens']})
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+n);n+=1
    decoder=dict(zip(map(chr,cs),bs))
    def token_bytes(i):
        s=inverse[i]
        if i in {t['id'] for t in tok['added_tokens']}:return s.encode()
        return bytes(decoder[c] for c in s)
    display=lambda i:token_bytes(i).decode('utf-8',errors='replace')
    published=[('Wait',14190),(' Wait',13824),('wait',11489),(' wait',3783),('But',3983),(' But',1988),('but',8088),(' but',714),('Alternatively',38478),(' Alternatively',38478),('Alternative',75763),(' Alternative',41109),('Hmm',80022),(' Hmm',88190)]
    table=[dict(paper_text=s,paper_id=i,actual_text=display(i),matches=s==display(i)) for s,i in published]
    corrected_ids=[14190,1988,714,13824,3983,3783,11489,8088,38478,41109,75763,92014,80022,88190]
    end_id=next(i for i,s in inverse.items() if s=='</think>')
    boundaries={i for i,s in inverse.items() if 'ĊĊ' in s}
    raw_path=source/'.codex_work/overnight_research_20260912/seal_comparison_all_20260912/seal_comparison130_20260912/math_train_original_dynamic.json'
    raw=read(raw_path);assert raw['max_tokens']==16000 and len(raw['records'])==100
    byhash={phash(r['problem']):i for i,r in enumerate(train)}
    used=[]; details=[]
    for r in raw['records']:
        h=phash(r['problem']);i=byhash[h];assert i in forbidden
        used.append(dict(train_index=i,problem_sha256=h,purpose='existing_dynamic_trace_CPU_development',status='used'))
        ids=r['token_ids']; assert len(ids)==r['tokens']
        think=ids[:ids.index(end_id)] if end_id in ids else ids
        head=True;head_hits=inner_hits=0;since=longest=0;boundary_count=0
        for t in think:
            if t in corrected_ids:
                if head:head_hits+=1
                else:inner_hits+=1
            if t in boundaries:head=True;since=0;boundary_count+=1
            else:
                since+=1;longest=max(longest,since)
                if token_bytes(t).strip():head=False
        details.append(dict(train_index=i,thinking_tokens=len(think),head_trigger_tokens=head_hits,interior_trigger_tokens=inner_hits,boundary_count=boundary_count,longest_nonboundary_run=longest,capped=len(ids)==16000))
    assert not set(x['train_index'] for x in used)&set(x['train_index'] for x in reservation)
    registry=dict(namespace=NS,normalization='Unicode NFKC then remove all Unicode whitespace; SHA256 UTF-8; preserve case/numbers/symbols',
                  coordination_status='pending_executor_global_A_B_reconciliation',source_plan_sha256=sha(plan_path),
                  excluded_unique=len(forbidden),eligible=len(eligible),selection='ascending SHA256(namespace + | + normalized_problem_sha256)',
                  used=used,proposed_reservations=reservation)
    save(args.output/'data_registry.json',registry)
    save(args.output/'trace_counts.json',details)
    tracked=['docs/research/00-研究交接.md','docs/research/03-ReBalance适配与运行.md',
             'integration/rebalance_easysteer/eval/rebalance_dynamic_eval.py',
             'sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/model_runner.py',
             'sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/sample/sampler.py',
             'sources/EasySteer/vllm-steer/vllm/v1/worker/gpu/steer_vector_utils.py',
             'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py']
    report=dict(namespace=NS,baseline_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                read_checkout_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                source_files={n:dict(sha256=sha(ROOT/n),git_blob=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD:'+n],text=True).strip()) for n in tracked},
                input_hashes={'train_LF':sha(train_path,True),'tokenizer':sha(tok_path),'dynamic_raw':sha(raw_path),'A_prior_plan':sha(plan_path)},
                formula_checks=formula_checks(),author_code=author_audit(args.author_root),token_table=table,
                end_token_id=end_id,actual_corrected_trigger_tokens={str(i):display(i) for i in corrected_ids},
                traces=dict(n=100,total_thinking=sum(x['thinking_tokens'] for x in details),
                            head_hits=sum(x['head_trigger_tokens'] for x in details),interior_hits=sum(x['interior_trigger_tokens'] for x in details),
                            questions_with_triggers=sum(x['head_trigger_tokens']+x['interior_trigger_tokens']>0 for x in details),
                            boundaries=sum(x['boundary_count'] for x in details),
                            capped=sum(x['capped'] for x in details),
                            note='No correctness recomputation; raw correct flags are not the author regrade. Lexical counts are not useless-reflection counts or intervention opportunities.'),
                budget=dict(new_answers=0,model_forward_calls=0,GPU_calls=0,server_connections=0,environment_installs=0),
                runtime_ready=False,data_frozen=False,elapsed_seconds=time.perf_counter()-start)
    save(args.output/'cpu_audit.json',report)
    print(json.dumps({k:report[k] for k in ['formula_checks','traces','budget','elapsed_seconds']},indent=2))


if __name__=='__main__':main()
