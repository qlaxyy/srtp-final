"""CPU checks for unchanged dispatch and preregistered confirmation decisions."""
import ast, hashlib, itertools, json, math, pathlib, subprocess, sys, tempfile
ROOT=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).resolve().parent
NS=HERE.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def function(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env[name]

def main():
    schedule=function(NS/'run_batch.py','formal_schedule');cases=0
    for arms in (['R'],['R','S','RS'],['U','R','S','RS']):
        for soft,mix in itertools.product((False,True),repeat=2):
            r={'expansion':{'roles':['math_mix05','gsm8k_mix05'],'arms':arms},'soft2_rows':soft,'mix05':mix}
            expected=[]
            for stage in r['expansion']['roles']:
                for name,use_r,mode in [('U',False,'absent'),('R',True,'off'),('S',False,'enforce'),('RS',True,'enforce')]:
                    if name not in arms:continue
                    if soft and mode=='enforce':mode='mix05' if mix else 'soft'
                    expected.append((stage,name,use_r,mode))
            assert list(schedule(r))==expected
            r['completed_engineering_reuse']={'source':'example'}
            assert list(schedule(r))==expected
            cases+=2
    assert list(schedule({}))==[('screening','U',False,'absent'),('screening','R',True,'off'),('screening','S',False,'enforce'),('screening','RS',True,'enforce')]
    # Check the analyzer decision on complete matched synthetic outputs, including a failed token gate.
    sys.path.insert(0,str(HERE));import analyze
    stats=0
    with tempfile.TemporaryDirectory() as tmp:
        folder=pathlib.Path(tmp);role='math_mix05'
        for length_rs,expect in ((80,True),(110,False)):
            for arm,length in [('R',100),('S',120),('RS',length_rs)]:
                dest=folder/role/arm;dest.mkdir(parents=True,exist_ok=True)
                rows=[dict(train_index=i,problem=str(i),prompt_token_ids=[i],tokens=length,thinking_tokens=length-10,answer_tokens=9,hybrid={'first_bias':4,'bias_count':1}) for i in range(200)]
                obj=dict(status='complete',cap=16000,records=rows,generation_seconds=10,checkpoint_io_seconds=0,startup_seconds=1,control={'statistics_and_mask_gpu_ms':1,'extra_model_forwards':0,'probe_tokens':0})
                raw=json.dumps(obj).encode();(dest/'result.json').write_bytes(raw)
                grade={'input_sha256':hashlib.sha256(raw).hexdigest(),'seconds':1,'records':[{'train_index':i,'author_correct':True} for i in range(200)]}
                (dest/'author_grade.json').write_text(json.dumps(grade),encoding='utf-8')
            result=analyze.analyze(folder,role)
            assert result['fixed_confirmation_gate_pass']==expect
            assert math.isclose(result['comparisons']['RS_minus_R']['net_loss_probability_upper95_conservative'],1-.025**(1/200),abs_tol=1e-12)
            stats+=1
    proposal=read(HERE/'data_proposal.json');hashes=[]
    for role in read(HERE/'plan.json')['roles']:
        p=HERE/(role+'.json');rows=read(p)
        assert len(rows)==200 and hashlib.sha256(p.read_bytes()).hexdigest()==proposal['question_files'][role]
        hashes += [r['problem_sha256'] for r in rows]
    assert len(set(hashes))==400
    for name,v in read(HERE/'patch_manifest.json')['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==v['patched_lf_sha256']
        if name.startswith('sources/'):assert v['unchanged']
    compiled=[]
    for p in [NS/'run_batch.py',*HERE.glob('*.py')]:
        compile(p.read_text(encoding='utf-8'),str(p),'exec');compiled.append(p.name)
    result=dict(status='pass',dispatch_cases=cases+1,confirmation_behavior_cases=stats,unique_proposed_questions=400,compiled=compiled,shared_inference_changes=0,limitations=['No local Torch or model forward','Remote cross-line reconciliation and asset checks pending','Synthetic statistics tests are not experiment results'])
    with (HERE/'cpu_checks.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
