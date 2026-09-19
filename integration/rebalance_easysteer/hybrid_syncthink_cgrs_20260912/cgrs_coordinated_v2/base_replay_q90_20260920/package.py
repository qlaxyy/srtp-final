"""Prepare an isolated copy of the already validated evaluation runner."""
from prepare import HOME,HERE,ROOT,WORK,read,save,sha
import ast,json,shutil,tarfile

def main():
    old=HERE/'iterative_recalibration_eval_20260919'
    support='/root/autodl-tmp/projects/iterative_recalibration_20260919/integration/rebalance_easysteer/hybrid_syncthink_cgrs_20260912/cgrs_coordinated_v2/iterative_recalibration_20260919'
    code=(old/'evaluate.py').read_text(encoding='utf-8')
    code=code.replace("sys.path.insert(0,str(HOME.parent/'iterative_recalibration_20260919'))",f"SUPPORT=Path({support!r})\nsys.path.insert(0,str(SUPPORT))")
    code=code.replace("    if not a.engineering:\n        gate=read(a.engineering_result/'complete.json')\n        assert gate['passed'] and gate['engineering'] and gate['identity']==identity\n", "    assert not a.engineering, 'This fixed batch runs complete MATH500 only'\n    assert fit['confidence_high_quantile']==.9\n    expected=read(HOME/'summary.json')['groups'][a.case]\n    assert identity['fit_sha256']==expected['fit_sha256'] and identity['vector_sha256']==expected['vector_sha256']\n    curve=read(a.fit_dir/'curve_check.json');assert curve['finite'] and curve['max_residual']<1e-8\n")
    code=code.replace("source=HOME.parent/'iterative_recalibration_20260919'",'source=SUPPORT')
    code=code.replace('str(HOME.parent),str(a.runtime_root', 'str(SUPPORT.parent),str(a.runtime_root')
    code=code.replace("name='iterated_'+a.case", "name='base_replay_q90_'+a.case")
    ast.parse(code);(HOME/'evaluate.py').write_bytes(code.encode())
    grade=(old/'grade.py').read_text(encoding='utf-8').replace("sys.path.insert(0,str(HOME.parent/'iterative_recalibration_20260919'))",f"sys.path.insert(0,{support!r})")
    ast.parse(grade);(HOME/'grade.py').write_bytes(grade.encode())
    shutil.copy2(old/'rows.json',HOME/'rows.json')
    refs=read(old/'historical_compact.json')
    archive=ROOT/'.codex_work/iterative_recalibration_20260919/full_evidence.tar.gz'
    assert sha(archive)=='79f647d8217276613bbb05dee759e8ca924d7eb4fab704bb40eb8f84715b60ae'
    with tarfile.open(archive) as t:
        for case in ['R','L27']:
            base='iterative_recalibration_20260919_run1/'+case+'_math500/'
            records=[json.loads(line) for line in t.extractfile(base+'author_labels.jsonl')]
            assert len(records)==500
            refs['groups'][case+'_recalibrated_q75']={'records':records,'source_archive_sha256':sha(archive),
                'analysis':json.load(t.extractfile(base+'analysis.json'))}
    save(HOME/'historical_compact.json',refs)
    shutil.copy2(WORK/'summary.json',HOME/'summary.json')
    for case in ['R','L27']:
        target=HOME/(case+'_fit');target.mkdir()
        for name in ['fit.json','auto_vector.pt','protocol.json','curve_check.json','selected_layer.json','complete.json']:
            shutil.copy2(WORK/(case+'_fit')/name,target/name)
    save(HOME/'plan.json',dict(model='DeepSeek-R1-Distill-Qwen-1.5B',dataset='MATH-500',rows_per_case=500,cases=['R','L27'],
        temperature=.7,top_p=.95,seed=42,max_tokens=16000,max_num_seqs=256,max_num_batched_tokens=32768,
        gpu_memory_utilization=.90,steer_graph_mode='in_graph',async_scheduling=True,
        reference='Archived matching q75 recalibration results; frozen R/L27 secondary references',
        only_change='confidence upper quantile .75 to .90 for label and curve; original base replay states and selected layers',
        gate='Both total/thinking token means decrease versus matched q75; accuracy point loss <=2pp; caps do not increase; report paired uncertainty',
        status='Exposed exploratory MATH500, not independent confirmation',process_limit_seconds_per_case=1800,
        stop='Preserve partial outputs on any error or timeout; no automatic retry or expansion',
        estimated_generation_minutes=[15,25],new_calibration_answers=0,new_calibration_forwards=0))
    names=['evaluate.py','grade.py','rows.json','historical_compact.json','summary.json','plan.json']
    names += [str(p.relative_to(HOME)).replace('\\','/') for c in ['R','L27'] for p in (HOME/(c+'_fit')).iterdir()]
    save(HOME/'manifest.json',{'files':{n:sha(HOME/n) for n in names}})
    with tarfile.open(WORK/'package.tar.gz','w:gz') as t:
        for n in names+['manifest.json']:t.add(HOME/n,arcname=n)
    print('Package SHA256',sha(WORK/'package.tar.gz'))

if __name__=='__main__':main()
