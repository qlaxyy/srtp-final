"""Capture the first failed synthetic check once, without continuing the batch."""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import time
from mechanism_candidates import ROOT, BASE, read, require, save, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(not out.exists(),'Immutable diagnostic output exists')
    plan_path=ROOT/BASE/'configs/radial_numeric_diagnostic_20260912.json'
    plan=read(plan_path)
    historical=subprocess.check_output(['git','show',plan['failed_commit']+':'+BASE+'scripts/check_radial_graph.py'],cwd=ROOT).decode()
    kernel_path='sources/EasySteer/vllm-steer/vllm/steer_vectors/graph_kernels.py'
    original=subprocess.check_output(['git','show',plan['failed_commit']+':'+kernel_path],cwd=ROOT)
    require(sha(ROOT/kernel_path,source=True)==hashlib.sha256(original).hexdigest(),'Kernel changed before diagnostic')
    tree=ast.parse(historical)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    context=next(n for n in function.body if isinstance(n,ast.With))
    stop=next(i for i,n in enumerate(context.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='require')
    context.body=context.body[:stop]+ast.parse('capture(locals())\nreturn').body
    function.body=function.body[:function.body.index(context)+1]
    tree.body=[n for n in tree.body if not isinstance(n,ast.If)]
    ast.fix_missing_locations(tree)
    out.mkdir(parents=True)
    started=time.perf_counter()
    def capture(values):
        import numpy as np
        import torch
        torch.cuda.synchronize()
        np.savez_compressed(out/'actual_arrays.npz',**{
            name:values[name].float().cpu().numpy() for name in
            ['h','residual','direction','mask','eager','compiled','error','envelope']})
        save(out/'receipt.json',dict(status='diagnostic_capture_not_a_passing_engineering_check',
            seconds=time.perf_counter()-started,violations=int((values['error']>values['envelope']).sum()),
            max_error=float(values['error'].max()),
            max_envelope_fraction=float((values['error']/values['envelope']).max()),
            arrays_sha256=sha(out/'actual_arrays.npz'),plan_sha256=sha(plan_path,source=True),
            historical_check_sha256=hashlib.sha256(historical.encode()).hexdigest(),
            source_kernel_sha256=sha(ROOT/kernel_path,source=True),model_loads=0,new_answers=0))
    namespace={'__name__':'isolated_historical_check','__file__':str(ROOT/BASE/'scripts/check_radial_graph.py'),'capture':capture}
    import sys
    sys.argv=['historical_check','--output',str(out/'unused.json'),'--vector',str(ROOT/BASE/'configs/radial_screen100_20260912/assets/original_dynamic/auto_vector.pt')]
    exec(compile(tree,'historical_radial_check_capture_only','exec'),namespace)
    namespace['main']()
    print(read(out/'receipt.json'))


if __name__=='__main__':main()
