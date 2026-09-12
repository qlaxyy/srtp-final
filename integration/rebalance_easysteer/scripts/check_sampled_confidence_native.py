"""Short real Torch and request-serialization checks in the existing environment."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import time
from mechanism_candidates import ROOT, require, save, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Immutable native receipt exists')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True).strip(),'Other GPU task')
    import torch
    import vllm
    require(torch.cuda.is_available(),'GPU not available; do not reinstall')
    require(Path(vllm.__file__).resolve()==(ROOT/'sources/EasySteer/vllm-steer/vllm/__init__.py').resolve(),
            'Different vLLM checkout')
    source=ROOT/'sources/EasySteer/vllm-steer/tests/steer_vectors/test_rebalance.py'
    spec=importlib.util.spec_from_file_location('local_prepared_native_checks',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=['test_sampled_confidence_owns_raw_logits_before_sampler_mutation',
        'test_sampled_confidence_survives_request_serialization_and_rejects_mixing',
        'test_sampled_confidence_is_request_local_across_reordering_and_replay',
        'test_rebalance_state_is_request_local_and_uses_arithmetic_mean',
        'test_rebalance_replays_past_scales_after_think_end_and_slot_reuse',
        'test_rebalance_reordering_think_end_and_slot_reuse']
    started=time.perf_counter();results=[]
    for name in names:
        begin=time.perf_counter();getattr(module,name)()
        results.append(dict(name=name,seconds=time.perf_counter()-begin))
    torch.cuda.synchronize()
    result=dict(status='native_torch_and_request_checks_passed',tests=results,
        seconds=time.perf_counter()-started,torch=torch.__version__,vllm=vllm.__version__,
        cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
        test_source_sha256=sha(source,source=True),checker_sha256=sha(Path(__file__),source=True),
        model_loads=0,new_answers=0,
        limits='Real Torch CPU/CUDA numerical checks, request serialization and controller lifecycle only. No throughput or compression benefit measured. The candidate shares the existing additive graph; full generation follows only the frozen engineering and opportunity gates.')
    save(args.output,result);print(result)


if __name__=='__main__':main()
