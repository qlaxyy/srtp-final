"""Check a fixed radial kernel and mutable CUDA graph, without model loading."""
import argparse
import subprocess
import time
from pathlib import Path
from mechanism_candidates import require, read_vector, save, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vector',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Immutable check exists')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU task')
    import torch
    from vllm.steer_vectors.graph_kernels import apply_decoder_families
    from vllm.steer_vectors.controllers import DecoderSteerController
    started=time.perf_counter();torch.manual_seed(20260922)
    rows=torch.ones(128,dtype=torch.long,device='cuda')
    controller=DecoderSteerController()
    controller.init_graph_table(1,1536,torch.bfloat16,torch.device('cuda'),128,rows,1,frozenset({'radial'}))
    direction=torch.tensor(read_vector(args.vector,args.vector),device='cuda',dtype=torch.bfloat16)
    controller.set_graph_row(1,'rebalance_radial',direction,1.)
    table=controller.graph_tables['radial']
    h=torch.randn(128,1536,device='cuda',dtype=torch.bfloat16)*4
    residual=torch.randn_like(h)*3
    mask=controller.graph_mask
    mask.copy_(torch.linspace(-1.5,.1,128,device='cuda').bfloat16())
    inputs=(mask,controller.replace_mask,controller.normalize_flag,rows,h,residual)
    compiler=torch.compile(apply_decoder_families,fullgraph=True)
    with torch.inference_mode():
        eager=apply_decoder_families(controller.graph_tables,*inputs).clone()
        compiled=compiler(controller.graph_tables,*inputs).clone()
        error=(eager.float()-compiled.float()).abs()
        envelope=(h.float().abs()+(compiled.float()-h.float()).abs()+1)/128
        require(bool(torch.all(error<=envelope)),'Compiled/eager outside fixed BF16 envelope')
        complete=(h+residual).float()
        actual=(compiled+residual).float()
        norm_error=abs(actual.norm(dim=-1)/complete.norm(dim=-1)-1)
        require(float(norm_error.max())<.005,'Restoration exceeds fixed BF16 norm tolerance')
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=compiler(controller.graph_tables,*inputs)
        graph.replay();torch.cuda.synchronize()
        require(torch.equal(captured,compiled),'Initial capture mismatch')
        table['E'].zero_();graph.replay();torch.cuda.synchronize()
        disabled=captured.clone()
        require(torch.equal(disabled,compiler(controller.graph_tables,*inputs)),'Mutable disabled table mismatch')
        original=compiler({'additive':{'V':table['V']}},*inputs).clone()
        disabled_original_differences=int((disabled!=original).sum())
        require(bool(torch.any(disabled!=compiled)),'No radial change in kernel check')
        for value in [0.,.1,-1.5]:
            mask.fill_(value);table['E'][1].fill_(1)
            graph.replay();torch.cuda.synchronize()
            require(torch.equal(captured,compiler(controller.graph_tables,*inputs)),'Changed-mask graph mismatch')
            if value==0:require(torch.equal(captured,h),'Zero-mask identity failed')
        controller.clear_graph_row(1);graph.replay();torch.cuda.synchronize()
        require(torch.equal(captured,h),'Cleared row identity failed')
    result=dict(status='compiled_and_cudagraph_checks_passed',
        seconds=time.perf_counter()-started,synthetic_rows=128,model_loads=0,new_answers=0,
        max_compiled_eager_error=float(error.max()),
        max_error_fraction_of_envelope=float((error/envelope).max()),
        max_complete_state_relative_norm_error=float(norm_error.max()),
        disabled_vs_original_differing_elements=disabled_original_differences,
        difference_policy='Recorded numerical control difference, not efficacy and not an identity failure; separate original and matched-off arms are mandatory.',
        vector_sha256=sha(args.vector),script_sha256=sha(Path(__file__),source=True))
    save(args.output,result);print(result)


if __name__=='__main__':main()
