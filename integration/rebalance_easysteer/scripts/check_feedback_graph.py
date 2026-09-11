"""Bounded compiled-kernel and CUDA-graph check; no model or answer generation."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
from mechanism_candidates import ROOT, require, save, sha, read, read_vector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Result exists')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU task')
    config=read(a.assets/'feedback.json')
    require(sha(a.assets/config['readout_file'])==config['readout_sha256'],'Readout changed')
    require(sha(a.assets/'auto_vector.pt')==config['vector_sha256'],'Direction changed')
    import torch
    from vllm.steer_vectors.graph_kernels import apply_decoder_families
    from vllm.steer_vectors.controllers import DecoderSteerController
    from vllm.steer_vectors.payloads import FeedbackDirection,materialize
    torch.set_num_threads(4);started=time.perf_counter();torch.manual_seed(20260912)
    payload=FeedbackDirection(read_vector(a.assets/'auto_vector.pt',a.assets/'auto_vector.pt'),
        np.load(a.assets/config['readout_file']),config['negative_centroid_score'],layer=20)
    data=materialize(payload.to_wire(),'cuda',torch.bfloat16,[20])[20]
    n=128;rows=torch.ones(n,dtype=torch.long,device='cuda');controller=DecoderSteerController()
    controller.init_graph_table(1,1536,torch.bfloat16,torch.device('cuda'),n,rows,1,frozenset({'feedback'}))
    controller.set_graph_row(1,'rebalance_feedback',data,1.)
    hidden=torch.randn(n,1536,device='cuda',dtype=torch.bfloat16)*4
    residual=torch.randn_like(hidden)*3
    mask=controller.graph_mask;mask.copy_(torch.linspace(-1.5,.1,n,device='cuda').bfloat16())
    args=(mask,controller.replace_mask,controller.normalize_flag,rows,hidden,residual)
    feedback=controller.graph_tables['feedback'];additive={'additive':{'V':feedback['V']}}
    compiler=torch.compile(apply_decoder_families,fullgraph=True)
    feedback['E'].zero_()
    with torch.inference_mode():
        expected=compiler(additive,*args).clone()
        disabled=compiler(controller.graph_tables,*args).clone()
        require(torch.equal(disabled,expected),'Compiled disabled feedback differs from original graph')
        feedback['E'][1].fill_(1)
        eager=apply_decoder_families(controller.graph_tables,*args).clone()
        compiled=compiler(controller.graph_tables,*args).clone()
        error=(eager.float()-compiled.float()).abs()
        rounding_bound=(hidden.float().abs()+(compiled.float()-hidden.float()).abs()+1)*2**-7
        require(torch.all(error<=rounding_bound),'Compiled feedback exceeds fixed BF16 rounding envelope')
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=compiler(controller.graph_tables,*args)
        feedback['stats'].zero_();graph.replay();torch.cuda.synchronize()
        require(torch.equal(captured,compiled),'CUDA graph replay changed output')
        stats1=feedback['stats'].clone();require(stats1[1,1]>0,'Synthetic check did not exercise clipping')
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(feedback['stats'],stats1*2)
        mask.zero_();graph.replay();torch.cuda.synchronize()
        require(torch.equal(captured,hidden),'Zero-mask graph replay is not identity')
        mask.fill_(.1);graph.replay();torch.cuda.synchronize()
        require(torch.equal(captured,compiler(additive,*args)),'Positive graph replay differs')
        mask.fill_(-1);feedback['E'].zero_();graph.replay();torch.cuda.synchronize()
        require(torch.equal(captured,compiler(additive,*args)),'Disabled replay with changed mask differs')
    result=dict(status='compiled_and_cudagraph_checks_passed',seconds=time.perf_counter()-started,
        scope='128 synthetic hidden+residual rows,1536 dimensions,BF16 activation and FP32 readout. No model loaded; no efficacy claim.',
        actual_clipping_counts=stats1.sum(0).cpu().tolist(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        compiled_eager_max_absolute_error=float(error.max()),
        compiled_eager_max_fraction_of_rounding_envelope=float((error/rounding_bound).max()),
        source_sha256=sha(Path(__file__),source=True),asset_sha256=sha(a.assets/'feedback.json'))
    save(a.output,result);print(result)


if __name__=='__main__':main()
