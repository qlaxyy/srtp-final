"""Native observer replay on CPU with independent scalar harmonic oracle."""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import torch
from harmonic_adapter import HarmonicAdapter
from test_alignment_device_cpu import owner
from prepare_label_alignment import load

ROOT=Path(__file__).resolve().parents[4]
RUNTIME=ROOT/'sources/EasySteer/vllm-steer/vllm'
rt=load('harmonic_test_runtime',RUNTIME/'steer_vectors/rebalance.py')
tree=ast.parse((RUNTIME/'v1/worker/gpu/steer_vector_utils.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SteerVectorState')
ns=dict(torch=torch,np=np,ReBalanceParams=rt.ReBalanceParams,compute_rebalance_coefficient=rt.compute_rebalance_coefficient)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])),'native-observer','exec'),ns)

class Tests(unittest.TestCase):
    def test_off(self):
        llm=N();a=HarmonicAdapter(llm,None);a.close();self.assertEqual(vars(llm),{})

    def test_native_accumulator_and_history(self):
        a=HarmonicAdapter.__new__(HarmonicAdapter);a.__dict__.update(vars(owner()))
        s=ns['SteerVectorState'](3,torch.device('cpu'),200);a.rstate=s;a.runner.steer_vector_state=s
        p=rt.ReBalanceParams(boundary_token_ids=(271,),think_start_token_id=151648,think_end_token_id=151649,
            q25c=.3,q75c=.9,q25v=.001,q75v=.02,low_val_1=-.5,low_val_2=-1.5,curve_tau=.01)
        s._dynamic_params={'a':p,'b':p};s._dynamic_indices={'a':2,'b':0};s._in_think[:]=True
        a.hready=torch.zeros(3,dtype=torch.bool);a.hprevious=torch.full((3,),float('nan'));a.hupdates=torch.zeros(3,dtype=torch.long)
        a.compute_coefficient=rt.compute_rebalance_coefficient;a.original_observe=s.observe_sample;a.original_record=s._record_scales
        s._record_scales=a.record_harmonic
        batch=N(num_reqs=2,num_draft_tokens=0,req_ids=['a','b'],idx_mapping=torch.tensor([2,0]),
            num_computed_tokens_np=np.array([10,10]),num_scheduled_tokens=np.array([1,1]),prefill_len_np=np.array([10,10]),seq_lens=torch.tensor([11,11]))
        # Consecutive empty boundary, ordinary words, end-think, invalid prefill,
        # and reordered slots all exercise the exact native count convention.
        streams=[([271,1],[.9,.5],[True,True]),([1,1],[.5,.8],[True,True]),
            ([1,271],[1.,.7],[True,True]),([271,271],[.2,.8],[True,True]),
            ([271,1],[.4,.1],[True,False]),([151649,1],[.7,.9],[True,True]),([271,271],[.8,.5],[True,True])]
        pending={0:[],2:[]};prev={0:None,2:None};updates={0:0,2:0}
        for k,(tokens,probs,valid) in enumerate(streams):
            batch.prefill_len_np=np.array([10 if x else 99 for x in valid]);batch.seq_lens[:]=11+k
            ids=batch.idx_mapping;ts=torch.tensor(tokens);ps=torch.tensor(probs)
            a.accept_lexical_sample(ids,ts,torch.tensor(valid))
            a.observe_sample(batch,ts[:,None],ps)
            for j,i in enumerate(ids.tolist()):
                if not valid[j]:continue
                if tokens[j]!=271:pending[i].append(probs[j])
                elif pending[i]:
                    c=len(pending[i])/sum(1/x for x in pending[i]);v=0 if prev[i] is None else (c-prev[i])**2/4
                    expected=rt.compute_rebalance_coefficient(torch.tensor([c]),torch.tensor([v]),p)[0]
                    self.assertAlmostEqual(float(s._prev_step_mean[i]),c,places=6)
                    self.assertAlmostEqual(float(s._coefs[i]),float(expected),places=6)
                    self.assertAlmostEqual(float(s._history[i,11+k]),float(expected*s._in_think[i]),places=6)
                    prev[i]=c;pending[i]=[];updates[i]+=1
                self.assertEqual(int(s._step_tok_count[i]),len(pending[i]))
                self.assertAlmostEqual(float(s._step_prob_sum[i]),sum(1/x for x in pending[i]),places=5)
        self.assertEqual(a.hupdates.tolist(),[updates[0],0,updates[2]])

if __name__=='__main__':unittest.main()
