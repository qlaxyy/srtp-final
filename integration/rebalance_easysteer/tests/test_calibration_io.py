import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import numpy as np
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root/'integration/rebalance_easysteer/scripts'))
import calibrate_auto as auto
import calibrate_own_vector as own
fake_t = ModuleType('transformers')
fake_t.AutoTokenizer = SimpleNamespace(from_pretrained=lambda _: SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[-1]['content']))
fake_v = ModuleType('vllm')
fake_v.SamplingParams = lambda **kw: kw
class LLM:
    fail = False
    def __init__(self, **kw): self.llm_engine=self; self.done=False; self.round=0
    def enqueue(self, prompts, params):
        self.prompts=prompts
        assert len(prompts)==500 and params['temperature']==0 and params['max_tokens']==16000
        return [str(i) for i in range(500)]
    def has_unfinished_requests(self): return not self.done
    def step(self):
        self.round+=1
        if self.fail and self.round==2: raise RuntimeError('simulated interrupt')
        if self.round==1:
            return [SimpleNamespace(finished=False)] + [self.output(i) for i in range(490,500)]
        self.done=True
        return [self.output(i) for i in reversed(range(490))]
    def output(self, i):
        seq=SimpleNamespace(token_ids=[10,11], text=self.prompts[i], logprobs=[{10:SimpleNamespace(logprob=-.2)},{11:SimpleNamespace(logprob=-.3)}], finish_reason='length' if i==2 else 'stop')
        return SimpleNamespace(finished=True, request_id=str(i), outputs=[seq], prompt_token_ids=[7])
fake_v.LLM=LLM

def test_generation_checkpoint():
    LLM.fail = False
    with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'transformers':fake_t,'vllm':fake_v}):
        out=Path(d)/'complete'; out.mkdir()
        own.generate(out, '/fake/model')
        rows=own.read(out/'generations.jsonl'); manifest=json.loads((out/'manifest.json').read_text())
        assert [r['train_index'] for r in rows]==manifest['train_indices']
        assert len(own.read(out/'generations.partial.jsonl'))==500
        summary=json.loads((out/'generation_summary.json').read_text())
        assert summary['count']==500 and summary['mean_tokens']==2 and summary['capped']==1
        assert rows[0]['logprobs']==[-.2,-.3]
        LLM.fail=True; interrupted=Path(d)/'interrupted'; interrupted.mkdir()
        try: own.generate(interrupted, '/fake/model')
        except RuntimeError as e: assert str(e)=='simulated interrupt'
        else: raise AssertionError('Expected interruption')
        assert len(own.read(interrupted/'generations.partial.jsonl'))==10
        assert not (interrupted/'generation_summary.json').exists()
        try: own.generate(interrupted, '/fake/model')
        except FileExistsError: pass
        else: raise AssertionError('Must not rerun saved requests')
    print('PASS checkpoint: completion order, logprobs, partial-output retention, repeat refusal')

def test_feature_storage_equivalence():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); rng=np.random.default_rng(42)
        c=np.tile(np.linspace(.3,.999,500),2)
        steps=[dict(question=i%500,confidence=float(v),lexical_hit=(i%11==0)) for i,v in enumerate(c)]
        arrays={1:rng.normal(size=(1000,4)).astype('float32'),2:np.stack([c,c*c,c*3,c*.5],1).astype('float32')}
        for mode in ('legacy','relocated'):
            out=root/mode; out.mkdir(); features=out if mode=='legacy' else root/'scratch'; features.mkdir(exist_ok=True)
            for layer,x in arrays.items(): np.save(features/f'layer_{layer}.npy',x)
            config=dict(layer_ids=[1,2])
            if mode=='relocated': config['feature_dir']=str(features)
            own.save(out/'collection.json',config); own.save(out/'steps.json',steps)
            own.save(out/'protocol.json',dict(confidence_quantiles=[.55,.9],variance_quantiles=[.001,.01],source_sha256='fixture'))
            args=SimpleNamespace(output=out,model=Path('/fake/model'))
            auto.select(args); auto.fit(args)
            assert json.loads((out/'selected_layer.json').read_text())['best']['layer']==2
        a=json.loads((root/'legacy/fit.json').read_text()); b=json.loads((root/'relocated/fit.json').read_text())
        assert a==b
        assert (root/'legacy/auto_vector.pt').read_bytes()==(root/'relocated/auto_vector.pt').read_bytes()
    print('PASS storage: identical layer, scores, parameters and vector bytes with external scratch')

if __name__ == "__main__":
    test_generation_checkpoint()
    test_feature_storage_equivalence()
