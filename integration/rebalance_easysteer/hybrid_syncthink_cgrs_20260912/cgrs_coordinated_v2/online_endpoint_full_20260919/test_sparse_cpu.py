"""Validate chunk filtering against real dense traces; create hashed GPU references."""
import sys,io,json,tarfile,hashlib
from pathlib import Path
import numpy as np
from prepare import HOME,HERE,ROOT,save
sys.path[:0]=[str(HERE/'online_endpoint_20260919'),str(HERE)]
from collector import SparseRows
def main():
    tok=json.loads((ROOT/'.codex_work/wsc_weights_20260917/tokenizer.json').read_text(encoding='utf-8'))
    bounds={i for piece,i in tok['model']['vocab'].items() if 'ĊĊ' in piece}
    archive=ROOT/'.codex_work/online_endpoint_20260919/evidence.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='387fa8508f550e0df2e4d879c2906e710a263aed3b92fa51ccecda9378e2ae00'
    (HOME/'reference').mkdir(exist_ok=False);refs={};total=0
    with tarfile.open(archive) as t:
      for arm in ['R','L27']:
        prefix='online_endpoint_20260919_engineering_run3/'+arm+'_observe/'
        refs[arm]=json.load(t.extractfile(prefix+'result.json'))['records']
        for i in range(8):
          z=np.load(io.BytesIO(t.extractfile(prefix+str(i)+'_hidden.npz').read()));ts=refs[arm][i]['token_ids'];p=int(z['prompt_tokens'])
          args=[np.zeros(512,dtype=int),z['positions'],z['input_ids'],np.array(ts),z['pre_hidden'],z['post_hidden']]
          a=SparseRows(bounds);b=SparseRows(bounds)
          a.add(*args)
          for start in range(0,512,128):b.add(*[x[start:start+128] for x in args])
          assert a.metadata==b.metadata
          expected=[j for j in range(1,512) if (j==1 or z['input_ids'][j-1] in bounds) and z['input_ids'][j] not in bounds and z['input_ids'][j]!=151649]
          got=b.hidden[0];assert [x[0] for x in got]==z['positions'][expected].tolist()
          pre=np.stack([x[1] for x in got]);post=np.stack([x[2] for x in got])
          assert np.array_equal(pre,z['pre_hidden'][expected]) and np.array_equal(post,z['post_hidden'][expected])
          np.savez(HOME/'reference'/f'{arm}_{i}.npz',starts=np.array([x[0]-p for x in got]),pre_hidden=pre,post_hidden=post)
          total+=len(got)
    save(HOME/'sparse_reference.json',refs)
    save(HOME/'cpu_checks.json',dict(real_dense_traces=16,token_positions=8192,retained_step_starts=total,chunk_size=128,exact=True,GPU_sparse_gate=False))
    print('PASS: 16 dense traces / 8192 positions; retained',total,'step starts exactly')
if __name__=='__main__':main()
