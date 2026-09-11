"""CPU-only class-prototype reweighting of saved selected-layer states.

Keep original labels, layer, raw-vector convention and curve rules. Each
question containing class k receives mass 1/Q_k; each of its states 1/(Q_k*n_qk).
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def moments(states, questions):
    """Population moments of the question-uniform, then step-uniform mixture."""
    ids = np.unique(questions)
    if not len(ids) or len(states) != len(questions):
        raise ValueError('Empty or mismatched class')
    means, within = [], []
    for q in ids:
        x = np.asarray(states[questions == q], dtype=np.float64)
        means.append(x.mean(0))
        within.append(x.var(0))
    means = np.asarray(means)
    mean = means.mean(0)
    variance = np.mean(within, axis=0) + ((means-mean)**2).mean(0)
    return mean, variance, len(ids)


def geometry(mo, mu, vo, vu, xo):
    d = mo-mu
    w = d/(vo+vu+1e-12)
    projection = float(w@d)
    if not np.isfinite(w).all() or projection <= 0:
        raise ValueError('Degenerate separator')
    threshold = float(.5*w@(mo+mu))
    moderate = float((w@mo-threshold)/projection)
    aggressive = float(np.max((xo@w-threshold)/projection))
    residual = float(np.max(xo@w-aggressive*projection-threshold))
    if not (np.isclose(moderate, .5) and aggressive >= moderate > 0
            and residual <= 1e-8*max(1., abs(threshold))):
        raise ValueError('Separator crossing check failed')
    return d, moderate, aggressive, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Require new candidate output directory')
    # Torch is used only for the existing CPU curve and tensor serialization.
    import torch
    from calibrate_auto import sha, load_module, ROOT, save
    torch.set_num_threads(8)
    started = time.time()
    src, out = args.original, args.output
    read = lambda name: json.loads((src/name).read_text())
    old, protocol, selected = read('fit.json'), read('protocol.json'), read('selected_layer.json')
    if sha(src/'fit.json') != '5bfe9feffc5cc10b494eb881651d863e434520c110e31f8bfa4e1f70778c7a83':
        raise ValueError('Not the fixed original 1.5B fit')
    if sha(src/'auto_vector.pt') != old['vector_sha256']:
        raise ValueError('Original vector hash mismatch')
    if sha(Path(protocol['source'])/'generations.jsonl') != protocol['source_sha256']:
        raise ValueError('Saved calibration answers changed')
    layer = old['hidden_state_index']
    if layer != 21 or selected['best']['layer'] != layer:
        raise ValueError('First candidate fixes hidden state index 21')
    steps, positions = read('steps.json'), read('positions.json')
    q = np.asarray([s['question'] for s in steps])
    if len(steps) != 84008 or len(positions) != 500 or len(np.unique(q)) != 500:
        raise ValueError('Unexpected calibration alignment size')
    if not np.array_equal(q, np.repeat(np.arange(500), [len(p['positions']) for p in positions])):
        raise ValueError('Saved position and step order mismatch')
    for i, p in enumerate(positions):
        if p['question'] != i:
            raise ValueError('Position question order mismatch')
    c = np.asarray([s['confidence'] for s in steps])
    v = np.asarray([s['variance'] for s in steps])
    lexical = np.asarray([s['lexical_hit'] for s in steps], dtype=bool)
    cl, ch = protocol['confidence_quantiles']
    vl, vh = protocol['variance_quantiles']
    np.testing.assert_allclose(np.quantile(c,[.25,.75]), [cl,ch], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.quantile(v,[.25,.75]), [vl,vh], rtol=0, atol=1e-14)
    over, under = lexical | (c < cl), ~lexical & (c > ch)
    if (int(over.sum()),int(under.sum())) != (48251,11309):
        raise ValueError('Frozen labels changed')
    feature = src/f'layer_{layer}.npy'
    x = np.load(feature, mmap_mode='r')
    if x.shape != (84008,1536) or not np.isfinite(x).all():
        raise ValueError('Invalid hidden states')
    xo, xu = x[over].astype(np.float64), x[under].astype(np.float64)
    # Reproduce the old fit before trusting this state/label pairing.
    original_d, original_m, original_a, _ = geometry(xo.mean(0),xu.mean(0),xo.var(0),xu.var(0),xo)
    saved = torch.load(src/'auto_vector.pt', map_location='cpu', weights_only=True)
    torch.testing.assert_close(torch.from_numpy(original_d.astype(np.float32)), saved, rtol=0, atol=0)
    np.testing.assert_allclose([-original_m,-original_a],
        [old['parameters']['low_val_1'],old['parameters']['low_val_2']], rtol=1e-10,atol=1e-12)
    mo, vo, qo = moments(xo,q[over])
    mu, vu, qu = moments(xu,q[under])
    direction, moderate, aggressive, residual = geometry(mo,mu,vo,vu,xo)
    ceiling = moderate*(1-ch)/(ch-cl)
    tau = min(.01,.5*ceiling)
    runtime = load_module('question_balanced_runtime', ROOT/'sources/EasySteer/vllm-steer/vllm/steer_vectors/rebalance.py')
    runtime.validate_curve_targets(cl,ch,-moderate,tau)
    params = dict(q25c=cl,q75c=ch,q25v=vl,q75v=vh,low_val_1=-moderate,
                  low_val_2=-aggressive,high_val_2=.1,initial_coef=-1.,curve_tau=tau)
    control = runtime.ReBalanceParams(boundary_token_ids=(1,),think_start_token_id=2,think_end_token_id=3,**params)
    mid,k,a,b,_,_ = runtime._curve_constants(cl,ch,-moderate,torch.device('cpu'),torch.float64,tau)
    if not 1e-6 < k < 1e6:
        raise ValueError('Curve solver boundary')
    targets = torch.tensor([-moderate,0.,tau],dtype=torch.float64)
    anchors = runtime._baseline(torch.tensor([cl,ch,1.],dtype=torch.float64),mid,k,a,b)
    torch.testing.assert_close(anchors,targets,rtol=1e-6,atol=1e-9)
    gc,gv = torch.meshgrid(torch.linspace(0,1,501),torch.linspace(0,.25,251),indexing='ij')
    surface = runtime.compute_rebalance_coefficient(gc,gv,control)
    if not torch.isfinite(surface).all():
        raise ValueError('Nonfinite coefficient surface')
    before = {name:sha(src/name) for name in ('fit.json','auto_vector.pt','steps.json',
              'positions.json','protocol.json','selected_layer.json','collection.json',feature.name)}
    out.mkdir(parents=True)
    tensor = torch.from_numpy(direction.astype(np.float32))
    torch.save(tensor,out/'auto_vector.pt')
    # version describes the compatible runtime schema; method_variant identifies
    # this distinct experiment and must never replace the frozen original fit.
    fitted = dict(old, parameters=params, vector_sha256=sha(out/'auto_vector.pt'),
        vector_norm=tensor.norm().item(), method_variant='question-balanced-class-prototypes-v1',
        method='Frozen original 1.5B layer and labels; class moments weighted uniformly per nonempty question, then per state; raw difference; same maximum crossing and feasible-curve rules.',
        weighting='1/(Q_k*n_qk)', original_fit_sha256=before['fit.json'])
    save(out/'fit.json',fitted)
    save(out/'curve_check.json',dict(k=k,anchors=anchors.tolist(),targets=targets.tolist(),
         max_residual=float((anchors-targets).abs().max()),grid_points=surface.numel(),finite=True,
         separator_crossing_residual=residual))
    report = dict(status='completed_cpu_fit',questions=500,steps=len(steps),class_questions=dict(over=qo,under=qu),
        original_vector_reproduced_bit_exact=True,original_low2_reproduced=original_a,
        vector_cosine=float(original_d@direction/(np.linalg.norm(original_d)*np.linalg.norm(direction))),
        original_vector_norm=old['vector_norm'],candidate_vector_norm=fitted['vector_norm'],
        original_parameters=old['parameters'],candidate_parameters=params,source_hashes=before,
        candidate_fit_sha256=sha(out/'fit.json'),candidate_vector_sha256=sha(out/'auto_vector.pt'),
        seconds=time.time()-started,model_forwards=0,new_generations=0)
    if before != {name:sha(src/name) for name in before}:
        raise ValueError('Original assets changed during fit')
    save(out/'fit_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
