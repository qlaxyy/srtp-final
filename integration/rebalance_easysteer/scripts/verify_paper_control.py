"""CPU contract checks for literal paper equations; no model generation."""
import json
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_control import PaperControl, PaperStepState, coefficient, geometric_confidence

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args = parser.parse_args()
torch.set_default_device(args.device)

p = PaperControl(.6, .9, .001, .01, 1., 3., .2, .02, .002)
c = torch.tensor([.2, .6, .9, .99], dtype=torch.float64)
v = torch.tensor([.1, .002, .01, .0001], dtype=torch.float64)
actual = coefficient(c, v, p)
assert actual[0] < 0 and actual[1] < 0 and actual[2] == 0 and actual[3] > 0
equal = PaperControl(.6, .9, .001, .01, 2., 2., 2., .02, .002)
assert torch.allclose(coefficient(c, v, equal), 2*torch.tanh(c-.9))
assert coefficient(torch.tensor(.3), torch.tensor(.1), p) < coefficient(
    torch.tensor(.3), torch.tensor(.002), p)
scaled = PaperControl(.6, .9, .001, .01, 10., 30., 2., .02, .002)
assert torch.allclose(coefficient(c, v, scaled), actual*10)
g = geometric_confidence(torch.log(torch.tensor([.9, .1])).sum(), torch.tensor(2))
assert torch.isclose(g, torch.tensor(.3))
assert not torch.isclose(g, torch.tensor(.5))
grid_c, grid_v = torch.meshgrid(torch.linspace(0, 1, 501),
                               torch.linspace(0, .25, 251), indexing="ij")
grid = coefficient(grid_c, grid_v, p)
assert torch.isfinite(grid).all()
state = PaperStepState(2, device=args.device)
boundary = torch.tensor([10])
def observe(ids, probabilities):
    return state.observe(torch.tensor(ids), torch.tensor(probabilities), boundary, 11, p)
assert torch.equal(observe([1, 1], [.9, .8]), torch.zeros(2))
assert torch.equal(observe([2, 2], [.1, .8]), torch.zeros(2))
assert torch.equal(observe([10, 3], [.99, .8]), torch.zeros(2))
assert torch.isclose(state.previous[0], torch.tensor(.3))
assert torch.isnan(state.previous[1])
assert torch.equal(observe([10, 10], [.99, .99]), torch.zeros(2))
assert torch.isclose(state.previous[0], torch.tensor(.3))
assert torch.isclose(state.previous[1], torch.tensor(.8))
assert (observe([3, 4], [.7, .7]) < 0).all()
assert torch.equal(observe([4, 11], [.7, .7]), torch.zeros(2))
assert torch.equal(observe([11, 5], [.7, .7]), torch.zeros(2))
print(json.dumps(dict(device=args.device, direction_and_zero=True, constant_amplitude_identity=True,
    overthinking_variance_response=True, amplitude_scaling=True,
    geometric_mean=True, finite_grid_points=grid.numel(),
    request_isolation=True, consecutive_boundaries=True,
    first_content_token_only=True, think_end_mask=True)))
