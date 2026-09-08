"""Freeze the paper reconstruction conventions before evaluating benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.calibration
    protocol = json.loads((root / "protocol.json").read_text())
    geometry = json.loads((root / "geometry.json").read_text())
    selected = json.loads((root / "selected_layer.json").read_text())
    layer = geometry["layer"]
    layers = json.loads((root / "collection.json").read_text())["layers"]
    if not 0 < layer < layers - 1:
        raise ValueError("Embedding/final norm need a separate injection mapping")
    assert selected["best"]["layer"] == layer
    sha = hashlib.sha256((root / "paper_unit_vector.pt").read_bytes()).hexdigest()
    assert sha == geometry["vector_sha256"]
    cl, ch = protocol["confidence_quantiles"]
    vl, vh = protocol["variance_quantiles"]
    distance = geometry["prototype_distance"]
    moderate_under, aggressive_under = .01 * distance, .1 * distance
    bm = max(geometry["moderate_overthinking"], moderate_under)
    result = dict(
        version="paper-reconstruction-v1",
        model=protocol["model"], hidden_state_index=layer,
        decoder_output_layer=layer-1, vector_sha256=sha,
        parameters=dict(q25c=cl, q75c=ch, q25v=vl, q75v=vh, initial_coef=0.,
            paper_parameters=[bm, geometry["aggressive_overthinking"],
                              aggressive_under, (ch-cl)/4, (vh-vl)/4]),
        parameter_order=["Bm", "Bo", "Bu", "eta_c", "eta_v"],
        conventions={
            "Bm": "max(d_O_m, d_U_m); direct moderate-target amplitude mapping",
            "Bo": "d_O_a (paper)", "Bu": "rho_a * d_prot (paper form)",
            "rho_m": .01, "rho_a": .1,
            "eta_c": "confidence IQR / 4", "eta_v": "variance IQR / 4",
            "f_to_B": "Use moderate target distances directly for Bm; no separate "
                        "tanh f fit because its mapping to final B is unspecified",
            "online": "Literal Appendix B.3 final smooth g, geometric max-probability "
                      "mean, adjacent population variance, exclude boundary tokens",
            "injection": "First content token of next step only; initial step zero",
            "calibration": "Reuse 500 seed42 greedy MATH-train answers; cap16000; "
                           "exclude incomplete final steps; author step-level Ridge split",
            "status": "Explicit reconstruction choices, not unique paper reproduction; "
                      "fixed before benchmark outcomes, no test-set parameter search",
        },
        calibration_protocol=protocol, geometry=geometry, selection=selected["best"],
        evaluation=dict(temperature=.7, top_p=.95, seed=42, max_tokens=16000,
                        max_model_len=32768, datasets={"MATH-500":500, "GSM8K":1319}),
    )
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(result["parameters"]))


if __name__ == "__main__":
    main()
