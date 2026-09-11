"""Resolve calibration placement before tokenizer, payload, or model loading."""
from pathlib import Path


def resolve_calibration_layer(fit, model, requested_layer=None):
    """Respect declared placement for every fit version; reject contradictions.

    Legacy/public assets without placement retain the historical decoder18
    default, but their extraction layer is unknown rather than guessed.
    """
    if requested_layer is not None and (
        not isinstance(requested_layer, int) or isinstance(requested_layer, bool)
        or requested_layer < 0
    ):
        raise ValueError("Decoder output layer must be a nonnegative integer")
    fit = fit or {}
    if fit.get("model") is not None:
        if Path(fit["model"]).resolve() != Path(model).resolve():
            raise ValueError("Calibration model mismatch")
    hidden = fit.get("hidden_state_index")
    output = fit.get("decoder_output_layer")
    if hidden is None and output is None:
        return (18 if requested_layer is None else requested_layer), None
    if any(not isinstance(value, int) or isinstance(value, bool)
           for value in (hidden, output)):
        raise ValueError("Calibration must declare both integer layer indices")
    if output < 0 or hidden != output + 1:
        raise ValueError("Calibration hidden-state/output-layer indices disagree")
    if requested_layer is not None and requested_layer != output:
        raise ValueError("Explicit output layer conflicts with calibration")
    return output, hidden
