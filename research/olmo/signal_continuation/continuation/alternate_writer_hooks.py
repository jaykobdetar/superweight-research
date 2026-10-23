"""Exact source-local L1 interventions for a streamed NumPy OLMo forward.

Inputs are [sequence, width], batch size one. Activations are POST-SwiGLU.
All treatments modify the down-projection output at selected positions only.
This module never changes weights, input arrays, or nonselected positions.
"""
from dataclasses import dataclass
import numpy as np

D, N, PRIMARY, COORDINATE = 2048, 8192, 1710, 1764
SOURCE_NEGATIVE_GROUP = (7579, 7845, 6947, 2637)
NEWLINE_POSITIVE_GROUP = (2476, 2637, 7845)
H14_QUERY_GROUP = (3950, 4786)


@dataclass(frozen=True)
class Intervention:
    name: str
    positions: tuple = (0,)
    drop: tuple = ()
    keep: tuple | None = None
    partial_group: tuple = ()
    remove_part: str | None = None
    addback: tuple = ()

    def __post_init__(self):
        assert self.remove_part in (None, "axis", "complement")
        assert (self.remove_part is None) == (not self.partial_group)
        assert self.keep is None or not self.drop
        assert not set(self.drop).intersection(self.partial_group)
        assert set(self.addback).issubset(self.drop)
        for group in (self.drop, self.keep or (), self.partial_group, self.addback):
            assert len(set(group)) == len(group)
            assert all(0 <= j < N for j in group)
        assert len(set(self.positions)) == len(self.positions)
        assert all(p >= 0 for p in self.positions)


def make_hook(down, intervention):
    """Return callback (layer, normalized_input, activations, output) -> output.

    `down` must have shape [2048,8192], matching stored HF weight orientation.
    Whole-column removal uses a fresh dot product, avoiding cancellation against
    the much larger primary output. The axis/complement treatments are explicitly
    vector-component interventions, not neuron ablations. Their sum is the raw
    group write, including its DC component in the complement.
    """
    assert down.shape == (D, N)
    axis = np.full(D, -1 / D, dtype=np.float64)
    axis[COORDINATE] += 1
    axis /= np.linalg.norm(axis)
    spec = intervention

    def group_write(a, ids):
        return a[:, list(ids)].astype(np.float64) @ np.asarray(
            down[:, list(ids)], dtype=np.float64).T

    def hook(layer_idx, normalized_input, activations, native_output):
        if layer_idx != 1:
            return native_output
        assert activations.ndim == native_output.ndim == normalized_input.ndim == 2
        assert activations.shape == (len(native_output), N)
        assert normalized_input.shape == native_output.shape == (len(native_output), D)
        assert np.issubdtype(native_output.dtype, np.floating)
        positions = np.asarray(spec.positions, dtype=int)
        assert np.all(positions < len(native_output))
        a = activations[positions].copy()
        native_a = a.copy()
        if spec.keep is not None:
            a.fill(0)
            a[:, list(spec.keep)] = native_a[:, list(spec.keep)]
        elif spec.drop:
            a[:, list(spec.drop)] = 0
        # FP32 arithmetic follows the caller's activation dtype for this oracle.
        selected = a @ down.T
        if spec.partial_group:
            group = group_write(native_a, spec.partial_group)
            axial = (group @ axis)[:, None] * axis[None, :]
            component = axial if spec.remove_part == "axis" else group - axial
            selected = selected.astype(np.float64) - component
        if spec.addback:
            selected = selected.astype(np.float64) + group_write(native_a, spec.addback)
        out = native_output.copy()
        out[positions] = selected
        return out

    hook.intervention = spec
    return hook


def group_factorial(group, positions=(0,), prefix="group"):
    """Fixed factorial, identity/reconstruction and mechanism controls."""
    group = tuple(group)
    return (
        Intervention("reconstruct_native", positions),
        Intervention("drop1710", positions, drop=(PRIMARY,)),
        Intervention(prefix + "_drop_only", positions, drop=group),
        Intervention(prefix + "_drop_with1710", positions, drop=(PRIMARY,) + group),
        Intervention(prefix + "_same_world_addback", positions,
                     drop=(PRIMARY,) + group, addback=group),
        Intervention(prefix + "_axis_remove_with1710", positions,
                     drop=(PRIMARY,), partial_group=group, remove_part="axis"),
        Intervention(prefix + "_complement_remove_with1710", positions,
                     drop=(PRIMARY,), partial_group=group, remove_part="complement"),
        Intervention(prefix + "_only", positions, keep=group),
        Intervention("carry_only", positions, keep=()),
    )
