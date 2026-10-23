"""Value-only operators. P is fixed externally; no Q/K modification occurs here."""
import numpy as np

def flatten_heads(values):
    return values.transpose(1,0,2).reshape(values.shape[1],-1)

def replay_context(probs, values):
    return flatten_heads(probs @ values)

def replacement_delta(probs, values, source, donor_value, query_start):
    delta = probs[:,:,source,None] * (donor_value - values[:,source,:])[:,None,:]
    delta[:,:query_start,:] = 0
    return flatten_heads(delta)

def match_output_norm(delta, reference_delta, project):
    """Match each query's aggregate post-W_O norm, never separate head/raw-V norms.

    Scalar rescaling in context space commutes with W_O. Zero direction is not
    invented: an impossible match raises rather than silently proceeding.
    """
    out = project(delta)
    ref = project(reference_delta)
    norm = np.linalg.norm(out.astype(np.float64),axis=-1)
    target = np.linalg.norm(ref.astype(np.float64),axis=-1)
    impossible = (norm <= 1e-12) & (target > 1e-10)
    if np.any(impossible): raise ValueError('Cannot norm-match zero control direction to a nonzero reference.')
    scale = np.divide(target,norm,out=np.zeros_like(target),where=norm>1e-12)
    result = delta * scale.astype(delta.dtype)[:,None]
    achieved = np.linalg.norm(project(result).astype(np.float64),axis=-1)
    error = float(np.max(np.abs(achieved-target)))
    tolerance = 1e-4 + 1e-5*float(np.max(target))
    if error > tolerance: raise AssertionError(('Post-W_O norm mismatch', error, tolerance))
    return result, dict(post_wo_norm_max_abs_error=error,
                        target_post_wo_norm_max=float(np.max(target)),
                        scale_max=float(np.max(scale)))
