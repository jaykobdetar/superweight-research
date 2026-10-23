# Adam: across-neuron damping and within-neuron axis preference

This is a separate post-06:15 UTC supplement. It does not modify the preceding
six-page training-origin report or its immutable evidence package.

## Result

At native OLMo-1B-0724 step954000, within neuron 1710, the focal residual
coordinate 1764 has a lower Adam second moment than all 128 preselected other
coordinates. Its inverse-denominator gain is 2.08 times the sample median;
its absolute first moment is only 1.29 times the median. There are 50 sampled
coordinates with larger |m|, but only 10 with larger |q| after adaptive scaling.

This complements, rather than contradicts, the earlier exact row comparison.
At fixed residual coordinate 1764, the primary has the largest v and lowest
gain across all 8,192 neurons. Adam can therefore damp this neuron relative to
other neurons while favoring its special output axis within the same neuron.
Neither comparison by itself establishes historical training causality.

Epsilon is small for this within-neuron comparison: epsilon/RMS is 0.0148 for
the focal axis and 0.00706 at the sample median. No sampled axis is epsilon-
dominated. The axis advantage principally reflects the second moment, whereas
epsilon substantially filters other neurons in the earlier row comparison.
An approximately twofold gain advantage is not a denominator explosion.

## Design and uncertainty

The 128 axes were selected uniformly without replacement from the 2,047 other
axes before tensor acquisition. NumPy default_rng seed: 202610230619. Focal
coordinate 1764 was excluded by design. The original protocol, including its
pre-acquisition status, is preserved unchanged; the completed receipt records
the later outcome. Its SHA256 is:

b1b962fcc6cfa9faf4bac0270eb17f4dbf74ddc209220f7ea118104f0c210d7f

All 256 authorized HTTP 206 ranges succeeded. Exactly 1,024 tensor bytes were
retrieved, with no model weights or new checkpoint sampled. Existing verified
optimizer headers were reused. At most eight requests ran concurrently. The
receipt preserves every exact official URL, absolute range and raw four-byte
value as hexadecimal data, with per-slice SHA256.

With zero sampled coordinates having higher gain, the exact two-sided 95%
hypergeometric interval allows zero to 56 higher-gain coordinates among 2,047
(0 to 2.74%). This is finite-population sampling uncertainty, not a measured
full-column rank, uncertainty over training seeds, or a confidence interval on
the optimizer's historical cause. Individual intervals for other metrics are
not adjusted for multiple correlated comparisons.

## Offline reproduction

    python analyze.py
    python audit.py

The analysis uses only Python's standard library. The independent audit uses
SciPy if already installed, otherwise still independently verifies the key
zero-count interval endpoint using a direct finite-product calculation. All
intervals were checked with SciPy in the recorded run. No installation is done.

The renderer requires ReportLab and DejaVu fonts:

    python render_supplement.py

Do not run acquire.py as part of reproduction. That script is a historical,
separately approved acquisition record with a rerun guard; its absolute source
path documents the preceding frozen package. It is not needed for offline
analysis and does not authorize future retries or download expansion.

## Interpretation limits

- This is one late checkpoint and one hidden neuron, not the emergence period.
- m is the saved first moment, not the current task gradient.
- The lower v may be a consequence of a mature circuit, not its origin.
- No matching output-column weights were retrieved, so this supplement does
  not estimate a weight-size/variance correlation across axes.
- No model inference, new gradients, optimizer application or training occurred.
- No whole-column or whole-model rank claim is made.

The next mechanistic hypothesis worth discriminating is whether normalization
suppresses gradient variance along a dominant residual direction and Adam
partly compensates. The present sample is consistent with that possibility,
but does not measure the required backward covariance or historical feedback.
Authentic causal continuation still requires a separate baseline and resource
plan; no such training is included or started here.
