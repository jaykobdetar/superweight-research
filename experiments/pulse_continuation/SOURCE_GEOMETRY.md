# Axis alignment versus complementary direction

Motivation: row deletion leaves the neuron's small off-axis output present,
while whole-column deletion removes it. Test whether the resulting downstream
source-normalized geometry explains their different effects. This is not a claim
that normalization or constant source vectors are new discoveries.

Use16natural source0 windows from upstream_run and12eligible unedited first-newline
sources from newline_natural. No fitting. Native donors C=clean,R=row-deleted,
N=whole-column-deleted. At each layer2..15 attention input normalization, capture
the source vector z. In FP32 center z, take u=(e1764-1/2048)/sqrt(1-1/2048), radius
r=||center(z)||, a=<center(z),u>/r, and v=(center(z)/r-a*u)/sqrt(1-a²).

In row-deleted recipient, for each of9 donor pairs(A,B), replace source norm by
mean(z_A)+r_A*(a_A*u+sqrt(1-a_A²)*v_B), castFP16. This preserves donorA's axis
alignment and radius while choosing donorB's orthogonal direction. Native
The implementation uses the measured orthogonal residual norm s instead of
sqrt(1-a²) to reduce cancellation; it is mathematically equivalent before
rounding. Same-donor reconstructions quantify numerical drift; separate direct full donor
patches must reproduce donor later-token outputs exactly. This all-layer source
clamp is an expected architectural closure, not proof of a unique biological
mechanism. Directions vary by layer and window; this is a conditional causal
decomposition, not a deployable fitted replacement.

Controls: three independently seeded centered directions orthogonal to u,
using C orN alignment (sixconditions), plus negative-u axis alone with C radius.
Same random vectors across cases, chosen before results; seeds20261021..23.
22forwards/case=616. NativeC/R/N baseline references all scored identically;
source0 score_start1, natural first-newline score_startp+1. Frozen donor replay
must match losses/predictions aftersource; own reconstruction drift is reported.
Source changes cannot affect earlier positions. Store donor norm/raw vectors,
all per-token outcomes, geometry scalars, and original model hashes. Smalln and
reused panels mean exploratory intervals; no universal low-dimensional claim.
