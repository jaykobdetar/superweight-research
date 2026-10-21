# Living hypothesis ledger — super-weight pulse research

Updated 21 October 2026 after report5; investigation continues. Layer/head indices
are zero-based. All claims below concern the pinned OLMo-1B-0724-hf checkpoint,
FP16 SDPA and stated text panels. Earlier ledger snapshots remain in archives.

## Established within tested interventions

1. **One-coordinate deletion is catastrophic conditionally on the rest of its
   neuron staying intact.** L1 down_proj[1764,1710] at source0 can be deleted to
   produce hundreds/thousands-fold PPL increases. Original runs were replicated
   exactly. This does NOT establish equal necessity of the whole producing neuron.
2. **Whole-column deletion is far less damaging.** In four-window natural-ID
   panels, row/whole-column ratios are Wiki1933.92/1.44506, DuBois537.43/1.73149,
   Hobbes648.00/1.03320, optimizer596.21/1.01114. Whole-column removal still has
   material errors (Wiki/DuBois agreement76.47/78.19%). Component2x2 interventions
   and exact source-deletion controls pass. Row1764 contains97.9053%of squared
   column norm; the remaining2.0947% strongly alters the loss interaction.
3. **Normalization connects the column components locally.** Row-only and
   whole-column deletion leave identical raw L2 coordinate1764, but different
   complements and normalized states. Wiki normalized coordinate is−44.80 clean,
   −10.39 row-deleted, −37.09 column-deleted. This verified mathematical change
   is not yet an exclusive downstream mechanism account.
4. **Source keys/values transmit the pulse effect.** Complete source-interface
   cuts L2–15 close effects as architecture predicts; partial cuts show distributed
   dependencies. H15 L2 local K/V effects differ, while later Q at that first
   broadcast is unchanged. Poisoning H15 does not imply clean-head necessity.
5. **Shared source vectors can preserve outputs closely.** Static mean K/V and
   low-rank normalized residual templates transfer across tested texts/identities/
   lengths. They preserve dynamic attention, other-token states and most model
   computation. Output agreement and free-running sequences are not identical;
   prior bypass timing showed no speedup.
6. **Dynamic gates and nonzero values matter.** Values across layers interact
   nonadditively with fixed source keys. Arithmetic global/four-bin gates fail
   to match shared-KV fidelity, including fresh corpus calibration. Centered gate
   covariance and PCA projections show concentration but not one mechanism:
   Wiki rank4/rank12 agreement80.5/91.9%; random matched directions are worse.
   Bounded projections include nonlinear clipping; rank16 is passthrough.
7. **Later gate changes causally transmit some value-ablation damage.** In fixed
   K worlds, mean-value donor gates change Wiki zeroV218.92× to10.11×, reverse
   donor gates yield7.99× with meanV. Layer2 gates agree exactly before effects
   propagate. Hume constant gates with zeroV yield3.05×, better than native-world
   donors; no optimality or additive mediation fraction follows.
8. **A controlled upstream pathway involves L1/H12.** Screened on8 separate Wiki
   development windows, H12 alone strongly induces/suppresses the boundary pulse
   across64 heldout identity/prefix/corpus cases. Full32-head donor cuts exactly
   close. SingletonH12 overshoots; larger sets combine nonmonotonically.
   Every original random control containedH12, so no absent-control advantage is
   claimed. Follow-up explicitly tested H0/H3/H15 and complement31.
9. **Source value identity and two scalar factors explain the controlled
   production effect.** Current source0 H12 value exactly reproduces isolated
   donor induction; current boundary value/self-routing does not. Earlier history
   has changed the value feature. Donor gate/up1710 swaps show both factors are
   required for the donor intermediate in this family. Changing a factor affects
   the whole down-column, unlike one-weight deletion. These interventions do not
   measure natural H12 routing or establish a universal semantic detector.
10. **The second known unit is not a complete backup explanation.** With source0
   columns1710(L1) and8041(L2) both removed, natural-ID PPL ratios are1.821,2.470,
   1.038,1.027. Conditional second-unit effects exist, but difference-in-differences
   are not uniformly significant. Its potential pulse increases to~−200 after
   first ROW removal despite catastrophic output, while becoming smaller after
   first COLUMN removal. Pulse size alone is not recovery evidence.

## Rejected or narrowed interpretations

- Catastrophic deletion of one coefficient does not show its whole neuron is
  equally indispensable, nor that every changed downstream coordinate is one
  mechanism. Full-row and whole-column interventions have different estimands.
- A pure zero-value attention anchor, tested constant/binned gates, and a few gate
  directions do not preserve all outputs. This does not reject every optimized
  scalar/gate approximation or establish useful compression.
- Empty early history is sufficient for a pulse at a later same-ID boundary;
  absolute position0 necessity and reset-procedure independence are not proved.
- Natural first-token identity alone does not explain cross-corpus gaps.
- Mean PPL near1 does not imply argmax or free-generation equivalence.

## Active fresh-window test

LATER_USES.md / later_uses.py:16 fresh512-token windows at original confirmation
starts+768, disjoint from every original calibration/evaluation span and each
other. Natural/The/period IDs; first-unit row/column deletion at0, later1+, orall.
Capture full tokenwise gate/up/neuron traces and second-unit activations. Test
later-only natural dependence and source×later interactions. No fitting. This is
motivated by nontrivial all-versus-source effects in books/code; the earlier Wiki
later-only null must not be generalized prematurely. Completed336forwards and independently audited. Natural later-only row PPL:
Wiki0.99986, DuBois16.4206, Hobbes60.0221, optimizer20.4495. Later-only whole-column
ratios1.0113/1.0106/1.0081 in the latter3. Whole source-column versus source-row
rescue reproduces on fresh windows, but whole-column errors remain substantial
for some contexts. Source×later interactions are nonadditive.

All10 natural-ID large later pulses occur at first single-newline token187;
acrossall48identityvariants34/35large later pulses are187, with one underscore64
exception. This is descriptive. Single newline encodes187 and double newline535,
so rendered line breaks and tokenidentity must not be conflated.

NEWLINE_HISTORY completed672forwards and independent audit. Fixedp64/q128,
explicit prior IDs, common382suffix targets: prior187 suppresses q187 pulse in
every window; prior535/tab/space/The/period/underscore does not. q535 stays small.
Removing L0(q,p), but not L1(q,p) or nearby L0sham, restores q pulse~−290..−304.
This causally identifies a direct L0 history route in the constructed prefixes.
It does not establish a unique head or universal first-occurrence detector.
Removing q row with no prior187 causes19.27–41.63×PPL; q whole-column removal
0.9982–1.0083×. With prior187, severe row sensitivity moves to p (34.73–88.94×),
while q removal≈1. Restoring a second pulse by edge removal shifts PPL only
~0.9943–1.0038× under intact p. Pulse amplitude is not itself a performance gain.

NEWLINE_HEADS completed880forwards and independent audit. L0/H14 is the sole
large singleton on16oldWiki/Hume development; mean signed fraction1.0030. H0
control and all heldout conditions were frozen before16fresh disjoint windows.
H14 p attention mass U≈0.001, R≈0.943–0.961. In U, joint pK+V donor fromR suppresses
q pulse; K-only routes mass without suppression andV-only lacks routing. InR,
either K orV fromU breaks suppression. Q535query donor partly restores pulse.
All64 local oracle controls are exact here. Conditionalcompensation fails:
induceq+deletep PPL25.9–64.7; qrowconditionalcost CIcrosses0 in3of4texts.

NEWLINE_BROADCAST completed432forwards. Blocking p forstrictlylaterqueriesL2..15
removes alladditionalrowdamage exactly (architecturalclosure), and cleanblocked
PPL1.0021–1.0070 with94.5–97.77%argmax. Thusnear-cleanloss≠exactfidelity.
EarlyL2..5blockingrepairsmostdamage; middle/latealonefail. Key/valueinteractions
strong: row←columnK PPL1.0045–1.0139, row←columnV599.7–3308.8. Mixedstatesare
incompatible; noadditivecomponentharmclaim. FixedcleanKzeroV differsslightly
fromcurrentKzeroV; bothremainimperfect. All80closurechecks pass.

NEWLINE_NATURAL completed228forwards. No inputID edits. Twelvebook/codewindows
qualifyfirsttwo187(second≤255); Wiki4have no187 and remainineligible. H14edge
restoresq pulse inall12 (−132.2..−298.0). H12contextswaps doNOTtransfer the
qualitativepulse distinction, thoughsmallchangesexist. Thisrejects thetempting
singleH12relaystory across empty-history andnewlineinterventions.
Naturalwholecolumnexceptions matter: DuBoisp0windows columnp11.37/13.06×,
edge+columnq7.33/149.70×originalclean; p>0DuBoisnear1. Optimizerp1windowcolumnp
4.224× andedge+columnq1.343×originalclean. Conditionaledgebaselinevalues saved
separately. No universallyharmlesswhole-neuronremoval established.

## Highest-value next distinctions

1. Split H14 downstream influence into L0 attention residual, L0 MLP output,
   andL1 attention output using frozen donor component factorials. H12 alone
   failed totransferthiseffect; determine where gate/up factors change.
2. Separate normalized target/complement magnitude from complementary feature
   direction, using factorial direction swaps with norm/mean constraints and
   matched random directions. The column intervention changes both geometry
   and specific features; neither is yet the sole explanation.
3. Trace where the pre-existing target signal comes from when the first entire
   column is absent. Test earlier residual/attention/MLP contributions before
   adopting a unique compensatory-circuit account.
4. Confirm generalization on independent contexts/lengths and evaluate practical
   simplifications only with heldout output fidelity and actual resource benefit.

## Scope and integrity

One checkpoint/backend, small windows per group, repeated exploratory panels,
pointwise bootstrap intervals conditional on fixed calibration. Identity/prefix
variants reuse text. Model training overlap unknown. Original fresh confirmation
included an identical final Gutenberg footer in both books; primary results were
preserved and reviewed sensitivity excludes it. Upstream/column tests use the
first4 windows and avoid that footer. No public publishing, paid compute, unrelated
UIT restart, credential changes or duplicate GPU model loads. Every batch retains
raw data, manifests, unchanged model hashes and parameter versions.

## Immutable periodic artifacts

- First causal report: [private Library archive identifier retained locally]
- Report2: [private Library archive identifier retained locally]
- Report3: [private Library archive identifier retained locally]
- Report4: [private Library archive identifier retained locally]
- Report5: [private Library archive identifier retained locally]
- Report6: [private Library archive identifier retained locally]

Each report has a local HTML, Library metadata and immutable ZIP with file hashes,
raw outputs, code and independent reviews. See bundle_identity_reportN.json.
Report5 has6064forwards in369.0934batchseconds; zip218464177bytes,
SHA256df03aba2879faaaff6a8a357b9ced95f99604924d782ead516c5364e547d2f04.

Continue from informative failures. A finished batch/report is not a declaration
that the research is fully understood or that useful ideas are exhausted.

## 2026-10-21 coordinate correction
The previously described known second weight atL2 was misattributed to the paper. Table2 places both OLMo1B weights atL1. See COORDINATE_ERRATUM.md; old L2 data remain an exploratory separate-coordinate experiment. The corrected same-layer factorial completed144forwards with48exact prior anchors. L1/8041 source0 contribution was tiny inthese16contexts and doesnotexplain theprimary row/column difference.

## 2026-10-21 residual component route
448 new forwards show M0 output swaps transfer the qualitative H14-induced later-pulse state in both directions across16constructed+12natural cases. Frozen M0 alone gives partial pulse (~96–120 magnitude groupmeans), M0+A1 ~224–305; A1 alone remains quiet. Independent numerical review passed; this is a conditional mediation result, not uniqueness of M0 or an additive attribution.

## 2026-10-21 geometry and reproduction milestone
The616-forward source-geometry factorial passed84full-donor plus84same-donor output closures. Clean alignment bundle protects against row-world orthogonal directions; column-world direction also removes much harm. Fixedaxis only is approximate (source0agreement73.78–95.78%), notfaithfulreplacement. Naturalexceptions persist. Nine portablecore batches reran4956forwards withidentical4716windowmetrics. Correctedpair,components,andgeometry receivedindependentCPUaudits; checkpoint wasnotindependentlyreplicated byreviewer. Transfer tolocalothermodels isnext, afterprivate repositorypublication.
