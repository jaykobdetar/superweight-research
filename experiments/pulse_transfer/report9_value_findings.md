Both the norm profile and direction matter causally in this factorial. Giving
E directions the B per-head norms produces natural-start PPL ratios
2.531–5.035; giving B directions the E norms reduces them to 1.162–1.329.
The B-direction/B-norm diagonal with E keys remains at 5.254–12.614. The
corresponding BOS mixed-cell ranges are 2.197–6.654 and 1.431–1.734.
All four simple effects—two norm effects and two direction effects—raise NLL
in each of the 32 input/start pairs. Thus neither an amplitude-only nor a
direction-only explanation describes this intervention family.

Define the signed norm/direction interaction as NLL(DE,NE) − NLL(DE,NB) −
NLL(DB,NE) + NLL(DB,NB). It is positive in 31 of 32 pairs and all
eight group intervals exclude zero. One Wiki natural-start passage has a small
negative interaction (−0.01484 nats/token), which is retained. Direction damage
is generally stronger at the B norm profile. These are conditional effects,
not percentages of mediated damage or proof of one underlying mechanism.
Even B direction with E norms agrees with native argmax predictions only
79.71–88.77% for natural starts and 70.00–80.29% for BOS starts; rescaling is
not native-output restoration.

The norm profile requires care: although the complete L11 value vector is
smaller in B, only head 8 has a smaller norm in every pair; its other 11 heads
are larger. Earlier whole-layer norm summaries would conceal that structure.
The signed-permuted controls are also not uniformly worse than their originals.
At E norms their natural-start ratios are 1.102–1.226; at B norms they are
3.598–8.474. Permuted E values improve the BOS optimizer group by 0.0699
nats/token on average (all four inputs; interval below zero), and permuted B
values improve over the unpermuted B/B cell in 30 of 32 inputs and all eight
group means. The random controls neither identify a unique semantic direction
nor supply a distribution over random interventions: one seed was frozen.

Zeroing these values in otherwise native C gives natural-start ratios
1.0014–1.0095 and BOS ratios 1.0021–1.0131. Native argmax agreement remains
95.15–97.30% and 94.41–96.91%, respectively. Therefore the low-loss zero-value
operation is an approximate intervention, not a faithful replacement of native
outputs. Together with the source-block comparison, the results motivate a
direct query/gating test while keeping values zero, rather than another sweep
of the original weight's numerical value.
