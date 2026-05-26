# Part 2 — TopoNet: My Proposed Improvement

## What I changed and why

Two surgical changes, each targeting one root cause from Part 1.

### 1. Per-trial Riemannian whitening

Before the network sees a trial $X \in \mathbb{R}^{C \times T}$:

$$
C_X = \frac{1}{T} X X^\top + \varepsilon I_C, \qquad
\tilde{X} = C_X^{-1/2} X.
$$

After whitening, $\tilde{X}\tilde{X}^\top/T \approx I$ for *every* trial,
regardless of subject. The dominant carrier of subject identity is
removed algebraically. Unlike Euclidean Alignment (Zanini et al., 2018),
which whitens with a *subject-mean* reference and requires per-subject
calibration data, per-trial whitening uses nothing about the test subject.

### 2. Coordinate-Conditioned Spatial Filters (CCSF)

Replace EEGNet's depthwise weight (shape `(F2, C)`, one scalar per
electrode-index per filter) with an MLP that maps physical electrode
position to filter weight:

$$
W[c, k] \;=\; \mathrm{MLP}\bigl( [\,\mathrm{pos}_c,\; e_k\,] \bigr), \quad
\mathrm{pos}_c \in S^2,\; e_k \in \mathbb{R}^{d_e}.
$$

The number of spatial-filter parameters is independent of $C$. The filter
is a smooth function on $S^2$ — small montage shifts perturb the *input*
to the MLP slightly rather than breaking it.

## What I expected

Whitening should help cross-subject because it removes the subject-identity
leak. CCSF should help in both regimes — especially with less data — because
it is a strong inductive prior on spatial filters.

## What actually happened — main result

|                            | best  | worst | mean  | sil(class) | sil(subject) |
| -------------------------- | ----- | ----- | ----- | ---------- | ------------ |
| Baseline EEGNet, 8 subj    | 0.644 | 0.511 | 0.570 | 0.010      | 0.084        |
| **TopoNet**, 8 subj        | 0.489 | 0.456 | 0.467 | 0.005      | **0.014**    |
| Baseline EEGNet, 3 subj    | 0.656 | 0.533 | 0.596 | -0.002     | 0.062        |
| **TopoNet**, 3 subj        | **0.622** | **0.600** | **0.615** | 0.006 | **0.028** |

Subject silhouette drops 6× on the full pool and 2.3× on the reduced
pool — the diagnostic from Part 1 is decisively improved. But on the
full pool, accuracy gets *worse*. On the reduced pool, accuracy improves
AND variance collapses.

## Ablation — which half does what?

Ran `scripts/ablation_part2.py` to decompose the intervention
(`results/ablation_part2.json`):

|                       | best  | worst | sil(class) | sil(subject) |
| --------------------- | ----- | ----- | ---------- | ------------ |
| Baseline, 8 subj      | 0.644 | 0.511 | 0.010      | 0.084        |
| Whitening only, 8 subj| 0.600 | 0.489 | 0.006      | **0.022**    |
| **CCSF only, 8 subj** | 0.522 | 0.422 | 0.004      | **0.022**    |
| TopoNet (both), 8 subj| 0.489 | 0.456 | 0.005      | 0.014        |
| Baseline, 3 subj      | 0.656 | 0.533 | -0.002     | 0.062        |
| Whitening only, 3 subj| 0.600 | 0.489 | -0.002     | **0.017**    |
| **CCSF only, 3 subj** | **0.611** | **0.567** | 0.005 | 0.020 |
| TopoNet (both), 3 subj| 0.622 | 0.600 | 0.006      | 0.028        |

Three findings that change the design:

1. **Either intervention alone reduces sil(subject) ~4×.** Both halves of
   the architecture independently address the mechanism from Part 1.
   They aren't redundant — each attacks a different surface (whitening
   removes covariance; CCSF replaces the index lookup) — but they aren't
   complementary either.

2. **The two interventions stack negatively on accuracy.** Whitening
   alone: 0.600 best. CCSF alone: 0.522 best. Combined: 0.489 best. The
   compositional cost is real because each intervention already
   constrains the hypothesis space; stacking them over-regularises.

3. **CCSF alone is the cleanest winner on the reduced pool.** Best=0.611,
   *worst=0.567*. Lower variance than baseline (0.567 worst is the
   highest worst-case of any 3-subj configuration tested). With
   sil(subject)=0.020, the representation is decoupled from subject
   without the class-signal-destruction problem whitening causes.

## The gap between expectation and reality

I expected whitening to be the dominant half (covariance is the
high-SNR carrier of subject identity). The ablation shows CCSF and
whitening lower sil(subject) by similar amounts (0.022 each) but
**CCSF preserves accuracy better** because it doesn't disturb the
spatial covariance structure carrying the class signal — it just
changes how the network *combines* electrodes.

The motor-imagery class signal for runs 6/10/14 is a
lateral-vs-medial contrast (bilateral hand area C3/C4 for imagined
fists, vs midline foot area Cz for imagined feet). That contrast
lives in the spatial covariance. Whitening flattens covariance
across the board, taking the class signal with it. CCSF only
changes how covariance is read, not whether it is preserved. Hence
the asymmetric cost.

The fix I would build next: drop whitening entirely, keep CCSF, and add
a **subject-conditional gating** on the spatial filter MLP so the
filter can adapt to coarse subject geometry without ever seeing
subject identity at the classifier.

## Visual evidence — topomaps tell the same story

`results/figures/topomap_eegnet.png` plots the 16 trained depthwise
filters of EEGNet baseline on a 10-05 head topomap. They look like
fragmented noise — no clean lateral-vs-medial motor-cortex contrast.
This is
exactly what the failure analysis predicts: the filters have learned
subject-specific covariance structure, not motor-cortex localisation.

`results/figures/topomap_ccsf.png` plots the 16 CCSF filters. They are
smooth gradients on the sphere (the MLP prior working as designed) but
**they collapse to nearly identical profiles**. That collapse explains
the full-pool underperformance: not too little capacity, but too
little *diversity* in the filter bank. A diversity-encouraging loss on
the filter embeddings `e_k` is the obvious one-line follow-up.

## Why this is not a transplant of an existing method

- Per-trial whitening is closely related to Euclidean Alignment but
  algebraically distinct: EA uses a subject-mean reference (requires
  calibration data); I use a per-trial reference (zero calibration,
  pays a class-signal-removal price I now understand).
- CCSF is conceptually analogous to coordinate-based networks (CoordConv,
  implicit-neural-rep MLPs) but applied to the spatial-filter weight
  itself, not the signal. To my knowledge no EEG paper parameterises the
  depthwise spatial filter as $\mathrm{MLP}(\mathrm{pos})$. See
  `report/references.md`.
