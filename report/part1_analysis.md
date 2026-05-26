# Part 1 — Where and Why EEGNet Fails

*Half a page. Mechanistic.*

## What we observe (real numbers from `results/metrics.json`)

Under cross-subject evaluation (train on subjects 1–8, test on 9–10), three seeds:

|                       | best  | worst | mean  | sil(class) | sil(subject) |
| --------------------- | ----- | ----- | ----- | ---------- | ------------ |
| Baseline, 8 subjects  | 0.644 | 0.511 | 0.570 | 0.010      | **0.084**    |
| Baseline, 3 subjects  | 0.656 | 0.533 | 0.596 | -0.002     | **0.062**    |
| Within-subject ref    | 0.778 | 0.222 | 0.522 | —          | —            |

(Within-subject: 80/20 random split within each of the 10 subjects, EEGNet
trained from scratch — see `scripts/within_subject.py`. The wide spread
is the small-data noise floor: 9 test trials per subject = 11 % resolution.)

Three non-trivial facts:

1. The full 8-subject pool gives **no consistent benefit** over the 3-subject
   pool. Adding subjects does not help generalisation to unseen subjects — the
   marginal value of an extra training subject is roughly zero for EEGNet on
   this paradigm.
2. **Cross-subject training (8 subj, 360 trials) gets better mean accuracy
   than within-subject training (1 subj, 36 trials).** 0.570 vs 0.522. The
   subset is too small for EEGNet to fit within-subject, but it does
   show that adding *out-of-distribution* trials helps when they come
   for free — the bottleneck is data volume more than distribution match.
3. The **silhouette score by subject is ~8× the silhouette by class**
   (0.084 vs 0.010). The penultimate-layer features carry far more information
   about *which subject* a trial belongs to than about *which class* it is.
   This is visible directly in `results/figures/baseline_*.png`: the UMAP plot
   colour-coded by subject is structured; the plot colour-coded by class is a
   blob.

These three facts together are the diagnostic.

## Why — the mechanism

EEGNet's failure is structural, not statistical:

1. **The depthwise spatial conv is an electrode-INDEX lookup.** Its weight has
   shape `(F1·D, C)` — one scalar per `(filter, electrode-index)` pair. The
   physical meaning of "electrode *c*" is set by what the cortical-source-to-
   scalp projection puts under that cap position for *this* subject. Skull
   thickness, gyral folding, fiducial registration and impedance vary
   subject-to-subject; the 10-05 montage standardises the *name* of the
   position, not the field it samples. So the same weight vector is forced
   to read a *different physical signal* across subjects.

2. **No mechanism for spatial adaptation.** With the filter bank frozen at
   inference time, the loss is minimised by aligning the filters to the
   training-set-mean spatial profile. When the test subject's geometry
   deviates from that mean, the network applies a stale spatial mask. That
   *is* the cross-subject failure mode — but the network has no way to know.

3. **Subject identity is a high-SNR signal that BatchNorm preserves.** The
   per-trial spatial covariance (impedance, reference-electrode contact, head
   shape) encodes subject identity with very low entropy. The classifier
   discovers it because it is *easier to learn than the class signal*. The
   silhouette-by-subject score is the receipt: 0.084 ≫ 0.010 says the
   network has effectively built a subject-identifier as its primary feature
   axis.

## The structural argument

EEGNet has **no learnable component that depends on electrode position**. Its
spatial filter is a function of *electrode index*. Position-invariance,
montage robustness, and subject-transfer all require a representation in
which spatial filters either (a) operate on a subject-normalised covariance,
or (b) are parameterised as smooth functions of physical position.

EEGNet has neither. This is the target of Part 2.

## Empirical support: classical CSP+LDA collapses

The classical BCI baseline (CSP + LDA) explicitly extracts spatial
filters that maximise per-class variance ratio — i.e. it is *built* to
exploit covariance structure. On this subset, cross-subject (see
`scripts/csp_baseline.py`):

| Pool | CSP+LDA best | CSP+LDA worst |
| ---- | -----------: | ------------: |
| 8 subj | 0.489 | 0.489 |
| 3 subj | 0.367 | 0.367 |

CSP is at chance on the full pool and **below chance** on the reduced
pool. Below-chance means CSP's filters are anti-aligned with the test
subjects' covariance — exactly what the mechanism above predicts when
subject-specific covariance learning meets test subjects with
incompatible covariance. EEGNet (0.570 mean) sits above CSP because it
has temporal-frequency capacity that CSP lacks — but the spatial-filter
half of EEGNet is doing the same kind of subject-specific learning,
just less catastrophically.

The topomap figure (`results/figures/topomap_eegnet.png`) makes this
visible: EEGNet's 16 learned depthwise filters look like fragmented
noise rather than the clean lateral-vs-medial contrast a fists-vs-feet
imagery task should produce (bilateral hand area C3/C4 vs midline
foot area Cz).
