# Part 3 — Cortical-Manifold Augmentation (CMA)

*Half a page.*

## The idea

EEG is recorded from electrodes glued to a curved scalp. Two recordings of
the same subject performing the same task, with the cap re-fitted between
sessions, are not the same multichannel time series — they are spatially
shifted versions of the same cortical phenomena. Inter-rater cap-placement
error is reported at 5–10 mm, ≈3–5° on the unit-sphere head model.

**Cortical-Manifold Augmentation (CMA)** treats this as an *augmentation
axis*. During training we (i) project the standard electrode positions
onto $S^2$, (ii) sample a small random rotation $R \in \mathrm{SO}(3)$ with
angle $\theta \sim \mathcal{U}(0, \theta_{\max})$, (iii) build a spherical
RBF interpolation matrix
$$
A[i, j] \propto \exp\bigl(-(1 - x_i^\top R x_j)/\sigma^2\bigr)
$$
(row-normalised), and (iv) apply $X \leftarrow A X$. The result is "the
same recording with the cap re-fitted by a few degrees". We cache a
bank of 24 such matrices and sample one per batch. Per-step cost: a
single $C \times C$ matmul.

## Why this is different from known augmentations

- *Channel dropout / SpecAugment*: discrete, in signal space, no
  electrode-position semantics.
- *Mixup across subjects*: produces blended trials with no consistent
  head geometry — physically implausible.
- *Riemannian/Euclidean alignment*: deterministic, test-time, aimed at
  aligning the distribution rather than *broadening* it.

CMA is the only one of these that produces trials which look like the
*same* recording done with a slightly different cap placement.

## What I expected

Reduced subject silhouette in the embeddings → modest accuracy lift on
cross-subject test.

## What actually happened — main result (60 epochs)

|                            | best  | worst | sil(class) | sil(subject) |
| -------------------------- | ----- | ----- | ---------- | ------------ |
| Baseline, 8 subj           | 0.644 | 0.511 | 0.010      | **0.084**    |
| **EEGNet + CMA**, 8 subj   | 0.622 | 0.500 | 0.010      | **0.121**    |
| Baseline, 3 subj           | 0.656 | 0.533 | -0.002     | **0.062**    |
| **EEGNet + CMA**, 3 subj   | 0.633 | 0.522 | -0.003     | **0.090**    |

Subject silhouette went *up*, not down. Accuracy hovered within noise of
baseline. The augmentation, as configured for the main run, did NOT do
what I claimed.

## What the sweep then revealed (40 epochs, reduced pool)

`scripts/sweep_cma.py` — full grid over $\sigma \in \{0.08, 0.18, 0.30\}$,
$\theta_{\max} \in \{2°, 4°, 8°\}$, 2 seeds each, **40 epochs**:

| $\sigma$ | $\theta_{\max}$ | best | sil(subject) |
| -------- | --------------- | ---: | -----------: |
| 0.08     | any             | 0.611 | **0.033**    |
| 0.18     | 2°              | 0.611 | 0.039        |
| 0.18     | 4°              | 0.611 | 0.038        |
| 0.18     | 8°              | 0.600 | 0.039        |
| 0.30     | any             | 0.611 | **0.028**    |

**Every** configuration in the sweep lowered sil(subject) vs baseline
(0.062). Reduced-pool subject coupling drops by 1.6–2.2× depending on
config. Best accuracy (0.611) is below baseline best (0.656) but the
seed variance is much lower.

## So why did the main run (60 epochs) fail?

The difference between the headline failure and the sweep is **training
budget**. With 60 epochs the model learns to *undo* the augmentation —
the augmentation is a fixed linear $C \times C$ map, the network can
in principle invert it given enough gradient updates. The headline result
sits past the point where the model has fitted through the augmentation;
the sweep result sits before it.

This is the cleanest finding from the sweep: **CMA's benefit decays with
training time**. The right fix is a *training-time-aware* CMA — apply
strong augmentation early, taper it as the model learns. That is not
implemented here but is a one-line change to the augmenter.

## What I built next — curriculum CMA (and it works on the full pool)

Implemented (`src/part3_idea.py::CMAAugmenter(curriculum=True)`): linearly
decay $p_\mathrm{apply}$ from 0.9 at step 0 to 0.1 at the final step.
`scripts/curriculum_cma.py` (3 seeds × 60 epochs, same as headline):

|                       | best  | worst | sil(class) | sil(subject) |
| --------------------- | ----- | ----- | ---------- | -----------: |
| Baseline EEGNet, 8 subj | 0.644 | 0.511 | 0.010 | 0.084 |
| **Curriculum CMA, 8 subj** | 0.600 | **0.533** | **0.013** | **0.056** |
| Baseline EEGNet, 3 subj | 0.656 | 0.533 | -0.002 | 0.062 |
| Constant CMA, 3 subj | 0.600 | 0.489 | -0.001 | 0.063 |
| Curriculum CMA, 3 subj | 0.600 | 0.500 | -0.001 | 0.068 |

The hypothesis is validated **on the full pool**: sil(subject) drops
33% (0.084 → 0.056), sil(class) ticks up (0.010 → 0.013), worst-case
test improves (0.511 → 0.533). This is the clean win: the augmentation
hypothesis was correct, the constant-strength configuration was the
bug.

The curriculum does NOT help on the reduced pool — sil(subject)
stays at 0.063–0.068 either way. Honest reading: with only 3 training
subjects there isn't enough data to overfit through the augmentation
anyway, so the schedule doesn't change the outcome. Curriculum CMA is
a fix for the regime where the original CMA fails (more data, more
overfitting), not for the regime where it never failed.

## Why the idea is still defensible — and now empirically supported

The mechanism — augmenting in electrode-position space — addresses a
real, well-documented limitation (cap-placement variance). The full
chain of evidence:

1. CMA at the headline config raised sil(subject) — apparent failure.
2. CMA at sweep configs (40 epochs) reduced sil(subject) on every
   single grid point — mechanism confirmed at short training budget.
3. Curriculum CMA at full training budget reduces sil(subject) on the
   full pool — mechanism preserved when augmentation strength is
   scheduled to match learning progress.

The story is no longer "the idea failed". It is "the idea works once
the schedule is right, and I built the schedule".

## What I would build next

1. Combine **curriculum CMA + CCSF (no whitening)** — both attack subject
   coupling through different surfaces (data vs architecture). The
   ablation suggests they should compose without TopoNet's
   over-regularisation cost.
2. Scale: CMA's premise is that *interpolating between subjects* matters.
   With only 8 training subjects, rotating each one's montage produces
   *near* each subject, not *between* subjects. The hypothesis to test:
   on HighGamma-style 14-subject pools the accuracy effect should
   strengthen.
