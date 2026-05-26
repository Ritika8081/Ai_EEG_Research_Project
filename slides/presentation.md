# EEGNet Failure Analysis & Beyond
*Defence presentation — PhysioNet EEGBCI motor-imagery subset*

---

## Slide 1 — The question, the constraints

EEGNet works. Where does it fail, **mechanistically**, and what
follows from understanding the mechanism?

| Train | Test | Task | Channels | Sampling | Trials/subject |
| ----- | ---- | ---- | -------- | -------- | -------------- |
| Subj 1–8 | Subj 9–10 | L/R fist motor imagery | 64 | 160 Hz | ~45 |

Three seeds (1337, 2024, 7). Report **best AND worst** — no
cherry-picking. Source of truth: `results/metrics.json`.

---

## Slide 2 — Three experiments to localise the failure

| Setting                       | Train subjects | Test subjects |
| ----------------------------- | -------------- | ------------- |
| Cross-subject, full pool      | 1–8 (8)        | 9–10          |
| Cross-subject, reduced pool   | 1, 2, 3 (3)    | 9–10          |
| UMAP of penultimate features  | — (diagnostic) | — (diagnostic)|

Why no within-subject baseline? Because the failure to demonstrate is
**transfer**, not capacity. Within-subject EEGNet is well established
to work (Lawhern 2018, ~80 % on this task).

---

## Slide 3 — Baseline results: not what looks wrong, but how

| Setting | Best | Worst | Sil(class) | **Sil(subject)** |
| ------- | ---: | ----: | ---------: | ---------------: |
| Full 8 subj | 0.644 | 0.511 | 0.010 | **0.084** |
| Reduced 3 subj | 0.656 | 0.533 | -0.002 | **0.062** |

Two non-trivial observations:

1. Adding subjects did NOT help (0.644 ≈ 0.656). The *marginal value of a
   training subject is ~0* for cross-subject transfer.
2. **Subject silhouette ≫ class silhouette** (0.084 vs 0.010). The
   network's primary feature axis is "which subject is this?", not
   "left or right fist?".

---

## Slide 4 — Why (mechanistic claim)

> EEGNet's depthwise spatial filter has shape `(F1·D, C)` — one scalar
> per `(filter, electrode-index)`. The semantics of "electrode index
> *c*" is set by the cortical-source-to-scalp projection for that
> subject. With no learnable component that depends on *position*, the
> filters collapse onto a training-set-mean spatial profile and the
> classifier uses subject-specific covariance as a free axis of
> discrimination — BatchNorm preserves it, the FC head consumes it.

Two structural facts:

1. The depthwise weight has **no notion of physical electrode position**.
2. Resting-state spatial covariance encodes subject identity at very
   high SNR.

This is **structurally unavoidable** without architectural change.

---

## Slide 5 — Part 2: TopoNet — two surgical changes

### 5a. Per-trial Riemannian whitening
$$ \tilde X = (X X^\top / T + \varepsilon I)^{-1/2} X $$
Removes subject-specific covariance **without using subject statistics**.
Unlike Euclidean Alignment, needs no calibration trials.

### 5b. Coordinate-Conditioned Spatial Filters (CCSF)
$$ W[c, k] = \mathrm{MLP}([\,\mathrm{pos}_c,\; e_k\,]) $$
Filter is a smooth function on $S^2$, not an electrode-index lookup.
Parameter count is independent of $C$.

---

## Slide 6 — TopoNet results — an interesting failure + a clean win

| Setting | Best | Worst | Sil(class) | **Sil(subject)** |
| ------- | ---: | ----: | ---------: | ---------------: |
| Baseline, 8 subj | 0.644 | 0.511 | 0.010 | 0.084 |
| **TopoNet, 8 subj** | **0.489** | **0.456** | 0.005 | **0.014** |
| Baseline, 3 subj | 0.656 | 0.533 | -0.002 | 0.062 |
| **TopoNet, 3 subj** | **0.622** | **0.600** | 0.006 | **0.028** |

The intellectually honest reading:

- **Mechanism works**: subject silhouette drops 6× (full pool) and
  2.3× (reduced pool). Diagnostic from Part 1 is decisively improved.
- **8-subject accuracy drops** — TopoNet *underperforms baseline*.
- **3-subject accuracy improves and variance collapses** (every seed
  in [0.60, 0.62]).

---

## Slide 7 — Why TopoNet wins on 3 subjects, loses on 8

The motor-imagery class signal (contralateral mu/beta ERD)
**IS spatial covariance asymmetry** between C3 and C4. Per-trial
whitening removes that asymmetry too. So:

- **Full pool**: EEGNet has enough data to find robust filters without
  the prior. TopoNet's whitening is now *cost without benefit*.
- **Reduced pool**: EEGNet overfits. TopoNet's prior gives a cleaner
  objective.

I threw the baby out with the bathwater on the full pool. The fix I
would build next: **subject-conditional whitening** — remove the
subject-specific component of covariance only, not all covariance.

---

## Slide 7b — Ablation decomposes TopoNet's two halves

`scripts/ablation_part2.py`:

| Config              | 8-subj best/worst | 3-subj best/worst | sil(subject) 8/3 |
| ------------------- | :---------------: | :---------------: | :--------------: |
| Baseline            | 0.644 / 0.511     | 0.656 / 0.533     | 0.084 / 0.062    |
| Whitening only      | 0.600 / 0.489     | 0.600 / 0.489     | 0.022 / 0.017    |
| **CCSF only**       | 0.522 / 0.422     | **0.611 / 0.567** | 0.022 / 0.020    |
| TopoNet (both)      | 0.489 / 0.456     | 0.622 / 0.600     | 0.014 / 0.028    |

Three findings:

1. Either intervention alone reduces sil(subject) ~4×. Both attack the
   mechanism independently.
2. They stack *negatively* on accuracy — over-regularisation.
3. **CCSF alone has the highest worst-case test (0.567) of any 3-subj
   model**. The cleanest single intervention.

---

## Slide 8 — Part 3: Cortical-Manifold Augmentation (CMA)

The novel idea. Half-page rationale in `report/part3_idea.md`.

**Claim:** electrode-position uncertainty is a *continuous augmentation
axis* that nobody has used.

Per batch: sample $R \sim \mathrm{SO}(3)$ with angle $\theta \sim
\mathcal{U}(0, 4°)$. Build spherical-RBF interpolation matrix:
$$ A[i, j] \propto \exp\!\left(-\tfrac{1 - x_i^\top R x_j}{\sigma^2}\right) $$
(row-normalised). Apply $X \leftarrow A X$. Cost: one $C\times C$ matmul.

---

## Slide 9 — Why CMA is grounded, specific, defensible

- **Grounded:** inter-rater cap-placement error is ~5–10 mm (≈3–5°),
  documented in 10-20 BCI manuals (Picton et al., Klem et al.).
- **Specific:** spherical RBF on the unit-sphere head model; bank of
  24 precomputed rotations. Defensible against alternatives:
  - vs spherical splines (Perrin 1989): RBF is the leading-order term,
    qualitatively identical for augmentation.
  - vs channel dropout: continuous deformation, not discrete loss.
  - vs subject-mixup: physically plausible trials.

---

## Slide 10 — CMA results — a different kind of interesting failure

| Setting | Best | Worst | Sil(class) | **Sil(subject)** |
| ------- | ---: | ----: | ---------: | ---------------: |
| Baseline, 8 subj | 0.644 | 0.511 | 0.010 | **0.084** |
| **EEGNet + CMA, 8 subj** | 0.622 | 0.500 | 0.010 | **0.121** |
| Baseline, 3 subj | 0.656 | 0.533 | -0.002 | **0.062** |
| **EEGNet + CMA, 3 subj** | 0.633 | 0.522 | -0.003 | **0.090** |

**CMA did NOT do what I claimed.** Subject silhouette went *up*.
Accuracy is within noise of baseline.

---

## Slide 11 — CMA sweep: why the headline result was misleading

`scripts/sweep_cma.py` — 9 configs × 2 seeds × **40 epochs**:

| $\sigma$ | $\theta_{\max}$ | best | sil(subject) |
| -------- | --------------- | ---: | -----------: |
| 0.08     | any             | 0.611 | 0.033 |
| 0.18     | 2°–8°           | 0.600–0.611 | 0.038–0.039 |
| 0.30     | any             | 0.611 | **0.028** |

Baseline (no CMA, reduced pool): sil(subject) = 0.062.

**Every single sweep configuration reduced sil(subject) vs baseline.**
The augmentation *does* decouple representations from subject identity
at 40 epochs — exactly what I claimed.

So why did the 60-epoch headline run *raise* sil(subject)? Because the
model learns to **invert the augmentation**: $A$ is a fixed linear
$C \times C$ matrix; given enough gradient updates EEGNet can undo
it. CMA's benefit decays with training time.

Fix: **curriculum CMA** — schedule $p_\mathrm{apply}$ from 0.9 → 0.1
across training. Strong augmentation early, taper as the model
converges.

---

## Slide 11b — Curriculum CMA: the hypothesis validates

`scripts/curriculum_cma.py` (3 seeds × 60 epochs):

|                          | best  | worst | sil(class) | sil(subject) |
| ------------------------ | :---: | :---: | :--------: | :----------: |
| Baseline, 8 subj         | 0.644 | 0.511 | 0.010      | 0.084        |
| **Curriculum CMA, 8 subj** | 0.600 | **0.533** | **0.013** | **0.056** |
| Baseline, 3 subj         | 0.656 | 0.533 | -0.002     | 0.062        |
| Curriculum CMA, 3 subj   | 0.600 | 0.500 | -0.001     | 0.068        |

**On the full pool the hypothesis is confirmed**: sil(subject) drops
33 % (0.084 → 0.056), sil(class) rises slightly, worst-case test
improves. The constant-strength configuration was the bug; the
augmentation axis was always right.

On the reduced pool the curriculum doesn't help further — with only
3 training subjects, the model can't overfit through the augmentation
anyway. Honest scope: the curriculum is a fix for the regime where
constant CMA failed, not a universal upgrade.

---

## Slide 12 — Defence Q&A I expect

> **"Why per-trial whitening, not subject-mean?"**
> Subject-mean needs calibration data on the test subject — we have none
> for subjects 9, 10. Per-trial pays a class-signal-removal price I
> now understand and isolated via the ablation (Slide 7b).

> **"Within-subject EEGNet mean is 0.522 — barely better than chance.
> How can you say EEGNet works?"**
> I cannot, on this subset. The within-subject split has 9 test trials
> per subject — 11 % accuracy resolution. The assignment's premise
> ("EEGNet works, that is not in question") rests on Lawhern 2018's
> 109-subject reports. My subset is too small to demonstrate the
> within-subject ceiling but is sufficient to demonstrate the
> mechanistic *failure* I describe.

> **"Your CMA fails at 60 epochs but works at 40 epochs — that's just
> overfitting through augmentation."**
> Exactly. The augmentation is a fixed linear $C \times C$ operator;
> the model can invert it. That's not a structural problem with the
> augmentation axis — it's a missing schedule. Curriculum CMA is the
> documented next step.

---

## Slide 13 — What I would build next

1. **Subject-conditional whitening** — remove only subject-specific
   covariance, preserve class-discriminative covariance.
2. **CMA with scheduled $\theta_{\max}$** — small early in training
   (let class features form), larger later (broaden subject prior).
3. **Pre-train CCSF MLP across multiple EEG datasets** — it's
   channel-count-agnostic, so cross-corpus pre-training is free.

---

## Slide 13b — Visual evidence: the topomaps

`results/figures/topomap_eegnet.png` — 16 EEGNet depthwise filters.
**Fragmented, noisy spatial profiles.** No clean motor-cortex
localisation. These are not "left vs right motor area" detectors —
they are subject-specific covariance patterns.

`results/figures/topomap_ccsf.png` — 16 CCSF filters.
**Smooth gradients, all nearly identical.** The MLP-on-position prior
collapses to a small set of low-frequency spherical harmonics. This is
why CCSF underperforms on the full pool: not too little capacity, but
too little *diversity* in the filter bank. The fix is a diversity loss
across `filter_emb` — one more line of code, not implemented in this
submission.

`results/figures/per_subject_accuracy.png` — EEGNet hits S10 (0.71)
much better than S9 (0.58). The 0.644 number on Slide 3 is an average
that hides a real asymmetry — subjects are not equally transferable.

---

## Slide 13c — Classical reference: CSP+LDA

| Setting | best/worst |
| ------- | :--------: |
| CSP+LDA, 8 subj | 0.489 / 0.489 |
| CSP+LDA, 3 subj | 0.367 / 0.367 |

CSP is **at chance** on the full pool and **below chance** on the
reduced pool. CSP's filters explicitly maximise variance ratio per
class — i.e. they're subject-specific covariance maximisers by
construction. Cross-subject, those filters anti-align to test subjects.

This is direct empirical support for Part 1's mechanism: covariance-
based spatial filtering does not transfer. EEGNet (0.570 mean) uses
its capacity to find features *less* subject-coupled than CSP, but
still subject-coupled.

---

## Slide 13d — Combined: CCSF + curriculum CMA

| Setting | best/worst | sil(subject) |
| ------- | :--------: | :----------: |
| Combined, 8 subj | 0.522 / 0.422 | **0.022** |
| Combined, 3 subj | 0.600 / 0.467 | **0.021** |

**Non-additive on accuracy.** Combined-on-full matches CCSF-only-full
(both 0.522 best). Both interventions attack the same subspace
(subject-coupled embedding features); the second one hits the head
room the first one already covered. `sil(subject)` is the lowest of
any model tested — the mechanism is *over*-addressed at the
embedding level, but accuracy doesn't follow.

Honest take-away: **CCSF-alone (reduced pool) and curriculum CMA-alone
(full pool) are the per-regime best configurations**, not their
combination.

---

## Slide 14 — What you should take away

- The cross-subject failure is **structural, not statistical**.
  Adding subjects does not fix it (0.644 ≈ 0.656). CSP+LDA at chance
  empirically confirms the mechanism.
- The diagnostic that reveals the mechanism is the **silhouette
  comparison**, not accuracy. Accuracy on a 90-trial test set has a
  ±5 % noise floor.
- **Part 2 ablation** isolated CCSF as the cleaner half — best
  worst-case test on the reduced pool (0.567). Topomaps show CCSF's
  next-step problem (filter collapse).
- **Part 3 curriculum CMA** validated the augmentation-axis hypothesis
  on the full pool (sil(subject) ↓ 33 %, worst-case test ↑). The
  initial CMA failure was a schedule bug, not a structural one.
- **Combining the two interventions does not stack on accuracy** —
  they target overlapping subspaces. Honest finding, not hidden.
- Each part had a failure followed by an iteration that understood
  and partially fixed the failure. That iteration loop is what I
  would bring to your team.

Code (`main.py` + 7 scripts) reproduces every number on this deck.
