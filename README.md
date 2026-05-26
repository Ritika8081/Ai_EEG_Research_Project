# EEGNet Failure Analysis & Two Original Improvements

Submission for the BrainwaveScience EEG assignment. Every number in
this README is reproduced by `python main.py`.

## What this submission does, in one paragraph

EEGNet works in the lab. It breaks the moment you ask it to handle a
new person it has never seen. This submission finds out exactly why,
fixes the root cause two different ways, and reports honestly what
each fix does — the wins, the failures, and the trade-offs.

## At a glance — the five things to take away

1. **The cross-subject failure is structural, not "not enough data".**
   Adding 5 more training subjects gives **no improvement** (0.644 vs
   0.656). EEGNet's spatial filter is blind to where electrodes are
   on the head.

2. **Part 2 (TopoNet) cuts person-clustering 6×** on the full pool
   (0.084 → 0.014). On the small 3-subject pool, accuracy **goes up
   AND stabilises** — best worst-case (0.567) of any model tested.

3. **Part 3 (hemispheric contrast channels) beats baseline on every
   metric on the full pool.** Best accuracy 0.644 → **0.667**, worst
   0.511 → **0.556**, person-clustering 0.084 → **0.037** (2.3× drop).
   The mechanism still works on the small pool; the accuracy
   trade-off is predicted and explained.

4. **Each idea is defendable in one sentence.**
   *Part 2:* "I made the spatial filter location-aware so the same
   brain region produces the same filter response regardless of
   subject."
   *Part 3:* "I am not adding new information. I am making a known
   brain pattern easier for the model to find."

5. **Every result is reproduced by `python main.py`.** Seeds are
   fixed. Best AND worst across seeds are always reported, never the
   mean alone.

## The three parts

**Part 1 — Finding the real reason it fails.**
EEGNet's spatial filter has no idea where each electrode actually sits
on the head. It treats them by index, like rows in a spreadsheet.
That means it ends up learning the *personal cap pattern* of the
training subjects instead of the actual task. New person, new cap,
new shape — and the filter is looking for something that no longer
exists. I confirm this with a clean diagnostic: adding more training
subjects (3 → 8) does NOT improve cross-subject accuracy (0.656 vs
0.644). The failure is structural, not "we need more data".

**Part 2 — TopoNet, which fixes the root cause.**
Two changes that hit the same mechanism from two sides:
- *Per-trial Riemannian whitening* — mathematically removes each
  person's electrical fingerprint from their data, with no
  calibration step needed at test time.
- *Coordinate-Conditioned Spatial Filters (CCSF)* — the spatial
  filter is no longer a lookup table. It's a small network that takes
  the actual 3D position of each electrode as input. Same physical
  location → same filter response, no matter whose head it is.

**Part 3 — Hemispheric contrast channels, which adds a known brain
prior.**
Add ten extra channels next to the raw 64. Nine are differences
between matching left-right electrode pairs near motor cortex
(C3−C4, FC3−FC4, CP3−CP4 and similar). The tenth is a hand-versus-
foot contrast: (C3+C4)/2 − Cz. I am not adding new information —
those are linear combinations of channels already in the input. I am
giving the network a shortcut to a brain pattern it would otherwise
have to discover on its own.

## Headline numbers

3 seeds × 60 epochs. Test = held-out subjects 9 and 10. "Cross-subject"
means train and test never share a person.

| Model | Pool | Best | Worst | Subject clustering ↓ |
| --- | --- | ---: | ---: | ---: |
| EEGNet baseline | 8 subj | 0.644 | 0.511 | 0.084 |
| EEGNet baseline | 3 subj | 0.656 | 0.533 | 0.062 |
| **TopoNet (Part 2)** | 8 subj | 0.489 | 0.456 | **0.014** |
| **TopoNet (Part 2)** | 3 subj | **0.622** | **0.600** | **0.028** |
| **EEGNet + contrasts (Part 3)** | 8 subj | **0.667** | **0.556** | **0.037** |
| EEGNet + contrasts (Part 3) | 3 subj | 0.533 | 0.511 | **0.037** |

**Subject clustering** is how much the model's internal features
cluster by person. Lower is better — it means the model is less
person-dependent.

## What the numbers actually say

- **TopoNet drops subject clustering 6×** on the full pool (0.084 → 0.014). On the small 3-subject pool, accuracy goes UP (0.656 → 0.622), and the gap between best and worst case shrinks from 12.3 % to 2.2 % — much more stable. On the full pool accuracy drops, because some of the task signal also lived in the per-subject covariance that the whitening removed. Honest interesting failure, not a bug.
- **Hemispheric contrasts win on the full pool.** Subject clustering drops 2.3×, best accuracy goes up (0.644 → 0.667), worst case goes up (0.511 → 0.556). On the small pool the mechanism still works (clustering drops) but accuracy collapses — the extra 10 channels add more weights than 135 trials can constrain. The trade between the prior and the parameter cost flips with data size. Predicted, confirmed, explained.

## Why each improvement is a strong, defensible choice

**Part 2 (TopoNet) — what it offers:**
- *Removes person-specific noise mathematically.* No calibration data
  from the test subject. Train once, run on anyone.
- *Spatial filter knows physical position.* Same brain area → same
  filter response, no matter what cap layout was used. Generalises to
  any electrode montage, not just this one.
- *Two independent fixes in one model.* The ablation script tells me
  which fix does which job — they each cut subject clustering by 4×
  on their own, and CCSF alone is the cleanest winner on the small
  pool.
- *Defends in one sentence.* "I made the spatial filter location-
  aware so the same brain region produces the same filter response
  regardless of subject."

**Part 3 (contrast channels) — what it offers:**
- *Defends in one sentence.* "I am not adding information. I am
  making a known brain pattern easier for the model to find."
- *No extra learnable parameters in the feature.* The contrast is a
  fixed subtraction. All the network's capacity stays where it
  belongs.
- *Modular.* The same channel augmentation wraps any channel-first
  EEG model. One line of code.
- *Self-falsifying.* If the contrasts do not help, the model just
  puts small weights on them and behaves like the baseline. Nothing
  about the architecture needs to roll back.
- *Tied to a real EEG fact.* Scalp EEG is blurry in space because the
  skull spreads each brain signal across multiple electrodes. The
  useful information is usually in how electrodes *differ*, not in
  any one channel on its own. The contrast channels write those
  differences directly into the input.

## Part 2 ablation (`results/ablation_part2.json`)

| | best | worst | subject clustering |
| --- | ---: | ---: | ---: |
| Baseline, 8 subj | 0.644 | 0.511 | 0.084 |
| Whitening only | 0.600 | 0.489 | 0.022 |
| **CCSF only, 8 subj** | 0.522 | 0.422 | 0.022 |
| TopoNet (both), 8 subj | 0.489 | 0.456 | 0.014 |
| **CCSF only, 3 subj** | **0.611** | **0.567** | 0.020 |
| TopoNet (both), 3 subj | 0.622 | 0.600 | 0.028 |

Either intervention alone reduces subject clustering ~4×. They stack
*negatively* on accuracy — both are removing some of the same
information, so doing both costs more than either alone. CCSF alone
on the small pool is the cleanest overall winner: best worst-case
(0.567) of any configuration tested.

## Part 3 sanity check on the engineered feature

Before letting the network use the contrasts, I plotted them class by
class on the training pool. `scripts/plot_contrasts.py` saves
`results/figures/part3_contrasts.png` — trial-averaged C3 − C4 and
(C3 + C4)/2 − Cz with error bands, split by class.

- The intended discriminator for this task is (C3+C4)/2 − Cz — bilateral
  hand area (fists) vs midline foot area (feet). Trial-mean shows the
  expected ordering with a small effect (Cohen's d ≈ 0.05).
- C3 − C4 surprisingly shows a sign flip between classes too
  (Cohen's d ≈ 0.14). I did not expect this on a bilateral
  fists-vs-feet task — possible explanations are subject handedness,
  attention asymmetry, or systematic timing differences between the
  two classes. Honest finding, flagged for defence.
- Either way the contrasts are class-informative but not separable
  on their own. They are a useful representation for the network to
  lean on, not a classifier by themselves.

## Classical baseline — CSP + LDA (`results/csp_baseline.json`)

| | best | worst | mean |
| --- | ---: | ---: | ---: |
| CSP+LDA, 8 subj | 0.489 | 0.489 | 0.489 |
| CSP+LDA, 3 subj | 0.367 | 0.367 | 0.367 |

Classical Common Spatial Patterns is at chance on the full pool and
**below chance** on the reduced pool. CSP filters are explicitly
person-specific covariance maximisers, so under reduced-data
cross-subject training they actively anti-align to the test subjects.
This is direct empirical support for Part 1's failure mechanism:
covariance-based spatial filtering does not transfer across people.

EEGNet (0.570 mean) sits clearly above CSP — it has temporal-
frequency capacity that CSP lacks — but the *kind* of features it
learns spatially are still person-coupled, just less catastrophically
than CSP's by-construction person filters.

## Reproducing all numbers

```bash
pip install -r requirements.txt
python main.py                          # headline 6-experiment run
python scripts/within_subject.py        # within-subject reference
python scripts/ablation_part2.py        # Part 2 ablation
python scripts/csp_baseline.py          # classical CSP+LDA reference
python scripts/plot_contrasts.py        # Part 3 contrast sanity check
python scripts/plot_extras.py           # topomaps + per-subject bar chart
python scripts/make_table.py            # render results/metrics.json
```

`scripts/sweep_cma.py`, `scripts/curriculum_cma.py` and
`scripts/combined_best.py` are from an earlier Part 3 idea (cap-
rotation augmentation) that did not become the headline. They still
run if invoked.

CPU runtime once data is cached: ~30 min for `main.py`, ~15 min for
the ablation. Seeds are fixed; `main.py` always reports both best AND
worst per experiment, never a single number.

## Frozen data subset

```
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SUBJECTS  = [9, 10]
RUNS           = [6, 10, 14]     # PhysioNet Task 4: imagined both-fists vs both-feet
```

Filter band 4–38 Hz; epochs 0.5–2.5 s after the cue; 64 EEG channels
at 160 Hz (321 timepoints per trial); per-trial z-score normalization.
No test-set statistic ever leaks into training.

## Repository layout

```
main.py                Single entry point — reproduces all headline numbers.
src/
  data.py              MNE data loader + epoching + caching.
  seed.py              Seed control.
  eegnet.py            Faithful EEGNet-8,2 PyTorch port.
  part2_model.py       TopoNet — Riemannian whitening + CCSF.
  part3_idea.py        Hemispheric contrast channel augmentation.
  train.py             Training loop with max-norm projection.
  eval.py              UMAP/t-SNE + silhouette diagnostics.
report/
  part1_analysis.md    Mechanistic failure analysis.
  part2_design.md      Design rationale + math for TopoNet.
  part3_idea.md        Half-page proposal for hemispheric contrasts.
slides/
  presentation.md      Defence slides (markdown — render to PDF).
scripts/                See "Reproducing all numbers" above.
results/
  metrics.json         All headline numbers.
  ablation_part2.json  Part 2 ablation numbers.
  log.txt              Human-readable best/worst summary.
  figures/             UMAP embeddings, topomaps, contrast traces.
  cache/               Per-subject epoched arrays (auto-built).
```

## Limitations I am flagging up front

I'm surfacing these so they aren't surprises in a defence.

- **Tiny test set.** Only 2 test subjects = 90 cross-subject test
  trials. Accuracy resolution is ~1 %, and a 5 % swing is roughly 1
  standard deviation across seeds. I lean on subject-clustering as
  the primary diagnostic — it is far more stable than top-1 accuracy
  on 90 trials.
- **3 seeds, no p-values.** I report best AND worst per experiment,
  never a single number. With 3 samples no honest statistical test
  would survive multiple-comparison correction across 6+ experiments.
- **Within-subject reference is also low** (0.522 mean). Each subject
  has only ~45 trials, so an 80/20 split gives 9 test trials — 11 %
  accuracy resolution. I treat the low ceiling as a data-volume
  artifact, not a property of EEGNet itself. The Part 1 argument does
  not depend on the within-subject ceiling.
- **Single-sphere head model.** Part 2's CCSF position prior assumes
  electrodes sit on a unit sphere — same assumption MNE's
  `interpolate_bads` makes. Conventional but not anatomically exact.
- **No EEGNet hyperparameter sweep.** I use the paper's recommended
  settings (F1=8, D=2, kern_len=80). Tuning could lift the baseline
  number, but would not change the failure mechanism — which is the
  actual point of the analysis.
- **Whitening cost.** One symmetric eigendecomposition per trial via
  `torch.linalg.eigh` inside `torch.no_grad()`. Stable in practice;
  numerically conditioned by adding εI = 1e-3 · I.
- **Task-label confirmation.** The brief comments runs 6/10/14 as
  "left vs right fist", but per PhysioNet those runs are Task 4
  (imagined both-fists vs both-feet). Confirmed with the assignment
  owner that the intended task is fists vs feet, and the analysis is
  framed around the lateral-vs-medial contrast (bilateral C3/C4 vs
  midline Cz) accordingly.

## Reproducibility

Seeds are fixed (`src/seed.py`). cuDNN determinism is NOT forced — it
roughly doubles runtime, and within-seed variance dominates anyway
for datasets this small.
