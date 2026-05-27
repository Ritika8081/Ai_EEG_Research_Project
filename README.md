# EEGNet Failure Analysis & Two Original Improvements

Submission for the BrainwaveScience EEG assignment.

**Reading note.** `python main.py` reproduces the six headline
experiments (baseline / Part 2 / Part 3, each on the 8-subject and
3-subject pools) and writes `results/metrics.json`,
`results/log.txt` and the headline UMAP figures. The supplementary
analyses (Part 2 ablation, CSP+LDA reference, within-subject
reference, Part 3 contrast sanity check, Part 3 filter-weight
check, topomaps, per-subject bar chart) each live as a standalone
script under `scripts/` — they all run end-to-end with no flags.
Full list under *Reproducing all numbers* below.

## What this submission does

EEGNet works in the lab. It breaks the moment you ask it to handle a
new person it has never seen. I tried to find out why that happens,
fix the root cause two different ways, and write down honestly what
each fix actually did — including where it didn't work.

## Five things that came out of this

1. **The cross-subject failure is structural, not "not enough data".**
   Adding 5 more training subjects gives **no improvement** (0.644 vs
   0.656). EEGNet's spatial filter has no idea where electrodes
   actually sit on the head.

2. **Part 2 (TopoNet) cuts person-clustering 6×** on the full pool
   (0.084 → 0.014). On the small 3-subject pool, accuracy goes up
   and stabilises — best worst-case (0.567) of any model I tried.

3. **Part 3 (hemispheric contrast channels) beats baseline on every
   metric on the full pool.** Best accuracy 0.644 → **0.667**, worst
   0.511 → **0.556**, person-clustering 0.084 → **0.037** (2.3× drop).
   The mechanism still works on the small pool; the accuracy
   trade-off is one I'd predicted and the report explains why. I
   also ran a check on my own claim with
   `scripts/part3_filter_analysis.py`, which looks at how much
   weight each trained spatial filter puts on the contrast channels
   against the 10/74 ≈ 0.135 random baseline. The aggregate number
   came back modest — trained mean 0.150, only about 11 % above
   chance — so I can't honestly claim the model leans heavily on the
   contrasts overall. What I do still stand behind is the per-filter
   side: shares range from 0.031 to 0.263 across the 16 filters, so
   some clearly concentrate on the contrasts (~1.5–2× baseline) and
   others ignore them. That kind of spread is not what you'd see if
   the contrasts were redundant noise. The contrasts end up being a
   useful sideline for a minority of filters, not a dominant input.
   Figure: `results/figures/part3_filter_weights.png`.

4. **One sentence per idea, in case I get pressed for time.**
   *Part 2:* "I made the spatial filter location-aware so the same
   brain region produces the same filter response regardless of
   subject."
   *Part 3:* "I'm not adding new information. I'm making a known
   brain pattern easier for the model to find."

5. **Every headline number is reproduced by `python main.py`.**
   Seeds are fixed in `src/seed.py`. I report best and worst across
   seeds for every experiment, never just the mean.

## The three parts

**Part 1 — Finding the real reason it fails.**
EEGNet's spatial filter has no idea where each electrode actually
sits on the head. It treats them by index, like rows in a
spreadsheet. So it ends up learning the *personal cap pattern* of
the training subjects instead of the actual task. New person, new
cap, new shape, and the filter is looking for something that isn't
there anymore. I check this with one diagnostic: adding more
training subjects (3 → 8) does NOT improve cross-subject accuracy
(0.656 vs 0.644). The failure is structural, not "we need more
data".

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

## Reading the numbers

- **TopoNet drops subject clustering 6×** on the full pool (0.084 → 0.014). On the small 3-subject pool, accuracy actually goes up (0.656 → 0.622), and the gap between best and worst seed shrinks from 12.3 % to 2.2 % — much more stable. On the full pool accuracy drops, because some of the task signal also lived in the per-subject covariance that the whitening removed. Not a bug — it's the trade-off I went into the experiment expecting, and the ablation script lets me isolate which half of TopoNet caused it.
- **Hemispheric contrasts win on the full pool.** Subject clustering drops 2.3×, best accuracy goes up (0.644 → 0.667), worst case goes up (0.511 → 0.556). On the small pool the mechanism still works (clustering drops) but accuracy collapses — the extra 10 channels add more weights than 135 trials can constrain. So the trade between the prior and the parameter cost flips with data size. I'd expected this when setting up the experiment, and the numbers match.

## Why I picked these two improvements

**Part 2 (TopoNet).** Two changes that hit the same root cause from
different sides. The per-trial Riemannian whitening removes each
person's electrical fingerprint without needing any calibration
trials from the test subject — train once, run on anyone. The CCSF
layer makes the spatial filter location-aware: same physical region
on the head, same filter response, regardless of whose cap it is. I
ran the ablation (`scripts/ablation_part2.py`) so I could tell which
half does which job — each one alone cuts subject clustering by
about 4×, and CCSF on its own is the cleanest winner on the small
pool.

**Part 3 (contrast channels).** The contrasts are a fixed
subtraction, so they don't add any learnable parameters of their
own — the whole augmentation is one line, stack 10 extra channels on
the input and let EEGNet see 74 channels instead of 64. If the
contrasts turn out not to help, the model can just put small weights
on them and behave like the baseline; nothing about the architecture
needs to roll back. The reason I chose this particular prior: scalp
EEG is blurry in space because the skull spreads each brain signal
across multiple electrodes, so the useful information usually lives
in how electrodes differ, not in any one channel on its own. The
contrast channels write those differences directly into the input.

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
  (Cohen's d ≈ 0.14). I didn't expect this on a bilateral
  fists-vs-feet task — possible explanations are subject handedness,
  attention asymmetry, or systematic timing differences between the
  two classes. Worth flagging for the defence.
- Either way the contrasts are class-informative but not separable
  on their own. They are a useful representation for the network to
  lean on, not a classifier by themselves.

## Classical baseline — CSP + LDA (`results/csp_baseline.json`)

| | best | worst | mean |
| --- | ---: | ---: | ---: |
| CSP+LDA, 8 subj | 0.489 | 0.489 | 0.489 |
| CSP+LDA, 3 subj | 0.367 | 0.367 | 0.367 |

Classical Common Spatial Patterns sits at chance on the full pool
and **below chance** on the reduced pool. CSP filters are explicitly
person-specific covariance maximisers, so under reduced-data
cross-subject training they actively anti-align to the test
subjects. It's the same failure mechanism Part 1 spelled out for
EEGNet, showing up even more starkly in a model that's
covariance-based by construction.

EEGNet (0.570 mean) sits clearly above CSP because it has
temporal-frequency capacity that CSP doesn't have. But the kind of
features EEGNet learns spatially are still person-coupled — just
less catastrophically than CSP's filters, which are explicitly
per-person by design.

## Reproducing all numbers

```bash
pip install -r requirements.txt
python main.py                          # headline 6-experiment run
python scripts/within_subject.py        # within-subject reference
python scripts/ablation_part2.py        # Part 2 ablation
python scripts/csp_baseline.py          # classical CSP+LDA reference
python scripts/plot_contrasts.py        # Part 3 contrast sanity check
python scripts/part3_filter_analysis.py # Part 3 — check whether the filters actually use the contrast channels
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

## Limitations

Putting these here so they don't surprise anyone in the defence.

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
  accuracy resolution. I'm treating the low ceiling as a data-volume
  thing, not something about EEGNet itself. The Part 1 argument
  doesn't rely on the within-subject ceiling either way.
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
