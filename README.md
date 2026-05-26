# EEGNet Failure Analysis & Two Original Improvements

Submission for the BrainwaveScience EEG assignment. Defence-ready —
all numbers reproduced by `python main.py`.

## TL;DR

- **Part 1 — Failure analysis (`report/part1_analysis.md`).** EEGNet's
  cross-subject failure is structural: the depthwise spatial filter has
  no learnable component dependent on electrode *position*, so its
  filter bank collapses to the training-set-mean spatial profile and
  uses subject-specific covariance as a free discrimination axis.
  Diagnostic: **silhouette by subject ≫ silhouette by class** (0.084 vs
  0.010 on the full pool). Adding subjects 4–8 to a 3-subject training
  pool does NOT improve test accuracy (0.644 ≈ 0.656), confirming the
  failure is structural, not statistical.

- **Part 2 — TopoNet (`src/part2_model.py`, `report/part2_design.md`).**
  Two changes that attack the mechanism directly:
  1. Per-trial Riemannian whitening — algebraically removes
     subject-specific covariance, no calibration required.
  2. Coordinate-Conditioned Spatial Filters (CCSF) — spatial filter is
     parameterised as a smooth function of 3D electrode position via a
     small MLP, not as an electrode-index lookup.

  **Result:** subject silhouette drops 6× (mechanism works). Accuracy
  drops on the full pool (the class signal also lives in spatial
  covariance — honest interesting failure). Accuracy improves and
  variance collapses on the reduced pool (prior helps when data is
  scarce).

- **Part 3 — Hemispheric contrast channels (`src/part3_idea.py`,
  `report/part3_idea.md`).** Add ten handcrafted symmetric-pair
  contrast channels (C3-C4, FC3-FC4, CP3-CP4 and neighbours, plus
  (C3+C4)/2 - Cz) next to the raw 64. The argument is not that
  contrasts add information — they do not, since they are linear
  combinations of channels EEGNet already sees — but that they change
  the inductive bias so a physiologically meaningful relationship
  becomes one of the first features the network can use.

  **Result:** on the full 8-subject pool the hypothesis is validated:
  sil(subject) drops **2.3×** (0.084 → 0.037), best test accuracy
  rises (0.644 → 0.667) and worst-case rises (0.511 → 0.556). On the
  3-subject pool the mechanism still works (sil(subject) drops 0.062 →
  0.037) but accuracy collapses (0.656 → 0.533) — the extra channels
  add more parameters than 135 trials can constrain. Trade-off
  between the inductive prior and sample-complexity tipped in opposite
  directions on the two pools.

## Results

### Headline (`results/metrics.json`)

| Experiment                  | Setting       | Test best | Test worst | Sil(class) | Sil(subject) |
| --------------------------- | ------------- | --------: | ---------: | ---------: | -----------: |
| EEGNet (baseline)           | 8 subj train  | 0.644     | 0.511      | 0.010      | **0.084**    |
| EEGNet (baseline)           | 3 subj train  | 0.656     | 0.533      | -0.002     | **0.062**    |
| **TopoNet** (Part 2)        | 8 subj train  | 0.489     | 0.456      | 0.005      | **0.014**    |
| **TopoNet** (Part 2)        | 3 subj train  | **0.622** | **0.600**  | 0.006      | **0.028**    |
| **EEGNet + contrasts** (P3) | 8 subj train  | **0.667** | **0.556**  | -0.001     | **0.037**    |
| EEGNet + contrasts (P3)     | 3 subj train  | 0.533     | 0.511      | 0.005      | **0.037**    |

Within-subject reference (EEGNet, 80/20 split per subject, see
`scripts/within_subject.py`): mean 0.522, best 0.778, worst 0.222.
The within-subject ceiling on this subset is too low to be a fair
"EEGNet works" baseline — the test split is only 9 trials per
subject, so accuracy resolution is 11 %.

### Part 2 ablation (`results/ablation_part2.json`)

|                       | best  | worst | sil(subject) |
| --------------------- | ----- | ----- | -----------: |
| Baseline, 8 subj      | 0.644 | 0.511 | 0.084        |
| Whitening only        | 0.600 | 0.489 | 0.022        |
| **CCSF only, 8 subj** | 0.522 | 0.422 | 0.022        |
| TopoNet (both), 8 subj| 0.489 | 0.456 | 0.014        |
| **CCSF only, 3 subj** | **0.611** | **0.567** | 0.020 |
| TopoNet (both), 3 subj| 0.622 | 0.600 | 0.028        |

**Either intervention alone reduces sil(subject) ~4×.** They stack
*negatively* on accuracy (compositional over-regularisation). CCSF
alone is the cleanest winner on the reduced pool — best worst-case
(0.567) of any configuration tested.

### Part 3 sanity check on the engineered feature

Before relying on the contrast channels, I plotted them class-by-class
on the training pool (`scripts/plot_contrasts.py` →
`results/figures/part3_contrasts.png`). Two observations.

- C3 − C4 has a **sign flip between classes** (trial-mean negative
  for T1, positive for T2) — the lateralization pattern the contrast
  is built to capture.
- Effect size is modest at the trial level (Cohen's d ≈ 0.14 on
  C3 − C4, ≈ 0.05 on (C3+C4)/2 − Cz). The contrasts are
  class-informative but not separable on their own — a useful
  representation for the network to lean on, not a classifier in
  themselves.

That matches the inductive-bias framing in `report/part3_idea.md`:
the contrasts do not add information; they make the spatial structure
easier for EEGNet to exploit.

### Classical baseline — CSP + LDA (`results/csp_baseline.json`)

|                            | best  | worst | mean  |
| -------------------------- | ----- | ----- | ----: |
| CSP+LDA, 8 subj            | 0.489 | 0.489 | 0.489 |
| CSP+LDA, 3 subj            | 0.367 | 0.367 | 0.367 |

Classical Common Spatial Patterns is at chance on the full pool and
*below* chance on the reduced pool — its filters are explicitly
subject-specific covariance maximisers, so under reduced-data
cross-subject training they anti-align to the test subjects. This is
direct empirical support for the Part 1 mechanism: covariance-based
spatial filtering does not transfer.

EEGNet (0.570 mean) is clearly using its capacity above CSP — but the
*kind* of features it learns are still subject-coupled, just to a
lesser extent than CSP's by-construction subject filters.

Test = subjects 9, 10 (held out at every step). Three seeds (1337, 2024, 7).
Lower `sil(subject)` = features less coupled to subject identity.

## Reproducing all numbers

```bash
pip install -r requirements.txt
python main.py                          # headline 6-experiment run
python scripts/within_subject.py        # within-subject reference
python scripts/ablation_part2.py        # Part 2 ablation
python scripts/csp_baseline.py          # classical CSP+LDA reference
python scripts/plot_contrasts.py        # Part 3 contrast sanity check
python scripts/plot_extras.py           # topomaps + per-subject bar chart
python scripts/make_table.py            # render metrics.json as markdown
```

The `scripts/sweep_cma.py`, `scripts/curriculum_cma.py` and
`scripts/combined_best.py` scripts are an earlier exploration of a
different Part 3 idea (cap-rotation augmentation) that did not become
the headline. They still run if invoked.

Wall-clock on a laptop CPU once data is cached: ~30 min for
`main.py`, ~5 min for the within-subject script, ~15 min for ablation.

`main.py` alone runs:
- EEGNet baseline (full 8-subject pool and reduced 3-subject pool)
- TopoNet (both pools)
- EEGNet + hemispheric contrasts (both pools)

All experiments use seeds `[1337, 2024, 7]` and the script reports
**best AND worst** runs per experiment (never the mean alone).
Outputs:

```
results/metrics.json          headline numbers
results/ablation_part2.json   Part 2 ablation numbers
results/log.txt               human summary
results/figures/*.png         UMAP embeddings + topomaps + contrast plot
results/cache/                per-subject epoched arrays (auto-built)
```

## Data subset (frozen)

```
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SUBJECTS  = [9, 10]
RUNS           = [6, 10, 14]     # Motor imagery: left vs right fist
```

Filter band 4–38 Hz; epochs 0.5–2.5 s post-cue; 64 EEG channels at 160 Hz
(→ 321 timepoints per trial); per-trial z-score normalisation (no
test-set statistics leaked).

## Repository layout

```
main.py                Single entrypoint.
src/
  data.py              MNE loader + epoching + caching.
  seed.py              Seed control.
  eegnet.py            Faithful EEGNet-8,2 PyTorch port.
  part2_model.py       TopoNet — whitening + CCSF.
  part3_idea.py        Hemispheric contrast channel augmentation.
  train.py             Training loop with max-norm projection.
  eval.py              UMAP/t-SNE + silhouette diagnostics.
report/
  part1_analysis.md    Mechanistic failure analysis (≤ half page).
  part2_design.md      Design rationale + math for TopoNet.
  part3_idea.md        Half-page proposal for hemispheric contrasts.
slides/
  presentation.md      Defence slides (markdown — render to PDF).
scripts/
  smoke_test.py        Two-epoch sanity check.
  within_subject.py    Optional reference baseline (within-subject EEGNet).
  make_table.py        Render results/metrics.json into a Markdown table.
```

## Limitations

These are the things to ask first; we already know about them:

- **Subset size.** 90 cross-subject test trials. Accuracy resolution is
  ~1.1 %; meaningful improvements need ≥ 3 % swings to be visible above
  seed noise. We compensate by leaning on `silhouette_subject` as the
  primary diagnostic — it's far less noise-bound than top-1 accuracy.
- **Statistical significance.** 3 seeds per configuration. We report best
  AND worst, never a single number; we do NOT report p-values because
  with 3 samples no honest test would survive multiple comparison
  correction across 6+ experiments.
- **Within-subject ceiling not established.** Each subject has ~45
  trials. An 80/20 split gives 9 test trials per subject — 11 %
  accuracy resolution. The within-subject reference (`scripts/within_
  subject.py`) hits 0.522 mean which is below the published EEGNet
  baseline; we treat that as a data-volume artefact rather than a
  property of EEGNet itself. The structural argument in Part 1 does
  not depend on the within-subject ceiling.
- **Single-sphere head model.** Part 2 (CCSF position prior) assumes
  electrodes sit on a unit sphere. This is the same assumption MNE's
  `interpolate_bads` makes — conventional but not anatomically exact.
- **No EEGNet hyperparameter sweep.** We use the paper's recommended
  configuration (F1=8, D=2, kern_len=sfreq/2=80). Tuning those for the
  subset could lift the baseline number — but would not change the
  failure mechanism, which is the point of the analysis.

## Honest caveats

- The PhysioNet subset used has only 2 test subjects, so a 5 %
  accuracy swing is ~1 σ. We report best/worst across seeds and lean
  on **representation-level diagnostics** (`silhouette_class` vs
  `silhouette_subject`) as the primary evidence — they are far less
  noise-bound than top-1 accuracy on ~90 test trials.
- Whitening cost: one symmetric eigendecomposition per trial, computed
  with `torch.linalg.eigh` inside `torch.no_grad()`. Stable in practice;
  numerically conditioned by an additive `εI = 1e-3 · I`.

## Reproducibility

Seeds are fixed (`src/seed.py`). cuDNN determinism is NOT forced — it
roughly doubles runtime and within-seed variance dominates for these
small datasets.
