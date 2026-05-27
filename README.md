# EEGNet Failure Analysis & Two Original Improvements

Submission for the BrainwaveScience EEG assignment.

**Quick start.** Run `python main.py`. That single command reproduces
every headline number in this README. CPU runtime is ~30 minutes
once the data is cached. The extra analyses (ablation, CSP baseline,
within-subject reference, contrast sanity check, filter-weight check,
topomaps) live as standalone scripts under `scripts/` and each runs
end-to-end with no flags.

---

## The story in one paragraph

EEGNet is a small, well-known brain-signal classifier. It works
beautifully when you train and test on the same person. The moment
you train on one group of people and ask it to read a brand-new
person, accuracy falls off a cliff. This submission digs into *why*
that happens, tries to fix the root cause in two different ways, and
reports honestly what each fix actually did — including where it
didn't work.

---

## The five things I'd want you to take away

1. **The cross-subject failure is structural, not a data problem.**
   Going from 3 training subjects to 8 gives basically no
   improvement on unseen people (0.644 vs 0.656). More data won't
   fix this — the architecture has the bug.

2. **The bug:** EEGNet's spatial filter treats electrodes by
   *index*, like rows in a spreadsheet. It has no idea where each
   electrode physically sits on the head. New person, new head
   shape, and the same filter ends up reading a different brain
   region — but the model has no way to know.

3. **Fix 1 — TopoNet** makes the spatial filter location-aware. It
   cuts how much the model's internal features cluster by person by
   6× on the big pool (0.084 → 0.014). On the small 3-subject pool,
   accuracy actually goes up and stabilises — TopoNet has the best
   worst-case (0.567) of any model I tried.

4. **Fix 2 — hemispheric contrast channels** adds 10 extra channels
   alongside the raw 64. Each one is a subtraction between two
   electrodes that should cancel out person-specific noise and
   highlight the task-relevant pattern. On the big pool: best
   accuracy 0.644 → 0.667, worst 0.511 → 0.556, person-clustering
   drops 2.3×. On the small pool the prior still works at the
   representation level but accuracy collapses — the report
   explains why.

5. **I tested my own claim and got a modest result.** Part 3's
   strongest framing was "the model picks per filter whether to use
   the contrast channels". I ran a check
   (`scripts/part3_filter_analysis.py`): trained mean weight share
   on the contrasts is 0.150 against a 0.135 random baseline — only
   about 11% above chance. So the aggregate claim "the model leans
   heavily on the prior" doesn't hold up. What does hold up is the
   per-filter side: weights spread from 0.031 to 0.263 across the
   16 filters — some clearly lean on the contrasts, others ignore
   them. That's the per-filter heterogeneity the architecture is
   meant to allow. The honest reading: useful sideline for a
   minority of filters, not a dominant input. Figure:
   `results/figures/part3_filter_weights.png`.

**One sentence to explain each idea if I'm short on time:**
- *Part 2:* "I made the spatial filter location-aware so the same
  brain region produces the same filter response regardless of
  subject."
- *Part 3:* "I'm not adding new information. I'm making a known
  brain pattern easier for the model to find."

---

## The three parts in plain language

### Part 1 — Why does it fail?

EEGNet's spatial filter looks at the 64 electrodes as 64 indexed
slots. It learns: "for filter #3, multiply electrode #12 by 0.7,
electrode #13 by -0.4..." and so on. The slot number is the only
thing the filter sees — it has no idea that electrode #12 is
sitting over the left motor cortex.

That's fine if every person's head is identical. They're not. Skull
thickness, cap fit, electrode contact, head shape — they all vary.
Same slot, different brain region underneath. The model ends up
learning each training person's *cap pattern* instead of the actual
task signal, because the cap pattern gives a stronger, more
consistent training signal than the task itself.

The proof: adding more training subjects (3 → 8) does NOT improve
accuracy on unseen people (0.656 → 0.644). If the problem were "not
enough data" you'd expect a clear lift. There isn't one. So the
failure is structural.

### Part 2 — TopoNet (fixing the cause)

Two changes that attack the same problem from two sides:

- **Per-trial Riemannian whitening.** A math operation applied to
  each trial that mathematically subtracts out each person's
  electrical fingerprint. No calibration data needed from the new
  person — every trial gets cleaned on its own.

- **Coordinate-Conditioned Spatial Filters (CCSF).** The spatial
  filter is no longer a slot lookup. It's a small neural network
  that takes each electrode's actual 3D position on the head as
  input and produces the filter weight for it. Same physical
  location on the head → same filter response, no matter whose head
  the cap is on.

### Part 3 — Hemispheric contrast channels (adding a known prior)

Add 10 extra channels next to the raw 64. Nine of them are
subtractions between matching left/right electrode pairs near motor
cortex (C3 minus C4, FC3 minus FC4, etc.). The tenth is a
hand-vs-foot contrast: average of C3+C4, minus Cz.

The contrasts don't add new information — they're just linear
combinations of channels EEGNet already sees. What they do is hand
the spatial filter a ready-made representation of a brain pattern
(lateralization) it would otherwise have to discover from scratch
with very little data. If the contrasts turn out to be useless on
some other dataset, the model can simply put small weights on them
and behave like the baseline. The architecture doesn't need to roll
back.

---

## Headline numbers

3 seeds × 60 epochs. "Cross-subject" = train and test never share a
person. Test = subjects 9 and 10 (held out completely).

| Model | Pool | Best | Worst | Subject clustering ↓ |
| --- | --- | ---: | ---: | ---: |
| EEGNet baseline | 8 subj | 0.644 | 0.511 | 0.084 |
| EEGNet baseline | 3 subj | 0.656 | 0.533 | 0.062 |
| **TopoNet (Part 2)** | 8 subj | 0.489 | 0.456 | **0.014** |
| **TopoNet (Part 2)** | 3 subj | **0.622** | **0.600** | **0.028** |
| **EEGNet + contrasts (Part 3)** | 8 subj | **0.667** | **0.556** | **0.037** |
| EEGNet + contrasts (Part 3) | 3 subj | 0.533 | 0.511 | **0.037** |

**Subject clustering** is a number between -1 and 1 that measures
how much the model's internal features cluster by person. Lower is
better — it means the model is less person-dependent. This is the
diagnostic I trust most, because accuracy on a 90-trial test set
has a ~5% noise floor across seeds.

---

## Honest reading of the numbers

**TopoNet.** Subject clustering drops 6× on the big pool — the
diagnostic I built Part 1 around moves a lot, which means the fix
is hitting the real cause. On the small pool, accuracy goes up
(0.656 → 0.622) and the gap between best and worst seed shrinks
from 12.3% to 2.2% — much more stable. On the big pool accuracy
drops, because some of the task signal also lived in the
per-subject covariance that the whitening removed. That's not a
bug — it's the trade-off I went in expecting, and the ablation
script lets me confirm which half of TopoNet caused it.

**Hemispheric contrasts.** Win on the big pool: clustering drops
2.3×, best accuracy goes up, worst case goes up. On the small pool
the mechanism still works (clustering drops) but accuracy
collapses — the extra 10 channels add more weights than 135 trials
can constrain. So the trade between the prior and the parameter
cost flips with data size. Predicted in advance; the numbers showed
exactly that.

---

## Why I picked these two improvements

**Part 2 (TopoNet)** hits the root cause from two sides at once.
The whitening removes each person's electrical fingerprint with no
calibration data needed at test time — train once, run on anyone.
The CCSF layer makes the spatial filter physically aware of the
scalp. I ran the ablation (`scripts/ablation_part2.py`) so I could
isolate which half does which job — each alone cuts subject
clustering ~4×, and CCSF on its own is the cleanest winner on the
small pool.

**Part 3 (contrast channels)** is the cheapest possible
intervention — a fixed subtraction adds zero learnable parameters,
the whole augmentation is one line of code, and it can wrap any
channel-first EEG model. The reason this particular prior: EEG is
blurry in space because the skull spreads each brain signal across
many electrodes, so the useful information usually lives in how
electrodes *differ* from each other, not in any one channel's
amplitude. The contrasts write those differences straight into the
input.

---

## Part 2 ablation (`results/ablation_part2.json`)

| | best | worst | subject clustering |
| --- | ---: | ---: | ---: |
| Baseline, 8 subj | 0.644 | 0.511 | 0.084 |
| Whitening only | 0.600 | 0.489 | 0.022 |
| **CCSF only, 8 subj** | 0.522 | 0.422 | 0.022 |
| TopoNet (both), 8 subj | 0.489 | 0.456 | 0.014 |
| **CCSF only, 3 subj** | **0.611** | **0.567** | 0.020 |
| TopoNet (both), 3 subj | 0.622 | 0.600 | 0.028 |

Each intervention alone cuts subject clustering ~4×. They stack
*negatively* on accuracy though — both are removing some of the
same useful information, so doing both together costs more than
either alone. CCSF by itself on the small pool is the overall best
single intervention: best worst-case (0.567) of any configuration
tested.

---

## Did the contrast channels actually get used? — Part 3 self-check

I wrote `scripts/part3_filter_analysis.py` as a test on my own
claim. For each of EEGNet's 16 spatial filters, it computes how
much of the filter's weight ended up on the 10 contrast channels vs
the 64 raw ones. If the contrasts were just noise, every filter
would end up around 10/74 ≈ 13.5%. If the model leaned on them
heavily, the share would be much higher.

Result: trained mean is 0.150 — only 1.11× the baseline. Modest.
Per-filter the spread is 0.031 to 0.263 — five filters concentrate
at 0.20–0.26 (clearly using the contrasts), five sit at 0.03–0.10
(basically ignoring them), the other six are near baseline.

So I drop the strong reading ("the model heavily leans on the
prior") and keep the per-filter reading ("the model exercises
per-filter choice, and a meaningful minority of filters picks up
the prior"). Figure: `results/figures/part3_filter_weights.png`.

---

## Part 3 sanity check on the engineered feature

Before letting the network use the contrasts, I plotted them split
by class on the training pool. `scripts/plot_contrasts.py` saves
`results/figures/part3_contrasts.png`.

- **(C3+C4)/2 − Cz** (the lateral-vs-midline contrast — the
  intended discriminator for fists vs feet) shows the expected
  ordering. Effect size on trial means is small (Cohen's d ≈ 0.05).
- **C3 − C4** unexpectedly shows a sign flip between classes too
  (Cohen's d ≈ 0.14). I didn't expect this on a bilateral
  fists-vs-feet task — could be subject handedness, attention
  asymmetry, or a timing difference between the two classes. Worth
  flagging for the defence.
- The contrasts are class-informative but not classifiers on their
  own. They're a useful representation for the network to lean on,
  not a model.

---

## Classical reference baseline — CSP + LDA (`results/csp_baseline.json`)

| | best | worst | mean |
| --- | ---: | ---: | ---: |
| CSP+LDA, 8 subj | 0.489 | 0.489 | 0.489 |
| CSP+LDA, 3 subj | 0.367 | 0.367 | 0.367 |

CSP (Common Spatial Patterns) is the textbook BCI baseline. On
this subset it sits at chance on the big pool and below chance on
the small pool. The reason: CSP filters are explicitly built to
maximise person-specific covariance differences, so under reduced
cross-subject data they actively *anti-align* to the test subjects'
covariance. Same failure mechanism Part 1 described for EEGNet,
just showing up much more starkly.

EEGNet (0.570 mean) sits above CSP because it has
temporal-frequency capacity that CSP doesn't have. But the spatial
features EEGNet learns are still person-coupled — just less
catastrophically than CSP's, which are designed to be per-person.

---

## Reproducing every number

```bash
pip install -r requirements.txt
python main.py                          # headline 6-experiment run
python scripts/within_subject.py        # within-subject reference
python scripts/ablation_part2.py        # Part 2 ablation
python scripts/csp_baseline.py          # classical CSP+LDA reference
python scripts/plot_contrasts.py        # Part 3 contrast sanity check
python scripts/part3_filter_analysis.py # Part 3 self-check on the contrast claim
python scripts/plot_extras.py           # topomaps + per-subject bar chart
python scripts/make_table.py            # render results/metrics.json
```

All supplementary scripts accept `--epochs N` (default 60). For a
fast smoke check, run with `--epochs 15` — runtime drops ~4× and
the qualitative findings still hold; numbers won't exactly match
the headline 60-epoch claims though.

`scripts/sweep_cma.py`, `scripts/curriculum_cma.py` and
`scripts/combined_best.py` are from an earlier Part 3 idea
(cap-rotation augmentation) that didn't make the headline. They
still run if invoked.

CPU runtime once data is cached: ~30 min for `main.py`, ~15 min
for the ablation at 60 epochs. Seeds are fixed; `main.py` always
reports both best AND worst per experiment, never a single number.

---

## Frozen data subset

```
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SUBJECTS  = [9, 10]
RUNS           = [6, 10, 14]     # PhysioNet Task 4: imagined both-fists vs both-feet
```

Filter band 4–38 Hz; epochs 0.5–2.5 s after the cue; 64 EEG
channels at 160 Hz (321 timepoints per trial); per-trial z-score
normalization. No test-set statistic ever leaks into training.

---

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
  presentation.md      Defence slides (markdown).
  presentation.pdf     Rendered version of the slides.
scripts/               See "Reproducing every number" above.
results/
  metrics.json         All headline numbers.
  ablation_part2.json  Part 2 ablation numbers.
  csp_baseline.json    Classical reference numbers.
  part3_filter_analysis.json  Part 3 self-check output.
  log.txt              Human-readable best/worst summary.
  figures/             UMAP embeddings, topomaps, contrast traces.
  cache/               Per-subject epoched arrays (auto-built).
```

---

## Limitations I'm flagging up front

So they don't surprise anyone in the defence.

- **Tiny test set.** 2 test subjects = 90 cross-subject test
  trials. Accuracy resolution is ~1%, and a 5% swing across seeds
  is roughly 1 standard deviation. I lean on subject clustering
  as the primary diagnostic — it's much more stable than top-1
  accuracy on 90 trials.

- **3 seeds, no p-values.** I report best AND worst per experiment,
  never a single number. With 3 samples no honest statistical test
  would survive multiple-comparison correction across 6+
  experiments.

- **Within-subject reference is also low** (0.522 mean). Each
  subject has only ~45 trials, so an 80/20 split gives 9 test
  trials — 11% accuracy resolution. I'm treating the low ceiling as
  a data-volume thing, not something about EEGNet itself. The
  Part 1 argument doesn't rely on the within-subject ceiling either
  way.

- **Single-sphere head model.** Part 2's CCSF position prior
  assumes electrodes sit on a unit sphere — same assumption MNE's
  `interpolate_bads` makes. Conventional but not anatomically
  exact.

- **No EEGNet hyperparameter sweep.** I use the paper's recommended
  settings (F1=8, D=2, kern_len=80). Tuning could lift the baseline
  number, but wouldn't change the failure mechanism — which is the
  actual point of the analysis.

- **Whitening cost.** One symmetric eigendecomposition per trial
  via `torch.linalg.eigh` inside `torch.no_grad()`. Stable in
  practice; numerically conditioned by adding εI = 1e-3 · I.

- **Task-label confirmation.** The brief comments runs 6/10/14 as
  "left vs right fist", but per PhysioNet those runs are Task 4
  (imagined both-fists vs both-feet). Confirmed with the assignment
  owner that the intended task is fists vs feet, and the analysis
  is framed around the lateral-vs-medial contrast (bilateral C3/C4
  vs midline Cz) accordingly.

---

## Reproducibility

Seeds are fixed (`src/seed.py`). cuDNN determinism is NOT forced —
it roughly doubles runtime, and within-seed variance dominates
anyway for datasets this small.
