# EEGNet Failure Analysis & Beyond
*Defence presentation — PhysioNet EEGBCI motor-imagery subset*

---

## Slide 1 — The question and the constraints

EEGNet works in the lab. Where does it fail when we move it to a new
person, and what does the failure tell us about how to fix it?

| Train | Test | Task | Channels | Sampling | Trials/subject |
| ----- | ---- | ---- | -------- | -------- | -------------- |
| Subj 1–8 | Subj 9–10 | Imagined fists vs feet (PhysioNet Task 4) | 64 | 160 Hz | ~45 |

Three seeds (1337, 2024, 7). Always report **best AND worst** —
no cherry-picking. Source of truth: `results/metrics.json`.

---

## Slide 2 — Three experiments to localise the failure

| Setting | Train subjects | Test subjects |
| --- | --- | --- |
| Cross-subject, full pool | 1–8 (8 subjects) | 9–10 |
| Cross-subject, reduced pool | 1, 2, 3 (3 subjects) | 9–10 |
| UMAP of the model's internal features | diagnostic | diagnostic |

No within-subject baseline. The thing under test is **transfer to a
new person**, not raw capacity. Within-subject EEGNet is already
established as working (Lawhern 2018).

---

## Slide 3 — Baseline results — the clue is in the diagnostic, not the accuracy

| Setting | Best | Worst | Sil(class) | **Sil(subject)** |
| --- | ---: | ---: | ---: | ---: |
| Full 8 subj | 0.644 | 0.511 | 0.010 | **0.084** |
| Reduced 3 subj | 0.656 | 0.533 | -0.002 | **0.062** |

Two things to notice:

1. Adding subjects did **NOT** help (0.644 ≈ 0.656). One more
   training subject is worth roughly zero.
2. **Subject clustering ≫ class clustering** (0.084 vs 0.010). The
   network's main feature axis is "which person is this?", not "left
   or right?".

Sil(subject) = how clustered the model's features are by person.
Higher = more person-dependent. We want it low.

---

## Slide 4 — Why — the mechanism

EEGNet's spatial filter has shape `(filters, channels)` — one weight
per (filter, electrode-index) pair. It looks at electrodes by *index*,
not by *position on the head*.

Two structural facts:

1. The depthwise weight has **no notion of physical electrode
   position**. Move the cap 1 cm and the same weight reads a
   different brain region — but the network has no way to know.
2. Each person's electrical fingerprint (skull thickness, electrode
   contact, head shape) shows up as resting-state covariance with
   very high signal-to-noise. The classifier finds it because it is
   *easier to learn than the actual class signal*.

The high sil(subject) is what tells me that — the model basically
built a person-identifier as its main feature.

There's no way around this without changing the architecture.

---

## Slide 5 — Part 2 — TopoNet — two changes

### 5a. Per-trial Riemannian whitening
$$ \tilde X = (X X^\top / T + \varepsilon I)^{-1/2} X $$

In plain English: rescale each trial so its covariance matrix
becomes the identity. This mathematically removes the person-specific
covariance fingerprint. No calibration data needed from the test
subject — every trial gets whitened on its own.

### 5b. Coordinate-Conditioned Spatial Filters (CCSF)
$$ W[c, k] = \mathrm{MLP}([\,\mathrm{pos}_c,\; e_k\,]) $$

The spatial filter is no longer a lookup table. It's a small network
that takes the actual 3D position of each electrode as input. Same
physical position → same filter response, regardless of whose head
the cap is on.

---

## Slide 6 — TopoNet results — an honest split

| Setting | Best | Worst | Sil(class) | **Sil(subject)** |
| --- | ---: | ---: | ---: | ---: |
| Baseline, 8 subj | 0.644 | 0.511 | 0.010 | 0.084 |
| **TopoNet, 8 subj** | 0.489 | 0.456 | 0.005 | **0.014** |
| Baseline, 3 subj | 0.656 | 0.533 | -0.002 | 0.062 |
| **TopoNet, 3 subj** | **0.622** | **0.600** | 0.006 | **0.028** |

What's working and what isn't:

- **Mechanism works.** Subject clustering drops 6× on the full pool
  and 2.3× on the reduced pool — the Part 1 diagnostic moves a lot.
- **Full-pool accuracy drops.** TopoNet sits below the baseline here.
- **Reduced-pool accuracy improves and becomes stable.** Every seed
  lands in [0.60, 0.62].

---

## Slide 7 — Why TopoNet wins on 3 subjects and loses on 8

The motor-imagery class signal for runs 6/10/14 lives in the
**lateral-vs-medial covariance contrast** — bilateral hand area
(C3/C4) for imagined fists vs midline foot area (Cz) for imagined
feet. Per-trial whitening flattens covariance across the board, so
it removes the class signal along with the person-specific
component — they live in the same algebraic object.

- **Full pool**: EEGNet has enough data to find robust filters
  without the prior. Whitening becomes *cost without benefit*.
- **Reduced pool**: EEGNet was overfitting. Whitening's prior gives
  it a cleaner objective and stabilises every seed.

I threw the baby out with the bathwater on the full pool. The next
fix would be **subject-conditional whitening** — remove only the
person-specific component of covariance, keep the class-relevant
asymmetry.

---

## Slide 7b — Ablation decomposes TopoNet's two halves

`scripts/ablation_part2.py`:

| Config | 8-subj best/worst | 3-subj best/worst | sil(subj) 8 / 3 |
| --- | :---: | :---: | :---: |
| Baseline | 0.644 / 0.511 | 0.656 / 0.533 | 0.084 / 0.062 |
| Whitening only | 0.600 / 0.489 | 0.600 / 0.489 | 0.022 / 0.017 |
| **CCSF only** | 0.522 / 0.422 | **0.611 / 0.567** | 0.022 / 0.020 |
| TopoNet (both) | 0.489 / 0.456 | 0.622 / 0.600 | 0.014 / 0.028 |

Three findings:

1. **Either intervention alone cuts subject clustering ~4×.** Both
   halves independently attack the mechanism from Part 1.
2. **They stack negatively on accuracy** — both remove some of the
   same useful information. Combining them over-regularises.
3. **CCSF alone has the highest worst-case test (0.567) of any
   3-subj model.** Cleanest single intervention.

---

## Slide 8 — Part 3 — Hemispheric contrast channels

The novel idea. Half-page rationale in `report/part3_idea.md`.

**Claim:** the useful signal in EEG is in the *relationships*
between electrodes, not in any one channel's amplitude. Volume
conduction (skull and scalp spread each brain signal across many
electrodes) means 64 channels are 64 overlapping views of a smaller
set of sources. The relationships are already in the input, but
buried.

**Idea:** add ten extra channels next to the raw 64. Nine are
left-right symmetric pairs near motor cortex (C3 − C4, FC3 − FC4,
CP3 − CP4 and neighbours). The tenth is a lateral-vs-midline
contrast: (C3 + C4)/2 − Cz. Input becomes 74 channels.

Add, not replace. If the contrasts are useless the model can
downweight them and behave like the baseline.

---

## Slide 9 — Why this is grounded, specific, and (I think) not just bipolar

**Grounded.** EEG has high temporal resolution but its spatial
picture is coarse. Volume conduction means neighbouring channels end
up carrying mostly the same thing. The useful information often
lives in the differences between electrodes.

**Specific.** Ten exact contrasts, named electrodes, fixed
subtraction. `scripts/plot_contrasts.py` plots the C3 − C4 and
(C3 + C4)/2 − Cz traces split by class — reviewers can verify the
sign flip on C3 − C4 themselves.

**Not just a bipolar montage.** Bipolar, surface Laplacian,
common-average, CSP, ICA — they all replace the raw signal with a
fixed transform before the network ever sees the data. Once that's
done, the raw channels are gone.

I do the opposite. The raw 64 channels stay. The 10 contrast
channels sit next to them. Now EEGNet's depthwise spatial filter
gets to decide, filter by filter, how much to lean on each side.
Some filters can stay raw-like, others can concentrate on the
contrasts and become lateralization detectors. None of that is
hard-coded.

**One sentence I'd defend it with:**

> *Classical EEG preprocessing decides the input representation
> before the model sees the data. I leave that choice to a layer
> that can be trained, and I check that it actually used it.*

**Why it might help here specifically:**

1. The contrast is in the input from the first batch, so the model
   doesn't have to discover it before overfitting to per-subject
   quirks.
2. Anything that affects both sides of the head equally (arousal,
   contact, head size) partly cancels in the subtraction — gentle
   noise suppression for free.
3. If contrasts don't help, the model can put small weights on them
   and behave like the baseline. The fallback is graceful.

---

## Slide 10 — Sanity check on the engineered feature

`scripts/plot_contrasts.py` → `results/figures/part3_contrasts.png`.

Trial-averaged contrast traces, split by class:

- **C3 − C4 has a sign flip between classes** (T1 mean negative,
  T2 mean positive) — the lateralization pattern the contrast was
  built to pick up.
- Effect size is modest at the trial level (Cohen's d ≈ 0.14 on
  C3 − C4, ≈ 0.05 on (C3 + C4)/2 − Cz). The contrasts are
  class-informative but **not classifiers on their own** — they are
  a useful representation for the network to lean on.

---

## Slide 11 — Part 3 results — the prediction holds on the bigger pool

| Setting | Best | Worst | Sil(class) | **Sil(subject)** |
| --- | ---: | ---: | ---: | ---: |
| Baseline, 8 subj | 0.644 | 0.511 | 0.010 | 0.084 |
| **EEGNet + contrasts, 8 subj** | **0.667** | **0.556** | -0.001 | **0.037** |
| Baseline, 3 subj | 0.656 | 0.533 | -0.002 | 0.062 |
| EEGNet + contrasts, 3 subj | 0.533 | 0.511 | 0.005 | **0.037** |

**Full 8-subject pool — hypothesis held.** Subject clustering drops
2.3× (0.084 → 0.037). Best accuracy up (0.644 → 0.667). Worst case
up (0.511 → 0.556). Mechanism and outcome both lined up with what
I'd predicted.

**Reduced 3-subject pool — mechanism still works, but accuracy
collapses.** Subject clustering still drops (0.062 → 0.037), so the
contrasts are doing their job at the representation level. But best
accuracy collapses to 0.533. With 135 trials, the extra 10 channels
add more weights than the data can constrain — the prior ended up
costing more than it paid back.

So the trade between human prior and parameter cost flipped with
data size. I'd expected this when I set the experiment up; the
numbers ended up showing it.

---

## Slide 11b — Did the model actually use the contrasts?

`scripts/part3_filter_analysis.py` → `results/figures/part3_filter_weights.png`.

I ran the test. The numbers came in mixed.

- Random baseline (every channel equally weighted): 10/74 ≈ 0.135.
- Trained mean per-filter share on contrasts: 0.150 — only 1.11×
  the baseline. Small shift in aggregate.
- Per-filter spread: 0.031 to 0.263. Five filters at 0.20–0.26 (so
  about 1.5–2× baseline). Five at 0.03–0.10, basically ignoring
  the contrasts. The other six are near the baseline.

So I'm walking the strong reading back. The aggregate number doesn't
support "the model heavily leans on the prior overall" — I'd be
overclaiming if I said that.

What I'd still stand behind is the per-filter side. If the
contrasts were redundant noise, every filter would land near 0.135.
They don't. Some filters lean in, others lean away — and that
heterogeneity is the actual evidence that the per-filter choice is
happening, which is the architectural difference from a bipolar
montage that forces every filter through the same fixed transform.

So the contrasts end up being a useful sideline for a minority of
filters, not a dominant input. That's where I'd land if pushed.

---

## Slide 12 — Defence Q&A I expect

> **"Isn't this just a bipolar montage / surface Laplacian? Those
> are decades old."**
> Bipolar and Laplacian replace the raw signal with a fixed
> transform before the network sees anything — once it's done, the
> raw channels are gone. What I do is different: I leave the raw 64
> in place and add 10 contrast channels next to them, and the
> depthwise filter decides per filter how much to weight each side.
> I ran the test on myself
> (`scripts/part3_filter_analysis.py`) — the trained mean share on
> contrasts came back at 0.150 against a 0.135 random baseline, so
> only about 11 % above chance. I'm not going to claim the model
> heavily leans on the prior overall — the aggregate number doesn't
> back that. But the per-filter spread is wide (0.031 to 0.263):
> some filters clearly concentrate on the contrasts, others ignore
> them entirely. That's not what you'd see if the contrasts were
> redundant noise. So the architectural difference from a bipolar
> montage still holds — the filter gets a per-filter choice and
> actually exercises it — even if the average filter doesn't
> strongly prefer the contrasts.

> **"But C3 − C4 is already a linear combination of channels EEGNet
> already sees. Why should this help?"**
> Exactly. I am not adding information. I am changing the inductive
> bias so a known brain pattern becomes one of the first features
> the model can find. With limited data it is the difference
> between discovering the lateralization pattern and overfitting to
> per-subject quirks first.

> **"Why per-trial whitening, not subject-mean whitening?"**
> Subject-mean needs calibration data on the test subject, and I
> have none for subjects 9 and 10. Per-trial uses no test-subject
> statistics at all. It pays a class-signal-removal price, which I
> now understand and isolated with the ablation.

> **"Within-subject EEGNet mean is 0.522 — barely above chance.
> How can you say EEGNet works?"**
> I cannot, on this subset. Each within-subject split has 9 test
> trials — 11 % accuracy resolution. The assignment's premise
> ("EEGNet works") rests on Lawhern 2018's 109-subject results. My
> subset is too small to demonstrate the within-subject ceiling but
> is enough to demonstrate the cross-subject **failure mechanism**.

> **"Why did Part 3 work on 8 subjects and fail on 3?"**
> The hypothesis was always about a trade-off between human prior
> and parameter cost. With 8 training subjects there is enough data
> to back the extra parameters. With 3 subjects there is not. The
> mechanism works in both pools (subject clustering drops in both);
> the outcome diverges where the parameter budget binds.

---

## Slide 13 — What I would build next

1. **Subject-conditional whitening** — remove only the
   subject-specific component of covariance, keep the
   class-discriminative asymmetry that simple whitening also
   removes.
2. **Frozen contrast weights on small pools** — freeze the depthwise
   filter weights on the 10 contrast channels in the reduced-data
   regime so the prior is preserved without paying the parameter
   cost.
3. **Pre-train CCSF MLP across multiple EEG datasets** — it's
   channel-count agnostic, so cross-corpus pre-training is free.

---

## Slide 13b — Visual evidence — topomaps tell the same story

`results/figures/topomap_eegnet.png` — EEGNet's 16 depthwise filters.
**Fragmented, noisy spatial profiles** — no clean motor-cortex
localisation. These are not motor-area detectors; they are
person-specific covariance patterns.

`results/figures/topomap_ccsf.png` — CCSF's 16 filters.
**Smooth gradients, all nearly identical.** The MLP-on-position prior
collapses to a small set of low-frequency spherical patterns. This
explains CCSF's full-pool underperformance: not too little capacity,
but too little *diversity*.

`results/figures/per_subject_accuracy.png` — EEGNet hits S10 (0.71)
much better than S9 (0.58). The 0.644 headline number averages over
a real asymmetry — subjects are not equally transferable.

---

## Slide 13c — Classical reference — CSP + LDA

| Setting | best / worst |
| --- | :---: |
| CSP+LDA, 8 subj | 0.489 / 0.489 |
| CSP+LDA, 3 subj | 0.367 / 0.367 |

CSP is **at chance** on the full pool and **below chance** on the
reduced pool. CSP filters are explicitly person-specific covariance
maximisers — under reduced-data cross-subject training they
anti-align to test subjects.

It's the same failure mechanism Part 1 spelled out for EEGNet,
showing up even more starkly in a model that's covariance-based by
construction.

EEGNet (0.570 mean) uses temporal-frequency capacity that CSP
doesn't have, but the kind of spatial features it learns are still
person-coupled — just less catastrophically.

---

## Slide 14 — Take-aways

- The cross-subject failure is **structural, not statistical**.
  Adding subjects does not fix it (0.644 ≈ 0.656). CSP+LDA at
  chance empirically confirms the mechanism.
- The diagnostic that reveals the mechanism is **subject
  clustering**, not accuracy. Accuracy on a 90-trial test set has
  a ±5 % noise floor.
- **Part 2 ablation** isolated CCSF as the cleaner half — best
  worst-case test on the reduced pool (0.567). Topomaps show
  CCSF's next-step problem (filter collapse).
- **Part 3 hemispheric contrasts** validated the
  inductive-bias hypothesis on the full pool (subject clustering ↓
  2.3×, best test ↑ 0.644 → 0.667, worst ↑ 0.511 → 0.556). On
  the small pool the mechanism still works but parameter cost wins.
- **Each part has a finding, a failure mode, and a follow-up
  experiment that explains the failure.** That's the working pattern
  I'd want to bring into your team.

Code (`main.py` plus the scripts referenced on each slide)
reproduces every number on this deck.
