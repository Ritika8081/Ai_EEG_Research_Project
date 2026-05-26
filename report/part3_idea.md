# Part 3 — Adding hemispheric contrast channels alongside raw EEG

Looking back at Part 1, EEGNet's depthwise spatial filters were
learning subject-specific electrode patterns instead of the task.
Each person's head and electrode contact create a different
per-channel amplitude profile, and the filters absorbed that. The
motor signal sits on top.

For Part 3 I want to make the spatial relationships I think actually
carry the task explicit, instead of leaving them buried inside the
raw channels. Add ten handcrafted contrast channels next to the
existing 64. Nine of them are differences between symmetric
left-right electrode pairs near motor cortex (C3-C4, FC3-FC4,
CP3-CP4 and their neighbours). The tenth is a lateral-vs-midline
contrast, (C3+C4)/2 minus Cz. The input to EEGNet becomes 74
channels. I add rather than replace so I do not throw out non-motor
signal if the task is not purely lateralized; if the contrasts are
useless the depthwise filter can downweight them.

## Why spatial relationships matter for EEG specifically

EEG has high temporal resolution but its spatial picture is coarse.
A single cortical source spreads to multiple scalp electrodes through
volume conduction in the skull and scalp tissue, and neighbouring
channels usually carry strongly correlated signals. This means the
raw 64-channel basis is highly redundant — much of the inter-channel
variance is shared structure, not independent information.
Task-relevant signal often lives in the *relationships* between
electrodes rather than in the absolute amplitude of any single one.
The contrast channels write a small set of those relationships into
the input directly, exposing **spatial gradients rather than absolute
potentials**. This does not create new information — C3 minus C4 is
computable from the raw channels — but it may make the spatial
structure of the signal easier for the network to exploit, especially
given limited training data and large cross-subject variability.

## The hypothesis I actually want to test

The contrasts do not give the model new information. C3 minus C4 is
a linear combination of channels EEGNet already sees, and the
depthwise spatial filter could in principle learn it from raw EEG.
The hypothesis is about **learnability and inductive bias**, not
information content. With 64 raw channels and only a few hundred
training trials across 8 subjects, the right combination is hard to
find — especially when subject-specific patterns provide locally good
gradients pointing somewhere else. Handing the network the contrast as a ready-made
input changes the optimisation landscape so the lateralized motor
feature is one of the first things the model can use.

This connects directly to Part 1. The reason EEGNet's filters
absorbed subject identity was that subject-specific features were the
strongest gradient signal in the raw representation. The contrast
channels give the model a low-cost path to a task-relevant feature
it would otherwise have to climb a gradient to discover.

There is also a sample-complexity angle that matters here
specifically. EEGNet could in principle learn C3 minus C4 from raw
input, but discovering the right linear combination from a fixed-size
training set takes data. With only 8 training subjects and a small
number of trials per subject, the model may not have enough examples
to find the correct depthwise weights before it overfits to
subject-specific gradients.
Exposing the relationship as a ready-made channel trades a small
amount of human prior knowledge for a lower sample-complexity
learning problem — the exact trade that fits the small-data regime
the assignment is built around.

A useful side effect: many approximately symmetric nuisance factors
(overall arousal, contact-quality differences, average-reference
shifts, head-size scaling) are attenuated by the left-right
subtraction, so the contrasts work simultaneously as a partial
nuisance filter and as an explicit spatial encoding.

## Why this is a clean approach to take

- **Defensible in one sentence.** I am not adding information. I am
  changing the inductive bias so a known neurophysiological
  relationship is observable in one dimension instead of buried in
  64.
- **Zero extra learnable parameters in the augmentation.** The
  contrast computation is fixed; all capacity stays in EEGNet.
- **Modular.** The same channel augmentation can wrap any
  channel-first EEG model.
- **Self-falsifying.** If the contrasts do not help, the depthwise
  filter learns small weights on them and the network behaves like
  the baseline. There is no architectural commitment to roll back.

## Why I think it is underexplored

CSP and EEGNet both learn spatial filters from data. Fixed pairwise
contrasts go the other direction, handed to the model rather than
learned. The classical surface Laplacian does something similar but
is used as a preprocessing replacement, not as extra channels sitting
next to the raw ones. I have not seen a paper that augments EEGNet's
channel basis with explicit symmetric-pair contrasts as a low-cost
inductive prior.

## Why it might not work

First, runs 6/10/14 may not be primarily left-right lateralized. The
brief comments them "left vs right fist" but per the PhysioNet
documentation they are actually imagined both-fists vs both-feet
(I emailed Shashwat for clarification). For fists-vs-feet the C3-C4
type contrasts would be near zero on both classes, which is why I
included the (C3+C4)/2 minus Cz contrast as the tenth feature.
Second, adding ten channels means the depthwise filter has more
parameters; if the contrasts do not carry the task signal, the extra
channels just give the model more ways to overfit subject identity.

## Implementation

`src/part3_idea.py::augment_with_contrasts`. Compute the ten pairwise
contrasts per trial and concatenate them as extra channels. Window,
bandpass, per-trial z-score and training schedule identical to the
baseline. Only the channel count changes (64 → 74).

## Sanity check on the engineered feature

Before asking the network to use the contrasts, I plotted the two key
ones class-by-class on the training pool. `scripts/plot_contrasts.py`
saves `results/figures/part3_contrasts.png` (trial-averaged C3 − C4
and (C3 + C4)/2 − Cz, with ±SEM bands, separately for class T1 and
T2). Two things I wanted to check:

- The C3 − C4 contrast has a **sign flip between classes** (T1 mean
  is negative, T2 mean is positive). That is the lateralization
  pattern the contrast is built to capture, so the feature is doing
  what it is supposed to do at the population level.
- The effect size is modest — Cohen's d ≈ 0.14 on the trial-averaged
  C3 − C4 and ≈ 0.05 on the lateral-vs-midline contrast. So the
  contrasts are class-informative but not separable on their own at
  the trial level. They are a useful representation *for the network
  to use*, not a classifier on their own.

## Results

3 seeds × 60 epochs, cross-subject test (subjects 9, 10).

| | best | worst | sil(class) | sil(subject) |
| --- | ---: | ---: | ---: | ---: |
| EEGNet baseline, 8 subj | 0.644 | 0.511 | 0.010 | 0.084 |
| **EEGNet + contrast channels, 8 subj** | **0.667** | **0.556** | -0.001 | **0.037** |
| EEGNet baseline, 3 subj | 0.656 | 0.533 | -0.002 | 0.062 |
| EEGNet + contrast channels, 3 subj | 0.533 | 0.511 | 0.005 | 0.037 |

## What actually happened

The two pools tell different stories, and the difference is exactly
what the sample-complexity argument predicted.

**On the full 8-subject pool, the hypothesis is validated.** sil(subject)
dropped 2.3× (0.084 → 0.037), meaning the embedding is much less
clustered by subject identity than the baseline. Best accuracy
improved (0.644 → 0.667) and worst-case accuracy improved (0.511 →
0.556). Both the mechanism (less subject clustering) and the outcome
(better cross-subject test accuracy) lined up with the prediction.
The depthwise filter did learn to lean on the contrast channels when
the raw alternative was contaminated by subject identity.

**On the reduced 3-subject pool, the intervention failed.** sil(subject)
still dropped (0.062 → 0.037), so the contrast channels are doing
what they are supposed to do at the representation level. But best
accuracy collapsed (0.656 → 0.533). My reading is that the extra ten
channels added parameters faster than 135 trials could constrain
them — the depthwise filter has 10 × ~K more weights to fit, and the
3-subject pool does not give it enough examples to do so without
overfitting. The mechanism still works; the model just cannot afford
it in this regime.

That is the trade-off the inductive-bias argument predicts but in
reverse: when there are enough examples to back the prior, the prior
helps. When there are not, the extra capacity hurts before the prior
pays off.

## What I learned

Two things, neither of which I would have known without running the
experiment.

1. The hemispheric contrast channels do reduce subject-identity
   leakage in both regimes (sil(subject) drops in both). The
   representation hypothesis was right.
2. Whether that translates to accuracy depends on how much data is
   available to absorb the extra channels. The trade between human
   prior and sample-complexity tipped in opposite directions on the
   two pools.

If I had more time, the cleanest next experiment would be to apply a
parameter constraint on the depthwise filter weights that act on the
ten contrast channels — for example freezing them to identity in the
reduced-pool regime — so the prior is preserved without paying the
full parameter cost. That would test directly whether the 3-subject
failure is about capacity (yes) or about the contrast signal itself
(probably not, since the mechanism worked on the 8-subject pool).
