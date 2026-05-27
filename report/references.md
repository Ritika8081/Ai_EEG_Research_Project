# References

Cited in `report/part1_analysis.md`, `report/part2_design.md`,
`report/part3_idea.md`, and the defence slides.

## Core architecture

- **Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
  Hung, C. P., & Lance, B. J. (2018).** *EEGNet: A compact
  convolutional neural network for EEG-based brain–computer
  interfaces.* Journal of Neural Engineering, 15(5), 056013.
  arXiv:1611.08024.

## Subject-transfer methods I looked at but didn't transplant

- **Zanini, P., Congedo, M., Jutten, C., Said, S., & Berthoumieu, Y.
  (2018).** *Transfer learning: A Riemannian geometry framework with
  applications to brain–computer interfaces.* IEEE Transactions on
  Biomedical Engineering, 65(5), 1107–1116.
  → Euclidean Alignment. Closest related method to my Part 2
  whitening; I use a per-trial reference rather than a subject-mean
  one.

- **He, H., & Wu, D. (2020).** *Transfer learning for brain–computer
  interfaces: A Euclidean space data alignment approach.* IEEE
  Transactions on Biomedical Engineering, 67(2), 399–410.

- **Ganin, Y., & Lempitsky, V. (2015).** *Unsupervised domain adaptation
  by backpropagation.* ICML. → Gradient-reversal subject discriminator.
  Not used because it removes subject signal *after* the spatial conv
  has already learned subject-coupled weights.

## Coordinate-based / implicit neural representations

- **Liu, R., Lehman, J., Molino, P., Such, F. P., Frank, E., Sergeev, A.,
  & Yosinski, J. (2018).** *An intriguing failing of convolutional
  neural networks and the CoordConv solution.* NeurIPS.
  → Inspiration for parameterising filters over spatial coordinates;
  I adapt the idea to the *filter weights* rather than the input
  feature map.

- **Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., &
  Wetzstein, G. (2020).** *Implicit neural representations with
  periodic activation functions.* NeurIPS. → General framework for
  representing fields by MLPs of coordinates.

## Cortical / scalp interpolation

- **Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989).**
  *Spherical splines for scalp potential and current density mapping.*
  Electroencephalography and Clinical Neurophysiology, 72(2), 184–187.
  → Standard tool for EEG channel interpolation. An earlier version
  of Part 3 (the CMA augmentation, now legacy) used a Gaussian RBF
  on the sphere — the leading-order term of the spherical-spline
  expansion. Same single-sphere head-model assumption.

## Classical BCI baseline (for Part 1 calibration)

- **Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000).**
  *Optimal spatial filtering of single trial EEG during imagined hand
  movement.* IEEE Transactions on Rehabilitation Engineering, 8(4),
  441–446. → Common Spatial Patterns (CSP).
- **Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R.
  (2008).** *Optimizing spatial filters for robust EEG single-trial
  analysis.* IEEE Signal Processing Magazine, 25(1), 41–56.

## Augmentation context

- **Lashgari, E., Liang, D., & Maoz, U. (2020).** *Data augmentation for
  deep-learning-based electroencephalography.* Journal of Neuroscience
  Methods, 346, 108885. → Surveys existing EEG augmentations
  (channel dropout, mixup, noise injection). I noticed
  position-space augmentation isn't in the survey, which is part of
  why I think the Part 3 angle isn't well-covered.

- **Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D.,
  & Le, Q. V. (2019).** *SpecAugment: A simple data augmentation method
  for automatic speech recognition.* Interspeech.
  → SpecAugment is the closest analog in spirit — masking on a known
  axis (time/frequency in speech, position in mine).

## Diagnostic metric

- **Rousseeuw, P. J. (1987).** *Silhouettes: A graphical aid to the
  interpretation and validation of cluster analysis.* Journal of
  Computational and Applied Mathematics, 20, 53–65.
  → Silhouette score; I use it to compare "by class" vs "by subject"
  organisation of the penultimate-layer embeddings — the headline
  Part 1 diagnostic.

## Cap-placement variance (motivation for Part 3)

- **Picton, T. W., Bentin, S., Berg, P., Donchin, E., Hillyard, S. A.,
  Johnson, R., et al. (2000).** *Guidelines for using human event-related
  potentials to study cognition: Recording standards and publication
  criteria.* Psychophysiology, 37(2), 127–152.
- **Klem, G. H., Lüders, H. O., Jasper, H. H., & Elger, C. (1999).**
  *The ten–twenty electrode system of the International Federation.*
  Electroencephalography and Clinical Neurophysiology Supplement, 52,
  3–6. → Cap-placement tolerance ~5–10 mm under standardised
  electrode-localisation protocol.
