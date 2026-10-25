# Pre-registration: shortcut-aware training (revision R1c)

Written before any run on the real data (28 Sep 2026). This is the last
detection experiment of the revision; every result below is reported.

**Question.** Does a detector trained so that the environment carries no
reward find an eye/head cybersickness signal that transfers to unseen people?

**Data and protocol.** Benchmark v1.1, seated sources (47 participants),
label FMS > 2, repeats 0-2 of the stratified 5-fold participant CV already
used for the published-recipe ensemble (identical folds and windows), all 14
channels, raw.

**Methods** (`analysis/revision_debias.py`, 3-member ensembles, seeds 42, 123, 456,
selection and early stopping by validation within-environment AUC):
`select` (control), `balanced`, `adversarial`, `ranking`.

**Primary outcome.** Within-environment-and-minute AUC, participant-cluster
bootstrap 95% CI (2,000 replicates).

**Decision rule.** A method supports a restricted detection claim only if
(1) its primary-outcome CI lies above 0.5 and (2) its increment over
environment + elapsed time has a CI above 0. Four methods are tested; a
claim additionally requires (1) and (2) to hold at 98.75% (Bonferroni,
4 methods), reported alongside.

**Sensitivity check (done on synthetic data before this run).** With a
planted within-environment signal and a strong location shortcut, the four
methods reached within-environment AUCs of 0.60-0.70 versus 0.50 for an
environment-only score.
