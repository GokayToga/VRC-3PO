# Manuscript-ready corrections

## Headline result

The five-model CNN ensemble achieved a pooled AUC of 0.757
on 768 windows from
9 unseen participants in the two passive
paradigms. Because windows overlapped and were clustered within participants,
uncertainty was estimated by resampling participants rather than individual
windows. The participant-cluster bootstrap 95% interval was
[0.696, 0.814].

Among the 5 test participants who
contributed both elevated and non-elevated windows, mean participant-level AUC
was 0.753 (participant-bootstrap 95% interval
[0.613, 0.899]). The remaining
participants contributed only one outcome class and therefore had no defined
within-participant AUC.

## Required caption correction

Replace "17 unseen participants" in the passive-only result and Figure 2
caption with "9 unseen participants".
Seventeen is the size of the complete all-paradigm test split, not the passive
subset evaluated by the CNN ensemble.

## Interpretation constraint

The pooled result supports cross-participant discrimination in this test split.
The participant-level analysis should be reported alongside it because a pooled
window AUC alone can mix within-participant change with between-participant
differences.
