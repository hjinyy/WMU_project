# Why Bus 5 is the selected minimum WMU placement

## Purpose

This analysis explains why Bus 5 was selected in the fault/non-fault
hard-constraint minimum-WMU study. It does not change the objective to fault
localization.

## Existing result

- Minimum count: k=1
- Selected placement: Bus 5
- Margin: 0.170377
- Normal FP / LoadSwitch FP / SLG FN / ThreePhase FN: 0 / 0 / 0 / 0

## Feasible single-WMU comparison

Six buses are feasible. Margin ranking: Bus 5 (0.170), Bus 6 (0.124), Bus 27 (0.087), Bus 25 (0.028), Bus 24 (0.025), Bus 22 (0.002). Bus 5 has the largest gap
because its non-fault maximum is only 0.024301, while its
fault minimum remains 0.194678. Other feasible buses also
achieve zero errors, but their score boundaries are much closer.

## Why infeasible buses fail

- Normal false alarm only: 16 buses
- LoadSwitch false alarm only: 0 buses
- SLG missed fault only: 2 buses
- Three-phase missed fault only: 2 buses
- Combined failure: 4 buses

The dominant weakness is therefore elevated Normal score rather than
LoadSwitch rejection. Bus 12 is the only single-WMU candidate with a
LoadSwitch false alarm under the selected rule.

## Feature-level explanation

At Bus 5, balanced 67–77 Hz voltage evidence is high for LoadSwitch and enters
the fixed score as a rejection penalty. Fault cases retain strong
voltage/current severity and sequence/unbalance evidence, particularly
zero-sequence current for SLG faults. The resulting non-fault maximum
(0.024301) and fault minimum
(0.194678) leave a 0.170377 safety gap
around threshold 0.109490.

The processed table contains the direct 67–77 Hz (`E72`) band for every bus.
Requested 60–80 and 70–80 Hz terms are documented as nearest-band proxies;
raw waveforms were not recomputed in this task.

## Topology explanation

Bus 5 has degree 2, betweenness 0.002463,
and closeness 0.271028.
It is not a topological hub. The Spearman correlation between closeness and
margin is 0.042. Centrality alone therefore
does not justify Bus 5; waveform separation is the direct selection basis.
Bus 5 is the largest configured LoadSwitch target in the current metadata
(added active-power setting 0.14130), which may strengthen its local switching
signature, but this remains a dataset-specific explanation.

## Bus 5 versus Bus 1

Bus 1 was sufficient for the earlier four-class macro-F1 greedy objective, but
that objective did not maximize the strict fault/non-fault safety margin.
Under the fixed hard-constraint score, Bus 1 has margin
-0.101784 and 0 LoadSwitch false alarms but
fails because Normal cases cross the fault decision region. Bus 5 and Bus 1
have identical unweighted topology centrality, confirming that their different
outcomes arise from waveform-feature behavior rather than graph position.

## Conclusion

Bus 5 is the most robust single-WMU placement in the current dataset because
it simultaneously provides strong LoadSwitch rejection and a retained lower
bound on fault evidence. This is a deterministic-dataset conclusion and must
be re-tested under varied LoadSwitch size, SSO conditions, fault parameters,
noise, missing channels, and operating points.
