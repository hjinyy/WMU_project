# Final Waveform Event Analysis Summary

- 기존 bad `SLG_Fault_Bus*.xlsx` 30개를 archive한 뒤, MATLAB R2025b direct exact-parameter fault automation으로 `SLG_Fault_Bus01.csv` ~ `SLG_Fault_Bus30.csv`를 재생성했다.
- direct smoke v2 (`Bus02`, `Bus10`, `Bus30`)는 모두 통과했다.
- raw dataset은 다시 `Normal 3 / LoadSwitch 20 / ThreePhase 30 / SLG 30 = total 83`으로 복구되었다.
- 재생성 후 `Normal`과 `SLG_Fault`는 더 이상 동일 feature-hash 그룹이 아니다.
- 즉 이전의 `Normal == SLG` raw-data corruption 문제는 해결되었다.

## Current classification outcome

### Flat best
- Model: `RandomForest`
- Accuracy: `0.2771`
- Macro-F1: `0.5000`
- Balanced accuracy: `0.5000`
- Normal recall: `1.0000`
- LoadSwitch recall: `1.0000`
- SLG recall: `0.0000`
- ThreePhase recall: `0.0000`
- Fault precision: `1.0000`

### Hierarchical best
- Trigger: `rule_based`
- Event model: `RandomForest`
- Accuracy: `0.2771`
- Macro-F1: `0.5000`
- Balanced accuracy: `0.5000`
- Normal recall: `1.0000`
- LoadSwitch recall: `1.0000`
- SLG recall: `0.0000`
- ThreePhase recall: `0.0000`
- Fault precision: `1.0000`

## Interpretation

- The previous failure mode (`Normal -> SLG_Fault`) disappeared after SLG raw regeneration.
- However, the current regenerated SLG cases are now classified almost entirely as `ThreePhase_Fault`, while `ThreePhase_Fault` is classified as `SLG_Fault`.
- So the dataset is no longer corrupted in the old way, but **SLG vs ThreePhase separability is still not solved**.
- This suggests the present case-level feature set or fault-generation diversity is still insufficient for reliable subtype discrimination between single-line-to-ground and three-phase faults.

## Sensor-count snapshot

- Greedy sensor-count curve remains flat at macro-F1 `0.5000` and balanced accuracy `0.5000`.
- Normal recall remains `1.0000`.
- LoadSwitch recall remains `1.0000`.
- Fault precision remains `1.0000`.

## Fault localization snapshot

- Both self-matching and strict LOO show `1.0000` exact/top3/1-hop accuracy on the current regenerated dataset.
- This should still be treated only as a **preliminary separability check**, because there is only one fault case per bus per class.

## Current limitations

- `Normal` class still has only 3 samples.
- The direct SLG automation path is now good enough to generate clearly non-Normal fault waveforms, but subtype diversity is still limited.
- `SLG_Fault` and `ThreePhase_Fault` remain systematically swapped by the current classifier, so the present feature set / dataset should not yet be treated as a final event-classification benchmark.
- The MATLAB automation scripts used for successful regeneration currently live in `wmu_work2` and are not yet fully productized inside the git repo.
