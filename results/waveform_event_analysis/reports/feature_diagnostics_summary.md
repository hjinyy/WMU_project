# Feature Diagnostics Summary

## 분석 목적
현재 결과는 최종 classification 성능 주장보다 **feature separability / Normal false alarm / raw data integrity / dataset limitation** 분석에 초점을 둔다.

## 핵심 진단
- Flat best: RandomForest | macro-F1=0.5000 | Normal_recall=1.0000
- Hierarchical best: rule_based + RandomForest | macro-F1=0.5000 | Normal_recall=1.0000
- Normal raw sanity check에서 exact duplicate non-Normal matches 총 6건이 확인되었다.
- Normal과 다수 SLG raw/feature가 동일하면 이는 classifier 문제가 아니라 **raw dataset integrity issue** 신호다.

## class separability 상위 feature
- max_sag
- max_I0_ratio
- max_I_unbalance
- max_Z_drop_ratio
- max_HF_ratio
- max_liss_area_norm_min

## correlation summary
- max_I0_ratio vs max_I_unbalance: corr=0.994
- max_sag vs max_Z_drop_ratio: corr=0.955
- max_Z_drop_ratio vs max_liss_area_norm_min: corr=-0.936
- max_sag vs max_liss_area_norm_min: corr=-0.930
- max_dV_energy vs max_liss_area_norm_min: corr=-0.816

## RandomForest top features
- Bus11__I2_ratio: 0.0056
- Bus26__Delta_Z_app: 0.0054
- Bus12__dV_HF_ratio_A: 0.0052
- Bus05__I_rms_jump_C: 0.0044
- Bus05__V0_ratio: 0.0044
- Bus08__I_rms_jump_B: 0.0043
- Bus20__dV_HF_ratio_A: 0.0042
- Bus06__dV_HF_ratio_C: 0.0041
- Bus05__Z_app_post: 0.0041
- Bus26__Z_drop_ratio: 0.0040

## important buses
- 5: count=4, summed importance=0.0164
- 26: count=3, summed importance=0.0126
- 20: count=2, summed importance=0.0079
- 8: count=2, summed importance=0.0079
- 25: count=2, summed importance=0.0070
- 13: count=2, summed importance=0.0068
- 18: count=2, summed importance=0.0061
- 11: count=1, summed importance=0.0056

## spatial heatmap summary
- heatmap_slg_dv_energy.png: strongest average response at TargetBus 8 / ObservedBus 6.
- heatmap_threephase_dv_energy.png: strongest average response at TargetBus 28 / ObservedBus 6.
- heatmap_loadswitch_dv_energy.png: strongest average response at TargetBus 8 / ObservedBus 9.
- heatmap_slg_i0_ratio.png: strongest average response at TargetBus 5 / ObservedBus 5.
- heatmap_threephase_di_energy.png: strongest average response at TargetBus 5 / ObservedBus 5.
- heatmap_loadswitch_di_energy.png: strongest average response at TargetBus 8 / ObservedBus 8.

## 한계
- Normal sample은 3개뿐이다.
- Hierarchical trigger threshold는 preliminary이다.
- 현재 raw folder 안의 일부 SLG file은 no-event/Normal과 동일할 가능성이 높다.
- 따라서 현재 SLG 관련 metric은 raw data integrity audit 없이는 최종 주장에 쓰면 안 된다.

## Representation study update

The subtype-debug follow-up confirms that the swap problem is representation-driven rather than a confusion-matrix labeling bug. New event-type summary features and grouped-by-targetbus evaluation artifacts were added to separate subtype semantics from location fingerprinting.

## Fault subtype representation study

The follow-up representation study indicates that, after fully regenerating both SLG and ThreePhase raw datasets, the current feature tables no longer show the old subtype swap. In the regenerated dataset:
- fault-pair distance diagnostics no longer indicate a same-bus opposite-type nearest-neighbor bias,
- grouped-by-targetbus evaluation remains stable,
- event-type summary features are sufficient for perfect subtype separation on the current 83-case dataset.
