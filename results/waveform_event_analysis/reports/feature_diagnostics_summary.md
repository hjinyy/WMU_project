# Feature Diagnostics Summary

## 1. 분석 목적
현재 83개 WMU waveform dataset에서 추출한 feature들이 **Normal / LoadSwitch / SLG_Fault / ThreePhase_Fault**를 어떻게 구분하는지 시각적으로 해석하는 데 초점을 두었다. 본 결과는 최종 classification 성능 주장보다 **feature separability, dataset limitation, 오분류 원인 분석**을 우선한다.

## 2. 사용 데이터셋 83개 구성
- Total: 83
- Normal: 3
- LoadSwitch: 20
- SLG_Fault: 30
- ThreePhase_Fault: 30
- Quality summary from prior run: OK 83 / WARNING 0 / FAILED 0

## 3. Feature 분포 분석 요약
주요 case-level/core feature에 대해 EventType별 boxplot/violin plot과 요약 통계를 생성했다.

가장 큰 class-level separation을 보인 핵심 feature(ANOVA F-score 기준):
- max_sag
- max_I0_ratio
- max_dV_energy
- max_Res_ratio
- max_I_unbalance
- max_Z_drop_ratio

분포 평균 기준 관찰:
- max_sag: highest mean response in ThreePhase_Fault (0.9336).
- max_I0_ratio: highest mean response in ThreePhase_Fault (0.8581).
- max_dV_energy: highest mean response in ThreePhase_Fault (0.2808).
- max_Res_ratio: highest mean response in ThreePhase_Fault (0.3718).

## 4. LoadSwitch와 Fault 구분에 유효해 보이는 feature
LoadSwitch vs Fault(SLG + ThreePhase) 평균 차이를 표준화한 effect size 기준 상위 feature:
- max_I0_ratio
- max_I_unbalance
- max_dV_energy
- max_dI_energy
- max_sag
- max_liss_area_norm_min

해석적으로는 dV/dI disturbance energy, sag, sequence/unbalance, impedance-drop 계열이 LoadSwitch와 Fault를 가장 직접적으로 가르는 축으로 보인다.

## 5. Normal 오분류 원인 분석
- Only 3 Normal cases exist, so Normal false-alarm interpretation is unstable.
- Normal and SLG_Fault are numerically overlapped on several core case-level features: max_I0_ratio, max_I_unbalance, max_V0_ratio, max_V_unbalance, max_Z_drop_ratio, max_dI_energy.
- The misclassified Normal cases lie closer to the SLG_Fault medians than to LoadSwitch or ThreePhase medians for: max_I0_ratio, max_I_unbalance, max_V0_ratio, max_V_unbalance, max_Z_drop_ratio, max_dI_energy.
- max_I0_ratio: Normal value=1.26e-05, SLG median=1.26e-05, LoadSwitch median=2.638e-05, ThreePhase median=0.8736.
- max_I_unbalance: Normal value=2.475e-05, SLG median=2.475e-05, LoadSwitch median=0.0007133, ThreePhase median=1.142.
- max_V0_ratio: Normal value=0.4618, SLG median=0.4618, LoadSwitch median=0.1889, ThreePhase median=0.4713.
- 현재 데이터에서는 Normal이 3개뿐이므로, Normal false alarm의 일반화 해석은 매우 불안정하다.
- 그럼에도 misclassified Normal의 특정 feature 값이 SLG_Fault 분포 내부 또는 경계에 들어가는지 feature 관점에서 직접 확인할 수 있다.

## 6. Correlation / importance 결과 요약
상관이 특히 큰 feature 쌍:
- max_I0_ratio vs max_I_unbalance: corr=0.997
- max_I_unbalance vs max_Z_drop_ratio: corr=0.991
- max_sag vs max_I0_ratio: corr=0.988
- max_I0_ratio vs max_Z_drop_ratio: corr=0.987
- max_I0_ratio vs max_liss_area_norm_min: corr=-0.981

RandomForest 상위 중요 feature:
- Bus07__dI_energy_C: 0.0062
- Bus16__dI_energy_B: 0.0050
- Bus13__I2_ratio: 0.0050
- Bus05__dI_Res_ratio_C: 0.0050
- Bus09__dI_energy_C: 0.0050
- Bus30__dI_energy_C: 0.0050
- Bus26__Z_drop_ratio: 0.0050
- Bus01__Delta_Z_app: 0.0050
- Bus27__dV_HF_ratio_3ph_max: 0.0050
- Bus06__dV_HF_ratio_C: 0.0050

Top-30 중요 feature의 ObservedBus 분포:
- 5: top-feature count=4, summed importance=0.0175
- 13: top-feature count=4, summed importance=0.0163
- 6: top-feature count=3, summed importance=0.0125
- 1: top-feature count=2, summed importance=0.0087
- 11: top-feature count=2, summed importance=0.0087
- 25: top-feature count=2, summed importance=0.0075

## 7. Spatial heatmap 해석
TargetBus × ObservedBus mean-response heatmap으로 disturbance spatial footprint를 요약했다.
- heatmap_slg_dv_energy.png: strongest average response at TargetBus 1 / ObservedBus 14.
- heatmap_threephase_dv_energy.png: strongest average response at TargetBus 8 / ObservedBus 6.
- heatmap_loadswitch_dv_energy.png: strongest average response at TargetBus 8 / ObservedBus 9.
- heatmap_slg_i0_ratio.png: strongest average response at TargetBus 1 / ObservedBus 27.
- heatmap_threephase_di_energy.png: strongest average response at TargetBus 5 / ObservedBus 5.
- heatmap_loadswitch_di_energy.png: strongest average response at TargetBus 8 / ObservedBus 8.
- LoadSwitch heatmap의 TargetBus index가 20개만 나오는 것은 load bus만 생성된 현재 dataset 구성상 정상이다.

## 8. 현재 데이터에서 DV_energy가 지배적인 이유에 대한 해석
- 현재 dataset은 event timing이 고정되어 있고 disturbance type별 파형 변화 크기가 크기 때문에, one-cycle difference 기반 dV_energy가 가장 직접적인 trigger strength를 제공한다.
- Fault 계열은 전압 파형 붕괴/급변이 크고, LoadSwitch는 상대적으로 전류 변화 비중이 커서 dV_energy 및 dI_energy가 먼저 class boundary를 만든다.
- 데이터 수가 작고 조건 다양성이 제한되어 있으므로, 강한 energy feature 하나가 다른 미세 feature보다 훨씬 안정적으로 작동한다.

## 9. 추가 feature가 아직 성능 향상으로 이어지지 않은 이유에 대한 가설
- 현재 83개 데이터는 이미 separability가 높아 richer feature가 macro-F1을 더 올릴 여지가 작다.
- Normal class가 3개뿐이라 sequence/lissajous/impedance feature의 장점이 안정적으로 검증되지 않는다.
- Fault resistance, inception angle, clearing time, switching intensity variation이 부족해 고급 feature가 드러날 조건 변화가 충분하지 않다.
- 일부 feature는 dV/dI energy와 강하게 상관되어 정보 중복이 발생한다.

## 10. 다음 실험 제안
- LoadSwitch 강도를 5%, 15%, 25% 이상으로 늘려 switching manifold를 확장한다.
- Fault resistance / clearing time / inception angle variation을 추가해 fault morphology 다양성을 늘린다.
- Normal baseline을 충분히 늘려 false alarm 해석의 신뢰구간을 확보한다.
- 반복 시뮬레이션을 추가해 case-level train/test split이 가능한 dataset으로 확장한다.
- WMU bus importance를 target-bus distance 또는 network topology와 함께 해석해 spatial diagnostics를 강화한다.
