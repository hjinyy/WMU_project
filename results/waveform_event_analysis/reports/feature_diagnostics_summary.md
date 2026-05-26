# Feature Diagnostics Summary

- Flat best: RandomForest | macro-F1=0.7381 | Normal_recall=0.0000
- Hierarchical best: rule_based + RandomForest | macro-F1=0.5417 | Normal_recall=1.0000
- Feature-hash 기준으로 Normal 3개와 SLG_Fault 30개가 동일 그룹으로 묶였다.
- 이는 Normal→SLG confusion의 핵심 원인이 raw/event integrity issue임을 시사한다.

## Limitations
- Normal sample 3개뿐임
- trigger threshold는 preliminary임
- verified SLG raw 재생성이 필요함
