# Final zone/community definition

The four localization zones were not defined by bus-number ranges and were not tuned to improve localization accuracy. They were generated from `data/ieee30_edges.csv` using NetworkX greedy modularity community detection on the unweighted IEEE 30-bus topology graph.

## Membership

- **Zone1:** 1, 2, 3, 4, 5, 6, 7, 8, 28
- **Zone2:** 9, 10, 11, 16, 17, 21, 22
- **Zone3:** 12, 13, 14, 15, 18, 19, 20
- **Zone4:** 23, 24, 25, 26, 27, 29, 30

## Limitation

This is a topology-community partition, not an electrical-distance clustering. Line impedance, operating point, voltage sensitivity, and IBR dynamics are not used. It is acceptable as a transparent supplementary diagnostic, but it should not be interpreted as an optimized protection-zone definition. An electrical-distance study would be more physically grounded, but must be fixed independently of localization labels and evaluated without tuning zones to improve the result.
