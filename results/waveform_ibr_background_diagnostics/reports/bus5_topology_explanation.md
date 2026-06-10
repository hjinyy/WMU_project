# Bus 5 topology explanation

Topology was computed only from `data/ieee30_edges.csv` as an unweighted graph.

## Bus 5 metrics

- Degree: 2
- Degree centrality: 0.068966
- Betweenness centrality: 0.002463
- Closeness centrality: 0.271028
- Average shortest-path distance: 3.690
- Maximum shortest-path distance: 6
- Configured LoadSwitch/load bus: True
- Configured added active power at Bus 5: 0.14130

Bus 5 is a degree-2 peripheral branch connected through Bus 2; it is not a topological hub. Bus 1 has identical degree, betweenness, closeness, and path-distance metrics in this unweighted topology, yet Bus 1 is infeasible while Bus 5 has the largest positive margin.
Bus 5 is also the largest configured LoadSwitch target in the current metadata (added active-power setting 0.14130). This may help expose a stable local switching signature, but it is a dataset-specific operating condition rather than a general topology property.

## Topology-margin relationship

- Spearman correlation for DegreeCentrality: 0.198
- Spearman correlation for BetweennessCentrality: 0.172
- Spearman correlation for ClosenessCentrality: 0.042
- Spearman correlation for AverageShortestPathDistance: -0.042

Therefore topology centrality alone does not explain the selection. The direct evidence is waveform-feature separation: Bus 5 suppresses all Normal/LoadSwitch scores while retaining a clear lower bound for every fault.

Generator/PV and explicit IBR-candidate bus metadata were not available in the repository data used here, so no unsupported assignment was inferred.
