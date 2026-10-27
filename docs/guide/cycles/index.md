# Find cycles

A cycle is a repeating swing of a series around its trend. These pages
measure the length of the dominant cycle with John Ehlers' Hilbert transform
tools and mark the turns of that cycle, on BTC daily closes.

<!-- tasks:start -->

1. [Measure the dominant cycle length](dominant-cycle.md)
2. [Mark cycle turning points](turning-points.md)

```{toctree}
:hidden:

/guide/cycles/dominant-cycle
/guide/cycles/turning-points
/notebooks/19-cycle-and-spectral-analysis
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Isolate a band of cycle lengths | `RoofingFilter` |
| Measure the dominant cycle length | `DominantCycle` |
| Mark cycle turning points | `CycleSine` |
| Phase, frequency, or amplitude of the cycle | `CyclePhase`, `CycleFrequency`, `CycleAmplitude` |
| Tell trending from cycling periods | `TrendMode` |
