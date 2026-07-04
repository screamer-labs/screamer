# Task F Report: DAG boundary – Stream feeds, (values, index) returns, key→index sweep

## Files changed

- `screamer/dag.py` — main implementation changes (details below)
- `tests/test_dag_identity.py` — feed helpers flipped to values-first; oracle comparison renamed; two new test cases added
- `tests/test_dag_dropna.py` — feeds and result unpacking updated to values-first
- `tests/test_dag_select.py` — feeds and result unpacking updated to values-first
- `tests/test_dag_resample.py` — feeds and result unpacking updated to values-first

`tests/_dag_oracle.py` was NOT changed — it stores results internally as `(index, values)` tuples (engine format) and calls `_align_results` at the end, which now returns `(values, index)`. Compatible without modifications.

## New Dag input/output shapes

**Input (feeds):** `_as_stream(feed)` now accepts three forms and converts them to the `(index_array, values_array)` format the C++ engine needs:
- bare value array → positional index via `np.arange(n, dtype=int64)`
- `Stream` object → uses `.values` / `.index` (None → `np.arange`)
- `(values, index)` pair (values-first user convention) → flipped to `(index, values)` for the engine

**Output (`__call__` / `stream`):**
- M == 1 → single `(values, index)` pair (values-first)
- M > 1, `align_outputs=False` → tuple of `(values, index)` pairs
- M > 1, `align_outputs=True` → tuple of equal-length `(values, index)` pairs co-indexed via `combine_latest`

## `np.unique` dedup removal

Removed the `np.unique` keep-last dedup block from `_align_results`. The old code:

```python
_, inv_idx = np.unique(aligned_keys[::-1], return_index=True)
last_idx = np.sort(len(aligned_keys) - 1 - inv_idx)
aligned_keys = aligned_keys[last_idx]
aligned = aligned[last_idx]
```

was dead work: `combine_latest` already coalesces to one row per distinct index (Task B). After removal, the aligned path simply uses `out.index` and `out.values` directly from `combine_latest`. Alignment correctness confirmed: all `_divergent` (align_outputs=True, 2 outputs) tests pass, batch==stream holds.

## `Dag.stream` merge fix

The old code called `merge(*streams)` where `streams` were `(index_arr, values_arr)` tuples, passing the tuple objects as the value arrays — causing the C++ merge to treat each tuple as a 2-element "stream" and returning `ms=None` (positional mode). The fix splits streams into separate lists before calling merge:

```python
idx_arrays = [s[0] for s in streams]
val_arrays = [s[1] for s in streams]
merged_vals, merged_sources, merged_index = merge(*val_arrays, index=idx_arrays)
for v, src, k in zip(merged_vals, merged_sources, merged_index):
    self._cg.push_event(int(src), int(k), float(v))
```

This restores byte-identical batch==stream results for all graph shapes.

## Residual `key` renames

All user-facing `key`/`keys` in `screamer/dag.py` were renamed to `index`:

| Location | Before | After |
|---|---|---|
| `_as_stream` docstring | `keys = row-number int64` | `index = row-number int64 via np.arange` |
| `_align_results` docstring | "one-row-per-key", "co-indexed tuple" | "one row per distinct index", updated |
| `Dag` class docstring | "shared, sorted key axis", "(keys, values) pairs" | "shared, sorted index axis", "(values, index) pairs" |
| `_compile_cpp.build` local var | `key = id(node)` | `node_id = id(node)` (memoization key; renamed for clarity) |
| Resample comment in `_compile_cpp` | `# 0=ByKey,1=ByCount` | `# 0=ByIndex,1=ByCount` |

## Verification

```
poetry run pytest tests/test_dag_identity.py tests/test_dag_dropna.py tests/test_dag_select.py tests/test_dag_resample.py -q
```

**Result: 53 passed in 1.06s**

All tests pass. batch==stream identity holds across all graph shapes (chain, fanout, combine, divergent, dropna, select, resample variants). Two new tests added:
- `test_stream_feed_matches_array_feed`: Stream-fed Dag matches `(values, index)` pair-fed Dag
- `test_values_index_feed_matches_array_feed`: `(values, index)` feed batch==stream

## Concerns

None. The `np.unique` removal is safe because `combine_latest` already produces one row per distinct index. The `Dag.stream` merge fix resolves the pre-existing `TypeError: 'NoneType' object is not iterable` failure on all streaming tests.
