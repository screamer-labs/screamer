# Changelog

All notable changes to this project are documented in this file.

Unreleased
----------

This is a major release, 3.0.0: the results listed under "Changed" differ from
2.8.0. Each entry gives the way to get the old result.

### Added

* JavaScript constructors also take one options object: `EwMean({ span: 20 })`.
  The names are the camelCase parameter names, each operator has a TypeScript
  `<Op>Options` type, and an unknown name raises `TypeError`
  ("EwMean: unknown option 'spn'; valid options are com, span, halflife,
  alpha"). Only a plain object counts, and it must be the only argument. An
  operator without parameters takes an empty object.
* JavaScript `combineLatest` joins arrays and streams directly, with `layout`.
  An indexed input is marked with an object, as a `Pipeline` feed is:
  `{values, index}` for an array and `{value, index}` for an event of an
  iterator, the counterparts of Python's tuples. A plain Array is always
  values. JavaScript `split` returns `{values, index}` objects, so its outputs
  feed `combineLatest` and a `Pipeline` as they are.
* JavaScript `split`.
* `split` takes values of any rank with time first, `(T, ...)`, in Python and
  JavaScript: each event is a row, and each output keeps the row shape
  `(rows, ...)` (an `NdArray` in JavaScript). 2.8.0 raised `TypeError` for
  values that were not 1-D.
* JavaScript `forecastPairs`.
* JavaScript `backtestReport`.
* The microstructure operators `SignedVolume`, `RollingKyleLambda`,
  `EwKyleLambda`, `RollingOrderImbalance` and `QueueImbalance` are C++
  operators and available in JavaScript.
* `Pipeline.live().push(input, index, value)` takes a sequence of numbers as one
  multi-column event, in Python and JavaScript; JavaScript's `pushWide` stays as
  an alias. The first event of an input after a reset fixes its width (a number
  is width 1), and an event of another width raises `ValueError` (`RangeError`
  in JavaScript). An empty or 2-D value, or a nested sequence, also raises
  (`ValueError` / `RangeError`). Inside a sequence, `None` (`null`) is `NaN`.
  A bare `None` is now `NaN` in Python, as it already was in JavaScript.
  A numeric Python `memoryview` is one multi-column event, as an array is.
  Python's `push` is now a C++ method and about 3x faster.
* **Walk-forward evaluation:** `walk_forward` (Python) and `walkForward`
  (JavaScript) take a `(T, K)` array of strategy variants, pick the best
  variant on each training window by any `backtest_report` statistic, trade it
  on the following test window, and return the stitched out-of-sample backtest
  with its statistics, the choice per fold, and every training score. Rolling
  and anchored training windows.
* **`help()` for every operator:** the Python classes now carry a numpydoc
  docstring built from their documentation page: signature, summary,
  parameters with types and defaults, description, and See also.
* **Real multi-asset data for the docs:** daily Binance bars of 20 coins from
  2022 to 2026 and the BTC perpetual's funding rates.
* Shape contract: every array reads as `(T, *A, F?)`, with time, assets and
  fields, and six rules hold for every operator in Python and JavaScript (see
  Polymorphic API).
* Multi-input operators broadcast over assets: `RollingCorr()(X, y)` with
  `X (T, K)` and `y (T,)`, and backtest engines with `(T, K)` targets and one
  `(T,)` price, in Python and JavaScript.
* `strategy_test` accepts `(T, K)` inputs: each randomized copy applies one time
  permutation to every asset, so the correlation between assets is kept.
  JavaScript `strategyTest` accepts the same panels.
* A packed `(T, K, N)` array feeds an N-input operator.
* Iterators of asset events give the batch result, one operator state per
  asset, in Python and JavaScript. A Python list of asset events gives a list
  of the same results.
* Every Python operator takes async iterables: `N` async sources, or one async
  source of `N`-field events, of numbers or asset arrays, giving the batch
  result. This includes the row operators (cross-sectional operators and
  portfolio engines, one `(K,)` row per input) and `PortfolioReport` (`(K, 4)`
  engine rows). Before, only one-input one-output operators of numbers did.
  Mixing async and synchronous inputs raises `TypeError`.
* Every JavaScript operator streams from sync and async iterables, including
  multi-input operators, row operators and `PortfolioReport`: one source per
  input, or one source of packed events, of numbers or asset arrays, giving
  the batch result. Operators expose `numInputs` and `numOutputs`.
* Row operators with several inputs (`BacktestPortfolioTarget`,
  `BacktestPortfolioOHLCTarget`) take iterators of rows, one per input or one
  iterator of tuples of rows.
* `SignedVolume`, `RollingKyleLambda`, `EwKyleLambda`, `QueueImbalance`, and
  `RollingOrderImbalance` accept every input form: a packed array, iterators,
  lists of events, and asset axes, as well as their keyword inputs such as
  `signed_flow=` and `return_=`.
* `reshape(x, layout)` for arrays, `(values, index)` tuples, DataFrames, lists
  and iterators of events; `reshape` in JavaScript; `from_pandas(layout=)`,
  where `layout="TAF"` reads complete two-level `(asset, field)` columns as
  `(T, A, F)`. Without a layout `from_pandas` returns what it did before.
* `CombineLatest` concatenates multi-column streams and broadcasts assets with
  `layout=`; inside a `Pipeline` it joins an OHLC node and a signal directly.
  The iterator forms read every event as the batch form reads a row, so a
  positional iterator of `(1,)` rows (a `(T, 1)` array) gives the batch rows;
  2.8.0 raised `TypeError` for it.
* `Resample` aggregates every column of `(T, X)` data; `agg="ohlc"` gives
  `(T', X, 4)`. Inside a `Pipeline` a stream carries fields only, so
  `Resample(agg="ohlc")` there gives a flat `(T', 4 * X)` frame in
  column-major field order, `[open_0, high_0, low_0, close_0, open_1, ...]`;
  `values.reshape(len(values), X, 4)` gives the eager result.
* JavaScript one-input operators accept `(T, K)` `NdArray` input.
* `PortfolioReport()(out, prices)` reports turnover as traded notional, the
  sum of `abs(position change * price)`, in the money units of equity and
  cost; without prices it is unchanged (units summed across assets). Python
  and JavaScript, for arrays, lists, and streams.
* JavaScript `Pipeline` takes an options object, `{ alignOutputs }`, Python's
  `align_outputs`: with several outputs and `alignOutputs: true` (the
  default), every output is put on one index, as in Python, and the outputs
  share one `index` array. `live().result()` returns every output of a
  multi-output pipeline, in the shape of the batch call; it threw before.
* JavaScript `Pipeline` takes `{ maxPending }`, Python's `max_pending`, with
  the same default (1,000,000) and rule. It caps every `combineLatest` node
  without its own `maxPending`, every `filter` node, and the join of a
  cumulative `resample` with its driver.

* `Resample(max_fill=...)` (JavaScript: `resample(x, { maxFill })`) bounds
  the empty bars that a fill resample (`fill="nan"` or `"carry"`) emits for
  one gap, the empty buckets that one event or one `advance(now)` closes. The
  default is 1,000,000; a setting is an integer from 1 to 2\*\*53, under the
  rule of `max_pending`. See the Changed entry below.

### Changed

* A decreasing index raises `ValueError` (JavaScript: `RangeError`) at every
  stream entry point, naming the input and the event: "the index of input 0
  decreases at event 2, from 20 to 10; a stream's index must be non-decreasing
  (equal indexes are allowed)". This covers `merge` and `Merge`,
  `combine_latest` and `CombineLatest` (arrays and iterators), `Pipeline`
  batch, lazy and live feeds (`live().push()`, `pushWide()`), `Resample`,
  `Delay`, `Dropna`, `Select`, `Filter`, `split` and `forecast_pairs`, in
  both languages. A live event below an earlier `advance(now)` raises too
  ("the index of input 0 is 40 at event 1, below an earlier advance(50)"). A
  rejected live event leaves the session as it was. Equal indexes are allowed
  as before. 2.8.0 accepted a decreasing index on all of these and returned
  rows out of order, merged or dropped without an error: `merge` and
  `combine_latest` emitted rows in a mixed order, a `Pipeline` and `Resample`
  passed the rows through in input order, `split` returned decreasing output
  indexes, and a live push below an `advance(now)` was lost from aligned
  outputs whose rows had settled with the older values. Sort each stream by
  its index first, for example `order = np.argsort(index, kind="stable")`,
  then `values[order]` and `index[order]`. Across inputs a live session still
  takes events in any order, each input in its own order.
* A fill resample (`fill="nan"` or `"carry"`, `freq=` or `every=`) raises
  `ValueError` (JavaScript: `RangeError`) when one gap would emit more than
  `max_fill` empty bars (default 1,000,000), before it builds any bar: "a fill
  Resample would emit 2000000 empty bars for one gap, more than
  max_fill=1000000". This holds in batch, lazy and live calls. The limit
  counts the empty bars of one step (one event or one `advance(now)`), so a
  live session that advances in small steps fills a gap that batch rejects.
  A rejected `advance(now)` leaves the session usable, as a rejected event
  does; later events must still come at or after `now`. 2.8.0 filled every
  empty bucket, so a large gap, such as
  two events 2\*\*53 index units apart with `every=10` or a big
  `advance(now)` jump after a pause, allocated until the process ran out of
  memory. For the 2.8.0 result on a gap up to 2\*\*53 buckets, pass a larger
  `max_fill`.

* The eager clock resample, `Resample(values, clock, clock=True, agg="ohlc")`
  on `(T, X)` values, returns `(T', X, 4)`, as `freq=` and `count=` do, in
  batch and lazy calls (each lazy bar is an `(X, 4)` array). 2.8.0 returned
  `(T', X)` holding each column's open only. For a flat `(T', 4 * X)` row per
  bar, the `Pipeline` layout, use `out.reshape(len(out), -1)`; `out[..., 0]`
  is the 2.8.0 result. A lazy resample of array events (`clock=True`,
  `freq=` or `count=`, any builtin per-column agg) yields each bar as an array
  of the batch row's shape, whatever the event's dtype or memory layout
  (lists of lists too), and every event must have the first event's shape.
  For `(X,)` events 2.8.0 yielded a tuple of floats for a clock resample:
  `tuple(v[:, 0])` (the opens) is the old value for `ohlc`, `tuple(v)` for
  the other aggs. For events of two or more axes, such as `(2, 3)`, 2.8.0
  raised `std::bad_cast`. Scalar events still give a float, or a tuple for
  `ohlc`, as any lazy operator does.
* A tuple given as a stream source always marks `(values, index)`, as an
  object `{values, index}` does in JavaScript. This holds for every stream
  source: `combine_latest` (`CombineLatest`), `merge` (`Merge`), `split`,
  `dropna` (`Dropna`), `select` (`Select`), `Filter`, `delay` (`Delay`),
  `resample` (`Resample`), `forecast_pairs`, `reshape` and `Pipeline` feeds.
  A 2-tuple of two sequences (a list, a tuple, a range, an array, a pandas
  Series or Index; the index may also be `None` or a NumPy scalar) is
  `(values, index)`, and its items follow the value and index rules. Any
  other tuple raises `TypeError` ("a tuple marks a (values, index) stream
  ..."). 2.8.0 read a tuple of numbers, such as `(1.0, 2.0, 3.0)`, as values;
  for values, pass a list or an array. 2.8.0 read a tuple whose values are
  not an array as values too: `dropna(([1.0, nan], [0, 1]))` gave
  `[[0.0, 1.0]]` (the index read as a row of values) and gives
  `(array([1.0]), array([0]))` now; `([1.0, 2.0], [0, 1])` raised
  `RuntimeError` "std::bad_cast" in `combine_latest` and `merge`. For the
  2.8.0 reading of such a tuple as rows, pass `np.array(t)`.
* An index whose length differs from its values in `combine_latest` raises
  `ValueError` ("combine_latest: input 0 has 30 rows of values but 5 index
  values"), as JavaScript's `combineLatest` raises `RangeError`. 2.8.0 raised
  `RuntimeError` ("streams: a child's index/values length differ"); catch
  `ValueError` instead.
* A JavaScript `Pipeline` feed `{values, index: null}` is positional, as
  Python's `(values, None)` is and as `{values}` already was; 2.8.0 threw
  `TypeError`. To get the error, check for a null index before the call.
* JavaScript `combineLatest` checks its options: an unknown option raises
  `TypeError`, and an `emit` other than `"when_all"` or `"on_any"` raises
  `RangeError` (`TypeError` when it is not a string). 2.8.0 ignored an
  unknown option and read any other `emit` as `"on_any"`; pass
  `emit: "on_any"` for that result.
* JavaScript operator constructors check their parameters:
  * `null` means "not given" only for the parameters that Python marks
    Optional and takes `None` for (the `com`, `span`, `halflife` and `alpha`
    of the exponentially weighted operators, the bounds of `Clip`, and so on):
    `EwMean(null, 20)` is `EwMean(undefined, 20)`, where 2.8.0 read `null` as
    0 and raised `RangeError`. For every other parameter `null` raises
    `TypeError`, as `None` does in Python ("RollingMean: windowSize must be a
    number, got null"); 2.8.0 read it as 0 for a number parameter
    (`SchmittTrigger(null, 3)` ran with a lower threshold of 0). Pass `0` for
    the 2.8.0 result, or `undefined` for the default.
  * A number parameter takes only a number. A boolean, an array, an object,
    an `NdArray` or a `BigInt` raises `TypeError`; 2.8.0 converted it (`true`
    as 1, `[5]` as 5, an object as NaN). Pass `Number(x)`. Python refuses
    `True` and `False` too (see the bool entry below).
  * A parameter without a default that is left out raises `TypeError`
    ("Hold: n is required"); 2.8.0 converted it to NaN or 0.
  * More arguments than the operator has parameters raise `TypeError`
    ("RollingMean: takes at most 2 arguments, got 3"; "Abs: takes no
    settings, got number"); 2.8.0 ignored them. Drop the extra arguments.
  * Every message a constructor raises for a parameter starts with the
    operator name ("EwMean: span must be a number, got a string"). Messages
    raised by the C++ core are unchanged.
* `split` runs its checks and its partition in C++, shared with JavaScript's
  `split`. Results for valid inputs are unchanged. Other changes:
  * A source must be a whole number. A bool, fractional, NaN, infinite,
    `None` or string source raises `TypeError` ("split: a source must be a
    whole number, got 0.5"; "... got a bool"), checked before any conversion,
    in a list as in an array. 2.8.0 read a bool as 0 or 1, truncated a
    fractional source and sent a NaN source to output 0. Pass
    `np.asarray(sources).astype(int)` for the 2.8.0 result.
  * A whole source must be from 0 to 2147483646 (one less than the most
    outputs, 2**31 - 1); another raises `ValueError` ("split: a source must
    be from 0 to 2147483646, got -1"). 2.8.0 raised `OverflowError` or
    nanobind's `TypeError` for a negative source and silently wrapped an
    int64 or uint64 source of 2**32 or more modulo 2**32 (2**40 went to
    output 0).
  * A 2-D `sources` or `index` raises `ValueError` ("split: index must be
    1-D (one index per event), got 2 dimensions"); 2.8.0 raised nanobind's
    `TypeError`. A length mismatch and a negative `n` raise `ValueError` with
    the 2.8.0 message, where 2.8.0 raised `RuntimeError`.
  * `n` is at most 2**31 - 1; a larger `n` raises `ValueError`, where 2.8.0
    raised nanobind's `TypeError`. A float `n` raises `TypeError` ("split: n
    must be an integer, got 2.0"), as for every integer setting; 2.8.0
    raised nanobind's `TypeError`, or `ValueError` when the float was too
    small. Pass `int(n)`.
* `forecast_pairs` runs one graph defined in C++, shared with JavaScript's
  `forecastPairs`, in every regime. Results for valid inputs are unchanged,
  except for a 2-D `X` in duration mode (see "Fixed"). Other changes:
  * A `(T, 1)` `X` in duration mode gives a `(rows, 1)` `X_shifted`, where
    2.8.0 gave a 1-D array; the result now has the rank of `X` in both modes.
    The values are the same. Pass `X[:, 0]` for the 1-D result.
  * `y` must be one column, `(T,)` or `(T, 1)`, in both modes and every
    regime. A 2-D `y` with more columns raises `ValueError` ("forecast_pairs:
    y holds 2 columns, expected 1"); 2.8.0 used the first column of `y` and
    dropped the others. Pass `y[:, 0]` for that result.
  * `X` must be `(T,)` or `(T, K)` with `K` at least 1. A rank-3 `X` raises
    `ValueError` ("forecast_pairs: X has 3 dimensions, expected (T,) or
    (T, K)"); 2.8.0 raised `RuntimeError` in count mode and returned the first
    feature only in duration mode. A `(T, 0)` `X` raises `ValueError`
    ("forecast_pairs: X has 0 columns, expected at least 1") in the eager call
    and in a graph node, in both modes, before any event is fed. 2.8.0 raised
    `IndexError` in the eager count call and `RuntimeError` in a count-mode
    node, and a duration-mode node returned no rows. Reshape `X` to `(T, K)`.
  * In count mode, `(values, index)` pairs raise `TypeError` ("count= mode
    takes X and y as values"); 2.8.0 read each pair as a 2-row array and
    returned unrelated numbers. Pass the values, or use `duration=`.
  * A negative `duration` raises `ValueError` ("duration must be at least
    0"). 2.8.0 accepted it and paired each target with later features, which
    leaks the future into the training set.
  * `count` is from 1 to 2147483647 and `duration` from 0 to the int64
    maximum. A value out of range raises `ValueError` "forecast_pairs: count
    must be at least 1 and at most 2147483647, got 0". 2.8.0 raised
    `ValueError` with `Lag`'s message ("Size must be at least 1.", or "vector"
    for a negative count), and `TypeError` with nanobind's signature text for
    a count of 2**31 or more or a duration beyond int64. A graph node checks
    both when it is built.
  * A `count` or `duration` given as a string raises `TypeError`, as every
    operator setting does (see "A string is never a number"); 2.8.0 read
    `count="3"` with `int()`. Pass `int(count)`.
  * `dropna` must be a bool (or a NumPy bool); anything else raises
    `TypeError` ("forecast_pairs: dropna must be a bool, got str"), as in
    JavaScript. 2.8.0 read it by truth value, so `dropna="no"` dropped rows.
    Pass `bool(dropna)`.
  * Inside a `Pipeline`, `forecast_pairs(X_node, y_node, ...)` is one graph
    node, and it takes an `X` of any width in count mode too (2.8.0 raised for
    a wide `X` in a graph or lazy call). `to_dict()`, `to_json()`, `to_text()`
    and `to_dot()` show one `forecast_pairs` node, where 2.8.0 showed its
    `Lag`, `Delay`, `Resample`, `CombineLatest` and `Dropna` nodes, and
    `from_dict()` restores it. Results are unchanged. A graph saved by 2.8.0
    still loads.
  * Given one `Node` and one array, it raises `TypeError` ("pass X and y both
    as Nodes to build a graph"). In count mode 2.8.0 returned a node that a
    `Pipeline` could not run; in duration mode it raised `Delay`'s
    `TypeError`.
* A Python `_GraphBuilder` or JavaScript `GraphBuilder` compiles once: a
  second `compile()` raises ("this graph builder was compiled already"), since
  two graphs over one builder would share its operators' state. `Pipeline`
  compiles each builder once, so its behavior is unchanged.
* The microstructure operators report a wrong set of inputs with their own
  name and input names. `SignedVolume`, `QueueImbalance`, `RollingKyleLambda`
  and `EwKyleLambda` raise `TypeError` with messages such as
  `RollingKyleLambda() missing 1 required input: 'return_'` and
  `SignedVolume() takes 2 inputs, got 3 positional`, where 2.8.0 gave Python's
  `RollingKyleLambda.__call__() ...` messages. `RollingOrderImbalance` raises
  the `TypeError` of every one-input operator, which names its `signed_flow`
  input. The exception class is unchanged.
* `backtest_report` reports a `periods_per_year` that is not a number as
  `TypeError: backtest_report: periods_per_year is a str, expected a number`,
  where 2.8.0 gave nanobind's message for `BacktestReport.__init__`. The
  exception class is unchanged, and so are the results.
* JavaScript: a `Pipeline` with several outputs aligns them on one index by
  default (`alignOutputs`, above). 2.8.0 ignored the options argument and
  returned each output on its own index; pass `{ alignOutputs: false }` for
  that. An option other than `alignOutputs`, or an `alignOutputs` that is not a
  boolean, throws `TypeError`. `live().result()` is typed `Output | Output[]`,
  as the batch call is.
* `Pipeline(max_pending=...)` takes an integer from 1 to `2**53`: 0, a
  negative number or one above `2**53` raises `ValueError`, and a value that is
  not an integer (`1.5`, `True`, `None`) raises `TypeError`. 2.8.0 read it with
  `int()`, so `1.5` became 1 and `True` 1; it took 0 (every `CombineLatest`
  node then overflowed at its first event), and a negative number when the
  pipeline had no `CombineLatest` node. Pass `int(max_pending)`. JavaScript
  `combineLatest(..., { maxPending })` follows the same rule (`RangeError` /
  `TypeError`); 2.8.0 did not check it, so 1.5 became 1.
* `Pipeline(max_pending=...)` also caps the `Filter` nodes of the pipeline,
  which used 1,000,000 whatever the setting. `to_dict()` and `to_json()` save
  `max_pending`, and `from_dict()` restores it; a graph saved by 2.8.0 has none
  and loads with the default, as before.
* A lazy `Pipeline` call that a rejected event ended (a `CombineLatest` or
  `Filter` overflow) raises at every later `next()` without reading more of
  its feeds ("this session rejected an event"); 2.8.0 read one more event at
  each `next()` and raised the overflow again.
* `Resample(value, driver, threshold=...)` inside a `Pipeline` is one graph
  node with two inputs; the engine joins value and driver itself. Results are
  unchanged, but `to_dict()`, `to_json()`, `to_text()` and `to_dot()` show one
  `Resample` node fed by the value and the driver, where 2.8.0 showed a
  `CombineLatest` node feeding the `Resample`. Graphs serialized by 2.8.0 still
  load and give the same results.
* One index rule, owned by the C++ engine, at every `Pipeline` entry point:
  Python `live().push()`, `advance()`, batch calls and iterator feeds of
  `(value, index)` pairs (also those of the stream operators, such as `Merge`),
  and JavaScript `live().push()`, `pushWide()`, `advance()` and batch calls. An
  index is a number that is finite, whole and inside the int64 range; anything
  else raises `TypeError`, whose message names the problem ("index must be a
  number", "index must be finite", "index must be integer-valued", "index is
  outside the int64 range"). Python takes `int`, `float` and NumPy integer and
  floating scalars and arrays (each element of an object array follows the
  scalar rule); JavaScript takes `number` and `Float64Array`. What
  2.8.0 did, and how to get it:
  * JavaScript truncated a number (`1.5` gave 1, `NaN` gave 0, `2**63`
    saturated) and converted any other value with `Number()` (`"5"` gave 5,
    `null` 0, `true` 1, an object its `valueOf()`). Pass `Math.trunc(Number(index))`.
  * Python `live().push()` and `advance()` truncated with `int(index)`, so
    `1.5`, `"5"`, `b"6"`, `True`, `Decimal` and `Fraction` indices were
    accepted. Pass `int(index)`.
  * Python batch calls saturated a float index of `2**63` or more to
    `2**63 - 1`, wrapped a `uint64` index of `2**63` or more to a negative one,
    and read `str`, `bool` and `Decimal` index arrays through `int()`. Pass
    `np.asarray(index).astype(np.int64)`.
  * Python iterator feeds truncated a NumPy float, `Decimal` or `Fraction`
    index (`np.float32(1.5)` gave 1) and accepted `"5"`, `b"6"`, `True` and
    `np.bool_`. Pass `int(index)`. A stream merge whose first index is a float
    read `True` and `np.bool_` as 1.0 and `Decimal("2.5")` and
    `Fraction(5, 2)` as 2.5, and raised `ValueError` for `None`; these now
    raise `TypeError` ("index must be a number (an int or a float)"). Pass
    `float(index)`. An int index there rounds to the nearest float64, as a
    batch float index does; one too large for a float64 raises `TypeError`
    ("index does not fit a float64").
* Python reads a time index as int64 ticks of its own unit at every `Pipeline`
  entry point, the same way a batch call reads a datetime64 or timedelta64
  array. This covers `live().push()`, `advance()`, iterator feeds (also those
  of `Merge` and the other stream operators) and object index arrays. A time
  index is a `np.datetime64`, `np.timedelta64`, pandas `Timestamp` or pandas
  `Timedelta`. So `Pipeline(...)(zip(values, index))` over a pandas
  `DatetimeIndex` equals the batch call `Pipeline(...)((values, index))`.
  * NaT (and `pd.NaT`) raises `TypeError` ("index must be finite, got NaT"),
    in batch arrays too.
  * A `Pipeline` call, a live session (with its `advance()`) and a stream
    merge (`merge`, `Merge`, `combine_latest`, batch and iterator forms
    alike) keep one index kind across every input: plain numbers, or times of
    one kind and unit. An index of another kind or unit raises `TypeError`
    ("index is a datetime64[ns], but earlier indexes were a datetime64[s]"),
    because ticks of two units, or ticks and plain numbers, cannot be
    compared. This includes a bare feed (whose index is its row number) next
    to a time-indexed feed. A rejected push or `advance()` does not set the
    kind. 2.8.0 merged and joined the raw ticks of two units, or ticks and
    plain numbers, silently. For one unit, convert every index first, for
    example `index.as_unit("ns")` for a pandas index or
    `index.astype("datetime64[ns]")` for a NumPy array (in pandas 2 and later,
    `.asi8` gives ticks of the index's own unit, so it does not unify them).
  * 2.8.0 took only a `datetime64[ns]` or `timedelta64[ns]` scalar (through
    `int()`) and raised `TypeError` for other units, `Timestamp` and
    `Timedelta`. It read NaT in a batch array as -2**63. For the old result
    with NaT, drop those rows or replace them first.
* A `Pipeline` iterator feed reads a `None` value as NaN, as a live push and
  a batch call do (A9); 2.8.0 raised `TypeError`. An event with no values
  (`([], index)`) raises `ValueError` ("the value of event 1 of input 0 is
  empty"), as a live push does; 2.8.0 pushed 0.0. A `bytes` or `bytearray`
  value raises `TypeError`; 2.8.0 read `b"3"` as the one-column event 51.0
  (its byte code). These rules hold for the iterator feeds of the stream
  operators (`Merge` and the others) too.
* A string is never a number. Any string value (`"abc"`, `""` or `"3"`,
  also NumPy's `str_` and `bytes_`, a `bytearray`, and a JavaScript `String`
  object) raises `TypeError` on every value path in both languages:
  operator scalars, lists, arrays (a string array, or an object array or
  pandas object holding a string), iterator and async feeds, row operators
  (`CrossRank` and the others), `PortfolioReport` and the portfolio engines,
  `Pipeline` batch, iterator and live feeds, the Python stream functions
  (`merge`, `combine_latest` batch and iterator, `dropna`, `select`, `split`,
  `resample`, `delay`, `Filter`), `reshape` and `from_pandas`, and the
  validation and backtest helpers. An operator setting given as a string
  raises `TypeError` too: an operator's number settings (`RollingMean("3")`
  raises "RollingMean: window_size must be an integer, got str", and
  `Lag(b"2")`),
  its per-asset and list settings, alone or as an element
  (`MovingAverage(["0.5", "0.5"])`, `BacktestPortfolioTarget(fee="0.01")`),
  and `threshold`, `every`, `count`, `origin`, `max_age`, a `delay`
  duration, `select` columns, `max_pending` and JavaScript's `maxPending`. A
  bool setting raises `TypeError` too (see the bool entry below). What 2.8.0
  did:
  * Python parsed numeric strings in string and object arrays, pandas
    `Series`, `Pipeline` batch feeds and live pushes, row operators,
    `PortfolioReport`, the stream functions and the lazy positional
    `combine_latest`, in integer operator settings
    (`RollingMean("3")` had a window of 3) and in settings read with `int()`
    or `float()` (`every="2"`), and raised NumPy's `ValueError` for other
    strings. An operator iterator read a string event as a one-asset array.
  * JavaScript converted a string or `String` object in a `Pipeline` feed,
    `live().push()` or `pushWide()` with `Number()` (`"abc"` gave NaN, `""`
    gave 0). It read a string operator setting through embind
    (`RollingMean("3")` had a window of 3; `MovingAverage(["0.5", "0.5"])`
    built a two-tap filter, and `MovingAverage("0.5")` read the characters
    as taps), a resample setting with
    `Number()` or `Math.trunc()`, `delay(node, "1")` and `select(node, "0")`
    with `Math.trunc()`, and did not check `maxPending`.
  * For the old result, parse the text first: `pd.to_numeric(column)` or
    `values.astype(float)` (and `int(setting)`) in Python, `Number(value)` or
    `Float64Array.from(values, Number)` in JavaScript.
* JavaScript: a list operator setting (a `number | number[]` parameter such
  as `MovingAverage` taps or a portfolio engine's `fee`) is a number, an
  `Array` or a typed array, and every element is a number, as in Python. A
  `Set`, a generator, a `Map` or another object throws `TypeError` ("taps must
  be a number or an array of numbers, got a Set"), and so does an element that
  is not a number (a nested array, an object, a boolean). 2.8.0 read any
  iterable and passed each element to embind, which converted it with
  `Number()` (a nested `[[3]]` gave 3). For the old result, pass
  `Array.from(x, Number)`.
* `None` (Python) and `null` (JavaScript) are a missing value, NaN, on every
  value path in both languages: an operator scalar (`RollingMean(2)(None)`),
  a list or array element, an iterator or async event, a row of a row
  operator, every `Pipeline` feed, and the lazy positional `combine_latest`,
  as its batch form already did. 2.8.0 raised `TypeError` for a Python `None`
  given to an operator directly, in a list or in an iterator (an object array
  already read it as NaN) or to the lazy positional `combine_latest`, and
  JavaScript operators threw `TypeError` for `null`. For the old error,
  check for missing values before the call.
* The standalone `resample(values, clock, clock=True)` applies the index rule
  to both indexes, so a fractional index raises `TypeError`; 2.8.0 cast it to
  int64, which truncated 1.5 to 1. Pass `np.trunc(index)` for the old result.
* Python `live().push()` raises `TypeError` for a `memoryview` over `bytes` or
  `bytearray` (or of format `"c"`); 2.8.0 read `memoryview(b"12")` as 12.0.
  Pass `float(value)` for the old result. A numeric `memoryview` is one
  multi-column event (2.8.0 raised `ValueError`).
* A complex value raises `TypeError` in `live().push()` (a number, a NumPy
  complex scalar, or a sequence or array holding one) and in a `Pipeline`
  batch feed, as operators do; 2.8.0 dropped the imaginary part of a NumPy
  complex value with a `ComplexWarning`. Pass `np.real(value)` for the old
  result. An object array
  is read as before.
* A `Pipeline` checks the whole graph against an input's width, before any
  operator sees an event: a live session when the input's first event
  arrives, a batch call before feeding. A width that a node cannot take raises
  with nothing pushed, and a live session behaves as if the event had not been
  sent. The check covers operators (each input of an operator must carry the
  operator's columns; to join streams into one row, use `CombineLatest`),
  `select` columns, `resample` widths (a cumulative resample, a reducer,
  the fixed plans of `ohlc_bars`, `ohlcv` and `ohlcv2`, `ohlcv_bars` and
  `max_age`) and `Filter` (see below). The error is `ValueError` (JavaScript
  `RangeError`) with a message such as "select needs column 5, but its input
  has 2 column(s)"; 2.8.0 raised `RuntimeError` from inside the graph, after
  earlier nodes had taken the event. A width that a node sees only at run
  time raises the same `ValueError` from the node itself. A rejection that only a node can decide later (for example
  a stalled `combine_latest` buffer) poisons the session: `push`, `advance`,
  `flush` and `result()` raise until `reset()` or a new `live()`, so no row
  that the rejected event produced is returned.
* `Filter` gates one data value by one mask value, so its data and its mask
  must each be one column. A wider data or mask stream raises `ValueError`
  (JavaScript `RangeError`) before any event is fed, in a `Pipeline`, a live
  session and the eager `Filter()(data, mask)`. 2.8.0 used column 0 of a wide
  data or mask stream and dropped the other columns silently. For the old
  result, select column 0 first: `Filter()(Select(columns=[0])(data), mask)`.
* `agg="ohlcv_bars"` takes at least 5 columns (`[open, high, low, close,
  volume, ...]`) on every path, as its reference page says: a `Pipeline`
  resample node, a clock resample and JavaScript `resample`, as the eager
  `resample()` already did. 2.8.0 accepted 4 columns inside a `Pipeline`
  (and in JavaScript) and returned `[open, high, low, close]` bars. Fewer
  than 5 columns now raise `ValueError` (JavaScript `RangeError`); for the old
  result on 4 columns, use `agg="ohlc_bars"`.
* The batch forms of `merge`, `Merge` and `combine_latest` accept a stream
  with an int index next to one with a float index, as the iterator forms
  do. The first non-empty index sets the kind, as in the iterator: after a
  float index every index is read as float64; after an int index a float
  index must be whole (else `TypeError` "integer-valued"), and the merged
  index is int64. 2.8.0 raised `TypeError` ("all streams must share one index
  type") for the batch forms. For the old error, check the index dtypes
  before the call. A bool index array is not an index in these batch forms,
  as in a `Pipeline` batch call and the iterator forms: it raises `TypeError`
  ("index must be a number ..., got an array of bool"). 2.8.0 read it as 0
  and 1; pass `index.astype(np.int64)` for the old result.
* Python `live().push()` raises `TypeError` for a `bytes` or `bytearray` value;
  2.8.0 read `b"12"` as 12.0. Pass `float(value)` for the old result.
* A list of rows passed to a row operator (`CrossRank()([r1, r2])`) returns a list of result rows, following rule 3; 2.8.0 returned an array. To get the array, pass `np.vstack(rows)`. A tuple of rows and a matrix still return an array. A mix of lists and arrays (for example a list target with an ndarray of prices) returns an array. The JavaScript row operators follow the same rule: an `Array` of rows gives an `Array` of `Float64Array` rows, and an `NdArray` gives an `NdArray`. To get the old `NdArray` from an array of rows in JavaScript, pass `{ data, shape: [T, K] }`.
* pandas `Series` and `DataFrame` inputs are read as their values and give a NumPy array (2.8.0 treated them as iterators). `str`, `bytes`, `dict` and `set` inputs raise `TypeError` at the call.
* **JavaScript error classes:** every shape, length or input-count mismatch throws `RangeError` (the counterpart of Python's `ValueError`); wrong kinds still throw `TypeError`.
* JavaScript: the WebAssembly module uses native WebAssembly exceptions; it needs Chrome 95, Firefox 100, Safari 15.2 or Node 17 or later. Errors thrown by your own iterators now reach you unchanged.
* JavaScript: a string parameter given a non-string (for example `RollingMean(3, 5)`) throws `TypeError`.
* JavaScript: streaming one value at a time (`op(x)`) is about 2.3 times faster; heap views are made in JavaScript instead of through Embind.
* One-input operators such as `RollingMean` raise `TypeError` for an input
  they cannot read (was `ValueError` with "Unsupported input type for call").
  Code that caught `ValueError` for this case should catch `TypeError`.
* `strategy_test` (Python) and `strategyTest` (JavaScript) run in one C++
  implementation that calls your strategy back. They are faster, 1.1 to 5
  times in our measurements, most when the strategy itself is quick.
  The error for a strategy that returns no engine output now ends "or a
  pipeline output holding one" in both languages (was "or a (values, index)
  pair holding one" in Python). JavaScript errors now use the Python text, for
  example `method="shift" needs the feature positions in columns` and
  `strategy_test needs at least one input`; the classes are unchanged. In
  JavaScript, a portfolio of 8 or more assets is summed into its book in
  NumPy's pairwise order, as in Python, so its statistics can differ from
  2.8.0 in the last bits; compare with a tolerance.
  `n` above 2147483647 raises "n is too large" (`ValueError` / `RangeError`),
  in both languages, and
  `n=True` raises `TypeError` "n must be an integer, not bool". Integers
  outside int64 in `price`, `volume`, `fixed`, `ohlc`, `columns`, or
  `min_shift` raise the `ValueError` of that argument. In JavaScript,
  `volume: null` means no volume (as `None` in Python; 2.8.0 raised
  `RangeError`), and `minShift: null` the default.
* `position_test` and `walk_forward` (Python) and `positionTest` and
  `walkForward` (JavaScript) run in one C++ implementation that calls your
  engine and a custom metric back. Results are unchanged (except for infinite
  metrics, see "Fixed"), and the calls are
  1.2 to 3 times faster in our measurements. Changes you can see:
  * JavaScript errors now use the Python text, with the same classes:
    `metric must be one of ['avg_holding', ...] or a callable`,
    `metric "turnover" has no natural direction; pass larger_is_better`,
    `several strategies need method="shift", which shifts them jointly`,
    `a portfolio is tested with method="shift", which moves all assets together`,
    and `walk_forward needs a single-asset engine; portfolio engines are not
    supported`. The metric texts also apply to `strategyTest`.
  * The seed is checked in C++ for every randomization function, so the
    JavaScript text is `seed must be an integer in [0, 2**53)` (was `2^53`) in
    `positionTest`, `strategyTest`, `circularShift`, `segmentShuffle`,
    `permuteBars`, and `permuteReturns`.
  * A custom metric (here and in `strategy_test` / `strategyTest`) must
    return one number per strategy in its input. The wrong count raises
    `ValueError` / `RangeError` "metric must return one value per strategy in
    its input"; in Python the metric sees k strategies at once, in JavaScript
    one, so a JavaScript metric returns a number or a one-element array. In
    2.8.0 Python spread a single value over every randomized run of a chunk,
    and took the first of several values with a portfolio engine and in
    `strategy_test`; JavaScript stored whatever the metric returned, so an
    array or `undefined` became NaN. Both languages accept the same kinds of
    result: a number, or a 1-D array of numbers (Python: an `int` or `float`
    or NumPy number, a 1-D numeric `ndarray`, or a flat `list` or `tuple` of
    numbers; JavaScript: a `number`, a typed array of numbers, or a flat
    `Array` of numbers). Anything else raises `TypeError` "metric must return
    a number or a 1-D array of numbers": `None`/`undefined`, a string or a
    numeric string, a bool or an array of bools, a nested list or array, a 2-D
    `ndarray` (also of shape (1, k): flatten it), and an object. 2.8.0 Python
    converted bools, numeric strings, nested lists, and 2-D arrays to numbers.
  * An engine output of the wrong shape raises `ValueError` / `RangeError`
    "engine must return a backtest output of shape (T, 4) ..." (2.8.0 scored
    what it could read, for example a JavaScript output with fewer values
    than its shape). A target with no columns raises "target needs at least
    one column" (2.8.0: a NumPy error). `walk_forward` with no candidate
    columns now raises "targets must hold at least one candidate" (2.8.0
    returned a result with every choice -1 in Python and raised `TypeError`
    in JavaScript).
  * `walk_forward`: a non-finite `train` or `test` raises `ValueError` "train
    (test) must be an integer of at least 2 (1)"; 2.8.0 Python raised
    `OverflowError`, and JavaScript keeps its 2.8.0 text.
  * `n` above 2147483647 raises "n is too large" (`ValueError` /
    `RangeError`), in JavaScript too: the results stream into JavaScript
    arrays, so the WebAssembly heap no longer bounds n. With K strategies the
    JavaScript limit is 2147483647 divided by K, since each result array holds
    n * K values. `n=True` raises `TypeError` "n must be an integer, not
    bool", and in JavaScript `n: Infinity` raises "n is too large" (was "n
    must be at least 1").
  * JavaScript `positionTest` checks its market inputs as Python does: each
    must be a series for one strategy or K strategies ("market input must be a
    1-D series"), and must have the target's (T, K) shape for a portfolio
    engine. It also accepts an `NdArray` target: (T,) for one strategy, and
    (T, K) for K strategies, which gives a `PositionFamilyResult` (2.8.0
    raised `RangeError` for both).
    The TypeScript overloads type the result from the shapes: a (T,) target
    gives a `PositionTestResult`; a (T, K) target (K columns, or an `NdArray`
    whose shape type is `[number, number]`) with series markets (arrays or
    `NdArray`s of shape `[number]`) gives a `PositionFamilyResult`; panel
    markets give the portfolio `PositionTestResult`; where the rank is not in
    the types, the result is the union. The new types `NdSeries`, `NdPanel`,
    `SeriesInput`, and `PanelInput` are exported.
  * JavaScript: a portfolio of 8 or more assets is summed into its book in
    NumPy's pairwise order, as in Python, so its statistics can differ from
    2.8.0 in the last bits; compare with a tolerance.
  * JavaScript call patterns, with the same results: with K strategies the
    engine gets one (T, k * K) block per chunk instead of K blocks; one
    strategy now uses Python's chunks (at most about 32 million values of
    working arrays), so a long series gets several (T, k) blocks where 2.8.0
    passed one (T, n) block (for example (T, 7), (T, 7), and (T, 3) at
    T = 200000 and n = 17); and `walkForward` no longer calls a custom metric
    on the out-of-sample window, whose value it never used.
  * JavaScript: `null` in the options of `positionTest` and `walkForward`, and
    `minShift: null` in `circularShift` (was a `RangeError`), mean the default.
  * Python: the family p-value no longer warns "All-NaN slice encountered" for
    a round where every strategy's metric is NaN.
* `permute_bars` and `permute_returns` on one series write 8 copies per pass
  and are 1.2 to 2 times faster; the copies are unchanged.
* `strategy_test` input errors: a `str` or `None` input raises `TypeError`
  (2.8.0 raised `ValueError`), so catch `TypeError` for these. A 0-d scalar
  input still raises `ValueError`, now with the message "input must be 1-D
  (T,) or 2-D (T, K)".
* A trailing axis of size 1 on an array of rank 3 or more is the single input
  field: `(T, K, 1)` input gives `(T, K)` (was `(T, K, 1)`), and `(T, K, M)`
  for M-output operators (was `(T, K, 1, M)`). Old result: `out[..., None]`,
  or `np.expand_dims(out, -2)` for M outputs.
* A generator passed to `IsNan`, `Not`, `IsFinite`, `TrendMode`, `CyclePhase`,
  `CycleAmplitude`, `CycleFrequency`, `DominantCycle`, or
  `InstantaneousTrendline` returns a lazy iterator, as for every other
  operator (was a list). Old result: `list(IsNan()(gen))`.
* `EwMean` on numbers, streams and multi-column arrays uses the decay step of
  its 1-D array path, so every path gives the same bits. Those results differ
  from 2.8.0 in the last bits (at most 5.7e-14 on a random walk near 100);
  there is no switch back, so compare with stored 2.8.0 output using a tolerance such
  as `np.allclose`.
* GCC builds (the Linux wheels) no longer fuse a multiply and an add across
  statements into one instruction (`-ffp-contract=off`), so batch and
  streaming results stay identical there. Clang builds (macOS wheels and
  WebAssembly) keep fusing within one expression, which both paths share.
  Results can therefore differ in the last bits between platforms; compare
  across platforms with a tolerance such as `np.allclose`.
* `Return` and `ROCP` on a 1-D array compute `(x[t] - x[t-k]) / x[t-k]`, as
  their other paths do (was `x[t] / x[t-k] - 1`), so a 1-D result equals the
  same column of a 2-D one. The values differ by at most one unit in the last
  place. Old result: `x / Lag(k)(x) - 1`.
* Every call other than `op(number)` is an independent run, in Python and
  JavaScript. A stream (iterator or async iterator) starts clean even on an
  operator that already holds state, runs on copies of its own, and resets the
  operator's running series when it opens, like a batch call. Two streams of
  one operator never share state, and live calls `op(x)` never touch a running
  stream. A Python list starts clean and leaves the operator reset, like an
  array; before, a list into a multi-input or multi-output operator continued
  its state, and a list into a one-input operator left its state behind. In
  JavaScript a stream already started clean, but ran on the operator itself,
  so interleaved streams shared one state. Streaming an operator built outside
  the `screamer` namespace (Python) or without its factory (JavaScript) raises
  `TypeError`. Old result: `(op(x) for x in source)` continues the running
  series as a 2.8.0 Python stream did, and `[op(x) for x in values]` (or
  `op(*event)` for events of several inputs) does the same for a list.
* A value of the wrong kind inside a list, tuple stream or iterator raises `TypeError` that names its type and position; before, it raised `RuntimeError: std::bad_cast`. For example, `RollingMean(2)([1.0, {}])` raises `TypeError: RollingMean: event 1 is a dict, expected a number`. An element that holds only numbers but is not one number, such as `[1.0, 2.0]` in `RollingMean(2)([1.0, [1.0, 2.0]])`, raises `ValueError: RollingMean: event 1 has shape (2,), expected a number` on the list, iterator and async paths alike (a list raised `RuntimeError: std::bad_cast`; an iterator raised `ValueError` with "every event must have the same shape").
* JavaScript: the same cases throw `TypeError` (wrong kind) or `RangeError` (wrong shape) with the same messages. Before, a list of events or an iterator item of the wrong kind threw `RangeError`, and other wrong-kind list elements threw `TypeError` without the position. A `Pipeline` feed, index or `live().push()` / `pushWide()` value that is a function, or an object that does not convert to a number (for example `{}`), throws `TypeError`; before, it was converted with `Number()` (`{}` and functions gave `NaN`). Numbers, booleans, `undefined` and `Date` values convert as before; a string raises `TypeError` and a `null` value is `NaN` (see above and below). An index follows the index rule above: it must be a number.
* `PortfolioReport` given a list of `(K, 4)` engine rows returns a list of `(6,)` result rows, following rule 3; 2.8.0 returned a `(T, 6)` array. To get the array, pass `np.stack(rows)` or a tuple of rows. A tuple of rows and an array still return an array. In JavaScript an `Array` of rows (nested arrays or `[K, 4]` `NdArray` events) gives an `Array` of `Float64Array` rows; to get the old `NdArray`, pass `{ data, shape: [T, K, 4] }`.
* A multi-input operator given a mix of lists and arrays returns an array, in either argument order: `RollingCorr(5)(x_list, y_array)` returned a list and `RollingCorr(5)(y_array, x_list)` raised `TypeError`; both now return the array that two arrays give. To get the old list, pass lists for every input (`list(y_array)`), or call `.tolist()` on the result. A list among arrays reads as a 1-D array of its length and broadcasts over the asset axes of the other inputs; lengths that differ still raise `ValueError`. All lists still return a list. JavaScript already returned an array for these mixes.
* JavaScript: a `null` value in a `Pipeline` feed, `live().push()` or `pushWide()` is a missing value, `NaN`, as Python's `None` is; before, it was 0. To get the old value, replace `null` with 0 before the call. A `null` index raises `TypeError` (the index rule above). A `null` given to an operator is NaN too (see above).
* JavaScript: an `Array` of events into an operator returns an `Array` with one result per event, as a Python list does. `Add()([[1, 2], [3, 4]])` returns `[3, 7]` (was a `Float64Array`), and an `Array` of `[K, N]` `NdArray` events returns an `Array` of per-event results (was one `NdArray`). An `Array` of `Float64Array` or `NdArray` asset events into a one-input operator, `RollingMean(2)([a, b])`, is a sequence of events and returns an `Array` of per-event arrays; before, it threw `RangeError`. To get the old container, pass the columns or one packed `NdArray`: `Add()(Float64Array.from(evs, (e) => e[0]), Float64Array.from(evs, (e) => e[1]))`. Likewise, a multi-input operator given only `number[]` inputs returns an `Array` (an `Array` of `number[]` for a multi-output operator), as Python lists give a list: `Add()([1, 2], [3, 4])` returns `[4, 6]` (was a `Float64Array`). To get the old result, pass `Float64Array` inputs (`Float64Array.from(x)`); one `Float64Array` or `NdArray` among the inputs is enough.
* JavaScript: in an `Array` of asset events into a one-input operator, the
  first event (a `Float64Array` or an `NdArray`) fixes the event shape, and a
  later event may be a plain `Array` of that shape (nested for an `NdArray`),
  as in Python: `RollingMean(2)([Float64Array.from([1, 2]), [3, 4]])` returns
  `[[NaN, NaN], [2, 3]]`. Before, a mix of a `Float64Array` and a `number[]`
  threw `RangeError`. A first event that is a `number[]` keeps its meaning
  (an event of numbers), as in Python.

* `RollingZscore` follows `start_policy` as `RollingMean` and `RollingStd`
  do: `"expanding"` gives the z-score over the values seen so far (`NaN` for
  the first value only) and `"zero"` counts the missing values as zeros. In
  2.8.0 every policy gave `NaN` until the window was full. The default,
  `"strict"`, gives the same results as before, bit for bit. Old result:
  `RollingZscore(window_size)` (the default policy) for any policy.
* `RollingOU` raises `ValueError` (JavaScript: `RangeError`) for
  `window_size` below 3. A window of 2 values holds one pair, which has no
  slope, so 2.8.0 returned `NaN` at every step for every output. Use
  `window_size >= 3`, or `>= 4` for `output="std"`.
* The Ehlers cycle operators follow Ehlers' *Rocket Science for Traders* as
  TA-Lib implements it, so their results change. There is no setting for the
  2.8.0 results; each one is listed below with what changed.
* `CycleSine` computes Ehlers' sinewave indicator from the dominant cycle
  phase (a Fourier sum over one measured cycle), as TA-Lib `HT_SINE` does.
  `sine` follows the cycle in phase and `leadsine` leads it by 45 degrees. In
  2.8.0 both used the `CyclePhase` angle, which runs backwards, so `leadsine`
  lagged `sine` by one eighth of a cycle. The values match `HT_SINE` after
  the start-up transient.
* `TrendMode` applies Ehlers' trend-mode rule, as TA-Lib `HT_TRENDMODE` does:
  cycle near a crossing of the sinewave lines or while the phase rotates at
  the cycle rate, trend whenever the smoothed price is 1.5% or more away from
  the instantaneous trendline. It has no settings; the `phase_rate_frac`
  parameter of 2.8.0 is gone, and passing it raises `TypeError`. A steady
  ramp now reads trend on every sample (2.8.0: 64%). The 1.5% test is
  relative, so the rule is meant for price series: a series that oscillates
  around zero reads trend throughout. The values match `HT_TRENDMODE` after
  the start-up transient.
* `InstantaneousTrendline` is Ehlers' instantaneous trendline, as TA-Lib
  `HT_TRENDLINE` computes it: the mean over one measured dominant cycle, then
  a 4-tap weighted mean. It removes the dominant cycle, where the 2-pole
  filter of 2.8.0 kept about 57% of it, and in exchange lags a ramp by half a
  cycle plus one sample (2.8.0: no lag). The values match `HT_TRENDLINE`
  after the start-up transient.
* `MAMA` uses Ehlers' published phase, `arctan(Q1 / I1)`, as TA-Lib `MAMA`
  does, instead of the `atan2` phase of `CyclePhase`. The phase wraps every
  half cycle, so the smoothing constant steps to `fast_limit` twice per cycle
  (2.8.0: once). The values match TA-Lib `MAMA` once its start-up (both
  averages start at 0 there) has faded.
* `HilbertPhasor` returns Ehlers' in-phase and quadrature components `I1` and
  `Q1`, the pair of TA-Lib `HT_PHASOR`, each divided by its filter gain at the
  measured period, so a sine of amplitude `A` gives components of amplitude
  `A` at periods from about 7 to 50. 2.8.0 returned the smoothed homodyne pair
  `(I2, Q2)`, whose size depended on the period. `CycleAmplitude`, the
  magnitude of the pair, now reads 1.00 for a unit sine of period 10, 20 or
  40 (2.8.0: 0.41, 1.08, 1.41). This keeps Ehlers' convention, the magnitude
  of `(I1, Q1)` as the cycle amplitude at about unit gain, and makes the
  gain exact. To get TA-Lib's `HT_PHASOR` values,
  multiply each component by its gain (see the `HilbertPhasor` page).

* A bool is not a number, in Python as in JavaScript. A number setting or a
  function's number argument refuses `True`, `False`, a NumPy bool and a 0-d
  NumPy bool array with `TypeError`, and the message names the operator (or
  function) and the setting: "RollingMean: window_size must be an integer,
  got bool", "BollingerBands: num_std must be a number, got bool". This
  covers every operator's integer, float, string, per-asset and list
  settings (`MovingAverage([True, 0.5])` and
  `BacktestPortfolioTarget(fee=True)` raise too); the stream settings,
  `Resample`'s `freq`, `every`, `count`, `threshold`, `origin` and
  `max_fill`, a `Delay` duration, `Select` columns, `split`'s `n`,
  `forecast_pairs`' `count` and `duration`, and a `Pipeline`'s
  `max_pending`; and the number arguments of `probabilistic_sharpe`
  (`benchmark`), `circular_shift`, `segment_shuffle`, `permute_bars`,
  `permute_returns`, `position_test`, `strategy_test` (`n`, `seed`,
  `min_shift`, the column positions, `periods_per_year`), `walk_forward`
  (`train`, `test`, `periods_per_year`) and `backtest_report`
  (`periods_per_year`). JavaScript does the same for its counterparts
  (`resample`'s `count`, `every`, `threshold`, `origin` and `maxAge`,
  `delay`, `select`, `split`'s `n`, `forecastPairs`' `count` and `duration`,
  `probabilisticSharpe`, `circularShift`, `segmentShuffle`, `permuteBars`,
  `permuteReturns`, `positionTest`, `strategyTest`, `walkForward` and
  `backtestReport`), with "boolean" for the type: "RollingMean: windowSize
  must be an integer, got boolean". 2.8.0 read a bool as 1 or 0:
  `RollingMean(True)` had a window of 1 and `probabilistic_sharpe(pnl, True)`
  a benchmark of 1. Pass `int(flag)` (`Number(flag)` in JavaScript).
* A setting of the wrong kind names the operator and the setting in Python,
  in the form JavaScript uses: "BollingerBands: num_std must be a number, got
  str", "RollingMean: start_policy must be a string, got int",
  "RollingMean: window_size must be an integer, got 2.5". 2.8.0 raised
  nanobind's "incompatible function arguments" for these; the type of the
  error, `TypeError`, is unchanged. An integer setting says "must be an
  integer" in both languages, also for a string or null, and a list given
  for one number is a wrong kind too ("RollingMean: window_size must be an
  integer, got list"). A setting that is not finite where the operator needs
  a finite number raises `ValueError` (`RangeError` in JavaScript) in one
  form, naming the setting passed: "EwMean: span must be finite, got inf"
  (JavaScript: "got Infinity"), or "must be finite or NaN" for a setting
  that also takes NaN. 2.8.0 gave each operator's own text, which often named
  another setting ("Alpha must be a finite value between 0 and 1").
* A JavaScript integer setting refuses a fraction, NaN and an infinity with
  `TypeError` ("RollingMean: windowSize must be an integer, got 2.5"), as
  Python does. 2.8.0 truncated it: `RollingMean(2.5)` had a window of 2.
  Pass `Math.trunc(x)`.
* `BollingerBands`' `num_std` and `SchmittTrigger`'s `lower` and `upper`,
  documented as finite, refuse an infinity with `ValueError` (`RangeError`
  in JavaScript), as they refused NaN. 2.8.0 took `BollingerBands(20, inf)`,
  whose bands were infinite, and `SchmittTrigger(-inf, 3)`.
* `Resample(fill="nan")` and `fill="carry"` raise `ValueError` (`RangeError`
  in JavaScript) under `count=` and `threshold=`: "resample: fill= applies to
  freq= (every=) and clock= bars; a count= bar is never empty". 2.8.0
  accepted them and ignored them. Leave `fill` out (`"skip"`) there.
* A JavaScript `Pipeline` call takes one argument, the feed or the feeds by
  input name; a second argument, such as an index, throws `TypeError` (a
  trailing `undefined` is not an argument).
  2.8.0 ignored it, so `p(values, index)` ran on the row numbers. Pass
  `p({ values, index })`.
* `Resample(fill="carry")` carries each column on its own, under `freq=` (and
  `every=`) and `clock=`, in Python and JavaScript, eager, lazy, in a
  `Pipeline` and live: a NaN cell of a bar takes its column's last non-NaN
  value, and an empty bar repeats each column's last non-NaN value. A
  `(T, K)` input gives the values of `K` one-column calls. 2.8.0 repeated
  the previous row as it was: under `clock=` a bar with no finite value in
  any column took the previous row, while a NaN cell next to a value stayed
  NaN, and under `freq=` a bar whose events were all NaN stayed NaN, and so
  did the empty bars after it. `forecast_pairs` in duration mode, a clock
  resample with carry, follows: a NaN in one column of `X` gives that
  column's last value. For the 2.8.0 values, resample with `fill="nan"` and
  fill the rows yourself: under `freq=`, give each bucket that the
  `fill="skip"` result lacks the row before it; under `clock=`, give each
  row that is all NaN the row before it.
* An empty result keeps its width when the graph or the input fixes it, in
  Python and JavaScript, batch and live. A `Pipeline` fed an empty `(0, 3)`
  array returns `(0, 3)` values (2.8.0: `(0,)`), a `Resample(agg="ohlc")`
  of that feed inside it `(0, 12)` (2.8.0: `(0, 4)`), and an eager `Delay`
  of an empty `(0, 3)` stream `(0, 3)` (2.8.0: `(0,)`). An empty
  `live().result()` keeps each output's width, aligned or not, and an output
  whose width the graph fixes (an operator's outputs, a `Select`, a reducer
  `Resample`) has it before any event. Only a one-column output is `(0,)`.
  An output whose width comes from a live input that has had no event yet
  has the width it would have for a one-column input (`(0,)`, or `(0, 4)`
  for `ohlc`). For the 2.8.0 shape, take `values.reshape(0)` when
  `len(values) == 0`.
* `Pipeline.to_dict()` holds a `Resample`'s functor reducer as its name and
  parameters, `{"functor": "RollingMean", "params": {"window_size": 5}}`, so
  `to_json()` writes it; 2.8.0 put the functor object itself in the dict, and
  `to_json()` raised "Object of type RollingMean is not JSON serializable".
  `from_dict()` and `from_json()` build a new functor from it, so the
  restored pipeline shares no state with the original (2.8.0's `from_dict()`
  reused the object). To get the object, restore the pipeline with
  `Pipeline.from_dict(d)`, which builds each functor, positional and nested
  parameters included.

### Fixed

* A clock `Resample` inside a `Pipeline` whose values come through a node
  that releases them late (a `Filter`, a `threshold=` or `count=` `Resample`,
  a `CombineLatest`) put a value that came out after a later clock tick into
  the later bar, so the bars depended on the node's timing, and a live
  session with `advance(now)` gave other bars than a batch call. A tick now
  emits its bar once no value at or below it can still come, so each bar
  holds the values with index in `(previous tick, tick]` in batch, lazy and
  live calls whatever the `advance()` and `result()` timing. Example: a value
  at 5 behind a `Filter` whose mask moves past 5 only at 20, with clock ticks
  at 6, 9 and 12, gave the bar at 12 in 2.8.0 and gives the bar at 6 now.
  Clock events at one index give one bar, also when a value ahead of the
  clock (a `Delay`) closed the first one. A live session whose inputs come
  in a different order across inputs (each in its own order) gives the batch
  bars too. There is no setting for the 2.8.0 result, which depended on when
  the upstream node released the value. In a live session a bar now comes
  out once the value stream has passed its tick `t`: a value above `t`, an
  `advance(now)` with `now > t`, or `flush()`. After a value at 50 and ticks
  at 60 and 70, bar 60 comes out at `advance(61)` (`advance(60)` is not
  enough, since a value at 60 still joins it). 2.8.0 emitted it at the next
  event of either input, here the tick at 70; to get a bar at that time,
  call `advance(now)` with the time of the event.
* A clock `Resample` now passes watermarks on, so `advance(now)` releases the
  other outputs' aligned rows while the clock is silent, in a live session
  and a lazy call. In 2.8.0 those rows waited for the next clock tick, and
  the resample kept every value since the last tick in memory: a live session
  whose clock stopped grew by about 60 bytes per value, with or without
  `advance()`. With periodic `advance()` its memory is now flat.

* A clock resample (`clock=True`) of an input with two or more asset axes,
  such as `(T, A0, A1)` or `(T, A0, A1, A2)`, returned a wrong array in 2.8.0
  for any agg: `(T', A0)`, or `(T',)` when `A0` is 1. It now resamples every
  column and keeps the asset axes, `(T', A0, A1)`, or `(T', A0, A1, 4)` for
  `ohlc`, as `freq=` and `count=` do.
* A Python `Pipeline` batch feed of rank 3 or more raises `ValueError`
  ("Pipeline: the values of input 0 have shape [30,1,3], expected (T,) or
  (T, w)"), as JavaScript's `Pipeline` does. 2.8.0 read such a feed as
  `(T, shape[1])` over the wrong stride, so every graph got misaligned values
  without an error (a `(T, 3, 2)` feed gave the rows `[0, 1, 2]`,
  `[3, 4, 5]`, ...). Reshape the feed to `(T, w)`.
* `forecast_pairs` in duration mode with a 2-D `X` of `K` columns returns all
  `K` shifted feature columns, as a `(rows, K)` array, as count mode does.
  2.8.0 returned only the first column, as a 1-D array. For the 2.8.0 result,
  pass `X[:, 0]`.
* JavaScript: an array of rows given to `positionTest` (a portfolio target)
  or `strategyTest` (an input) reads `null` as `NaN` and rejects a string
  with `TypeError`, as every other value path does. Before, `null` read as 0
  and a string was converted.
* A `Resample(every=..., fill="nan")` or `fill="carry"` fed through a
  `CombineLatest` or a `Filter` hung at the end of the input and grew memory
  until it was killed (`std::bad_alloc` in JavaScript), in a batch call and in
  `live()`: the end-of-input watermark made it fill every empty bucket up to
  the int64 maximum. The end of input now emits the open bucket only if it has
  data, as for a resample fed directly, and `advance(now)` emits the buckets up
  to `now`.
* A label-left `Resample(every=...)` whose first event came after a watermark
  (rows that a `Dropna` or `Filter` dropped, or an `advance()`) told the nodes
  downstream that no row would come below that watermark, then emitted its
  first bar below it. A `CombineLatest` downstream then lost rows: with
  `a = [NaN] * 10 + [1, 2, 3, 4, 5]` on index 0 to 14,
  `CombineLatest()(a, Resample(every=100)(Dropna()(a)))` gave index 0 and 10 to
  14 in 2.8.0, and now gives 0 to 14. Before its first event the resample now
  holds its watermark at the bucket that event can open.
* `Pipeline.from_dict()` and `from_json()` raised "unknown DAG operator" for a
  graph with a `Filter` or `Delay` node, which `to_dict()` saves. Both load now.
* `live().result()` of a `Pipeline` with several aligned outputs aligned each
  call's rows on their own, so calling it more than once dropped rows: with
  events a0 b1 a2 b3 a4 a5 b6 and `result()` after b3, after a5 and at the end,
  2.8.0 gave the indexes [1, 2], [4], [6] where a batch call gives 1 to 6. The
  alignment now carries each output's latest value from one call to the next,
  and a call holds back a row until every output has moved past its index, so
  the calls, put end to end, give the batch rows ([1], [2], [3, 4, 5, 6] here).
  After `flush()` a call returns every row.
* A `Pipeline` with several outputs and `align_outputs=True` (the default)
  returned the wrong columns when an output had more than one column, in a
  batch call and in `live().result()`: `Pipeline([a], [a, Select(columns=[0])(a)])`
  on `[[1, 2], [3, 4]]` gave `([1, 3], [2, 4])`. Each output now keeps its own
  columns (`([[1, 2], [3, 4]], [1, 3])`), as the lazy call already did.
* JavaScript: `resample(x, { threshold, driver })` threw `RangeError` at the
  first event, because the value and the driver reached the resample node as
  two separate streams. They are now joined into one `[value, driver]` row
  first, as in Python.
* JavaScript: a `Pipeline` batch feed whose values are an `NdArray`
  (`{ values: { data, shape: [T, w] }, index }`) was read as empty, and a bare
  `NdArray` feed raised "missing feed". Both are now read as `T` rows of `w` columns; a shape that is not
  1-D or 2-D, or an index whose length is not `T`, throws `RangeError`, and
  values that are not an array, typed array or `NdArray` throw `TypeError`.
* JavaScript: a `Pipeline` feed that is a bare typed array other than
  `Float64Array` (`p(new Int32Array(...))`, or `{ a: new Int32Array(...) }`)
  is read as values, as `{ values: Int32Array }` is; it threw a `TypeError`
  saying the values were a `Function`. A feed of the wrong kind (`{ a: 5 }`,
  a `Map`) throws a `TypeError` that names the input and the kind ("the feed
  for input 'a' is a number"), and `BigInt64Array` / `BigUint64Array` values
  throw a `TypeError` that says to convert them, instead of the engine's raw
  BigInt message. A bare feed of another kind (a `DataView`, a `Set`, a
  generator, `{values() {...}}`) throws a `TypeError` naming it instead of
  "missing feed". Error messages use "an" before a vowel ("an Object").
* A reused `Pipeline` or a new `live()` session forgets the plan a `resample`
  node learned from the previous stream's width. Before, a clock resample fed
  2 columns and then 3 dropped a column (3 then 2 padded with `NaN`), and an
  `ohlcv_bars` resample fed 5 columns and then 6 raised for the life of the
  `Pipeline`, in Python and JavaScript.
* A clock resample (`clock=True`) of a multi-column value stream with
  `agg="ohlc"` wrote past its output row; it now gives (open, high, low, close)
  per column (`(T', X, 4)` from the eager call, a flat `(T', 4 * X)` inside a
  `Pipeline`). With `agg="ohlcv_bars"` it crashed; it now builds the bars. A
  fixed-plan agg (`ohlc_bars`, `ohlcv`, `ohlcv2`) given the wrong number of
  columns silently dropped or padded columns; it now raises `ValueError`
  (`RangeError` in JavaScript).
* JavaScript: events of different widths on one input of a live session (a
  number then a sequence, or sequences of different lengths, in either order,
  through `push` and `pushWide`) read past the end of the output buffer or
  returned wrong values. They now raise `RangeError`.
* JavaScript operators and pipelines that are dropped without `dispose()` are
  now freed when they are garbage-collected. Before, every operator kept about
  4 to 10 KB on the JS heap and its C++ state on the WASM heap for the life of
  the program, and every `Pipeline` its compiled graph, so a program that
  builds a new operator per call ran out of memory over a long run. The
  finalizer's held value reached the operator it was meant to free, so the
  operator was never collected. Building and dropping 100,000 operators now
  leaves the JS and WASM heaps where they were.
* Infinite metrics now count in the null distribution of `position_test` /
  `positionTest` and `strategy_test` / `strategyTest`. A randomized run is
  left out only when its metric is NaN, as documented; 2.8.0 also left out
  runs whose metric was `+inf` or `-inf`. This applies to every p-value: the
  per-strategy p-value (one strategy, each of K strategies, and a portfolio),
  the `strategy_test` p-value, and the family p-value with K strategies, where
  a round counts when any strategy has a metric and its best, infinite or
  not, is compared with each observed value. Infinite runs also appear in the
  returned `null` (one strategy, portfolio, `strategy_test`).
  * Which results change: only tests where some randomized run has an
    infinite metric. That is a statistic that can be infinite, such as
    `profit_factor` (no losing trade), `calmar` (no drawdown), or `sortino`
    (no downside), or a custom metric that returns `inf`. Tests whose null
    has no infinite value give the same results as 2.8.0.
  * Direction: a run better than every finite value (`+inf` with larger is
    better, `-inf` with smaller is better) is a hit, so 2.8.0's p-values were
    too small there; a run worse than every value adds a non-hit, so 2.8.0's
    were slightly too large there. When the observed metric itself is not
    finite the p-value stays NaN.
  * Examples, with a custom metric that is `inf` above a threshold: two
    strategies, n = 120, `inf` in 41 rounds: p-values (0.106, 0.030) become
    (0.372, 0.207), and the family p-values (0.15, 0.0125) become (0.438,
    0.347); `strategy_test`, n = 200, 32 infinite runs: 0.0059 becomes 0.164.
* The `nan_policy` of the backtest engines that give a `NaN` target or order a defined meaning is now listed as `nan-aware`, not `ignore`: `BacktestOHLCTarget`, `BacktestOHLCVTarget`, `BacktestTradesTarget`, `BacktestL1Target` (a `NaN` target keeps the last target), and `BacktestOHLCOrders`, `BacktestTradesOrders`, `BacktestL1Orders`, `BacktestL1TradesOrders` (a `NaN` own quote is a market order or an unquoted side). Documentation only; no result changes.
* A list passed to a multi-input operator whose events do not hold every input
  (for example `Add()([v, w])` or `Add()([1.0, 2.0])`) raises a `ValueError`
  (JavaScript: `RangeError`) that explains the list reading and the fix,
  `Add()(v, w)`, in place of `std::bad_cast` or a misleading shape `(1, 30)`.
* A list of N-element lists into an N-input operator (`Add()([[1.0, 2.0], [3.0, 4.0]])`)
  now reads as events, like a list of tuples, in place of `std::bad_cast`.
  JavaScript reads a list of N-value arrays or of `(..., N)` NdArray events the same way.

* JavaScript: after a few thousand caught operator errors (for example
  `RollingMean(0)` in a loop) the WebAssembly stack overran the heap and later
  calls failed with "memory access out of bounds". Each caught error now
  resets the stack.
* JavaScript: a rejected constructor with a string parameter (for example
  `RollingMean(0, "strict")`), a rejected `reshape` layout, and a rejected
  constructor with an array parameter no longer leak WebAssembly memory.
* **JavaScript:** row operators, reducers and row streams check an `NdArray`'s shape against its data before reading it.
* JavaScript: when one of several stream sources throws, the stream no longer
  calls `return()` on that source; the other sources are still closed, as
  `for...of` does.
* JavaScript: an operator copies its array parameters when it is made, so
  changing the array afterwards no longer changes later runs.
* `BacktestPortfolioTarget` and `BacktestPortfolioOHLCTarget` dropped an order
  that fell due on a step where its asset had no bar, so one leg of a pair
  could trade while the other never did. The order now waits for the asset's
  next quoted bar, unless a newer order for that asset is due first.
* `CombineLatest` given a multi-column stream kept only its first column inside
  a `Pipeline` and raised an opaque `std::bad_cast` when called directly. Both
  paths now concatenate its columns.
* A tuple of arrays passed to a one-input operator raises a `TypeError` that
  says what to do: a `(values, index)` stream goes to a stream operator or a
  `Pipeline`, and a list of arrays is a sequence of asset events. In 2.8.0 a
  one-output operator raised `RuntimeError: std::bad_cast`, and a
  multi-output operator raised `TypeError: Iterable must contain numbers.`
* Inputs whose shapes do not match raise `ValueError` that names the problem
  ("do not broadcast", "same length along time"); they raised `TypeError`.
* Passing an object whose class has a non-string `__module__` to an operator
  raised a segmentation fault; it now raises `TypeError`.
* Operator constructors and other calls no longer crash on an object whose
  class has a non-string `__module__` or a module name that is not valid
  UTF-8 (nanobind upgraded to 2.13.0, with a local patch for the UTF-8 case).
* An input that an operator cannot read raises `TypeError` that names the
  operator and the input's type (`RollingMean: a M is not a valid input; ...`)
  on every dispatch path; most paths said only "Unsupported input type."
* JavaScript errors raised in C++ carry their message text, and reducer error
  messages say "assets" instead of "groups".
* A stream stored on the object that feeds it is now freed by the garbage
  collector; before, the cycle leaked for the life of the process. This covers
  the sync and async operator streams, lazy `Pipeline` calls and lazy `Merge`.
* `RollingSkew` and `RollingKurt` (and `RollingVar`, `RollingStd`,
  `RollingZscore`, `RollingSharpe`, `BollingerBands`, `RollingSigmaClip`,
  `RollingCov`, `RollingCorr`, `RollingBeta`, `RollingKyleLambda`,
  `RollingSpread`, `RollingAlpha`, `RollingTLSBeta`, `RollingLinearRegression`,
  `RollingResidualStd`, `ExpandingVar`, `ExpandingStd`, `ExpandingSkew`,
  `ExpandingKurt`, `EwVar`, `EwStd`, `EwSkew`, `EwKurt`, `EwZscore`, `EwCov`,
  `EwCorr`, `EwBeta`, `EwKyleLambda`, `EwSpread`, `EwAlpha`, `EwResidualStd`,
  `EwYangZhangVar` and `EwYangZhangVol`) were inaccurate when the mean is large next to the spread,
  for example on price levels; they now use a numerically stable update. On a
  series near 1000 with unit moves, `RollingKurt` was off by up to 8e-2 and
  `RollingVar` by 4e-9 of their values; at 1e6 the raw sums had no correct
  digits for skewness and kurtosis. Each operator now matches its estimator,
  evaluated in exact arithmetic, to within 3e-13 of the statistic's scale at
  levels up to 1e6, also right after a spike of 1e4 or a level jump of 1e5
  standard deviations (most errors are below 1e-14; see "Accuracy at large
  levels" in the conventions page). On data near zero the results change only
  in the last bits (below 1e-13).
* On constant input these operators now return the exact result where they
  returned rounding noise or infinity: `RollingKurt` and `ExpandingKurt` give
  `NaN` for a constant window (they gave `inf` or `-inf`); `EwSkew`,
  `EwKurt`, `EwZscore`, `EwCorr`, `EwBeta`, `EwSpread`, `EwAlpha` and
  `EwResidualStd` give `NaN` while the weighted spread is exactly zero (they
  gave values such as -1.6e10, -1e17, 0 or 1); `EwVar` and `EwCov` give
  exactly 0 (they gave values of about 1e-3); `EwStd` gives 0 where it gave
  `NaN`.
* `EwZscore` returned 0 instead of `NaN` for its first sample on some
  platforms, as rounding decided; it is `NaN`, as documented.
* `RollingLinearRegression(start_policy="zero")` treated the window as empty
  rather than zero-filled until its first `reset()`; it now starts zero-filled.
* `RollingOU` fitted its AR(1) regression with mismatched counts: it divided
  the intercept by `window_size` instead of the `window_size - 1` pairs in the
  window, and mixed both counts in the noise variance. Its `"mean"` was
  pulled toward zero by `mean / window_size` (adding a level L to the input
  moved it by L - L / window_size), `"relmean"` with it, and `"std"` grew with
  the level (142 instead of about 1 at a level of 1000). With
  `start_policy="expanding"`, until the window was full, the first value was
  also counted as a target of the fit, which moved every output. The fit is
  now the documented least-squares fit over the pairs: `"mean"` moves by
  exactly the level, `"mrr"`, `"relmean"` and `"std"` do not move, and
  `"std"` is the residual standard deviation with `window_size - 3` degrees
  of freedom. `"std"` therefore needs `window_size >= 4`: with
  `window_size=3` it is now `NaN` at every step (2.8.0 gave finite values).
  With `start_policy="expanding"`, `"std"` now starts at the fourth value
  (three pairs), where it started at the third, or later where the old
  formula took the square root of a negative number. Its sums are taken about
  an origin near the data. Against the fit evaluated in exact arithmetic,
  `"mrr"`, `"relmean"` and `"std"` are within 5e-13 of their scale at levels
  up to 1e6; `"mean"`, a value of the size of the level, is within 5e-13 plus
  the rounding of the result itself (half a unit in the last place of the
  level, 6e-11 at 1e6). On data near zero, `"mrr"` changes only in the last
  bits for the strict and zero policies. There is no switch back: the old
  values were wrong.
* `RollingPoly1`, `RollingPoly2`, `RollingTSF`, `ExpandingSlope` and
  `RollingADF` were inaccurate when the level is large next to the spread,
  the same raw-sum problem as the moment operators above, and the rolling fits
  also drifted with the length of the series, since their sums were never
  recomputed. Against the fit in exact arithmetic, window 50, unit noise at a
  level of 1e6, the error was up to 5e-7 of the spread for `RollingPoly2`,
  1e-7 for `RollingTSF`, 1e-8 for `RollingPoly1` and 3e-9 for
  `ExpandingSlope`, and `RollingADF` returned `NaN` (its regression looked
  singular). They now keep their sums about an origin near the data and fit
  about the mean time. Slopes, curvatures and the ADF statistic match the
  exact fit to within 2e-13 of their scale at levels up to 1e6, also after a
  spike or a level jump (`RollingADF` once the jump has left its window). A
  fitted value (the `RollingPoly1` and `RollingPoly2` endpoint, the
  `RollingTSF` forecast) is within the same 2e-13 of the window's standard
  deviation plus the rounding of the result itself, half a unit in the last
  place of the level (6e-11 at 1e6, which is 2e-10 of the standard deviation
  of a window of 4 unit-noise values). On data near zero the results change
  in the last bits: at most 2e-11 relative for `RollingPoly2` on 5000 values
  (its old sums had drifted), 4e-13 or less for the others. `RollingTSF`
  still matches `talib.TSF` to rounding.
* `RollingPoly1`, `RollingPoly2`, `RollingTSF` and `RollingOU` returned `NaN`
  forever after an infinity entered the window (their running sums became
  inf - inf). They now return finite values again once the infinity has left
  the window, as the moment operators do.
* The standard error of `RollingLinearRegression` was 0 at the steps where
  an infinity in the window made the residual `NaN`, while the slope and the
  other outputs were `NaN`. It is now `NaN` there too. A residual that rounds
  to zero or below still gives 0.

* A `CombineLatest` of inputs whose widths are known has their sum as its
  width, a known width of 0 included, so the width check before any event
  sees it: `forecast_pairs(CombineLatest()(a, b), y, ...)` of two `(T, 0)`
  feeds raises "forecast_pairs: X has 0 columns, expected at least 1", where
  it returned an empty result, and an operator fed such a join raises
  `ValueError` before any event. A `forecast_pairs` count lag that runs out of
  memory on an event no longer emits that event's warm-up row first.
* The eager `Resample` of an empty wide stream, `(0, 3)` or `(0, 3, 2)`,
  raised "cannot reshape array of size 0"; it returns `(0, 3)` (`(0, 3, 4)`
  for `ohlc`).

### Performance

* A `Pipeline` that joins inputs runs faster, in Python and JavaScript.
  `CombineLatest` emits its settled row without copying it, a batch run of up
  to 8 inputs merges them by a scan of their next rows instead of a heap, and
  an output takes a one-column row without a range insert. The results are
  unchanged. Python, 10^6 rows per input (ms, against this branch before the
  change and 2.8.0): `Sub()(CombineLatest()(a, b))` 46.5 (was 57.2; 2.8.0
  52.9), `forecast_pairs(X, y, count=5)` 34.9 (was 43.3; 40.2), a one-input
  `Lag(1)` graph 5.5 (was 6.9; 10.1), a clock `Resample` 19.3 (was 22.2;
  33.9). JavaScript, the two-input graph: 111 ms (was 135).
* An operator reads a float64 C-contiguous array argument in place instead
  of copying it, and a packed `(T, ..., N)` argument costs what N separate
  arrays cost, in every multi-input operator. The fields of a packed array
  are read in place too; a tile of a `(T, K, N)` array gathers the N fields
  of a row in one pass; `WilliamsR` reads strided fields in place and takes
  the rolling high and low in one pass; `ContOFI` keeps its state in
  registers. The results are unchanged. Python, 200,000 values per field (ms,
  separate / packed, was): `Sub()` 0.057 / 0.057 (0.112 / 0.112),
  `WilliamsR(20)` 0.58 / 0.58 (0.77 / 0.95), `WilliamsR(20)` on
  `(2000, 100)` fields 0.82 / 0.83 (1.02 / 1.08), `Where()` on `(2000, 100)`
  0.33 / 0.35 (0.39 / 0.47), `ContOFI()` 0.33 / 0.33 (1.35 / 1.39). Over
  the 94 operator and shape pairs the separate call takes 0.90 of its time
  (geometric mean) and the packed call 0.87, and the packed call is within
  1.05x of the separate one (was up to 1.23x; 2.8.0 up to 4x), `VPIN` aside,
  which runs at one of two speeds per process in either spelling.
* JavaScript calls the per-event steps of a live `Pipeline` and of the
  `combineLatest` iterators as plain WASM exports instead of Embind methods,
  and checks the keys of a `{ value, index }` event without building a
  message or an array. The steps are the same C++. Node 22, 10^5 events per
  input (ms): the indexed iterator join of two sources 33 (was 86), `push`
  of a number 4.3 (was 16.8), `push` of a two-value event 22 (was 39), the
  positional iterator join of array events 22 (was 31).
* The layout join of `combine_latest` (asset broadcast, field concatenation,
  row assembly and the positional pairing of iterators) is C++, shared with
  JavaScript. The positional iterator form is about 2.9x faster than 2.8.0
  (2 sources of 1e5 floats: 47 ms to 16 ms); with a layout, 6x to 13x faster
  than the Python join it replaces.
* `split` reads, checks and counts the sources in one pass, reads a uint32
  source array (the output of `merge`) in place, and no longer builds and
  drops a row-number index in a positional call. On 1e6 events it is 5-20%
  faster than 2.8.0 (2.3x for a positional call with 1000 sources), and 4x
  faster on a list of NumPy integers.
* The string check before a float64 conversion of a list (shared by every
  operator and stream function) reads a list of NumPy scalars about 25x
  faster: it checks the first item of each NumPy scalar type and compares the
  type of the rest.
* A `Pipeline` with several aligned outputs aligns them in C++, for a batch
  call, a lazy call and `live().result()`. Python, two outputs of 10^6 rows:
  a batch call takes 15 ms (was 39 ms), a lazy call of 10^5 generator events
  38 ms (was 49 ms), and `live().result()` of 10^5 rows 0.05 ms (was 2.4 ms).
* A batch call of a one-input `Pipeline` feeds its events in order without the
  merge step: `RollingMean(5)` over 10^6 rows takes 7.9 ms in Python (was
  10.2 ms) and 22 ms in JavaScript (was 35-36 ms).
* The Python async iterators no longer copy each event's items into a buffer;
  the operator reads the awaited items in place, about 0.04 to 0.06 us less
  per event.
* `position_test`, `strategy_test` and `walk_forward` no longer copy prices to
  K columns; JavaScript `positionTest` and `walkForward` run the engine once
  per chunk.
* nanobind 2.13.0 lowers the cost of a single-event call: `RollingMean(20)(1.0)`
  takes about 0.065 µs (was 0.077 µs) and `RollingCorr(20)(1.0, 2.0)` about
  1.0 µs (was 1.6 µs).
* A scalar call to a multi-input operator is about 11x faster:
  `RollingCorr(20)(1.0, 2.0)` takes about 0.09 µs (was 1.0 µs), close to the
  0.06 µs of `RollingMean(20)(1.0)`. Exact Python floats and ints and NumPy
  number scalars (`np.float64`, `np.float32`, `np.int64` and the other float
  and integer types) call the kernel directly; any other argument takes the
  full dispatch as before. A scalar call to a one-input, multi-output
  operator such as `BollingerBands(20)(1.0)` drops from 0.65 µs to 0.10 µs.
* A batch call with N arrays (`RollingCorr(20)(x, y)`) no longer reads each
  array's `ndim` to look for a mixed list, about 0.5 µs less per call.
* JavaScript batch calls run each asset column in one C++ loop, as Python
  does, instead of one virtual call per element. On 1M elements (ns per
  element, Apple M-series): `RollingMean(20)` 6.6 to 1.6-2.3, `RollingCorr(20)`
  14.7 to 5.9, `Add()` 10.1 to 0.8. Multi-input arrays are no longer
  interleaved in JavaScript, and the broadcast is planned once per call.
* Python and JavaScript share one batch column loop, which calls the
  operator's per-event step directly. On 1M elements `Add()` takes 0.6 ns per
  element (was 1.4). With clang (the macOS wheels, measured on Apple M) the
  step is also inlined into the loop: `RollingCorr(20)` 3.9 (was 5.3),
  `RollingBeta(20)` 3.7 (was 4.3). GCC and MSVC builds keep their compilers'
  own inlining.
* Long columns of a `(T, K)` array run in tiles of 8 adjacent assets copied to
  contiguous memory, so the column loop no longer stalls on strided reads. The
  gain is largest at power-of-two K: `Add()` on `(2000, 256)` takes 1.4 ns per
  element (was 5.0), `RollingCorr(20)` on `(20000, 128)` 6.8 (was 15.9).
* More operators keep their running state in registers in a batch call.
  Before, the state was stored and reloaded every step, and the speed of that
  depended on the process (some runs were 2 to 4 times slower than others).
  The results are unchanged. On 200,000 values (ns per element, Apple M3):
  * IIR filters: `Butter()` 2.7 (was 4.9 to 6.5), `ButterBandpass()` 2.4 (was
    5.3 to 6.6), `SuperSmoother(20)` 2.7 (was 3.9 to 6.5), `RoofingFilter(48,
    10)` 2.7 (was 2.8 to 5.0); also `ButterHighpass`, `ButterBandstop` and
    `Decycler`.
  * `CumSum` and `ExpandingSum` 0.66 (was 0.8 to 3.3), `ExpandingMean` 0.56
    (was 3.3), `CumProd` and `ExpandingProd` 1.1 (was 1.1 to 3.3), `Diff2`
    0.54 (was 1.4 to 2.6).
  * The two-input EW operators: `EwCov` 3.0 (was 4.9), `EwBeta` 3.1 (was 4.7),
    `EwCorr` 3.7 (was 4.8), `EwSpread` 2.7 (was 5.0), `EwAlpha` 3.6 (was 5.9),
    `EwResidualStd` 4.7 (was 5.8); `ADOSC` 1.8 (was 5.1).
  * The DMI and ATR family: `ADX` 5.8 (was 15), `ADXR` 5.9 (was 19), `PlusDI`
    and `MinusDI` 5.8 (was 7.7), `DX` 5.7 (was 7.8), `PlusDM` and `MinusDM`
    5.6 (was 11 to 13), `ATR` 4.2 (was 7.2 to 7.9), `NATR` 4.2 (was 5.7 to
    7.6).
* `EwMean` on a `(T, K)` array runs each column in its batch loop, 1.5 ns per
  element on `(2000, 100)` (was 4.3).
* The moment operators made stable (see Fixed) run their batch loop on a local
  copy of their state, which stays in registers, but the stable update and its
  re-centering tests cost time in the rolling and two-input operators. On
  200,000 values (ns per element on a 1-D series, Apple M3, `window_size=20`
  or `span=20`, against the build before this change; for the slower ones
  the ratio is the larger of the 1-D and `(2000, 100)` shapes): faster are
  `ExpandingVar` 1.2 (was 4.4), `ExpandingStd` 1.4 (was 4.3), `EwVar` 2.8
  (was 4.9), `EwStd` 2.8 (was 5.1), `EwZscore` 2.9 (was 5.5), `EwSkew` 3.9 (was 5.5), `EwKurt` 4.5 (was 5.6), `ExpandingSkew`
  3.1 (was 4.3), `ExpandingKurt` 3.8 (was 4.6), `RollingSharpe` 3.2 (was
  4.1), `EwCorr` 3.2 (was 3.7) and `RollingVar` 3.2 (was 3.3). Slower by more
  than 1.05x in either shape are `RollingCorr` 5.5 (was 3.8, 1.43x),
  `RollingLinearRegression` 6.8 (was 4.8, 1.40x), `RollingSpread` 5.0 (was
  3.6, 1.40x), `RollingTLSBeta` 5.8 (was 4.2, 1.40x), `EwResidualStd` 6.6
  (was 4.7, 1.40x), `RollingBeta` 5.0 (was 3.7, 1.36x), `RollingSkew` 4.6
  (was 3.5, 1.33x), `RollingCov` 4.8 (was 3.6, 1.32x), `EwSpread` 3.5 (was
  2.7, 1.30x), `RollingResidualStd` 9.0 (was 7.0, 1.29x), `RollingSigmaClip`
  5.7 (was 4.5, 1.26x), `BollingerBands` 4.2 (was 3.5, 1.24x), `RollingKurt`
  5.0 (was 4.1, 1.22x), `EwAlpha` 4.4 (was 3.6, 1.22x), `RollingZscore` 3.7
  (was 3.4, 1.16x), `EwYangZhangVar` 23.4 (was 20.5, 1.14x), `EwBeta` 3.4
  (was 3.0, 1.13x), `EwCov` 3.3 (was 3.0, 1.10x), `RollingAlpha` 6.5 (was
  6.0, 1.09x) and `RollingStd` 3.4 (was 3.3, 1.06x). The rolling operators
  recompute their sums about once per window; in streaming, that step takes
  about 8 microseconds with `window_size=10000` (23 after a spike).
* `RealizedSpread` takes 1.35 ns per element (was 2.5).
* The regressions made stable (see Fixed) run a batch loop on a local copy of
  their state. On 200,000 values (ns per element, Apple M3, default
  parameters, against the build before this change; the ratio is the larger
  of the 1-D and `(2000, 100)` shapes): `ExpandingSlope` 1.7 (was 4.4),
  `RollingPoly2` 4.3 (was 5.4) and `RollingADF` 81 (was 87) are faster;
  `RollingOU` 4.7 (was 4.2, 1.13x), `RollingPoly1` 3.5 (was 3.3, 1.10x) and
  `RollingTSF` 3.6 (was 3.3, 1.07x) are slower.

### Documentation

* The Polymorphic API page states the six rules every operator follows and
  the shape contract (time, assets, fields), a new `reshape` page documents
  layouts, and the streams guide covers multi-column `CombineLatest` and
  `Resample`. The JavaScript page gives the same rules in JavaScript and lists
  the remaining differences between the packages.
* A new Streams in depth page describes live calls, stream state, laziness,
  several and async sources, asset and packed events, errors, and row
  operators on streams.
* Function pages show a generated header from their frontmatter: Python and
  JavaScript signatures, a parameter table, output columns, See also, and
  references. Titles and tags are indexed, so a search for "ema", "z-score",
  or "p-value" finds the function.
* New function groups and topics: "Portfolio & pairs", "Signals & position
  rules", "Significance & overfitting", "Bars & resampling", and "Volatility
  indicators". The homepage group grid is generated from the same source.
* New recipes: your first strategy, stops and exits, volatility-targeted
  sizing, funding costs on a perpetual, a parameter grid and its robustness,
  walk-forward evaluation, a cross-sectional momentum book, and from research
  to a live loop. JavaScript backtesting and significance tests in the
  JavaScript guide.
* The portfolios and pairs guide runs on real daily data. The pairs recipe is
  the single pairs example and fills at the next open. Notebook 14 no longer
  fills at the close its signal was computed from, and notebook 18 compares
  hedge ratio estimators.
* The sidebar is organized as Get started, User Guide, Examples, Concepts, API
  reference, and Release notes.
* The Ehlers pages are corrected: `CyclePhase` decreases with time (it said
  it advances); the `MAMA` page and its parameters describe the adaptation
  the right way round (the smoothing constant reaches `fast_limit` when the
  phase stalls or wraps); `DominantCycle` states its `[6, 50]` period range
  and that a trending series should go through `RoofingFilter` first; the
  cycle pages say which in-phase and quadrature pair each one uses instead of
  "the same I and Q"; `Decycler` defines its cutoff as the half-power point;
  `InstantaneousTrendline` cites *Rocket Science for Traders*, the source of
  the routine it now implements.

2.8.0 - 2026-10-10
------------------

### Added

* **Cross-sectional operators:** `CrossRank`, `CrossZscore`, `CrossDemean`,
  `CrossWinsorize`, `CrossNormalize`, and `CrossTopBottom` transform each row
  of a `(T, K)` array across its `K` assets, ignoring assets that are NaN on
  that row. `CrossNormalize` scales scores to portfolio weights with a gross
  exposure and an optional net exposure (`net=0` is dollar-neutral). They run
  on a `(T, K)` array, one row, or a stream of rows, in Python and
  JavaScript/WASM.
* **Portfolio backtests:** `BacktestPortfolioTarget` (prices) and
  `BacktestPortfolioOHLCTarget` (OHLC bars, with `fill_at` and `delay`)
  backtest `K` assets from one shared capital account. Targets are weights of
  portfolio equity, notionals, or units; a finite target is an order and NaN
  holds, so a rebalancing schedule is a target array with weights on the
  rebalance steps. `rebalance_band` trades drifted assets back (`rebalance=
  "drifted"` or `"all"`), `max_gross` and `max_net` cap the book's exposure,
  and fees, slippage, tick sizes, and contract multipliers can differ per
  asset. Assets that are not quoted on a step hold their position. The output
  is `(T, K, 4)` per asset, ready for `PortfolioReport`.
* **Pairs trading:** `KalmanRegression` estimates a time-varying intercept
  and hedge ratio with a Kalman filter and returns the spread and its
  predicted deviation (a ready z-score); `RollingTLSBeta` is a total least
  squares hedge ratio that does not depend on which leg is dependent;
  `RollingADF` is a rolling augmented Dickey-Fuller statistic for testing
  whether a spread still mean-reverts; `BandSignal` turns a z-score into a
  short/flat/long position with entry, exit, and stop levels and a lockout
  after a stop.
* **Significance tests on portfolios:** `position_test` with a portfolio
  engine shifts all `K` assets of the book together and scores the whole
  book, and `strategy_test` accepts strategies that return a `(T, K, 4)`
  portfolio output. Book statistics sum the assets, with turnover counted per
  asset. Python and JavaScript give the same results.
* **Docs:** a "Portfolios and pairs" guide
  (weights from scores, rebalancing rules, a cross-sectional book, a
  Kalman-filter pairs trade with a cointegration gate, and significance tests
  on a book), and the pairs-trade recipe rewritten on real BTC/ETH data with
  the new operators, fees, and a significance test.

### Fixed

* Docs: pages with several plotly examples in separate blocks wrote every
  plot to the same file, so each example showed the first plot.

* **Every statistic of every randomized run:** `position_test` and
  `strategy_test` (and `positionTest`/`strategyTest` in JavaScript) return
  `observed_stats` and `null_stats`: all `backtest_report` summary statistics
  of the real run and of each randomized run, in run order with NaN kept, for
  plotting distributions and scatter plots of pairs of statistics. `metric`
  accepts any of those statistics (for example `"sortino"` or `"win_rate"`),
  and `periods_per_year` annualizes the ratios. Statistics with no natural
  direction, such as `"turnover"`, need `larger_is_better`.

2.7.0 - 2026-10-10
------------------

### Added

* **Strategy significance tests:** `strategy_test` runs a whole strategy (a
  function of screamer operators, a `Pipeline`, or any callable returning a
  backtest engine output) on the real data and on randomized copies, and
  returns the null distribution and a p-value. `method="bars"` permutes OHLC
  bars (Masters' method, `permute_bars`), `"returns"` permutes a price
  series (`permute_returns`), and `"shift"` rotates chosen feature inputs to
  test whether they carry information. `mode` shuffles log returns (`"log"`),
  differences (`"diff"`, for prices that can be zero or negative), or the
  values themselves (`"level"`), and `volume_mode` moves a volume input with
  its bar unchanged (`"keep"`) or rescaled to keep the traded amount
  (`"notional"`). Because the strategy runs
  again on each copy, fitting and tuning on the data are part of the test.
  Python and JavaScript/WASM (`strategyTest`, `permuteBars`,
  `permuteReturns`) give the same randomized inputs for the same seed.
* **Testing several strategies:** `position_test` accepts a `(T, K)` target,
  shifts all `K` variants jointly, and adds `family_p_value`, which compares
  each variant with the best of the `K` randomized results per round (White's
  Reality Check max statistic). It corrects for reporting the best of several
  tried variants.
* **Probabilistic and deflated Sharpe ratios:** `probabilistic_sharpe` gives
  the probability that a strategy's true Sharpe ratio is above a benchmark,
  adjusted for sample length, skewness, and kurtosis; `deflated_sharpe` raises
  the benchmark to the Sharpe ratio the best of the tried variants would reach
  by luck (Bailey and Lopez de Prado). Analytic, in C++, Python and JavaScript.

* **Backtest risk exits:** the target engines (`BacktestOHLCTarget`,
  `BacktestPriceTarget`, `BacktestTradesTarget`, `BacktestL1Target`) take
  `stop_loss`, `take_profit`, and `trailing_stop`, as fractions of the average
  entry fill. On OHLC bars a gap exits at the open and an intrabar touch at the
  level; when both levels fall inside one bar the stop is taken. Exits are
  market orders (taker fee and slippage), and same-side targets are ignored
  after an exit until the target turns flat or reverses.
* **Backtest statistics:** `BacktestStats` adds running exposure, round-trip
  count, win rate, profit factor, average win and loss, average holding period,
  Sortino, and Calmar. `BacktestReport`, `BacktestStats`, and `backtest_report`
  take `periods_per_year` to annualize the ratios (the default 1 keeps the
  per-step values), and `backtest_report`'s summary includes the new statistics.
* **Backtest sizing:** the target engines take `size="units" | "notional" |
  "weight"` with `initial_capital`: a notional amount or a fraction of equity
  (which compounds) is converted to units at the execution price when the
  target value changes, and `rebalance_band` trades back to the target when the
  holding drifts by more than that fraction of equity.
* **Volume limits:** `BacktestOHLCVTarget` takes the bar's volume and fills an
  order at most `max_participation` of it per bar, carrying the rest to later
  bars; `BacktestTradesTarget` gains `max_participation` of each print's size.
  The target engines now keep working toward the last finite target, so a
  `NaN` target continues a partly filled order (with complete fills this is the
  same as holding).
* **Backtest carry:** `BacktestCarry` adds perpetual funding, borrow, or
  financing to any engine's output from a per-step rate series: the position
  held into each step pays `position * price * rate`, moved from pnl into cost.

### Documentation

* A "Backtesting best practices" guide: look-ahead, realistic fills, costs and
  carry, timing robustness, luck, and selection, each with the screamer tool
  that addresses it and runnable examples.
* A capability overview, a gallery of function plots, eight recipes on real
  data, plots in the foundation notebooks, and a JavaScript page.

2.6.0 - 2026-10-10
------------------

### Added

* **Backtest significance tests:** `position_test` asks whether a backtest's
  timing beats randomly timed copies of the same strategy and returns the null
  distribution and a p-value. The copies come from `circular_shift` (rotate the
  target series) or `segment_shuffle` (shuffle its trades and flat stretches),
  which keep the trading profile and break only the link to the market. The
  generators run in C++ with a portable seeded generator, so a seed gives the
  same result in Python and JavaScript/WASM (`positionTest`, `circularShift`,
  `segmentShuffle`). These are batch-only validation tools, not streaming
  operators.

* **Backtest slippage:** every backtest engine takes `slippage` and
  `slippage_unit`. One number is read in one unit: `"fraction"` of the
  reference price (5 bps = `5e-4`), `"ticks"` of `tick_size`, or `"price"`.
  Slippage moves taker fills against the trader; resting limit fills never
  slip. `BacktestPriceTarget` also gains `tick_size`.
* **Backtest execution timing:** `BacktestOHLCTarget` takes `fill_at`
  (`"open"`, `"close"`, `"typical"`, `"ohlc4"`) to choose the execution price
  within the bar, and `delay` to choose how many bars after the decision it
  executes. `delay=0` with `fill_at="close"` is market-on-close of the signal
  bar.

### Fixed

* **Read-only arrays:** every op raised `std::bad_cast` on a read-only numpy
  array. pandas returns read-only arrays from `Series.to_numpy()` under
  copy-on-write, so `RollingMean(3)(df["price"].to_numpy())` failed. Inputs
  are now accepted as read-only views, with no copy.
* **Backtest cost on OHLC engines:** the `cost` column of `BacktestOHLCTarget`
  and `BacktestOHLCOrders` counted the move from the fill price to the bar's
  close as cost. It now holds only the trade's friction: slippage plus fees,
  measured against the fill's reference price (the bar's `fill_at` price for a
  market fill, the limit price for a resting fill). Equity and pnl are
  unchanged; `cum_cost` and `total_cost` in `backtest_report` change.

### Changed

* **Backtest `tick_size`:** `tick_size` now means only the instrument's price
  grid. On `BacktestOHLCTarget` and `BacktestOHLCOrders` it no longer adds a
  one-tick slippage to market fills. To keep the old fills, pass
  `slippage=1, slippage_unit="ticks"` with the same `tick_size`. On the L1
  engines, a marketable order still walks one tick past the displayed size.

### Documentation

* Seeded plot examples for the twelve Ehlers cycle and filter pages that had
  none, a References page that shows the concept guides as cards, and new
  sections in "Choosing an operator" for cycles and regimes, order flow and
  liquidity, risk and performance, data cleaning, bars from ticks, and backtest
  significance. The engine guide gains a section on execution price, slippage,
  and cost.
* `SuperSmoother`, `Decycler`, and `RoofingFilter` docs now state that the
  period (or cutoff) is required; the value they list is a suggested starting
  point, not a constructor default.
* Fixed: the install docs named Python 3.11 and pybind11 (now 3.12 and
  nanobind), and `examples/lag_example.py` used a removed API.

2.5.0 - 2026-08-11
------------------

### Added

* **Volatility (EW):** `EwYangZhangVar` and `EwYangZhangVol`, the
  exponentially-weighted Yang-Zhang range-based volatility estimator. They
  complete the EW range-based family: Parkinson, Garman-Klass, and
  Rogers-Satchell already had EW forms, but Yang-Zhang did not. The estimator
  composes two `EwVar` (overnight and open-to-close) and one `EwMean`
  (Rogers-Satchell per bar), and it uses the effective sample size for the k
  weight. Both Python and JavaScript/WASM include them, verified against a
  pandas transcription and the OHLC simulation test that recovers a known sigma.

2.4.0 - 2026-08-10
------------------

### Added

* **Regression family (EW):** `EwSpread`, `EwAlpha`, and `EwResidualStd`, the
  exponentially-weighted analogs of `RollingSpread` / `RollingAlpha` /
  `RollingResidualStd`. They complete the rolling/EW symmetry of the
  pairs-trading and regression operators. Each one composes the existing EW
  building blocks (`EwBeta`, `EwMean`, `EwStd`). Both Python and JavaScript/WASM
  include them, verified against independent pandas baselines.

2.3.1 - 2026-08-09
------------------

### Changed

* **JavaScript:** an `op(array)` call now processes the whole array in one C++
  pass (`evalBatchInto`) instead of one JS/WASM boundary crossing per element.
  This is about 65x faster (EWMA goes from ~2.4 to ~180 million updates/s in
  Node). The streaming (`op(x)`) path and every result are unchanged.

### Added

* **JavaScript demo:** the dashboard warm-starts from the recent public trade
  tape over REST, so the chart and every indicator fill immediately instead of
  building up over a minute.

* `js/benchmarks/throughput.mjs` (`npm run bench`): a reproducible batch and
  streaming throughput benchmark that self-checks batch results against the
  streaming path before it reports speed.

2.3.0 - 2026-08-06
------------------

### Added

* **JavaScript demo:** the live-trades page becomes a multi-window market
  dashboard. It adds a VPIN signal view, a volume-clock view, a responsive
  indicator picker with about twenty indicators, and time-aware buffered replay.
  Contributions by Édouard Tallent (@tagomatech).

2.2.1 - 2026-08-04
------------------

### Fixed

* **JavaScript:** the WASM module now runs in the browser. The 2.1.0 and 2.2.0
  builds compiled the emscripten module for Node only, so the glue called
  `require`/`process` and threw in any browser. The module now builds for
  `web`, `worker`, and `node`. Verified in a real browser.

### Added

* **JavaScript:** a "Node, bundlers, and the browser" guide, and a live-trades
  demo page on the documentation site.

2.2.0 - 2026-08-03
------------------

### Added

* **JavaScript:** `PortfolioReport` is now part of the JavaScript/WASM binding,
  closing the one gap between the Python and JavaScript operator sets. It needed
  a second marshalling path: every other operator reads a fixed number of values
  per event, while a reducer folds a variable number of groups (the assets in a
  portfolio) into one event and reads that count from the data. The WASM layer
  gained `reduceInto` / `reduceBatchInto` alongside the existing fixed-width
  `evalInto`, the manifest gained a `dynamic_reducer` category that drives both
  generators, and the codegen freshness gate no longer excludes the operator, so
  it now enforces the parity it previously allowed to lapse. A JavaScript event
  is a `(groups, 4)` block and a batch is `(events, groups, 4)`, matching the
  Python shapes; results are held to a Python fixture covering both the batch
  and the per-event path.

* Generic backtest contract sizing and fees, plus the `PortfolioReport`
  operator. Contributed by Édouard Tallent (@tagomatech).

* Python and JavaScript are now required to expose the same operator set. A
  capability in one binding but not the other is a defect, not a trade. The
  rule is written into `CONTRIBUTING.md`.

* `make patch` / `minor` / `major` now also create a `js-v<version>` tag, so one
  command releases the Python and the JavaScript package together.

### Fixed

* The WASM smoke harness constructed four backtest engines with stale
  hand-written arguments and failed on all of them once those engines gained
  contract-sizing parameters. The arguments now match, and a mismatch reports
  which operator to update instead of an Embind arity error.

2.1.0 - 2026-08-03
------------------

### Added

* **JavaScript / WebAssembly build.** screamer now compiles the same C++ core
  to WebAssembly and runs in Node and the browser, published to npm as
  `@screamer-labs/screamer`. Highlights:

  * A uniform Embind runtime over `EvalOp` (`evalInto` / `evalBatchInto` /
    `reset` / heap helpers), and generators that read the nanobind bindings to
    emit both the Embind layer and typed TypeScript factories for all 226
    operators. Freshness gates keep the generated files in sync with the
    bindings.
  * One polymorphic op wrapper with four call regimes (scalar, typed array,
    sync iterable, async iterable), explicit lifetime management, and an
    ndarray-lite shape.
  * A symbolic `Input` / `Pipeline` graph (define-then-bind) with combinators
    (`combineLatest`, `resample`, `select`, and friends), backed by an Embind
    DAG layer. Pipeline results are checked against a Python oracle, and batch
    equals streaming on the same data.
  * A self-contained single-file package (embedded WASM, ESM + CJS), a
    pack-and-install smoke test, tag-gated npm publishing over OIDC, and a
    TypeDoc reference plus a hand-written guide on GitHub Pages.

2.0.0 - 2026-08-02
------------------

### Changed

* **Build:** the C++ extension is now bound with
  [nanobind](https://github.com/wjakob/nanobind) instead of pybind11. nanobind
  is fetched by CMake at configure time, so it is no longer a pip build
  dependency. Every operator's observable behavior is unchanged; the full test
  suite is the oracle for that.

* **Packaging:** releases now ship a single stable-ABI (`cp312-abi3`) wheel per
  platform rather than one wheel per Python version. The wheel is built on
  CPython 3.12 and installs on 3.12 and every later 3.x.

* **Internal:** `ScreamerBase` and `FunctorBase` are split into a pure
  compute base and a relocated nanobind dispatch layer (`common/dispatch.h`
  / `common/dispatch.cpp`). Operator headers now depend only on the pure
  base, so they compile with no binding library on the include path; a
  `check-pure-headers` CMake target and `devtools/check_pure_headers.sh`
  enforce this as a standing gate. This is groundwork for a future JS/WASM
  binding and does not change any Python-visible behavior.

* `EwMean`'s array path holds its accumulators in locals for the duration of
  the loop, 4.0 to 1.0 ns/sample on 1M samples. It was the one operator where
  an alternative was meaningfully ahead in the batch comparison, and it now
  leads its row: TA-Lib's `EMA` is 1.74 ns and pandas 4.20. Streaming and every
  result are unchanged.

  Same cause as `RollingMean`: the accumulators live behind `this`, and the
  compiler cannot prove the caller's output buffer does not alias the operator,
  so in a batch loop it reloads them every sample.

* `RollingMean`'s array path hoists the recurrence state into locals for the
  duration of the loop, 3.00 to 0.87 ns/sample on 1M samples. Streaming and
  every result are unchanged.

  The per-sample path keeps the running sum, buffer index and count behind
  `this`. In a batch loop the compiler cannot prove the caller's output buffer
  does not alias the operator, so it reloads all of it every iteration. Loading
  it once, running the loop, and writing it back is worth 3.4x here, and it
  takes the operator past an equivalent hand-written numba loop, which measures
  1.42 ns/sample.

  The same shape applies to every stateful operator's array path, so this is
  one instance of a general opportunity rather than a one-off.

* `WilliamsR` computes its array result by block decomposition, 19.9 to 3.4
  ns/sample on 1M samples, which puts it just ahead of TA-Lib's `WILLR` at 3.8.
  Streaming is unchanged, and the array result is asserted identical to the
  per-sample one.

  `FunctorBase` gains an optional `process_columns` hook for this, the
  multi-input counterpart of the `process_array_no_stride` override that
  `ScreamerBase` already had. It defaults to declining, so every other
  operator is untouched, and an override is expected to decline rather than
  change a result when it cannot apply: here, non-contiguous input or a `NaN`
  in the bars, which under `ignore` skips a bar and breaks the fixed block
  structure.

* `RollingMax`, `RollingMin`, `RollingRange`, `RollingArgmax` and
  `RollingArgmin` compute their array results by
  block decomposition instead of by replaying the streaming recurrence, which
  is 5 to 7 times faster and no longer depends on the shape of the data:
  `RollingMax(50)` on 1M random samples goes from 8.6 to 1.2 ns/sample, and
  from 10.6 to 1.2 on a random walk; `RollingArgmax` goes from 8.9 to 2.1.
  Streaming is untouched.

  The argmin/argmax form carries indices through the same decomposition and
  has to break ties the way the deque does, newest index first, or the two
  paths disagree on flat stretches. That is asserted against the scalar path
  on integer input, where ties are frequent.

  The deque's real cost on random input is not its comparisons but its pop
  loop, whose trip count depends on the data, so the loop-exit branch
  mispredicts on most samples. Splitting the array into blocks of `window` and
  taking a running extremum forward and backward through each gives the same
  answers with two comparisons per element and no data-dependent branch. It
  needs up to `window` samples of lookahead, so it can only be the array path;
  the event path keeps the deque. The array path falls back to the streaming
  recurrence when the operator already holds state or the input contains
  `NaN`, so results are identical either way.

* The sliding-window extremum used by `RollingMin`, `RollingMax`,
  `RollingRange`, `RollingArgmin`, `RollingArgmax`, `WilliamsR`, `Stoch`,
  `StochRSI` and `DonchianChannels` now stores its candidates in a fixed ring
  buffer instead of a `std::deque`. Nothing is allocated after construction,
  which matters more for streaming latency than for throughput: it is about 12%
  faster on the array path (`RollingMax(50)`, 9.9 to 8.7 ns/sample) and removes
  the allocator from the per-event path. Results are unchanged.

  This came out of a comparison against TA-Lib, where the family looked 6 to 7
  times slower. Most of that gap is not fixable and should not be fixed: the
  monotonic deque is O(1) per sample in the worst case, while the usual
  alternative tracks the extremum and rescans the window when it expires, which
  is quicker on typical data and degrades to O(window) on a monotone run. On a
  monotonically falling series, `RollingMax(50)` costs 3.0 ns/sample against
  TA-Lib's 16.2. A falling series is a sell-off, not a contrived input, so the
  bounded worst case is the right trade for a streaming library.
  `tests/test_extremum_worst_case.py` pins that property.

### Removed

* **Breaking:** dropped support for Python 3.11. `requires-python` is now
  `>=3.12`. The stable-ABI wheel needs nanobind's `STABLE_ABI`, which is
  available from CPython 3.12 onward, so 3.11 users must stay on an earlier
  screamer release or upgrade their interpreter.

### Added

* References for `RollingCorr`, `RollingCov`, `RollingBeta`, `RollingIqr`,
  `HullMA` and `RollingVWAP`, taking baseline coverage to 174 of 231. The pair
  statistics come from `pandas.rolling().corr()` / `.cov()` / `.var()`, which
  are external implementations rather than transcriptions.

### Fixed

* `ROC`, `ROCP` and `ROCR` defaulted to `window_size=1` while their pages
  documented `10`, TA-Lib's `timeperiod` default and the value the sibling
  `Momentum` uses. `ROC()` computed a 1-period rate of change and the page said
  otherwise. The bindings now default to 10. **Passing `window_size` explicitly
  is unaffected.** `tests/test_documented_defaults.py` now asserts every
  documented default against the pybind11 signature, so a page and its binding
  cannot drift apart again.

* `EwSkew` and `EwKurt` returned a value scaled down by roughly the effective
  sample size, so the result shrank as the window lengthened. Both fed
  *mean*-based moment ratios into bias corrections written for *sum*-based
  quantities. On exponential data (true skew 2.0) `EwSkew(span=100)` returned
  0.019; `EwKurt` returned about 0 for every distribution, which looked correct
  only on normal data, where the true excess kurtosis is 0. Both now use the
  adjusted Fisher-Pearson estimators, the same ones `scipy.stats` returns for
  `bias=False`:

      G1 = sqrt(n(n-1))/(n-2) * m3 / m2^(3/2)
      G2 = (n-1)/((n-2)(n-3)) * ((n+1) * m4/m2^2 - 3(n-1))

  **Output changes for both operators, by roughly a factor of `n_eff`.** Any
  stored value or threshold calibrated against the old output needs recomputing.

  Their reference implementations were wrong too, in a different way, which is
  why the comparison had been disabled with a `# todo baselines` note rather
  than investigated. Both baselines were rewritten from the estimators above,
  the comparison is enabled, and `tests/test_ew_moments.py` anchors them: the
  formulas reduce to `scipy.stats.skew/kurtosis(bias=False)` under equal
  weights, they recover the known moments of four distributions, and the
  estimate must not drift with window length on stationary input.

* A baseline whose optional dependency is not installed (TA-Lib needs a C
  library, so it sits in an opt-in extras group) raised out of
  `devtools/baselines/__init__.py` and took down collection for the entire test
  suite. Such a baseline is now skipped and recorded in
  `baselines.missing_dependencies()`. Only `ImportError` is swallowed; a
  baseline that fails for any other reason still raises.

* `FillNa` on a strided array tested the *output* buffer for `NaN` instead of the
  input, so it filled arbitrary elements with the fill value and passed real `NaN`
  inputs through unchanged. Reachable from the public API on any strided input;
  the contiguous path was correct. Found by giving the parity harness coverage of
  every operator (below).

* Twelve operators declared `nan_policy: ignore` but let a `NaN` input reach
  their state, so a single `NaN` cost more than the one output slot the policy
  promises. Two mechanisms: the `NaN` was stored into a carried previous-value
  slot that also served as the warmup sentinel (`TrueRange`, `ATR`, `TRIX`,
  `RollingYangZhangVar`/`Vol`), or it occupied a position in a fixed-size window
  and advanced warmup (`RollingIqr`, `RollingMaxDrawdown`, `RollingHurst`,
  `RollingTSF`, `RollingHitRate`, `Stoch`, `WilliamsR`). `RollingVWAP` and
  `KeltnerChannels` had a third form: they consumed one input into their own
  state while delegating the rest to a sub-operator, so a bar with one missing
  field left the two halves permanently out of step. `NATR` inherited the `ATR`
  defect and is fixed with it.

  **Output changes on input containing `NaN`.** A series with gaps now produces
  the documented result, which differs from what these operators returned
  before. Output on gap-free input is unchanged.

  Two compliance properties were added to `tests/test_nan_input_compliance.py`
  to catch this class: an ignored `NaN` must cost exactly one output slot, and
  for a multi-input operator one missing field must skip the whole row. The
  existing properties only checked that state recovered eventually and that the
  `NaN` index itself was `NaN`, which every one of these operators satisfied.

### Added

* 59 reference implementations, taking baseline coverage from 62 of 213
  operators to 121. Elementwise math (`Acos`, `Asin`, `Atan`, `Ceil`, `Cos`,
  `Cube`, `Floor`, `Identity`, `IsFinite`, `IsNan`, `Not`, `Round`, `Sin`,
  `Square`), cumulative and positional (`CumSum`, `CumProd`, `CumMax`,
  `CumMin`, `Diff2`, `Momentum`, `ROC`, `ROCP`, `ROCR`, `Detrend`, `Drawdown`,
  `MaxDrawdown`), and windowed (`RollingRange`, `RollingMad`,
  `RollingMedianAD`, `RollingHitRate`, `WMA`, `TRIMA`, `DEMA`, `TEMA`). Each is
  written from the documented definition rather than from screamer's
  implementation, and each is compared on every run. Writing them is what
  exposed the `ROC` default above.

* TA-Lib references for the technical-indicator family, taking baseline
  coverage to 158 of 213: `ATR`, `NATR`, `TrueRange`, `CCI`, `WilliamsR`,
  `ADX`, `Stoch`, `StochRSI`, `UltimateOscillator`, `BOP`, `AD`, `ADOSC`,
  `MFI`, `OBV`, `RollingRSI`, `KAMA`, `TRIX`, `MACD`, `RollingArgmax` and
  `RollingArgmin`. Several of these pages already cite TA-Lib as the definition
  they implement, so it is the right oracle rather than a second opinion.
  Fourteen of the twenty matched on the first run.

  `TA-Lib` is now a test dependency. Its wheels have bundled the C library
  since 0.6, so it installs in CI like any other package.

* `tests/test_baseline_defaults.py`: a reference's default arguments must match
  the operator's documented ones. Three of this batch's apparent failures were
  a reference defaulting to a different window than the operator (`KAMA` at 30
  against 10, `CCI` at 20 against 14, `TRIX` at span 15 against 14), which
  reads as an operator bug. Together with `tests/test_documented_defaults.py`
  this pins operator, page and reference to one value.

* References for the exponentially weighted volatility twins and the EW pair
  statistics, taking baseline coverage to 138 of 213. `EwParkinson*` and
  `EwGarmanKlass*` reuse the QuantLib per-bar kernel with EW weighting in place
  of rolling; `EwRogersSatchell*` is transcribed and anchored by the simulation
  test, which now covers the EW forms including the drift-robustness split.
  `EwCov`, `EwCorr` and `EwBeta` are checked against `pandas.ewm().cov()`,
  `.corr()` and `.var()`.

* References for the eight range-based volatility operators, taking baseline
  coverage to 129 of 213. `RollingParkinson*` and `RollingGarmanKlass*` are
  checked against **QuantLib** (`ParkinsonSigma` and `GarmanKlassSigma5`), an
  independent implementation of the formula rather than a transcription of it.
  `RollingRogersSatchell*` and `RollingYangZhang*` are not in QuantLib and are
  transcribed from their documented formulas.

  Because a transcription cannot catch a wrong formula,
  `tests/test_ohlc_volatility.py` anchors all four families against ground
  truth: a geometric Brownian motion is simulated at fine intra-bar resolution,
  aggregated into OHLC bars, and each estimator must recover the sigma that
  generated it, in the regime its page claims. The negative claims are asserted
  too. Under a drift of five times sigma, Rogers-Satchell stays on target while
  Parkinson reads about 3x and Garman-Klass about 2x, and with overnight gaps
  only Yang-Zhang reports the gap-inclusive total.

  `QuantLib` is now a test dependency. It is a plain wheel with no system
  library behind it, so unlike TA-Lib it installs in CI and these comparisons
  run on every push.

* The Garman-Klass pages now say *which* estimator from the paper is
  implemented (the simplified sigma_5 form), why the higher-order sigma_4 and
  the gap-aware sigma_1 / sigma_3 / sigma_6 variants are not shipped, and that
  Yang-Zhang supersedes the gap-aware ones. Identifying the variant previously
  meant testing all five against the formula.

* The baseline harness now drives multi-input operators, one independent array
  per input, which is what let the two- and three-input math operators be
  compared at all: `Add`, `Sub`, `Mul`, `Div`, `Hypot`, `Atan2`, `Linear2`,
  `Cart2Polar`, `Polar2Cart`, `Where`, and the comparison and logic family.
  Independence matters, since feeding one array to both sides of `Sub` compares
  all zeros. The `Butter*` family, `MovingAverage`, `SchmittTrigger`, `Hold` and
  `KalmanFilter` gained references in the same pass.

* `array_type: "discrete"`, small integers with repeats and real zeros, for the
  logic, equality and select operators. On continuous input two independent
  draws are never equal and are always both nonzero, so `Equal`, `And`, `Or`
  and `Where` were each compared on one branch only and asserted nothing.

* `array_type: "unit"` in the test harness, giving the inverse trigonometric
  operators input inside their domain. Driven with the previous `positive`
  input (uniform 0.1 to 10) most samples fell outside the domain and both sides
  returned `NaN`, so the comparison passed without asserting anything.


* `FracDiff(d, window_size, threshold, start_policy)` - fractional differentiation
  (Lopez de Prado, *Advances in Financial Machine Learning*, chapter 5). An FIR
  filter with taps `w_k = (-1)^k * binom(d, k)`, truncated at `window_size` taps or
  at the first weight below `threshold`. `d=0` is the identity, `d=1` reproduces
  `Diff(1)`, `d=2` reproduces `Diff2()`; fractional orders in between remove the
  trend while keeping memory an integer difference discards. Contributed by
  Mohammadjavad Vakili.

* `Resample(..., clock=True, agg='last', fill='carry', max_age=N)` bounds the
  age of a carried scalar value in a target-clock as-of join. Once its age
  exceeds `N`, the output is `NaN`; omitting `max_age` retains unbounded carry.
  Contributed by Édouard Tallent (@tagomatech).

1.1.1 - 2026-07-29
------------

### Fixed

* Value-initialized several latent uninitialized class members flagged by
  clang-tidy's member-init check (the resample plan entry, `BacktestReport`,
  `BacktestL1TradesOrders`, the L1 fill and PnL-account helpers, `TickRuleSign`,
  `LeeReadySign`, the stream `Event`, and the DAG node spec). These were
  uninitialized-read hazards of the class that can surface as platform-specific
  misbehavior; each fix is behavior-preserving (the values are set before use).

1.1.0 - 2026-07-29
------------

### Added

* `Hold(n, release=0.0)` - a time-based latch. On a nonzero input it holds that value
  for `n` steps, then returns `release`. Complements `SchmittTrigger` (level hysteresis)
  with time hysteresis.
* `Resample` information bars: `Resample(value, driver, threshold=T, agg=...)` closes a
  bar when a cumulative `driver` reaches `threshold`, giving volume bars, dollar bars,
  and other event-clock bars from one mechanism.
* `Resample` bar re-aggregation aggs `ohlc_bars` and `ohlcv_bars`: downsample
  already-built `[O,H,L,C]` / `[O,H,L,C,V,...]` bars into coarser bars (first/max/min/last
  on OHLC, sum on trailing columns) in a single C++ pass.
* `Resample` target-clock mode: `Resample(value, clock, clock=True, agg='last',
  fill='carry')` samples the last-known value as of each event of a separate clock stream
  (an as-of / `merge_asof` / reindex-forward-fill).

### Changed

* The `ohlcv` and `ohlcv2` resample aggregations now run in every regime (eager, graph,
  and lazy), not just eager. Batch results are unchanged; previously-rejected graph and
  lazy calls now succeed.
* `forecast_pairs` now runs in every regime (eager, graph, and lazy) for both `count=` and
  `duration=` modes, so a training set can be built by streaming a live feature engine.
  The public API and return values are unchanged.

1.0.1 - 2026-07-26
------------

### Changed

* screamer now installs with no runtime dependencies. `pybind11` is a build-time-only,
  header-only dependency and has been removed from the runtime requirements; it stays in
  the build-system requirements. Existing installs are unaffected.

1.0.0 - 2026-07-26
------------

First stable release. The public API is stable from this point and follows
semantic versioning.

### Changed (breaking)

* `forecast_pairs` now returns `(X_shifted, y)` instead of `(X_shifted, y, as_of)`.
  The third return (the row index) was redundant: the function delays `X` and passes
  `y` through, so the output rows sit on the caller's own timeline. Keep your own index
  alongside `X` and `y` if you need to map rows back to time.

### Added

* `Pipeline` accepts `max_pending` (default 1,000,000), the cap on the `CombineLatest`
  reorder buffer. A stream that stalls long enough to exceed the cap raises a
  `RuntimeError` instead of buffering without bound.

### Fixed

* A `Delay` feeding a `CombineLatest` merge inside a single `Pipeline` produced
  lookahead: the merge aligned each delayed frame against inputs that had not yet
  advanced, so the result was wrong in both batch and live mode. The DAG now carries an
  event-time watermark (a monotone lower bound on future frame indices) that gates a
  bounded reorder buffer at every fan-in, releasing frames in global index order once
  every input has advanced past them. `Filter` and `Dropna` forward the watermark past
  dropped frames, and `Resample` closes its windows on it, in both index and count modes.

### Documentation

* Full pass over the notebooks, guides, and every function reference page: corrected
  formulas and claims that did not match the implementation, and rewrote descriptions to
  lead with what each operator computes. Added a documentation writing guide to
  `CONTRIBUTING.md`.

0.13.0 - 2026-07-24
------------

### Changed (breaking)

* `EwKyleLambda` now accepts only the standard EW mutex (`com`, `span`, `halflife`,
  `alpha`), exactly one required. The previous positional `window` parameter is removed.
* `Propagator` parameter renamed from `window` to `window_size` for consistency with
  other windowed operators. Passing `window=` raises `TypeError`.
* `BayesianRegression` output columns reordered to `[slope, intercept, pred_mean, pred_std]`
  (was `[pred_mean, pred_std, slope, intercept]`), so slope and intercept sit at columns
  0 and 1 as in `RollingLinearRegression`.
* `Hampel`, `ImpulseClip`, `RollingSigmaClip`, `RollingOU`: the `output` parameter now
  accepts a string mode name instead of an integer. Passing an integer raises an error.
  Valid values: `Hampel`/`ImpulseClip` use `"cleaned"`, `"flag"`, `"nan"`;
  `RollingSigmaClip` uses `"clipped"`, `"mean"`, `"std"`, `"nan"`;
  `RollingOU` uses `"mrr"`, `"mean"`, `"relmean"`, `"std"`.

### Added

* `RollingSigmaClip` now exposes `start_policy` in its binding (previously omitted),
  defaulting to `"strict"` to match the other rolling operators.

### Fixed

* `EwMean`, `EwStd`, `BayesianRegression` and all other `Ew*` operators now raise
  `ValueError` when `alpha` is non-finite (`NaN` or `inf`).
* Removed contradictory "NaN values should be preprocessed" sentences from 16 docs pages
  that also carried a generated `Policy: ignore` footnote. The footnote is the authoritative
  statement; the body sentences were stale.
* Fixed incorrect comment in `bindings/bindings_misc.cpp` that said NaN "propagates" for
  `CumSum`/`CumProd`/`CumMax`/`CumMin`; they follow the standard `ignore` policy.

0.12.0 - 2026-07-23
------------

### Added

* `BayesianRegression`: online Bayesian univariate regression that emits a causal
  one-step-ahead predictive mean and standard deviation plus the current slope and
  intercept. Normal-Inverse-Gamma posterior (noise learned online, Student-t
  predictive) with exponential forgetting (`com`/`span`/`halflife`/`alpha`) and a weak
  prior, so estimates and intervals are defined from the first sample.
* Example notebook 16, "Supervised forecasting with `forecast_pairs`": build a
  leak-safe training set, fit a least-squares model, check it out of sample, and
  tie the prediction into a backtest.

### Fixed

* `forecast_pairs(count=..., dropna=True)` now also drops rows whose target is NaN
  (not only feature-warmup rows), matching `duration=` mode, so it returns a clean
  training set.

### Docs

* Documentation rewritten to the NumPy and scikit-learn voice. Removed the repeated
  batch-vs-live-stream property from per-operator docs and notebook closers (it is
  stated once, where streaming is the subject), the verification cells that only
  demonstrated equivalence, and the closing summary sections.

0.11.0 - 2026-07-22
------------

### Added

* Backtesting: engines are renamed and reorganised into a
  `Backtest<DataModel><OrderDef>` grid. The data model (Price, OHLC, Trades,
  L1, L1Trades) names the market feed; the order definition (Target, Orders)
  names the strategy's output contract. Eight engines fill eight of the ten
  cells of the 5x2 matrix; `BacktestPriceOrders` and `BacktestL1TradesTarget`
  are intentionally not provided (see `choosing_a_backtest_engine`).
* **Target engines** (`BacktestPriceTarget`, `BacktestOHLCTarget`,
  `BacktestTradesTarget`, `BacktestL1Target`): receive a scalar target
  position each event, compute the delta to the current position, and take
  liquidity to reach it. The target is clamped to the static
  `[min_position, max_position]` cap before sizing the order.
  `BacktestOHLCTarget` defers execution to the next bar's open (causal;
  no manual lag needed). `BacktestTradesTarget` and `BacktestL1Target` execute
  immediately against the current event.
* **Orders engines** (`BacktestOHLCOrders`, `BacktestTradesOrders`,
  `BacktestL1Orders`, `BacktestL1TradesOrders`): receive a two-sided resting
  quote `(bid_price, bid_size, ask_price, ask_size)` each event plus the
  market data columns. Either or both sides can fill on the same event. A
  quote submitted already crossing the spread is a taker.
* New engines completing the useful grid cells:
  `BacktestOHLCTarget` and `BacktestOHLCOrders` (replacing `BacktestOHLC` /
  `BacktestOHLCMaker`), `BacktestTradesTarget` and `BacktestTradesOrders`
  (replacing `BacktestTrades` / `BacktestTradesMaker`), `BacktestL1Target` and
  `BacktestL1Orders` (replacing `BacktestL1`), `BacktestL1TradesOrders`
  (replacing `BacktestL1Trades`), and `BacktestPriceTarget` (replacing
  `BacktestSignal`).
* All engines accept `min_position` and `max_position` (default unbounded).
  Fills are capped by a three-way minimum: order size, participation limit
  (where applicable), and remaining room to the position bound.
* MARKET/NaN/inf encoding shared across all engines: a finite price is a
  resting limit (maker); `NaN` is a side-agnostic market order (taker);
  `+inf` / `screamer.MARKET` is a market buy (never-fill sell); `-inf` is a
  market sell (never-fill buy).
* Docs: `choosing_a_backtest_engine` grid overview page with the 5x2 matrix,
  order-definition interfaces, MARKET encoding table, and fill-cap rule.
* `SchmittTrigger` gains an `initial` latch seed (`0.0`, `1.0`, or `NaN`).
* `Delay(duration)` stream op: re-stamp each event's index by a time offset (the
  time-based counterpart of `Lag`). Requires an explicit index; lossless, 1:1,
  no warmup.
* `screamer.supervised.forecast_pairs(X, y, count=|duration=)`: build a forecasting
  training set by lagging features to align with a future causal target. Returns
  `(X_shifted, y, as_of)`; `count=` is event-based, `duration=` is time-based (uses
  `Delay`, needs an index). Fully causal (lags X, never leads y).

### Changed (breaking)

* `SchmittTrigger` now seeds its latch with `initial` (default `0.0`, low) instead
  of `NaN`, so a signal that starts inside the dead band reads low during warmup
  rather than `NaN`. Pass `initial=nan` to restore the previous undefined-until-first-crossing
  behavior; pass `initial=1.0` to start high.

0.10.0 - 2026-07-18
------------

### Added

* Risk statistics: `RollingDownsideDeviation` (downside semideviation, the Sortino
  denominator), `RollingOmega` (Omega ratio of gains to losses about a threshold),
  and `RollingCVaR` (historical Conditional Value-at-Risk / Expected Shortfall,
  the mean loss in the worst alpha tail; VaR is `-RollingQuantile`). Each with a
  reference page, a plotted example, and tests.
* Backtesting: a suite of five causal C++ engines named by the market data they
  consume, all sharing one accounting core (`detail::PnLAccount`) and the
  `[equity, pnl, position, cost]` output schema. Fills follow one rule set: a
  trade or quote *through* your price fills the full remaining, a fill *at* your
  price is `min(remaining, participation_ratio * available_size)`, and a
  marketable order fills fully with `tick_size` slippage. Resting orders fill at
  their limit price (maker); an order submitted already crossing is a taker.
  * `BacktestSignal` (2 inputs): marks a position signal to a price, with a
    fractional `spread` (crossing cost) and `fee`.
  * `BacktestOHLC` (6 inputs): a directional target-position strategy on OHLC
    bars, with market orders (fill at the open, crossing half the `spread`, paying
    `taker_fee`) and limit orders (`"touch"`/`"breach"` of the bar range, paying
    `maker_fee`); bars fill in full. Causal by design: the target decided on a
    bar's close is deferred and traded on the next bar, so no manual lag is needed.
  * `BacktestTrades` (4 inputs): a resting limit order against the trade tape;
    a through-print sweeps the full order, an at-print fills a `participation_ratio`
    share.
  * `BacktestL1` (8 inputs): a two-sided maker against top-of-book quotes only.
    Fills are a documented heuristic: `"breach"` (default) fills on a through,
    `"touch"` adds a participation partial once per lock episode. Inventory cap,
    `taker_fee`, and `tick_size` for marketable orders.
  * `BacktestL1Trades` (10 inputs): the preferred maker engine. Quotes mark the
    position, the trade tape drives fills (each trade on its own event row, so no
    fill-versus-cancel ambiguity), a quote cross with no explaining trade is the
    run-over fallback.
  * `BacktestReport` (4 inputs, 6 outputs): the C++ node that turns an engine's
    `[equity, pnl, position, cost]` into the running report columns (dollar
    drawdown, cumulative cost, turnover, trade count, running max drawdown, and
    running Sharpe), so pure-C++ callers get the statistics too.
  The `backtest_report` helper wraps `BacktestReport`, labeling its columns and
  reading the last row into a summary (total PnL, max drawdown, cost, turnover,
  trades, Sharpe). It returns plain dicts of numpy arrays and needs no pandas.
  Reference pages with plotted examples, tests, and two demo notebooks (a signal
  on bars, and the event-driven engines on a real tape).

[0.9.0] - 2026-07-17
--------------------

More microstructure: order-flow toxicity, book pressure, and spread
decomposition, plus a re-classified docs taxonomy.

### Added

* Microstructure tranche 2, order-flow toxicity and book features:
  * `VPIN` (Easley-Lopez de Prado-O'Hara 2012): order-flow toxicity over a
    self-contained volume clock (equal-volume buckets, boundary-splitting).
  * `MicroPrice` (Stoikov 2018, first-order): the imbalance-weighted mid, a fair
    value that leans toward the thinner side of the book.
  * `QueueImbalance`: L1 book (queue) imbalance, a documented synonym of `OFI`
    applied to resting bid/ask sizes.
* Microstructure tranche 3, order-book flow and spread decomposition:
  * `ContOFI` (Cont-Kukanov-Stoikov 2014): the canonical order-flow imbalance
    from L1 book events, distinct from the trade-flow `OFI`.
  * `EffectiveSpread` (`2*|price - mid|`) and `RealizedSpread` (the liquidity
    part kept after the price moves); their difference is the price-impact /
    adverse-selection component.
* Each ships with a reference page, a plotted usage example, and tests.
* Docs: the flat "Microstructure" function topic is split into four groups -
  Trade signing, Order-flow imbalance, Price impact & liquidity, and Order-flow
  arrivals - with "microstructure" kept as a search tag.
* Docs: a third microstructure notebook (toxicity, book pressure, and spreads)
  covering VPIN, QueueImbalance, MicroPrice, ContOFI, and the effective/realized
  spread decomposition.

[0.8.0] - 2026-07-16
--------------------

Microstructure and order-flow operators, and the lazy Pipeline path moved into
the C++ core.

### Added

* Microstructure and order-flow operators, all implemented as C++ core nodes:
  * Trade signing: `TickRuleSign`, `LeeReadySign`, `SignedVolume`, and the
    bar-level `BulkVolumeClassifier` (BVC).
  * Order flow: `OFI` (order-flow imbalance) and `RollingOrderImbalance`.
  * Price impact and liquidity: `RollingKyleLambda`, `EwKyleLambda`,
    `AmihudIlliquidity`, `RollSpread` (Roll effective spread), and the Bouchaud
    `Propagator`.
  * Event intensity: `HawkesIntensity`, a self-exciting arrival-rate model.
* Two demo notebooks driven by a small committed real-data slice (six hours of
  Deribit BTC- and ETH-perpetual trades): order flow and trade signing, and
  price impact and liquidity.

### Internal

* The lazy (Python-iterator) `Pipeline` path now runs in the C++ core. A C++
  driver merges the input feeds, drives the compiled graph, and runs the
  multi-output watermark as-of join, replacing the previous Python driver. The
  `batch == lazy == graph` invariant is now enforced by one implementation
  rather than by parallel ones agreeing. The k-way merge refills lazily, so it
  never reads further ahead than requested (safe for endless streams).
* Eager `combine_latest` coalesces per index in C++ (the Python dedup is gone),
  and `split` runs in C++.

[0.7.0] - 2026-07-12
--------------------

The v2 API: one consistent call shape, streams as plain tuples, and pipelines.

### Changed (breaking)

* Stream operators are now CamelCase config-first classes, called like the
  functors as `Op(config)(data)`: `Resample`, `Dropna`, `Select`, `Filter`,
  `CombineLatest`, `Merge`. The lowercase function forms are removed from the
  public API.
* The `Stream` class is removed. A stream is a plain `(values, index)` tuple;
  `to_pandas` / `from_pandas` convert to and from pandas.
* `Dag` is renamed to `Pipeline` (`from screamer import Pipeline`).
* `Resample` takes `freq=` (a time window) or `count=` (a number of arrivals),
  and `agg=` a functor or string. The `every=`, `func=`, and `agg={dict}` forms
  are gone; compose several reducers with `CombineLatest` instead.
* `replay` and `multi_resample` are removed; compose the existing operators.
* The per-operator streaming variants (`*_iter`) were removed in 0.5; the
  unified operators handle both batch and lazy inputs:

  | Old (removed) | New |
  |---|---|
  | `resample_iter(events, ...)` | `Resample(...)(events)` |
  | `dropna_iter(events)` | `Dropna()(events)` |
  | `filter_iter(events, pred)` | `Filter()(data, mask)` (mask replaces predicate) |
  | `select_iter(events, cols)` | `Select(cols)(events)` |
  | `combine_latest_iter(a, b)` | `CombineLatest()(a, b)` |
  | `merge_iter(a, b)` | `Merge()(a, b)` |

### Added

* Comparison and logic operators: `GreaterThan`, `LessThan`, `GreaterEqual`,
  `LessEqual`, `Equal`, `NotEqual`, `And`, `Or`, `Not`, `Where`, `IsNan`,
  `IsFinite`. These build the masks the new `Filter` gate consumes.
* Expanding whole-history statistics: `ExpandingMean`, `ExpandingVar`,
  `ExpandingStd`, `ExpandingSkew`, `ExpandingKurt`, `ExpandingSlope`, and the
  running `ExpandingSum` / `ExpandingMax` / `ExpandingMin` / `ExpandingProd`.
* `PosPart` (`max(x, 0)`) and `NegPart` (`max(-x, 0)`).
* OHLC bar aggregations for `Resample`: `agg="ohlc"`, `"ohlcv"`, `"ohlcv2"`.

### Internal

* All stream-operator compute moved into C++ (dropna, select, filter, and merge
  as graph nodes), so the Python bindings stay thin.

### Fixed

* `Clip` vectorizes to SIMD min/max, about 2x faster; it had regressed to 2x
  slower than `np.clip`.


[0.6.0] - 2026-07-06
--------------------

Multi-stream and pipeline infrastructure.

### Added

* The streams layer, for aligning and reshaping feeds that do not tick together:
  `CombineLatest`, `Merge`, `Dropna`, `Filter`, `Select`, `split`, and time- or
  count-based `Resample`.
* Pipelines (then named `Dag`): wire operators into a reusable graph, run it in
  batch or live with identical results, serialize it to JSON, and visualize it as
  a text tree or Graphviz.
* Despiking functors: `RollingMedianAD`, `Hampel`, `ImpulseClip`.
* A topic taxonomy and unified frontmatter across all function reference pages.


[0.5.0] - 2026-05-21
--------------------

### Added

* `SchmittTrigger`, a hysteresis comparator.

### Changed

* The `ignore` NaN policy is applied consistently across the library: a `NaN`
  input is skipped and never corrupts a function's internal state.


[0.4.0] - 2026-05-20
-------------------------

### Changed (breaking - JSON consumers)

* `screamer/data/help.json` schema: the freeform `body_markdown` field has been
  removed and replaced with two structured fields:
  - `details` (string) - markdown prose, guaranteed to contain no fenced
    code blocks. Use this when rendering the description / math / notes.
  - `examples` (list of `{language, caption, code}`) - extracted code
    examples, one entry per `### Caption` heading in the source markdown.
    `{eval-rst} .. plotly::` directives are unwrapped to plain python.
  
  Consumers that read `body_markdown` must switch to `details` (and
  optionally render `examples` separately). No backwards-compatibility
  shim is provided.

### Changed

* Function reference docs (`docs/functions_*/<Name>.md`) now follow a
  canonical layout: prose lives under H2 sub-headings (Description,
  Formula, …), examples live under a single `## Examples` H2 with one
  `### Caption` per example. The sphinx-rendered pages adopt the same
  structure.

[0.3.0] - 2026-05-11
--------------------

This release more than doubles the indicator surface (67 → 153 exposed
classes) and closes six of the seven roadmap sections. Almost every
new class is cross-validated against TA-Lib, pandas, scipy, or
pandas-ta-classic to floating-point precision; documented divergences
(EMA convention, Cutler vs Wilder RSI) are pinned by tests so future
drift fires. Full test suite: 2126 passing.

### Added

#### Dispatcher infrastructure

* **Plan E (`N→M` dispatch)** completed. `FunctorBase` now supports
  any combination of multi-input + multi-output classes (1→1, N→1,
  1→M, N→M). First consumers: `Cart2Polar` / `Polar2Cart` (2→2),
  `KeltnerChannels` (3→3), `RollingLinearRegression` (2→4).
* New shared primitives: `detail::MonotonicDeque<bool IsMax>` (six+
  classes share it), `detail::WilderSmoother`.
* Documented multi-output shape rule
  `output.shape == single_input.shape + (M,)` with column-by-column
  pairing in `docs/polymorphic_api.md`.

#### Math

* Element-wise: `Floor`, `Ceil`, `Round` (banker's rounding),
  `Square`, `Cube`, `Sin`, `Cos`, `Atan`, `Asin`, `Acos`, `Identity`,
  `Hypot` (2→1), `Atan2` (2→1).
* `Linear2(a, b, c)` - two-input affine combination
  `a*x + b*y + c`, pairs with `Sign`/`Relu`/`Sigmoid` for compact
  composed expressions.
* `Cart2Polar` / `Polar2Cart` - 2→2 coordinate conversions; first
  consumers of the N→M dispatcher.

#### Misc / data transforms

* `CumSum`, `CumProd`, `CumMax`, `CumMin` - `O(1)` running
  reductions matching numpy.
* `Diff2` - second-order finite difference (discrete second
  derivative).
* `Detrend(window)` - `x − rolling_mean(x)`.
* `Momentum(k)` - alias of `Diff(k)` (TA-Lib's `MOM`).

#### Rolling-window statistics

* `RollingMad`, `RollingIqr`, `RollingRange` - composition of
  existing primitives or sharper variants.
* `RollingArgmin`, `RollingArgmax` - window-offset of the
  rolling extremum (TA-Lib's `MININDEX` / `MAXINDEX`).
* `RollingRank`, `RollingPercentile` - pandas-style position
  metrics with average tie rule.

#### Exponentially-weighted statistics

* `EwCov`, `EwCorr`, `EwBeta` - 2-input pair statistics. Matches
  pandas `ewm(adjust=True).cov / .corr` bit-exactly. EwBeta follows
  the CAPM `(target, regressor)` convention.

#### Moving averages

* `WMA` - linearly-weighted moving average, O(1) per step via the
  identity `W[t] − W[t−1] = w·x[t] − S[t−1]`.
* `DEMA`, `TEMA` - Mulloy's double/triple EMA compositions.
* `TRIMA` - triangular MA (`SMA(SMA(x))`).
* `HullMA` - `WMA(2·WMA(n/2) − WMA(n), √n)`.
* `KAMA` - Kaufman Adaptive MA with O(1) per step; matches TA-Lib
  bit-exactly.

#### Momentum / oscillators

* `MACD` (1→3), `WilliamsR` (3→1), `Stoch` (3→2, fast and slow
  via `smooth_k`), `StochRSI` (1→2), `TRIX`, `BOP` (4→1), `CCI`
  (3→1), `UltimateOscillator` (3→1), `ADX` (3→3 returning
  `+DI` / `-DI` / `ADX`).
* `ROC`, `ROCP`, `ROCR` - TA-Lib rate-of-change family.
* `RollingRSI` default changed to **Wilder's smoothing** (matches
  TA-Lib and pandas-ta); `method="cutler"` preserves the old
  Cutler form. Earlier versions diverged from TA-Lib's `RSI` by
  ~11 RSI points -- now bit-exact.

#### Volatility / range

* Range-based volatility quartet: **Parkinson**, **Garman-Klass**,
  **Rogers-Satchell**, **Yang-Zhang**. Each ships in `Var` and
  `Vol` variants, with rolling and EW smoothing (for the first
  three) -- 14 classes total.
* `TrueRange`, `ATR(window)`, `NATR(window)` - Wilder's
  bar-aware volatility family.
* `DonchianChannels` (2→3), `KeltnerChannels` (3→3) - channel
  indicators.

#### Volume-aware

* `RollingVWAP`, `OBV`, `AD`, `ADOSC`, `MFI` - first volume-aware
  primitives in screamer; all bit-exact to TA-Lib counterparts
  (except `ADOSC`, which inherits the documented EMA-convention
  divergence).

#### Performance / risk

* `Drawdown`, `MaxDrawdown`, `RollingMaxDrawdown`,
  `RollingSharpe(window, periods_per_year)`,
  `RollingSortino(window, ppy, target)`,
  `RollingInfoRatio(window, ppy)`,
  `RollingCalmar(window, ppy)`,
  `RollingHitRate(window)` - backtest-evaluation metrics. None of
  these are in TA-Lib; they're a real differentiator for screamer
  in trading pipelines.

#### Statistical / regression

* `RollingAlpha` - companion intercept to `RollingBeta`.
* `RollingResidualStd` - std of the per-bar `RollingSpread`.
* `RollingLinearRegression` (2→4) - full OLS fit returning
  `(slope, intercept, r², stderr)`. First 2→4 consumer of the
  N→M dispatcher. `stderr` is the RMSE of residuals (standard
  error of estimate, not slope-stderr).
* `RollingTSF` - TA-Lib's Time-Series Forecast (regression vs
  time projected one step ahead), bit-exact to `talib.TSF`.
* `RollingHurst(window, min_scale=4, method='rs')` - rolling Hurst
  exponent via Anis-Lloyd corrected rescaled-range analysis at
  dyadic scales. Bit-exact to the reference Python implementation;
  ~0.5 on white noise, >0.5 on integrated processes.

#### Signal processing

* `ButterHighpass`, `ButterBandpass`, `ButterBandstop` -
  high-pass, band-pass, band-stop Butterworth IIR filters.
  Added the underlying `lp2hp_zpk` / `lp2bp_zpk` / `lp2bs_zpk`
  ZPK transformations so future Bessel/Cheby/Elliptic families
  also get all four btypes once their prototypes are written.
* `MovingAverage(taps)` - FIR with arbitrary user-supplied taps
  (pair with `np.hamming` / `np.kaiser` / `scipy.signal.firwin`).
* `KalmanFilter(process_var, observation_var)` - scalar 1-D
  Kalman, O(1) per step.

### Validation

* New `tests/test_third_party_alignment.py` runs against **TA-Lib**
  and **pandas-ta-classic** under the new optional `validation`
  install group. Tests are skipped gracefully when the libraries
  are unavailable.
* New `docs/conventions.md` documents the few deliberate
  divergences (EMA `adjust=True` vs TA-Lib's adjust=False + SMA
  seed; `ddof=1` vs ddof=0 in `RollingStd`; Wilder vs Cutler RSI).
  Each divergence is asserted in the expected direction so future
  drift in either screamer or the third-party library trips the
  test.

### Changed

* `RollingRSI` default is now Wilder smoothing (was Cutler);
  `method="cutler"` preserves the previous behaviour.
* Pre-existing pending items now part of this release: general-
  order Butterworth filter, `RollingOU`, refactored devtools.

Version v0.1.46 (2024-11-02)
-------------------------

### Added

* RollingSigmaClip
* Relu
* Elu
* Selu
* Sigmoid
* Tanh
* Softsign
* Linear
* Power

Version v0.1.35 (2024-11-01)
-------------------------

* Improved documentation

Version v0.1.34 (2024-10-31)
-------------------------

Version v0.1.33 (2024-10-31)
-------------------------

### Added

#### Data handeling

* fillna
* ffill
* clip

#### Math

* Abs
* Sign
* Exp
* Log
* Sqrt
* Erf
* Erfc

#### Simple transforms

* Diff
* Lag
* Return 
* LogReturn

#### Rolling window functions

* rolling std
* rolling skew
* rolling kurtosis
* rolling zscore
* rolling min
* rolling max
* rolling median
* rolling quantile
* rolling rms
* rolling poly1, 1rst order polynomial fit
* rolling poly2, 2nd order polynomial fit

#### Exponentiually weighted functions

* EwMean
* EwStd
* EwVar
* EwSkew
* EwKurt
* EwRms
  
#### Filters

* 2nd order Butterworth


#### Interface
* support for iterator / generator processing

### Fixed
* Fixed incorrect results when applying transforms to a view on a numpy array.

### Changed
* removed the transform member functions

Version v0.1.32 (2024-10-20)
-------------------------

### Added

* Differences, Simple Return, Log Return, Rolling Sum, Simple Moving Average

### Changed
* removed initial_value from the constructor, we (for now) return NaN values when we cant resolve indicators.

Version v0.1.31 (2024-10-20)
-------------------------

### Added
* The indicator.transform() member functions can now transform multi-dimensional numpy arrays.
* Added documentation.


