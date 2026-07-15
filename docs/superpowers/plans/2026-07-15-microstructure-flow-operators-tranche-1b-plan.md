# Microstructure & Order-Flow Operators (Tranche 1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the screamer `microstructure` module with the rest of the flow toolkit - order-imbalance, Lee-Ready and bulk-volume trade signing, Roll's effective spread - and the two predictor centerpieces, a self-exciting Hawkes intensity and a Bouchaud propagator.

**Architecture:** Same as tranche 1a. Composition/alias operators (RollingOrderImbalance, LeeReadySign, BulkVolumeClassifier, RollSpread) wrap existing C++ ops, inheriting causality and batch==stream. The two novel operators (HawkesIntensity, Propagator) are pure-Python stateful streaming operators whose per-sample update logic is used identically for a whole-array call and a one-sample-at-a-time call, so batch==stream holds by construction. Every operator exposes `reset()`.

**Tech Stack:** Python 3.11, numpy, the screamer C++ bindings (`RollingSum`, `RollingStd`, `RollingCov`, `Diff`, `Lag`, `Sign`, `Erf`), pytest.

## Global Constraints

- Operators are added to the EXISTING module `screamer/microstructure.py` (do not create a new module); append to its `__all__`. Tranche 1a already wired this module into the `__init__` generator, `tests/param_cases.py` (pure-Python ops excluded from the C++ functor sweep by `screamer.microstructure.__all__`), and `tests/test_doc_coverage.py` - so no changes to those files are needed; new ops are picked up automatically.
- All operators are **causal** (value at t uses only inputs <= t).
- Library-wide `nan_policy: ignore`: output at t is NaN iff an input at t is NaN or t is warmup, and a NaN must not permanently poison recursive state. The compliance harness `tests/test_nan_start_policy_compliance.py` drives every `help.json` entry (including these) scalar-by-scalar vs whole-array and asserts **batch==stream**; the novel ops must pass it.
- Every operator exposes `reset()` (stateless -> `pass`; stateful -> reset held sub-ops / clear buffers / zero state).
- Windowed composition ops use `window_size` (int, default 20, min 2) + `start_policy in {strict, expanding, zero}`.
- An operator class with an `__init__` taking parameters MUST carry a pybind11-style first-line signature in its `__init__` docstring (e.g. `"""__init__(self: RollSpread, window_size: int = 20, start_policy: str = 'strict') -> None"""`) so `devtools/build_help_registry.py`'s parameter cross-check passes. This is load-bearing; do not omit it.
- Each public operator MUST have a `docs/functions_micro/<Name>.md` (frontmatter `name,title,implementation_family: micro,topics,tags,short,inputs,outputs,parameters,nan_policy` + a `## Description` body that teaches the model with references; synonyms in `tags`). Topics must exist in `docs/topics.yml` (`microstructure` and `statistics` already exist). No em-dashes in docs.
- NEVER hand-edit generated `screamer/__init__.py` / `screamer/data/help.json` / version files; regenerate with `make regen-init` and `poetry run python devtools/build_help_registry.py`.
- Run everything with `poetry run` from `/Users/thijs/screamer`. Do NOT touch `examples/deribit_research.ipynb`.
- Per-op verify command: `make regen-init && poetry run python devtools/build_help_registry.py && poetry run python -m pytest tests/test_doc_coverage.py tests/test_microstructure.py tests/test_nan_start_policy_compliance.py -q`

## File Structure

- Modify `screamer/microstructure.py` - append the six operator classes + `__all__` entries.
- Modify `tests/test_microstructure.py` - append tests.
- Create `docs/functions_micro/{RollingOrderImbalance,LeeReadySign,BulkVolumeClassifier,RollSpread,HawkesIntensity,Propagator}.md`.
- Regenerated (do not hand-edit): `screamer/__init__.py`, `screamer/data/help.json`.

---

### Task 1: `RollingOrderImbalance` and `LeeReadySign`

**Files:** Modify `screamer/microstructure.py`, `tests/test_microstructure.py`; Create `docs/functions_micro/RollingOrderImbalance.md`, `docs/functions_micro/LeeReadySign.md`

**Interfaces:**
- Consumes: `RollingSum`, `Sign` (from `. import`), and the existing `TickRuleSign` (same module).
- Produces:
  - `RollingOrderImbalance(window_size=20, start_policy="strict")` - `(signed_flow) -> ndarray`, the trailing-window sum of signed flow (a `RollingSum` alias). Inputs 1, outputs 1.
  - `LeeReadySign()` - `(price, mid) -> ndarray` of trade signs: `+1` when `price > mid`, `-1` when `price < mid`, and the tick-rule sign of `price` when `price == mid`. Inputs 2, outputs 1.

- [ ] **Step 1: Write the failing tests** in `tests/test_microstructure.py`

```python
def test_rolling_order_imbalance_equals_rolling_sum():
    from screamer import RollingSum
    from screamer.microstructure import RollingOrderImbalance
    flow = np.array([1.0, -2.0, 3.0, -1.0, 2.0])
    np.testing.assert_allclose(RollingOrderImbalance(window_size=3)(flow),
                               RollingSum(3)(flow), equal_nan=True)


def test_lee_ready_sign_uses_mid_then_tick_fallback():
    from screamer.microstructure import LeeReadySign
    price = np.array([100.0, 101.0, 101.0, 100.0])
    mid   = np.array([100.5, 100.5, 101.0, 100.5])
    # p<mid -> -1 ; p>mid -> +1 ; p==mid -> tick rule (101 vs prev 101 = unchanged -> carry +1) ; p<mid -> -1
    out = LeeReadySign()(price, mid)
    np.testing.assert_allclose(out, [-1.0, 1.0, 1.0, -1.0])


def test_lee_ready_sign_is_causal():
    from screamer.microstructure import LeeReadySign
    price = np.array([100.0, 101.0, 100.5, 101.0]); mid = np.array([100.0, 100.0, 100.0, 100.0])
    full = LeeReadySign()(price, mid)
    trunc = LeeReadySign()(price[:3], mid[:3])
    np.testing.assert_allclose(full[:3], trunc, equal_nan=True)
```

- [ ] **Step 2: Run, confirm fail** `poetry run python -m pytest tests/test_microstructure.py -k "order_imbalance or lee_ready" -q` - FAIL (ImportError).

- [ ] **Step 3: Implement** in `screamer/microstructure.py` (add `"RollingOrderImbalance", "LeeReadySign"` to `__all__`; extend the `from . import ...` line with `RollingSum` if not present):

```python
class RollingOrderImbalance:
    """Trailing-window sum of signed order flow (Chordia-Roll-Subrahmanyam order
    imbalance). Specializes RollingSum.
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: RollingOrderImbalance, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._sum = RollingSum(window_size, start_policy)

    def __call__(self, signed_flow):
        return self._sum(signed_flow)

    def reset(self):
        self._sum.reset()


class LeeReadySign:
    """Lee-Ready (1991) trade sign: +1 when the trade prints above the mid, -1
    below, and the tick-rule sign of price when it prints exactly at the mid.

    The tick-rule fallback is fed every price (not only at-mid ones) so its state
    stays consistent whether the operator is driven by a whole array or one
    sample at a time - so batch == stream holds. NaN in price or mid yields NaN.
    """

    def __init__(self):
        self._tick = TickRuleSign()

    def __call__(self, price, mid):
        price = np.asarray(price, dtype=float)
        mid = np.asarray(mid, dtype=float)
        tick = self._tick(price)                     # advance tick state on every sample
        above = np.sign(price - mid)                 # +1 / 0 / -1 (NaN preserved)
        out = np.where(above != 0.0, above, tick)    # at-mid (0) -> tick-rule fallback
        return np.where(np.isnan(price) | np.isnan(mid), np.nan, out)

    def reset(self):
        self._tick.reset()
```

- [ ] **Step 4: Run, confirm pass** `poetry run python -m pytest tests/test_microstructure.py -k "order_imbalance or lee_ready" -q` - PASS.

- [ ] **Step 5: Write help pages.** `RollingOrderImbalance.md` (inputs 1, outputs 1, params `window_size`+`start_policy`, tags `[order imbalance, order flow, imbalance, microstructure]`, `see_also: [RollingSum, OFI]`, cite Chordia-Roll-Subrahmanyam). `LeeReadySign.md` (inputs 2, outputs 1, `parameters: []`, tags `[trade sign, lee ready, classification, flow, microstructure]`, `see_also: [TickRuleSign, SignedVolume]`, cite Lee, Ready 1991). No em-dashes.

- [ ] **Step 6: Regenerate + verify** `make regen-init && poetry run python devtools/build_help_registry.py && poetry run python -m pytest tests/test_doc_coverage.py tests/test_microstructure.py tests/test_nan_start_policy_compliance.py -q` - PASS.

- [ ] **Step 7: Commit**

```bash
git add screamer/microstructure.py tests/test_microstructure.py \
        docs/functions_micro/RollingOrderImbalance.md docs/functions_micro/LeeReadySign.md \
        screamer/__init__.py screamer/data/help.json
git commit -m "feat(micro): RollingOrderImbalance + LeeReadySign"
```

---

### Task 2: `BulkVolumeClassifier`

**Files:** Modify `screamer/microstructure.py`, `tests/test_microstructure.py`; Create `docs/functions_micro/BulkVolumeClassifier.md`

**Interfaces:**
- Consumes: `RollingStd`, `Erf`.
- Produces: `BulkVolumeClassifier(window_size=20, start_policy="strict")` - `(return_) -> buy_fraction in [0,1]`, the BVC estimate of the buy-initiated share of a bar's volume: `Phi(return_ / sigma)` where `sigma` is the trailing-window std of `return_` and `Phi` is the standard normal CDF. Inputs 1, outputs 1.

- [ ] **Step 1: Write the failing test** in `tests/test_microstructure.py`

```python
def test_bvc_is_normal_cdf_of_standardized_return():
    from screamer import RollingStd, Erf
    from screamer.microstructure import BulkVolumeClassifier
    rng = np.random.default_rng(0)
    ret = rng.normal(scale=0.01, size=200)
    out = BulkVolumeClassifier(window_size=50)(ret)
    sigma = np.asarray(RollingStd(50)(ret))
    z = ret / sigma
    ref = 0.5 * (1.0 + np.asarray(Erf()(z / np.sqrt(2.0))))
    np.testing.assert_allclose(out, ref, equal_nan=True)
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0   # a fraction
```

- [ ] **Step 2: Run, confirm fail** `poetry run python -m pytest tests/test_microstructure.py -k bvc -q` - FAIL (ImportError).

- [ ] **Step 3: Implement** (add `"BulkVolumeClassifier"` to `__all__`; extend imports with `RollingStd, Erf`):

```python
class BulkVolumeClassifier:
    """Bulk Volume Classification (Easley-Lopez de Prado-O'Hara 2012): the
    buy-initiated share of a bar's volume, estimated as the standard normal CDF
    of the bar return divided by its trailing-window volatility. Works on
    aggregate bars, no tick data needed. Output is a fraction in [0, 1].
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: BulkVolumeClassifier, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._std = RollingStd(window_size, start_policy)
        self._erf = Erf()

    def __call__(self, return_):
        ret = np.asarray(return_, dtype=float)
        sigma = np.asarray(self._std(ret), dtype=float)
        z = ret / sigma
        return 0.5 * (1.0 + np.asarray(self._erf(z / np.sqrt(2.0))))

    def reset(self):
        self._std.reset()
        self._erf.reset()
```

- [ ] **Step 4: Run, confirm pass** `poetry run python -m pytest tests/test_microstructure.py -k bvc -q` - PASS.

- [ ] **Step 5: Write help page** `BulkVolumeClassifier.md` (inputs 1, outputs 1, params `window_size`+`start_policy`, tags `[trade sign, bulk volume, bvc, order flow, toxicity, microstructure]`, `see_also: [TickRuleSign, LeeReadySign]`, cite Easley, Lopez de Prado, O'Hara 2012). No em-dashes.

- [ ] **Step 6: Regenerate + verify** (same command as Task 1 Step 6) - PASS.

- [ ] **Step 7: Commit**

```bash
git add screamer/microstructure.py tests/test_microstructure.py \
        docs/functions_micro/BulkVolumeClassifier.md screamer/__init__.py screamer/data/help.json
git commit -m "feat(micro): BulkVolumeClassifier (BVC) bar-level trade signing"
```

---

### Task 3: `RollSpread`

**Files:** Modify `screamer/microstructure.py`, `tests/test_microstructure.py`; Create `docs/functions_micro/RollSpread.md`

**Interfaces:**
- Consumes: `Diff`, `Lag`, `RollingCov`.
- Produces: `RollSpread(window_size=20, start_policy="strict")` - `(price) -> effective_spread`, Roll's (1984) estimator `2*sqrt(-cov(dP_t, dP_{t-1}))` over a trailing window, where `dP = Diff(price)`. When the serial covariance is non-negative (no bid-ask bounce detected) the estimate is undefined and returns NaN. Inputs 1, outputs 1.

- [ ] **Step 1: Write the failing test** in `tests/test_microstructure.py`

```python
def test_roll_spread_recovers_bounce_and_is_nan_when_undefined():
    from screamer.microstructure import RollSpread
    # a clean +/-0.1 bid-ask bounce around 100 -> serial cov of price changes is
    # negative -> Roll spread is defined and positive
    price = 100.0 + 0.1 * np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    out = RollSpread(window_size=6)(price)
    assert np.isfinite(out[-1]) and out[-1] > 0.0
    # a monotonic ramp -> serial cov >= 0 -> undefined -> NaN
    ramp = np.arange(12.0)
    assert np.isnan(RollSpread(window_size=6)(ramp)[-1])
```

- [ ] **Step 2: Run, confirm fail** `poetry run python -m pytest tests/test_microstructure.py -k roll_spread -q` - FAIL (ImportError).

- [ ] **Step 3: Implement** (add `"RollSpread"` to `__all__`; extend imports with `Diff, Lag, RollingCov`):

```python
class RollSpread:
    """Roll's (1984) effective spread from trade prices alone:
    2 * sqrt(-cov(dP_t, dP_{t-1})) over a trailing window, where dP is the price
    change. Bid-ask bounce makes successive price changes negatively correlated;
    a non-negative covariance leaves the estimate undefined (returns NaN).
    """

    def __init__(self, window_size=20, start_policy="strict"):
        """__init__(self: RollSpread, window_size: int = 20, start_policy: str = 'strict') -> None"""
        self._diff = Diff(1)
        self._lag = Lag(1)
        self._cov = RollingCov(window_size, start_policy)

    def __call__(self, price):
        d = self._diff(price)                        # dP_t
        cov = np.asarray(self._cov(d, self._lag(d)), dtype=float)   # cov(dP_t, dP_{t-1})
        return 2.0 * np.sqrt(np.where(cov < 0.0, -cov, np.nan))

    def reset(self):
        self._diff.reset()
        self._lag.reset()
        self._cov.reset()
```

Note: verify `RollingCov`'s constructor accepts `(window_size, start_policy)` by checking its help page `docs/functions_fin/RollingCov.md`; if it takes only `window_size`, drop `start_policy` from the `_cov` construction and from this operator's parameters/frontmatter accordingly.

- [ ] **Step 4: Run, confirm pass** `poetry run python -m pytest tests/test_microstructure.py -k roll_spread -q` - PASS.

- [ ] **Step 5: Write help page** `RollSpread.md` (inputs 1, outputs 1, params matching the actual `_cov` constructor, tags `[spread, roll, effective spread, liquidity, microstructure]`, `see_also: [RollingCov, RollingSpread]`, cite Roll 1984). No em-dashes.

- [ ] **Step 6: Regenerate + verify** (same command as Task 1 Step 6) - PASS.

- [ ] **Step 7: Commit**

```bash
git add screamer/microstructure.py tests/test_microstructure.py \
        docs/functions_micro/RollSpread.md screamer/__init__.py screamer/data/help.json
git commit -m "feat(micro): Roll (1984) effective spread from trade prices"
```

---

### Task 4: `HawkesIntensity` (self-exciting predictor)

**Files:** Modify `screamer/microstructure.py`, `tests/test_microstructure.py`; Create `docs/functions_micro/HawkesIntensity.md`

**Interfaces:**
- Produces: `HawkesIntensity(decay=0.9, alpha=1.0, mu=0.0)` - `(x) -> intensity`, the conditional intensity of an exponential-kernel Hawkes process driven by the event-mark series `x` (e.g. trade counts, signed-flow magnitude, or a 0/1 event flag). Recursion: `lambda_t = mu + kappa_t`, `kappa_{t+1} = decay*(kappa_t + alpha*x_t)`, `kappa_0 = 0`. Causal (`lambda_t` uses `x` up to `t-1`). Inputs 1, outputs 1. `decay` in `(0, 1)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_microstructure.py`

```python
def test_hawkes_intensity_hand_recursion_and_stream_equals_batch():
    from screamer.microstructure import HawkesIntensity
    x = np.array([1.0, 0.0, 0.0, 2.0, 0.0])
    # lam0=0 ; lam1=0.9*(0+1)=0.9 ; lam2=0.9*0.9=0.81 ; lam3=0.9*0.81=0.729 ;
    # lam4=0.9*(0.729+2)=2.4561
    batch = HawkesIntensity(decay=0.9, alpha=1.0, mu=0.0)(x)
    np.testing.assert_allclose(batch, [0.0, 0.9, 0.81, 0.729, 2.4561], atol=1e-9)
    op = HawkesIntensity(decay=0.9, alpha=1.0, mu=0.0)
    stream = np.array([op(float(v)) for v in x])   # one sample at a time
    np.testing.assert_allclose(batch, stream, equal_nan=True)


def test_hawkes_nan_does_not_poison_state():
    from screamer.microstructure import HawkesIntensity
    x = np.array([1.0, np.nan, 1.0])
    out = HawkesIntensity(decay=0.5, alpha=1.0, mu=0.0)(x)
    assert np.isnan(out[1])           # NaN input -> NaN output
    assert np.isfinite(out[2])        # state recovered (not poisoned)


def test_hawkes_reset_restarts_state():
    from screamer.microstructure import HawkesIntensity
    x = [1.0, 0.5, 2.0]
    op = HawkesIntensity(decay=0.8)
    a = [op(v) for v in x]; op.reset(); b = [op(v) for v in x]
    np.testing.assert_allclose(a, b)
```

- [ ] **Step 2: Run, confirm fail** `poetry run python -m pytest tests/test_microstructure.py -k hawkes -q` - FAIL (ImportError).

- [ ] **Step 3: Implement** (add `"HawkesIntensity"` to `__all__`):

```python
class HawkesIntensity:
    """Conditional intensity of an exponential-kernel Hawkes process - a
    self-exciting model where each event raises the near-term rate of further
    events (order-flow clustering / momentum). Recursion:
    lambda_t = mu + kappa_t, kappa_{t+1} = decay*(kappa_t + alpha*x_t), kappa_0 = 0.
    Causal (lambda_t depends on x up to t-1). A NaN event mark is ignored (output
    NaN at that step, state left unchanged), so it does not poison the state. The
    per-sample update is identical for array and scalar driving, so batch==stream.
    """

    def __init__(self, decay=0.9, alpha=1.0, mu=0.0):
        """__init__(self: HawkesIntensity, decay: float = 0.9, alpha: float = 1.0, mu: float = 0.0) -> None"""
        self.decay = decay
        self.alpha = alpha
        self.mu = mu
        self._kappa = 0.0

    def reset(self):
        self._kappa = 0.0

    def _step(self, x):
        lam = self.mu + self._kappa
        if np.isnan(x):
            return np.nan                       # ignore: do not fold NaN into state
        self._kappa = self.decay * (self._kappa + self.alpha * x)
        return lam

    def __call__(self, x):
        scalar = np.ndim(x) == 0
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.array([self._step(float(v)) for v in arr])
        return out[0] if scalar else out
```

- [ ] **Step 4: Run, confirm pass** `poetry run python -m pytest tests/test_microstructure.py -k hawkes -q` - PASS.

- [ ] **Step 5: Write help page** `HawkesIntensity.md` (inputs 1, outputs 1, `parameters` = `decay` (float, default 0.9), `alpha` (float, default 1.0), `mu` (float, default 0.0); tags `[hawkes, self-exciting, intensity, order flow, clustering, microstructure]`, `see_also: [EwMean]`, cite Hawkes (1971) / Bacry-Muzy). Explain the self-excitation intuition and the exponential kernel. No em-dashes.

- [ ] **Step 6: Regenerate + verify** (same command as Task 1 Step 6). Confirm HawkesIntensity passes `tests/test_nan_start_policy_compliance.py::test_stream_matches_batch` (it is driven scalar-vs-array there).

- [ ] **Step 7: Commit**

```bash
git add screamer/microstructure.py tests/test_microstructure.py \
        docs/functions_micro/HawkesIntensity.md screamer/__init__.py screamer/data/help.json
git commit -m "feat(micro): HawkesIntensity self-exciting flow-intensity predictor"
```

---

### Task 5: `Propagator` (long-memory flow -> price predictor)

**Files:** Modify `screamer/microstructure.py`, `tests/test_microstructure.py`; Create `docs/functions_micro/Propagator.md`

**Interfaces:**
- Produces: `Propagator(window=20, g0=1.0, gamma=0.5)` - `(signed_flow) -> predicted_impact`, the Bouchaud propagator model's price impact as a decaying-kernel convolution over past signed flow: `impact_t = sum_{k=0}^{window-1} G(k)*flow_{t-k}`, `G(k) = g0*(k+1)^(-gamma)`. Warmup (fewer than `window` samples seen) returns NaN. Inputs 1, outputs 1.

- [ ] **Step 1: Write the failing tests** in `tests/test_microstructure.py`

```python
def test_propagator_kernel_convolution_and_warmup():
    from screamer.microstructure import Propagator
    flow = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    out = Propagator(window=3, g0=1.0, gamma=0.5)(flow)
    # G = [1, 2^-0.5, 3^-0.5] = [1, 0.70711, 0.57735]
    # warmup: t=0,1 -> NaN ; t=2: G0*f2+G1*f1+G2*f0 = 0.57735 ;
    # t=3: G0*f3+G1*f2+G2*f1 = 1.0 ; t=4: G2*f2? = G0*f4+G1*f3+G2*f2 = 0.70711 ;
    # t=5: G0*f5+G1*f4+G2*f3 = 0.57735
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [0.57735, 1.0, 0.70711, 0.57735], atol=1e-4)


def test_propagator_stream_equals_batch():
    from screamer.microstructure import Propagator
    rng = np.random.default_rng(3); flow = rng.normal(size=60)
    batch = Propagator(window=5)(flow)
    op = Propagator(window=5); stream = np.array([op(float(v)) for v in flow])
    np.testing.assert_allclose(batch, stream, equal_nan=True)


def test_propagator_reset_clears_buffer():
    from screamer.microstructure import Propagator
    flow = [1.0, 2.0, 3.0, 4.0]
    op = Propagator(window=2)
    a = [op(v) for v in flow]; op.reset(); b = [op(v) for v in flow]
    np.testing.assert_allclose(a, b, equal_nan=True)
```

- [ ] **Step 2: Run, confirm fail** `poetry run python -m pytest tests/test_microstructure.py -k propagator -q` - FAIL (ImportError).

- [ ] **Step 3: Implement** (add `"Propagator"` to `__all__`):

```python
class Propagator:
    """Bouchaud-Gefen-Potters-Wyart (2004) propagator model: price impact as a
    decaying-kernel convolution over past signed order flow,
    impact_t = sum_k G(k)*flow_{t-k} with G(k) = g0*(k+1)^(-gamma) over a fixed
    window. Captures that flow moves price with a memory that decays but does not
    vanish at once. Warmup (fewer than `window` samples) is NaN. The per-sample
    update keeps a fixed-length buffer, identical for array and scalar driving, so
    batch==stream.
    """

    def __init__(self, window=20, g0=1.0, gamma=0.5):
        """__init__(self: Propagator, window: int = 20, g0: float = 1.0, gamma: float = 0.5) -> None"""
        self.window = window
        self._g = g0 * (np.arange(window) + 1.0) ** (-gamma)
        self._buf = []

    def reset(self):
        self._buf = []

    def _step(self, x):
        self._buf.append(x)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) < self.window:
            return np.nan                       # warmup
        # buf[-1] is x_t (k=0), buf[-1-k] is x_{t-k}
        return float(sum(self._g[k] * self._buf[-1 - k] for k in range(self.window)))

    def __call__(self, flow):
        scalar = np.ndim(flow) == 0
        arr = np.atleast_1d(np.asarray(flow, dtype=float))
        out = np.array([self._step(float(v)) for v in arr])
        return out[0] if scalar else out
```

- [ ] **Step 4: Run, confirm pass** `poetry run python -m pytest tests/test_microstructure.py -k propagator -q` - PASS.

- [ ] **Step 5: Write help page** `Propagator.md` (inputs 1, outputs 1, `parameters` = `window` (int, default 20, min 2), `g0` (float, default 1.0), `gamma` (float, default 0.5); tags `[propagator, price impact, order flow, long memory, bouchaud, microstructure]`, `see_also: [RollingKyleLambda, HawkesIntensity]`, cite Bouchaud, Gefen, Potters, Wyart 2004). Explain the decaying-kernel-over-past-flow intuition. No em-dashes.

- [ ] **Step 6: Regenerate + verify** (same command as Task 1 Step 6). Confirm Propagator passes the stream-vs-batch compliance test.

- [ ] **Step 7: Commit**

```bash
git add screamer/microstructure.py tests/test_microstructure.py \
        docs/functions_micro/Propagator.md screamer/__init__.py screamer/data/help.json
git commit -m "feat(micro): Bouchaud propagator flow-to-price impact predictor"
```

---

## Deferred to a later tranche

- `MicroPrice` (Stoikov 2018) - needs the proper conditional-expectation / G-function treatment and explicit bid/ask/imbalance inputs.
- `VPIN` (volume-clock resampling) and `PIN` (structural EM fit).
- L2 order-book models (queue-reactive, Cont-Stoikov-Talreja) - require level-2 data.
- A C++ node-core port of `HawkesIntensity` / `Propagator` for performance and Pipeline participation (the Python versions here are the validated reference).

## Self-Review

**Spec coverage:** Implements the deferred flow toolkit from the tranche-1a spec's "Deferred" section: signing (`LeeReadySign`, `BulkVolumeClassifier`), imbalance (`RollingOrderImbalance`), spread (`RollSpread`), and the two predictor centerpieces (`HawkesIntensity`, `Propagator`). `MicroPrice`/`VPIN`/`PIN`/L2 remain deferred with reasons. No spec item silently dropped.

**Placeholder scan:** No TBD/TODO; every code and command step is concrete, and the numeric test values were verified against a prototype run (Hawkes `[0, 0.9, 0.81, 0.729, 2.4561]`; Propagator `[.., 0.57735, 1.0, 0.70711, 0.57735]`; BVC CDF `0.0228/0.5/0.9772`; Roll spread positive on a bounce).

**Type consistency:** Inputs are `RollingOrderImbalance(signed_flow)`, `LeeReadySign(price, mid)`, `BulkVolumeClassifier(return_)`, `RollSpread(price)`, `HawkesIntensity(x)`, `Propagator(signed_flow)`. Every parameterized class carries the pybind-style `__init__` docstring, exposes `reset()`, and honors `nan_policy: ignore`. Windowed composition ops use `window_size`+`start_policy`; the novel ops use their documented `decay/alpha/mu` and `window/g0/gamma`.
