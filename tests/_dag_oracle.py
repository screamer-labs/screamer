"""DAG-1 reference oracle — the original pure-Python executor, kept for tests."""
import numpy as np
from screamer.dag import is_node, _align_results
from screamer.dag import _as_stream  # module-level in dag.py
from screamer.streams import Stream


def run_oracle(dag, feeds):
    memo = {}

    def ev(node):
        k = id(node)
        if k in memo:
            return memo[k]
        op = node.op
        if isinstance(op, tuple) and op[0] == "input":
            result = _as_stream(feeds[op[1]])
        elif isinstance(op, tuple) and op[0] == "operator":
            fn, kwargs = op[1], op[2]
            raw_inputs = [ev(i) for i in node.inputs]
            # Operator functions (e.g. combine_latest) expect Stream objects, not
            # (keys, values) tuples; wrap tuple inputs in Streams.
            stream_inputs = [
                Stream(x[1], x[0]) if isinstance(x, tuple) and len(x) == 2
                and isinstance(x[0], np.ndarray) else x
                for x in raw_inputs
            ]
            # CamelCase class operators are config-first: Op(**cfg)(*inputs).
            # Legacy function operators are data-first: fn(*inputs, **cfg).
            if isinstance(fn, type):
                out = fn(**kwargs)(*stream_inputs)
            else:
                out = fn(*stream_inputs, **kwargs)
            # Convert Stream output back to (keys, values) for oracle consistency
            if isinstance(out, Stream):
                result = (out.index, out.values)
            else:
                result = out
        else:
            ins = [ev(i) for i in node.inputs]
            result = (ins[0][0], op(*[v for (_, v) in ins]))
        memo[k] = result
        return result

    results = [ev(o) for o in dag.outputs]
    return _align_results(results, dag.align_outputs)


def lazy_batch(dag_obj, *feeds):
    """Run a dag through the lazy iterator path (generators), reshaped to batch form.

    Each feed is a (values_arr, keys_arr) pair (values-first). Returns the same
    shape as ``dag(*feeds)``: a (values, index) pair for a single-output dag, or a
    tuple of such pairs for a multi-output dag (align_outputs=True only). Lets a
    test assert the lazy path equals the batch oracle with one call.
    """
    def _gen(v_arr, k_arr):
        return ((float(v), int(k)) for v, k in zip(v_arr, k_arr))

    n_out = len(dag_obj.outputs)
    gen_feeds = [_gen(v_arr, k_arr) for v_arr, k_arr in feeds]
    events = list(dag_obj(*gen_feeds))
    if n_out == 1:
        if not events:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
        sv = np.array([e[0] for e in events], dtype=np.float64)
        sk = np.array([e[1] for e in events], dtype=np.int64)
        return sv, sk
    if not events:
        empty = np.array([], dtype=np.float64)
        return tuple((empty, np.array([], dtype=np.int64)) for _ in range(n_out))
    sk = np.array([e[-1] for e in events], dtype=np.int64)
    return tuple((np.array([e[i] for e in events], dtype=np.float64), sk)
                 for i in range(n_out))
