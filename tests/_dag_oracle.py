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
