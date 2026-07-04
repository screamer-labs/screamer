"""DAG-1 reference oracle — the original pure-Python executor, kept for tests."""
import numpy as np
from screamer.dag import is_node, _align_results
from screamer.dag import _as_stream  # module-level in dag.py


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
            result = fn(*[ev(i) for i in node.inputs], **kwargs)
        else:
            ins = [ev(i) for i in node.inputs]
            result = (ins[0][0], op(*[v for (_, v) in ins]))
        memo[k] = result
        return result

    results = [ev(o) for o in dag.outputs]
    return _align_results(results, dag.align_outputs)
