// Symbolic graph node for the define-then-bind Pipeline API. A Node is a lazy
// placeholder in a computation DAG: `Input(name)` is a source, and passing a
// Node into an op factory (see the node branch in runtime.ts `wrapOp`) defers
// the call and returns a functor Node instead of computing eagerly.
export class Node {
  readonly isNode = true as const;
  constructor(public op: unknown, public inputs: Node[]) {}
}

// A named input placeholder. Feeds are bound at call time (`pipeline(feeds)`).
export function Input(name: string): Node {
  return new Node({ input: name }, []);
}

export function isNode(x: unknown): x is Node {
  return !!x && (x as any).isNode === true;
}
