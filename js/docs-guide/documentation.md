---
title: Built-in documentation
---

# Built-in documentation

The documentation of every function ships with the package, in JavaScript names: TypeScript
signatures, camelCase options with their defaults and allowed values, inputs and outputs, NaN
policy, and related functions. It matches the installed version, which makes it a reliable
source for people and for coding agents that write screamer code.

## From a shell

The package installs a `screamer-docs` command:

```bash
npx screamer-docs topics                          # the topic groups and their functions
npx screamer-docs search "order flow imbalance"   # find functions by keywords
npx screamer-docs search volatility --topic statistics --limit 5
npx screamer-docs show RollingRSI                 # the full documentation of one function
npx screamer-docs guide                           # how to use screamer, written for coding agents
```

Add `--json` to any command for structured output.

## From code

The same data is a separate entry point, `@screamer-labs/screamer/docs`, so a bundle that
imports only the operators does not include it. It needs no `ready()` call.

```js
import { search, show, topics, guide } from "@screamer-labs/screamer/docs";

search("order flow imbalance", { limit: 3 }).map((r) => r.name);
// [ 'RollingOrderImbalance', 'ContOFI', 'OFI' ]

show("RollingRSI").signatures;
// [ 'RollingRSI({ windowSize: 14, method: "wilder", startPolicy: "strict" })' ]

topics().map((g) => g.slug);   // [ 'statistics', 'smoothing-filters', 'indicators', ... ]
guide();                       // the agent guide as Markdown text
```

`show(name)` returns the parameters, inputs, outputs, examples, related functions, and a link
to the full reference page.

[Using screamer with AI agents](https://screamer.readthedocs.io/en/latest/agents.html) describes
the MCP server and the other ways to give an agent this documentation.
