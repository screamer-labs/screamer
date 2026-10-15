# Using screamer with AI agents

Coding agents write better screamer code when they can look up which functions
exist and how to call them, instead of guessing or rewriting a calculation in
numpy. screamer ships its documentation in a form agents can query: a
command-line tool and Python module, an MCP server, an Agent Skill, and
machine-readable files on this site. All of them serve the same content: every
function's summary, parameters (type, default, allowed values, meaning), inputs
and outputs, NaN policy, related functions, details, and examples, plus a short
guide to the library's conventions, reproduced [at the end of this
page](#the-agent-guide).

The first three run on your machine and describe the screamer version you have
installed. The files on this site describe the latest release.

## The command line and Python

Any agent that can run shell commands can use this directly; it needs nothing
beyond screamer itself.

```bash
python -m screamer.docs topics          # topics and their functions
python -m screamer.docs search "order flow imbalance"
python -m screamer.docs show RollingStd # one function in full
python -m screamer.docs guide           # the agent guide
```

Add `--json` for structured output. From Python:

```python
from screamer import docs

docs.search("moving average", limit=5)  # list of {name, kind, short, topics}
docs.show("RollingStd")["parameters"]    # list of {name, type, default, ...}
docs.topics()                            # groups, topics, and functions
```

In a JavaScript project, the npm package has the same tool and the same
documentation in JavaScript names, with each function's TypeScript signatures
and camelCase options:

```bash
npx screamer-docs search "order flow imbalance"
npx screamer-docs show positionTest
```

```javascript
import { search, show, topics } from "@screamer-labs/screamer/docs";
show("RollingRSI").signatures;   // ['RollingRSI({ windowSize: 14, ... })']
```

The documentation is a separate entry point, so a bundle that imports only the
functions does not include it.

To point an agent at it, add a line to the instructions file it reads, such as
`AGENTS.md` or `CLAUDE.md`:

```text
This project uses the screamer library. Before writing a time-series
calculation, find the function with `python -m screamer.docs search <keywords>`
and read it with `python -m screamer.docs show <Name>`.
```

## The MCP server

The [Model Context Protocol](https://modelcontextprotocol.io) is how many agents
and editors connect to tools. screamer's MCP server is an optional extra:

```bash
pip install "screamer[mcp]"
```

It runs on stdio as `screamer-mcp` and offers four tools: `list_topics`,
`search_functions`, `get_function`, and `get_guide`. It also sends the agent
guide as its instructions, so a connected agent reads the conventions before it
uses the tools.

With Claude Code:

```bash
claude mcp add screamer -- screamer-mcp
```

Clients configured with JSON (Claude Desktop, Cursor, VS Code, and others) take
an entry like this:

```json
{
  "mcpServers": {
    "screamer": { "command": "screamer-mcp" }
  }
}
```

If screamer is installed in a virtual environment, use the full path of the
command, such as `/path/to/venv/bin/screamer-mcp`, or run the server with that
environment's Python as `/path/to/venv/bin/python -m screamer.mcp`. The server
then describes the version installed in that environment.

## An Agent Skill

Agents that support [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview),
such as Claude Code, load a skill when a task calls for it. Install the guide as
a skill for all your projects:

```bash
mkdir -p ~/.claude/skills/screamer
python -m screamer.docs skill > ~/.claude/skills/screamer/SKILL.md
```

or for one project, in its `.claude/skills/screamer/SKILL.md`. Re-run the
command after upgrading screamer.

## Files on this site

For agents that read the web, this site follows the [llms.txt](https://llmstxt.org)
convention:

- [llms.txt](https://screamer.readthedocs.io/en/latest/llms.txt) is an index of
  the guides, every function grouped by topic, and the worked examples, each
  linked to its page source in markdown.
- [llms-full.txt](https://screamer.readthedocs.io/en/latest/llms-full.txt) holds
  the agent guide and the full documentation of every function in one file.

The markdown source of any page is at `_sources/<page>.md.txt`, for example
[`_sources/functions_rolling/RollingStd.md.txt`](https://screamer.readthedocs.io/en/latest/_sources/functions_rolling/RollingStd.md.txt).

(the-agent-guide)=
## The agent guide

This is the text that `python -m screamer.docs guide`, the MCP server, and the
skill give an agent.

```{include} ../screamer/data/agent_guide.md
:heading-offset: 2
```
