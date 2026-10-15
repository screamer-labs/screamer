"""An MCP server that lets coding agents read screamer's documentation.

It serves the documentation of the installed screamer version over the Model
Context Protocol, on stdio. It needs the optional ``mcp`` package::

    pip install "screamer[mcp]"
    screamer-mcp                    # or: python -m screamer.mcp

To use it from Claude Code, run ``claude mcp add screamer -- screamer-mcp``.
Other clients take the command ``screamer-mcp`` in their MCP server settings.

The tools wrap :mod:`screamer.docs`: ``list_topics``, ``search_functions``,
``get_function``, and ``get_guide``. The agent guide is also sent as the
server's instructions, so a connected agent reads it before using the tools.
"""
import sys

from screamer import docs

INSTRUCTIONS = (
    "This server documents the screamer library installed on this machine. "
    "Before writing a calculation by hand, call search_functions to find the "
    "screamer function for it, and get_function for its parameters. The "
    "command-line examples in the guide below map to these tools: search is "
    "search_functions, show is get_function, topics is list_topics.\n\n"
    + docs.guide()
)


def list_topics() -> str:
    """List screamer's topic groups, each topic's slug, and the functions in it."""
    return docs._format_topics(docs.topics())


def search_functions(query: str, topic: str | None = None, limit: int = 10) -> str:
    """Find screamer functions by keywords, best match first.

    query: keywords, such as "order flow imbalance" or "exponential moving average".
    topic: optional topic slug from list_topics, to search only that topic.
    limit: the most results to return.
    """
    return docs._format_search(docs.search(query, topic=topic, limit=limit))


def get_function(name: str) -> str:
    """The full documentation of one screamer function, in markdown.

    It gives the call signature, each parameter's type, default, allowed values,
    and meaning, the number of inputs and outputs, the NaN policy, related
    functions, a description, examples, and a link to the online page.
    name: the function name, such as "RollingStd".
    """
    try:
        return docs._format_show(docs.show(name))
    except KeyError as exc:
        return exc.args[0]


def get_guide() -> str:
    """How to use screamer: the calling conventions, causality, streams,
    pipelines, backtests, significance tests, and the JavaScript package."""
    return docs.guide()


TOOLS = [list_topics, search_functions, get_function, get_guide]


def build_server():
    """The MCP server with the documentation tools registered."""
    try:
        from mcp.server.mcpserver import MCPServer          # mcp 2.x
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer   # mcp 1.x
        except ImportError:
            raise ImportError(
                "the screamer MCP server needs the mcp package: "
                'pip install "screamer[mcp]"') from None
    import inspect
    import screamer
    kwargs = {"name": "screamer", "instructions": INSTRUCTIONS}
    if "version" in inspect.signature(MCPServer.__init__).parameters:   # mcp 2.x
        kwargs["version"] = screamer.__version__
    server = MCPServer(**kwargs)
    for fn in TOOLS:
        server.tool()(fn)
    return server


def main() -> int:
    try:
        server = build_server()
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
