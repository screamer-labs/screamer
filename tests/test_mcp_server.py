"""The screamer MCP server: its tools return what screamer.docs returns, and a
client can connect to it over stdio."""
import sys

import pytest

pytest.importorskip("mcp")
anyio = pytest.importorskip("anyio")

from screamer import docs  # noqa: E402
from screamer.mcp import INSTRUCTIONS, build_server  # noqa: E402

EXPECTED = {
    "list_topics": ({}, docs._format_topics(docs.topics())),
    "search_functions": ({"query": "kalman"}, docs._format_search(docs.search("kalman"))),
    "get_function": ({"name": "RollingStd"}, docs._format_show(docs.show("RollingStd"))),
    "get_guide": ({}, docs.guide()),
}


def _text(result):
    return "".join(c.text for c in result.content)


def test_tools_match_screamer_docs():
    async def run():
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=sys.executable, args=["-m", "screamer.mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert init.instructions == INSTRUCTIONS
                names = {t.name for t in (await session.list_tools()).tools}
                assert names == set(EXPECTED)
                for name, (args, expected) in EXPECTED.items():
                    assert _text(await session.call_tool(name, args)) == expected, name
                missing = _text(await session.call_tool("get_function", {"name": "VPNI"}))
                assert "did you mean VPIN" in missing

    anyio.run(run)


def test_build_server_registers_every_tool():
    server = build_server()
    tools = anyio.run(server.list_tools)
    assert {t.name for t in tools} == set(EXPECTED)
