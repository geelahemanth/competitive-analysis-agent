
import sys
from pathlib import Path

# Add the project root so this file can import the existing tools package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server import MCPServer

from tools.competitor_tools import (
    search_competitors as existing_search_tool,
    get_competitor_profile as existing_profile_tool,
)

mcp = MCPServer("competitive-intelligence")


@mcp.tool()
def search_competitors(query: str) -> str:
    """Search the competitor knowledge base using a natural-language query."""
    return str(existing_search_tool.invoke({"query": query}))


@mcp.tool()
def get_competitor_profile(competitor_name: str) -> str:
    """Retrieve the stored profile for a specific competitor."""
    return str(
        existing_profile_tool.invoke(
            {"competitor_name": competitor_name}
        )
    )


if __name__ == "__main__":
    mcp.run()
