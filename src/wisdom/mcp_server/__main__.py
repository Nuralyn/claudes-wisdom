"""Entry point: python -m wisdom.mcp_server"""

from wisdom.mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run()
