"""
MCP Client for Aria.

Connects to external MCP servers (filesystem, fetch, git) via JSON-RPC over stdio.
This allows Aria to leverage external tools through the Model Context Protocol.
"""

import subprocess
import json
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Represents an MCP server configuration and state."""
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    process: Optional[subprocess.Popen] = None
    tools: List[Dict] = field(default_factory=list)
    resources: List[Dict] = field(default_factory=list)
    initialized: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)


class MCPClient:
    """
    Client for managing and calling MCP servers.

    This client can:
    - Start MCP server subprocesses
    - Communicate via JSON-RPC over stdio
    - Call tools on servers
    - List available tools from all servers
    """

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self._request_id = 0
        self._id_lock = threading.Lock()

    def _next_request_id(self) -> int:
        """Thread-safe request ID generation."""
        with self._id_lock:
            self._request_id += 1
            return self._request_id

    def register_server(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Register an MCP server configuration.

        Args:
            name: Unique name for this server
            command: The command to run (e.g., 'npx', 'python')
            args: Arguments for the command
            env: Optional environment variables
        """
        self.servers[name] = MCPServer(
            name=name,
            command=command,
            args=args,
            env=env,
            tools=[],
            resources=[]
        )
        logger.info(f"Registered MCP server: {name} ({command} {' '.join(args[:2])}...)")

    def start_server(self, name: str) -> bool:
        """
        Start an MCP server subprocess.

        Args:
            name: Name of the registered server to start

        Returns:
            True if server started successfully, False otherwise
        """
        if name not in self.servers:
            logger.error(f"Server {name} not registered")
            return False

        server = self.servers[name]

        with server._lock:
            # Check if already running
            if server.process is not None and server.process.poll() is None:
                logger.debug(f"Server {name} already running")
                return True

            try:
                # Build environment
                import os
                env = os.environ.copy()
                if server.env:
                    env.update(server.env)

                # Start the subprocess
                server.process = subprocess.Popen(
                    [server.command] + server.args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    env=env
                )

                logger.info(f"Started MCP server {name} (PID: {server.process.pid})")

                # Initialize the server
                if self._initialize_server(server):
                    server.initialized = True
                    return True
                else:
                    # Initialization failed, stop the process
                    self._kill_process(server)
                    return False

            except FileNotFoundError:
                logger.error(f"Command not found: {server.command}")
                return False
            except Exception as e:
                logger.error(f"Failed to start MCP server {name}: {e}")
                return False

    def _kill_process(self, server: MCPServer):
        """Safely terminate a server process."""
        if server.process:
            try:
                server.process.terminate()
                server.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.process.kill()
            except Exception:
                pass
            finally:
                server.process = None
                server.initialized = False

    def _send_request(self, server: MCPServer, method: str, params: Optional[dict] = None) -> dict:
        """
        Send JSON-RPC request to server.

        Args:
            server: The MCPServer to send the request to
            method: The JSON-RPC method name
            params: Optional parameters for the method

        Returns:
            The JSON-RPC response as a dictionary
        """
        if not server.process or server.process.poll() is not None:
            return {"error": {"code": -1, "message": "Server not running"}}

        request_id = self._next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            server.process.stdin.write(request_line)
            server.process.stdin.flush()

            # Read response
            response_line = server.process.stdout.readline()
            if not response_line:
                return {"error": {"code": -1, "message": "No response from server"}}

            return json.loads(response_line)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return {"error": {"code": -32700, "message": f"Parse error: {e}"}}
        except BrokenPipeError:
            logger.error(f"Broken pipe to server {server.name}")
            server.process = None
            server.initialized = False
            return {"error": {"code": -1, "message": "Connection lost"}}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": {"code": -1, "message": str(e)}}

    def _send_notification(self, server: MCPServer, method: str, params: Optional[dict] = None):
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            server: The MCPServer to send to
            method: The notification method name
            params: Optional parameters
        """
        if not server.process or server.process.poll() is not None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params

        try:
            notification_line = json.dumps(notification) + "\n"
            server.process.stdin.write(notification_line)
            server.process.stdin.flush()
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    def _initialize_server(self, server: MCPServer) -> bool:
        """
        Initialize MCP server and get tool list.

        Args:
            server: The MCPServer to initialize

        Returns:
            True if initialization successful, False otherwise
        """
        # Send initialize request
        response = self._send_request(server, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "aria",
                "version": "2.0.0"
            }
        })

        if "error" in response:
            logger.error(f"Initialize failed for {server.name}: {response['error']}")
            return False

        # Log server capabilities
        if "result" in response:
            result = response["result"]
            server_info = result.get("serverInfo", {})
            logger.info(f"Connected to {server_info.get('name', server.name)} v{server_info.get('version', '?')}")

            capabilities = result.get("capabilities", {})
            if capabilities.get("tools"):
                logger.debug(f"Server {server.name} supports tools")
            if capabilities.get("resources"):
                logger.debug(f"Server {server.name} supports resources")

        # Send initialized notification
        self._send_notification(server, "notifications/initialized")

        # Get available tools
        tools_response = self._send_request(server, "tools/list")
        if "result" in tools_response and "tools" in tools_response["result"]:
            server.tools = tools_response["result"]["tools"]
            logger.info(f"Server {server.name} has {len(server.tools)} tools")

        # Get available resources (optional)
        resources_response = self._send_request(server, "resources/list")
        if "result" in resources_response and "resources" in resources_response["result"]:
            server.resources = resources_response["result"]["resources"]
            logger.debug(f"Server {server.name} has {len(server.resources)} resources")

        return True

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the server to call
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool

        Returns:
            Tool result dictionary with 'content' or 'error'
        """
        if server_name not in self.servers:
            return {"error": f"Server {server_name} not found"}

        server = self.servers[server_name]

        # Start server if not running
        if not server.initialized or server.process is None or server.process.poll() is not None:
            if not self.start_server(server_name):
                return {"error": f"Failed to start server {server_name}"}

        # Call the tool
        response = self._send_request(server, "tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        if "result" in response:
            return response["result"]
        elif "error" in response:
            return {"error": response["error"]}
        return {"error": "Unknown response format"}

    def list_tools(self, server_name: Optional[str] = None) -> List[dict]:
        """
        List available tools from one or all servers.

        Args:
            server_name: Optional specific server, or None for all servers

        Returns:
            List of tool definitions with server name included
        """
        tools = []

        if server_name:
            if server_name not in self.servers:
                return []
            servers_to_check = [self.servers[server_name]]
        else:
            servers_to_check = list(self.servers.values())

        for server in servers_to_check:
            # Start server if needed
            if not server.initialized:
                self.start_server(server.name)

            for tool in server.tools or []:
                tool_copy = tool.copy()
                tool_copy["server"] = server.name
                tools.append(tool_copy)

        return tools

    def get_tool(self, server_name: str, tool_name: str) -> Optional[dict]:
        """
        Get a specific tool definition.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool

        Returns:
            Tool definition or None if not found
        """
        if server_name not in self.servers:
            return None

        server = self.servers[server_name]
        if not server.initialized:
            self.start_server(server_name)

        for tool in server.tools or []:
            if tool.get("name") == tool_name:
                return tool
        return None

    def stop_server(self, name: str):
        """
        Stop an MCP server.

        Args:
            name: Name of the server to stop
        """
        if name not in self.servers:
            return

        server = self.servers[name]
        with server._lock:
            if server.process:
                logger.info(f"Stopping MCP server {name}")
                self._kill_process(server)
                server.tools = []
                server.resources = []

    def stop_all(self):
        """Stop all MCP servers."""
        for name in list(self.servers.keys()):
            self.stop_server(name)

    def get_server_status(self, name: str) -> dict:
        """
        Get status information for a server.

        Args:
            name: Server name

        Returns:
            Status dictionary with running, initialized, tool_count
        """
        if name not in self.servers:
            return {"exists": False}

        server = self.servers[name]
        running = server.process is not None and server.process.poll() is None

        return {
            "exists": True,
            "running": running,
            "initialized": server.initialized,
            "tool_count": len(server.tools),
            "resource_count": len(server.resources),
            "pid": server.process.pid if running else None
        }

    def get_all_server_status(self) -> Dict[str, dict]:
        """Get status for all registered servers."""
        return {name: self.get_server_status(name) for name in self.servers}


# Default server configurations
DEFAULT_MCP_SERVERS = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.home())],
        "description": "File system operations in home directory",
    },
    "fetch": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "description": "HTTP fetch operations",
    },
}


# Singleton instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """
    Get the global MCP client singleton.

    Returns:
        The shared MCPClient instance with default servers registered
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
        # Register default servers
        for name, config in DEFAULT_MCP_SERVERS.items():
            _mcp_client.register_server(
                name,
                config["command"],
                config["args"],
                config.get("env")
            )
        logger.info(f"MCP client initialized with {len(DEFAULT_MCP_SERVERS)} default servers")
    return _mcp_client


def reset_mcp_client():
    """Reset the MCP client singleton (mainly for testing)."""
    global _mcp_client
    if _mcp_client:
        _mcp_client.stop_all()
    _mcp_client = None
