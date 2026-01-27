"""
Aria Daemon - Background Service

Runs Aria as a persistent background service with:
- REST API for external control
- WebSocket for real-time updates
- Ambient intelligence loop
- Scheduled task execution
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import ARIA_HOME, DATA_PATH, DAEMON_PORT, DAEMON_HOST

logger = logging.getLogger(__name__)

# Daemon state
PID_FILE = ARIA_HOME / "daemon.pid"
LOG_FILE = ARIA_HOME / "daemon.log"
SOCKET_FILE = ARIA_HOME / "daemon.sock"

# Global instances - stored in builtins to survive __main__ vs module imports
import builtins
_DAEMON_KEY = "_aria_daemon_instance"

def _get_global_daemon():
    """Get daemon from builtins to survive re-imports."""
    return getattr(builtins, _DAEMON_KEY, None)

def _set_global_daemon(instance):
    """Set daemon in builtins to survive re-imports."""
    setattr(builtins, _DAEMON_KEY, instance)


class AriaDaemon:
    """
    Main daemon controller.

    Manages:
    - FastAPI server for REST/WebSocket API
    - Ambient intelligence loop
    - Scheduler for cron jobs
    - Notification delivery
    - Telegram bridge
    """

    def __init__(self, port: int = None):
        self.port = port or DAEMON_PORT
        self.app = self._create_app()
        self.mcp_server = None
        self.ambient = None
        self.scheduler = None
        self.telegram = None
        self.running = False
        self._websocket_clients: list[WebSocket] = []
        # Set global instance so get_daemon() returns this instance
        _set_global_daemon(self)

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Aria daemon starting...")
            await self._startup()
            yield
            # Shutdown
            logger.info("Aria daemon stopping...")
            await self._shutdown()

        app = FastAPI(
            title="Aria Daemon API",
            version="4.0.0",
            lifespan=lifespan
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Import and include routers
        from .daemon_api import router
        app.include_router(router)

        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._websocket_clients.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming WebSocket messages
                    response = await self._handle_ws_message(data)
                    await websocket.send_json(response)
            except WebSocketDisconnect:
                self._websocket_clients.remove(websocket)

        return app

    async def _startup(self):
        """Initialize daemon components."""
        # Write PID file
        PID_FILE.write_text(str(os.getpid()))

        # Lazy import MCP server
        try:
            from .mcp_server import AriaMCPServer
            self.mcp_server = AriaMCPServer()
        except Exception as e:
            logger.warning(f"MCP server not available: {e}")

        # Start ambient system
        try:
            from .ambient import get_ambient_system
            self.ambient = get_ambient_system()
            await self.ambient.start()
            logger.info("Ambient system started")
        except Exception as e:
            logger.warning(f"Ambient system not available: {e}")

        # Start scheduler
        try:
            from .scheduler import get_scheduler
            self.scheduler = get_scheduler()
            await self.scheduler.start()
            logger.info("Scheduler started")
        except Exception as e:
            logger.warning(f"Scheduler not available: {e}")

        # Start Telegram bridge if configured
        try:
            from .config import TELEGRAM_ENABLED
            if TELEGRAM_ENABLED:
                from .bridges import get_telegram_bridge
                self.telegram = get_telegram_bridge()
                await self.telegram.start()
                logger.info("Telegram bridge started")

                # Register Telegram as push notification handler
                if self.ambient:
                    from .ambient import DeliveryChannel
                    delivery_engine = self.ambient._delivery_engine

                    async def telegram_notification_handler(action):
                        """Send notifications via Telegram."""
                        try:
                            title = action.message[:50] if action.message else "Aria"
                            body = action.message or ""
                            await self.telegram.send_message(f"*{title}*\n\n{body}")
                            return True
                        except Exception as e:
                            logger.error(f"Telegram notification failed: {e}")
                            return False

                    delivery_engine.register_handler(
                        DeliveryChannel.PUSH_NOTIFICATION,
                        telegram_notification_handler
                    )
                    logger.info("Telegram registered as push notification handler")
        except Exception as e:
            logger.warning(f"Telegram bridge not available: {e}")

        self.running = True
        logger.info(f"Aria daemon running on port {self.port}")

    async def _shutdown(self):
        """Cleanup on shutdown."""
        self.running = False

        # Stop Telegram bridge
        if self.telegram:
            try:
                await self.telegram.stop()
            except Exception as e:
                logger.error(f"Error stopping Telegram: {e}")

        # Stop scheduler
        if self.scheduler:
            try:
                await self.scheduler.stop()
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")

        # Stop ambient system
        if self.ambient:
            try:
                await self.ambient.stop()
            except Exception as e:
                logger.error(f"Error stopping ambient: {e}")

        # Remove PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

    async def _handle_ws_message(self, data: str) -> dict:
        """Handle WebSocket message."""
        import json
        try:
            msg = json.loads(data)
            action = msg.get("action", "")

            if action == "ping":
                return {"status": "pong"}
            elif action == "status":
                return self.get_status()
            elif action == "briefing":
                if self.ambient:
                    briefing = self.ambient.get_briefing()
                    return {"briefing": briefing}
                return {"error": "Ambient system not running"}
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}

    async def broadcast(self, message: dict):
        """Broadcast message to all WebSocket clients."""
        for client in self._websocket_clients:
            try:
                await client.send_json(message)
            except Exception:
                pass

    def get_status(self) -> dict:
        """Get daemon status."""
        return {
            "running": self.running,
            "pid": os.getpid(),
            "port": self.port,
            "ambient_running": self.ambient.is_running if self.ambient else False,
            "scheduler_running": self.scheduler.is_running if self.scheduler else False,
            "telegram_running": self.telegram.is_running if self.telegram else False,
            "websocket_clients": len(self._websocket_clients),
        }

    def run(self):
        """Run the daemon."""
        uvicorn.run(
            self.app,
            host=DAEMON_HOST,
            port=self.port,
            log_level="info",
        )


def get_daemon() -> AriaDaemon:
    """Get or create daemon instance."""
    daemon = _get_global_daemon()
    if daemon is None:
        daemon = AriaDaemon()
    return daemon


def is_daemon_running() -> bool:
    """Check if daemon is already running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        return True
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink()
        return False


def start_daemon(foreground: bool = False):
    """Start the daemon."""
    if is_daemon_running():
        print("Aria daemon is already running")
        return

    if foreground:
        # Run in foreground (for debugging)
        daemon = get_daemon()
        daemon.run()
    else:
        # Daemonize
        import subprocess
        subprocess.Popen(
            [sys.executable, "-m", "aria.daemon", "--foreground"],
            stdout=open(LOG_FILE, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"Aria daemon started. Log: {LOG_FILE}")


def stop_daemon():
    """Stop the daemon."""
    if not PID_FILE.exists():
        print("Aria daemon is not running")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print("Aria daemon stopped")
    except ProcessLookupError:
        print("Daemon process not found")
        PID_FILE.unlink()


def daemon_status():
    """Get daemon status."""
    if is_daemon_running():
        pid = int(PID_FILE.read_text().strip())
        print(f"Aria daemon is running (PID: {pid})")
        print(f"API: http://{DAEMON_HOST}:{DAEMON_PORT}")
        print(f"Log: {LOG_FILE}")
    else:
        print("Aria daemon is not running")


if __name__ == "__main__":
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ]
    )

    parser = argparse.ArgumentParser(description="Aria Daemon")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    parser.add_argument("--stop", action="store_true", help="Stop the daemon")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--port", "-p", type=int, default=DAEMON_PORT, help="Port to listen on")
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
    elif args.status:
        daemon_status()
    else:
        if args.port != DAEMON_PORT:
            _daemon = AriaDaemon(port=args.port)
        start_daemon(foreground=args.foreground)
