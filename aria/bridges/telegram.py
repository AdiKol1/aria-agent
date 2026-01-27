"""
Aria Telegram Bridge

Access Aria from Telegram with full functionality.

Features:
- Natural language processing
- Ambient commands (worlds, entities, goals)
- Briefings on demand
- Receive insights as they happen
- Voice message support (future)
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from ..config import DATA_PATH, TELEGRAM_BOT_TOKEN

logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_CONFIG_FILE = DATA_PATH / "telegram_config.json"


class TelegramBridge:
    """
    Telegram bot that bridges to Aria.

    Usage:
        bridge = TelegramBridge()
        await bridge.start()
    """

    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN or ""
        self.allowed_user_ids: List[int] = []  # Security: only allowed users
        self.app: Optional[Application] = None
        self._running = False

        self._load_config()

    def _load_config(self):
        """Load config from file."""
        if TELEGRAM_CONFIG_FILE.exists():
            try:
                with open(TELEGRAM_CONFIG_FILE) as f:
                    data = json.load(f)
                    self.bot_token = data.get("bot_token", self.bot_token)
                    self.allowed_user_ids = data.get("allowed_user_ids", [])
            except Exception as e:
                logger.warning(f"Failed to load Telegram config: {e}")

    def _save_config(self):
        """Save config to file."""
        try:
            with open(TELEGRAM_CONFIG_FILE, "w") as f:
                json.dump({
                    "bot_token": self.bot_token,
                    "allowed_user_ids": self.allowed_user_ids,
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Telegram config: {e}")

    def configure(self, bot_token: str = None, user_id: int = None):
        """Configure the bridge."""
        if bot_token:
            self.bot_token = bot_token
        if user_id and user_id not in self.allowed_user_ids:
            self.allowed_user_ids.append(user_id)
        self._save_config()

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        # If no users configured, allow anyone (first-time setup)
        if not self.allowed_user_ids:
            return True
        return user_id in self.allowed_user_ids

    async def start(self):
        """Start the Telegram bot."""
        if not self.bot_token:
            logger.error("Telegram bot token not configured")
            return

        self.app = Application.builder().token(self.bot_token).build()

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("briefing", self._cmd_briefing))
        self.app.add_handler(CommandHandler("worlds", self._cmd_worlds))
        self.app.add_handler(CommandHandler("insights", self._cmd_insights))
        self.app.add_handler(CommandHandler("remember", self._cmd_remember))
        self.app.add_handler(CommandHandler("recall", self._cmd_recall))
        self.app.add_handler(CommandHandler("notify", self._cmd_notify))
        self.app.add_handler(CommandHandler("schedule", self._cmd_schedule))

        # Handle all other messages
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))

        self._running = True
        logger.info("Telegram bridge starting...")

        # Start polling
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bridge running")

    async def stop(self):
        """Stop the Telegram bot."""
        if self.app:
            self._running = False
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bridge stopped")

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user

        # First-time setup: register user
        if not self.allowed_user_ids:
            self.allowed_user_ids.append(user.id)
            self._save_config()
            await update.message.reply_text(
                f"Hi {user.first_name}! I'm Aria, your AI assistant.\n\n"
                f"You're now registered as an authorized user.\n\n"
                f"Try these commands:\n"
                f"/briefing - Get your daily briefing\n"
                f"/worlds - List your monitored domains\n"
                f"/insights - Get pending insights\n"
                f"/help - See all commands\n\n"
                f"Or just send me a message!"
            )
        elif self._is_authorized(user.id):
            await update.message.reply_text(
                f"Welcome back, {user.first_name}!\n"
                f"Send /help to see available commands."
            )
        else:
            await update.message.reply_text(
                "Sorry, you're not authorized to use this bot."
            )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self._is_authorized(update.effective_user.id):
            return

        help_text = """
*Aria Commands*

*Information*
/status - Daemon and system status
/briefing - Get your current briefing
/worlds - List monitored worlds
/insights - Get pending insights

*Memory*
/remember <fact> - Store a fact
/recall <query> - Search memory

*Scheduling*
/schedule - List scheduled tasks

*Notifications*
/notify <message> - Send yourself a notification

*Natural Language*
Just send any message to interact naturally!
Examples:
- "What's going on?"
- "Track Compass as a competitor"
- "My goal is to close 3 deals"
- "Schedule morning briefing at 8am"
        """
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            from ..daemon import is_daemon_running, get_daemon

            if is_daemon_running():
                daemon = get_daemon()
                status = daemon.get_status()

                text = (
                    f"*Aria Status*\n\n"
                    f"Daemon: Running\n"
                    f"Port: {status['port']}\n"
                    f"Ambient: {'Running' if status['ambient_running'] else 'Stopped'}\n"
                    f"Scheduler: {'Running' if status['scheduler_running'] else 'Stopped'}\n"
                    f"WebSocket clients: {status['websocket_clients']}"
                )
            else:
                text = "*Aria Status*\n\nDaemon: Not running"

            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error getting status: {e}")

    async def _cmd_briefing(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /briefing command."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            from ..ambient import get_ambient_system

            ambient = get_ambient_system()
            briefing = ambient.get_briefing(format="text")

            if not briefing:
                briefing = "No updates available at the moment."

            # Split if too long for Telegram (4096 char limit)
            if len(briefing) > 4000:
                parts = [briefing[i:i+4000] for i in range(0, len(briefing), 4000)]
                for i, part in enumerate(parts):
                    await update.message.reply_text(
                        f"*Briefing ({i+1}/{len(parts)})*\n\n{part}",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(
                    f"*Briefing*\n\n{briefing}",
                    parse_mode="Markdown"
                )
        except Exception as e:
            await update.message.reply_text(f"Error getting briefing: {e}")

    async def _cmd_worlds(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /worlds command."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            from ..ambient import get_ambient_system

            ambient = get_ambient_system()
            worlds = ambient.list_worlds()

            if worlds:
                lines = ["*Your Worlds*\n"]
                for world in worlds:
                    lines.append(f"*{world.name}*")
                    lines.append(f"   {world.description}")
                    lines.append(f"   Goals: {len(world.goals)}, Entities: {len(world.entities)}")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            else:
                await update.message.reply_text(
                    "No worlds configured yet.\n\n"
                    "Create one by saying something like:\n"
                    "\"I work in real estate\""
                )
        except Exception as e:
            await update.message.reply_text(f"Error listing worlds: {e}")

    async def _cmd_insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /insights command."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            from ..ambient import get_ambient_system

            ambient = get_ambient_system()
            insights = ambient.get_pending_insights(limit=5)

            if insights:
                lines = ["*Pending Insights*\n"]
                for insight in insights:
                    priority = insight.priority.value if hasattr(insight.priority, 'value') else str(insight.priority)
                    priority_emoji = {
                        "critical": "!",
                        "high": "*",
                        "medium": "-",
                        "low": ".",
                    }.get(priority.lower(), "-")
                    lines.append(f"{priority_emoji} *{insight.title}*")
                    if hasattr(insight, 'suggested_action') and insight.suggested_action:
                        lines.append(f"   -> {insight.suggested_action}")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            else:
                await update.message.reply_text("No pending insights.")
        except Exception as e:
            await update.message.reply_text(f"Error getting insights: {e}")

    async def _cmd_remember(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remember command."""
        if not self._is_authorized(update.effective_user.id):
            return

        fact = " ".join(context.args) if context.args else ""

        if not fact:
            await update.message.reply_text("Usage: /remember <fact to remember>")
            return

        try:
            from ..memory import get_memory
            memory = get_memory()
            success = memory.remember_fact(fact, "telegram")

            if success:
                await update.message.reply_text(f"Remembered: {fact}")
            else:
                await update.message.reply_text("Failed to remember")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_recall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /recall command."""
        if not self._is_authorized(update.effective_user.id):
            return

        query = " ".join(context.args) if context.args else ""

        if not query:
            await update.message.reply_text("Usage: /recall <search query>")
            return

        try:
            from ..memory import get_memory
            memory = get_memory()
            results = memory.recall_facts(query, n_results=5)

            if results:
                lines = ["*Memories found:*\n"]
                for r in results:
                    lines.append(f"- {r['fact']}")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            else:
                await update.message.reply_text("No memories found for that query.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_notify(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notify command (test notifications)."""
        if not self._is_authorized(update.effective_user.id):
            return

        message = " ".join(context.args) if context.args else "Test notification from Aria"

        try:
            from ..notifications import get_notifier
            notifier = get_notifier()
            success = await notifier.send("Aria", message)

            if success:
                await update.message.reply_text("Notification sent")
            else:
                await update.message.reply_text("Notification failed (check ntfy config)")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_schedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /schedule command."""
        if not self._is_authorized(update.effective_user.id):
            return

        try:
            from ..scheduler import get_scheduler
            scheduler = get_scheduler()
            tasks = scheduler.list_tasks()

            if tasks:
                lines = ["*Scheduled Tasks*\n"]
                for t in tasks:
                    status = "Enabled" if t["enabled"] else "Disabled"
                    lines.append(f"*{t['task_type']}* at {t['schedule']}")
                    lines.append(f"   Status: {status}")
                    if t.get("next_run"):
                        lines.append(f"   Next: {t['next_run'][:16]}")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            else:
                await update.message.reply_text("No scheduled tasks.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    # =========================================================================
    # MESSAGE HANDLER
    # =========================================================================

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle natural language messages."""
        if not self._is_authorized(update.effective_user.id):
            return

        text = update.message.text

        # Send typing indicator
        await update.message.chat.send_action("typing")

        try:
            # Process through MCP server (same as Claude Code)
            from ..mcp_server import AriaMCPServer
            server = AriaMCPServer()

            # Try ambient command first
            result = server._process_ambient_command(text)

            # If not an ambient command, try general processing
            if "didn't understand" in result.lower():
                try:
                    from ..agent import get_agent
                    agent = get_agent()
                    result = agent.process_request(text, include_screen=False)
                except Exception:
                    # Keep the ambient result if agent fails
                    pass

            # Send response
            await update.message.reply_text(result)

        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing that request."
            )

    # =========================================================================
    # PROACTIVE MESSAGING
    # =========================================================================

    async def send_insight(self, insight, user_id: int = None):
        """Send an insight to user(s)."""
        if not self.app or not self._running:
            return

        # Send to specified user or all allowed users
        recipients = [user_id] if user_id else self.allowed_user_ids

        for uid in recipients:
            try:
                priority = insight.priority.value if hasattr(insight.priority, 'value') else str(insight.priority)
                priority_emoji = {
                    "critical": "!",
                    "high": "*",
                    "medium": "-",
                    "low": ".",
                }.get(priority.lower(), "-")

                text = (
                    f"{priority_emoji} *New Insight*\n\n"
                    f"*{insight.title}*\n\n"
                    f"{insight.body}\n\n"
                )
                if hasattr(insight, 'suggested_action') and insight.suggested_action:
                    text += f"_Suggested: {insight.suggested_action}_"

                await self.app.bot.send_message(uid, text, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Failed to send insight to {uid}: {e}")

    async def send_briefing(self, briefing: str, user_id: int = None):
        """Send a briefing to user(s)."""
        if not self.app or not self._running:
            return

        recipients = [user_id] if user_id else self.allowed_user_ids

        for uid in recipients:
            try:
                await self.app.bot.send_message(
                    uid,
                    f"*Daily Briefing*\n\n{briefing}",
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Failed to send briefing to {uid}: {e}")

    async def send_message(self, text: str, user_id: int = None):
        """Send a message to user(s)."""
        if not self.app or not self._running:
            return

        recipients = [user_id] if user_id else self.allowed_user_ids

        for uid in recipients:
            try:
                await self.app.bot.send_message(uid, text)
            except Exception as e:
                logger.error(f"Failed to send message to {uid}: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "configured": bool(self.bot_token),
            "authorized_users": len(self.allowed_user_ids),
        }
