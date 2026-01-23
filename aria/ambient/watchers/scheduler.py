"""
Aria Ambient Intelligence - Watcher Scheduler

Background scheduler that runs watchers at their configured intervals
and collects signals for the cortex to process.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .base import Watcher
from ..models import Signal
from ..constants import WatcherStatus

logger = logging.getLogger(__name__)


class WatcherScheduler:
    """
    Background scheduler for running watchers at their intervals.

    The scheduler:
    - Runs each watcher at its configured interval
    - Collects signals and queues them for processing
    - Handles watcher errors gracefully
    - Provides status reporting

    Usage:
        scheduler = WatcherScheduler()
        scheduler.register_watcher(news_watcher)
        scheduler.register_watcher(calendar_watcher)

        await scheduler.start()
        # ... signals are collected automatically ...
        await scheduler.stop()
    """

    def __init__(self, signal_callback: Callable[[List[Signal]], None] = None):
        """
        Initialize the scheduler.

        Args:
            signal_callback: Optional callback to receive signals as they're collected.
                            Called with List[Signal] after each watcher observation.
        """
        self._watchers: Dict[str, Watcher] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._signal_buffer: List[Signal] = []
        self._signal_callback = signal_callback
        self._running = False
        self._started_at: Optional[str] = None
        self._total_signals_collected = 0
        self._lock = asyncio.Lock()

    # =========================================================================
    # WATCHER REGISTRATION
    # =========================================================================

    def register_watcher(self, watcher: Watcher) -> bool:
        """
        Register a watcher with the scheduler.

        Args:
            watcher: Watcher instance to register

        Returns:
            True if registered successfully
        """
        if watcher.name in self._watchers:
            logger.warning(f"Watcher {watcher.name} already registered, replacing")

        self._watchers[watcher.name] = watcher
        logger.info(f"Registered watcher: {watcher.name}")

        # If scheduler is running, start the watcher task
        if self._running and watcher.is_enabled:
            self._start_watcher_task(watcher)

        return True

    def unregister_watcher(self, name: str) -> bool:
        """
        Unregister a watcher.

        Args:
            name: Name of the watcher to unregister

        Returns:
            True if successfully unregistered
        """
        if name not in self._watchers:
            logger.warning(f"Watcher {name} not found")
            return False

        # Stop the task if running
        if name in self._tasks:
            self._tasks[name].cancel()
            del self._tasks[name]

        del self._watchers[name]
        logger.info(f"Unregistered watcher: {name}")
        return True

    def get_watcher(self, name: str) -> Optional[Watcher]:
        """Get a watcher by name."""
        return self._watchers.get(name)

    def list_watchers(self) -> List[str]:
        """List all registered watcher names."""
        return list(self._watchers.keys())

    # =========================================================================
    # SCHEDULER LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """
        Start the scheduler.

        Begins running all enabled watchers at their configured intervals.
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._started_at = datetime.now().isoformat()
        logger.info(f"Starting watcher scheduler with {len(self._watchers)} watchers")

        # Start a task for each enabled watcher
        for watcher in self._watchers.values():
            if watcher.is_enabled:
                self._start_watcher_task(watcher)

    async def stop(self) -> None:
        """
        Stop the scheduler.

        Gracefully stops all watcher tasks.
        """
        if not self._running:
            return

        self._running = False
        logger.info("Stopping watcher scheduler")

        # Cancel all tasks
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._tasks.clear()
        logger.info("Watcher scheduler stopped")

    def _start_watcher_task(self, watcher: Watcher) -> None:
        """Start the background task for a watcher."""
        if watcher.name in self._tasks:
            # Task already exists
            return

        task = asyncio.create_task(
            self._run_watcher(watcher),
            name=f"watcher_{watcher.name}"
        )
        self._tasks[watcher.name] = task
        logger.debug(f"Started task for watcher: {watcher.name}")

    async def _run_watcher(self, watcher: Watcher) -> None:
        """
        Run a watcher in a loop.

        This method runs until the scheduler is stopped or the task is cancelled.
        """
        logger.info(f"Starting watcher loop: {watcher.name} (interval: {watcher.get_config().check_interval}s)")

        while self._running and watcher.is_enabled:
            try:
                # Observe and collect signals
                signals = await watcher.safe_observe()

                if signals:
                    await self._handle_signals(signals)

                # Wait for next interval
                await asyncio.sleep(watcher.get_config().check_interval)

            except asyncio.CancelledError:
                logger.debug(f"Watcher {watcher.name} task cancelled")
                break

            except Exception as e:
                logger.error(f"Error in watcher {watcher.name} loop: {e}")
                # Wait before retrying
                await asyncio.sleep(60)

        logger.info(f"Watcher loop ended: {watcher.name}")

    async def _handle_signals(self, signals: List[Signal]) -> None:
        """Handle signals collected from a watcher."""
        async with self._lock:
            self._signal_buffer.extend(signals)
            self._total_signals_collected += len(signals)

        # Call the callback if provided
        if self._signal_callback:
            try:
                self._signal_callback(signals)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    # =========================================================================
    # SIGNAL RETRIEVAL
    # =========================================================================

    async def get_signals(self, clear: bool = True) -> List[Signal]:
        """
        Get all collected signals.

        Args:
            clear: If True, clears the buffer after retrieving

        Returns:
            List of collected signals
        """
        async with self._lock:
            signals = list(self._signal_buffer)
            if clear:
                self._signal_buffer.clear()
            return signals

    async def get_signal_count(self) -> int:
        """Get the number of signals in the buffer."""
        async with self._lock:
            return len(self._signal_buffer)

    # =========================================================================
    # MANUAL TRIGGERS
    # =========================================================================

    async def run_watcher_now(self, name: str) -> List[Signal]:
        """
        Manually trigger a watcher observation.

        Args:
            name: Name of the watcher to run

        Returns:
            List of signals collected
        """
        watcher = self._watchers.get(name)
        if not watcher:
            logger.warning(f"Watcher not found: {name}")
            return []

        signals = await watcher.safe_observe()
        if signals:
            await self._handle_signals(signals)

        return signals

    async def run_all_now(self) -> List[Signal]:
        """
        Manually trigger all watchers.

        Returns:
            List of all signals collected
        """
        all_signals = []

        for watcher in self._watchers.values():
            if watcher.is_enabled:
                signals = await watcher.safe_observe()
                if signals:
                    all_signals.extend(signals)

        if all_signals:
            await self._handle_signals(all_signals)

        return all_signals

    # =========================================================================
    # WATCHER CONTROL
    # =========================================================================

    def enable_watcher(self, name: str) -> bool:
        """Enable a watcher and start its task if scheduler is running."""
        watcher = self._watchers.get(name)
        if not watcher:
            return False

        watcher.enable()

        if self._running and name not in self._tasks:
            self._start_watcher_task(watcher)

        return True

    def disable_watcher(self, name: str) -> bool:
        """Disable a watcher and stop its task."""
        watcher = self._watchers.get(name)
        if not watcher:
            return False

        watcher.disable()

        # Cancel the task if running
        if name in self._tasks:
            self._tasks[name].cancel()
            del self._tasks[name]

        return True

    def pause_watcher(self, name: str) -> bool:
        """Pause a watcher (keeps task but skips observations)."""
        watcher = self._watchers.get(name)
        if not watcher:
            return False

        watcher.pause()
        return True

    def resume_watcher(self, name: str) -> bool:
        """Resume a paused watcher."""
        watcher = self._watchers.get(name)
        if not watcher:
            return False

        watcher.resume()
        return True

    # =========================================================================
    # STATUS REPORTING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status and statistics.

        Returns:
            Dictionary with status information
        """
        watcher_statuses = {}
        for name, watcher in self._watchers.items():
            status = watcher.get_status()
            status["task_running"] = name in self._tasks and not self._tasks[name].done()
            watcher_statuses[name] = status

        return {
            "running": self._running,
            "started_at": self._started_at,
            "total_watchers": len(self._watchers),
            "active_tasks": len([t for t in self._tasks.values() if not t.done()]),
            "signals_in_buffer": len(self._signal_buffer),
            "total_signals_collected": self._total_signals_collected,
            "watchers": watcher_statuses,
        }

    def get_watcher_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific watcher."""
        watcher = self._watchers.get(name)
        if not watcher:
            return None

        status = watcher.get_status()
        status["task_running"] = name in self._tasks and not self._tasks[name].done()
        return status

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running
