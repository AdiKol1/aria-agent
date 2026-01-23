"""
Aria Ambient Intelligence - Watcher Base Class

Abstract base class for all watchers. Watchers are continuous signal collectors
that monitor various data sources and produce signals for the cortex to process.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import logging

from ..models import Signal, generate_id, now_iso
from ..constants import SignalType, WatcherStatus, CHECK_INTERVALS

logger = logging.getLogger(__name__)


@dataclass
class WatcherConfig:
    """Configuration for a watcher."""
    enabled: bool = True
    check_interval: int = 300  # seconds
    max_signals_per_check: int = 50
    retry_on_error: bool = True
    max_retries: int = 3
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "check_interval": self.check_interval,
            "max_signals_per_check": self.max_signals_per_check,
            "retry_on_error": self.retry_on_error,
            "max_retries": self.max_retries,
            "custom_settings": self.custom_settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatcherConfig":
        return cls(**data)


class Watcher(ABC):
    """
    Abstract base class for all signal watchers.

    Watchers are responsible for:
    1. Monitoring a specific data source (news, calendar, social, etc.)
    2. Collecting relevant data at regular intervals
    3. Converting raw data into Signal objects
    4. Managing their own state and configuration

    Subclasses must implement:
    - observe(): Collect signals from the data source
    - _validate_config(): Validate watcher-specific configuration

    Lifecycle:
    1. __init__(): Set up the watcher
    2. configure(): Apply custom configuration
    3. start(): Begin observing (if using scheduler)
    4. observe(): Called periodically to collect signals
    5. stop(): Stop observing
    """

    # Override in subclasses
    name: str = "base"
    description: str = "Base watcher class"
    default_signal_type: SignalType = SignalType.CUSTOM

    def __init__(self, config: WatcherConfig = None):
        """
        Initialize the watcher.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or WatcherConfig()
        self._status = WatcherStatus.DISABLED
        self._last_check: Optional[str] = None
        self._last_error: Optional[str] = None
        self._error_count: int = 0
        self._total_signals: int = 0
        self._running: bool = False

        # Apply default check interval based on watcher name
        if self._config.check_interval == 300:  # Default value
            self._config.check_interval = CHECK_INTERVALS.get(
                self.name, CHECK_INTERVALS["default"]
            )

    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================

    @abstractmethod
    async def observe(self) -> List[Signal]:
        """
        Collect signals from the data source.

        This method is called periodically by the scheduler. It should:
        1. Query the data source for new information
        2. Filter out already-seen items (deduplication)
        3. Convert relevant items to Signal objects
        4. Return the list of new signals

        Returns:
            List of Signal objects collected during this observation.

        Raises:
            Exception: If data source is unavailable (will be caught by scheduler)
        """
        pass

    def _validate_config(self) -> List[str]:
        """
        Validate watcher-specific configuration.

        Override in subclasses to validate custom settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def configure(self, settings: Dict[str, Any]) -> bool:
        """
        Apply custom configuration settings.

        Args:
            settings: Dictionary of settings to apply

        Returns:
            True if configuration was valid and applied
        """
        # Update standard settings
        if "enabled" in settings:
            self._config.enabled = settings["enabled"]
        if "check_interval" in settings:
            self._config.check_interval = max(10, settings["check_interval"])
        if "max_signals_per_check" in settings:
            self._config.max_signals_per_check = settings["max_signals_per_check"]

        # Store custom settings
        custom = {k: v for k, v in settings.items()
                  if k not in {"enabled", "check_interval", "max_signals_per_check"}}
        self._config.custom_settings.update(custom)

        # Validate
        errors = self._validate_config()
        if errors:
            logger.warning(f"Configuration errors for {self.name}: {errors}")
            return False

        logger.info(f"Configured watcher: {self.name}")
        return True

    def get_config(self) -> WatcherConfig:
        """Get current configuration."""
        return self._config

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a custom setting value."""
        return self._config.custom_settings.get(key, default)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def enable(self) -> None:
        """Enable the watcher."""
        self._config.enabled = True
        if self._status == WatcherStatus.DISABLED:
            self._status = WatcherStatus.ACTIVE
        logger.info(f"Enabled watcher: {self.name}")

    def disable(self) -> None:
        """Disable the watcher."""
        self._config.enabled = False
        self._status = WatcherStatus.DISABLED
        logger.info(f"Disabled watcher: {self.name}")

    def pause(self) -> None:
        """Pause the watcher (can be resumed)."""
        self._status = WatcherStatus.PAUSED
        logger.info(f"Paused watcher: {self.name}")

    def resume(self) -> None:
        """Resume a paused watcher."""
        if self._config.enabled:
            self._status = WatcherStatus.ACTIVE
            logger.info(f"Resumed watcher: {self.name}")

    @property
    def is_enabled(self) -> bool:
        """Check if watcher is enabled."""
        return self._config.enabled

    @property
    def is_active(self) -> bool:
        """Check if watcher is actively running."""
        return self._status == WatcherStatus.ACTIVE

    @property
    def status(self) -> WatcherStatus:
        """Get current watcher status."""
        return self._status

    # =========================================================================
    # OBSERVATION WRAPPER
    # =========================================================================

    async def safe_observe(self) -> List[Signal]:
        """
        Safely observe with error handling and retry logic.

        This wrapper method:
        1. Checks if watcher is enabled and active
        2. Handles exceptions and retries
        3. Updates status and statistics
        4. Returns signals or empty list on error

        Returns:
            List of signals, or empty list if error occurred
        """
        if not self._config.enabled:
            return []

        if self._status not in {WatcherStatus.ACTIVE, WatcherStatus.PAUSED}:
            return []

        retry_count = 0
        last_error = None

        while retry_count <= (self._config.max_retries if self._config.retry_on_error else 0):
            try:
                self._status = WatcherStatus.ACTIVE

                signals = await self.observe()

                # Limit signals per check
                if len(signals) > self._config.max_signals_per_check:
                    signals = signals[:self._config.max_signals_per_check]

                # Update stats
                self._last_check = now_iso()
                self._error_count = 0
                self._total_signals += len(signals)

                logger.debug(f"{self.name} observed {len(signals)} signals")
                return signals

            except asyncio.CancelledError:
                raise  # Don't catch cancellation

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                logger.warning(
                    f"{self.name} observation error (attempt {retry_count}): {e}"
                )

                if retry_count <= self._config.max_retries and self._config.retry_on_error:
                    await asyncio.sleep(min(2 ** retry_count, 30))  # Exponential backoff
                    continue
                break

        # All retries failed
        self._status = WatcherStatus.ERROR
        self._last_error = last_error
        self._error_count += 1

        logger.error(f"{self.name} observation failed after {retry_count} attempts")
        return []

    # =========================================================================
    # SIGNAL CREATION HELPERS
    # =========================================================================

    def create_signal(
        self,
        title: str,
        content: str,
        signal_type: SignalType = None,
        url: str = None,
        raw_data: Dict[str, Any] = None,
        expires_in_seconds: int = None,
    ) -> Signal:
        """
        Create a new signal with common fields populated.

        Args:
            title: Signal title
            content: Signal content/body
            signal_type: Type of signal (defaults to watcher's default type)
            url: Source URL
            raw_data: Original data from source
            expires_in_seconds: When this signal expires

        Returns:
            New Signal object
        """
        expires_at = None
        if expires_in_seconds:
            expires_at = (
                datetime.now() +
                __import__('datetime').timedelta(seconds=expires_in_seconds)
            ).isoformat()

        return Signal(
            id=generate_id("sig"),
            source=self.name,
            type=signal_type or self.default_signal_type,
            title=title,
            content=content,
            url=url,
            timestamp=now_iso(),
            expires_at=expires_at,
            raw_data=raw_data or {},
        )

    # =========================================================================
    # STATUS REPORTING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get current watcher status and statistics.

        Returns:
            Dictionary with status information
        """
        return {
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "enabled": self._config.enabled,
            "check_interval": self._config.check_interval,
            "last_check": self._last_check,
            "last_error": self._last_error,
            "error_count": self._error_count,
            "total_signals": self._total_signals,
            "config": self._config.to_dict(),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._error_count = 0
        self._total_signals = 0
        self._last_error = None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, status={self._status.value})>"
