"""
Aria Ambient Intelligence - Delivery Engine

Delivers insights to the user at the right time through the right channel.
Manages timing, batching, and prioritization of deliveries.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, time as dt_time
from typing import Any, Callable, Dict, List, Optional
import heapq

from ..models import Insight, PreparedAction, now_iso
from ..constants import (
    DeliveryChannel,
    InsightPriority,
    InsightStatus,
    UrgencyLevel,
    DELIVERY_WINDOWS,
    PRIORITY_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class DeliveryQueue:
    """
    Priority queue for prepared actions.

    Actions are prioritized by:
    1. Priority score (higher first)
    2. Urgency level
    3. Creation time (older first within same priority)
    """

    def __init__(self, max_size: int = 100):
        self._queue: List[tuple] = []  # Heap: (-priority, timestamp, action)
        self._max_size = max_size
        self._action_ids: set = set()

    def push(self, action: PreparedAction, priority_score: float) -> bool:
        """
        Add an action to the queue.

        Returns:
            True if added, False if duplicate or queue full
        """
        if action.id in self._action_ids:
            return False

        if len(self._queue) >= self._max_size:
            # Remove lowest priority item
            if self._queue:
                _, _, removed = heapq.heappop(self._queue)
                self._action_ids.discard(removed.id)

        # Use negative priority for max-heap behavior
        heapq.heappush(
            self._queue,
            (-priority_score, action.created_at, action)
        )
        self._action_ids.add(action.id)
        return True

    def pop(self) -> Optional[PreparedAction]:
        """Remove and return highest priority action."""
        if not self._queue:
            return None

        _, _, action = heapq.heappop(self._queue)
        self._action_ids.discard(action.id)
        return action

    def peek(self) -> Optional[PreparedAction]:
        """Look at highest priority action without removing."""
        if not self._queue:
            return None
        return self._queue[0][2]

    def remove(self, action_id: str) -> bool:
        """Remove a specific action by ID."""
        if action_id not in self._action_ids:
            return False

        self._queue = [
            (p, t, a) for p, t, a in self._queue
            if a.id != action_id
        ]
        heapq.heapify(self._queue)
        self._action_ids.discard(action_id)
        return True

    def clear(self) -> int:
        """Clear all actions."""
        count = len(self._queue)
        self._queue.clear()
        self._action_ids.clear()
        return count

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def get_all(self) -> List[PreparedAction]:
        """Get all actions in priority order (non-destructive)."""
        sorted_items = sorted(self._queue)
        return [action for _, _, action in sorted_items]


class DeliveryEngine:
    """
    Manages the delivery of prepared actions to users.

    Responsibilities:
    - Queue management with priority ordering
    - Timing decisions (when to deliver)
    - Channel selection (how to deliver)
    - Delivery execution via registered handlers
    - Batching for digests
    """

    def __init__(
        self,
        delivery_handlers: Dict[DeliveryChannel, Callable] = None,
    ):
        """
        Initialize the delivery engine.

        Args:
            delivery_handlers: Dict mapping channels to delivery functions.
                              Functions should accept (PreparedAction) -> bool
        """
        self._queue = DeliveryQueue()
        self._handlers: Dict[DeliveryChannel, Callable] = delivery_handlers or {}
        self._delivered_count = 0
        self._last_delivery: Optional[str] = None

        # User state (for timing decisions)
        self._user_focus_mode = False
        self._user_dnd = False
        self._quiet_hours = True

        # Batching for digests
        self._morning_batch: List[PreparedAction] = []
        self._evening_batch: List[PreparedAction] = []

    # =========================================================================
    # HANDLER REGISTRATION
    # =========================================================================

    def register_handler(
        self,
        channel: DeliveryChannel,
        handler: Callable[[PreparedAction], bool]
    ) -> None:
        """
        Register a delivery handler for a channel.

        Args:
            channel: The delivery channel
            handler: Function that delivers the action, returns success bool
        """
        self._handlers[channel] = handler
        logger.info(f"Registered handler for {channel.value}")

    def unregister_handler(self, channel: DeliveryChannel) -> None:
        """Unregister a delivery handler."""
        self._handlers.pop(channel, None)

    # =========================================================================
    # QUEUEING
    # =========================================================================

    def queue(
        self,
        action: PreparedAction,
        priority_score: float = 0.5
    ) -> bool:
        """
        Queue an action for delivery.

        Args:
            action: The prepared action to queue
            priority_score: Priority score (0-1)

        Returns:
            True if queued successfully
        """
        # Check if this should go to a digest instead
        if self._should_batch(action):
            return self._add_to_batch(action)

        return self._queue.push(action, priority_score)

    def _should_batch(self, action: PreparedAction) -> bool:
        """Check if action should be batched for digest."""
        if action.preferred_channel in [
            DeliveryChannel.MORNING_DIGEST,
            DeliveryChannel.EVENING_DIGEST,
            DeliveryChannel.WEEKLY_DIGEST
        ]:
            return True
        return False

    def _add_to_batch(self, action: PreparedAction) -> bool:
        """Add action to appropriate digest batch."""
        if action.preferred_channel == DeliveryChannel.MORNING_DIGEST:
            self._morning_batch.append(action)
        elif action.preferred_channel == DeliveryChannel.EVENING_DIGEST:
            self._evening_batch.append(action)
        else:
            # Weekly or other: add to evening for now
            self._evening_batch.append(action)
        return True

    # =========================================================================
    # DELIVERY LOGIC
    # =========================================================================

    async def process_queue(self) -> int:
        """
        Process the delivery queue.

        Delivers actions that should be delivered now.

        Returns:
            Number of actions delivered
        """
        delivered = 0

        while self._queue:
            action = self._queue.peek()
            if not action:
                break

            if not self.should_deliver_now(action):
                break

            action = self._queue.pop()
            if action:
                success = await self._deliver(action)
                if success:
                    delivered += 1

        return delivered

    def should_deliver_now(self, action: PreparedAction) -> bool:
        """
        Determine if an action should be delivered now.

        Considers:
        - User focus mode / DND
        - Quiet hours
        - Action priority and urgency
        - Delivery window preferences
        """
        # High priority always delivers (unless hard DND)
        if self._is_high_priority(action):
            if not self._user_dnd:
                return True
            # Even DND can be overridden for critical
            # (would need insight info, simplified here)

        # Check quiet hours
        if self._is_quiet_hours():
            return False

        # Check focus mode
        if self._user_focus_mode:
            return False

        # Check delivery window
        if action.delivery_window_start and action.delivery_window_end:
            now = datetime.now().time()
            start = datetime.strptime(action.delivery_window_start, "%H:%M").time()
            end = datetime.strptime(action.delivery_window_end, "%H:%M").time()
            if not (start <= now <= end):
                return False

        return True

    def _is_high_priority(self, action: PreparedAction) -> bool:
        """Check if action is high priority."""
        return action.preferred_channel == DeliveryChannel.PUSH_NOTIFICATION

    def _is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        if not self._quiet_hours:
            return False

        now = datetime.now().time()
        quiet = DELIVERY_WINDOWS.get("quiet_hours", {})
        start_str = quiet.get("start", "22:00")
        end_str = quiet.get("end", "07:00")

        start = datetime.strptime(start_str, "%H:%M").time()
        end = datetime.strptime(end_str, "%H:%M").time()

        # Handle overnight quiet hours
        if start > end:
            return now >= start or now <= end
        else:
            return start <= now <= end

    async def _deliver(self, action: PreparedAction) -> bool:
        """
        Execute delivery of an action.

        Args:
            action: The action to deliver

        Returns:
            True if delivered successfully
        """
        channel = action.preferred_channel
        handler = self._handlers.get(channel)

        if not handler:
            logger.warning(f"No handler for channel: {channel.value}")
            # Try fallback channels
            handler = self._get_fallback_handler(channel)
            if not handler:
                return False

        try:
            if asyncio.iscoroutinefunction(handler):
                success = await handler(action)
            else:
                success = handler(action)

            if success:
                action.status = "delivered"
                action.delivered_at = now_iso()
                self._delivered_count += 1
                self._last_delivery = now_iso()
                logger.debug(f"Delivered action via {channel.value}")

            return success

        except Exception as e:
            logger.error(f"Delivery failed: {e}")
            return False

    def _get_fallback_handler(self, channel: DeliveryChannel) -> Optional[Callable]:
        """Get a fallback handler for a channel."""
        fallbacks = {
            DeliveryChannel.PUSH_NOTIFICATION: [
                DeliveryChannel.DESKTOP_POPUP,
                DeliveryChannel.IN_APP
            ],
            DeliveryChannel.VOICE_BRIEFING: [
                DeliveryChannel.DESKTOP_POPUP,
                DeliveryChannel.IN_APP
            ],
            DeliveryChannel.DESKTOP_POPUP: [
                DeliveryChannel.IN_APP
            ],
        }

        for fallback in fallbacks.get(channel, []):
            if fallback in self._handlers:
                return self._handlers[fallback]

        return None

    def select_channel(
        self,
        action: PreparedAction,
        context: Dict[str, Any] = None
    ) -> DeliveryChannel:
        """
        Select the best delivery channel based on context.

        Args:
            action: The action to deliver
            context: Current context (active app, focus mode, etc.)

        Returns:
            Selected delivery channel
        """
        context = context or {}

        # Already specified
        if action.preferred_channel:
            return action.preferred_channel

        # High priority: push
        if self._is_high_priority(action):
            return DeliveryChannel.PUSH_NOTIFICATION

        # User in meeting/call: queue for later
        if context.get("in_meeting"):
            return DeliveryChannel.IN_APP

        # Default
        return DeliveryChannel.VOICE_BRIEFING

    # =========================================================================
    # USER STATE
    # =========================================================================

    def set_focus_mode(self, enabled: bool) -> None:
        """Set user focus mode."""
        self._user_focus_mode = enabled

    def set_dnd(self, enabled: bool) -> None:
        """Set Do Not Disturb mode."""
        self._user_dnd = enabled

    def set_quiet_hours(self, enabled: bool) -> None:
        """Enable/disable quiet hours."""
        self._quiet_hours = enabled

    # =========================================================================
    # DIGEST ACCESS
    # =========================================================================

    def get_morning_batch(self, clear: bool = True) -> List[PreparedAction]:
        """Get actions batched for morning digest."""
        batch = list(self._morning_batch)
        if clear:
            self._morning_batch.clear()
        return batch

    def get_evening_batch(self, clear: bool = True) -> List[PreparedAction]:
        """Get actions batched for evening digest."""
        batch = list(self._evening_batch)
        if clear:
            self._evening_batch.clear()
        return batch

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get delivery engine status."""
        return {
            "queue_size": len(self._queue),
            "morning_batch_size": len(self._morning_batch),
            "evening_batch_size": len(self._evening_batch),
            "delivered_count": self._delivered_count,
            "last_delivery": self._last_delivery,
            "user_focus_mode": self._user_focus_mode,
            "user_dnd": self._user_dnd,
            "quiet_hours_active": self._is_quiet_hours(),
            "registered_channels": [c.value for c in self._handlers.keys()],
        }

    def get_queue_preview(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Preview items in the queue."""
        actions = self._queue.get_all()[:limit]
        return [
            {
                "id": a.id,
                "insight_id": a.insight_id,
                "type": a.type.value,
                "channel": a.preferred_channel.value,
                "created": a.created_at,
            }
            for a in actions
        ]
