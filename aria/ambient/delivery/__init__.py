"""
Aria Ambient Intelligence - Delivery

The Delivery system handles surfacing insights to the user at the right time
through the right channel. It manages timing, batching, and prioritization.

Components:
- DeliveryEngine: Orchestrates delivery decisions and queue management
- DeliveryQueue: Priority queue for prepared actions
- DigestCompiler: Builds morning/evening/weekly digests
"""

from .engine import DeliveryEngine, DeliveryQueue
from .digest import DigestCompiler

__all__ = [
    "DeliveryEngine",
    "DeliveryQueue",
    "DigestCompiler",
]
