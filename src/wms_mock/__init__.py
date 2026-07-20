"""Mock WMS: in-process order source posting orders like {"item": "red marker", "bin": "A"}."""

from .orders import DEMO_ORDERS, Order, OrderSource

__all__ = ["DEMO_ORDERS", "Order", "OrderSource"]
