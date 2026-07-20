"""Mock WMS order source — emits warehouse orders shaped ``{"item", "bin"}``.

In-process and network-free by design so the whole order→pick→place→alert loop is
unit-testable offline (CLAUDE.md D9, sim-first). A real WMS will later POST these
same-shaped orders to the agent via a Cyberwave **webhook** workflow (see runbook);
until then :class:`OrderSource` replays a small hardcoded list.

Kept deliberately tiny: an ``Order`` value object + an iterable source. No FastAPI
yet — swap the source for a webhook handler when the loop is proven in sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class Order:
    """A single fulfillment order: pick ``item``, drop it in ``bin``."""

    item: str
    bin: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Order":
        """Parse (and validate) an order dict. Raises ``ValueError`` if malformed."""
        try:
            item = str(data["item"]).strip()
            bin_ = str(data["bin"]).strip()
        except KeyError as e:
            raise ValueError(f"order missing required field: {e}") from e
        if not item:
            raise ValueError("order 'item' must be a non-empty string")
        if not bin_:
            raise ValueError("order 'bin' must be a non-empty string")
        return cls(item=item, bin=bin_)

    def to_dict(self) -> dict[str, str]:
        return {"item": self.item, "bin": self.bin}


# A small, demo-friendly batch. Items/bins here must have hardcoded poses in
# src/agent_service/poses.py or the loop will (correctly) reject them as unknown.
DEMO_ORDERS: list[dict[str, str]] = [
    {"item": "red marker", "bin": "A"},
    {"item": "blue marker", "bin": "B"},
    {"item": "eraser", "bin": "A"},
]


class OrderSource:
    """Replays a list of order dicts as validated :class:`Order` objects.

    ``for order in OrderSource(): ...`` — deterministic, no network. Defaults to
    :data:`DEMO_ORDERS`; pass your own list to drive tests.
    """

    def __init__(self, orders: Iterable[dict[str, Any]] | None = None) -> None:
        self._raw = list(DEMO_ORDERS if orders is None else orders)

    def __iter__(self) -> Iterator[Order]:
        for raw in self._raw:
            yield Order.from_dict(raw)

    def __len__(self) -> int:
        return len(self._raw)

    def first(self) -> Order:
        """The first order — handy for single-order sim runs."""
        if not self._raw:
            raise ValueError("OrderSource is empty")
        return Order.from_dict(self._raw[0])
