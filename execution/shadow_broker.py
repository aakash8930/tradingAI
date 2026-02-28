# execution/shadow_broker.py

from datetime import datetime
from typing import Optional
from execution.position import Position


class ShadowBroker:
    """
    Shadow trading broker.
    Behaves like live execution but never sends real orders.
    """

    def __init__(self):
        self.position: Optional[Position] = None

    def open_position(self, side: str, price: float, qty: float, symbol: str):
        self.position = Position(
            side=side,
            entry_price=price,
            qty=qty,
            entry_time=datetime.utcnow(),
        )
        return self.position

    def close_position(self, price: float, symbol: str):
        if not self.position:
            return 0.0

        pnl = self.position.pnl(price)
        self.position = None
        return pnl