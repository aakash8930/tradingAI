from datetime import datetime
from typing import Optional

from v2.execution.position import Position


class PaperBroker:
    """
    Simulates order execution.
    """

    def __init__(self):
        self.position: Optional[Position] = None

    def open_position(self, side: str, price: float, qty: float) -> Position:
        self.position = Position(
            side=side,
            entry_price=price,
            qty=qty,
            entry_time=datetime.utcnow(),
        )
        return self.position

    def close_position(self, price: float) -> float:
        if self.position is None:
            return 0.0

        pnl = self.position.pnl(price)
        self.position = None
        return pnl
