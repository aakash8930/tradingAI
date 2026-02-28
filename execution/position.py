
# execution/position.py

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Position:
    """
    Represents a single open trading position.
    """
    side: str            # "LONG" or "SHORT"
    entry_price: float
    qty: float
    entry_time: datetime

    def pnl(self, exit_price: float) -> float:
        """
        Calculate profit / loss at given exit price.
        """
        if self.side == "LONG":
            return (exit_price - self.entry_price) * self.qty

        # SHORT (future-proofing)
        return (self.entry_price - exit_price) * self.qty
