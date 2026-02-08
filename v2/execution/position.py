from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    side: str              # "LONG" or "SHORT"
    entry_price: float
    qty: float
    entry_time: datetime

    def pnl(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry_price) * self.qty
        else:
            return (self.entry_price - price) * self.qty
