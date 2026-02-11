# limits.py

from dataclasses import dataclass
from datetime import date


@dataclass
class RiskLimits:
    max_daily_loss_pct: float = 0.03     # 3%
    max_weekly_loss_pct: float = 0.06    # 6%
    max_trades_per_day: int = 10


class RiskState:
    """
    Tracks losses and trade counts.
    """

    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance

        self.today = date.today()
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0

        self.current_date = None


    def reset_if_new_day(self, current_date):
        if self.current_date != current_date:
            self.current_date = current_date
            self.trades_today = 0
            self.daily_pnl = 0.0


    def register_trade(self, pnl: float):
        self.trades_today += 1
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.current_balance += pnl

    def daily_loss_pct(self) -> float:
        return max(0.0, -self.daily_pnl / self.starting_balance)

    def weekly_loss_pct(self) -> float:
        return max(0.0, -self.weekly_pnl / self.starting_balance)

    def trading_allowed(self, limits: RiskLimits) -> bool:
        if self.trades_today >= limits.max_trades_per_day:
            return False
        if self.daily_loss_pct() >= limits.max_daily_loss_pct:
            return False
        if self.weekly_loss_pct() >= limits.max_weekly_loss_pct:
            return False
        return True
