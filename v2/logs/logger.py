import csv
from pathlib import Path
from datetime import datetime


class TradeLogger:
    def __init__(self, path: str = "data_outputs/v2_trades.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time",
                    "side",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "pnl",
                    "balance",
                    "prob_up_entry",
                ])

    def log(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        pnl: float,
        balance: float,
        prob_up: float,
    ):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                side,
                round(entry_price, 4),
                round(exit_price, 4),
                round(qty, 6),
                round(pnl, 2),
                round(balance, 2),
                round(prob_up, 4),
            ])
