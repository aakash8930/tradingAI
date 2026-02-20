# logs/logger.py

import csv
from pathlib import Path
from datetime import datetime


class TradeLogger:
    def __init__(self, path: str = "data_outputs/v2_trades.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = [
            "time",
            "symbol",
            "side",
            "entry_price",
            "exit_price",
            "qty",
            "pnl",
            "balance",
            "prob_up_entry",
        ]

        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        else:
            self._upgrade_schema_if_needed()

    def _upgrade_schema_if_needed(self):
        with open(self.path, "r", newline="") as f:
            rows = list(csv.reader(f))

        if not rows:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.headers)
            return

        header = rows[0]
        if "symbol" in header:
            return

        upgraded = [self.headers]
        for row in rows[1:]:
            if not row:
                continue
            if len(row) == 8:
                upgraded.append([row[0], "", row[1], row[2], row[3], row[4], row[5], row[6], row[7]])
            else:
                upgraded.append(row)

        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(upgraded)

    def log(
        self,
        symbol: str,
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
                symbol,
                side,
                round(entry_price, 4),
                round(exit_price, 4),
                round(qty, 8),
                round(pnl, 6),
                round(balance, 6),
                round(prob_up, 4),
            ])
