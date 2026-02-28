
# logs/logger.py

import csv
from pathlib import Path
from datetime import datetime


class TradeLogger:
    HEADERS = [
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

    def __init__(self, path: str = "data_outputs/v2_trades.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self._write_header()
        else:
            self._upgrade_schema_if_needed()

    def _write_header(self) -> None:
        with self.path.open("w", newline="") as f:
            csv.writer(f).writerow(self.HEADERS)

    def _upgrade_schema_if_needed(self) -> None:
        with self.path.open("r", newline="") as f:
            rows = list(csv.reader(f))

        if not rows or rows[0] == self.HEADERS:
            return

        upgraded = [self.HEADERS]
        for row in rows[1:]:
            if len(row) == 8:  # legacy schema
                upgraded.append([row[0], "", *row[1:]])
            else:
                upgraded.append(row)

        with self.path.open("w", newline="") as f:
            csv.writer(f).writerows(upgraded)

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
    ) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([
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

