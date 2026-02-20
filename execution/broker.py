# execution/broker.py

from datetime import datetime
from typing import Optional

import ccxt

from execution.position import Position


class PaperBroker:
    """
    Simulates order execution.
    """

    def __init__(self):
        self.position: Optional[Position] = None

    def open_position(self, side: str, price: float, qty: float, symbol: Optional[str] = None) -> Position:
        self.position = Position(
            side=side,
            entry_price=price,
            qty=qty,
            entry_time=datetime.utcnow(),
        )
        return self.position

    def close_position(self, price: float, symbol: Optional[str] = None) -> float:
        if self.position is None:
            return 0.0

        pnl = self.position.pnl(price)
        self.position = None
        return pnl


class LiveBroker:
    """
    Executes real market orders through ccxt.
    Current implementation is spot-safe: LONG only.
    """

    def __init__(
        self,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ):
        if exchange_name != "binance":
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                },
            }
        )

        if testnet and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

        self.exchange.load_markets()
        self.position: Optional[Position] = None

    def get_balance_usdt(self) -> float:
        balance = self.exchange.fetch_balance()
        free = balance.get("free", {}).get("USDT")
        total = balance.get("total", {}).get("USDT")
        if free is not None:
            return float(free)
        if total is not None:
            return float(total)
        return 0.0

    def _normalize_qty(self, symbol: str, qty: float) -> float:
        market = self.exchange.market(symbol)
        normalized = float(self.exchange.amount_to_precision(symbol, qty))

        min_amount = market.get("limits", {}).get("amount", {}).get("min")
        if min_amount is not None and normalized < float(min_amount):
            raise ValueError(
                f"Order qty too small for {symbol}: {normalized} < min {min_amount}"
            )

        return normalized

    def _validate_notional(self, symbol: str, qty: float, price: float) -> None:
        market = self.exchange.market(symbol)
        min_cost = market.get("limits", {}).get("cost", {}).get("min")
        if min_cost is None:
            return

        notional = qty * price
        if notional < float(min_cost):
            raise ValueError(
                f"Order notional too small for {symbol}: {notional:.4f} < min {min_cost}"
            )

    def open_position(self, side: str, price: float, qty: float, symbol: str) -> Position:
        if self.position is not None:
            raise RuntimeError("Position already open.")

        if side != "LONG":
            raise ValueError("Live spot broker supports LONG only.")

        qty = self._normalize_qty(symbol, qty)
        self._validate_notional(symbol, qty, price)

        order = self.exchange.create_order(symbol, "market", "buy", qty)
        fill_price = float(order.get("average") or price)
        filled_qty = float(order.get("filled") or qty)

        if filled_qty <= 0:
            raise RuntimeError(f"Buy order was not filled: {order}")

        self.position = Position(
            side=side,
            entry_price=fill_price,
            qty=filled_qty,
            entry_time=datetime.utcnow(),
        )
        return self.position

    def close_position(self, price: float, symbol: str) -> float:
        if self.position is None:
            return 0.0

        qty = self._normalize_qty(symbol, self.position.qty)
        order = self.exchange.create_order(symbol, "market", "sell", qty)
        exit_price = float(order.get("average") or price)

        pnl = self.position.pnl(exit_price)
        self.position = None
        return pnl
