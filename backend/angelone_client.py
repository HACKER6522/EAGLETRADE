"""Angel One SmartAPI historical market-data client for chart candles.

Credentials are read only from server-side environment variables. Never expose
ANGELONE_API_KEY or ANGELONE_JWT_TOKEN to the browser.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from typing import Any
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger(__name__)

SMARTAPI_BASE = "https://apiconnect.angelone.in"
IST = ZoneInfo("Asia/Kolkata")

INTERVALS = {
    "1": ("ONE_MINUTE", 30),
    "5": ("FIVE_MINUTE", 100),
    "15": ("FIFTEEN_MINUTE", 200),
    "60": ("ONE_HOUR", 400),
    "1D": ("ONE_DAY", 2000),
}

INDEX_ALIASES = {
    "NIFTY50-INDEX": ("NSE", "NIFTY"),
    "NIFTYBANK-INDEX": ("NSE", "BANKNIFTY"),
    "SENSEX-INDEX": ("BSE", "SENSEX"),
    "FINNIFTY-INDEX": ("NSE", "FINNIFTY"),
    "MIDCPNIFTY-INDEX": ("NSE", "MIDCPNIFTY"),
}
# Stable SmartAPI index tokens from Angel One's exchange instrument master.
INDEX_TOKENS = {
    ("NSE", "NIFTY"): "99926000",
    ("NSE", "BANKNIFTY"): "99926009",
    ("NSE", "FINNIFTY"): "99926037",
    ("NSE", "MIDCPNIFTY"): "99926074",
    ("BSE", "SENSEX"): "99919000",
}


class AngelOneConfigurationError(RuntimeError):
    pass


class AngelOneAPIError(RuntimeError):
    pass


class AngelOneClient:
    """Small REST client for SmartAPI scrip lookup and historical candles."""

    def __init__(self) -> None:
        self._token_cache: dict[tuple[str, str], tuple[str, float, str]] = {}
        self._cache_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(os.getenv("ANGELONE_API_KEY", "").strip() and
                    os.getenv("ANGELONE_JWT_TOKEN", "").strip())

    def _headers(self) -> dict[str, str]:
        api_key = os.getenv("ANGELONE_API_KEY", "").strip()
        jwt_token = os.getenv("ANGELONE_JWT_TOKEN", "").strip()
        if not api_key or not jwt_token:
            raise AngelOneConfigurationError(
                "Angel One SmartAPI is not configured. Set ANGELONE_API_KEY and "
                "ANGELONE_JWT_TOKEN in the backend environment."
            )
        return {
            "Authorization": f"Bearer {jwt_token}",
            "X-PrivateKey": api_key,
            "X-UserType": "USER",
            "X-SourceID": os.getenv("ANGELONE_SOURCE_ID", "WEB").strip() or "WEB",
            "X-ClientLocalIP": os.getenv("ANGELONE_CLIENT_LOCAL_IP", "127.0.0.1").strip(),
            "X-ClientPublicIP": os.getenv("ANGELONE_CLIENT_PUBLIC_IP", "127.0.0.1").strip(),
            "X-MACAddress": os.getenv("ANGELONE_MAC_ADDRESS", "00:00:00:00:00:00").strip(),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _resolve_symbol(symbol: str) -> tuple[str, str, str]:
        raw = (symbol or "").strip().upper()
        if ":" in raw:
            exchange, ticker = raw.split(":", 1)
        else:
            exchange, ticker = "NSE", raw
        if exchange not in {"NSE", "BSE", "NFO", "BFO", "MCX"}:
            raise AngelOneAPIError(f"Unsupported Angel One exchange: {exchange}")
        alias = INDEX_ALIASES.get(ticker)
        if alias:
            return alias[0], alias[1], "index"
        if ticker.endswith("-EQ"):
            return exchange, ticker, "equity"
        if ticker.endswith("-FUT"):
            futures_exchange = "BFO" if exchange == "BSE" else "NFO"
            return futures_exchange, ticker[:-4], "future"
        if ticker.endswith("-INDEX"):
            return exchange, ticker[:-6], "index"
        return exchange, ticker, "equity"

    def _find_instrument(self, exchange: str, query: str, kind: str) -> str:
        cache_key = (exchange, query)
        index_token = INDEX_TOKENS.get(cache_key) if kind == "index" else None
        if index_token:
            return index_token
        now = time.monotonic()
        with self._cache_lock:
            cached = self._token_cache.get(cache_key)
            if cached and cached[1] > now:
                return cached[0]

        headers = self._headers()
        url = f"{SMARTAPI_BASE}/rest/secure/angelbroking/order/v1/searchScrip"
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(
                    url, headers=headers,
                    json={"exchange": exchange, "searchscrip": query},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Angel One scrip lookup failed for %s: %s", query, exc)
            raise AngelOneAPIError("Angel One instrument lookup failed. Check the SmartAPI session and network.") from exc

        if not payload.get("status") or not isinstance(payload.get("data"), list):
            code = payload.get("errorcode", "unknown")
            message = payload.get("message", "Instrument lookup failed")
            raise AngelOneAPIError(f"Angel One instrument lookup failed ({code}): {message}")

        instruments = payload["data"]
        exact = [item for item in instruments
                 if str(item.get("tradingsymbol", "")).upper() == query.upper()]
        if kind == "future":
            futures = [item for item in instruments
                       if str(item.get("tradingsymbol", "")).upper().startswith(query.upper())
                       and str(item.get("tradingsymbol", "")).upper().endswith("FUT")]
            # SmartAPI's search endpoint returns currently listed matches; use the
            # nearest future expiry when expiry metadata is present.
            def expiry_key(item: dict[str, Any]) -> dt.date:
                expiry = str(item.get("expiry", "")).upper()
                try:
                    return dt.datetime.strptime(expiry, "%d%b%Y").date()
                except ValueError:
                    return dt.date.max
            futures.sort(key=expiry_key)
            candidates = futures or exact
        elif kind == "index":
            candidates = exact or [
                item for item in instruments
                if str(item.get("tradingsymbol", "")).upper() == query.upper()
            ]
        else:
            candidates = exact or [
                item for item in instruments
                if str(item.get("tradingsymbol", "")).upper() == query.upper()
            ]

        if not candidates:
            raise AngelOneAPIError(
                f"Angel One could not find {query} on {exchange}. "
                "This instrument may not be supported by SmartAPI."
            )
        token = str(candidates[0].get("symboltoken", "")).strip()
        if not token:
            raise AngelOneAPIError(f"Angel One returned no token for {query}.")
        with self._cache_lock:
            self._token_cache[cache_key] = (token, now + 3600, str(candidates[0].get("tradingsymbol", query)))
        return token

    def get_history(self, symbol: str, resolution: str = "5", days: int = 3) -> list[dict[str, Any]]:
        if resolution not in INTERVALS:
            raise AngelOneAPIError("Unsupported chart timeframe. Use 1m, 5m, 15m, 1h, or 1D.")
        interval, max_days = INTERVALS[resolution]
        exchange, query, kind = self._resolve_symbol(symbol)
        token = self._find_instrument(exchange, query, kind)
        safe_days = max(1, min(int(days), max_days))
        end = dt.datetime.now(IST).replace(second=0, microsecond=0)
        start = end - dt.timedelta(days=safe_days)
        payload = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": start.strftime("%Y-%m-%d %H:%M"),
            "todate": end.strftime("%Y-%m-%d %H:%M"),
        }
        url = f"{SMARTAPI_BASE}/rest/secure/angelbroking/historical/v1/getCandleData"
        try:
            with httpx.Client(timeout=25.0) as client:
                response = client.post(url, headers=self._headers(), json=payload)
                response.raise_for_status()
                result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Angel One candle request failed for %s: %s", symbol, exc)
            raise AngelOneAPIError("Angel One historical candle request failed. Check your session, IP settings, and API limits.") from exc

        if not result.get("status") or not isinstance(result.get("data"), list):
            code = result.get("errorcode", "unknown")
            message = result.get("message", "Historical data request failed")
            raise AngelOneAPIError(f"Angel One historical data error ({code}): {message}")

        candles = []
        for row in result["data"]:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            try:
                timestamp = dt.datetime.fromisoformat(str(row[0])).timestamp()
                candles.append({
                    "time": int(timestamp),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5] or 0),
                })
            except (TypeError, ValueError, OverflowError):
                continue
        candles.sort(key=lambda candle: candle["time"])
        return candles
