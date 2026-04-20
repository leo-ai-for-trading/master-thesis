from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from app.config import Settings

RETRIABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class PolymarketClient:
    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("polymarket_research.client")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=settings.http_timeout_seconds,
            follow_redirects=True,
            headers={
                "Accept": "application/json, text/plain;q=0.9",
                "User-Agent": "master-thesis-polymarket-research/0.1",
            },
        )

    async def __aenter__(self) -> "PolymarketClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        allow_404: bool = False,
        expect_text: bool = False,
    ) -> Any:
        for attempt in range(self.settings.http_max_retries + 1):
            try:
                response = await self._client.get(url, params=params)
                if allow_404 and response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.text.strip() if expect_text else response.json()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if allow_404 and status_code == 404:
                    return None
                if status_code not in RETRIABLE_STATUS_CODES or attempt >= self.settings.http_max_retries:
                    raise
                await self._sleep_before_retry(attempt, status_code=status_code)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
                if attempt >= self.settings.http_max_retries:
                    raise
                await self._sleep_before_retry(attempt, error_type=type(exc).__name__)

        raise RuntimeError(f"request retries exhausted for {url}")

    async def _sleep_before_retry(
        self,
        attempt: int,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
    ) -> None:
        delay = self.settings.http_base_backoff_seconds * (2**attempt)
        delay += random.uniform(0.0, 0.1)
        self.logger.warning(
            "retrying request",
            extra={
                "attempt": attempt + 1,
                "delay_seconds": round(delay, 3),
                "status_code": status_code,
                "error_type": error_type,
            },
        )
        await asyncio.sleep(delay)

    async def get_server_time(self) -> int:
        raw = await self._request(
            f"{self.settings.clob_base_url}/time",
            expect_text=True,
        )
        return int(raw)

    async def get_event_by_slug(self, slug: str) -> dict[str, Any] | None:
        payload = await self._request(
            f"{self.settings.gamma_base_url}/events/slug/{slug}",
            allow_404=True,
        )
        if payload is None:
            return None
        return dict(payload)

    async def get_market_by_slug(self, slug: str) -> dict[str, Any] | None:
        payload = await self._request(
            f"{self.settings.gamma_base_url}/markets/slug/{slug}",
            allow_404=True,
        )
        if payload is None:
            return None
        return dict(payload)

    async def get_clob_market(self, condition_id: str) -> dict[str, Any]:
        payload = await self._request(
            f"{self.settings.clob_base_url}/clob-markets/{condition_id}",
        )
        return dict(payload)

    async def get_book(self, token_id: str) -> dict[str, Any]:
        payload = await self._request(
            f"{self.settings.clob_base_url}/book",
            params={"token_id": token_id},
        )
        return dict(payload)

    async def get_midpoint(self, token_id: str) -> dict[str, Any]:
        payload = await self._request(
            f"{self.settings.clob_base_url}/midpoint",
            params={"token_id": token_id},
        )
        return dict(payload)

    async def get_spread(self, token_id: str) -> dict[str, Any]:
        payload = await self._request(
            f"{self.settings.clob_base_url}/spread",
            params={"token_id": token_id},
        )
        return dict(payload)

    async def get_best_price(self, token_id: str, side: str) -> dict[str, Any]:
        payload = await self._request(
            f"{self.settings.clob_base_url}/price",
            params={"token_id": token_id, "side": side},
        )
        return dict(payload)
