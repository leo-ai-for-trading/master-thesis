from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import typer

from app.config import Settings, setup_logging
from app.polymarket_client import PolymarketClient
from app.quote_service import OrderBookCollector

cli = typer.Typer(add_completion=False, no_args_is_help=True)


def _build_settings(
    *,
    poll_interval: float | None = None,
    start_slug: str | None = None,
) -> Settings:
    settings = Settings.from_env()
    if poll_interval is not None:
        settings = replace(settings, poll_interval_seconds=poll_interval)
    if start_slug is not None:
        settings = replace(settings, start_slug=start_slug)
    return settings


async def _run_once(settings: Settings) -> list[dict[str, object]]:
    logger = setup_logging(settings.log_level)
    async with PolymarketClient(settings, logger=logger) as client:
        collector = OrderBookCollector(settings, client, logger=logger)
        return await collector.collect_once(persist=True)


async def _run_stream(settings: Settings, max_iterations: int | None) -> None:
    logger = setup_logging(settings.log_level)
    async with PolymarketClient(settings, logger=logger) as client:
        collector = OrderBookCollector(settings, client, logger=logger)
        await collector.stream(max_iterations=max_iterations)


async def _run_inspect(settings: Settings) -> dict[str, object]:
    logger = setup_logging(settings.log_level)
    async with PolymarketClient(settings, logger=logger) as client:
        collector = OrderBookCollector(settings, client, logger=logger)
        return await collector.inspect_current()


@cli.command()
def once(
    start_slug: str | None = typer.Option(None),
) -> None:
    """Fetch one YES/NO order-book snapshot pair and write it to JSONL and CSV."""

    records = asyncio.run(_run_once(_build_settings(start_slug=start_slug)))
    typer.echo(json.dumps(records, indent=2, default=str))


@cli.command()
def stream(
    poll_interval: float | None = typer.Option(None, min=0.25),
    max_iterations: int | None = typer.Option(None, min=1),
    start_slug: str | None = typer.Option(None),
) -> None:
    """Continuously fetch live order-book data and append it to JSONL and CSV."""

    settings = _build_settings(poll_interval=poll_interval, start_slug=start_slug)
    try:
        asyncio.run(_run_stream(settings, max_iterations=max_iterations))
    except KeyboardInterrupt:
        typer.echo("stopped")


@cli.command("inspect-current")
def inspect_current(
    start_slug: str | None = typer.Option(None),
) -> None:
    """Print the currently resolved BTC 5-minute market."""

    payload = asyncio.run(_run_inspect(_build_settings(start_slug=start_slug)))
    typer.echo(f"Server time UTC: {payload['server_time']}")
    typer.echo(f"Current window start UTC: {payload['current_window_start']}")
    typer.echo(f"Current window end UTC: {payload['current_window_end']}")
    typer.echo(f"Resolved slug: {payload['resolved_slug']}")
    typer.echo(f"Condition id: {payload['condition_id']}")
    typer.echo(f"YES token ({payload['yes_label']}): {payload['token_yes']}")
    typer.echo(f"NO token ({payload['no_label']}): {payload['token_no']}")
    typer.echo(f"Seconds remaining: {payload['seconds_remaining']:.3f}")


if __name__ == "__main__":
    cli()
