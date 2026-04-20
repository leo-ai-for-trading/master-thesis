from __future__ import annotations

import asyncio
import json
import webbrowser
from dataclasses import replace

import typer

from app.config import Settings, setup_logging
from app.mm_dashboard import DashboardStateStore, MarketMakerDashboardServer, MarketMakerGraphRunner
from app.paper_market_maker import PaperMarketMakerConfig
from app.polymarket_client import PolymarketClient
from app.quote_service import QuoteService
from app.reinforcement_learning import QLearningConfig, RLTrainingRunner

cli = typer.Typer(add_completion=False, no_args_is_help=True)


def _build_settings(
    *,
    poll_interval: float | None = None,
    use_ws: bool | None = None,
) -> Settings:
    settings = Settings.from_env()
    if poll_interval is not None:
        settings = replace(settings, poll_interval_seconds=poll_interval)
    if use_ws is not None:
        settings = replace(settings, use_ws=use_ws)
    return settings


async def _run_once(settings: Settings) -> list[dict[str, object]]:
    logger = setup_logging(settings.log_level)
    async with PolymarketClient(settings, logger=logger) as client:
        service = QuoteService(settings, client, logger=logger)
        return await service.collect_once(persist=True)


async def _run_stream(settings: Settings, max_iterations: int | None) -> None:
    logger = setup_logging(settings.log_level)
    async with PolymarketClient(settings, logger=logger) as client:
        service = QuoteService(settings, client, logger=logger)
        await service.stream(max_iterations=max_iterations)


async def _run_inspect(settings: Settings) -> dict[str, object]:
    logger = setup_logging(settings.log_level)
    async with PolymarketClient(settings, logger=logger) as client:
        service = QuoteService(settings, client, logger=logger)
        return await service.inspect_current()


async def _run_graph_mm(
    *,
    settings: Settings,
    runner: MarketMakerGraphRunner,
    server: MarketMakerDashboardServer,
    max_iterations: int | None,
) -> None:
    try:
        await runner.run(max_iterations=max_iterations)
    finally:
        server.stop()


async def _run_simulate_mm(
    *,
    runner: MarketMakerGraphRunner,
    max_iterations: int | None,
) -> None:
    await runner.run(max_iterations=max_iterations)


async def _run_train_rl_mm(
    *,
    runner: RLTrainingRunner,
    server: MarketMakerDashboardServer | None,
    max_iterations: int | None,
) -> None:
    try:
        await runner.run(max_iterations=max_iterations)
    finally:
        if server is not None:
            server.stop()


@cli.command()
def once() -> None:
    """Fetch one YES/NO snapshot pair and write it to JSONL and CSV."""

    records = asyncio.run(_run_once(_build_settings()))
    typer.echo(json.dumps(records, indent=2, default=str))


@cli.command()
def stream(
    poll_interval: float | None = typer.Option(None, min=0.25, max=1.0),
    use_ws: bool | None = typer.Option(None, "--use-ws/--no-use-ws"),
    max_iterations: int | None = typer.Option(None, min=1),
) -> None:
    """Continuously collect read-only quote data."""

    settings = _build_settings(poll_interval=poll_interval, use_ws=use_ws)
    try:
        asyncio.run(_run_stream(settings, max_iterations=max_iterations))
    except KeyboardInterrupt:
        typer.echo("stopped")


@cli.command("inspect-current")
def inspect_current() -> None:
    """Print the currently resolved BTC 5-minute market and top-of-book summary."""

    payload = asyncio.run(_run_inspect(_build_settings()))
    typer.echo(f"Server time UTC: {payload['server_time']}")
    typer.echo(f"Current window start UTC: {payload['current_window_start']}")
    typer.echo(f"Current window end UTC: {payload['current_window_end']}")
    typer.echo(f"Resolved slug: {payload['resolved_slug']}")
    typer.echo(f"Condition id: {payload['condition_id']}")
    typer.echo(
        f"YES token ({payload['yes_label']}): {payload['token_yes']}"
    )
    typer.echo(
        f"NO token ({payload['no_label']}): {payload['token_no']}"
    )
    typer.echo(
        "Top of book YES: "
        f"bid={payload['yes_top_of_book']['best_bid']} "
        f"ask={payload['yes_top_of_book']['best_ask']} "
        f"bid_size_1={payload['yes_top_of_book']['bid_size_1']} "
        f"ask_size_1={payload['yes_top_of_book']['ask_size_1']}"
    )
    typer.echo(
        "Top of book NO: "
        f"bid={payload['no_top_of_book']['best_bid']} "
        f"ask={payload['no_top_of_book']['best_ask']} "
        f"bid_size_1={payload['no_top_of_book']['bid_size_1']} "
        f"ask_size_1={payload['no_top_of_book']['ask_size_1']}"
    )
    typer.echo(f"Midpoint YES: {payload['yes_midpoint']}")
    typer.echo(f"Midpoint NO: {payload['no_midpoint']}")
    typer.echo(f"Seconds remaining: {payload['seconds_remaining']:.3f}")


@cli.command("graph-mm")
def graph_mm(
    poll_interval: float | None = typer.Option(None, min=0.25, max=1.0),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765, min=1024, max=65535),
    max_points: int = typer.Option(600, min=100, max=5000),
    max_iterations: int | None = typer.Option(None, min=1),
    open_browser: bool = typer.Option(False, "--open-browser/--no-open-browser"),
    order_size: float = typer.Option(5.0, min=1.0),
    max_inventory_abs: float = typer.Option(25.0, min=1.0),
) -> None:
    """Run a live mean-field paper market-maker monitor with a local dashboard."""

    settings = _build_settings(poll_interval=poll_interval)
    logger = setup_logging(settings.log_level)
    mm_config = PaperMarketMakerConfig(
        order_size=order_size,
        max_inventory_abs=max_inventory_abs,
    )
    store = DashboardStateStore(max_points=max_points, inventory_cap=max_inventory_abs)
    server = MarketMakerDashboardServer(host=host, port=port, store=store)
    server.start()
    logger.info("market maker dashboard started", extra={"url": server.url})
    if open_browser:
        webbrowser.open(server.url)
    typer.echo(f"Dashboard: {server.url}")

    runner = MarketMakerGraphRunner(
        settings=settings,
        logger=logger,
        store=store,
        mm_config=mm_config,
    )
    try:
        asyncio.run(
            _run_graph_mm(
                settings=settings,
                runner=runner,
                server=server,
                max_iterations=max_iterations,
            )
        )
    except KeyboardInterrupt:
        typer.echo("stopped")


@cli.command("simulate-mm")
def simulate_mm(
    poll_interval: float | None = typer.Option(None, min=0.25, max=1.0),
    max_iterations: int | None = typer.Option(None, min=1),
    order_size: float = typer.Option(5.0, min=1.0),
    max_inventory_abs: float = typer.Option(25.0, min=1.0),
) -> None:
    """Run the live mean-field paper-trading environment without the dashboard."""

    settings = _build_settings(poll_interval=poll_interval)
    logger = setup_logging(settings.log_level)
    store = DashboardStateStore(max_points=200, inventory_cap=max_inventory_abs)
    runner = MarketMakerGraphRunner(
        settings=settings,
        logger=logger,
        store=store,
        mm_config=PaperMarketMakerConfig(
            order_size=order_size,
            max_inventory_abs=max_inventory_abs,
        ),
    )
    try:
        asyncio.run(
            _run_simulate_mm(
                runner=runner,
                max_iterations=max_iterations,
            )
        )
    except KeyboardInterrupt:
        typer.echo("stopped")


@cli.command("train-rl-mm")
def train_rl_mm(
    poll_interval: float | None = typer.Option(None, min=0.25, max=1.0),
    max_iterations: int | None = typer.Option(None, min=1),
    order_size: float = typer.Option(5.0, min=1.0),
    max_inventory_abs: float = typer.Option(25.0, min=1.0),
    alpha: float = typer.Option(0.15, min=0.01, max=1.0),
    gamma: float = typer.Option(0.95, min=0.0, max=1.0),
    epsilon: float = typer.Option(0.20, min=0.0, max=1.0),
    replay: bool = typer.Option(True, "--replay/--no-replay"),
    replay_epochs: int = typer.Option(3, min=1, max=100),
    replay_sample_size: int = typer.Option(2048, min=1),
    replay_min_rows: int = typer.Option(256, min=1),
    replay_max_rows: int = typer.Option(50000, min=1),
) -> None:
    """Run online Q-learning on the live paper market-making environment."""

    settings = _build_settings(poll_interval=poll_interval)
    logger = setup_logging(settings.log_level)
    runner = RLTrainingRunner(
        settings=settings,
        logger=logger,
        mm_config=PaperMarketMakerConfig(
            order_size=order_size,
            max_inventory_abs=max_inventory_abs,
        ),
        rl_config=QLearningConfig(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            replay_enabled=replay,
            replay_epochs=replay_epochs,
            replay_sample_size=replay_sample_size,
            replay_min_rows=replay_min_rows,
            replay_max_rows=replay_max_rows,
        ),
    )
    try:
        asyncio.run(
            _run_train_rl_mm(
                runner=runner,
                server=None,
                max_iterations=max_iterations,
            )
        )
    except KeyboardInterrupt:
        typer.echo("stopped")


@cli.command("graph-rl-mm")
def graph_rl_mm(
    poll_interval: float | None = typer.Option(None, min=0.25, max=1.0),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8775, min=1024, max=65535),
    max_points: int = typer.Option(600, min=100, max=5000),
    max_iterations: int | None = typer.Option(None, min=1),
    open_browser: bool = typer.Option(False, "--open-browser/--no-open-browser"),
    order_size: float = typer.Option(5.0, min=1.0),
    max_inventory_abs: float = typer.Option(25.0, min=1.0),
    alpha: float = typer.Option(0.15, min=0.01, max=1.0),
    gamma: float = typer.Option(0.95, min=0.0, max=1.0),
    epsilon: float = typer.Option(0.20, min=0.0, max=1.0),
    replay: bool = typer.Option(True, "--replay/--no-replay"),
    replay_epochs: int = typer.Option(3, min=1, max=100),
    replay_sample_size: int = typer.Option(2048, min=1),
    replay_min_rows: int = typer.Option(256, min=1),
    replay_max_rows: int = typer.Option(50000, min=1),
) -> None:
    """Run online Q-learning with a live local dashboard."""

    settings = _build_settings(poll_interval=poll_interval)
    logger = setup_logging(settings.log_level)
    store = DashboardStateStore(max_points=max_points, inventory_cap=max_inventory_abs)
    server = MarketMakerDashboardServer(host=host, port=port, store=store)
    server.start()
    logger.info("rl market maker dashboard started", extra={"url": server.url})
    if open_browser:
        webbrowser.open(server.url)
    typer.echo(f"RL Dashboard: {server.url}")

    runner = RLTrainingRunner(
        settings=settings,
        logger=logger,
        store=store,
        mm_config=PaperMarketMakerConfig(
            order_size=order_size,
            max_inventory_abs=max_inventory_abs,
        ),
        rl_config=QLearningConfig(
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            replay_enabled=replay,
            replay_epochs=replay_epochs,
            replay_sample_size=replay_sample_size,
            replay_min_rows=replay_min_rows,
            replay_max_rows=replay_max_rows,
        ),
    )
    try:
        asyncio.run(
            _run_train_rl_mm(
                runner=runner,
                server=server,
                max_iterations=max_iterations,
            )
        )
    except KeyboardInterrupt:
        typer.echo("stopped")


if __name__ == "__main__":
    cli()
