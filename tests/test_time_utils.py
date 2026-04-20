from app.time_utils import (
    build_market_slug,
    candidate_window_starts,
    window_end_from_start,
    window_start_from_server_ts,
)


def test_exact_five_minute_boundary_handling() -> None:
    assert window_start_from_server_ts(600) == 600
    assert window_start_from_server_ts(899) == 600
    assert window_end_from_start(600) == 900


def test_rollover_from_one_window_to_the_next() -> None:
    assert window_start_from_server_ts(899) == 600
    assert window_start_from_server_ts(900) == 900
    assert candidate_window_starts(900) == [900, 600, 1200]


def test_slug_generation_from_server_timestamp() -> None:
    server_ts = 1_776_708_616
    window_start = window_start_from_server_ts(server_ts)
    assert window_start == 1_776_708_600
    assert build_market_slug("btc-updown-5m", window_start) == "btc-updown-5m-1776708600"
