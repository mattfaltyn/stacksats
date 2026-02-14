from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
from matplotlib.axes import Axes

from stacksats.plot_weights import (
    fetch_weights_for_date_range,
    get_date_range_options,
    get_db_connection,
    plot_dca_weights,
    validate_date_range,
)


def test_get_db_connection_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL environment variable is not set"):
        get_db_connection()


def test_get_date_range_options_returns_typed_dataframe() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("2024-01-01", "2024-12-31", 366),
        ("2024-02-01", "2025-01-31", 366),
    ]

    df = get_date_range_options(conn)

    assert list(df.columns) == ["start_date", "end_date", "count"]
    assert pd.api.types.is_datetime64_any_dtype(df["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["end_date"])
    assert df.iloc[0]["count"] == 366


def test_get_date_range_options_raises_when_no_rows() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []

    with pytest.raises(ValueError, match="No data found"):
        get_date_range_options(conn)


def test_validate_date_range_returns_true_when_present() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (1,)

    assert validate_date_range(conn, "2024-01-01", "2024-12-31") is True


def test_validate_date_range_returns_false_when_absent() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (0,)

    assert validate_date_range(conn, "2024-01-01", "2024-12-31") is False


def test_fetch_weights_for_date_range_returns_dataframe() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = [
        ("2024-01-01", 0.5, 50000.0, 1),
        ("2024-01-02", 0.5, None, 2),
    ]

    df = fetch_weights_for_date_range(conn, "2024-01-01", "2024-12-31")

    assert list(df.columns) == ["DCA_date", "weight", "btc_usd", "id"]
    assert pd.api.types.is_datetime64_any_dtype(df["DCA_date"])
    assert len(df) == 2


def test_fetch_weights_for_date_range_raises_when_no_rows() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []

    with pytest.raises(ValueError, match="No data found for date range"):
        fetch_weights_for_date_range(conn, "2024-01-01", "2024-12-31")


def test_plot_dca_weights_draws_boundary_for_mixed_past_and_future(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    boundary_calls: list[pd.Timestamp] = []
    original_axvline = Axes.axvline

    def _spy_axvline(self, *args, **kwargs):
        x = kwargs.get("x", args[0] if args else None)
        boundary_calls.append(pd.Timestamp(x))
        return original_axvline(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", _spy_axvline)
    monkeypatch.setattr("stacksats.plot_weights.plt.savefig", lambda *_args, **_kwargs: None)

    df = pd.DataFrame(
        {
            "DCA_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "weight": [0.2, 0.3, 0.25, 0.25],
            "btc_usd": [50000.0, 51000.0, None, None],
            "id": [1, 2, 3, 4],
        }
    )

    plot_dca_weights(
        df,
        start_date="2024-01-01",
        end_date="2024-01-04",
        output_path=str(tmp_path / "weights.svg"),
    )

    assert len(boundary_calls) == 1
    assert boundary_calls[0] == pd.Timestamp("2024-01-02")
