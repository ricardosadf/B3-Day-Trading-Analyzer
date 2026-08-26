"""
Testes unitários para o motor de estratégia e métricas de Day Trade na B3.
"""

import pytest
import pandas as pd
import numpy as np
from src.strategy_engine import run_daytrade_backtest
from src.metrics import calculate_performance_metrics


@pytest.fixture
def sample_ohlcv_data():
    """Gera um conjunto controlado de 10 dias para validação matemática exata."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
    # Cenários:
    # Dia 1: Open=100, High=115, Low=95, Close=105 -> Gatilho 97 atingido (Low 95 <= 97), Alvo 106.7 atingido (High 115 >= 106.7) -> TAKE_PROFIT
    # Dia 2: Open=100, High=102, Low=98, Close=99  -> Gatilho 97 NÃO atingido (Low 98 > 97) -> NO_TRADE
    # Dia 3: Open=100, High=104, Low=96, Close=98  -> Gatilho 97 atingido (Low 96 <= 97), Alvo 106.7 NÃO atingido -> Saída no Close 98
    data = {
        "Open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "High": [115.0, 102.0, 104.0, 112.0, 101.0, 115.0, 103.0, 114.0, 102.0, 115.0],
        "Low":  [95.0,  98.0,  96.0,  95.0,  99.0,  94.0,  96.0,  95.0,  98.0,  94.0],
        "Close": [105.0, 99.0,  98.0,  108.0, 100.0, 107.0, 97.0,  110.0, 99.0,  108.0],
        "Volume": [1000000] * 10
    }
    df = pd.DataFrame(data, index=dates)
    df["Prev_Close"] = df["Close"].shift(1).fillna(100.0)
    df["Gap_Perc"] = ((df["Open"] - df["Prev_Close"]) / df["Prev_Close"]) * 100.0
    return df


def test_strategy_execution(sample_ohlcv_data):
    # Desconto de 3% sobre abertura (Entrada em 97.0)
    # Alvo de 10% (Venda em 106.7)
    result = run_daytrade_backtest(
        df=sample_ohlcv_data,
        entry_mode="percent_below_open",
        entry_discount=3.0,
        target_profit_perc=10.0,
        stop_loss_perc=None,
        initial_capital=10000.0,
        b3_fee_perc=0.0
    )

    trades_df = result["trades_df"]
    assert len(trades_df) == 10

    # Dia 1 deve ser TAKE_PROFIT
    assert trades_df.iloc[0]["Triggered"] == True
    assert trades_df.iloc[0]["Outcome"] == "TAKE_PROFIT"
    assert trades_df.iloc[0]["Gross_Return_Perc"] == 10.0

    # Dia 2 não atingiu mínima de 97.0
    assert trades_df.iloc[1]["Triggered"] == False
    assert trades_df.iloc[1]["Outcome"] == "NO_TRADE"

    # Dia 3 atingiu 97.0 de entrada, mas máxima 104 < 106.7 -> saída no close 98.0
    assert trades_df.iloc[2]["Triggered"] == True
    assert trades_df.iloc[2]["Outcome"] == "EXIT_CLOSE_PROFIT"  # comprou em 97, saiu em 98 (+1.03%)


def test_metrics_calculation(sample_ohlcv_data):
    result = run_daytrade_backtest(
        df=sample_ohlcv_data,
        entry_mode="percent_below_open",
        entry_discount=3.0,
        target_profit_perc=10.0,
        initial_capital=10000.0,
        b3_fee_perc=0.0
    )

    metrics = calculate_performance_metrics(result, initial_capital=10000.0)

    assert metrics["total_days"] == 10
    assert metrics["num_trades"] > 0
    assert metrics["win_rate_perc"] >= 0.0
    assert "expectancy_perc" in metrics
    assert "profit_factor" in metrics
    assert "max_drawdown_perc" in metrics
