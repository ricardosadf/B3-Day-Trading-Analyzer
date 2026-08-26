"""
Testes unitários para simulação de Monte Carlo e Inferência Bayesiana.
"""

import pytest
import pandas as pd
import numpy as np
from src.forecast_model import run_monte_carlo_simulation, bayesian_recurrence_analysis, statistical_significance_test


def test_monte_carlo_simulation():
    # 50 dias com retornos diários simulados
    dates = pd.date_range(start="2024-01-01", periods=50, freq="B")
    profits = np.random.normal(loc=150.0, scale=300.0, size=50)
    returns = profits / 10000.0 * 100.0

    equity_df = pd.DataFrame({
        "Capital": 10000.0 + np.cumsum(profits),
        "Profit_BRL": profits,
        "Net_Return_Perc": returns,
        "Triggered": [True] * 50
    }, index=dates)

    mc_result = run_monte_carlo_simulation(
        equity_df=equity_df,
        future_days=126,
        num_simulations=1000,
        initial_capital=10000.0,
        reinvest=False
    )

    assert "prob_profit_perc" in mc_result
    assert 0.0 <= mc_result["prob_profit_perc"] <= 100.0
    assert "percentile_curves" in mc_result
    assert len(mc_result["percentile_curves"]["days"]) == 127
    assert len(mc_result["percentile_curves"]["p50"]) == 127


def test_bayesian_recurrence_analysis():
    # 20 trades, 14 acertos do alvo
    res = bayesian_recurrence_analysis(total_events=20, successful_events=14)

    assert "posterior_mean_perc" in res
    assert "ci_lower_perc" in res
    assert "ci_upper_perc" in res
    assert res["ci_lower_perc"] < res["posterior_mean_perc"] < res["ci_upper_perc"]
    assert 0.0 <= res["prob_above_50_perc"] <= 100.0


def test_statistical_significance():
    # Retornos positivos consistentes
    returns = np.array([2.5, 3.1, -1.0, 4.0, 2.8, 3.5, 1.2, 5.0, -0.5, 3.0])
    sig = statistical_significance_test(returns)

    assert "p_value" in sig
    assert "is_significant" in sig
    assert sig["p_value"] < 0.05
