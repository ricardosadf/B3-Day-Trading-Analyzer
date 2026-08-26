"""
Módulo de cálculo de métricas estatísticas, financeiras e de recorrência para Day Trade.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def calculate_performance_metrics(
    backtest_result: Dict[str, Any],
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Calcula métricas aprofundadas de recorrência, probabilidade e risco/retorno.
    """
    trades_df = backtest_result.get("trades_df", pd.DataFrame())
    equity_df = backtest_result.get("equity_curve", pd.DataFrame())

    if trades_df.empty:
        return {}

    total_days = len(trades_df)
    triggered_trades = trades_df[trades_df["Triggered"] == True].copy()
    num_trades = len(triggered_trades)

    if num_trades == 0:
        return {
            "total_days": total_days,
            "num_trades": 0,
            "trigger_rate_perc": 0.0,
            "target_hits": 0,
            "target_hit_rate_trades": 0.0,
            "target_hit_rate_all_days": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
            "even_trades": 0,
            "win_rate_perc": 0.0,
            "total_profit_brl": 0.0,
            "total_return_perc": 0.0,
            "avg_trade_return_perc": 0.0,
            "avg_win_perc": 0.0,
            "avg_loss_perc": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy_perc": 0.0,
            "max_drawdown_perc": 0.0,
            "max_drawdown_brl": 0.0,
            "sharpe_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    # Desfechos
    target_hits = (triggered_trades["Outcome"] == "TAKE_PROFIT").sum()
    win_trades = (triggered_trades["Net_Return_Perc"] > 0).sum()
    loss_trades = (triggered_trades["Net_Return_Perc"] < 0).sum()
    even_trades = (triggered_trades["Net_Return_Perc"] == 0).sum()

    trigger_rate_perc = (num_trades / total_days) * 100.0
    target_hit_rate_trades = (target_hits / num_trades) * 100.0
    target_hit_rate_all_days = (target_hits / total_days) * 100.0
    win_rate_perc = (win_trades / num_trades) * 100.0

    # Retornos
    returns = triggered_trades["Net_Return_Perc"].values
    profits_brl = triggered_trades["Profit_BRL"].values

    total_profit_brl = float(np.sum(profits_brl))
    total_return_perc = (total_profit_brl / initial_capital) * 100.0
    avg_trade_return = float(np.mean(returns))

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = float(np.mean(gains)) if len(gains) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0

    # Payoff
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (avg_win if avg_win > 0 else 0.0)

    # Profit Factor
    sum_gains_brl = float(np.sum(profits_brl[profits_brl > 0])) if np.any(profits_brl > 0) else 0.0
    sum_losses_brl = float(abs(np.sum(profits_brl[profits_brl < 0]))) if np.any(profits_brl < 0) else 0.0
    profit_factor = (sum_gains_brl / sum_losses_brl) if sum_losses_brl > 0 else (999.0 if sum_gains_brl > 0 else 0.0)

    # Expectativa Matemática por operação (E = (P_win * Avg_Win) - (P_loss * Avg_Loss))
    prob_win = win_trades / num_trades
    prob_loss = loss_trades / num_trades
    expectancy_perc = (prob_win * avg_win) - (prob_loss * avg_loss)

    # Drawdown na curva de capital
    capital_series = equity_df["Capital"] if "Capital" in equity_df.columns else pd.Series([initial_capital])
    peak = capital_series.cummax()
    drawdown_brl_series = peak - capital_series
    drawdown_perc_series = (drawdown_brl_series / peak) * 100.0

    max_drawdown_brl = float(drawdown_brl_series.max())
    max_drawdown_perc = float(drawdown_perc_series.max())

    # Sharpe Ratio Diário Anualizado (252 dias úteis)
    all_daily_returns = equity_df["Net_Return_Perc"].values if "Net_Return_Perc" in equity_df.columns else returns
    std_dev = float(np.std(all_daily_returns))
    sharpe_ratio = (np.mean(all_daily_returns) / std_dev * np.sqrt(252)) if std_dev > 0 else 0.0

    # Sequências consecutivas
    is_win = (triggered_trades["Net_Return_Perc"] > 0).astype(int).values
    max_consec_wins = 0
    max_consec_losses = 0
    curr_wins = 0
    curr_losses = 0

    for res in is_win:
        if res == 1:
            curr_wins += 1
            curr_losses = 0
            if curr_wins > max_consec_wins:
                max_consec_wins = curr_wins
        else:
            curr_losses += 1
            curr_wins = 0
            if curr_losses > max_consec_losses:
                max_consec_losses = curr_losses

    return {
        "total_days": int(total_days),
        "num_trades": int(num_trades),
        "trigger_rate_perc": round(trigger_rate_perc, 2),
        "target_hits": int(target_hits),
        "target_hit_rate_trades": round(target_hit_rate_trades, 2),
        "target_hit_rate_all_days": round(target_hit_rate_all_days, 2),
        "win_trades": int(win_trades),
        "loss_trades": int(loss_trades),
        "even_trades": int(even_trades),
        "win_rate_perc": round(win_rate_perc, 2),
        "total_profit_brl": round(total_profit_brl, 2),
        "total_return_perc": round(total_return_perc, 2),
        "avg_trade_return_perc": round(avg_trade_return, 2),
        "avg_win_perc": round(avg_win, 2),
        "avg_loss_perc": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_perc": round(expectancy_perc, 2),
        "max_drawdown_perc": round(max_drawdown_perc, 2),
        "max_drawdown_brl": round(max_drawdown_brl, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_consecutive_wins": int(max_consec_wins),
        "max_consecutive_losses": int(max_consec_losses),
        "std_dev_returns": round(std_dev, 2)
    }
