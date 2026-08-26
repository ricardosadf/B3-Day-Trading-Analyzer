"""
Módulo de Otimização e Varredura de Parâmetros (Grid Search) para Day Trade na B3.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from src.strategy_engine import run_daytrade_backtest
from src.metrics import calculate_performance_metrics


def run_parameter_grid_search(
    df: pd.DataFrame,
    entry_mode: str = "percent_below_open",
    discounts: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    targets: List[float] = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0],
    stop_loss_perc: Any = None,
    initial_capital: float = 10000.0,
    b3_fee_perc: float = 0.03
) -> pd.DataFrame:
    """
    Realiza uma varredura bidimensional de parâmetros (Gatilho de Entrada vs. Alvo de Lucro)
    para encontrar a combinação ideal de rentabilidade, taxa de acerto e expectativa matemática.

    Retorna:
        DataFrame com colunas: Discount, Target, Win_Rate_Perc, Total_Return_Perc,
        Total_Profit_BRL, Num_Trades, Target_Hits, Profit_Factor, Expectancy_Perc
    """
    if df.empty:
        return pd.DataFrame()

    results = []

    for disc in discounts:
        for tgt in targets:
            bt_res = run_daytrade_backtest(
                df=df,
                entry_mode=entry_mode,
                entry_discount=disc,
                target_profit_perc=tgt,
                stop_loss_perc=stop_loss_perc,
                initial_capital=initial_capital,
                b3_fee_perc=b3_fee_perc
            )

            metrics = calculate_performance_metrics(bt_res, initial_capital=initial_capital)

            results.append({
                "Desconto_Entrada": disc,
                "Alvo_Lucro_Perc": tgt,
                "Trades": metrics.get("num_trades", 0),
                "Alvos_Atingidos": metrics.get("target_hits", 0),
                "Taxa_Acerto_Perc": metrics.get("win_rate_perc", 0.0),
                "Taxa_Alvo_Trades_Perc": metrics.get("target_hit_rate_trades", 0.0),
                "Retorno_Total_Perc": metrics.get("total_return_perc", 0.0),
                "Lucro_Total_BRL": metrics.get("total_profit_brl", 0.0),
                "Expectativa_Perc": metrics.get("expectancy_perc", 0.0),
                "Profit_Factor": metrics.get("profit_factor", 0.0),
                "Max_Drawdown_Perc": metrics.get("max_drawdown_perc", 0.0)
            })

    return pd.DataFrame(results)
