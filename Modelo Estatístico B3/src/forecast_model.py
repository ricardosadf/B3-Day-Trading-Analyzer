"""
Módulo de Modelagem Preditiva Estatística para Day Trade na B3.
Inclui Simulação de Monte Carlo com Bootstrap, Inferência Bayesiana e Testes de Hipótese.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def run_monte_carlo_simulation(
    equity_df: pd.DataFrame,
    future_days: int = 126,  # 6 meses úteis (~126 pregões)
    num_simulations: int = 5000,
    initial_capital: float = 10000.0,
    reinvest: bool = False,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executa a Simulação de Monte Carlo por Bootstrap dos retornos diários
    para projetar a distribuição de capital nos próximos 6 meses.

    Parâmetros:
        equity_df: Histórico com a coluna 'Profit_BRL' ou 'Net_Return_Perc'
        future_days: Número de pregões futuros a simular (~126 para 6 meses)
        num_simulations: Quantidade de caminhos simulados
        initial_capital: Capital de partida
        reinvest: Se True, juros compostos; se False, resultado somatório fixo
        random_seed: Semente para reprodutibilidade

    Retorna:
        Dicionário com curvas percentílicas, probabilidades de lucro e distribuição final.
    """
    if equity_df.empty or "Profit_BRL" not in equity_df.columns:
        return {}

    np.random.seed(random_seed)

    # Amostra histórica de resultados diários (inclui dias com 0 e dias com trade)
    historical_profits = equity_df["Profit_BRL"].values
    historical_returns_perc = equity_df["Net_Return_Perc"].values

    if len(historical_profits) == 0:
        return {}

    # Matriz de simulação: [num_simulations, future_days]
    sampled_indices = np.random.choice(
        len(historical_profits),
        size=(num_simulations, future_days),
        replace=True
    )

    if reinvest:
        # Crescimento multiplicativo
        sampled_returns = (historical_returns_perc[sampled_indices] / 100.0) + 1.0
        # Cumprod ao longo dos dias
        cumulative_paths = initial_capital * np.cumprod(sampled_returns, axis=1)
        # Adicionar o ponto inicial (dia 0)
        initial_col = np.full((num_simulations, 1), initial_capital)
        cumulative_paths = np.hstack([initial_col, cumulative_paths])
    else:
        # Crescimento aditivo (capital fixo por trade)
        sampled_profits = historical_profits[sampled_indices]
        cumulative_profits = np.cumsum(sampled_profits, axis=1)
        cumulative_paths = initial_capital + cumulative_profits
        initial_col = np.full((num_simulations, 1), initial_capital)
        cumulative_paths = np.hstack([initial_col, cumulative_paths])

    final_capitals = cumulative_paths[:, -1]
    final_profits_brl = final_capitals - initial_capital
    final_returns_perc = (final_profits_brl / initial_capital) * 100.0

    # Probabilidades Chave
    prob_profit = float(np.mean(final_profits_brl > 0) * 100.0)
    prob_double = float(np.mean(final_capitals >= (initial_capital * 2.0)) * 100.0)
    prob_loss_10 = float(np.mean(final_capitals <= (initial_capital * 0.90)) * 100.0)
    prob_loss_20 = float(np.mean(final_capitals <= (initial_capital * 0.80)) * 100.0)

    # Cálculo dos Cones Percentílicos dia a dia
    # Percentis: 5%, 25%, 50% (mediana), 75%, 95%
    p5_curve = np.percentile(cumulative_paths, 5, axis=0)
    p25_curve = np.percentile(cumulative_paths, 25, axis=0)
    p50_curve = np.percentile(cumulative_paths, 50, axis=0)
    p75_curve = np.percentile(cumulative_paths, 75, axis=0)
    p95_curve = np.percentile(cumulative_paths, 95, axis=0)

    # Cálculo do Max Drawdown em cada simulação
    peaks = np.maximum.accumulate(cumulative_paths, axis=1)
    drawdowns = (peaks - cumulative_paths) / peaks * 100.0
    sim_max_drawdowns = np.max(drawdowns, axis=1)
    median_max_dd = float(np.median(sim_max_drawdowns))
    p95_max_dd = float(np.percentile(sim_max_drawdowns, 95))

    days_axis = list(range(future_days + 1))

    return {
        "future_days": future_days,
        "num_simulations": num_simulations,
        "initial_capital": initial_capital,
        "prob_profit_perc": round(prob_profit, 2),
        "prob_double_perc": round(prob_double, 2),
        "prob_loss_10_perc": round(prob_loss_10, 2),
        "prob_loss_20_perc": round(prob_loss_20, 2),
        "median_final_capital": round(float(np.median(final_capitals)), 2),
        "median_final_profit_brl": round(float(np.median(final_profits_brl)), 2),
        "median_final_return_perc": round(float(np.median(final_returns_perc)), 2),
        "p5_final_profit_brl": round(float(np.percentile(final_profits_brl, 5)), 2),
        "p95_final_profit_brl": round(float(np.percentile(final_profits_brl, 95)), 2),
        "expected_max_drawdown_perc": round(median_max_dd, 2),
        "worst_case_max_drawdown_perc": round(p95_max_dd, 2),
        "percentile_curves": {
            "days": days_axis,
            "p5": [round(x, 2) for x in p5_curve],
            "p25": [round(x, 2) for x in p25_curve],
            "p50": [round(x, 2) for x in p50_curve],
            "p75": [round(x, 2) for x in p75_curve],
            "p95": [round(x, 2) for x in p95_curve],
        },
        "final_returns_sample": final_returns_perc[:1000].tolist()
    }


def bayesian_recurrence_analysis(
    total_events: int,
    successful_events: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0
) -> Dict[str, Any]:
    """
    Estima a probabilidade futura de recorrência usando Inferência Bayesiana (Modelo Beta-Binomial).

    Parâmetros:
        total_events: Total de oportunidades / trades observados no semestre
        successful_events: Número de acertos (ex: batimento do alvo ou trades positivos)
        prior_alpha, prior_beta: Parâmetros da distribuição a priori (Beta(1,1) = Uniforme)

    Retorna:
        Estatísticas a posteriori (média, mediana, intervalo de credibilidade de 95%).
    """
    if total_events <= 0:
        return {
            "posterior_mean_perc": 0.0,
            "ci_lower_perc": 0.0,
            "ci_upper_perc": 0.0,
            "prob_above_50_perc": 0.0,
            "prob_above_60_perc": 0.0,
        }

    post_alpha = prior_alpha + successful_events
    post_beta = prior_beta + (total_events - successful_events)

    # Distribuição a posteriori
    post_dist = stats.beta(post_alpha, post_beta)

    post_mean = post_dist.mean() * 100.0
    post_median = post_dist.median() * 100.0
    ci_lower = post_dist.ppf(0.025) * 100.0
    ci_upper = post_dist.ppf(0.975) * 100.0

    # Probabilidades de a taxa real ser maior que 50% ou 60%
    prob_above_50 = (1.0 - post_dist.cdf(0.50)) * 100.0
    prob_above_60 = (1.0 - post_dist.cdf(0.60)) * 100.0

    return {
        "posterior_mean_perc": round(post_mean, 2),
        "posterior_median_perc": round(post_median, 2),
        "ci_lower_perc": round(ci_lower, 2),
        "ci_upper_perc": round(ci_upper, 2),
        "prob_above_50_perc": round(prob_above_50, 2),
        "prob_above_60_perc": round(prob_above_60, 2),
        "post_alpha": post_alpha,
        "post_beta": post_beta
    }


def statistical_significance_test(returns: np.ndarray) -> Dict[str, Any]:
    """
    Testa se o retorno médio diário é estatisticamente diferente de zero (T-Test).
    """
    if len(returns) < 5:
        return {"p_value": 1.0, "is_significant": False, "t_stat": 0.0}

    t_stat, p_val = stats.ttest_1samp(returns, 0.0, alternative="greater")

    return {
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "is_significant": bool(p_val < 0.05),
        "confidence_level_perc": round((1.0 - p_val) * 100.0, 2) if p_val < 1.0 else 0.0
    }
