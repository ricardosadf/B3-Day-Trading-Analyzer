"""
Módulo do Motor de Simulação (Backtesting) de Estratégias de Day Trade na B3.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def run_daytrade_backtest(
    df: pd.DataFrame,
    entry_mode: str = "percent_below_open",
    entry_discount: float = 3.0,
    target_profit_perc: float = 10.0,
    stop_loss_perc: Optional[float] = None,
    initial_capital: float = 10000.0,
    b3_fee_perc: float = 0.03,  # Emolumentos e taxa de liquidação B3 (~0.03% por perna)
    slippage_perc: float = 0.0,  # Slippage estimado
    reinvest_profits: bool = False
) -> Dict[str, Any]:
    """
    Executa a simulação estatística de Day Trade no histórico do ativo.

    Parâmetros:
        df: DataFrame com cotações (Open, High, Low, Close, Prev_Close, Gap_Perc)
        entry_mode: Modo de entrada ('percent_below_open', 'points_below_open', 'percent_below_prev_close')
        entry_discount: Valor do desconto para entrada (ex: 3.0 para 3% ou 0.03 reais/pontos)
        target_profit_perc: Alvo de lucro percentual no mesmo dia (ex: 10.0%)
        stop_loss_perc: Limite de perda percentual (None para sair no fechamento do dia)
        initial_capital: Capital base alocado para a operação (R$)
        b3_fee_perc: Taxa B3 por perna da operação (%)
        slippage_perc: Desconto por slippage (%)
        reinvest_profits: Se True, acumula o capital nos trades; se False, aloca capital fixo por dia

    Retorna:
        Dicionário com:
          - trades_df: Tabela com o detalhe de cada pregão e operação
          - equity_curve: Evolução diária do capital
          - summary: Resumo agregado da estratégia
    """
    if df.empty or len(df) < 5:
        return {"trades_df": pd.DataFrame(), "equity_curve": pd.DataFrame(), "summary": {}}

    data = df.copy()
    
    trades = []
    current_capital = initial_capital
    equity_records = []

    for dt, row in data.iterrows():
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        prev_close = float(row.get("Prev_Close", open_price))
        gap_perc = float(row.get("Gap_Perc", 0.0))

        # Determinar Preço Alvo de Entrada (Gatilho)
        if entry_mode == "percent_below_open":
            target_entry_price = open_price * (1.0 - (entry_discount / 100.0))
        elif entry_mode == "points_below_open":
            target_entry_price = open_price - entry_discount
        elif entry_mode == "percent_below_prev_close":
            target_entry_price = prev_close * (1.0 - (entry_discount / 100.0))
        else:
            target_entry_price = open_price * (1.0 - (entry_discount / 100.0))

        # O gatilho foi acionado se a mínima do dia foi igual ou inferior ao preço de entrada
        trade_triggered = low_price <= target_entry_price

        trade_info = {
            "Date": dt,
            "Prev_Close": prev_close,
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close_price,
            "Gap_Perc": gap_perc,
            "Target_Entry": round(target_entry_price, 4),
            "Triggered": trade_triggered,
            "Entry_Price": np.nan,
            "Exit_Price": np.nan,
            "Target_Price": np.nan,
            "Stop_Price": np.nan,
            "Outcome": "NO_TRADE",
            "Gross_Return_Perc": 0.0,
            "Net_Return_Perc": 0.0,
            "Profit_BRL": 0.0,
            "Capital": current_capital
        }

        if trade_triggered and target_entry_price > 0:
            # Preço real de execução da entrada
            # Se o ativo abriu já abaixo do preço alvo, a compra é na abertura; senão é no target_entry_price
            entry_price = min(open_price, target_entry_price)
            
            # Ajuste de slippage na compra
            if slippage_perc > 0:
                entry_price *= (1.0 + (slippage_perc / 100.0))

            # Preço Alvo de Venda (Take Profit)
            take_profit_price = entry_price * (1.0 + (target_profit_perc / 100.0))
            
            # Preço de Stop Loss (se houver)
            stop_price = None
            if stop_loss_perc is not None and stop_loss_perc > 0:
                stop_price = entry_price * (1.0 - (stop_loss_perc / 100.0))

            trade_info["Entry_Price"] = round(entry_price, 4)
            trade_info["Target_Price"] = round(take_profit_price, 4)
            if stop_price:
                trade_info["Stop_Price"] = round(stop_price, 4)

            # Avaliar Desfecho Intraday
            # Cenário 1: Bateu no Alvo de Lucro (Take Profit)
            if high_price >= take_profit_price:
                exit_price = take_profit_price
                outcome = "TAKE_PROFIT"
                gross_ret = target_profit_perc
            # Cenário 2: Stop Loss acionado (caso configurado)
            elif stop_price is not None and low_price <= stop_price:
                exit_price = stop_price
                outcome = "STOP_LOSS"
                gross_ret = -stop_loss_perc
            # Cenário 3: Saída no Fechamento do dia (Day Trade obrigatório)
            else:
                exit_price = close_price
                gross_ret = ((exit_price - entry_price) / entry_price) * 100.0
                if gross_ret > 0:
                    outcome = "EXIT_CLOSE_PROFIT"
                elif gross_ret < 0:
                    outcome = "EXIT_CLOSE_LOSS"
                else:
                    outcome = "EXIT_CLOSE_EVEN"

            # Custos operacionais totais (entrada + saída)
            total_fee_perc = (b3_fee_perc * 2.0)
            net_ret = gross_ret - total_fee_perc

            trade_capital = current_capital if reinvest_profits else initial_capital
            profit_brl = trade_capital * (net_ret / 100.0)

            if reinvest_profits:
                current_capital = max(0.0, current_capital + profit_brl)
            else:
                current_capital += profit_brl

            trade_info["Exit_Price"] = round(exit_price, 4)
            trade_info["Outcome"] = outcome
            trade_info["Gross_Return_Perc"] = round(gross_ret, 4)
            trade_info["Net_Return_Perc"] = round(net_ret, 4)
            trade_info["Profit_BRL"] = round(profit_brl, 2)
            trade_info["Capital"] = round(current_capital, 2)

        trades.append(trade_info)
        equity_records.append({
            "Date": dt,
            "Capital": current_capital,
            "Profit_BRL": trade_info["Profit_BRL"],
            "Net_Return_Perc": trade_info["Net_Return_Perc"],
            "Triggered": trade_triggered
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_records)
    
    if "Date" in trades_df.columns:
        trades_df.set_index("Date", inplace=True)
    if "Date" in equity_df.columns:
        equity_df.set_index("Date", inplace=True)

    return {
        "trades_df": trades_df,
        "equity_curve": equity_df,
        "parameters": {
            "entry_mode": entry_mode,
            "entry_discount": entry_discount,
            "target_profit_perc": target_profit_perc,
            "stop_loss_perc": stop_loss_perc,
            "initial_capital": initial_capital,
            "b3_fee_perc": b3_fee_perc,
            "slippage_perc": slippage_perc,
            "reinvest_profits": reinvest_profits
        }
    }
