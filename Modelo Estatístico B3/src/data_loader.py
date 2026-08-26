"""
Módulo de carregamento e pré-processamento de dados de ativos da B3 via Yahoo Finance.
"""

from typing import List, Optional, Tuple
import datetime
import pandas as pd
import yfinance as yf

# Lista de tickers populares da B3 (Ibovespa)
DEFAULT_B3_TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
    "MGLU3.SA", "WEGE3.SA", "ABEV3.SA", "RENT3.SA", "PRIO3.SA",
    "SUZB3.SA", "B3SA3.SA", "GGBR4.SA", "CSNA3.SA", "ELET3.SA",
    "HAPV3.SA", "LREN3.SA", "RADL3.SA", "JBSS3.SA", "RAIZ4.SA",
    "AZUL4.SA", "CMIN3.SA", "EMBR3.SA", "CPLE6.SA", "VBBR3.SA",
    "RDOR3.SA", "UGPA3.SA", "ASAI3.SA", "KLBN11.SA", "CSAN3.SA"
]


def format_ticker(ticker: str) -> str:
    """Garante que o ticker termine com .SA para ativos brasileiros."""
    ticker_clean = ticker.strip().upper()
    if not ticker_clean.endswith(".SA") and not ticker_clean.startswith("^"):
        return f"{ticker_clean}.SA"
    return ticker_clean


def fetch_ticker_data(
    ticker: str,
    period: str = "6mo",
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Baixa dados diários de cotações para um ativo da B3.
    
    Parâmetros:
        ticker: Código da ação (ex: PETR4 ou PETR4.SA)
        period: Período padrão (ex: '6mo', '1y', '2y', 'max')
        start_date: Data inicial opcional (substitui period)
        end_date: Data final opcional (substitui period)
        auto_adjust: Ajustar por desdobramentos e proventos
        
    Retorna:
        DataFrame com colunas: Open, High, Low, Close, Volume, Prev_Close, Gap_Perc, Range_Perc
    """
    formatted_ticker = format_ticker(ticker)
    
    try:
        if start_date and end_date:
            df = yf.download(
                formatted_ticker,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,
                progress=False
            )
        else:
            df = yf.download(
                formatted_ticker,
                period=period,
                auto_adjust=auto_adjust,
                progress=False
            )
            
        if df.empty:
            return pd.DataFrame()
            
        # Tratar MultiIndex se retornado pelo yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        # Garantir colunas padrão
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()
                
        df = df[required_cols].copy()
        df.dropna(inplace=True)
        
        # Calcular Fechamento do dia anterior
        df["Prev_Close"] = df["Close"].shift(1)
        
        # Calcular Gap de abertura (%)
        df["Gap_Perc"] = ((df["Open"] - df["Prev_Close"]) / df["Prev_Close"]) * 100.0
        
        # Amplitude intraday máxima possível do dia (%)
        df["Range_Perc"] = ((df["High"] - df["Low"]) / df["Open"]) * 100.0
        
        # Variação do dia abertura até máxima (%)
        df["Max_Runup_Perc"] = ((df["High"] - df["Open"]) / df["Open"]) * 100.0
        
        # Variação do dia abertura até mínima (%)
        df["Max_Drawdown_Perc"] = ((df["Low"] - df["Open"]) / df["Open"]) * 100.0
        
        # Remover primeira linha que não tem Prev_Close
        df.dropna(subset=["Prev_Close"], inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Erro ao baixar dados para {ticker}: {e}")
        return pd.DataFrame()


def batch_fetch_tickers(
    tickers: List[str],
    period: str = "6mo"
) -> dict[str, pd.DataFrame]:
    """Baixa dados históricos para múltiplos ativos simultaneamente."""
    results = {}
    for t in tickers:
        df = fetch_ticker_data(t, period=period)
        if not df.empty and len(df) >= 20:
            results[format_ticker(t)] = df
    return results
