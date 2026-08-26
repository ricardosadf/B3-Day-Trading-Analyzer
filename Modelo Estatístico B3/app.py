"""
Sistema de Análise Estatística & Preditiva de Day Trade na B3
Dashboard Interativo Streamlit
"""

import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io

from src.data_loader import fetch_ticker_data, format_ticker, DEFAULT_B3_TICKERS, batch_fetch_tickers
from src.strategy_engine import run_daytrade_backtest
from src.metrics import calculate_performance_metrics
from src.forecast_model import (
    run_monte_carlo_simulation,
    bayesian_recurrence_analysis,
    statistical_significance_test
)
from src.optimizer import run_parameter_grid_search

# Configuração de Página Streamlit
st.set_page_config(
    page_title="Day Trade B3 - Modelo Estatístico & Preditivo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_ticker_data(ticker: str, period: str):
    return fetch_ticker_data(ticker=ticker, period=period)


def main():
    st.markdown('<div class="main-header">📈 Modelo Estatístico & Preditivo - Day Trade B3</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Análise de recorrência, backtesting intraday e projeção probabilística para os próximos 6 meses</div>', unsafe_allow_html=True)

    # BARRA LATERAL: PARÂMETROS
    st.sidebar.header("⚙️ Configurações do Setup")

    # Seleção do Ativo
    ticker_mode = st.sidebar.radio("Seleção do Ativo:", ["Lista Principal B3", "Digitar Código Customizado"], horizontal=True)
    if ticker_mode == "Lista Principal B3":
        selected_ticker = st.sidebar.selectbox("Ativo da B3:", DEFAULT_B3_TICKERS, index=0)
    else:
        custom_input = st.sidebar.text_input("Código do Ativo (ex: PETR4, VALE3, MGLU3):", value="PETR4")
        selected_ticker = format_ticker(custom_input)

    # Período Histórico
    period_options = {
        "Últimos 6 Meses (1 Semestre)": "6mo",
        "Últimos 3 Meses": "3mo",
        "Último 1 Ano": "1y",
        "Últimos 2 Anos": "2y"
    }
    selected_period_label = st.sidebar.selectbox("Histórico para Análise:", list(period_options.keys()), index=0)
    selected_period = period_options[selected_period_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Regras Operacionais")

    entry_mode_map = {
        "Desconto % sobre a Abertura do dia": "percent_below_open",
        "Desconto em Reais/Pontos abaixo da Abertura": "points_below_open",
        "Desconto % sobre o Fechamento Anterior": "percent_below_prev_close"
    }
    entry_mode_label = st.sidebar.selectbox("Critério de Entrada (Gatilho de Compra):", list(entry_mode_map.keys()), index=0)
    entry_mode = entry_mode_map[entry_mode_label]

    if entry_mode == "points_below_open":
        entry_discount = st.sidebar.number_input("Desconto de Entrada (Pontos/Reais):", min_value=0.01, max_value=20.0, value=0.30, step=0.05)
    else:
        entry_discount = st.sidebar.number_input("Desconto de Entrada (%):", min_value=0.1, max_value=15.0, value=3.0, step=0.5, help="Compra executada se o papel recuar X% em relação à referência.")

    target_profit_perc = st.sidebar.slider("Alvo de Lucro Intraday (Take Profit %):", min_value=1.0, max_value=20.0, value=10.0, step=0.5, help="Percentual de ganho desejado para encerrar com lucro no dia.")

    use_stop_loss = st.sidebar.checkbox("Configurar Stop Loss fixo", value=False, help="Se desmarcado, posições não atingidas são encerradas no fechamento do dia.")
    stop_loss_perc = None
    if use_stop_loss:
        stop_loss_perc = st.sidebar.slider("Limite de Perda (Stop Loss %):", min_value=1.0, max_value=15.0, value=4.0, step=0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Gestão de Capital & Custos")
    initial_capital = st.sidebar.number_input("Capital Base por Operação (R$):", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0)
    reinvest_profits = st.sidebar.checkbox("Reinvestir Lucros (Juros Compostos)", value=False)
    b3_fee_perc = st.sidebar.number_input("Taxas B3 / Emolumentos por perna (%):", min_value=0.0, max_value=1.0, value=0.03, step=0.01)

    # Carregar Dados do Ativo Selecionado
    with st.spinner(f"Carregando cotações de {selected_ticker}..."):
        df_ticker = load_cached_ticker_data(selected_ticker, selected_period)

    if df_ticker.empty:
        st.error(f"❌ Não foi possível carregar dados para o ticker **{selected_ticker}**. Verifique se o código está correto na B3.")
        return

    # Executar Backtest
    backtest_result = run_daytrade_backtest(
        df=df_ticker,
        entry_mode=entry_mode,
        entry_discount=entry_discount,
        target_profit_perc=target_profit_perc,
        stop_loss_perc=stop_loss_perc,
        initial_capital=initial_capital,
        b3_fee_perc=b3_fee_perc,
        reinvest_profits=reinvest_profits
    )

    metrics = calculate_performance_metrics(backtest_result, initial_capital=initial_capital)
    trades_df = backtest_result["trades_df"]
    equity_df = backtest_result["equity_curve"]

    # ABAS PRINCIPAIS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Backtest & Recorrência",
        "🔮 Previsão Próximos 6 Meses",
        "🚀 Scanner B3 (Ranking)",
        "🎯 Otimizador de Alvos",
        "📥 Tabela de Trades & Exportação"
    ])

    # -------------------------------------------------------------
    # TAB 1: BACKTEST & RECORRÊNCIA
    # -------------------------------------------------------------
    with tab1:
        st.markdown(f"### 📈 Desempenho Histórico: **{selected_ticker}** ({selected_period_label})")
        
        # Linha 1 de Métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Pregões Analisados", f"{metrics.get('total_days', 0)} dias", f"{metrics.get('num_trades', 0)} disparos")
        with col2:
            st.metric("Taxa de Disparo (Oportunidades)", f"{metrics.get('trigger_rate_perc', 0.0)}%", f"{metrics.get('target_hits', 0)} alvos atingidos")
        with col3:
            st.metric("Taxa de Acerto (Win Rate)", f"{metrics.get('win_rate_perc', 0.0)}%", f"Alvo {target_profit_perc}%: {metrics.get('target_hit_rate_trades', 0.0)}%")
        with col4:
            profit_brl = metrics.get('total_profit_brl', 0.0)
            ret_perc = metrics.get('total_return_perc', 0.0)
            st.metric("Lucro Líquido Acumulado", f"R$ {profit_brl:,.2f}", f"{ret_perc:+.2f}%")
        with col5:
            st.metric("Expectativa Matemática", f"{metrics.get('expectancy_perc', 0.0):+.2f}% / trade", f"Payoff: {metrics.get('payoff_ratio', 0.0):.2f}")

        # Linha 2 de Métricas
        col6, col7, col8, col9 = st.columns(4)
        with col6:
            st.metric("Fator de Lucro (Profit Factor)", f"{metrics.get('profit_factor', 0.0):.2f}")
        with col7:
            st.metric("Max Drawdown", f"-{metrics.get('max_drawdown_perc', 0.0):.2f}%", f"-R$ {metrics.get('max_drawdown_brl', 0.0):,.2f}")
        with col8:
            st.metric("Ganho Médio vs Perda Média", f"+{metrics.get('avg_win_perc', 0.0):.2f}% / -{metrics.get('avg_loss_perc', 0.0):.2f}%")
        with col9:
            st.metric("Índice Sharpe Anualizado", f"{metrics.get('sharpe_ratio', 0.0):.2f}")

        st.markdown("---")

        # Gráfico da Curva de Capital (Equity Curve)
        st.subheader("💰 Curva de Evolução Patrimonial (R$)")
        fig_equity = px.line(
            equity_df.reset_index(),
            x="Date",
            y="Capital",
            title=f"Evolução do Capital - Setup {selected_ticker} (Capital Inicial R$ {initial_capital:,.2f})",
            labels={"Date": "Data", "Capital": "Patrimônio Líquido (R$)"},
            color_discrete_sequence=["#10B981"]
        )
        fig_equity.update_layout(
            hovermode="x unified",
            template="plotly_white",
            height=380
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Gráfico Candlestick com Marcações de Trades
        st.subheader("🕯️ Gráfico de Candlestick com Trades Executados")
        
        candle_df = df_ticker.copy().reset_index()
        fig_candle = go.Figure()
        
        # Candles
        fig_candle.add_trace(go.Candlestick(
            x=candle_df["Date"],
            open=candle_df["Open"],
            high=candle_df["High"],
            low=candle_df["Low"],
            close=candle_df["Close"],
            name="Cotação Diária",
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444"
        ))

        # Adicionar marcadores de compras e saídas
        triggered_trades = trades_df[trades_df["Triggered"] == True].reset_index()
        
        if not triggered_trades.empty:
            # Compras
            fig_candle.add_trace(go.Scatter(
                x=triggered_trades["Date"],
                y=triggered_trades["Entry_Price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#3B82F6"),
                name="Entrada (Compra)"
            ))
            
            # Take Profit Alvo Atingido
            tp_trades = triggered_trades[triggered_trades["Outcome"] == "TAKE_PROFIT"]
            if not tp_trades.empty:
                fig_candle.add_trace(go.Scatter(
                    x=tp_trades["Date"],
                    y=tp_trades["Exit_Price"],
                    mode="markers",
                    marker=dict(symbol="star", size=12, color="#EAB308"),
                    name=f"Take Profit (+{target_profit_perc}%)"
                ))

            # Saídas no Fechamento / Stop
            other_trades = triggered_trades[triggered_trades["Outcome"] != "TAKE_PROFIT"]
            if not other_trades.empty:
                fig_candle.add_trace(go.Scatter(
                    x=other_trades["Date"],
                    y=other_trades["Exit_Price"],
                    mode="markers",
                    marker=dict(symbol="x", size=9, color="#9333EA"),
                    name="Saída Fechamento/Stop"
                ))

        fig_candle.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: PREVISÃO PRÓXIMOS 6 MESES
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### 🔮 Projeção Estatística e Previsão para os Próximos 6 Meses")
        st.write("Modelo estocástico baseado em **Simulação de Monte Carlo com Bootstrap** (5.000 iterações sobre a distribuição real do histórico) e **Inferência Bayesiana Beta-Binomial**.")

        if metrics.get("num_trades", 0) < 3:
            st.warning("⚠️ Poucos trades executados no período selecionado para gerar uma projeção estatística confiável. Experimente ajustar o desconto de entrada ou aumentar o histórico.")
        else:
            col_mc1, col_mc2 = st.columns([2, 1])

            with col_mc2:
                st.subheader("⚙️ Parâmetros da Simulação")
                sim_days = st.slider("Horizonte de Previsão (Pregões):", min_value=20, max_value=252, value=126, step=10, help="126 pregões úteis correspondem a aproximadamente 6 meses.")
                num_sims = st.selectbox("Iterações de Monte Carlo:", [1000, 3000, 5000, 10000], index=2)

                mc_res = run_monte_carlo_simulation(
                    equity_df=equity_df,
                    future_days=sim_days,
                    num_simulations=num_sims,
                    initial_capital=initial_capital,
                    reinvest=reinvest_profits
                )

                # Análise Bayesiana
                bayes_res = bayesian_recurrence_analysis(
                    total_events=metrics.get("num_trades", 0),
                    successful_events=metrics.get("target_hits", 0)
                )

                # Teste de Significância
                sig_res = statistical_significance_test(trades_df[trades_df["Triggered"] == True]["Net_Return_Perc"].values)

                st.markdown("---")
                st.markdown("#### 🎯 Probabilidades Preditivas")
                st.metric("Probabilidade de Lucro > 0", f"{mc_res.get('prob_profit_perc', 0.0)}%", help="Chance estimada de fechar o próximo semestre no positivo.")
                st.metric("Lucro Mediano Esperado", f"R$ {mc_res.get('median_final_profit_brl', 0.0):,.2f}", f"{mc_res.get('median_final_return_perc', 0.0):+.2f}%")
                st.metric("Pior Cenário Provável (VaR 95%)", f"R$ {mc_res.get('p5_final_profit_brl', 0.0):,.2f}")
                st.metric("Drawdown Máximo Esperado", f"{mc_res.get('expected_max_drawdown_perc', 0.0):.2f}%")

            with col_mc1:
                st.subheader("📊 Cone de Probabilidades - Projeção de Capital")
                
                # Plotar Cone de Monte Carlo
                p_curves = mc_res.get("percentile_curves", {})
                days_x = p_curves.get("days", [])
                
                fig_mc = go.Figure()
                
                # Faixa P5 - P95 (90% de confiança)
                fig_mc.add_trace(go.Scatter(
                    x=days_x + days_x[::-1],
                    y=p_curves.get("p95", []) + p_curves.get("p5", [])[::-1],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalo 90% (P5 - P95)'
                ))
                
                # Faixa P25 - P75 (50% de confiança)
                fig_mc.add_trace(go.Scatter(
                    x=days_x + days_x[::-1],
                    y=p_curves.get("p75", []) + p_curves.get("p25", [])[::-1],
                    fill='toself',
                    fillcolor='rgba(59, 130, 246, 0.30)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalo 50% (P25 - P75)'
                ))

                # Linha Mediana P50
                fig_mc.add_trace(go.Scatter(
                    x=days_x,
                    y=p_curves.get("p50", []),
                    mode='lines',
                    line=dict(color='#2563EB', width=3),
                    name='Cenário Mediano (P50)'
                ))

                # Linha de Capital Inicial (Breakeven)
                fig_mc.add_hline(y=initial_capital, line_dash="dash", line_color="#EF4444", annotation_text="Capital Inicial")

                fig_mc.update_layout(
                    title=f"Cone de Monte Carlo ({num_sims:,} simulações para os próximos {sim_days} pregões)",
                    xaxis_title="Dias Futuros de Pregão",
                    yaxis_title="Patrimônio Projetado (R$)",
                    template="plotly_white",
                    height=450,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_mc, use_container_width=True)

            st.markdown("---")
            st.subheader("🔬 Inferência Bayesiana da Taxa de Acerto do Alvo")
            
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            with bcol1:
                st.metric("Taxa Estimada a Posteriori", f"{bayes_res.get('posterior_mean_perc', 0.0)}%")
            with bcol2:
                st.metric("Intervalo de Credibilidade (95%)", f"[{bayes_res.get('ci_lower_perc', 0.0)}% - {bayes_res.get('ci_upper_perc', 0.0)}%]")
            with bcol3:
                st.metric("Probabilidade de Taxa > 50%", f"{bayes_res.get('prob_above_50_perc', 0.0)}%")
            with bcol4:
                st.metric("Significância Estatística (p-valor)", f"{sig_res.get('p_value', 1.0):.4f}", "Significante (p < 0.05)" if sig_res.get("is_significant") else "Não Significante")

    # -------------------------------------------------------------
    # TAB 3: SCANNER B3 (RANKING)
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### 🚀 Scanner de Oportunidades na B3")
        st.write("Varredura comparativa das principais ações da bolsa com os parâmetros atuais para identificar os papéis mais lucrativos e recorrentes.")

        if st.button("🔍 Executar Scanner em Lote nos Papéis da B3", type="primary"):
            with st.spinner("Analisando múltiplos papéis da B3..."):
                scan_results = []
                for t in DEFAULT_B3_TICKERS[:15]:  # Top 15 mais líquidos para resposta rápida
                    df_t = load_cached_ticker_data(t, selected_period)
                    if not df_t.empty:
                        bt_t = run_daytrade_backtest(
                            df=df_t,
                            entry_mode=entry_mode,
                            entry_discount=entry_discount,
                            target_profit_perc=target_profit_perc,
                            stop_loss_perc=stop_loss_perc,
                            initial_capital=initial_capital,
                            b3_fee_perc=b3_fee_perc
                        )
                        m_t = calculate_performance_metrics(bt_t, initial_capital=initial_capital)
                        scan_results.append({
                            "Ativo": t.replace(".SA", ""),
                            "Trades": m_t.get("num_trades", 0),
                            "Taxa de Disparo (%)": m_t.get("trigger_rate_perc", 0.0),
                            "Alvos Atingidos": m_t.get("target_hits", 0),
                            "Taxa de Acerto (%)": m_t.get("win_rate_perc", 0.0),
                            "Retorno Total (%)": m_t.get("total_return_perc", 0.0),
                            "Lucro Líquido (R$)": m_t.get("total_profit_brl", 0.0),
                            "Expectativa (%)": m_t.get("expectancy_perc", 0.0),
                            "Profit Factor": m_t.get("profit_factor", 0.0),
                            "Sharpe": m_t.get("sharpe_ratio", 0.0),
                            "Max Drawdown (%)": m_t.get("max_drawdown_perc", 0.0)
                        })

                if scan_results:
                    scan_df = pd.DataFrame(scan_results).sort_values(by="Lucro Líquido (R$)", ascending=False).reset_index(drop=True)
                    st.dataframe(
                        scan_df.style.format({
                            "Taxa de Disparo (%)": "{:.1f}%",
                            "Taxa de Acerto (%)": "{:.1f}%",
                            "Retorno Total (%)": "{:+.2f}%",
                            "Lucro Líquido (R$)": "R$ {:,.2f}",
                            "Expectativa (%)": "{:+.2f}%",
                            "Profit Factor": "{:.2f}",
                            "Sharpe": "{:.2f}",
                            "Max Drawdown (%)": "{:.2f}%"
                        }).background_gradient(subset=["Retorno Total (%)", "Expectativa (%)"], cmap="RdYlGn"),
                        use_container_width=True
                    )

                    # Gráfico de barras do ranking
                    fig_rank = px.bar(
                        scan_df,
                        x="Ativo",
                        y="Retorno Total (%)",
                        color="Retorno Total (%)",
                        title="Ranking de Rentabilidade por Ativo no Semestre (%)",
                        color_continuous_scale="RdYlGn"
                    )
                    st.plotly_chart(fig_rank, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: OTIMIZADOR DE ALVOS (HEATMAP)
    # -------------------------------------------------------------
    with tab4:
        st.markdown(f"### 🎯 Otimizador de Parâmetros e Mapa de Calor: **{selected_ticker}**")
        st.write("Avalie como diferentes níveis de **Desconto de Entrada** e **Alvo de Lucro (Take Profit)** afetam o retorno total e a taxa de acerto.")

        if st.button("🔥 Gerar Matriz de Otimização", type="secondary"):
            with st.spinner("Processando combinações de parâmetros..."):
                grid_df = run_parameter_grid_search(
                    df=df_ticker,
                    entry_mode=entry_mode,
                    discounts=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                    targets=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0],
                    stop_loss_perc=stop_loss_perc,
                    initial_capital=initial_capital,
                    b3_fee_perc=b3_fee_perc
                )

                if not grid_df.empty:
                    # Pivot para Retorno Total
                    pivot_ret = grid_df.pivot(index="Desconto_Entrada", columns="Alvo_Lucro_Perc", values="Retorno_Total_Perc")
                    
                    fig_heat_ret = px.imshow(
                        pivot_ret,
                        labels=dict(x="Alvo de Lucro (Take Profit %)", y="Desconto de Entrada (%)", color="Retorno Total (%)"),
                        x=[f"{c}%" for c in pivot_ret.columns],
                        y=[f"{r}%" for r in pivot_ret.index],
                        color_continuous_scale="Viridis",
                        text_auto=".1f",
                        title=f"Matriz de Retorno Total Acumulado (%) - {selected_ticker}"
                    )
                    st.plotly_chart(fig_heat_ret, use_container_width=True)

                    # Pivot para Taxa de Acerto
                    pivot_win = grid_df.pivot(index="Desconto_Entrada", columns="Alvo_Lucro_Perc", values="Taxa_Acerto_Perc")
                    fig_heat_win = px.imshow(
                        pivot_win,
                        labels=dict(x="Alvo de Lucro (Take Profit %)", y="Desconto de Entrada (%)", color="Taxa de Acerto (%)"),
                        x=[f"{c}%" for c in pivot_win.columns],
                        y=[f"{r}%" for r in pivot_win.index],
                        color_continuous_scale="Blues",
                        text_auto=".1f",
                        title=f"Matriz de Taxa de Acerto (Win Rate %) - {selected_ticker}"
                    )
                    st.plotly_chart(fig_heat_win, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: TABELA DE TRADES & EXPORTAÇÃO
    # -------------------------------------------------------------
    with tab5:
        st.markdown(f"### 📥 Histórico Completo de Pregões e Trades: **{selected_ticker}**")
        
        filter_option = st.radio("Filtrar Tabela:", ["Apenas Pregões com Operação Disparada", "Todos os Pregões"], horizontal=True)
        
        display_df = trades_df[trades_df["Triggered"] == True] if filter_option == "Apenas Pregões com Operação Disparada" else trades_df
        
        st.dataframe(
            display_df.reset_index().style.format({
                "Open": "R$ {:.2f}",
                "High": "R$ {:.2f}",
                "Low": "R$ {:.2f}",
                "Close": "R$ {:.2f}",
                "Prev_Close": "R$ {:.2f}",
                "Gap_Perc": "{:+.2f}%",
                "Target_Entry": "R$ {:.2f}",
                "Entry_Price": "R$ {:.2f}",
                "Exit_Price": "R$ {:.2f}",
                "Net_Return_Perc": "{:+.2f}%",
                "Profit_BRL": "R$ {:,.2f}",
                "Capital": "R$ {:,.2f}"
            }),
            use_container_width=True
        )

        # Exportações
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = display_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📄 Baixar Tabela em CSV",
                data=csv_data,
                file_name=f"trades_{selected_ticker}_{datetime.date.today()}.csv",
                mime="text/csv"
            )

        with col_exp2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                display_df.to_excel(writer, sheet_name="Trades")
                equity_df.to_excel(writer, sheet_name="Curva_Capital")
            st.download_button(
                label="📊 Baixar Relatório em Excel (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"relatorio_daytrade_{selected_ticker}_{datetime.date.today()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
