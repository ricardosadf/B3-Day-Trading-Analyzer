# Modelo Estatístico e Preditivo para Day Trade na B3

Software de análise quantitativa, backtesting estatístico e modelagem preditiva para operações de Day Trade no mercado de ações brasileiro (B3).

---

## 📌 Funcionalidades Principais

1. **Coleta de Dados em Tempo Real / Histórico B3**:
   - Integração com Yahoo Finance (`yfinance`) para download automático dos últimos 6 meses (ou períodos customizados).
   - Suporte aos ativos do Ibovespa e qualquer ação da B3 (`PETR4`, `VALE3`, `ITUB4`, `MGLU3`, etc.).
   - Ajuste automático para dividendos, JCP e desdobramentos de ações.

2. **Motor de Simulação Operacional (Backtesting Intraday)**:
   - **Gatilho de Compra**: Compra executada se a mínima do dia atingir um recuo de $X\%$ ou $-X$ pontos/centavos em relação à abertura do dia ou fechamento anterior.
   - **Take Profit (Alvo de Lucro)**: Saída no intraday se a máxima atingir o patamar configurado (ex: $+10\%$, $+5\%$, etc.).
   - **Stop Loss / Saída no Fechamento**: Se o alvo não for atingido, encerramento obrigatório no leilão de fechamento ou em nível de stop configurado.
   - **Custos Reais**: Inclusão de emolumentos/taxas B3 (~0.03% por perna) e slippage.

3. **Métricas de Performance & Recorrência**:
   - Frequência de disparo (% dos dias em que houve oportunidade).
   - Taxa de Acerto (*Win Rate* %) e Taxa de Batimento do Alvo.
   - Lucro Líquido Total (R$ e %) e Curva de Evolução Patrimonial.
   - Payoff ($Gain_{médio} / Loss_{médio}$) e Fator de Lucro (*Profit Factor*).
   - Expectativa Matemática por operação:
     \[
     E = (P_{win} \times \bar{G}) - (P_{loss} \times \bar{L})
     \]
   - *Max Drawdown* (maior queda de capital) e Sharpe Ratio anualizado.

4. **Modelagem Preditiva para os Próximos 6 Meses**:
   - **Simulação de Monte Carlo com Bootstrap**: 5.000 a 10.000 iterações projetando cones de probabilidade (P5, P25, P50, P75, P95) para os próximos 126 pregões úteis.
   - **Inferência Bayesiana (Beta-Binomial)**: Estimativa da taxa de acerto futura com intervalo de credibilidade de 95% e probabilidade a posteriori.
   - **Teste de Significância Estatística**: Teste de hipótese para avaliar se o retorno médio é estatisticamente significante ($p < 0.05$).

5. **Scanner B3 & Otimizador de Parâmetros**:
   - Scanner em lote para rankear as melhores ações para o setup.
   - Matrizes de calor (*Heatmaps*) para identificar os melhores alvos e descontos de entrada.

---

## 🚀 Como Executar

### 1. Pré-requisitos e Instalação
Certifique-se de estar no diretório do projeto e execute:
```bash
pip install -r requirements.txt
```

### 2. Iniciar o Dashboard Interativo
Execute o Streamlit no terminal:
```bash
streamlit run app.py
```
O navegador abrirá automaticamente no endereço `http://localhost:8501`.

### 3. Executar os Testes Automatizados
```bash
python3 -m pytest tests/ -v
```

---

## 📂 Estrutura do Código

```
Modelo Estatístico B3/
├── app.py                      # Dashboard Web Interativo (Streamlit)
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação detalhada
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Download de cotações B3 e cálculo de Gaps
│   ├── strategy_engine.py      # Backtest com regras de Abertura/Fechamento/Alvos
│   ├── metrics.py              # Cálculo de métricas estatísticas e financeiras
│   ├── forecast_model.py       # Simulação de Monte Carlo e Inferência Bayesiana
│   └── optimizer.py            # Otimizador de parâmetros em grade (Grid Search)
└── tests/
    ├── test_strategy.py        # Testes unitários do motor de estratégia
    └── test_forecast.py        # Testes unitários dos modelos preditivos
```
