# Função de Recompensa Composta (Composite Reward)
## Base Teórica e Explicação Matemática

Este documento detalha o funcionamento matemático da função de recompensa composta utilizada pelos agentes de DRL neste projeto, implementada no módulo `CompositeRewardCalculator`. A função foi desenvolvida para ir além de maximizar retornos, penalizando riscos assimétricos e simulando fricções reais do mercado da B3 (como corretagem e emolumentos). Além disso, ela utiliza pesos de regime de mercado dinâmicos informados pela rede PINN.

A recompensa total no instante $t$ é dada por:

$$ R_{t} = w_1(s) \cdot R_{exc} - w_2(s) \cdot P_{down} + w_3(s) \cdot R_{\alpha} - w_4(s) \cdot C_{trans} $$

Onde $w_i(s)$ são os pesos ajustados dinamicamente baseados no regime de mercado $s$ detectado pela PINN (ex: *Bull Market*, *High Volatility*, *Crash*).

---

### Componentes da Função

#### 1. Retorno Excedente (Excess Return) - $R_{exc}$
O retorno excedente ($R_{exc}$) mede o quão superior foi o rendimento do portfólio em relação a um investimento na taxa livre de risco (Aproximada pela Taxa Selic anual convertida em fator diário).

$$ R_{exc} = R_{port} - R_{rf} $$

*   $R_{port}$: Retorno percentual diário alcançado pelas ações combinadas.
*   $R_{rf}$: Taxa *Risk-Free* diária baseada em dados históricos da Selic (`taxa_selic.csv`).
*   **Motivação:** Um agente que simplesmente gannha do CDI agrega valor. Um que empata assume risco à toa.

#### 2. Penalidade de Risco de Queda (Downside Risk Penalty) - $P_{down}$
Diferente do Índice de Sharpe que trata a volatilidade positiva e negativa igualmente (punindo *upsides* expressivos), nossa recompensa bebe da fonte do Índice de Sortino. A função **penaliza de forma quadrática apenas retornos negativos**.

$$ P_{down} = \max(0, -R_{port})^2 $$

*   **Motivação:** A elevação quadrática garante que pequenas oscilações negativas no dia a dia sejam toleradas pelo agente, mas quedas acentuadas num único dia resultem em "dor" exponencial (prevenindo que o agente invista em ativos que *derretem* de vez em quando).

#### 3. Alpha ou Retorno Diferencial (Differential Return) - $R_{\alpha}$
O "Alpha" recompensa ou penaliza a rede baseando-se no desempenho comparativo contra o Ibovespa ($R_{bench}$).

$$ R_{\alpha} = R_{port} - R_{bench} $$

*   **Motivação:** Força o DRL a encontrar vetores de ações ou pesos descorrelacionados positivamente de marés generalizadas de baixa.

#### 4. Custos de Transação da B3 (Transaction Costs) - $C_{trans}$
Cada ordem enviada desconta os custos inerentes à B3 e a spread/comissões. A métrica se baseia no giro diário (quantidade de transações do agente em um dia, englobando buys e sells).

$$ C_{trans} = N \times (C_{corretagem} + C_{emolumentos}) $$

*   No Brasil: Um número aceitável na pesquisa é $0.15\%$ ($0.0015$) para Custos Institucionais/Varejo somado a $0.05\%$ ($0.0005$) para emolumentos líquidos da bolsa, totalizando $0.20\%$ ($0.0020$).
*   **Motivação:** Evita modelar algoritmos *High-Frequency* e pune o excesso de *churning*. Se a ação sobe 1%, mas o DRL executou 10 negociações no dia para caçar isso, a punição devorou os lucros reais.

---

### Exemplo Numérico Interativo

Suponhamos que num determinado "Step" (um dia) dentro do ambiente virtual tenhamos os seguintes dados informados pelo mercado normal de Baixa Volatilidade (Onde os pesos do regime, suavizados por EMA, são: $w=\{1.0, 1.0, 1.0, 1.0\}$):

*   Retorno do Portfólio (DRL no dia): **-1,5%** ($R_{port} = -0.015$)
*   Retorno do Ibovespa no dia: **-3,0%** ($R_{bench} = -0.030$)
*   Taxa Selic Diária: **0,03%** ($R_{rf} = 0.0003$)
*   Número de ordens (Trades do agente): **2**

#### Cálculo Passo-a-Passo:

**1) Excess Return:**
$$ R_{exc} = -0.015 - 0.0003 = -0.0153 $$
$$ \to R_{exc\ weighted} = 1.0 \times (-0.0153) = -0.0153 $$

**2) Downside Penalty:**
*   A queda foi de $-0.015$. Portanto o termo negativo é acionado.
$$ P_{down} = \max(0, - (-0.015))^2 = (0.015)^2 = 0.000225 $$
$$ \to R_{down\ weighted} = 1.0 \times (-0.000225) = -0.000225 $$

**3) Alpha (Retorno Relativo):**
*   Caiu -1,5%, mas a bolsa caiu o dobro. Alpha positivo!
$$ R_{\alpha} = -0.015 - (-0.030) = +0.015 $$
$$ \to R_{\alpha\ weighted} = 1.0 \times 0.015 = 0.015 $$

**4) Custos Friccionais:**
*   Agente rodou sua carteira 2 vezes, a 0.2% cada ($0.002$).
$$ C_{trans} = 2 \times 0.002 = 0.004 $$
$$ \to P_{trans\ weighted} = 1.0 \times 0.004 = 0.004 $$

**Recompensa Final:**
$$ R_{total} = (-0.0153) + (-0.000225) + (+0.015) - (+0.004) = -0.004525 $$

**Conclusão do Exemplo:**
Embora o agente tenha atuado de forma defensiva frente a um colapso do IBOV (gerando Alpha de +0.015), o saldo do dia ainda foi negativo frente à Selic, com multas por volatilidade de queda. Somando o frete operacionado 2 vezes, a penalização final de recompensa foi de `-0.004525`. 
Em vez de simplesmente passar `-0.015` (recompensa puramente ingênua de retorno) ou passar `+0.015` (focada unicamente em vencer Ibovespa com ordens sem limite de custo), o sinal engloba todos os fenômenos quantitativos.
