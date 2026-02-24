# Detalhamento Fundo e Arquitetural das Técnicas Aplicadas no Projeto

Este documento detalha pormenorizadamente as metodologias de vanguarda implementadas na fundação deste projeto. A base de código entrelaça Aprendizado de Máquina Dinâmico (RL), Aprendizado Supervisionado com Restrição Física (PINN) e Validação Estatística de Séries Temporais (Purged Validation). Abaixo, exploramos as nuances matemáticas, a lógica algorítmica e a aplicabilidade de cada módulo da nossa infraestrutura.

---

## 1. Deep Reinforcement Learning (DRL) Algorithms

A camada de tomada de decisão, instanciada em `src/agents/drl_agents.py`, adota um triplo conjunto de algoritmos para maximizar a assertividade do portfólio. Ao invés de usar apenas um paradigma de aprendizado de máquina, alavancamos algoritmos complementares (*On-Policy* e *Off-Policy*).

### 1.1 Proximal Policy Optimization (PPO)
- **Classificação:** Algoritmo de Política Direta (*Policy Gradient*) *On-Policy* e *Actor-Critic*.
- **Mecânica Matemática e Operacional:** O PPO inibe passos de atualização destrutiva através do *clipping* restritivo da razão de verossimilhança $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$. No nosso código, o parâmetro `clip_range=0.2` diz que o tamanho da passada de aprendizado não deve fugir a 20% da confiança da política anterior. Acoplado a uma entropia moderada (`ent_coef=0.01`), ele mantém as ações exploratórias equilibradas.
- **Papel no Projeto:** O PPO assume o papel de "Navegador Central", é o mais estável para convergência matemática linear frente a dados complexos do mercado acionário diário, focado em entender tendências de curto prazo sem apostar a banca toda cegamente em sinais errados.

### 1.2 Deep Deterministic Policy Gradient (DDPG)
- **Classificação:** Algoritmo *Off-Policy* determinístico para ações contínuas.
- **Mecânica Matemática e Operacional:** Diferente do PPO que estima um espaço de probabilidade, o DDPG estima uma função determinística exata $a = \mu(s|\theta^\mu)$. Ele explora o ambiente injetando ruído gaussiano contínuo através da `NormalActionNoise(mean, sigma)`. 
- **O Diferencial (Experience Replay):** O DDPG faz amostras de experiências passadas, gravadas em um formidável buffer (configurado em nosso código para `buffer_size=100_000` transições temporais).
- **Papel no Projeto:** Funciona como o lado "Analítico da Memória Histórica". Se amanhã o mercado formar um *Crash* que reverbera o mesmo regime visto há 6 meses atrás nas épocas de treino inicial, o DDPG será brutalmente vantajoso, pois varre o buffer através de mini-batches lembrando "o que fiz naquela vez e como evitei a perda sistêmica?", trazendo uma contraparte analítica temporal superior ao PPO.

### 1.3 Advantage Actor-Critic (A2C)
- **Classificação:** *On-Policy* Síncrono.
- **Mecânica Matemática e Operacional:** Subtrai o "Valor Esperado" da recompensa bruta de um sub-estado, criando o *Advantage* ($A = Q(s, a) - V(s)$). Ou seja, se a IA ganha $1,000 enquanto a média natural e fácil do mercado (Ibovespa) ganharia $900, o *Advantage* é apenas $100.
- **Papel no Projeto:** Ele não quer apenas "Bater a meta", ele quer bater a média local, ajustando seus vetores para obter prêmios líquidos exponenciais num horizonte de tempo minúsculo iterativo (`n_steps=5`).

---

## 2. Modelagem Estratégica Multi-Agentes (Ensemble Learning)

Para não depender do sucesso de apenas um modelo de DRL em uma janela hostil de mercado, agregamos vetores. Implementado intrinsecamente no módulo `src/agents/ensemble_agent.py`.

- **Mecânica de Fusão e Tomada de Decisão:** O `EnsembleAgent` recolhe a política de previsão `action = predict(obs)` instanciada separadamente na tríade de Redes (PPO, DDPG, A2C), e aplica métodos de votação ponderados por histórico de competência.
- **Sharpe-Weighted Voting (*Estratégia Weighted*):**
  - O código roda inicialmente testes (avaliações em *test_env*) unitários para ranquear agentes perante um sub-trecho de mercado.
  - O agente que gerou um Índice de Sharpe e lucro alto recebe pesos majoritários: $w_i = \frac{Sharpe_i}{\sum Sharpe}$. 
  - Na matriz inferencial final: $\text{Ação Global}  = (0.7 \times DDPG) + (0.2 \times PPO) + (0.1 \times A2C)$.
  - **Tratamento de Edge-Cases:** Se os retornos forem catastróficos para TODOS (`sharpe < 0`), o módulo detecta esse limite com `total_weight <= 0`, aplicando um mecanismo defensivo ("fallback") e voltando passivamente à tática equalitária rígida (`Mean_Voting/Uniform = 1.0/3.0 pesos iguais`).

---

## 3. Physics-Informed Neural Network (Market Regime Sensor via PINN)

Provavelmente a peça técnica mais impressionante. Integrado na arquitetura sobre o diretório `src/pinn/`, ele rompe as barreiras tradicionais do cálculo quantitativo estocástico com **Boundary Conditions Criptográficas (Redes Rígidas baseadas na Físico-Matemática DGM)**.

- **Mecânica:** Os mercados não são randômicos perfeitos, derivativos são controlados por equações diferencias termodinâmicas (Black-Scholes / Heston Model). O `heston_residual` em `physics.py` penaliza o gradiente da rede caso esta não satisfaça a Equação Fundamental Diferencial Parcial (EDP) de Heston (utilizando `torch.autograd.grad` simultâneo para extrair $V_{SS}$ [Gamma], $V_{\nu\nu}$ [Vol of Vol Curvature] e $V_{S, \nu}$).
- **Parâmetros Físicos Aprendidos e Extraídos de uma janela Real Movel (30-Dias):**
  - **$\kappa$ (Kappa):** Velocidade em que o pânico cessa e revete à normalidade do pregão.
  - **$\theta$ (Theta):** Variânica perene a Longo Prazo.
  - **$\xi$ (Xi):** Dinâmica da volatilidade da volatilidade ("Vol of Vol"), revelador nato de *Crashes*.
  - **$\rho$ (Rho):** Efeitos de Alavancagem Negativa; reflete pânicos onde Preço ($S$) cai e o Medo (Volatilidade) espirra catastroficamente em direção à Lua.
- **Interação do PINN com o Agente Central DRL:**
  1. A cada nova bateria técnica, o modulo `PINNInferenceEngine` lê estatísticas cruas dos últimos 30 dias de cotação.
  2. Infere $\kappa, \theta, \xi, \rho$ latentes de acordo com equações de Heston.
  3. Pluga e repassa ("Enrichment") esses 4 numérios recém-calculados como +4 *features* acopladas como colunas de `dataframes` e enviadas puras nas matrizes de Observação ($O_t$) alimentadas ao PPO. 
  Isso ensina o bot silenciosamente sobre se a conjuntura macropolítica oculta de 30 dias está saudável ou colapsando a partir da "Superficie Gamma do Options Desk".

---

## 4. Validação Cruzada Walk-Forward (Purged \& Embargo Rolling Window)

As séries temporais do IBOW violam a premissa de **Observações I.I.D. (Livres e Independentes)**. Construído através do artefato engenhoso `src/data/rolling_window.py`.

- **Mecânica da Falha Estática Tradicional:** A maioria dos "papers" aplicam Split Estático `Treino: 2018-2022 | Teste: 2023`. Ao se passar 5 meses rodando em predição de Janelas, a serial-correlação dos fechamentos infecta o mercado (Leakage). Se treina no dia D, e testa no dia D+1: o preço adjacente afeta o resultado.
- **Nossa Arquitetura (Purged Validation with Embargo Gap):**
  A classe `RollingWindowStrategy` não só "Rola" as datas fixas e progressivamente, mas injeta um intervalo estéril paramétrico (`purge_days = 5` dias por default) expurgado do gráfico e ignorado. 
- Essa muralha temporal cede as correlações transientes estocásticas a morrerem ao fim de um período de "Embargo", limpando qualquer vício preditivo adjacente e expondo a IA ao mais letal rigor quantitativo antes do *Release* final. Conta simultaneamente com "Purge de K-Fold" também injetável!

---

## 5. Função de Recompensa Dinâmica por Regimes (Composite Adaptive Reward)

A função meta de ganho no sistema de reforço engole o princípio das Regras Modulares ("The reward is not just pure unguided return"). Documentado massivamente ao longo de `src/reward/composite_reward.py`.

- **Quatro Equações Agregadas Simultaneamente:**
  1. $R_{exc}$: Retorno Excedente - Punição quando fica a baixo da taxa Taxa Livre de Risco (Selic) local lida dinamicamente do dataset do Brasil.
  2. $R_{diff}$: Alpha - Retribuição pelo Lucro Líquido Subtraindo passivamente quanto cresceu o Mercado Indice Ibovespa base (Benchmarking). 
  3. $\sigma_{down}$: Punição à Desvio Negativo (Quadratic Penalty de Sortino Ratio).
  4. Constante Exata de Custos baseados na Taxação Transacional e Emolumentos na B3 (-0.15% corretagem -0.05% b3 fixo na recompensa).
- **Adaptatividade Multi-Regimes (RegimeDetector):**
  Utilizando os dados latentes ($\xi$ Vol-of-vol e $\rho$) trazidos pelo PINN, a classe quantifica regimes globais em categorias (`STABLE_TRENDING`, `TURBULENT_SHOCK`, `ELEVATED_VOLATILITY`).
  - *Exemplo Catastrófico:* Se o PINN gritar que fomos para um *Turbulent_shock* (Altíssimo $\xi$, baixíssimo $\rho$). A função de Reward entra em pânico técnico: Seu sub-peso de Punição a riscos de queda ($\text{downside\_risk\_weight}$) **dobra**, seu incentivo de premiação a retornos excedentes **cai pela metade**. O PPO e A2C desistem de caçar a alta no trade e correm desesperadamente para posições em dinheiro aliviando draw-downs letais automaticamente e liquidando tudo em carteira.

---

## 6. Otimização Bayesiana Estocástica (Optuna / TPE)

Ao invés de tentar treinar infinitamente e cegamente no Grid paramétrico. Embutimos um Pipeline de Otimização `optuna-optimize`.

- O Optuna baseia seu Sampler inferencial através do modelo de **Estimadores Árvore de Parzen (TPE)**. 
- O Pipeline (no `main.py` e `hyperparameter_optimizer.py`) mapeia quais sub-parâmetros das sub-arquiteturas das camadas ocultas DRL devem ser rodados em instâncias efêmeras (com timeos-outs fixos), explorando matematicamente onde as pontuações do Sharpe global tendem a se cruzar na malha geométrica, achando o *Graal* do Gamma de Desconto ideal antes de congelar os pesos para a vida do robô.

---

## 7. A/B Testing Científico Nativo

Para cravamento em documentações/TCCs universitárias. Implementamos flag de orquestração explícita no módulo central do `main.py`.

- A flag `--ab-testing` desencadeia um processamento paralelo cronológico independente no Pipeline.
  - Ramo primário lança um **Grupo B de Baseline** (Apenas o PPO e seu reator técnico correndo pelo calendário cego, sem as Features Matemáticas do Regime).
  - Ramo secundário lança o **Grupo A de Experimento** com as Injeções Enriquecidas PINN habilitadas sob a mesma Semente Numérica Genética do pytorch (`set_all_seeds(100)`).
- Logistica exata da tabulação: Devolve comparações analíticas cruas baseadas se as taxas de Análise Físico Críticas mitigaram Risco sem ceder Sharpe Ratio e sem gerar Viés. Retornando os deltas ($\Delta_{\text{Alpha}}$, $\Delta_{\text{Drawdowns}}$) isolados na tela.
