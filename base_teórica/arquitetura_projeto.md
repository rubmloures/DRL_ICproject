# Arquitetura Estrutural, Dados Transacionais e Módulos Operacionais

Este documento estende cientificamente a taxonomia de fluxos e interdependências de classes, subsistemas Python acoplados, orquestração por processamento GPU com Optuna e topologias transacionais simuladas implementadas utilizando as interfaces operantes do `Stable Baselines 3`, `FinRL/Gymnasium` e módulos originais.

## 1. Topologia Base e Interatividade FinRL/Stable Baselines

O ecossistema implementa rigor arquitetural ao delegar subsistemas aos frameworks canônicos da comunidade Open-Source de "Deep Reinforcement Learning for Finance" e pesquisa de IA segura, garantindo flexibilidade a refatoramentos:

*   **Gymnasium Base Environments (Gym):** Herança de `gym.Env` que determina de forma canônica quatro funções do pipeline: `step()`, `reset()`, `render()` e o espaço observacional e o de ação transacional multivariada. No projeto, as fronteiras complexas de caixa rotativo (dinâmica de restrições transacionais e balanço *account_value* sob a ação do vetor alocativo de Markowitz e *Cost Penalty*) são abstraídas em um sub-ambiente customizado que encapsula o *spread / slip* para derivar uma curva patrimonial verossímil sob rigor da liquidez intradiária.
*   **SB3 Extensibility Engine / Features:** O Stable Baselines não injeta arquiteturas simples; ele viabiliza injeções diretas polimórficas das Extrações Neurais por bibliotecas nativas de *PyTorch*, reescritas no bloco `Custom Feature Extractor`. Esta ponte desacopla toda a densidade das equações diferencias computacionais de "Actor-Critic Policy" dos fluxos do backend otimizado do PPO.

## 2. Ingestões Dimensionais: Pipeline e Engenharia de Features de Alta Frequência

O vetor original da infraestrutura (séries extraídas diretamente das APIs dos terminais via `data/raw`) precisa ser condicionado ao padrão auto-regressivo analítico viável ao treinamento:

### 2.1 Dimensionalidade Expandida Não-Linear de Tempo (Features Técnicas)
As séries brutas ($OHLCV$, Open/High/Low/Close/Volume) não portam invariância temporal ou escala ideal ao descida gradiente; a volatilidade corrompe grandezas escalares. Extratores do pacote analítico computam funções de transferência com transformadas no domínio do processamento digital (Fast Fourier/Autocovariâncias discretas ou derivativos via TA-Lib) e as encapsulam via padronização min-max diferencial Z-Score ao tensor:
1.  **RSI/MACD/CCI:** Medidas discretas de regressões de *momentum* oscilatórios para captar divergências estatísticas nos picos de suporte direcional de derivadas da microestrutura (força do empuxo temporal local).
2.  **Bollinger e ATR (Average True Range):** Estruturas dispersivas baseadas no coeficiente local contínuo de osciladores, mensurando o volume temporal e desvio de normalidade restrita por Box-Jenkins de séries integradas univariáveis cruzadas pela banda e larguras vetoriais.

### 2.2 Estratificação da Matriz Temporal Causal (Lookback Windows)

Diferente de ML convencional (MLP, Tabular XGBoost, Support Vector Machines lineares), a série causal num nó relacional neural obedece formalmente a blocos empilhados da sub-trama contigua $L \times N$, engessando ao PyTorch uma dependência auto-regressiva densa e contextual `np.reshape() -> torch.Tensor` do histórico contínuo dos últimos estresses no ativo base $H(t)$.

## 3. Extrator Customizado de Redes Funcionais: "TransformerActorCriticPolicy"

O coração do projeto (onde atua o PPO principal base) suplanta as arquiteturas triviais das MLP. Sob implementação de sub-classes do `BaseFeaturesExtractor`, a política (peso base otimizável via PyTorch $AdamW$/$RMSProp$ e LR Schedules Decaying), processada nos fluxos Actor $\phi_{act}$ e Critic $\phi_{crit}$, instiga a entrada multivariável a rotear, antes das topologias densas, pelos encoders e redes *Head/Bottleneck* residuais de:

*   **Linear / LSTM e GRUs:** Blocos de estado de celda retida.
*   **Scaled Attention Transform Engines (*Transformers*):** Reduzindo distúrbios da memória das RNN baseadas na sequência de entrada, computando blocos paralelos num subespaço matricial rotacional que avalia implicitamente se as dependências não lineares da transição pontual no passo temporal das *lookback windows* $W$ impacta as inferências no momento futuro. Cada camada aplica o *dropout stochastic residual layer* minimizando oscilador instável e otimizando generalizações para instabilidades fora de cotações normais.

## 4. Engenharia Funcional Recompensatória Restrita

No domínio matemático Actor-Critic de otimização estocástica, as restrições da função recompensa contínua moldam indiretamente as curvaturas probabilísticas a longo-alcance, induzindo ou desencorajando caminhos e distribuições estocásticas da política de locações a fim de não criar sistemas unicamente especulativos ao pico.

O gradiente do PPO, de maximização, integra a "Loss function" de política $\pi$ para penalizar sistematicamente via hiper-variáveis modulares de *Penalties*: 

$$ J^{Actor} (\theta_{Actor}) = \max_{\theta} \, \mathbb{E}_t [ \mathbf{f_{NAV\_Var}} - \psi_{dd} ( f(MDD) ) - \gamma_{rot} (\sum_{k=1}^{N_{A}} | a_i^{t} - a_i^{t-1} | )] $$
O resíduo penalizador condensa:
*   $f_{NAV}$: Valorização percentual contínua total agregada (*Account Value* líquido da transição anterior vs a transição atual do fechamento).
*   $f(MDD)$ ou Sortino Penalties: Ajuste severo a variações bruscas que puxem o retorno negativo além das predições estatísticas de VaR (Value at Risk) de portfólios paramétricos ou a subjacente métrica puramente contínua máxima relativa de pico profundo. 
*   **Fricções Restritivas de Comissões e Modificadores:** Custos relativos restritos aos *Slips* absolutos do movimento temporal para estabilidade de carteiras e freio explícito de oscilações puramente ruídosas, forçando uma otimização causal nas inferências latentes com tendências prolongadas (Hold & Buy / Swing trades), e não simples especulação instável hiper-frequente de chattering temporal ruidosamente inconstante gerado pelas cadeias aleatórias.

## 5. Orquestradores Otimizados e Arquiteturas Backtesting / A-B Check 

1.  **Framework Otimizador *Optuna*:** A calibragem contínua não é empírica local/manual. Funções Obejtivos interativas e robustas, guiadas pelas cadeias de *Tree-structured Parzen Estimator* (TPE) mapeiam o amplo espaço estocástico multivariado ($\epsilon$-PPO Clip, Learning Rate, Batch Sizes nas mini-fases temporais e Gamma $\gamma$ discount). Em instâncias parciais e ciclos repetitivos curtos, extrai os blocos das recompensas densas para escalar, por amostragem probabilística baiesiana contínua iterativa, uma matriz global dos melhores vetores paramétricos globais aos ensaios PPO sem sofrer perdas pesadas de *grid searching* exponencial e cego.
2.  **Robustez de A/B (Walk-Forward e In-Distribution Validation):** Os *Backtestings* dos pesos e distribuições estaduais são inferidos de modo congelado nas porções da divisão linear OOS (Out of Sample Test Matrix) gerando inferências reais em um ambiente paralelo. Se extrai as métricas tradicionais da estatística do ambiente (Índices SR (Sharpe), Retornos Líquidos, Drawdowns), comparando as versões de Agentes e Modelos Integrados, validando conclusivamente a assertividade funcional contínua.
