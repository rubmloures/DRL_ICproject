# Integração do Projeto: Sensor de Regimes, PINNs em Superfícies de Volatilidade de Heston

Este documento detalha o paradigma de fusão estrutural arquitetônica onde Equações Diferenciais Estocásticas (SDEs), integradas via métodos computacionais de Redes Neurais Informadas pela Física (PINNs), fornecem matrizes latentes contínuas a políticas estocásticas de atuação em RL, formalizando o projeto como um híbrido exótico de "Machine Learning Causal e Otimização Direcionada".

## 1. O Modelo Estocástico Difusivo de Heston 

A limitação crônica do balanço de portfólio guiado por Black-Scholes resvala diretamente na premissa estática e log-normal contínua do caminho do desvio padrão do mercado (suposta volatilidade $\sigma$ determinística no tempo e predeterminada). Os mercados demonstram empíricamente curtose acentuada (*fat-tails*) e *volatility smiles*.

O modelo de Heston (1993) postula dois Processos Brownianos Padrões $\mathbb{Q}$ acoplados. O retorno da ação $S_t$ se manifesta sob a estrutura difusiva de salto contínuo condicionado instantaneamente pela dinâmica de retorno de variâncias de alta ordem temporal (mean-reverting Square-Root process ou equação de Cox-Ingersoll-Ross (CIR)):

$$ dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_1^S $$
$$ dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_2^v $$

*Correlação Estrutural Limitadora:* $\langle dW_1, dW_2 \rangle_t = \rho dt$.
Esta assimetria browniana é basilar. A correlação de alavanca $\rho \in [-1, 1]$ usualmente altamente negativa dita um efeito estrutural de que quebras nos retornos instigam assimetrias inflacionárias na volatilidade real. $\kappa$ projeta a velocidade com que a volátil variância de estado caótica regride ao platô calmo natural secular $\theta$, onde a variância na variação é a sensibilidade ruidosa do *vol-of-vol* $\xi$. O termo de deriva $v_t \geq 0$ (Condição de Feller exigida analiticamente via $2\kappa\theta > \xi^2$).

### 1.1 A EDP Fundamental de Estruturação Heston para PINNs
Usando o lema de Itô multidimensional para compor o valor hipotético temporal exato do prêmio de deriva de um derivativo europeu, a função restritora canônica do projeto é materializada em:

$$ \frac{\partial U}{\partial t} + \frac{1}{2} v S^2 \frac{\partial^2 U}{\partial S^2} + \rho \xi v S \frac{\partial^2 U}{\partial S \partial v} + \frac{1}{2} \xi^2 v \frac{\partial^2 U}{\partial v^2} + r S \frac{\partial U}{\partial S} + [\kappa(\theta - v) - \lambda v] \frac{\partial U}{\partial v} - r U = 0 $$

O treinamento restritivo por EDP do operador do resíduo (Autograd Backprop na componente estrutural do *Loss Function*) mapeia, no espaço paramétrico e de pesos da arquitetura base `DeepHestonHybrid`, o quão bem os derivativos simulados tangem analiticamente a lei não-linear deste processo e, consequentemente, destila do aprendizado inverso, as medidas pontuais desconhecidas contemporâneas $\{\kappa_t, \theta_t, \xi_t, \rho_t\}_{OOS}$.

## 2. Abstraindo Regimes Dinâmicos: O Fluxo Tensor Híbrido

O agente inteligente do portfólio, no cômputo clássico de Time Series Prediction, recebe puramente matrizes baseadas nos preços (ex. High/Low quantis cruzados ou janelas diárias temporais), que falham intrinsecamente ao separar uma reversão estatística média de estresse de transição modal estruturada "Crash Pattern".

No núcleo arquitetônico da integração proposta pelo projeto, o método *PINN Parameter Estimation* fornece as chaves latentes contínuas a serem anexadas como extrator do fator de estado ao POMDP do *Actor-Critic*.

### 2.1 A Inversões Constantes por Treinamento Incremental Calibrado
Dada a superfície empírica dos dados $C^{mkt}(S_i, v_i, T_i, K_i)$ alimentados na entrada do orquestrador via pipeline diário:

$$ \min_{\mathbf{Weights_{NN}}, \mathbf{\Omega_{Heston}}} \left( \sum ( C^{PINN} - C^{mkt} )^2_{Data} + \lambda_{pde} ||f_{Heston}||^2_{pde} \right) $$

Como o treinamento do modelo é dinâmico por blocos de *sliding window*, os parâmetros escalares da otimização das restrições restam calibrados iterativamente (os hiperparametros físicos reais da SDE subjacente, $\mathbf{\Omega} \equiv \{\kappa_t, \theta_t, \xi_t, \rho_t, v_0\}$ em $t$, emergem da minimização global e servem como termômetro das entranhas sensíveis e ocultas de sentimento).

## 3. O Tensor Objeto do Ambiente: Extratificada Multidimensional

O Agente DRL `DeepHestonTradingEnv` (O environment contínuo no PPO), no estado em $t$, observa a sobreposição concatenada transacional:
O vetor $s_t$ compõe um bloco semântico multivariado, agregando a camada causal e o reflexo técnico micro:
$$ s_t = \{ \Psi_{Price}^{1:L}, \ \Upsilon_{TechIndicators}^{1:F}, \ \Phi(\mathbf{\Omega_{Heston}})_t \} \in \mathbb{R}^{d_{obs}} $$
*   $\Psi_{Price}$: Matriz das variações dimensionais em cascata relativas dos preços ($L$ dias atrasados preenchendo as topologias dos *Transformers* ou os módulos baseados num loop do `Stable Baselines 3`).
*   $\Upsilon_{TechIndicators}$: Transformadas técnicas determinísticas do *Pandas-TA* (Boiling Band Width, Índice Force e Volume, RSI cross).
*   $\Phi(\mathbf{\Omega_{Heston}})_t$: Parâmetros de calibração extraídos *ex ante*, permitindo um filtro probabilístico que precondiciona a matriz de transição do ambiente latente de observação do agente ao ciclo real oculto no qual o modelo dinâmico financeiro opera nos subjacentes macroeconômicos não refletidos integralmente ainda nos atrasos (lags) clássicos do sinal empírico puro de preços passados.
