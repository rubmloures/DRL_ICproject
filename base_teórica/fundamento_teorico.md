# Base Teórica: Arquiteturas de Redes Neurais e Fundamentação Matemática Avançada

Este documento estabelece o escopo analítico-matemático e a fundamentação teórica rigorosa (nível *stricto sensu*) das três principais arquiteturas de redes neurais que compõem o *core* computacional deste projeto: Deep Reinforcement Learning (DRL) focado em Actor-Critic, Transformadores Espaço-Temporais e Redes Neurais Informadas pela Física (PINNs) aplicadas a equações diferenciais estocásticas.

## 1. Deep Reinforcement Learning (DRL) e o Processo de Decisão de Markov (MDP)

No paradigma adotado, o agente não explora um ambiente estático, mas sim o fluxo de microestrutura do mercado financeiro, cuja aderência é perfeitamente mapeada como um Processo de Decisão de Markov Parcialmente Observável (POMDP). Diferentemente do agrupamento simples de dados, DRL permite otimizar políticas sequenciais onde decisões correntes influenciam trajetórias e retornos futuros em interações multi-agente complexas.

### 1.1 Formalismo do POMDP Financeiro

Formalizamos o sistema pela tupla $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$:
*   $\mathcal{S}$: O Espaço de Estados Verdadeiros (não observável diretamente). Engloba todos os fatores ocultos macro e microeconômicos reais.
*   $\Omega$: O Espaço de Observações (o que o agente enxerga), composto por *arrays* multivariados da série histórica $X_t \in \mathbb{R}^{W \times F}$, em janelas temporais $W$.
*   $\mathcal{O}(o|s)$: A função de observação, mapeando a probabilidade condicional da observação empírica dado o estado real.
*   $\mathcal{A}$: O Espaço de Ações Contínuas $\mathcal{A} \subset \mathbb{R}^{N_{ativos}}$, sendo um vetor de pesos de alocação de portfólio restrito pelo simplex padrão: $\sum_{i} a_i = 1, \ a_i \in [-1, 1]$ (admitindo *short-selling* ou dependendo da formulação, apenas $[0,1]$ longo).
*   $\mathcal{T}(s'|s,a)$: Matriz (ou função de densidade de probabilidade) de Transição Estocástica de estados.
*   $\mathcal{R}(s, a) \rightarrow \mathbb{R}$: Função de recompensa densa.
*   $\gamma \in [0, 1)$: Fator de recuo de desconto (*discount factor*), penalizando recompensas tardias frente a imediatas para limitar o raio de convergência do valor ótimo final.

## 2. Proximal Policy Optimization (PPO) sob Arquitetura Actor-Critic

Em vez de métodos tradicionais como Q-Learning (como DQN, focado em espaços de ações discretos e propício a superestimação de valor continuo), adota-se o *Proximal Policy Optimization* (PPO). O PPO resolve a sensibilidade instável de métodos *Policy Gradient* (como A2C puro e TRPO), evitando que uma atualização ríspida deteriore uma política pré-otimizada através da criação empírica de um espaço de "região de confiança" (*Trust-Region*).

### 2.1 Formulação do Gradient Ascent no PPO

O PPO implementa dois estimadores neurais simultaneamente compartimentados (Actor-Critic): a Rede *Actor* $\pi_\theta(a|s)$, parametrizada por pesos associados aos pesos $\theta$, encarregada de destilar o fluxo contínuo de probabilidades de ações diretas, e a *Critic* (ou *Value Function*) $V_\phi(s)$, associada a parâmetros $\phi$ que projeta o valor cumulativo remanescente do estado.

A otimização principal do PPO restringe a distância de *Kullback-Leibler* (KL divergence) estipulativamente através da função objetivo clipada (surrogate objective):
$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right] $$
Onde $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ exprime a razão da política nova mediante a velha. $\epsilon$ tipicamente restrito entre $0.1$ a $0.3$ força estabilidade transacional durante as convergências de portfólio num mercado caótico.

A Função Vantagem $\hat{A}_t$ (*Advantage Function*), calcula quão superior foi uma ação específica com relação ao plano esperado no estado. O projeto consolida a estimativa *Generalized Advantage Estimation* (GAE):
$$ \hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $$
Onde o resíduo do erro TD (Temporal Difference) $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. Este formalismo ajusta incisivamente os dilemas entre viés e variância (*Bias-Variance Tradeoff*). O GAE atua como mecanismo supressor dos grandes ruídos do mercado nas rotas otimizadas.

## 3. Arquitetura Transformer Otimizada para Séries Temporais Financeiras

Enquanto as matrizes tradicionais baseadas em LSTMs sofrem perda de retenção informacional na modelagem do caminho estocástico num horizonte $\mathcal{O}(T)$ linear e sofrem de problemas na difusão da retropropagação cruzada (exploding/vanishing gradients), *Transformers* executam a contextualização do tempo por pareamento completo em tempo $\mathcal{O}(T^2)$ via matriz de dependência em apenas uma etapa neural através da operação combinada de Auto-Atenção.

### 3.1 Teorema de Self-Attention Multi-Head

O mecanismo de atenção é o núcleo das inferências de *Transformers*. Para uma matriz temporal transformada $X \in \mathbb{R}^{W \times d_{model}}$ com embutimentos posicionais somados (garantindo que o transformer perceba a casualidade longitudinal dos eventos):
São mapeados os vetores canônicos Lineares de *Query* ($Q=XW_Q$), *Key* ($K=XW_K$), e *Value* ($V=XW_V$).
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

No entanto, o aspecto vital do mapeamento se encontra na matriz *Multi-Head*, onde o tensor final compreende $h$ representações aprendidas projetadas num espaço ortogonal conjunto, asseverando que a rede foque em diferentes espectros da série de preços. Os pesos do agrupamento da cabeça, assim representados:
$$ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O $$
Desta forma, os padrões latentes no *micro-time* (e.g. *bid-ask bounce*, distúrbios de variação de liquidez em curto prazo) são observados assimetricamente pelas cabeças 1-2, enquanto dinâmicas lentas como autocorrelações do índice referencial ou ciclos de taxa de juros ressoam nas *heads* remanescentes.

## 4. Physics-Informed Neural Networks (PINNs): Restrição Dinâmica Não-Linear

Na precificação avançada por quantitativos e previsão de estresse dinâmico de capital, as redes genéricas perdem performance quando encaram distribuições e transições exóticas longtail não explícitas no dataset linearmente modelado. PINNs (Physics-Informed Neural Networks - Raissi et al., 2019) integram regularizadores explícitos calcados nas interações analíticas estabelecidas pela literatura sobre derivativos (e.g., EDPs fundamentais das dinâmicas das equações diferenciais estocásticas subjacentes).

$$ f(x,t) = \frac{\partial u_\theta}{\partial t} + \mathcal{N}[u_\theta, \lambda] = 0 $$
Onde $u_\theta$ é o aproximador universal da física contido nos pesos de rede (Ex: Precificação teórica profunda), $\mathcal{N}$ é o operador funcional da derivada (Ex. Black-Scholes Operator ou o Jacobiano Difusivo da equação de Heston).

### 4.1 Otimização das Operações Computacionais Numéricas Inversas (Autograd e Operadores Diferenciais)

Ao contrário dos esquemas numéricos complexos por Discretização e Diferenças Finitas (FDM), que impõem limites de restrito Courant-Friedrichs-Lewy (CFL constraints) e "A Maldição da Dimensionalidade" extrema em resoluções com matriz Custo-Efetiva (como a discretização complexa para PDEs multi-variadas em American Options), o mecanismo de Diferenciação Automática contínuo dos domínios Tensores baseados no Python PyTorch calcula a derivada real com precisão paramétrica. 
$$ \mathcal{L}_{Total} = \omega_{data}\frac{1}{N_{d}}\sum_{i=1}^{N_d} |u(x_i, t_i) - u_i|^2 + \omega_{pde}\frac{1}{N_{f}}\sum_{j=1}^{N_f} |f(x_j, t_j)|^2 + \omega_{ic_{bc}}\mathcal{L}_{bc} $$

Através disso, os módulos estocásticos conseguem aprender através do *Problema Inverso*. A rede não apenas estima $u$, mas simultaneamente aprende as grandezas desconhecidas e variantes estocásticas do mercado $\lambda(t)$ no tempo (como as matrizes do vetor estocástico de longo prazo do regime local, atuando como calibração global por *Deep Inverse Method* sem usar OLS estocástico iterativo).
