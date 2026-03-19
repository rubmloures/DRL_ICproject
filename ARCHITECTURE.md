"""
TECHNICAL ARCHITECTURE GUIDE
============================

This document describes the complete architecture of the DRL Stock Trading project,
organized into 3 main layers following industry best practices (FinRL, Stable-Baselines3).

## Layer 1: Data Layer (ETL)
=====================================

Location: `src/data/`

### Modules:
- `data_loader.py`: Raw data ingestion
  - Load CSV files with Brazilian decimal/separator format
  - Cache data for performance
  - Handle missing values gracefully
  - Parse date columns and encode categorical variables

- `data_processor.py`: Feature engineering & preprocessing
  - Calculate technical indicators (MACD, RSI, Bollinger, ATR)
  - Normalize financial features using StandardScaler/MinMaxScaler
  - Aggregate options data to daily summaries
  - Split data into train/test sets by date

### Data Flow:
```
CSV Files (data/raw/)
    ↓
DataLoader.load_asset() → Parsed DataFrame
    ↓
DataProcessor.clean_data() → Cleaned DataFrame
    ↓
DataProcessor.add_technical_indicators() → Enhanced Features
    ↓
DataProcessor.fit_scaler() → Fit on Training Data
    ↓
DataProcessor.transform() → Normalized Features
    ↓
Environment (Training)
```

### Key Design Patterns:
1. **Separation of Concerns**: Loading ≠ Processing
2. **Scalability**: Cache layer for repeated loads
3. **Reproducibility**: Stateless functions, no side effects
4. **Robustness**: Error handling, logging, validation

### Example Usage:
```python
from src.data import DataLoader, DataProcessor

# Load raw data
loader = DataLoader(data_path="data/raw")
df = loader.load_multiple_assets(
    assets=["PETR4", "VALE3"],
    start_date="2023-01-01",
    end_date="2024-12-31",
)

# Preprocess
processor = DataProcessor()
clean_df = processor.clean_data(df)
featured_df = processor.add_technical_indicators(clean_df)

# Split and scale
train_data, test_data = DataProcessor.split_data(featured_df, train_ratio=0.8)
processor.fit_scaler(train_data, columns=['SMA_20', 'RSI_14', ...])
train_scaled = processor.transform(train_data)
test_scaled = processor.transform(test_data)
```

---

## Layer 2: Environment Layer (Market Simulation)
=================================================

Location: `src/env/`

### Modules:
- `base_env.py`: Abstract Gymnasium environment
  - Defines common interface for all trading environments
  - Portfolio management, position tracking
  - Cost handling (transaction costs, slippage)
  - Reward calculation framework

- `stock_env_b3.py`: Specialized B3 environment
  - Multi-asset portfolio simulation (2-8 stocks)
  - Continuous action space [-1, 1]
  - B3-specific costs (0.03% transactions)
  - Sharpe-based reward function

### Environment Design:
```
Observation (State):
├── Cash position (normalized)
├── Holdings per stock (# shares / hmax)
├── Current prices (normalized)
├── Holdings values (normalized)
└── Technical features (MACD, RSI, ATR, etc.)

Action (Continuous [-1, 1]):
├── Per stock: magnitude controls position size
├── Sign controls direction (+ buy, - sell)
└── Execution enforces position limits

Reward (Sharpe-based):
├── Daily return reward
├── Transaction cost penalty
└── Volatility scaling
```

### Key Features:
1. **Realistic B3 Costs**: 0.03% transaction cost
2. **Position Constraints**: Max 30% per stock
3. **Portfolio Tracking**: Full state history
4. **Sharpe Reward**: Incentivizes risk-adjusted returns

### Example Usage:
```python
from src.env import StockTradingEnvB3

env = StockTradingEnvB3(
    df=train_data,  # From DataProcessor
    stock_dim=3,
    hmax=100,
    initial_amount=100_000,
    buy_cost_pct=0.0003,  # 0.03% B3 costs
    sell_cost_pct=0.0003,
)

obs, _ = env.reset()
for step in range(252):  # 1 year
    action = agent.predict(obs)[0]  # [-1, 1] per stock
    obs, reward, done, _, info = env.step(action)
    print(f"Portfolio: ${info['portfolio_value']:,.0f}, Reward: {reward:.4f}")
    if done:
        break
```

---

## Layer 3: Agent Layer (Deep Reinforcement Learning)
====================================================

Location: `src/agents/`

### Modules:
- `base_agent.py`: Abstract base class
  - Common interface for all DRL agents
  - Training, prediction, evaluation, save/load
  - Evaluation metrics (Sharpe, return, length)

- `drl_agents.py`: Concrete implementations
  - **PPOAgent**: Proximal Policy Optimization
    - Best for: Stable training, sample efficiency
  - **DDPGAgent**: Deep Deterministic Policy Gradient  
    - Best for: Off-policy efficiency, continuous control
  - **A2CAgent**: Advantage Actor-Critic
    - Best for: Fast training, parallelization

### Agent Architecture:
```
Environment
    ↓
Observe State Vector
    ↓
Policy Network (Actor):
├── Input: State (45-dim)
├── Hidden: 2 x 256 units
└── Output: Action (3-dim), μ and σ

Value Network (Critic):
├── Input: State (45-dim)
├── Hidden: 2 x 256 units
└── Output: Scalar value

Learning Loop:
├── Collect Rollout (n_steps = 2048)
├── Compute Advantages (GAE)
├── Update Policy (clip_range = 0.2)
├── Update Value (vf_coef = 0.5)
└── Repeat
```

### Algorithm Comparison:
| Algorithm | Type | Stability | Sample Eff. | Speed | Best For |
|-----------|------|-----------|-------------|-------|----------|
| PPO | On-policy | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Default choice |
| DDPG | Off-policy | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Sample efficiency |
| A2C | On-policy | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Speed & parallelization |

### Example Usage:
```python
from src.agents import PPOAgent

agent = PPOAgent(
    env=env,
    model_name="ppo_trading",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    device="cuda",
)

# Train
agent.train(
    total_timesteps=100_000,
    save_dir="trained_models/"
)

# Evaluate
metrics = agent.evaluate(env, num_episodes=10)
print(f"Mean Reward: {metrics['mean_reward']:.4f}")

# Save/Load
agent.save("trained_models/ppo_best.zip")
agent.load("trained_models/ppo_best.zip")

# Make predictions
obs, _ = env.reset()
action, _ = agent.predict(obs, deterministic=True)
```

---

## Layer 4: Optimization Layer (Hyperparameter Tuning)
=====================================================

Location: `src/optimization/`

### Modules:
- `hyperparameter_optimizer.py`: Bayesian optimization
  - Uses Optuna for hyperparameter search
  - Tree Parzen Estimator sampling
  - Median pruner for early stopping
  - Parallel trial execution

### Optimization Workflow:
```
Define Search Space:
├── learning_rate: [1e-5, 1e-3]
├── n_steps: [512, 4096]
├── batch_size: [32, 256]
└── ... (agent-specific parameters)

For Each Trial:
├── Sample hyperparameters
├── Create agent with sampled params
├── Train for 10,000 steps
├── Evaluate on validation set
└── Return Sharpe ratio

Update Search Distribution:
├── Observe trial results
├── Model distribution with TPE
└── Sample new point likely to improve

After n_trials:
└── Return best parameters
```

### Example Usage:
```python
from src.optimization import HyperparameterOptimizer

def create_env():
    return StockTradingEnvB3(
        df=train_data,
        stock_dim=3,
        initial_amount=100_000,
    )

optimizer = HyperparameterOptimizer(
    agent_type="PPO",
    env_fn=create_env,
    n_jobs=-1,  # Parallel execution
)

results = optimizer.optimize(
    n_trials=100,
    show_progress_bar=True,
)

best_params = results['best_params']
best_value = results['best_value']

# Train final agent with best params
final_agent = PPOAgent(env, **best_params)
final_agent.train(total_timesteps=500_000)
```

---

## Complete Workflow Example
============================

```python
# 1. DATA LAYER
from src.data import DataLoader, DataProcessor

loader = DataLoader()
df = loader.load_multiple_assets(
    assets=["PETR4", "VALE3", "BBAS3"],
    start_date="2023-01-01",
    end_date="2025-12-31",
)

processor = DataProcessor()
df_clean = processor.clean_data(df)
df_features = processor.add_technical_indicators(df_clean)
train_data, test_data = DataProcessor.split_data(df_features)
processor.fit_scaler(train_data, columns=['SMA_20', 'RSI_14'])

# 2. ENVIRONMENT LAYER
from src.env import StockTradingEnvB3

train_env = StockTradingEnvB3(
    df=processor.transform(train_data),
    stock_dim=3,
    initial_amount=100_000,
)

test_env = StockTradingEnvB3(
    df=processor.transform(test_data),
    stock_dim=3,
    initial_amount=100_000,
)

# 3. OPTIMIZATION LAYER (Optional: find best hyperparams)
from src.optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    agent_type="PPO",
    env_fn=lambda: train_env,
)
results = optimizer.optimize(n_trials=50)
best_params = results['best_params']

# 4. AGENT LAYER
from src.agents import PPOAgent

agent = PPOAgent(
    env=train_env,
    **best_params,
)

# Train
agent.train(total_timesteps=200_000, save_dir="trained_models/")

# Evaluate
metrics = agent.evaluate(test_env, num_episodes=10)
print(f"Test Sharpe: {metrics['mean_reward']:.4f}")
```

---

## File Structure
==================

```
DRL_ICproject/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── constants.py           # Shared constants
│   │
│   ├── data/                      # Layer 1: Data
│   │   ├── __init__.py
│   │   ├── data_loader.py         # CSV loading
│   │   └── data_processor.py      # Feature engineering
│   │
│   ├── env/                       # Layer 2: Environment
│   │   ├── __init__.py
│   │   ├── base_env.py            # Abstract base
│   │   └── stock_env_b3.py        # B3 implementation
│   │
│   ├── agents/                    # Layer 3: Agents
│   │   ├── __init__.py
│   │   ├── base_agent.py          # Abstract base
│   │   ├── drl_agents.py          # PPO/DDPG/A2C
│   │   └── models.py              # Legacy PINN integration
│   │
│   ├── optimization/              # Layer 4: Tuning
│   │   ├── __init__.py
│   │   └── hyperparameter_optimizer.py  # Optuna integration
│   │
│   ├── pinn/                      # PINN Models
│   │   ├── __init__.py
│   │   └── heston_pinn.py
│   │
│   ├── backtest/                  # Metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   ├── preprocessor.py            # Legacy
│   ├── data_loader.py             # Legacy
│   └── __init__.py
│
├── config/
│   ├── config.py
│   └── hyperparameters.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── trained_models/
├── results/
├── main.py
├── test_setup.py
├── requirements.txt
└── README.md
```

---

## Best Practices Applied
==========================

### From FinRL:
✓ Clear layer separation (Data, Environment, Agent)
✓ Shared data processing infrastructure  
✓ Modular agent implementations
✓ Standard interface across algorithms

### From Stable-Baselines3:
✓ Gymnasium-compatible environments
✓ Standard policy class hierarchy
✓ Common training callbacks
✓ Save/load functionality

### From Optuna:
✓ Bayesian hyperparameter optimization
✓ Parallel trial execution
✓ Early stopping (pruning)
✓ Tree Parzen Estimator sampling

### General:
✓ Type hints throughout
✓ Comprehensive logging
✓ Error handling & validation
✓ DRY principle (no code duplication)
✓ Configuration-driven design
✓ Abstract base classes for extensibility

---

## Performance Tips
====================

1. **Data Layer**:
   - Use caching to avoid reloading CSVs
   - Parallelize preprocessing with multiprocessing

2. **Environment**:
   - Normalize observations to [-1, 1] for stability
   - Use vectorized environments for parallel collection

3. **Agent**:
   - Start with default PPO, switch to DDPG if sample efficiency needed
   - Use learning rate schedule for fine-tuning
   - Enable gradient clipping to prevent divergence

4. **Optimization**:
   - Use -n_jobs=-1 for parallel optimization
   - Start with 50 trials, increase if computational budget allows
   - Use TPE sampler for Bayesian optimization

---

## Troubleshooting
===================

**Issue**: "No data files found"
- Check filenames in data/raw/ match asset symbols

**Issue**: NaN values in features
- DataProcessor.clean_data() handles this with forward fill

**Issue**: Agent training diverges
- Reduce learning rate (try 1e-4)
- Increase n_steps (try 4096)
- Add entropy coefficient (try ent_coef=0.05)

**Issue**: Poor backtest performance
- Ensure proper train/test split
- Check for lookahead bias in features
- Validate environment costs are realistic

"""