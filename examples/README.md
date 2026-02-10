# Examples Directory

This directory contains practical examples demonstrating how to use the DRL Stock Trading Agent system.

**Supports multiple markets**: B3 (Brazil), NYSE/NASDAQ (US), and other exchanges via configurable parameters.

## Examples Overview

### 1. Data Pipeline (`01_data_pipeline.py`)
**Learn:** How to load, process, and prepare real trading data

**What it covers:**
- Loading raw CSV data in Brazilian format (semicolon separator, comma decimals)
- Cleaning and validating data
- Adding technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Normalizing features with StandardScaler or MinMaxScaler
- Splitting data into train/test sets
- Saving processed data for use in trading environments

**Expected output:**
```
✓ Loaded 754 records
✓ Cleaned data: 754 records
✓ Added 8 indicators
✓ Train: 603 | Test: 151
✓ Saved processed data to data/processed/
```

**Run:**
```bash
python examples/01_data_pipeline.py
```

---

### 2. Trading Environment (`02_environment.py`)
**Learn:** How to interact with the Gymnasium-compatible trading environment

**What it covers:**
- Creating a `StockTradingEnvB3` environment with multiple assets
- Resetting and stepping through trading days
- Understanding the observation space (OHLCV, indicators, portfolio state)
- Understanding the action space (continuous [-1, +1] per stock)
- Monitoring portfolio value, costs, and rewards
- Sharpe ratio-based reward calculation
- Transaction cost modeling (0.03% buy/sell costs)

**Key parameters:**
- `stock_dim`: Number of stocks to trade (default: 3)
- `hmax`: Maximum shares per transaction (default: 100)
- `initial_amount`: Starting capital (default: $100,000)
- `buy_cost_pct` / `sell_cost_pct`: Transaction costs (default: 0.0003 = 0.03%)

**Expected output:**
```
Day     Action       Cash ($)    Portfolio ($)   Reward
0       -0.1234      99,987.32   99,987.32      -0.001234
5       0.0456       99,998.45   100,045.67     0.000456
...
[Summary] 30 days simulated, portfolio value $XXX,XXX.XX
```

**Run:**
```bash
python examples/02_environment.py
```

---

### 3. Training Agents (`03_training_agents.py`)
**Learn:** How to train Deep Reinforcement Learning agents using 3 algorithms

**What it covers:**
- Creating and training a **PPO agent** (Proximal Policy Optimization)
  - Recommended for beginners
  - On-policy, stable, good sample efficiency
- Creating and training a **DDPG agent** (Deep Deterministic Policy Gradient)
  - Off-policy, very sample efficient
  - Best for continuous control with limited data
- Creating and training an **A2C agent** (Advantage Actor-Critic)
  - Fast training with parallel workers
  - Good for distributed architectures
- Evaluating agent performance
- Saving and loading trained models

**Training parameters (PPO):**
```python
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
gamma: 0.99
clip_range: 0.2
```

**Expected output:**
```
Training PPO Agent...
[PPO] Episode 1 | Reward: 0.1234, Avg: 0.1234
[PPO] Episode 2 | Reward: 0.2345, Avg: 0.1789
...
✓ PPO Agent trained
Evaluation: Mean reward: 0.1567, Std: 0.0234
```

**Run:**
```bash
python examples/03_training_agents.py
```

---

### 4. Hyperparameter Optimization (`04_hyperparameter_optimization.py`)
**Learn:** How to use Optuna for Bayesian hyperparameter optimization

**What it covers:**
- Setting up `HyperparameterOptimizer` with TPE sampler
- Defining search spaces for different algorithms
- Running Bayesian optimization trials
- Early stopping with median pruning
- Analyzing optimization results
- Training final agent with optimized parameters
- Comparison of PPO vs DDPG vs A2C

**Search space example (PPO):**
```python
learning_rate: [1e-5, 1e-3]
n_steps: [512, 4096]
batch_size: [32, 256]
n_epochs: [5, 20]
gamma: [0.95, 0.9999]
gae_lambda: [0.8, 1.0]
clip_range: [0.1, 0.4]
```

**Expected output:**
```
[Trial 1] learning_rate=0.0005, n_steps=2048, ... → Sharpe: 0.1234
[Trial 2] learning_rate=0.0003, n_steps=1024, ... → Sharpe: 0.1567
...
Best parameters found:
  learning_rate: 0.0004
  n_steps: 2048
  batch_size: 128
  ...
```

**Run:**
```bash
python examples/04_hyperparameter_optimization.py
```

---

### 5. Complete Workflow (`05_complete_workflow.py`)
**Learn:** Full end-to-end integration of all components

**What it covers:**
Step 1: Data Pipeline
- Load raw CSV files from 3 B3 stocks
- Process and normalize data
- Split into train/test sets

Step 2: Environment Setup
- Create training environment (80% of data)
- Create test environment (20% of data)

Step 3: Agent Training
- Train PPO agent on training data
- Train DDPG agent on training data

Step 4: Evaluation
- Evaluate both agents on test data
- Compute mean reward, std, min/max

Step 5: Backtesting
- Run agents on test data with deterministic actions
- Calculate Sharpe ratio, maximum drawdown
- Count number of trades executed

Step 6: Performance Comparison
- Compare evaluation scores
- Compare backtest metrics
- Identify best algorithm

**Expected workflow:**
```
Step 1: Data Pipeline ✓
  ✓ Loaded 1,000+ records from 3 assets
  ✓ Train: 800, Test: 200

Step 2: Environment Setup ✓
Step 3: Agent Training ✓
Step 4: Evaluation ✓
Step 5: Backtesting ✓
Step 6: Comparison ✓

Results:
  PPO:  Sharpe=0.1234, Max DD=-0.0545
  DDPG: Sharpe=0.0987, Max DD=-0.0678
```

**Run:**
```bash
python examples/05_complete_workflow.py
```

---

### 6. Ensemble Strategy (`06_ensemble_strategy.py`)
**Learn:** How to combine multiple agents using different voting strategies

**What it covers:**
- Creating individual agents (PPO, DDPG, A2C)
- Combining predictions using ensemble voting
- Setting weights based on individual performance
- Evaluating ensemble vs individual agents
- Analyzing voting patterns

**Voting Strategies:**
- `mean`: Simple average of all action suggestions
- `weighted`: Actions weighted by agent performance (Sharpe ratio)
- `majority`: Discrete voting (buy/hold/sell consensus)
- `best`: Always use best-performing agent

**Example:**
```
Individual Agent Performance:
  PPO:  Sharpe=0.50
  DDPG: Sharpe=0.30
  A2C:  Sharpe=0.20

Ensemble Weights (normalized):
  PPO:  50%
  DDPG: 30%
  A2C:  20%

Ensemble Sharpe: 0.58 (beat all individual agents!)
```

**Run:**
```bash
python examples/06_ensemble_strategy.py
```

---

### 7. Rolling Window with Ensemble (`07_rolling_window_ensemble.py`) ⭐
**Learn:** Complete production-grade pipeline with walk-forward validation

**What it covers:**
- Loading multi-asset data
- Deploying rolling window cross-validation (14 weeks train / 4 weeks test)
- Training ensemble on each window
- Evaluating on out-of-sample data
- Aggregating metrics across all windows
- Comparing algorithm performance statistically

**Rolling Window Strategy:**
```
Window 0:  Week 0-15  (14s train) | Week 15-19  (4s test)
Window 1:  Week 13-27 (14s train) | Week 27-31 (4s test)  [2s overlap]
Window 2:  Week 26-40 (14s train) | Week 40-44 (4s test)  [2s overlap]
...
```

**Benefits:**
- Tests on truly out-of-sample data
- Reduces overfitting bias
- Produces robust performance estimates
- Better for regime-changing markets

**Output Example:**
```
Rolling Window Results (5 windows):
┌────────────┬────────────┬────────────┬──────────┐
│ Algorithm  │ Avg Sharpe │ Std Sharpe │ Best/Worst
├────────────┼────────────┼────────────┼──────────┤
│ PPO        │    0.5234  │    0.1203  │ 0.7/0.3
│ DDPG       │    0.4891  │    0.1567  │ 0.6/0.2
│ A2C        │    0.4123  │    0.2001  │ 0.7/0.1
│ ENSEMBLE   │    0.5612  │    0.0891  │ 0.7/0.4  ← BEST!
└────────────┴────────────┴────────────┴──────────┘
```

**Run:**
```bash
python examples/07_rolling_window_ensemble.py
```

---

## Running the Examples

### Prerequisites
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data (run Example 1 first):
```bash
python examples/01_data_pipeline.py
```

### Sequential Order (Recommended)
```bash
# First time setup
python examples/01_data_pipeline.py                          # Creates processed data

# Understanding individual components
python examples/02_environment.py                            # Understand environment
python examples/03_training_agents.py                        # Train single agents
python examples/04_hyperparameter_optimization.py            # Optimize params

# Advanced strategies
python examples/05_complete_workflow.py                      # Simple pipeline
python examples/06_ensemble_strategy.py                      # Multi-agent voting
python examples/07_rolling_window_ensemble.py                # Production pipeline ⭐
```

### Recommended for First Time
1. Start with Example 1 (data loading)
2. Skip to Example 7 (rolling window ensemble) for complete pipeline
3. Return to Examples 2-6 for deep dives into specific components

### Direct Usage
Each example is self-contained. If you skip Example 1:
- Examples 2-6 will load from `data/processed/` (auto-skip if not found)
- Example 7 loads raw CSV directly

---

## Architecture Integration

These examples demonstrate the **complete 4-layer architecture**:

```
┌──────────────────────────────────────────────────────────────┐
│ Data Layer (Examples 1, 7)                                   │
│ DataLoader → DataProcessor → RollingWindowStrategy           │
│                                                               │
│ Features: Multi-format CSV, technical indicators, rolling    │
│ windows (14s train / 4s test / 2s overlap)                   │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Environment Layer (Example 2)                                │
│ BaseStockTradingEnv → StockTradingEnv (Multi-market)         │
│                                                               │
│ Features: Continuous actions, transaction costs,             │
│ Sharpe-based rewards, portfolio tracking                     │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Agent Layer (Examples 3, 4, 6, 7)                            │
│ BaseDRLAgent → [PPOAgent, DDPGAgent, A2CAgent]              │
│                           ↓                                   │
│                    EnsembleAgent (voting)                    │
│                                                               │
│ Features: 3 DRL algorithms, ensemble voting strategies       │
│ (mean, weighted, majority, best)                             │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│ Backtesting & Metrics Layer (Examples 5, 7)                  │
│ PortfolioMetrics: Sharpe, Sortino, Calmar, Max DD, Win Rate │
│                                                               │
│ Features: Aggregate metrics across rolling windows,          │
│ statistical comparison of algorithms                         │
└──────────────────────────────────────────────────────────────┘
```
└─────────────────────────────────────────┘
```

---

## Common Issues & Solutions

### Issue: "FileNotFoundError: data/raw/PETR4.csv not found"
**Solution:** Make sure you have CSV files in `data/raw/`. Download historical data for:
- PETR4.csv (Petrobras)
- VALE3.csv (Vale)
- BBAS3.csv (Banco do Brasil)
- ABEV3.csv (Ambev) [optional]
- MGLU3.csv (Magazine Luiza) [optional]

### Issue: "processed data not found" in Example 2+
**Solution:** Run Example 1 first to generate `data/processed/train_data.csv` and `test_data.csv`

### Issue: Out of memory during training
**Solution:** Reduce:
- `n_steps` parameter (from 2048 to 1024)
- `batch_size` parameter (from 64 to 32)
- Total training timesteps (from 100K to 50K)

### Issue: Optimization taking too long
**Solution:**
- Reduce `n_trials` (from 10 to 5)
- Reduce timesteps per trial in hyperparameter_optimizer
- Set `n_jobs` to parallelize across CPU cores

---

## Output Files

After running examples, you'll have:

```
data/
  processed/
    train_data.csv          ← Created by Example 1
    test_data.csv           ← Created by Example 1

trained_models/
  ppo/                      ← Created by Example 3
    best_model.zip
  ddpg/                     ← Created by Example 3
    best_model.zip
  a2c/                      ← Created by Example 3
    best_model.zip
  ppo_optimized/            ← Created by Example 4
    final_model.zip
```

---

## Parameters Reference

### DataLoader
```python
loader = DataLoader(data_path="data/raw")
loader.load_asset(asset="PETR4", start_date="2022-01-01", end_date="2024-12-31")
```

### DataProcessor
```python
processor = DataProcessor()
processor.clean_data(df)
processor.add_technical_indicators(df)
processor.fit_scaler(df, columns=['SMA_20', 'RSI_14'], scaler_name="features")
df_scaled = processor.transform(df, scaler_name="features")
train, test = DataProcessor.split_data(df, train_ratio=0.8)
```

### StockTradingEnvB3
```python
env = StockTradingEnvB3(
    df=data,
    stock_dim=3,            # Number of stocks
    hmax=100,               # Max shares per trade
    initial_amount=100_000, # Starting capital
    buy_cost_pct=0.0003,    # 0.03% transaction cost
    sell_cost_pct=0.0003,
    gamma=0.99,             # Discount factor
    reward='sharpe_ratio'   # Reward function
)
```

### PPOAgent
```python
agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)
agent.train(total_timesteps=100_000, save_dir="trained_models/ppo/")
metrics = agent.evaluate(n_episodes=10, env=env)
```

### HyperparameterOptimizer
```python
optimizer = HyperparameterOptimizer(
    agent_type="PPO",
    env_fn=lambda: StockTradingEnvB3(...),
    n_jobs=-1  # Parallel execution
)
results = optimizer.optimize(n_trials=100)
```

---

## Next Steps

After running through these examples:

1. **Custom data:** Replace CSV files with your own market data
2. **Different stocks:** Modify `load_multiple_assets(['YOUR_STOCK1', 'YOUR_STOCK2'])`
3. **Longer training:** Increase `total_timesteps` in training
4. **Advanced tuning:** Manually adjust search space in hyperparameter optimizer
5. **Live deployment:** Connect to market data API and use trained models

---

## Further Learning

- See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed architecture documentation
- See [README.md](../README.md) for installation and setup
- See `src/` directory for actual implementations

---

**Good luck with your DRL trading agent! 🚀**
