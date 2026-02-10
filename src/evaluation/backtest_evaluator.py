"""
Backtesting and Financial Evaluation Module with Pyfolio Integration

Handles:
1. Converting account value time series to returns (required by Pyfolio)
2. Fetching benchmark data (Ibovespa, CDI)
3. Generating tear sheets and financial metrics
4. Comparing against risk-free rate (CDI/SELIC) and market benchmark
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyfolio
except ImportError:
    pyfolio = None

logger = logging.getLogger(__name__)


class BenchmarkData:
    """
    Handles benchmark data retrieval and management.
    
    For B3/Brazil:
    - Market benchmark: Ibovespa (^BVSP)
    - Risk-free rate: CDI (Selic equivalent)
    """
    
    # CDI Daily rates for 2023-2025 (approximate)
    # Real CDI is from Central Bank, these are averages
    CDI_ANNUAL_RATES = {
        2023: 0.1065,  # 10.65% annual
        2024: 0.0975,  # 9.75% annual  
        2025: 0.0945,  # 9.45% annual (estimate)
    }
    
    @staticmethod
    def get_cdi_daily_rate(year: int) -> float:
        """Get daily CDI rate for given year."""
        annual_rate = BenchmarkData.CDI_ANNUAL_RATES.get(year, 0.10)
        daily_rate = (1 + annual_rate) ** (1/252) - 1
        return daily_rate
    
    @staticmethod
    def create_cdi_returns(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Create synthetic CDI returns series for comparison.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Series of daily CDI returns indexed by date
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        daily_returns = []
        
        for date in dates:
            year = date.year
            daily_return = BenchmarkData.get_cdi_daily_rate(year)
            daily_returns.append(daily_return)
        
        cdi_series = pd.Series(daily_returns, index=dates)
        cdi_series.name = 'CDI'
        
        return cdi_series
    
    @staticmethod
    def fetch_ibovespa(
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """
        Fetch Ibovespa benchmark returns.
        
        Uses yfinance to download ^BVSP (Ibovespa index).
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Series of daily returns
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance required. Install: pip install yfinance")
            return None
        
        logger.info(f"Fetching Ibovespa data ({start_date} to {end_date})")
        
        ibov = yf.download(
            "^BVSP",
            start=start_date,
            end=end_date,
            progress=False,
        )
        
        if ibov is None or ibov.empty:
            logger.error("Failed to fetch Ibovespa data")
            return None
        
        # Calculate returns from closing prices
        returns = ibov['Close'].pct_change().dropna()
        returns.name = 'Ibovespa'
        
        return returns


class BacktestEvaluator:
    """
    Comprehensive backtesting evaluator with Pyfolio integration.
    
    Usage:
        evaluator = BacktestEvaluator('/path/to/results.csv')
        evaluator.run_full_analysis()
        evaluator.save_tear_sheet('tear_sheet.html')
    """
    
    def __init__(
        self,
        results_df: pd.DataFrame,
        initial_cash: float = 100000.0,
        save_dir: str = "./backtest_results",
    ):
        """
        Initialize evaluator.
        
        Args:
            results_df: DataFrame with columns ['date', 'account_value', ...]
                        or 'returns' column directly
            initial_cash: Initial portfolio value for returns calculation
            save_dir: Directory to save results
        """
        self.results_df = results_df.copy()
        self.initial_cash = initial_cash
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert date column if needed
        if 'date' in self.results_df.columns:
            self.results_df['date'] = pd.to_datetime(self.results_df['date'])
            self.results_df.set_index('date', inplace=True)
        
        # Calculate returns if needed
        if 'returns' not in self.results_df.columns:
            if 'account_value' in self.results_df.columns:
                self.results_df['returns'] = (
                    self.results_df['account_value'].pct_change()
                )
            else:
                raise ValueError("DataFrame must have 'account_value' or 'returns' column")
        
        self.returns_series = self.results_df['returns'].dropna()
        
        # Metrics storage
        self.metrics = {}
        self.benchmark_returns = None
        self.cdi_returns = None
        self.tear_sheet_html = None
        
        logger.info(f"BacktestEvaluator initialized with {len(self.returns_series)} returns")
    
    def _calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        if len(self.returns_series) == 0:
            return {}
        
        annual_returns = self.returns_series.mean() * 252
        annual_volatility = self.returns_series.std() * np.sqrt(252)
        
        cumulative_returns = (1 + self.returns_series).prod() - 1
        
        # Sharpe ratio (using CDI as risk-free rate)
        cdi_daily = BenchmarkData.get_cdi_daily_rate(datetime.now().year) / 252
        excess_returns = self.returns_series - cdi_daily
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio (downside only)
        downside_returns = self.returns_series[self.returns_series < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (excess_returns.mean() * 252 / downside_std) if downside_std > 0 else 0
        
        # Max drawdown
        cumval = (1 + self.returns_series).cumprod()
        running_max = cumval.expanding().max()
        drawdown = (cumval - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (self.returns_series > 0).sum() / len(self.returns_series)
        
        # Calmar ratio
        calmar = annual_returns / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': float(cumulative_returns),
            'annual_return': float(annual_returns),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'num_trades': int(self.results_df['num_trades'].sum()) if 'num_trades' in self.results_df else 0,
        }
    
    def _calculate_relative_metrics(self) -> Dict[str, float]:
        """Calculate metrics relative to benchmarks."""
        metrics = {}
        
        # vs. CDI
        if self.cdi_returns is not None:
            # Align indices
            common_dates = self.returns_series.index.intersection(self.cdi_returns.index)
            agent_ret = self.returns_series.loc[common_dates]
            cdi_ret = self.cdi_returns.loc[common_dates]
            
            outperformance = (agent_ret - cdi_ret).mean() * 252
            metrics['outperformance_vs_cdi_annual'] = float(outperformance)
            
            cumulative_agent = (1 + agent_ret).prod() - 1
            cumulative_cdi = (1 + cdi_ret).prod() - 1
            metrics['cumulative_outperformance_vs_cdi'] = float(cumulative_agent - cumulative_cdi)
        
        # vs. Ibovespa
        if self.benchmark_returns is not None:
            common_dates = self.returns_series.index.intersection(self.benchmark_returns.index)
            agent_ret = self.returns_series.loc[common_dates]
            bench_ret = self.benchmark_returns.loc[common_dates]
            
            alpha = (agent_ret - bench_ret).mean() * 252
            metrics['alpha_vs_ibovespa_annual'] = float(alpha)
            
            # Beta and correlation
            covariance = np.cov(agent_ret, bench_ret)[0, 1]
            bench_variance = np.var(bench_ret)
            beta = covariance / bench_variance if bench_variance > 0 else 1.0
            metrics['beta_vs_ibovespa'] = float(beta)
            
            correlation = agent_ret.corr(bench_ret)
            metrics['correlation_with_ibovespa'] = float(correlation)
        
        return metrics
    
    def calculate_metrics(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        cdi_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive financial metrics.
        
        Args:
            benchmark_returns: Ibovespa returns series (optional)
            cdi_returns: CDI returns series (optional)
            
        Returns:
            Dictionary of all metrics
        """
        self.benchmark_returns = benchmark_returns
        self.cdi_returns = cdi_returns
        
        # Combine metrics
        self.metrics = {}
        self.metrics.update(self._calculate_basic_metrics())
        self.metrics.update(self._calculate_relative_metrics())
        
        logger.info(f"Calculated {len(self.metrics)} metrics")
        return self.metrics
    
    def generate_tear_sheet(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        show_plots: bool = True,
    ) -> str:
        """
        Generate Pyfolio tear sheet (requires pyfolio installed).
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            show_plots: Whether to display plots in Jupyter
            
        Returns:
            HTML tear sheet (if available)
        """
        if pyfolio is None:
            logger.warning("Pyfolio not installed. Install: pip install pyfolio")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import warnings
            warnings.filterwarnings('ignore')
            
            logger.info("Generating Pyfolio tear sheet...")
            
            # Create tear sheet
            tear_sheet_df = pyfolio.create_full_tear_sheet(
                returns=self.returns_series,
                benchmark_rets=benchmark_returns,
                set_context=False,
                return_dict=True,
            )
            
            if show_plots:
                plt.tight_layout()
            
            logger.info("Tear sheet generated successfully")
            return tear_sheet_df
            
        except Exception as e:
            logger.error(f"Error generating tear sheet: {e}")
            return None
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table: Agent vs. CDI vs. Ibovespa.
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison = {
            'Metric': [
                'Total Return',
                'Annual Return',
                'Annual Volatility',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Calmar Ratio',
                'Win Rate',
            ],
            'Agent': [
                f"{self.metrics.get('total_return', 0):.2%}",
                f"{self.metrics.get('annual_return', 0):.2%}",
                f"{self.metrics.get('annual_volatility', 0):.2%}",
                f"{self.metrics.get('sharpe_ratio', 0):.2f}",
                f"{self.metrics.get('sortino_ratio', 0):.2f}",
                f"{self.metrics.get('max_drawdown', 0):.2%}",
                f"{self.metrics.get('calmar_ratio', 0):.2f}",
                f"{self.metrics.get('win_rate', 0):.2%}",
            ],
        }
        
        # Add benchmark columns if available
        if self.benchmark_returns is not None:
            bench_metrics = self._get_benchmark_metrics(self.benchmark_returns, "Ibovespa")
            comparison['Ibovespa'] = [
                f"{bench_metrics.get('total_return', 0):.2%}",
                f"{bench_metrics.get('annual_return', 0):.2%}",
                f"{bench_metrics.get('annual_volatility', 0):.2%}",
                f"{bench_metrics.get('sharpe_ratio', 0):.2f}",
                f"{bench_metrics.get('sortino_ratio', 0):.2f}",
                f"{bench_metrics.get('max_drawdown', 0):.2%}",
                f"{bench_metrics.get('calmar_ratio', 0):.2f}",
                f"{bench_metrics.get('win_rate', 0):.2%}",
            ]
        
        if self.cdi_returns is not None:
            cdi_metrics = self._get_benchmark_metrics(self.cdi_returns, "CDI")
            comparison['CDI'] = [
                f"{cdi_metrics.get('total_return', 0):.2%}",
                f"{cdi_metrics.get('annual_return', 0):.2%}",
                f"{cdi_metrics.get('annual_volatility', 0):.2%}",
                f"{cdi_metrics.get('sharpe_ratio', 0):.2f}",
                f"{cdi_metrics.get('sortino_ratio', 0):.2f}",
                f"{cdi_metrics.get('max_drawdown', 0):.2%}",
                f"{cdi_metrics.get('calmar_ratio', 0):.2f}",
                f"{cdi_metrics.get('win_rate', 0):.2%}",
            ]
        
        return pd.DataFrame(comparison)
    
    @staticmethod
    def _get_benchmark_metrics(returns: pd.Series, name: str = "Benchmark") -> Dict[str, float]:
        """Calculate metrics for a benchmark series."""
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        cumulative_returns = (1 + returns).prod() - 1
        
        cdi_daily = BenchmarkData.get_cdi_daily_rate(datetime.now().year) / 252
        excess_returns = returns - cdi_daily
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (excess_returns.mean() * 252 / downside_std) if downside_std > 0 else 0
        
        cumval = (1 + returns).cumprod()
        running_max = cumval.expanding().max()
        drawdown = (cumval - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).sum() / len(returns)
        
        calmar = annual_returns / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': float(cumulative_returns),
            'annual_return': float(annual_returns),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
        }
    
    def save_metrics(self, filename: str = "metrics.csv") -> str:
        """Save metrics to CSV."""
        output_path = self.save_dir / filename
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Metrics saved to {output_path}")
        return str(output_path)
    
    def print_summary(self) -> None:
        """Print evaluation summary."""
        comparison = self.generate_comparison_table()
        
        print("\n" + "=" * 90)
        print("BACKTESTING EVALUATION SUMMARY")
        print("=" * 90)
        print(comparison.to_string(index=False))
        print("=" * 90)
        
        # Additional insights
        if 'outperformance_vs_cdi_annual' in self.metrics:
            outperf = self.metrics['outperformance_vs_cdi_annual']
            print(f"\nOutperformance vs. CDI (Annual): {outperf:+.2%}")
            if outperf > 0:
                print("✓ Agent beat the risk-free rate (CDI)")
            else:
                print("✗ Agent underperformed CDI (risk not justified)")
        
        if 'alpha_vs_ibovespa_annual' in self.metrics:
            alpha = self.metrics['alpha_vs_ibovespa_annual']
            print(f"\nAlpha vs. Ibovespa (Annual): {alpha:+.2%}")
            if alpha > 0:
                print("✓ Agent generated positive alpha")
            else:
                print("✗ Agent underperformed benchmark")


def setup_backtesting(
    results_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_cash: float = 100000.0,
) -> Tuple[BacktestEvaluator, pd.Series, pd.Series]:
    """
    Convenience function to setup backtesting with benchmarks.
    
    Args:
        results_df: Results DataFrame
        start_date: Start date for benchmarks (YYYY-MM-DD)
        end_date: End date for benchmarks (YYYY-MM-DD)
        initial_cash: Initial portfolio value
        
    Returns:
        (evaluator, ibovespa_returns, cdi_returns)
    """
    evaluator = BacktestEvaluator(
        results_df,
        initial_cash=initial_cash,
        save_dir="./backtest_results",
    )
    
    if start_date is None:
        start_date = results_df.index.min().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = results_df.index.max().strftime('%Y-%m-%d')
    
    # Fetch benchmarks
    ibov_returns = BenchmarkData.fetch_ibovespa(start_date, end_date)
    cdi_returns = BenchmarkData.create_cdi_returns(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date),
    )
    
    # Calculate metrics
    evaluator.calculate_metrics(
        benchmark_returns=ibov_returns,
        cdi_returns=cdi_returns,
    )
    
    return evaluator, ibov_returns, cdi_returns
