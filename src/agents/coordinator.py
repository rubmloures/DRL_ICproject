"""
MARL Coordinator / Mixing Network
==================================
Implements the Multi-Agent Reinforcement Learning Coordinator for Portfolio Management.

The Coordinator receives Conviction Scores from SingleAssetTradingEnv specialists
and allocates the portfolio balance (Cash) according to Value Decomposition (VDN)
principles and Difference Rewards (D_z).
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class PortfolioCoordinator:
    """
    Acts as the Meta-Agent orchestrating the Specialist Agents.
    
    Responsibility:
    - Receive [Conviction Scores] from N specialized agents.
    - Compute the optimal [Target Portfolio Weights].
    - Calculate [Difference Rewards (D_z)] to train the specialists locally.
    
    Algorithm (Softmax Routing):
    W_i = exp(Conviction_i * Temperature) / sum(exp(Conviction_j * Temperature))
    """
    
    def __init__(
        self,
        assets: List[str],
        temperature: float = 2.0,
        cash_buffer_penalty: float = 0.05
    ):
        """
        Initialize the Portfolio Coordinator.
        
        Args:
            assets: List of asset names (e.g., ['PETR4', 'VALE3'])
            temperature: Softmax temperature. Higher = more aggressive routing to the highest conviction.
            cash_buffer_penalty: Minimum cash allocation constraint (if any).
        """
        self.assets = assets
        self.temperature = temperature
        self.cash_buffer_penalty = cash_buffer_penalty
        
        # Tracking history for Difference Rewards
        self.history = defaultdict(list)
        logger.info(f"Initialized MARL PortfolioCoordinator for assets: {assets}")

    def route_capital(self, convictions: Dict[str, float]) -> Dict[str, float]:
        """
        Convert raw Conviction Scores [-1, 1] into Portfolio Weights [0, 1].
        
        Args:
            convictions: Dict mapping asset name to its conviction score.
                         e.g., {'PETR4': 0.8, 'VALE3': -0.2}
                         
        Returns:
            Dict of target weights that sum to 1.0 (including 'Cash').
        """
        weights = {}
        
        # Step 1: Filter out negative convictions (we only go Long in this B3 configuration)
        # In a Long/Short env, we would allocate negative weights to Short positions.
        positive_convictions = {k: max(0.0, v) for k, v in convictions.items()}
        
        # Step 2: Compute Softmax
        total_conviction = sum(positive_convictions.values())
        
        if total_conviction <= 1e-4:
            # Defensive Routing: All agents are bearish/neutral. Allocate everything to Cash.
            weights['Cash'] = 1.0
            for asset in self.assets:
                weights[asset] = 0.0
        else:
            # Aggressive Routing: Softmax distribution
            exp_sum = sum(np.exp(v * self.temperature) for v in positive_convictions.values() if v > 0)
            
            allocated_to_assets = 0.0
            for asset in self.assets:
                v = positive_convictions.get(asset, 0.0)
                if v > 0:
                    weight = np.exp(v * self.temperature) / exp_sum
                    # Discount a minimal cash buffer
                    weight = weight * (1.0 - self.cash_buffer_penalty)
                    weights[asset] = weight
                    allocated_to_assets += weight
                else:
                    weights[asset] = 0.0
                    
            weights['Cash'] = 1.0 - allocated_to_assets
            
        return weights

    def compute_difference_reward(
        self, 
        global_reward: float, 
        asset_name: str, 
        asset_return: float, 
        risk_free_rate: float
    ) -> float:
        """
        Compute Difference Reward D_z(i) = G(z) - G(z - i).
        
        Evaluates the marginal contribution of a specific Specialist Agent (i)
        to the Global Portfolio Reward (G(z)).
        
        Args:
            global_reward: The actual return/reward of the entire portfolio step.
            asset_name: Name of the asset to compute D_z for.
            asset_return: The standalone return of ONLY this asset during this step.
            risk_free_rate: Return of the 'Cash' asset (e.g., Daily SELIC).
            
        Returns:
            D_z: The unique reward to be fed back to the specialist agent.
        """
        # G(z) is the global_reward.
        
        # Fast Approximation of G(z - i): 
        # "What if the capital given to agent 'i' had simply stayed in Cash rendering SELIC?"
        # The penalty/bonus is exactly the alpha generated over the risk-free rate.
        
        # Local Value Added:
        local_alpha = asset_return - risk_free_rate
        
        # D_z represents Global Reward dynamically penalized/boosted by the Local Alpha
        # Meaning: You only get the global team bonus if you actually helped the team.
        d_z = global_reward + local_alpha
        
        return float(d_z)
