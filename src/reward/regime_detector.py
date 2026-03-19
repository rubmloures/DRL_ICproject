"""
Regime Detector
===============

Detects market regimes (stable/turbulent) using PINN signals
and market microstructure indicators. Drives dynamic reward weighting.

Based on:
- Heston volatility of volatility (xi from PINN)
- Historical volatility patterns
- Market microstructure entropy
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Enumeration of market regimes detected by PINN."""
    STABLE_TRENDING = "stable_trending"       # Low volatility, clear trend
    NORMAL_RANGING = "normal_ranging"         # Normal vol, rangebound
    ELEVATED_VOLATILITY = "elevated_vol"      # Medium-high vol, still tradeable
    TURBULENT_SHOCK = "turbulent_shock"       # High vol, structural breaks, crisis


class RegimeDetector:
    """
    Detects market regime using PINN output and market indicators.
    
    The PINN Heston parameters provide a physics-informed view of market:
    - nu (ν): Long-run mean volatility
    - xi (ξ): Volatility of volatility (vol clustering strength)
    - rho (ρ): Correlation between price and volatility
    
    These serve as primary signals for regime classification.
    
    Attributes
    ----------
    threshold_stable : float
        Threshold for long-run vol (nu) to be considered stable [0.05-0.15]
    threshold_xi_high : float
        Threshold for vol-of-vol (xi) to trigger turbulence [0.3-0.5]
    threshold_rho_negative : float
        Threshold for negative correlation (leverage effect) [-0.8 to -0.2]
    recent_window : int
        Number of recent obs to consider for trend detection
    """
    
    def __init__(
        self,
        threshold_stable: float = 0.12,
        threshold_xi_high: float = 0.4,
        threshold_rho_negative: float = -0.5,
        recent_window: int = 20,
        volatility_percentile_high: float = 0.75,
    ):
        """
        Initialize regime detector.
        
        Parameters
        ----------
        recent_window : int
            Number of periods for recent vol calculation
        volatility_percentile_high : float
            Percentile of historical vol to classify as "high"
        """
        self.threshold_stable = threshold_stable
        self.threshold_xi_high = threshold_xi_high
        self.threshold_rho_negative = threshold_rho_negative
        self.recent_window = recent_window
        self.volatility_percentile_high = volatility_percentile_high
        
        # History for regime tracking
        self.vol_history: List[float] = []
        self.regime_history: List[str] = []
        
        # KMeans cluster components
        self.kmeans = None
        self.scaler = None
        self.cluster_to_regime_map = {}
        
        logger.info(
            f"RegimeDetector initialized (K-Means Ready):\n"
            f"  Recent window: {recent_window} periods"
        )
        
    def fit_kmeans(self, feature_df: pd.DataFrame) -> None:
        """
        Train unsupervised KMeans on historical PINN features to determine dynamic thresholds.
        
        Parameters
        ----------
        feature_df : pd.DataFrame
            Historical data containing 'nu', 'xi', and 'rho' columns for the specific asset.
        """
        if not all(col in feature_df.columns for col in ['nu', 'xi', 'rho']):
            logger.warning("Feature DataFrame missing required PINN columns (nu, xi, rho) for K-Means. Falling back to static.")
            return

        X = feature_df[['nu', 'xi', 'rho']].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 4 Clusters representing the 4 Regimes
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        self.kmeans.fit(X_scaled)
        
        # Map clusters to regimes based on their centroids' risk profile
        # Higher nu (mean vol) and xi (vol-of-vol), and lower rho (negative correlation) = Higher Risk
        centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        # Compute a synthetic "Risk Score" for each cluster centroid
        # Risk = nu(normalized) + xi(normalized) - rho(normalized)
        risk_scores = centroids[:, 0] + centroids[:, 1] - centroids[:, 2]
        
        # Sort cluster indices by risk score
        sorted_indices = np.argsort(risk_scores)
        
        # Map ordered indices to Regimes
        regimes_ordered = [
            MarketRegime.STABLE_TRENDING,
            MarketRegime.NORMAL_RANGING,
            MarketRegime.ELEVATED_VOLATILITY,
            MarketRegime.TURBULENT_SHOCK
        ]
        
        self.cluster_to_regime_map = {
            sorted_indices[i]: regimes_ordered[i] for i in range(4)
        }
        
        logger.info(f"KMeans Regime Detector fitted. Centroid Risk Map: {self.cluster_to_regime_map}")
    
    def detect_regime(
        self,
        heston_params: Dict[str, float],
        current_volatility: float,
        vol_history: Optional[List[float]] = None,
        recent_returns: Optional[np.ndarray] = None,
    ) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect current market regime using PINN Heston parameters.
        
        Parameters
        ----------
        heston_params : dict
            PINN output with keys: 'nu', 'theta', 'kappa', 'xi', 'rho'
        current_volatility : float
            Realized volatility at current step
        vol_history : list, optional
            Historical volatility values for reference
        recent_returns : np.ndarray, optional
            Recent returns for additional analysis
        
        Returns
        -------
        regime : MarketRegime
            Detected regime enum
        confidence : dict
            Confidence scores for each regime indicator
        """
        self.vol_history.append(current_volatility)
        if len(self.vol_history) > 500:
            self.vol_history = self.vol_history[-500:]
        
        # Extract Heston parameters
        nu = heston_params.get('nu', 0.1)        # Mean volatility
        xi = heston_params.get('xi', 0.3)        # Vol-of-vol
        rho = heston_params.get('rho', -0.3)     # Price-vol correlation
        
        confidence = {'kmeans_used': False}
        
        if self.kmeans is not None and self.scaler is not None:
            # Dynamic Unsupervised Classification
            X_new = np.array([[nu, xi, rho]])
            X_scaled = self.scaler.transform(X_new)
            
            # Predict cluster
            cluster_idx = self.kmeans.predict(X_scaled)[0]
            regime = self.cluster_to_regime_map[cluster_idx]
            
            # Distance to cluster center as confidence proxy
            distances = self.kmeans.transform(X_scaled)[0]
            min_dist = distances[cluster_idx]
            
            # Simple confidence inverse to distance
            confidence['kmeans_used'] = True
            confidence['centroid_distance'] = float(min_dist)
            
            logger.debug(f"K-Means Inference: Predicted cluster {cluster_idx} -> Regime {regime.value} (dist={min_dist:.4f})")
            
        else:
            # Fallback to legacy static threshold logic if K-Means is not trained
            vol_stability_score = 1.0 - np.clip(nu / 0.5, 0, 1)
            confidence['vol_stability'] = float(vol_stability_score)
            
            vol_clustering_score = np.clip(xi / self.threshold_xi_high, 0, 1)
            confidence['vol_clustering'] = float(vol_clustering_score)
            
            correlation_score = 1.0 if rho < self.threshold_rho_negative else 0.0
            confidence['leverage_effect'] = float(correlation_score)
            
            if self.vol_history:
                vol_percentile = (np.sum(np.array(self.vol_history[:-1]) < current_volatility) / max(len(self.vol_history) - 1, 1))
                confidence['historical_percentile'] = float(vol_percentile)
            else:
                vol_percentile = 0.5
                confidence['historical_percentile'] = 0.5
            
            is_stable = (vol_stability_score > 0.6 and vol_clustering_score < 0.4 and vol_percentile < self.volatility_percentile_high)
            is_turbulent = (vol_clustering_score > 0.6 or (correlation_score > 0.5 and vol_percentile > 0.7) or (current_volatility > np.percentile(self.vol_history, 85) if self.vol_history else False))
            is_elevated = (vol_percentile > self.volatility_percentile_high and not is_turbulent)
            
            if is_turbulent: regime = MarketRegime.TURBULENT_SHOCK
            elif is_elevated: regime = MarketRegime.ELEVATED_VOLATILITY
            elif is_stable: regime = MarketRegime.STABLE_TRENDING
            else: regime = MarketRegime.NORMAL_RANGING
        
        self.regime_history.append(regime.value)
        
        logger.debug(
            f"Regime detected: {regime.value} | "
            f"nu={nu:.3f}, xi={xi:.3f}, rho={rho:.3f}, "
            f"KMeans={confidence.get('kmeans_used', False)}"
        )
        
        return regime, confidence
    
    def get_regime_weights(
        self,
        regime: MarketRegime,
        base_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Get dynamic reward weights for the detected regime.
        
        Parameters
        ----------
        regime : MarketRegime
            Current market regime
        base_weights : dict, optional
            Base weights to multiply by regime factors
        
        Returns
        -------
        dict
            Adjusted weights: {
                'excess_return': float,
                'downside_risk': float,
                'alpha_return': float,
                'transaction_cost': float,
            }
        """
        if base_weights is None:
            base_weights = {
                'excess_return': 1.0,
                'downside_risk': 1.0,
                'alpha_return': 1.0,
                'transaction_cost': 1.0,
            }
        
        # Regime-specific weight multipliers
        regime_multipliers = {
            MarketRegime.STABLE_TRENDING: {
                'excess_return': 1.0,      # Normal focus on returns
                'downside_risk': 0.5,      # Accept vol during trends
                'alpha_return': 1.5,       # Strong focus on beating market
                'transaction_cost': 0.8,   # Lower penalty, can trade more
            },
            MarketRegime.NORMAL_RANGING: {
                'excess_return': 1.0,      # Balanced
                'downside_risk': 1.0,      # Normal risk aversion
                'alpha_return': 1.0,       # Normal focus on alpha
                'transaction_cost': 1.0,   # Standard penalty
            },
            MarketRegime.ELEVATED_VOLATILITY: {
                'excess_return': 0.8,      # Reduce aggressive chasing
                'downside_risk': 1.5,      # Increase risk penalty
                'alpha_return': 0.8,       # De-emphasize alpha in chaos
                'transaction_cost': 1.2,   # Higher penalty to reduce trading
            },
            MarketRegime.TURBULENT_SHOCK: {
                'excess_return': 0.6,      # Increased from 0.3 for risk seeking
                'downside_risk': 1.8,      # Reduced from 3.0 to avoid paralysis
                'alpha_return': 0.5,       # Increased from 0.1
                'transaction_cost': 1.5,   # Reduced from 2.0
            },
        }
        
        # Get multipliers for current regime
        multipliers = regime_multipliers.get(
            regime,
            regime_multipliers[MarketRegime.NORMAL_RANGING]
        )
        
        # Apply multipliers to base weights
        adjusted_weights = {
            component: base_weights[component] * multipliers[component]
            for component in base_weights.keys()
        }
        
        # Normalize so they sum to a reasonable value
        # (optional: sum to 1 for probability-like interpretation)
        # For now, keep absolute values for interpretability
        
        logger.debug(f"Adjusted weights for {regime.value}: {adjusted_weights}")
        
        return adjusted_weights
    
    def get_regime_stats(self) -> Dict[str, Any]:
        """
        Get statistics about detected regimes.
        
        Returns
        -------
        dict
            Statistics including regime distribution, durations, etc.
        """
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for regime in self.regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        stats = {
            'total_periods': len(self.regime_history),
            'regime_distribution': regime_counts,
            'current_regime': self.regime_history[-1],
            'recent_regimes': self.regime_history[-20:],
        }
        
        # Compute average duration per regime (runs of same regime)
        if len(self.regime_history) > 1:
            durations = {}
            current_regime = self.regime_history[0]
            current_duration = 1
            
            for regime in self.regime_history[1:]:
                if regime == current_regime:
                    current_duration += 1
                else:
                    if current_regime not in durations:
                        durations[current_regime] = []
                    durations[current_regime].append(current_duration)
                    current_regime = regime
                    current_duration = 1
            
            # Average duration per regime
            avg_durations = {
                regime: np.mean(durs) for regime, durs in durations.items()
            }
            stats['avg_regime_duration'] = avg_durations
        
        return stats
    
    def reset(self) -> None:
        """Reset history for new episode."""
        self.vol_history = []
        self.regime_history = []
        logger.debug("RegimeDetector reset")
