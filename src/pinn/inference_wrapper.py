"""
PINN Inference Engine
====================
Loads pre-trained DeepHestonHybrid model and performs safe inference
with validation, caching, and error handling.
Includes LRU cache with TTL (Time-To-Live) to prevent memory leaks.
"""

import torch
import torch.nn as nn
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from collections import OrderedDict
import logging
import sys

logger = logging.getLogger(__name__)


class PINNInferenceEngine:
    """
    Safe wrapper for PINN (DeepHestonHybrid) inference.
    
    Responsibilities:
    - Load checkpoint from best_hybrid_model.pth
    - Validate inputs against training ranges
    - Execute batched inference
    - Denormalize outputs
    - Cache results to avoid re-computation
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_stats_path: str,
        device: str = "cpu",
        dtype: str = "float32",
        enable_validation: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600,  # Cache TTL in seconds (1 hour default)
        verbose: bool = True
    ):
        """
        Initialize PINN inference engine.
        
        Args:
            checkpoint_path: Path to best_hybrid_model.pth
            data_stats_path: Path to data_stats.json (normalization stats)
            device: 'cpu' or 'cuda'
            dtype: 'float32' or 'float64'
            enable_validation: Validate inputs against training ranges
            cache_size: LRU cache maximum items (default: 1000)
            cache_ttl: Cache Time-To-Live in seconds (default: 3600=1 hour)
            verbose: Print logging info
        """
        self.checkpoint_path = checkpoint_path
        self.data_stats_path = data_stats_path
        self.device = device
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.enable_validation = enable_validation
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.verbose = verbose
        
        # Load normalization statistics
        self._load_data_stats()
        
        # Load model architecture and weights (optional - checkpoint may not exist yet)
        try:
            self.model = self._load_checkpoint()
            self.model.to(device)
            self.model.eval()
            self.model_loaded = True
        except FileNotFoundError:
            logger.warning(f"⚠️  Checkpoint not found at {checkpoint_path}")
            logger.warning("   PINN model will be initialized with random weights on first training")
            self.model = None
            self.model_loaded = False
        except RuntimeError as e:
            # Architecture mismatch - this is expected if checkpoint was trained with different config
            error_msg = str(e)
            if "size mismatch" in error_msg or "Unexpected key" in error_msg or "Missing key" in error_msg:
                logger.warning(f"⚠️  Checkpoint architecture mismatch - will initialize fresh")
                logger.warning(f"   Reason: Model was trained with different configuration")
                logger.warning(f"   Example: Different embedding_dim, hidden_size, or neuron counts")
                logger.info(f"   This is normal. Model will learn from current training data.")
                self.model = None
                self.model_loaded = False
            else:
                logger.error(f"❌ Unexpected error loading checkpoint: {e}")
                self.model = None
                self.model_loaded = False
        
        # LRU cache with TTL support
        self.inference_cache = OrderedDict()
        self.cache_timestamps = {}  # Track creation time for TTL
        
        if verbose:
            status = "✅" if self.model_loaded else "⚠️ "
            model_status = "initialized" if self.model_loaded else "pending (will load on first training)"
            logger.info(f"{status} PINN InferenceEngine {model_status} on {device} (cache_size={cache_size}, TTL={cache_ttl}s)")
    
    def _load_data_stats(self):
        """Load normalization statistics from PINN training."""
        try:
            with open(self.data_stats_path, 'r') as f:
                stats = json.load(f)
            
            # Store key statistics
            self.data_stats = stats
            
            # Derive valid ranges (mean ± 3*std for outlier detection)
            self.valid_ranges = {
                'S': (
                    stats.get('S_mean', 1.0) - 3 * stats.get('S_std', 0.5),
                    stats.get('S_mean', 1.0) + 3 * stats.get('S_std', 0.5)
                ),
                'K': (
                    stats.get('K_mean', 1.0) - 3 * stats.get('K_std', 0.5),
                    stats.get('K_mean', 1.0) + 3 * stats.get('K_std', 0.5)
                ),
                'T': (0.01, 10.0),  # 1 day to ~10 years
                'r': (0.0, 0.5),    # 0-50% interest rates
                'q': (0.0, 0.5),    # 0-50% dividend yield
            }
            
            if self.verbose:
                logger.info(f"✅ Loaded data statistics from {self.data_stats_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load data stats: {e}")
            raise
    
    def _load_checkpoint(self) -> nn.Module:
        """
        Load PINN model checkpoint.
        Imports DeepHestonHybrid from src/pinn/model.py
        """
        try:
            # Import model from current package
            from .model import DeepHestonHybrid
            
            # Check if checkpoint exists
            if not os.path.exists(self.checkpoint_path):
                logger.info(f"ℹ️  Checkpoint not found - model will be trained from scratch")
                return None
            
            # Load checkpoint state dict
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Get model config from data_stats (should now match checkpoint architecture)
            model_config = self.data_stats.get('config', {})
            
            if not model_config:
                # Infer config from data_stats if not available
                num_assets = len(self.data_stats.get('asset_map', {}))
                model_config = {
                    'lstm_hidden_size': 64,
                    'lstm_layers': 2,
                    'lstm_dropout': 0.1,
                    'lstm_input_size': 10,
                    'use_asset_embeddings': False,
                    'num_assets': num_assets if num_assets > 0 else 14,
                    'asset_embedding_dim': 4,
                    'use_fourier_features': True,
                    'fourier_features': 128,
                    'fourier_sigma': 10.0,
                    'pinn_hidden_layers': 4,
                    'pinn_neurons': 64,
                    'activation': 'silu',
                    'dropout': 0.0,
                    'deep_layers': [64, 64, 64, 64]
                }
            
            # Instantiate model with config and data_stats
            model = DeepHestonHybrid(config=model_config, data_stats=self.data_stats)
            
            # Load checkpoint weights (should match now)
            try:
                model.load_state_dict(state_dict, strict=True)
                if self.verbose:
                    logger.info(f"✅ Loaded PINN checkpoint successfully")
                model.eval()
                return model
            except RuntimeError as e:
                error_msg = str(e)
                logger.warning(f"⚠️  Checkpoint incompatible: {str(error_msg)[:100]}...")
                logger.info(f"   Model will be trained from scratch")
                return None
        
        except Exception as e:
            logger.debug(f"Could not load checkpoint: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def validate_inputs(
        self,
        x_seq: np.ndarray,
        x_phy: np.ndarray,
        asset_ids: Optional[np.ndarray] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate inputs are within training ranges.
        
        Args:
            x_seq: [batch, 30, 6] - LSTM sequences
            x_phy: [batch, 5] - [S, K, T, r, q]
            asset_ids: [batch] - Asset IDs (optional)
        
        Returns:
            (is_valid: bool, warnings: List[str])
        """
        warnings = []
        
        # Check for NaNs
        if np.isnan(x_seq).any():
            warnings.append("x_seq contains NaN values")
        if np.isnan(x_phy).any():
            warnings.append("x_phy contains NaN values")
        
        # Check x_phy ranges [S, K, T, r, q]
        if x_phy.shape[1] >= 1:
            S = x_phy[:, 0]
            if (S < self.valid_ranges['S'][0]).any() or (S > self.valid_ranges['S'][1]).any():
                warnings.append(
                    f"Spot price S outside range {self.valid_ranges['S']}: "
                    f"[{S.min():.2f}, {S.max():.2f}]"
                )
        
        if x_phy.shape[1] >= 2:
            K = x_phy[:, 1]
            if (K < self.valid_ranges['K'][0]).any() or (K > self.valid_ranges['K'][1]).any():
                warnings.append(
                    f"Strike K outside range {self.valid_ranges['K']}: "
                    f"[{K.min():.2f}, {K.max():.2f}]"
                )
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def preprocess_inputs(
        self,
        x_seq: np.ndarray,
        x_phy: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize inputs using Z-score from training data.
        
        Args:
            x_seq: [batch, 30, 6]
            x_phy: [batch, 5]
        
        Returns:
            Normalized torch tensors on self.device
        """
        
        # Z-score normalization
        x_seq_mean = self.data_stats.get('x_seq_mean', 0.0)
        x_seq_std = self.data_stats.get('x_seq_std', 1.0)
        
        x_phy_mean = self.data_stats.get('x_phy_mean', np.zeros(5))
        x_phy_std = self.data_stats.get('x_phy_std', np.ones(5))
        
        # Compute z-scores
        x_seq_norm = (x_seq - x_seq_mean) / (x_seq_std + 1e-8)
        x_phy_norm = (x_phy - np.array(x_phy_mean)) / (np.array(x_phy_std) + 1e-8)
        
        # Convert to torch tensors
        x_seq_tensor = torch.from_numpy(x_seq_norm).to(
            self.device, dtype=self.dtype
        )
        x_phy_tensor = torch.from_numpy(x_phy_norm).to(
            self.device, dtype=self.dtype
        )
        
        return x_seq_tensor, x_phy_tensor
    
    def infer_heston_params(
        self,
        x_seq: np.ndarray,
        x_phy: np.ndarray,
        asset_ids: Optional[np.ndarray] = None,
        return_price: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run PINN inference and extract Heston parameters.
        
        Args:
            x_seq: [batch, 30, 6] - LSTM sequences
            x_phy: [batch, 5] - [S, K, T, r, q]
            asset_ids: [batch] - Asset IDs (optional)
            return_price: Include option price in output
        
        Returns:
            {
                'nu': [batch, 1] - Instantaneous variance
                'theta': [batch, 1] - Long-term variance (mean)
                'kappa': [batch, 1] - Mean reversion speed
                'xi': [batch, 1] - Volatility of volatility
                'rho': [batch, 1] - Spot-volatility correlation
                'price': [batch, 1] - Theoretical option price (if return_price=True)
            }
        """
        
        # Check if model is loaded
        if not self.model_loaded or self.model is None:
            logger.warning("⚠️ PINN model not loaded yet, returning default Heston parameters")
            batch_size = x_seq.shape[0]
            return {
                'nu': np.ones((batch_size, 1)) * 0.25,      # Default variance
                'theta': np.ones((batch_size, 1)) * 0.25,   # Default long-term variance
                'kappa': np.ones((batch_size, 1)) * 1.0,    # Default mean reversion
                'xi': np.ones((batch_size, 1)) * 0.5,       # Default vol of vol
                'rho': np.ones((batch_size, 1)) * -0.5,     # Default correlation
                'price': np.ones((batch_size, 1)) * 0.0     # Default price
            }
        
        # Validate
        if self.enable_validation:
            is_valid, warnings = self.validate_inputs(x_seq, x_phy, asset_ids)
            for w in warnings:
                logger.warning(f"⚠️ PINN validation: {w}")
        
        # Preprocess
        x_seq_norm, x_phy_norm = self.preprocess_inputs(x_seq, x_phy)
        
        if asset_ids is not None:
            asset_ids_tensor = torch.from_numpy(asset_ids).to(self.device).long()
        else:
            asset_ids_tensor = torch.zeros(x_seq.shape[0]).to(self.device).long()
        
        # Forward pass
        with torch.no_grad():
            try:
                outputs = self.model(x_seq_norm, x_phy_norm, asset_ids_tensor)
                
                # Parse outputs (assuming model returns: price, nu, theta, kappa, xi, rho)
                if isinstance(outputs, tuple):
                    price = outputs[0]
                    heston_params = outputs[1:]
                else:
                    # If model returns dict
                    price = outputs.get('price', torch.zeros(x_seq.shape[0], 1))
                    heston_params = [
                        outputs.get(f'param_{i}', torch.zeros(x_seq.shape[0], 1))
                        for i in range(5)
                    ]
                
            except Exception as e:
                logger.error(f"❌ PINN inference failed: {e}")
                # Return dummy features
                batch_size = x_seq.shape[0]
                price = torch.zeros(batch_size, 1, device=self.device)
                heston_params = [torch.zeros(batch_size, 1, device=self.device) for _ in range(5)]
        
        # Convert to numpy
        results = {
            'nu': heston_params[0].cpu().numpy(),
            'theta': heston_params[1].cpu().numpy(),
            'kappa': heston_params[2].cpu().numpy(),
            'xi': heston_params[3].cpu().numpy(),
            'rho': heston_params[4].cpu().numpy(),
        }
        
        if return_price:
            results['price'] = price.cpu().numpy()
        
        return results
    
    def batch_infer(
        self,
        dataloader: 'torch.utils.data.DataLoader'
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on entire dataset via dataloader.
        
        Args:
            dataloader: PyTorch DataLoader with batches
        
        Returns:
            Aggregated results across all batches
        """
        all_results = {
            'nu': [], 'theta': [], 'kappa': [], 'xi': [], 'rho': [], 'price': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                x_seq, x_phy, asset_ids = batch[0], batch[1], batch[2] if len(batch) > 2 else None
            else:
                x_seq, x_phy = batch['x_seq'], batch['x_phy']
                asset_ids = batch.get('asset_ids', None)
            
            batch_results = self.infer_heston_params(
                x_seq.cpu().numpy(),
                x_phy.cpu().numpy(),
                asset_ids.cpu().numpy() if asset_ids is not None else None
            )
            
            for key in all_results.keys():
                if key in batch_results:
                    all_results[key].append(batch_results[key])
        
        # Concatenate
        return {k: np.vstack(v) if v else np.array([]) for k, v in all_results.items()}
    
    def clear_cache(self):
        """Clear LRU inference cache."""
        self.inference_cache.clear()
        self.cache_timestamps.clear()
        if self.verbose:
            logger.info("✅ Cleared PINN inference cache")
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """
        Get item from cache, checking TTL.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if expired/missing
        """
        if key not in self.inference_cache:
            return None
        
        # Check TTL
        if time.time() - self.cache_timestamps[key] > self.cache_ttl:
            # Expired, remove
            del self.inference_cache[key]
            del self.cache_timestamps[key]
            return None
        
        # Move to end (LRU updating)
        self.inference_cache.move_to_end(key)
        return self.inference_cache[key]
    
    def _cache_put(self, key: str, value: Any) -> None:
        """
        Put item in cache with LRU eviction.
        
        Args:
            key: Cache key
            value: Value to store
        """
        # Update if exists
        if key in self.inference_cache:
            self.inference_cache.move_to_end(key)
        
        self.inference_cache[key] = value
        self.cache_timestamps[key] = time.time()
        
        # LRU eviction: remove oldest if exceeds size
        while len(self.inference_cache) > self.cache_size:
            oldest_key = next(iter(self.inference_cache))
            del self.inference_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
            if self.verbose:
                logger.debug(f"Evicted cache key {oldest_key} (size limit reached)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'size': len(self.inference_cache),
            'max_size': self.cache_size,
            'ttl_seconds': self.cache_ttl,
        }
