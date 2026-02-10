"""
Reproducibility & Seeding Module
=================================

Enforces reproducibility for scientific studies and regulatory compliance.
Centralizes all random seed initialization.

Usage:
    from src.core.reproducibility import set_all_seeds, get_seed_status
    
    set_all_seeds(seed=42)  # Call FIRST before any random operations
    status = get_seed_status()
    print(status)
"""

import random
import numpy as np
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global seed state
_SEED_STATE = {
    'seed': None,
    'is_set': False,
    'timestamp': None
}


def set_all_seeds(seed: int = 42, verbose: bool = True) -> Dict[str, Any]:
    """
    Set all random seeds for 100% reproducibility.
    
    MUST be called FIRST before any random operations, imports, or network initialization.
    
    Args:
        seed: Random seed (default: 42)
        verbose: Log seed initialization (default: True)
    
    Returns:
        Dictionary with seed status
    
    Raises:
        TypeError: If seed is not an integer
        ValueError: If seed is negative or too large
    
    Examples:
        >>> from src.core.reproducibility import set_all_seeds
        >>> set_all_seeds(seed=42)  # Call FIRST in main()
        >>> # Then proceed with imports and model initialization
    """
    # Validate inputs
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"seed must be in [0, 2^32-1], got {seed}")
    
    # Set all random sources
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # GPU seeds for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # All GPUs
    
    # PyTorch determinism (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Update global state
    from datetime import datetime
    _SEED_STATE['seed'] = seed
    _SEED_STATE['is_set'] = True
    _SEED_STATE['timestamp'] = datetime.now().isoformat()
    
    if verbose:
        logger.info(
            f"✓ Random seeds set to {seed}\n"
            f"  - random.seed({seed})\n"
            f"  - np.random.seed({seed})\n"
            f"  - torch.manual_seed({seed})\n"
            f"  - torch.cuda.manual_seed_all({seed})\n"
            f"  - torch.backends.cudnn.deterministic = True"
        )
    
    return _SEED_STATE.copy()


def get_seed_status() -> Dict[str, Any]:
    """
    Get current seed status.
    
    Returns:
        Dictionary with seed value, set status, and timestamp
    
    Examples:
        >>> status = get_seed_status()
        >>> print(f"Seeds set: {status['is_set']}, Seed value: {status['seed']}")
    """
    return _SEED_STATE.copy()


def assert_reproducible() -> bool:
    """
    Assert that reproducibility is properly set.
    
    Raises:
        RuntimeError: If seeds are not set
    
    Examples:
        >>> assert_reproducible()  # Raises if not seed
    """
    if not _SEED_STATE['is_set']:
        raise RuntimeError(
            "❌ Random seeds not set! Call set_all_seeds() FIRST before any random operations.\n"
            "This is required for reproducibility in scientific studies."
        )
    return True
