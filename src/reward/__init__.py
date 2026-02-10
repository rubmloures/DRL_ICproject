"""
Reward Functions Module
=======================

Contains composite and adaptive reward functions for RL agents,
with PINN-driven regime detection for dynamic weight adjustment.
"""

from .composite_reward import CompositeRewardCalculator
from .regime_detector import RegimeDetector

__all__ = ['CompositeRewardCalculator', 'RegimeDetector']
