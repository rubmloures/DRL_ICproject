"""
Training Utilities
==================
Timeout handling, checkpointing, and recovery mechanisms for agent training.
"""

import signal
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Callable
import threading
import time

logger = logging.getLogger(__name__)


class TrainingTimeoutError(Exception):
    """Raised when training exceeds timeout duration."""
    pass


class TimeoutHandler:
    """
    Handle training timeouts using signals (Unix) or threading (Windows).
    
    Allows graceful training interruption with model checkpoint.
    """
    
    def __init__(self, timeout_seconds: int, callback: Optional[Callable] = None):
        """
        Initialize timeout handler.
        
        Args:
            timeout_seconds: Timeout duration in seconds
            callback: Optional callback to execute before timeout (e.g., save model)
        """
        self.timeout_seconds = timeout_seconds
        self.callback = callback
        self.timer = None
        
    def _timeout_callback(self):
        """Internal callback for timeout event."""
        if self.callback:
            logger.warning(f"Training timeout in {self.timeout_seconds}s - executing callback")
            try:
                self.callback()
            except Exception as e:
                logger.error(f"Error in timeout callback: {e}")
        
        raise TrainingTimeoutError(
            f"Training exceeded timeout of {self.timeout_seconds} seconds"
        )
    
    def start(self) -> None:
        """Start the timeout timer."""
        if self.timeout_seconds <= 0:
            return
        
        self.timer = threading.Timer(
            self.timeout_seconds,
            self._timeout_callback
        )
        self.timer.daemon = True
        self.timer.start()
        logger.info(f"Training timeout set to {self.timeout_seconds}s")
    
    def cancel(self) -> None:
        """Cancel the timeout timer."""
        if self.timer:
            self.timer.cancel()
            logger.debug("Training timeout cancelled")
    
    @contextmanager
    def timeout_context(self):
        """Context manager for timeout handling."""
        try:
            self.start()
            yield
        finally:
            self.cancel()


class CheckpointManager:
    """
    Manage model checkpoints during training.
    
    Allows resuming training from checkpoints if interrupted.
    """
    
    def __init__(self, checkpoint_dir: Path, agent_name: str = "agent"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            agent_name: Name of the agent (for checkpoint naming)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.agent_name = agent_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_checkpoint = None
        
        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(self, model: Any, timesteps: int) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to checkpoint
            timesteps: Number of training timesteps
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = (
            self.checkpoint_dir / f"{self.agent_name}_checkpoint_{timesteps}"
        )
        
        try:
            model.save(str(checkpoint_path))
            self.current_checkpoint = checkpoint_path
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(self, model: Any = None) -> Optional[tuple]:
        """
        Load the most recent checkpoint.
        
        Args:
            model: Model object to load into (optional)
        
        Returns:
            Tuple of (model, timesteps) or None if no checkpoint exists
        """
        if not self.current_checkpoint:
            # Find the most recent checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob(f"{self.agent_name}_checkpoint_*"))
            if not checkpoints:
                logger.info("No checkpoint found")
                return None
            self.current_checkpoint = checkpoints[-1]
        
        try:
            # Extract timesteps from filename
            filename = self.current_checkpoint.name
            timesteps = int(filename.split('_')[-1])
            
            logger.info(f"Loaded checkpoint from {self.current_checkpoint}")
            return timesteps
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """
        Remove old checkpoints, keeping only the N most recent.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = sorted(self.checkpoint_dir.glob(f"{self.agent_name}_checkpoint_*"))
        
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                try:
                    # Try to remove both the checkpoint and its metadata
                    import shutil
                    if checkpoint.is_file():
                        checkpoint.unlink()
                    elif checkpoint.is_dir():
                        shutil.rmtree(checkpoint)
                    logger.debug(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")


def safe_train_with_timeout(
    model: Any,
    total_timesteps: int,
    timeout_seconds: int = 300,
    save_callback: Optional[Callable] = None,
    **learn_kwargs
) -> bool:
    """
    Train a model with timeout protection.
    
    Args:
        model: Agent model to train
        total_timesteps: Total training timesteps
        timeout_seconds: Maximum training time in seconds (0 = no timeout)
        save_callback: Callback to save model before timeout
        **learn_kwargs: Additional kwargs for model.learn()
    
    Returns:
        True if training completed successfully, False if timeout occurred
    
    Example:
        >>> success = safe_train_with_timeout(
        ...     agent.model,
        ...     total_timesteps=50000,
        ...     timeout_seconds=300,
        ...     save_callback=lambda: agent.save(Path("backup"))
        ... )
    """
    if timeout_seconds <= 0:
        # No timeout
        model.learn(total_timesteps=total_timesteps, **learn_kwargs)
        return True
    
    timeout_handler = TimeoutHandler(
        timeout_seconds=timeout_seconds,
        callback=save_callback
    )
    
    try:
        with timeout_handler.timeout_context():
            model.learn(total_timesteps=total_timesteps, **learn_kwargs)
        return True
    except TrainingTimeoutError as e:
        logger.error(f"Training timeout: {e}")
        return False
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False
