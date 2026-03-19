import logging
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

class CurriculumCallback(BaseCallback):
    """
    Callback for Curriculum Learning to mitigate Frictional Policy Collapse.
    Progressively increases transaction costs during training.
    
    Phases:
    1) Discovery Phase (Pure Exploration): Transaction costs are zeroed.
       The agent learns to find Alpha (beat benchmark) without fear of being punished for trading.
    2) Adaptation Phase: Costs gradually increase from 0% to 100% using a non-linear (logarithmic-like) curve.
       The agent, which already knows how to make money, now learns to optimize 
       the number of trades so as not to return profits in fees.
    """
    
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        # Configuration for phases
        self.discovery_fraction = 0.2  # First 20%: zero costs
        self.adaptation_fraction = 0.6  # From 20% to 80%: gradual increase
        
        self.last_logged_multiplier = -1.0
        self.start_timesteps = 0
        
    def _on_training_start(self) -> None:
        """Capture the starting timesteps of the agent for warm-start scenarios."""
        self.start_timesteps = self.num_timesteps
        logger.info(f"[Curriculum] Starting from step {self.start_timesteps}")
        
    def _on_step(self) -> bool:
        # Calculate relative progress within the current training session (window)
        current_session_steps = self.num_timesteps - self.start_timesteps
        progress = current_session_steps / self.total_timesteps
        
        # Clamp progress to [0, 1] to avoid artifacts if total_timesteps is exceeded
        progress = max(0.0, min(1.0, progress))
        
        if progress < self.discovery_fraction:
            multiplier = 0.0
        elif progress < (self.discovery_fraction + self.adaptation_fraction):
            elapsed_in_adaptation = progress - self.discovery_fraction
            linear_progress = elapsed_in_adaptation / self.adaptation_fraction
            # Non-linear scaling: square root or logarithmic
            # Using x^(0.5) makes it grow quickly at first, then slow down as it approaches 1.0
            # multiplier = linear_progress ** 0.5 
            # Alternatively, an exponential approach if we want slow then fast
            
            # For logarithmic-like curve that quickly reaches a decent penalty but slowly approaches 100%:
            # log(1 + 9x) / log(10) maps [0,1] -> [0,1] with a steep start and flat end
            import math
            multiplier = math.log10(1 + 9 * linear_progress)
        else:
            multiplier = 1.0
            
        # Try to apply the multiplier to the environment(s)
        # Stable Baselines often wraps environments in a DummyVecEnv
        if hasattr(self.training_env, "env_method"):
            # It's a vectorized environment
            self.training_env.env_method("set_cost_multiplier", multiplier)
        elif hasattr(self.training_env, "envs"):
            # Alternative unwrapping
            for env in self.training_env.envs:
                if hasattr(env.unwrapped, "set_cost_multiplier"):
                    env.unwrapped.set_cost_multiplier(multiplier)
        elif hasattr(self.training_env, "set_cost_multiplier"):
            # Not vectorized
            self.training_env.set_cost_multiplier(multiplier)
        
        # Log to terminal (as requested: [Curriculum] Multiplicador de Custos: X.XX)
        rounded_mult = round(multiplier, 2)
        # Avoid logging the same value too many times; log roughly every 10% change
        if rounded_mult >= self.last_logged_multiplier + 0.1 or (multiplier == 1.0 and self.last_logged_multiplier < 1.0):
            print(f"[Curriculum] Multiplicador de Custos: {rounded_mult:.2f}")
            logger.info(f"[Curriculum] Multiplicador de Custos: {rounded_mult:.2f}")
            self.last_logged_multiplier = rounded_mult
            
        return True
