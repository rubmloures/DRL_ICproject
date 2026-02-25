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
    2) Adaptation Phase: Costs gradually increase from 0% to 100%.
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
        
    def _on_step(self) -> bool:
        # Calculate current progress (0.0 to 1.0)
        progress = self.num_timesteps / self.total_timesteps
        
        if progress < self.discovery_fraction:
            multiplier = 0.0
        elif progress < (self.discovery_fraction + self.adaptation_fraction):
            elapsed_in_adaptation = progress - self.discovery_fraction
            multiplier = elapsed_in_adaptation / self.adaptation_fraction
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
