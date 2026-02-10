"""
Results Manager
===============
Save and load training results, metrics, and models.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Manage saving and loading of training results, metrics, and models.
    """
    
    def __init__(self, results_dir: Path = None):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory to save results (default: results/)
        """
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.metrics_dir = self.results_dir / "metrics"
        self.models_dir = self.results_dir / "models"
        self.plots_dir = self.results_dir / "plots"
        
        for dir_path in [self.metrics_dir, self.models_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResultsManager initialized: {self.results_dir}")
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        name: str,
        timestamp: bool = True,
    ) -> Path:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            name: Name of the metric file (without extension)
            timestamp: Append timestamp to filename
        
        Returns:
            Path to saved file
        """
        # Convert non-serializable types
        clean_metrics = self._serialize_metrics(metrics)
        
        # Create filename
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{ts}.json"
        else:
            filename = f"{name}.json"
        
        filepath = self.metrics_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(clean_metrics, f, indent=2)
            logger.info(f"Saved metrics to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return None
    
    def save_metrics_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        timestamp: bool = True,
    ) -> Path:
        """
        Save metrics as CSV (for time series data).
        
        Args:
            df: DataFrame with metrics
            name: Name of the CSV file (without extension)
            timestamp: Append timestamp to filename
        
        Returns:
            Path to saved file
        """
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{ts}.csv"
        else:
            filename = f"{name}.csv"
        
        filepath = self.metrics_dir / filename
        
        try:
            df.to_csv(filepath)
            logger.info(f"Saved metrics dataframe to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save metrics dataframe: {e}")
            return None
    
    def save_model(
        self,
        model: Any,
        agent_name: str,
        metrics: Optional[Dict] = None,
    ) -> Path:
        """
        Save a trained model.
        
        Args:
            model: Model to save
            agent_name: Name of the agent (PPO, DDPG, A2C, ensemble)
            metrics: Optional metrics associated with model
        
        Returns:
            Path to saved model
        """
        model_dir = self.models_dir / agent_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"{agent_name}_{ts}"
        
        try:
            model.save(str(model_path))
            logger.info(f"Saved {agent_name} model to {model_path}")
            
            # Save metadata if metrics provided
            if metrics:
                metadata = {
                    'agent': agent_name,
                    'timestamp': ts,
                    'metrics': self._serialize_metrics(metrics),
                }
                metadata_path = model_dir / f"{agent_name}_{ts}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved model metadata to {metadata_path}")
            
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
    
    def save_summary(
        self,
        pipeline_type: str,
        results: Dict[str, Any],
        agents: Optional[Dict] = None,
    ) -> Path:
        """
        Save a comprehensive training summary.
        
        Args:
            pipeline_type: Type of pipeline (simple, rolling, optuna)
            results: Results dictionary from pipeline
            agents: Optional dict of trained agents to save
        
        Returns:
            Path to summary file
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.results_dir / f"summary_{pipeline_type}_{ts}.json"
        
        # Prepare summary
        summary = {
            'pipeline_type': pipeline_type,
            'timestamp': ts,
            'results': self._serialize_metrics(results),
        }
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved training summary to {summary_path}")
            
            # Save best models if provided
            if agents:
                logger.info("Saving best models...")
                for agent_name, agent in agents.items():
                    metrics = results.get(f'{agent_name.lower()}_metrics', {})
                    self.save_model(agent.model, agent_name, metrics)
            
            return summary_path
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            return None
    
    def save_plot(
        self,
        fig: Any,
        name: str,
        fmt: str = 'png',
    ) -> Path:
        """
        Save a matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            name: Name for the plot
            fmt: File format (png, pdf, jpg)
        
        Returns:
            Path to saved plot
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{ts}.{fmt}"
        filepath = self.plots_dir / filename
        
        try:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
            return None
    
    def load_metrics(self, filepath: Path) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}
    
    @staticmethod
    def _serialize_metrics(obj: Any) -> Any:
        """
        Convert non-serializable objects to serializable format.
        
        Args:
            obj: Object to serialize
        
        Returns:
            Serialized object
        """
        if isinstance(obj, dict):
            return {k: ResultsManager._serialize_metrics(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ResultsManager._serialize_metrics(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (Path, datetime)):
            return str(obj)
        else:
            return obj
    
    def get_latest_model(self, agent_name: str) -> Optional[Path]:
        """Get path to latest saved model for an agent."""
        agent_dir = self.models_dir / agent_name
        if not agent_dir.exists():
            return None
        
        # Find latest model file (excluding metadata)
        models = sorted([f for f in agent_dir.glob(f"{agent_name}_*") 
                        if not f.name.endswith('_metadata.json')])
        
        return models[-1] if models else None
    
    def list_results(self) -> Dict[str, list]:
        """List all saved results."""
        return {
            'metrics': sorted([f.name for f in self.metrics_dir.glob('*.json')]),
            'models': sorted([d.name for d in self.models_dir.iterdir() if d.is_dir()]),
            'plots': sorted([f.name for f in self.plots_dir.glob('*')]),
        }
