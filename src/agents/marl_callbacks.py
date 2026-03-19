"""
MARL Rich Live Dashboard Callbacks
====================================
Professional real-time training callbacks for MARL Specialist Pipeline.
Uses the Rich library to render a Bloomberg-style live terminal panel.

Features:
- Per-step live update of: Timesteps, Reward, Regime, Loss
- Color-coded regime indicators
- Normalization stats
- Training progress bar
"""

import time
import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
import csv
import os
from rich import box
from pathlib import Path

console = Console(highlight=False, force_terminal=True)

# Regime colors: ASCII-safe labels
REGIME_DISPLAY = {
    "stable_trending": "[bold green]STABLE TRENDING[/bold green]",
    "normal_ranging":  "[bold cyan]NORMAL RANGING[/bold cyan]",
    "elevated_vol":    "[bold yellow]ELEVATED VOL[/bold yellow]",
    "turbulent_shock": "[bold red]TURBULENT SHOCK[/bold red]",
    "N/A":             "[dim]N/A[/dim]"
}


class MARLRichDashboardCallback(BaseCallback):
    """
    Model 2: Rich Live Dashboard Callback
    ======================================
    Displays a live Bloomberg-style terminal panel during MARL specialist training.

    Live display shows:
    - Asset name being trained
    - Progress bar (timesteps)
    - Rolling reward mean +/- std
    - Current market regime (from env info)
    - Policy loss, value loss, entropy (from SB3 logger)
    - Episode count and length
    - Elapsed time
    """

    def __init__(
        self,
        asset_name: str,
        total_timesteps: int,
        refresh_rate: int = 200,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.asset_name = asset_name
        self.total_timesteps = total_timesteps
        self.refresh_rate = refresh_rate

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._start_time: Optional[float] = None

        # Policy diagnostics from SB3 logger
        self.policy_loss = 0.0
        self.value_loss  = 0.0
        self.entropy     = 0.0

        # Market regime (set by env info if available)
        self.current_regime = "N/A"

        # Rich Live panel context
        self._live: Optional[Live] = None

    # ──────────────────────────────────────────
    # SB3 Lifecycle Hooks
    # ──────────────────────────────────────────

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        console.print(f"\n[bold blue]>> Starting Specialist Training: {self.asset_name}[/bold blue]")
        self._live = Live(
            self._build_panel(),
            console=console,
            refresh_per_second=4,
            transient=False
        )
        self._live.start()

    def _on_step(self) -> bool:
        # Accumulate episode reward and length
        rewards = self.locals.get("rewards", [0.0])
        self._current_episode_reward += float(np.mean(rewards))
        self._current_episode_length += 1

        # Read policy diagnostics from SB3 internal logger
        if hasattr(self.model, "logger") and self.model.logger is not None:
            name_to_value = getattr(self.model.logger, "name_to_value", {})
            self.policy_loss = name_to_value.get("train/policy_loss", self.policy_loss)
            self.value_loss  = name_to_value.get("train/value_loss",  self.value_loss)
            self.entropy     = name_to_value.get("train/entropy_loss", self.entropy)

        # Detect episode end
        dones = self.locals.get("dones", [False])
        if np.any(dones):
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self._current_episode_length)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

        # Try to pull regime from env info
        infos = self.locals.get("infos", [{}])
        for info in infos:
            regime = info.get("current_regime")
            if regime is not None:
                self.current_regime = str(regime)
                break

        # Refresh Rich panel
        if self.n_calls % self.refresh_rate == 0 and self._live:
            self._live.update(self._build_panel())

        return True  # Always continue

    def _on_training_end(self) -> None:
        if self._live:
            self._live.update(self._build_panel(final=True))
            self._live.stop()

        elapsed = time.time() - (self._start_time or time.time())
        best = max(self.episode_rewards, default=0.0)
        console.print(
            f"[bold green]"
            f"[{self.asset_name}] Training done in {elapsed:.1f}s | "
            f"Episodes: {len(self.episode_rewards)} | "
            f"Best Reward: {best:.5f}"
            f"[/bold green]\n"
        )

    # ──────────────────────────────────────────
    # Dashboard Builder
    # ──────────────────────────────────────────

    def _build_panel(self, final: bool = False) -> Panel:
        elapsed = time.time() - (self._start_time or time.time())
        progress_pct = (self.num_timesteps / max(self.total_timesteps, 1)) * 100
        bar_len = 28
        filled  = int(progress_pct / 100 * bar_len)
        bar = "[green]" + "|" * filled + "[/green]" + "-" * (bar_len - filled)

        mean_r = float(np.mean(self.episode_rewards[-20:])) if self.episode_rewards else 0.0
        std_r  = float(np.std(self.episode_rewards[-20:]))  if self.episode_rewards else 0.0
        n_ep   = len(self.episode_rewards)

        regime_txt = REGIME_DISPLAY.get(self.current_regime, f"[dim]{self.current_regime}[/dim]")
        status_txt = (
            "[bold green]TRAINING COMPLETE[/bold green]" if final
            else "[bold cyan]RUNNING...[/bold cyan]"
        )

        # Build the display table
        tbl = Table(show_header=False, box=box.SIMPLE, padding=(0, 2), expand=True)
        tbl.add_column("Key",   style="bold white", width=22)
        tbl.add_column("Value", style="bold yellow")

        tbl.add_row("Asset",          f"[bold magenta]{self.asset_name}[/bold magenta]")
        tbl.add_row("Progress",       f"{bar} {progress_pct:.1f}%")
        tbl.add_row("Elapsed",        f"{elapsed:.1f}s")
        tbl.add_row("Timesteps",      f"{self.num_timesteps:,} / {self.total_timesteps:,}")
        tbl.add_row("Episodes",       str(n_ep))
        tbl.add_row("Mean Reward",    f"{mean_r:+.5f}  +/- {std_r:.5f}")
        tbl.add_row("Policy Loss",    f"{self.policy_loss:.6f}" if self.policy_loss else "---")
        tbl.add_row("Value Loss",     f"{self.value_loss:.6f}"  if self.value_loss  else "---")
        tbl.add_row("Entropy",        f"{self.entropy:.6f}"     if self.entropy      else "---")
        tbl.add_row("Market Regime",  regime_txt)
        tbl.add_row("Status",         status_txt)

        return Panel(
            tbl,
            title=f"[bold white on blue] MARL Specialist :: {self.asset_name} [/bold white on blue]",
            border_style="blue",
            subtitle="[dim]Ctrl+C to interrupt[/dim]"
        )

class MARLStepLoggerCallback(BaseCallback):
    """
    Detailed Step Logger Callback
    =============================
    Logs every step of the training process to a CSV file.
    Captures: Actions, Rewards, Features (Tech & PINN), Real values.
    """
    def __init__(
        self,
        asset_name: str,
        log_dir: str,
        window_idx: int = 0,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.asset_name = asset_name
        self.log_dir = log_dir
        self.window_idx = window_idx
        self.csv_path = os.path.join(log_dir, f"marl_step_log_{asset_name}.csv")
        self.initialized = False
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        info = infos[0] # Single env
        
        # Access the environment's current data row
        # training_env is usually a VecEnv, we need the raw env
        try:
            raw_env = self.training_env.get_attr("unwrapped")[0]
            # In our case it might be directly SingleAssetTradingEnv or wrapped
            if hasattr(raw_env, 'data'):
                features_row = raw_env.data.to_dict()
            else:
                # Fallback to just what's in info
                features_row = {}
        except Exception:
            features_row = {}

        # Build log record
        log_record = {
            "window": self.window_idx,
            "asset": self.asset_name,
            "timestep": self.num_timesteps,
            "date": features_row.get("date", features_row.get("data", "N/A")),
            "close": features_row.get("close", features_row.get("acao_close_ajustado", 0.0)),
            "conviction": info.get("conviction_score", 0.0),
            "reward": float(self.locals.get("rewards", [0.0])[0]),
            "daily_return": info.get("daily_return", 0.0),
            "regime": info.get("regime", "N/A"),
            "cash": info.get("cash", 0.0),
            "holdings": info.get("holdings", [0.0])[0], 
        }

        # Add Heston Params
        heston = info.get("heston_params", {})
        for k, v in heston.items():
            log_record[f"pinn_{k}"] = v

        # Add Tech Indicators if available in features_row
        for col in features_row.keys():
            if any(tech in col for tech in ["SMA", "RSI", "MACD", "ATR"]):
                log_record[col] = features_row[col]

        # Initialize CSV and Write Header
        if not self.initialized:
            os.makedirs(self.log_dir, exist_ok=True)
            self.file_exists = os.path.isfile(self.csv_path)
            self.initialized = True
            
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_record.keys())
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
            writer.writerow(log_record)

        return True
