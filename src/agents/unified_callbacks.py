"""
Unified Training Callbacks & Dashboards
=========================================
Professional real-time monitoring and auditing for all DRL pipelines.
Supports: MARL, Ensemble, and Parallel Asset-Specific training.
"""

import time
import os
import csv
import numpy as np
import datetime
import collections
from typing import Optional, Dict, Any, List
from stable_baselines3.common.callbacks import BaseCallback
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box

console = Console(highlight=False, force_terminal=True)

REGIME_DISPLAY = {
    "stable_trending": "[bold green]STABLE TRENDING[/bold green]",
    "normal_ranging":  "[bold cyan]NORMAL RANGING[/bold cyan]",
    "elevated_vol":    "[bold yellow]ELEVATED VOL[/bold yellow]",
    "turbulent_shock": "[bold red]TURBULENT SHOCK[/bold red]",
    "UNKNOWN":         "[dim]UNKNOWN[/dim]"
}

class UnifiedRichDashboard(BaseCallback):
    """
    Unified Bloomberg-style terminal dashboard for all RL training.
    """
    def __init__(
        self,
        name: str,
        total_timesteps: int,
        window_idx: int = 0,
        refresh_rate: int = 10,
        verbose: int = 0,
        queue: Optional[Any] = None  # NEW: For parallel monitoring
    ):
        super().__init__(verbose)
        self.name = name.upper().strip() # Ensure name is always uppercase and clean
        self.total_timesteps = total_timesteps
        self.window_idx = window_idx
        self.refresh_rate = refresh_rate
        self.queue = queue
        self.headless = queue is not None
        
        # Metrics
        self.episode_rewards = []
        self._current_reward = 0.0
        self._start_time = None
        
        # Diagnostics
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy_loss = 0.0
        self.learning_rate = 0.0
        self.fps = 0
        self.iterations = 0
        self.time_elapsed = 0.0
        self.ep_len_mean = 0.0
        self.ep_rew_mean_sb3 = 0.0
        
        self.regime = "UNKNOWN"
        self.regime_conf = 0.0
        self.heston = {}
        
        # Portfolio Stats (from info)
        self.portfolio_value = 0.0
        self.portfolio_history = collections.deque(maxlen=40)
        self.cash_pct = 0.0
        self.daily_return = 0.0
        
        self.live = None

    def _on_training_start(self):
        self._start_time = time.time()
        if not self.headless:
            self.live = Live(
                self._build_layout(), 
                console=console, 
                refresh_per_second=4,
                vertical_overflow="crop",
                transient=True
            )
            self.live.start()
        else:
            # Em modo headless, apenas notificamos o início
            self._send_update_to_queue()

    def _on_step(self) -> bool:
        # Update metrics
        rewards = self.locals.get("rewards", [0.0])
        self._current_reward += float(np.mean(rewards))
        
        dones = self.locals.get("dones", [False])
        if np.any(dones):
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0
            
        # Update diagnostics from logger
        if hasattr(self.model, "logger") and self.model.logger is not None:
            n2v = getattr(self.model.logger, "name_to_value", {})
            
            # Robust mapping for different algorithms
            self.policy_loss = n2v.get("train/policy_loss", n2v.get("train/actor_loss", n2v.get("train/policy_gradient_loss", self.policy_loss)))
            self.value_loss = n2v.get("train/value_loss", n2v.get("train/critic_loss", n2v.get("train/loss", self.value_loss)))
            self.entropy_loss = n2v.get("train/entropy_loss", self.entropy_loss)
            self.learning_rate = n2v.get("train/learning_rate", self.learning_rate)
            
            self.fps = n2v.get("time/fps", self.fps)
            self.iterations = n2v.get("time/iterations", self.iterations)
            self.time_elapsed = n2v.get("time/time_elapsed", self.time_elapsed)
            
            self.ep_len_mean = n2v.get("rollout/ep_len_mean", self.ep_len_mean)
            self.ep_rew_mean_sb3 = n2v.get("rollout/ep_rew_mean", self.ep_rew_mean_sb3)
            
        # Manual perf calculation
        if self.fps == 0 and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self.fps = int(self.num_timesteps / elapsed)

        # Update from info
        infos = self.locals.get("infos", [{}])
        if infos:
            info = infos[0]
            regime_val = info.get("current_regime", info.get("regime", self.regime))
            self.regime = str(regime_val)
            self.regime_conf = info.get("regime_conf", 0.0)
            self.heston = info.get("heston_params", {})
            
            self.portfolio_value = info.get("portfolio_value", info.get("account_value", self.portfolio_value))
            self.portfolio_history.append(self.portfolio_value)
            
            cash = info.get("cash", 0.0)
            if self.portfolio_value > 0:
                self.cash_pct = (cash / self.portfolio_value) * 100
            self.daily_return = info.get("daily_return", 0.0)

        if self.n_calls % self.refresh_rate == 0:
            if not self.headless and self.live:
                self.live.update(self._build_layout())
            elif self.headless and self.queue:
                self._send_update_to_queue()
            
        return True

    def _on_training_end(self):
        if not self.headless and self.live:
            self.live.update(self._build_layout(final=True))
            self.live.stop()
        elif self.headless:
            self._send_update_to_queue(final=True)

    def _send_update_to_queue(self, final=False):
        """Send current state to master process."""
        if not self.queue:
            return
            
        # Objeto de dados serializável
        data = {
            "name": self.name,
            "timesteps": self.num_timesteps,
            "total_timesteps": self.total_timesteps,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy_loss": self.entropy_loss,
            "portfolio_value": self.portfolio_value,
            "cash_pct": self.cash_pct,
            "daily_return": self.daily_return,
            "portfolio_history": list(self.portfolio_history),
            "regime": self.regime,
            "regime_conf": self.regime_conf,
            "heston": self.heston,
            "fps": self.fps,
            "final": final,
            "window_idx": self.window_idx
        }
        self.queue.put(data)

    def _build_layout(self, final=False) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1)
        )
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        layout["left"].split_column(
            Layout(name="metrics", ratio=1),
            Layout(name="rollout", ratio=1)
        )
        layout["right"].split_column(
            Layout(name="physics", ratio=1),
            Layout(name="performance", ratio=1)
        )

        status_style = "bold green" if final else "bold cyan"
        status_text = "COMPLETE" if final else "TRAINING"
        
        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left", ratio=1)
        header_table.add_column(justify="right", ratio=1)
        header_table.add_row(
            f" [bold white]PIPELINE:[/bold white] [magenta]{self.name}[/magenta] [dim]Window: {self.window_idx}[/dim]",
            f" [white]STATUS:[/white] [{status_style}]{status_text}[/{status_style}] "
        )
        layout["header"].update(Panel(header_table, style="blue", box=box.SQUARE))

        # RL Metrics
        m_tbl = Table(show_header=False, box=box.SIMPLE, expand=True)
        m_tbl.add_row("Timesteps", f"{self.num_timesteps:,} / {self.total_timesteps:,}")
        m_tbl.add_row("Policy Loss", f"[bold yellow]{self.policy_loss:.6f}[/bold yellow]")
        m_tbl.add_row("Value Loss", f"{self.value_loss:.6f}")
        m_tbl.add_row("Entropy Loss", f"{self.entropy_loss:.6f}")
        m_tbl.add_row("Learning Rate", f"{self.learning_rate:.2e}")
        layout["metrics"].update(Panel(m_tbl, title="[bold cyan]RL Training", border_style="cyan"))

        # Portfolio Stats
        r_tbl = Table(show_header=False, box=box.SIMPLE, expand=True)
        r_tbl.add_row("Episodes", str(len(self.episode_rewards)))
        if self.episode_rewards:
            last_20 = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
            mean_r = float(np.mean(last_20))
        else:
            mean_r = 0.0
        r_tbl.add_row("Mean Reward (L20)", f"[bold]{mean_r:+.4f}[/bold]")
        r_tbl.add_row("Portfolio Value", f"[bold green]${self.portfolio_value:,.2f}[/bold green]")
        r_tbl.add_row("Cash Position", f"{self.cash_pct:.1f}%")
        r_tbl.add_row("Daily Return", f"{self.daily_return*100:+.4f}%")
        layout["rollout"].update(Panel(r_tbl, title="[bold green]Live Portfolio", border_style="green"))

        # Physics Panel
        p_tbl = Table(show_header=False, box=box.SIMPLE, expand=True)
        p_tbl.add_row("Current Regime", REGIME_DISPLAY.get(self.regime, f"[dim]{self.regime}[/dim]"))
        for k, v in self.heston.items():
            p_tbl.add_row(f"Heston {k}", f"{v:.6f}")
        layout["physics"].update(Panel(p_tbl, title="[bold magenta]Market Physics (PINN)", border_style="magenta"))

        # Performance Panel
        perf_tbl = Table(show_header=False, box=box.SIMPLE, expand=True)
        perf_tbl.add_row("FPS", f"[bold yellow]{self.fps}[/bold yellow]")
        perf_tbl.add_row("Iterations", str(self.iterations))
        if self.fps > 0:
            remaining_steps = max(0, self.total_timesteps - self.num_timesteps)
            eta_secs = remaining_steps / self.fps
            eta_delta = datetime.timedelta(seconds=int(eta_secs))
            perf_tbl.add_row("Time Left (ETA)", f"[cyan]{eta_delta}[/cyan]")
        
        elapsed_delta = datetime.timedelta(seconds=int(time.time() - (self._start_time or time.time())))
        perf_tbl.add_row("Elapsed Time", str(elapsed_delta))
        layout["performance"].update(Panel(perf_tbl, title="[bold yellow]System Perf", border_style="yellow"))

        return layout

class StepAuditCallback(BaseCallback):
    """
    Standardized Audit Log for all DRL pipelines.
    """
    def __init__(self, filename: str, log_dir: str = "results/logs/audit"):
        super().__init__()
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, filename)
        self.initialized = False

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        record = {
            "step": self.num_timesteps,
            "reward": float(self.locals.get("rewards", [0.0])[0]),
            "regime": info.get("current_regime", "N/A"),
            "action": str(self.locals.get("actions", [0.0])[0]),
        }
        # Log normalized physics (used by agent)
        heston_norm = info.get("heston_params", {})
        for k, v in heston_norm.items():
            record[f"pinn_norm_{k}"] = v
            
        # Log raw physics (actual market state)
        heston_raw = info.get("heston_raw", {})
        for k, v in heston_raw.items():
            record[f"pinn_raw_{k}"] = v
            
        # Backward compatibility for existing plots
        if heston_norm:
            for k, v in heston_norm.items():
                record[f"pinn_{k}"] = v
            
        if not self.initialized:
            os.makedirs(self.log_dir, exist_ok=True)
            self.header = list(record.keys())
            with open(self.csv_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.header).writeheader()
            self.initialized = True

        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.header, extrasaction='ignore').writerow(record)
        return True

class TrainingLossAuditCallback(BaseCallback):
    """
    Detailed audit of training loss terms (Policy, Value, Entropy).
    """
    def __init__(self, filename: str, log_dir: str = "results/logs/audit"):
        super().__init__()
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, filename)
        self.initialized = False

    def _on_step(self) -> bool:
        if hasattr(self.model, "logger") and self.model.logger is not None:
            n2v = getattr(self.model.logger, "name_to_value", {})
            p_loss = n2v.get("train/policy_loss", n2v.get("train/actor_loss", 0.0))
            v_loss = n2v.get("train/value_loss", n2v.get("train/critic_loss", n2v.get("train/loss", 0.0)))
            ent_loss = n2v.get("train/entropy_loss", 0.0)
            lr = n2v.get("train/learning_rate", 0.0)
            
            record = {
                "step": self.num_timesteps,
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "entropy_loss": ent_loss,
                "learning_rate": lr,
                "fps": n2v.get("time/fps", 0)
            }
            if not self.initialized:
                os.makedirs(self.log_dir, exist_ok=True)
                self.header = list(record.keys())
                with open(self.csv_path, 'w', newline='') as f:
                    csv.DictWriter(f, fieldnames=self.header).writeheader()
                self.initialized = True
            with open(self.csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=self.header, extrasaction='ignore').writerow(record)
        return True


class ParallelMultiAssetDashboard:
    """
    Master Dashboard (Layout A) for multiple assets in Grid Formation.
    Used by the main process to collect data from all worker queues.
    """
    def __init__(self, assets: List[str]):
        self.assets = [a.upper().strip() for a in assets]
        self.asset_states = {a: {} for a in self.assets}
        self.console = Console()
        self.live = None
        self._start_time = time.time()

    def update_state(self, data: Dict):
        asset_name = str(data.get("name", "UNKNOWN")).upper().strip()
        
        # Determine the target key in asset_states
        target_asset = None
        if asset_name in self.asset_states:
            target_asset = asset_name
        else:
            for a in self.assets:
                if a in asset_name:
                    target_asset = a
                    break
        
        if target_asset:
            # Merge logic: ensure we don't lose data when different agents update
            prev_state = self.asset_states[target_asset]
            
            # 1. Timesteps should be cumulative or max
            data["timesteps"] = max(data.get("timesteps", 0), prev_state.get("timesteps", 0))
            
            # 2. Portfolio Persistence (don't reset to 0 if an agent is just starting)
            if data.get("portfolio_value", 0.0) <= 0.01 and prev_state.get("portfolio_value", 0.0) > 0.01:
                data["portfolio_value"] = prev_state["portfolio_value"]
                data["daily_return"] = prev_state.get("daily_return", 0.0)
                data["portfolio_history"] = prev_state.get("portfolio_history", [])
            elif not data.get("portfolio_history") and prev_state.get("portfolio_history"):
                 data["portfolio_history"] = prev_state["portfolio_history"]
                 
            # 3. Physics Persistence
            if not data.get("heston") and prev_state.get("heston"):
                data["heston"] = prev_state["heston"]
                data["regime"] = prev_state["regime"]
                data["regime_conf"] = prev_state.get("regime_conf", 0.0)
            
            # 4. Status Persistence
            if not data.get("status") and prev_state.get("status"):
                data["status"] = prev_state["status"]

            # 5. Final status should be sticky
            if prev_state.get("final", False):
                data["final"] = True
                
            self.asset_states[target_asset] = data

    def _generate_sparkline(self, values: List[float], width: int = 20) -> str:
        """Generate a Unicode sparkline from values."""
        if not values or len(values) < 2:
            return "[dim]Insufficient data[/]"
        
        # Unicode dynamic blocks
        chars = " ▂▃▄▅▆▇█"
        v_min, v_max = min(values), max(values)
        r = v_max - v_min
        if r == 0:
            return chars[4] * len(values)
        
        # Rescale and map to indices
        indices = [int((v - v_min) / r * (len(chars) - 1)) for v in values]
        spark = "".join(chars[i] for i in indices)
        
        # Color based on trend
        color = "green" if values[-1] >= values[0] else "red"
        return f"[{color}]{spark}[/]"

    def _get_asset_panel(self, asset_name: str) -> Panel:
        state = self.asset_states.get(asset_name, {})
        if not state:
            return Panel(f"[dim]Waiting for {asset_name}...[/dim]", title=f" {asset_name} ", border_style="dim")
        
        final = state.get("final", False)
        worker_status = state.get("status")
        
        if final:
            status = "[bold green]DONE[/]"
        elif worker_status:
            status = f"[bold yellow]{worker_status}[/]"
        else:
            status = f"[cyan]{state.get('timesteps', 0):,} steps[/]"
        
        # 1. Finance Section
        finance_info = (
            f"[bold green]${state.get('portfolio_value', 0):,.0f}[/] | "
            f"[bold]{state.get('daily_return', 0)*100:+.2f}%[/] | "
            f"Cash: {state.get('cash_pct', 0):.1f}%"
        )
        
        # 2. Physics Section
        heston = state.get("heston", {})
        xi = heston.get('xi', 0.0)
        rho = heston.get('rho', 0.0)
        regime_raw = state.get("regime", "UNKNOWN")
        regime_styled = REGIME_DISPLAY.get(regime_raw, f"[dim]{regime_raw}[/]")
        
        # If we have a worker status but no physics data yet, show status in physics line too
        if worker_status and not heston:
            physics_info = f"[yellow]Initializing Physics Engine...[/]"
        else:
            physics_info = (
                f"{regime_styled} | "
                f"ξ:{xi:.3f} ρ:{rho:.2f} | "
                f"Conf: {state.get('regime_conf', 0)*100:.1f}%"
            )
        
        # 3. AI Health Section
        health_info = (
            f"P-Loss: [yellow]{state.get('policy_loss', 0):.4f}[/] | "
            f"Ent: {state.get('entropy_loss', 0):.3f} | "
            f"FPS: {state.get('fps', 0)}"
        )
        
        # Graph (Sparkline)
        history = state.get("portfolio_history", [])
        sparkline = self._generate_sparkline(history)

        # Progress
        prog = (state.get('timesteps', 0) / max(1, state.get('total_timesteps', 1))) * 100
        prog_bar = f"[{'='*int(prog/10)}{' '*(10-int(prog/10))}] {prog:.1f}%"

        # Compilation Table
        table = Table.grid(expand=True)
        table.add_row("[dim]FINANCE :[/]", finance_info)
        table.add_row("[dim]PHYSICS :[/]", physics_info)
        table.add_row("[dim]HEALTH  :[/]", health_info)
        table.add_row("[dim]TREND   :[/]", sparkline)
        table.add_row("[dim]PROGRESS:[/]", f"{prog_bar} [dim](Win {state.get('window_idx', 0)})[/dim]")

        p_style = "green" if final else "blue"
        return Panel(
            table, 
            title=f" [bold white]{asset_name}[/] | {status} ", 
            border_style=p_style,
            box=box.ROUNDED
        )

    def _build_grid_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1)
        )
        
        # Header Master
        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")
        elapsed = datetime.timedelta(seconds=int(time.time() - self._start_time))
        header.add_row(
            f" [bold white]PARALLEL ASSET COMMAND CENTER[/] [dim]Assets: {len(self.assets)}[/]",
            f" [bold yellow]TOTAL ELAPSED: {elapsed}[/] "
        )
        layout["header"].update(Panel(header, style="magenta", box=box.DOUBLE))

        # Dynamic Grid splitting
        n = len(self.assets)
        if n <= 1:
            layout["main"].update(self._get_asset_panel(self.assets[0]))
        elif n <= 2:
            layout["main"].split_row(
                Layout(self._get_asset_panel(self.assets[0])),
                Layout(self._get_asset_panel(self.assets[1]))
            )
        else:
            # 2x2 or similar
            rows = []
            assets_per_row = 2
            for i in range(0, n, assets_per_row):
                row_assets = self.assets[i : i + assets_per_row]
                row_layout = Layout()
                # Use splat unpacking carefully or just add layouts
                row_layouts = [Layout(self._get_asset_panel(a)) for a in row_assets]
                row_layout.split_row(*row_layouts)
                rows.append(row_layout)
            
            layout["main"].split_column(*rows)
            
        return layout

    def start(self):
        self.live = Live(self._build_grid_layout(), console=self.console, refresh_per_second=2, transient=True)
        self.live.start()

    def refresh(self):
        if self.live:
            self.live.update(self._build_grid_layout())

    def stop(self):
        if self.live:
            self.live.update(self._build_grid_layout())
            self.live.stop()
