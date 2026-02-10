import sys
import subprocess
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import print as rprint

# Initialize Rich Console
console = Console()

# Default Configuration
DEFAULT_ASSETS = ["PETR4", "VALE3"]

def show_banner():
    """Display the welcome banner."""
    title = Text("DRL Trading Agent Orchestrator", style="bold white on blue", justify="center")
    console.print(Panel(title, style="blue"))
    console.print("[italic]An interactive launcher for the Deep Reinforcement Learning Trading System[/italic]\n", justify="center")

def select_mode() -> str:
    """Select the execution mode."""
    console.print(Panel("Select Execution Mode", style="bold green"))
    console.print("1. [bold cyan]Simple Pipeline[/bold cyan] (Quick Test: Train/Test Split)")
    console.print("2. [bold cyan]Rolling Ensemble[/bold cyan] (Production: Cross-Validation)")
    console.print("3. [bold cyan]Optuna Optimization[/bold cyan] (Hyperparameter Tuning)")
    
    choice = Prompt.ask("Choose an option", choices=["1", "2", "3"], default="2")
    
    mapping = {
        "1": "simple-pipeline",
        "2": "rolling-ensemble",
        "3": "optuna-optimize"
    }
    return mapping[choice]

def select_assets() -> List[str]:
    """Select assets to trade."""
    console.print(Panel("Asset Selection", style="bold yellow"))
    console.print(f"1. Use Defaults: [bold]{', '.join(DEFAULT_ASSETS)}[/bold]")
    console.print("2. Enter Custom Assets")
    
    choice = Prompt.ask("Choose an option", choices=["1", "2"], default="1")
    
    if choice == "2":
        assets_input = Prompt.ask("Enter assets separated by space (e.g., PETR4 ABEV3)")
        return assets_input.split()
    return DEFAULT_ASSETS

def configure_pinn() -> dict:
    """Configure PINN-related settings."""
    config = {}
    console.print(Panel("Physics-Informed Neural Network (PINN) Settings", style="bold magenta"))
    
    # 1. Enable PINN Features
    if Confirm.ask("Enable PINN Features? [dim](Adds Heston volatility params)[/dim]"):
        config["pinn-features"] = True
        
        # 2. Enable Fine-tuning (Only if PINN is enabled)
        if Confirm.ask("Enable PINN Fine-tuning? [dim](Retrain PINN on new data)[/dim]"):
            config["pinn-finetune"] = True
            
        # 3. Enable A/B Testing (Only if PINN is enabled)
        if Confirm.ask("Enable A/B Testing? [dim](Compare Performance With vs Without PINN)[/dim]"):
            config["ab-testing"] = True
            
    return config

def configure_optuna() -> dict:
    """Configure Optuna optimization settings."""
    config = {}
    console.print(Panel("Optuna Optimization Settings", style="bold red"))
    
    agent_type = Prompt.ask("Select Agent Type", choices=["PPO", "DDPG", "A2C"], default="PPO")
    trials = Prompt.ask("Number of Trials", default="20")
    
    config["agent-type"] = agent_type
    config["n-trials"] = trials
    return config

def build_command(mode: str, assets: List[str], pinn_config: dict, optuna_config: dict = None) -> List[str]:
    """Construct the command line arguments."""
    cmd = [sys.executable, "main.py", "--mode", mode]
    
    # Add Assets
    cmd.append("--assets")
    cmd.extend(assets)
    
    # Add PINN flags
    if pinn_config.get("pinn-features"):
        cmd.append("--pinn-features")
    if pinn_config.get("pinn-finetune"):
        cmd.append("--pinn-finetune")
    if pinn_config.get("ab-testing"):
        cmd.append("--ab-testing")
        
    # Add Optuna flags
    if mode == "optuna-optimize" and optuna_config:
        cmd.extend(["--agent-type", optuna_config["agent-type"]])
        cmd.extend(["--n-trials", optuna_config["n-trials"]])
        
    return cmd

def main():
    show_banner()
    
    # 1. Select Mode
    mode = select_mode()
    
    # 2. Select Assets
    assets = select_assets()
    
    # 3. Configure Features based on mode
    pinn_config = {}
    optuna_config = {}
    
    if mode == "rolling-ensemble":
        pinn_config = configure_pinn()
    elif mode == "optuna-optimize":
        optuna_config = configure_optuna()
    else:
        # Simple pipeline can also use PINN features if desired
        if Confirm.ask("Enable PINN features for simple pipeline?"):
            pinn_config = {"pinn-features": True}

    # 4. Build Command
    cmd = build_command(mode, assets, pinn_config, optuna_config)
    
    # 5. Review and Execute
    console.print("\n" + "="*50)
    console.print(Panel(f"[bold green]{' '.join(cmd)}[/bold green]", title="Generated Command"))
    console.print("="*50 + "\n")
    
    if Confirm.ask("Execute this command now?"):
        console.print("[bold green]🚀 Launching...[/bold green]\n")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]❌ Execution failed with code {e.returncode}[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[bold yellow]⚠️ Execution interrupted by user[/bold yellow]")
    else:
        console.print("[yellow]Execution cancelled.[/yellow]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
