import json
import os
from pathlib import Path

# Try to locate the file safely
nb_path = Path('notebooks/validation_dashboard.ipynb')
if not nb_path.exists():
    nb_path = Path(r'd:\UERJ\Programacao_e_Codigos\DRL_ICproject\notebooks\validation_dashboard.ipynb')

if not nb_path.exists():
    print(f"Error: Notebook not found at {nb_path}")
    exit(1)

print(f"Reading {nb_path}")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. SETUP CELL FIX Content (Cells are lists of strings in JSON)
setup_source = [
    "# Configuração de Caminhos e Imports\n",
    "\n",
    "import sys\n",
    "import os\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import json\n",
    "import glob\n",
    "\n",
    "# Ajuste dinâmico do PROJECT_ROOT\n",
    "current_path = Path(os.getcwd())\n",
    "if current_path.name == 'notebooks':\n",
    "    PROJECT_ROOT = current_path.parent\n",
    "else:\n",
    "    PROJECT_ROOT = current_path\n",
    "\n",
    "# Adiciona src ao path se não estiver lá\n",
    "if str(PROJECT_ROOT) not in sys.path:\n",
    "    sys.path.append(str(PROJECT_ROOT))\n",
    "\n",
    "# Imports do Projeto\n",
    "try:\n",
    "    # Tenta importar stable_baselines3 para verificar ambiente\n",
    "    import stable_baselines3\n",
    "    from src.evaluation.results_manager import ResultsManager\n",
    "    from config.config import RESULTS, TRAINED_MODELS\n",
    "    print(\"Ambiente configurado com sucesso.\")\n",
    "    print(f\"Raiz do Projeto: {PROJECT_ROOT}\")\n",
    "except ImportError as e:\n",
    "    print(f\"Erro de Importação: {e}\")\n",
    "    print(\"Dica: Verifique se o ambiente virtual está ativo e se 'stable-baselines3' está instalado.\")\n",
    "    print(\"Verifique também se a pasta 'src' está acessível a partir da raiz.\")\n",
    "\n",
    "# Configuração Visual\n",
    "sns.set_theme(style=\"darkgrid\")\n",
    "plt.rcParams['figure.figsize'] = (14, 6)\n"
]

count = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = "".join(cell['source'])
    
    # Fix 1: Setup
    if "# Configuração de Caminhos e Imports" in src:
        cell['source'] = setup_source
        count += 1
        print("Fixed Setup Cell")
        
    # Fix 2: PINN Path
    # Identifying the cell by unique PINN import or comment
    if "Validação de Física do PINN" in src or "from src.pinn.model import DeepHestonHybrid" in src:
        new_source = []
        for line in cell['source']:
            # Replace the relative path open with PROJECT_ROOT based path
            if "with open('src/pinn/weights/data_stats.json')" in line or 'with open("src/pinn/weights/data_stats.json")' in line:
                new_source.append("        # Instancia modelo (Hiperparâmetros devem bater com o config)\n")
                new_source.append("        stats_path = PROJECT_ROOT / 'src' / 'pinn' / 'weights' / 'data_stats.json'\n")
                new_source.append("        if not stats_path.exists():\n")
                new_source.append("            print(f\"Arquivo de estatísticas não encontrado: {stats_path}\")\n")
                new_source.append("            return\n")
                new_source.append("        with open(stats_path) as f:\n")
            else:
                new_source.append(line)
        
        # Only update if changed
        if new_source != cell['source']:
            cell['source'] = new_source
            count += 1
            print("Fixed PINN Path Cell")

    # Fix 3: Monte Carlo Logic
    if "class MonteCarloStressTest" in src and "def load_and_validate_data" in src:
        # Replacement block for return calculation
        new_logic = [
            "        # Extrair retornos\n",
            "        if 'ensemble_total_return' in df.columns:\n",
            "            # CORREÇÃO: Tratar cada janela como um período de performance independente\n",
            "            # Converter retorno total da janela em retorno diário geométrico médio\n",
            "            \n",
            "            # Tentar calcular dias reais se colunas de data existirem\n",
            "            days_col = 60 # Fallback padrão\n",
            "            if 'end_date' in df.columns and 'start_date' in df.columns:\n",
            "                try:\n",
            "                    df['days'] = (pd.to_datetime(df['end_date']) - pd.to_datetime(df['start_date'])).dt.days\n",
            "                    days_col = df['days'].clip(lower=1)\n",
            "                except:\n",
            "                    pass\n",
            "            \n",
            "            # (1 + R_total) = (1 + r_daily) ^ days\n",
            "            returns = (1 + df['ensemble_total_return'])**(1/days_col) - 1\n",
            "            returns = returns.dropna()\n",
            "            \n",
            "        elif 'daily_returns' in df.columns:\n",
            "            returns = df['daily_returns'].dropna()\n",
            "        else:\n",
            "            # Tentar identificar colunas numéricas genéricas\n",
            "            numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
            "            if len(numeric_cols) > 0:\n",
            "                returns = df[numeric_cols[0]].pct_change().dropna()\n",
            "            else:\n",
            "                raise ValueError(\"Não foi possível identificar colunas de retorno\")\n"
        ]
        
        # Strategy: find the block start and end in the current source list
        start_marker = "if 'ensemble_total_return' in df.columns:"
        end_marker = "raise ValueError(\"Não foi possível identificar colunas de retorno\")"
        
        start_idx = -1
        end_idx = -1
        
        current_lines = cell['source']
        for i, line in enumerate(current_lines):
            if start_marker in line:
                start_idx = i
            if end_marker in line:
                end_idx = i
                break
        
        if start_idx != -1 and end_idx != -1:
            # We have identified the block to replace
            # We assume indentation of start_marker is correct for the new block (which has indentation)
            # However, json list strings usually include \n and indentation. 
            # The new_logic list above includes \n but indentation matches the snippet context.
            
            # Need to ensure we don't break indentation of surrounding code.
            # The start_marker line usually has indentation "        ".
            
            # Let's verify indentation of the matched line
            # indent = current_lines[start_idx].split("if")[0]
            # Assumed "        " based on python structure inside method
            
            cell['source'] = current_lines[:start_idx] + new_logic + current_lines[end_idx+1:]
            count += 1
            print("Fixed Monte Carlo Logic")

print(f"Applying {count} fixes...")
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook saved successfully.")
