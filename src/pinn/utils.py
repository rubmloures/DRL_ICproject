# /src/utils.py

import pandas as pd
import os
import logging
from pathlib import Path

# Setup logger
logger = logging.getLogger('PINN_Utils')

def salvar_historico_treinamento(history: dict, results_dir: str = None):
    """
    Salva o histórico de treinamento em CSV de forma segura,
    lidando com listas de tamanhos diferentes.
    
    Args:
        history: Dicionário com histórico de treinamento
        results_dir: Diretório para salvar (opcional, usa ./results/pinn_training/ se não informado)
    """
    try:
        if results_dir is None:
            # Use default results directory
            results_dir = Path(__file__).parent.parent.parent / "results" / "pinn_training"
        else:
            results_dir = Path(results_dir)
        
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / 'training_history.csv'
        
        # Encontra o comprimento máximo entre as listas
        max_len = max(len(v) for v in history.values() if isinstance(v, list))
        
        # Normaliza as listas (preenche com NaN se faltar dado no final)
        history_normalized = {}
        for k, v in history.items():
            if isinstance(v, list):
                if len(v) < max_len:
                    # Estende com None/NaN
                    v = v + [None] * (max_len - len(v))
                history_normalized[k] = v
        
        df = pd.DataFrame(history_normalized)
        df.to_csv(save_path, index=False)
        logger.info(f"Histórico salvo com sucesso em: {save_path}")
        
    except Exception as e:
        logger.error(f"Erro crítico ao salvar histórico: {e}")
        # Tenta salvar backup cru
        try:
            backup_path = str(save_path) + ".bak"
            pd.DataFrame.from_dict(history, orient='index').transpose().to_csv(backup_path)
            logger.info("Backup salvo.")
        except:
            pass

# Adicione aqui sua função send_telegram_message se desejar usá-la
# Exemplo:
# import requests
# def send_telegram_message(message):
#     TOKEN = "SEU_TOKEN_AQUI"
#     CHAT_ID = "SEU_CHAT_ID_AQUI"
#     url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
#     params = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
#     try:
#         response = requests.post(url, data=params)
#     except Exception as e:
#         print(f"Erro ao enviar mensagem para o Telegram: {e}")


