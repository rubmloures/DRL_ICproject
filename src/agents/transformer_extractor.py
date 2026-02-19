import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Processa a sequência temporal (Lookback Window) usando Self-Attention.
    Inspirado em IC_Project_RL/src/agents/transformer_policy.py
    """
    def __init__(self, observation_space, features_dim=256, n_heads=4, n_layers=2):
        # Assume que o observation space é achatado, precisamos saber o shape original
        # Ex: (30 dias, N features). Se for vetor flat, precisamos fazer reshape.
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0] # Ajustar conforme seu env
        
        # Encoder Linear para projetar features
        self.embedding = nn.Linear(input_dim, 128)
        
        # Transformer Encoder Layer (PyTorch Nativo)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Camada final de compressão
        self.linear_out = nn.Linear(128, features_dim)
        self.activation = nn.Tanh()

    def forward(self, observations):
        # 1. Projeção Linear (Adicionar dimensão de tempo se necessário)
        # Se observations for [Batch, Features], transformamos em [Batch, 1, Features] para atenção
        x = self.embedding(observations).unsqueeze(1) 
        
        # 2. Passar pelo Transformer
        x = self.transformer_encoder(x)
        
        # 3. Flatten e Saída
        x = x.squeeze(1)
        return self.activation(self.linear_out(x))