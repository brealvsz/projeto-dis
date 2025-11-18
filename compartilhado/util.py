
import numpy as np
import datetime
import os
import json
from PIL import Image
import uuid
import pandas as pd 

from compartilhado.constantes import (
    PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR, PASTA_METADADOS_RECONSTRUCAO,
)

def calculo_fator_reducao(H: np.ndarray) -> float:
    
    return np.linalg.norm(H.T @ H, 2)

def calculo_coeficiente_regularizacao(H: np.ndarray, g: np.ndarray) -> float:
    
    return np.max(np.abs(H.T @ g)) * 0.05

def aplicar_ganho_sinal(g_original_vector: np.ndarray, N_sensores: int, S_amostras: int) -> np.ndarray:
    
    if g_original_vector.size != N_sensores * S_amostras:
        raise ValueError(f"Tamanho do vetor g ({g_original_vector.size}) não corresponde a N*S ({N_sensores * S_amostras}).")
       
    g_matrix_2d = g_original_vector.reshape((N_sensores, S_amostras), order='C') # Assumindo C-order (row-major) no flatten

    ganho_matrix_2d = np.ones_like(g_matrix_2d, dtype=float)
    for c_idx in range(N_sensores): # Loop sobre sensores (colunas da matriz H no contexto de Hf=g)
        for l_idx in range(S_amostras): # Loop sobre amostras (linhas da matriz H no contexto de Hf=g)
            
            ganho_matrix_2d[c_idx, l_idx] = 100 + (1/20) * l_idx * np.sqrt(l_idx)
    
    g_mod_matrix_2d = g_matrix_2d * ganho_matrix_2d
    
    return g_mod_matrix_2d.flatten() # Retorna o vetor g modificado e achatado novamente

def salvar_imagem_e_metadados(
    f_reconstruido: np.ndarray,
    identificacao_usuario: str,
    algoritmo_utilizado: str,
    data_hora_inicio: datetime.datetime,
    data_hora_termino: datetime.datetime,
    dimensoes_imagem: tuple,
    num_iteracoes: int
) -> str:
    
    # 1. Remodelar 'f' para as dimensões da imagem
    f_reshaped = f_reconstruido.reshape(dimensoes_imagem)

    # correção de orientação
    f_ajustado = np.flipud(np.rot90(f_reshaped, k=1))

    # 2. Normalizar para 0-255 e converter para uint8
    min_val = f_ajustado.min()
    max_val = f_ajustado.max()
    
    if max_val == min_val:
        f_normalized = np.zeros_like(f_ajustado, dtype=np.uint8) 
    else:
        f_normalized = ((f_ajustado - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    img = Image.fromarray(f_normalized, mode='L') # 'L' para escala de cinza

    # Garantir que as pastas existam
    os.makedirs(PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR, exist_ok=True)
    os.makedirs(PASTA_METADADOS_RECONSTRUCAO, exist_ok=True)

    # Gerar nomes de arquivo únicos
    id_reconstrucao = str(uuid.uuid4())
    nome_arquivo_imagem = f"imagem_reconstruida_{id_reconstrucao}.png"
    caminho_imagem = os.path.join(PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR, nome_arquivo_imagem)
    img.save(caminho_imagem)

    # 4. Salvar metadados
    metadados = {
        "id_reconstrucao": id_reconstrucao,
        "identificacao_usuario": identificacao_usuario,
        "algoritmo_utilizado": algoritmo_utilizado,
        "data_hora_inicio": data_hora_inicio.isoformat(),
        "data_hora_termino": data_hora_termino.isoformat(),
        "tempo_reconstrucao_ms": (data_hora_termino - data_hora_inicio).total_seconds() * 1000,
        "tamanho_pixels": f"{dimensoes_imagem[0]}x{dimensoes_imagem[1]}",
        "numero_iteracoes": num_iteracoes,
        "caminho_imagem": caminho_imagem
    }
    nome_arquivo_metadados = f"metadados_{id_reconstrucao}.json"
    caminho_metadados = os.path.join(PASTA_METADADOS_RECONSTRUCAO, nome_arquivo_metadados)
    with open(caminho_metadados, 'w') as f:
        json.dump(metadados, f, indent=4)
        
    print(f"Imagem e metadados salvos para reconstrução ID: {id_reconstrucao}")
    return nome_arquivo_imagem, metadados