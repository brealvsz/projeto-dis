import os
import datetime
import psutil
import pandas as pd
import numpy as np
import io
import json
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from compartilhado.constantes import (
    PORTA_SERVIDOR, HOST_SERVIDOR, PASTA_MODELOS_SERVIDOR, DIMENSOES_IMAGEM_PADRAO,
    PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR,
    DIMENSOES_H_30X30, S_PARA_GANHO_30X30, N_PARA_GANHO_30X30, MAX_ITERACOES_30X30, TOLERANCIA_30X30,
    DIMENSOES_H_60X60, S_PARA_GANHO_60X60, N_PARA_GANHO_60X60, MAX_ITERACOES_60X60, TOLERANCIA_60X60,
    DIMENSOES_IMAGEM_30X30, DIMENSOES_IMAGEM_60X60
)
from compartilhado.util import (
    aplicar_ganho_sinal, salvar_imagem_e_metadados,
    calculo_fator_reducao, calculo_coeficiente_regularizacao 
)
from servidor.algoritmos.cg_algoritmos import reconstruir_cgne, reconstruir_cgnr

# Crie as pastas se não existirem
os.makedirs(PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR, exist_ok=True)
os.makedirs(PASTA_MODELOS_SERVIDOR, exist_ok=True)


app = FastAPI(
    title="Servidor de Reconstrução de Imagens",
    description="API para reconstrução de imagens usando CGNE/CGNR."
)

class DadosReconstrucao(BaseModel):
    identificacao_usuario: str
    algoritmo_selecionado: str
    modelo_imagem_id: str
    dimensoes_imagem: tuple[int, int]

# Dicionário para armazenar as matrizes H carregadas em memória
MATRIZES_H_CARREGADAS = {}

def carregar_matriz_h(modelo_id: str) -> np.ndarray:
    
    if modelo_id in MATRIZES_H_CARREGADAS:
        print(f"Usando matriz H em cache para modelo {modelo_id}.")
        return MATRIZES_H_CARREGADAS[modelo_id]

    caminho_npy = os.path.join(PASTA_MODELOS_SERVIDOR, f"matriz_h_{modelo_id}.npy")
    
    if not os.path.exists(caminho_npy):
        
        raise FileNotFoundError(
            f"Arquivo da matriz H não encontrado para o modelo '{modelo_id}' em {caminho_npy}. "
            "Por favor, coloque os arquivos .npy das matrizes H na pasta 'servidor/modelos/'."
        )
    
    print(f"Carregando matriz H para modelo {modelo_id} de {caminho_npy}...")
    matriz_h = np.load(caminho_npy)
    MATRIZES_H_CARREGADAS[modelo_id] = matriz_h
    return matriz_h


@app.post("/reconstruir_imagem/")
async def rota_reconstruir_imagem(
    dados_json: str = Form(...),
    arquivo_sinal: UploadFile = File(...)
):
    #Endpoint para receber os dados do sinal

    data_hora_inicio_reconstrucao = datetime.datetime.now()
    
    # 1. Validar e carregar o vetor de sinal 'g' do CSV
    try:
        conteudo_csv = await arquivo_sinal.read()
        df_g = pd.read_csv(io.StringIO(conteudo_csv.decode('utf-8')), header=None)
        vetor_g_original = df_g.values.flatten()
        valor_max_abs = np.max(np.abs(vetor_g_original))
        if valor_max_abs > 100:  # Limite arbitrário ajustável
            print(f"[AVISO] Sinal original possui valor máximo alto ({valor_max_abs:.2e}), normalizando...")
            vetor_g_original = vetor_g_original / valor_max_abs
        g_max = np.max(np.abs(vetor_g_original))
        if g_max > 1e3:  # threshold ajustável 
            print(f"[AVISO] Sinal original possui valor máximo alto ({g_max:.2e}), normalizando...")
            vetor_g_original = vetor_g_original / g_max
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo CSV do sinal: {e}")

    # Desserializar a string JSON para o modelo Pydantic
    try:
        dados = DadosReconstrucao.model_validate_json(dados_json)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro de validação dos dados JSON: {e}. Recebido: {dados_json}")

    # 2. Carregar a matriz H e obter os parâmetros específicos do modelo
    try:
        matriz_H = carregar_matriz_h(dados.modelo_imagem_id)
        
        # Escolher parâmetros específicos com base no modelo_imagem_id
        if dados.modelo_imagem_id == "30x30_modelo1":
            S_usado, N_usado = S_PARA_GANHO_30X30, N_PARA_GANHO_30X30
            dimensoes_esperadas_h_matriz = DIMENSOES_H_30X30
            dimensoes_esperadas_imagem = DIMENSOES_IMAGEM_30X30
            max_iter_algo = MAX_ITERACOES_30X30
            tol_algo = TOLERANCIA_30X30
        elif dados.modelo_imagem_id == "60x60_modelo1":
            S_usado, N_usado = S_PARA_GANHO_60X60, N_PARA_GANHO_60X60
            dimensoes_esperadas_h_matriz = DIMENSOES_H_60X60
            dimensoes_esperadas_imagem = DIMENSOES_IMAGEM_60X60
            max_iter_algo = MAX_ITERACOES_60X60
            tol_algo = TOLERANCIA_60X60
        else:
            raise HTTPException(status_code=400, detail=f"Modelo de imagem '{dados.modelo_imagem_id}' não reconhecido. Verifique os IDs de modelo disponíveis.")

        # Validações de dimensão 
        if matriz_H.shape != dimensoes_esperadas_h_matriz:
            raise HTTPException(status_code=400, detail=f"Dimensões da matriz H carregada ({matriz_H.shape}) não correspondem às esperadas para o modelo '{dados.modelo_imagem_id}' ({dimensoes_esperadas_h_matriz}).")
        
        if matriz_H.shape[0] != vetor_g_original.shape[0]:
             raise HTTPException(status_code=400, detail=f"Incompatibilidade de dimensões: Linhas de H ({matriz_H.shape[0]}) != elementos do vetor g ({vetor_g_original.shape[0]}).")
        
        tamanho_vetor_imagem_esperado = dimensoes_esperadas_imagem[0] * dimensoes_esperadas_imagem[1]
        if matriz_H.shape[1] != tamanho_vetor_imagem_esperado:
            raise HTTPException(status_code=400, detail=f"Incompatibilidade de dimensões: Colunas de H ({matriz_H.shape[1]}) != tamanho vetor imagem ({tamanho_vetor_imagem_esperado}).")
        
        if dados.dimensoes_imagem != dimensoes_esperadas_imagem:
            raise HTTPException(status_code=400, detail=f"Dimensões de imagem solicitadas ({dados.dimensoes_imagem}) não correspondem às esperadas para o modelo '{dados.modelo_imagem_id}' ({dimensoes_esperadas_imagem}).")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar/validar Matriz H e parâmetros: {e}")

    # 3. Aplicar o ganho de sinal
    try:
        vetor_g_com_ganho = aplicar_ganho_sinal(vetor_g_original, N_sensores=N_usado, S_amostras=S_usado)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro ao aplicar ganho de sinal: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado ao aplicar ganho: {e}")

    # 4. Calcular o coeficiente de regularização (lambda)
    
    lambda_bruto = calculo_coeficiente_regularizacao(matriz_H, vetor_g_com_ganho)

    LIMITE_LAMBDA = 1e2 
    lambda_regularizacao = min(lambda_bruto, LIMITE_LAMBDA)

    print(f"Lambda calculado bruto: {lambda_bruto:.2e} | Lambda final usado: {lambda_regularizacao:.2e}")

    # 5. Executar o algoritmo de reconstrução
    imagem_reconstruida_vetor = None
    num_iteracoes_executadas = 0
    try:
        loop = asyncio.get_event_loop()
        if dados.algoritmo_selecionado.upper() == "CGNE":
            imagem_reconstruida_vetor, num_iteracoes_executadas = await loop.run_in_executor(
                None, reconstruir_cgne, vetor_g_com_ganho, matriz_H, lambda_regularizacao, max_iter_algo, tol_algo
            )
        elif dados.algoritmo_selecionado.upper() == "CGNR":
            imagem_reconstruida_vetor, num_iteracoes_executadas = await loop.run_in_executor(
                None, reconstruir_cgnr, vetor_g_com_ganho, matriz_H, lambda_regularizacao, max_iter_algo, tol_algo
            )
        else:
            raise HTTPException(status_code=400, detail="Algoritmo selecionado inválido. Use 'CGNE' ou 'CGNR'.")
    except Exception as e:
        print(f"Erro durante a execução do algoritmo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na execução do algoritmo de reconstrução: {e}")

    data_hora_termino_reconstrucao = datetime.datetime.now()

    # 6. Salvar a imagem e os metadados
    try:
        nome_arquivo_imagem_salva, metadados_completos = salvar_imagem_e_metadados(
            f_reconstruido=imagem_reconstruida_vetor,
            identificacao_usuario=dados.identificacao_usuario,
            algoritmo_utilizado=dados.algoritmo_selecionado,
            data_hora_inicio=data_hora_inicio_reconstrucao,
            data_hora_termino=data_hora_termino_reconstrucao,
            dimensoes_imagem=dados.dimensoes_imagem,
            num_iteracoes=num_iteracoes_executadas
        )
    except Exception as e:
        print(f"Erro ao salvar imagem/metadados: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar resultado da reconstrução: {e}")

    return JSONResponse(content={
        "status": "sucesso",
        "id_reconstrucao": metadados_completos["id_reconstrucao"],
        "mensagem": "Imagem reconstruída com sucesso!",
        "caminho_imagem_servidor": nome_arquivo_imagem_salva,
        "metadados": metadados_completos
    })

@app.get("/status_servidor/")
async def rota_status_servidor():
    
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem_info = psutil.virtual_memory()
    mem_percent = mem_info.percent
    
    return JSONResponse(content={
        "cpu_percent": cpu_percent,
        "memory_percent": mem_percent,
        "timestamp": datetime.datetime.now().isoformat()
    })

# Para rodar o servidor: python -m uvicorn servidor.main_servidor:app --host 0.0.0.0 --port 8000 --reload