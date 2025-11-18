import requests
import time
import random
import datetime
import uuid
import numpy as np
import pandas as pd
import io
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

from compartilhado.constantes import (
    URL_BASE_SERVIDOR, MIN_INTERVALO_ENVIO_SINAIS, MAX_INTERVALO_ENVIO_SINAIS,
    NUM_REQUISICOES_CLIENTE, PASTA_RELATORIOS_CLIENTE, PASTA_IMAGENS_CLIENTE,
    PASTA_DESEMPENHO_CLIENTE, PASTA_SINAIS_TESTE_CLIENTE,
    DIMENSOES_H_30X30, DIMENSOES_H_60X60, # Importa as dimensões das matrizes H
    DIMENSOES_IMAGEM_30X30, DIMENSOES_IMAGEM_60X60, # Importa as dimensões das imagens
    PASTA_MODELOS_SERVIDOR,
    PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR
)

# cria as pastas do cliente se não existirem
os.makedirs(PASTA_IMAGENS_CLIENTE, exist_ok=True)
os.makedirs(PASTA_DESEMPENHO_CLIENTE, exist_ok=True)
os.makedirs(PASTA_SINAIS_TESTE_CLIENTE, exist_ok=True)

MAPA_TESTES_VALIDOS = {
    "caso_30x30_1": {
        "modelo_imagem_id": "30x30_modelo1",
        "caminho_csv_sinal": os.path.join(PASTA_SINAIS_TESTE_CLIENTE, "sinal_30x30_caso1.csv"),
        "dimensoes_imagem_esperada": DIMENSOES_IMAGEM_30X30,
        "tamanho_vetor_g_esperado": DIMENSOES_H_30X30[0] # linhas da matriz H para 30x30
    },
    "caso_30x30_2": {
        "modelo_imagem_id": "30x30_modelo1",
        "caminho_csv_sinal": os.path.join(PASTA_SINAIS_TESTE_CLIENTE, "sinal_30x30_caso2.csv"),
        "dimensoes_imagem_esperada": DIMENSOES_IMAGEM_30X30,
        "tamanho_vetor_g_esperado": DIMENSOES_H_30X30[0]
    },
    "caso_60x60_1": {
        "modelo_imagem_id": "60x60_modelo1",
        "caminho_csv_sinal": os.path.join(PASTA_SINAIS_TESTE_CLIENTE, "sinal_60x60_caso1.csv"),
        "dimensoes_imagem_esperada": DIMENSOES_IMAGEM_60X60,
        "tamanho_vetor_g_esperado": DIMENSOES_H_60X60[0] # Linhas da matriz H para 60x60
    },
    "caso_60x60_2": {
        "modelo_imagem_id": "60x60_modelo1",
        "caminho_csv_sinal": os.path.join(PASTA_SINAIS_TESTE_CLIENTE, "sinal_60x60_caso2.csv"),
        "dimensoes_imagem_esperada": DIMENSOES_IMAGEM_60X60,
        "tamanho_vetor_g_esperado": DIMENSOES_H_60X60[0]
    },
    "caso_30x30_3": {
        "modelo_imagem_id": "30x30_modelo1",
        "caminho_csv_sinal": os.path.join(PASTA_SINAIS_TESTE_CLIENTE, "sinal_30x30_caso3.csv"),
        "dimensoes_imagem_esperada": DIMENSOES_IMAGEM_30X30,
        "tamanho_vetor_g_esperado": DIMENSOES_H_30X30[0]
    },
    "caso_60x60_3": {
        "modelo_imagem_id": "60x60_modelo1",
        "caminho_csv_sinal": os.path.join(PASTA_SINAIS_TESTE_CLIENTE, "sinal_60x60_caso3.csv"),
        "dimensoes_imagem_esperada": DIMENSOES_IMAGEM_60X60,
        "tamanho_vetor_g_esperado": DIMENSOES_H_60X60[0]
    }
}


def criar_csv_sinal_exemplo(filepath: str, tamanho_g: int):
    
    print(f"Gerando sinal de {tamanho_g} elementos para {os.path.basename(filepath)}...")
    sinal = np.random.rand(tamanho_g) * 50  # Sinal base

    num_picos = random.randint(5, 15)
    for _ in range(num_picos):
        idx = random.randint(0, tamanho_g - 1)
        sinal[idx] += random.uniform(100, 500)
        if idx > 0: sinal[idx - 1] += random.uniform(50, 200)
        if idx < tamanho_g - 1: sinal[idx + 1] += random.uniform(50, 200)
    
    sinal += np.random.normal(0, 5, tamanho_g) # Adiciona ruído

    pd.DataFrame(sinal).to_csv(filepath, index=False, header=False)
    print(f"Sinal de exemplo salvo em: {filepath}")

# Funções do Cliente

def simular_envio_requisicao():
    
    identificacao_usuario = f"usuario_{uuid.uuid4().hex[:8]}"
    algoritmo_selecionado = random.choice(["CGNE", "CGNR"])
    
    # Seleciona um caso de teste aleatoriamente
    caso_selecionado_key = random.choice(list(MAPA_TESTES_VALIDOS.keys()))
    caso_teste = MAPA_TESTES_VALIDOS[caso_selecionado_key]

    modelo_imagem_id = caso_teste["modelo_imagem_id"]
    caminho_csv_sinal = caso_teste["caminho_csv_sinal"]
    dimensoes_imagem = caso_teste["dimensoes_imagem_esperada"]

    # Verifica se o arquivo CSV do sinal existe
    if not os.path.exists(caminho_csv_sinal):
        print(f"ERRO: Arquivo de sinal CSV não encontrado: {caminho_csv_sinal}")
        print("Por favor, execute o bloco 'if __name__ == \"__main__\":' para criar os arquivos CSV de sinais.")
        return None

    # Ler o conteúdo do CSV
    try:
        with open(caminho_csv_sinal, 'r') as f_csv:
            csv_content = f_csv.read()
        csv_buffer = io.StringIO(csv_content) # Cria um buffer em memória para enviar
        csv_buffer.seek(0) # Retorna ao início do buffer para leitura
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV {caminho_csv_sinal}: {e}")
        return None

    # Dados JSON para o corpo da requisição
    payload_json = {
        "identificacao_usuario": identificacao_usuario,
        "algoritmo_selecionado": algoritmo_selecionado,
        "modelo_imagem_id": modelo_imagem_id,
        "dimensoes_imagem": dimensoes_imagem
    }

    # Arquivos para a requisição multipart/form-data
    files = {'arquivo_sinal': (os.path.basename(caminho_csv_sinal), csv_buffer.getvalue(), 'text/csv')}

    # Enviar a requisição
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Enviando requisição de {identificacao_usuario} para {algoritmo_selecionado} ({dimensoes_imagem[0]}x{dimensoes_imagem[1]}) usando sinal de {os.path.basename(caminho_csv_sinal)}...")
    try:
        response = requests.post(
            f"{URL_BASE_SERVIDOR}/reconstruir_imagem/",
            data={'dados_json': json.dumps(payload_json)},
            files=files,
            timeout=900 
        )
        response.raise_for_status()
        dados_resposta = response.json()
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Resposta recebida: {dados_resposta['status']} para ID {dados_resposta['id_reconstrucao']}")
        
        resultado_para_relatorio = dados_resposta['metadados'].copy()
        resultado_para_relatorio['caminho_imagem_servidor'] = dados_resposta['caminho_imagem_servidor']
        
        return resultado_para_relatorio

    except requests.exceptions.RequestException as e:

        return None

def coletar_desempenho_servidor():
    
    #Coleta dados de desempenho do servidor (CPU e memória).
    
    try:
        response = requests.get(f"{URL_BASE_SERVIDOR}/status_servidor/", timeout=10)
        response.raise_for_status()
        dados_desempenho = response.json()
        return dados_desempenho
    except requests.exceptions.RequestException as e:
        print(f"Erro ao coletar desempenho do servidor: {e}")
        return None

def gerar_relatorio_imagens_reconstruidas(resultados: list):
   
    #Gera um relatório consolidado das imagens reconstruídas.
    
    print("\n--- Gerando Relatório de Imagens Reconstruídas ---")
    if not resultados:
        print("Nenhuma imagem reconstruída para relatar.")
        return

    relatorio_html_path = os.path.join(PASTA_RELATORIOS_CLIENTE, "relatorio_imagens.html")

    html_content = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatorio de Imagens Reconstruidas</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
            .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .imagem-card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            .imagem-card h2 { color: #0056b3; margin-top: 0; }
            /* MUDANÇAS AQUI: Aumentar o tamanho da imagem */
            .imagem-card img {
                max-width: 400px; 
                width: 100%;     
                height: auto;
                display: block;
                margin: 10px auto;
                border: 2px solid #ccc; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.15); 
                image-rendering: optimizeSpeed;
                image-rendering: pixelated; 
            }
            
            .imagem-card p { margin: 5px 0; }
            .label { font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatorio de Imagens Reconstruidas</h1>
    """

    for res in resultados:
        if not isinstance(res, dict) or 'id_reconstrucao' not in res or 'caminho_imagem_servidor' not in res:
            print(f"Aviso: Ignorando resultado inválido na geração do relatório: {res}")
            continue # Pula para o próximo resultado no loop

        # Tenta copiar a imagem do servidor para o relatório do cliente
        
        caminho_imagem_servidor_completo = os.path.join(
            PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR, 
            res['caminho_imagem_servidor']
        )
        caminho_imagem_cliente_local = os.path.join(PASTA_IMAGENS_CLIENTE, res['caminho_imagem_servidor'])
        
        if os.path.exists(caminho_imagem_servidor_completo):
            try:
                Image.open(caminho_imagem_servidor_completo).save(caminho_imagem_cliente_local)
                caminho_para_html = os.path.relpath(caminho_imagem_cliente_local, start=PASTA_RELATORIOS_CLIENTE)
            except Exception as e:
                print(f"Erro ao copiar ou salvar imagem {res['caminho_imagem_servidor']}: {e}")
                caminho_para_html = "caminho/para/imagem/nao_encontrada.png" # Imagem placeholder
        else:
            print(f"Aviso: Imagem {caminho_imagem_servidor_completo} não encontrada no servidor (após sucesso do servidor?). Pode ter sido um problema de salvamento ou caminho.")
            caminho_para_html = "caminho/para/imagem/nao_encontrada.png" # Imagem placeholder


        html_content += f"""
            <div class="imagem-card">
                <h2>Reconstrucao ID: {res['id_reconstrucao']}</h2>
                <p><span class="label">Usuario:</span> {res['identificacao_usuario']}</p>
                <p><span class="label">Algoritmo:</span> {res['algoritmo_utilizado']}</p>
                <p><span class="label">Tamanho:</span> {res['tamanho_pixels']}</p>
                <p><span class="label">Inicio:</span> {res['data_hora_inicio']}</p>
                <p><span class="label">Termino:</span> {res['data_hora_termino']}</p>
                <p><span class="label">Tempo de Reconstrucao:</span> {res['tempo_reconstrucao_ms']:.2f} ms</p>
                <p><span class="label">Iteracoes:</span> {res['numero_iteracoes']}</p>
                <img src="{caminho_para_html}" alt="Imagem Reconstruída ID {res['id_reconstrucao']}">
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """

    with open(relatorio_html_path, 'w') as f:
        f.write(html_content)
    print(f"Relatório de imagens salvo em: {relatorio_html_path}")

def gerar_relatorio_desempenho_servidor(dados_desempenho: list):
    """
    Gera um relatório de desempenho do servidor (CPU e Memória) com gráficos.
    """
    print("\n--- Gerando Relatório de Desempenho do Servidor ---")
    if not dados_desempenho:
        print("Nenhum dado de desempenho coletado.")
        return

    timestamps = [datetime.datetime.fromisoformat(d['timestamp']) for d in dados_desempenho]
    cpu_percents = [d['cpu_percent'] for d in dados_desempenho]
    mem_percents = [d['memory_percent'] for d in dados_desempenho]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(timestamps, cpu_percents, marker='o', linestyle='-', color='b')
    ax1.set_title('Uso de CPU do Servidor')
    ax1.set_ylabel('Uso de CPU (%)')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)

    ax2.plot(timestamps, mem_percents, marker='o', linestyle='-', color='r')
    ax2.set_title('Uso de Memória do Servidor')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Uso de Memória (%)')
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Salvar o gráfico
    os.makedirs(PASTA_DESEMPENHO_CLIENTE, exist_ok=True)
    caminho_grafico = os.path.join(PASTA_DESEMPENHO_CLIENTE, "desempenho_servidor.png")
    plt.savefig(caminho_grafico)
    print(f"Gráfico de desempenho salvo em: {caminho_grafico}")
    plt.close(fig) # Fecha a figura para liberar memória

    relatorio_html_path = os.path.join(PASTA_RELATORIOS_CLIENTE, "relatorio_desempenho.html")
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatorio de Desempenho do Servidor</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatorio de Desempenho do Servidor</h1>
            <img src="{os.path.relpath(caminho_grafico, start=PASTA_RELATORIOS_CLIENTE)}" alt="Gráfico de Desempenho do Servidor">
        </div>
    </body>
    </html>
    """
    with open(relatorio_html_path, 'w') as f:
        f.write(html_content)
    print(f"Relatório de desempenho salvo em: {relatorio_html_path}")


if __name__ == "__main__":
    print("Iniciando simulação do cliente...")
    
    # Passo 1: Gerar os arquivos CSV de sinais de teste se não existirem 

    for key, val in MAPA_TESTES_VALIDOS.items():
        if not os.path.exists(val["caminho_csv_sinal"]):
            criar_csv_sinal_exemplo(val["caminho_csv_sinal"], val["tamanho_vetor_g_esperado"])


    # Passo 3: Iniciar a simulação de envio de requisições 
    resultados_reconstrucao = []
    dados_desempenho_servidor = []

    for i in range(NUM_REQUISICOES_CLIENTE):
        resultado = simular_envio_requisicao()
        if resultado:
            resultados_reconstrucao.append(resultado)
        
        # Coleta dados de desempenho periodicamente (ex: a cada 2 envios)
        if (i + 1) % 2 == 0 or (i + 1) == NUM_REQUISICOES_CLIENTE:
            desempenho = coletar_desempenho_servidor()
            if desempenho:
                dados_desempenho_servidor.append(desempenho)
            
        if i < NUM_REQUISICOES_CLIENTE - 1:
            tempo_espera = random.uniform(MIN_INTERVALO_ENVIO_SINAIS, MAX_INTERVALO_ENVIO_SINAIS)
            print(f"Aguardando {tempo_espera:.2f} segundos antes da próxima requisição...")
            time.sleep(tempo_espera)

    print("\n--- Simulação do cliente concluída ---")
    
    # Passo 4: Gerar os relatórios finais
    # Filtrar resultados bem-sucedidos antes de gerar o relatório de imagens
    resultados_sucesso = [res for res in resultados_reconstrucao if res is not None]
    gerar_relatorio_imagens_reconstruidas(resultados_sucesso)
    gerar_relatorio_desempenho_servidor(dados_desempenho_servidor)