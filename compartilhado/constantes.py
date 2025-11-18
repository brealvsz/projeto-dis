import os

# Dimensões da imagem
DIMENSOES_IMAGEM_30X30 = (30, 30)
DIMENSOES_IMAGEM_60X60 = (60, 60)
DIMENSOES_IMAGEM_PADRAO = DIMENSOES_IMAGEM_30X30

# Definindo as dimensões das matrizes H e parâmetros S/N
# Modelo 30x30 pixels
DIMENSOES_H_30X30 = (27904, 900)
S_PARA_GANHO_30X30 = 436 # Amostras do sinal
N_PARA_GANHO_30X30 = 64  # Elementos sensores 
MAX_ITERACOES_30X30 = 10 
TOLERANCIA_30X30 = 1e-4 

# Modelo 60x60 pixels
DIMENSOES_H_60X60 = (50816, 3600)
S_PARA_GANHO_60X60 = 794
N_PARA_GANHO_60X60 = 64
MAX_ITERACOES_60X60 = 10
TOLERANCIA_60X60 = 1e-4 

# Caminhos
PASTA_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_MODELOS_SERVIDOR = os.path.join(PASTA_PROJETO, 'servidor', 'modelos')
PASTA_IMAGENS_RECONSTRUIDAS_SERVIDOR = os.path.join(PASTA_PROJETO, 'servidor', 'imagens_reconstruidas')
PASTA_METADADOS_RECONSTRUCAO = os.path.join(PASTA_PROJETO, 'servidor', 'metadados_reconstrucao')
PASTA_RELATORIOS_CLIENTE = os.path.join(PASTA_PROJETO, 'cliente', 'relatorios')
PASTA_IMAGENS_CLIENTE = os.path.join(PASTA_RELATORIOS_CLIENTE, 'imagens_reconstruidas')
PASTA_DESEMPENHO_CLIENTE = os.path.join(PASTA_RELATORIOS_CLIENTE, 'desempenho_servidor')
PASTA_SINAIS_TESTE_CLIENTE = os.path.join(PASTA_PROJETO, 'cliente', 'sinais_teste')

# Configurações do servidor
PORTA_SERVIDOR = int(os.getenv('PORTA_SERVIDOR', 8000))
HOST_SERVIDOR = os.getenv('HOST_SERVIDOR', '127.0.0.1')
URL_BASE_SERVIDOR = f"http://{HOST_SERVIDOR}:{PORTA_SERVIDOR}"


# Configurações de simulação do cliente
MIN_INTERVALO_ENVIO_SINAIS = 0.5 # segundos
MAX_INTERVALO_ENVIO_SINAIS = 2.0 # segundos
NUM_REQUISICOES_CLIENTE = 6 # Número de imagens a serem enviadas pelo cliente