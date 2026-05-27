"""
Configurações e constantes do projeto TCC.
Previsão de Tendências Literárias com Base em Dados do Skoob.

NOTA: O projeto utiliza um dataset público (Public Domain) do Skoob
com ~12.000 livros, em conformidade com os Termos de Uso da plataforma.
"""

import os

# ============================================================
# Diretórios do projeto
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Criar diretórios se não existirem
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Fonte de dados — Dataset Público
# ============================================================
# Dataset com licença Public Domain contendo ~12.000 livros do Skoob
DADOS_CSV_PATH = os.path.join(BASE_DIR, "dados.csv")

# Descrição da fonte (para documentação e relatórios)
DADOS_FONTE = "Dataset Público do Skoob — Public Domain License"
DADOS_PERIODO = "Publicações até 2020"
DADOS_VOLUME = "~12.000 livros"

# ============================================================
# Configurações de NLP
# ============================================================
NLP_LANGUAGE = "portuguese"

# ============================================================
# Configurações dos Modelos de ML
# ============================================================
TEST_SIZE = 0.2            # Proporção do conjunto de teste
RANDOM_STATE = 42          # Seed para reprodutibilidade
CV_FOLDS = 5               # Folds para validação cruzada

# Holdout temporal: treinar em livros até ANO_CORTE, testar após
ANO_CORTE_TREINO = 2017    # Treinar com até 2017 inclusive
ANO_INICIO_ANALISE = 2000  # Início da análise temporal de gêneros
ANO_FIM_ANALISE = 2020     # Fim da análise temporal de gêneros

# Fórmula do Score de Popularidade (pesos somam 1.0)
PESO_LERAM = 0.5           # Peso: nº de leitores que concluíram
PESO_AVALIACAO = 0.3       # Peso: nº de avaliações
PESO_RESENHA = 0.2         # Peso: nº de resenhas escritas

# Top N gêneros a usar como features no modelo
TOP_GENEROS_FEATURES = 20  # Cria N colunas dummy de gênero
TOP_EDITORAS_FEATURES = 10 # Cria N categorias de editora

# ============================================================
# Arquivos de saída
# ============================================================
CLEANED_BOOKS_FILE = os.path.join(PROCESSED_DATA_DIR, "books_clean.csv")
GENRE_TRENDS_FILE = os.path.join(PROCESSED_DATA_DIR, "genre_trends_temporal.csv")
NLP_RESULTS_FILE = os.path.join(PROCESSED_DATA_DIR, "nlp_descricoes.csv")
FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, "features_ml.csv")
MODEL_RESULTS_FILE = os.path.join(RESULTS_DIR, "resultados_modelos.csv")
