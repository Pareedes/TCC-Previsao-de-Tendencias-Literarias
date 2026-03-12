"""
Configurações e constantes do projeto TCC.
Previsão de Tendências Literárias com Base em Dados do Skoob.
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
# Configurações do Skoob
# ============================================================
SKOOB_BASE_URL = "https://www.skoob.com.br"
SKOOB_BOOK_URL = SKOOB_BASE_URL + "/livro/{book_id}"
SKOOB_SEARCH_URL = SKOOB_BASE_URL + "/livro/lista"
SKOOB_API_BASE = SKOOB_BASE_URL + "/v1"

# ============================================================
# Configurações de Scraping
# ============================================================
REQUEST_DELAY_MIN = 0.5        # Delay mínimo entre requisições (segundos)
REQUEST_DELAY_MAX = 1.5        # Delay máximo entre requisições (segundos)
REQUEST_TIMEOUT = 20           # Timeout por requisição (segundos)
MAX_RETRIES = 3                # Número de tentativas em caso de falha
BACKOFF_FACTOR = 1.5           # Fator de backoff exponencial
MAX_WORKERS = 5                # Threads simultâneas para scraping concorrente

# Headers HTTP para simular navegador real
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# ============================================================
# Configurações de Coleta
# ============================================================
# Volume alvo de livros a coletar
TARGET_BOOKS_COUNT = 5000  # Conforme requisitado, ideal para o projeto

# Gêneros literários principais para busca direcionada
GENEROS_ALVO = [
    "Romance",
    "Fantasia",
    "Ficção Científica",
    "Terror",
    "Suspense",
    "Mistério",
    "Drama",
    "Aventura",
    "Infantojuvenil",
    "Jovem Adulto",
    "Poesia",
    "Biografia",
    "Autoajuda",
    "História",
    "Humor",
    "Clássicos",
    "Distopia",
    "HQ",
    "Mangá",
    "Religião",
    "Erótico",
    "Policial",
    "Crônica",
    "Conto",
    "Filosofia",
    "Psicologia",
    "Negócios",
]

# ============================================================
# Configurações de NLP
# ============================================================
NLP_LANGUAGE = "portuguese"
MAX_REVIEWS_PER_BOOK = 50  # Limite de resenhas por livro para NLP

# ============================================================
# Configurações dos Modelos de ML
# ============================================================
TEST_SIZE = 0.2            # Proporção do conjunto de teste
RANDOM_STATE = 42          # Seed para reprodutibilidade
CV_FOLDS = 5               # Folds para validação cruzada

# ============================================================
# Arquivos de saída
# ============================================================
RAW_BOOKS_FILE = os.path.join(RAW_DATA_DIR, "books_raw.csv")
RAW_REVIEWS_FILE = os.path.join(RAW_DATA_DIR, "reviews_raw.csv")
CLEANED_BOOKS_FILE = os.path.join(PROCESSED_DATA_DIR, "books_cleaned.csv")
CLEANED_REVIEWS_FILE = os.path.join(PROCESSED_DATA_DIR, "reviews_cleaned.csv")
ENRICHED_BOOKS_FILE = os.path.join(PROCESSED_DATA_DIR, "books_enriched.csv")
GENRE_TRENDS_FILE = os.path.join(PROCESSED_DATA_DIR, "genre_trends.csv")
NLP_RESULTS_FILE = os.path.join(PROCESSED_DATA_DIR, "nlp_sentiment.csv")
