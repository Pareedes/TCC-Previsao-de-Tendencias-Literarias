"""
Módulo de processamento de linguagem natural (NLP).

Utilizado para análise de sentimento e extração de palavras-chave
das descrições/sinopses dos livros no dataset público do Skoob.

Após a reestruturação do projeto (pivot de scraping → CSV),
este módulo opera sobre a coluna 'descricao' do dataset,
não mais sobre resenhas individuais de usuários.
"""

import re
import logging
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("NLPProcessor")

# Flag para controlar downloads do NLTK
_nltk_ready = False


def _setup_nltk():
    """Configura e baixa recursos do NLTK necessários."""
    global _nltk_ready
    if _nltk_ready:
        return

    import nltk
    resources = ["punkt", "punkt_tab", "stopwords"]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info(f"Baixando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)

    _nltk_ready = True


class NLPProcessor:
    """
    Processador de NLP para descrições de livros do Skoob.

    Funcionalidades:
    - Análise de sentimento das sinopses (escala -1.0 a 1.0)
    - Extração de palavras-chave mais frequentes (com remoção de stopwords PT-BR)
    """

    # Palavras de sentimento positivo em português
    POSITIVE_WORDS = {
        "ótimo", "excelente", "maravilhoso", "incrível", "fantástico",
        "lindo", "perfeito", "adorei", "amei", "recomendo", "bom",
        "interessante", "divertido", "emocionante", "surpreendente",
        "envolvente", "cativante", "genial", "magnífico", "espetacular",
        "imperdível", "sensacional", "apaixonante", "fascinante",
        "memorável", "inspirador", "tocante", "encantador",
    }

    # Palavras de sentimento negativo em português
    NEGATIVE_WORDS = {
        "ruim", "péssimo", "horrível", "terrível", "chato", "tedioso",
        "entediante", "decepcionante", "fraco", "desinteressante",
        "cansativo", "monótono", "confuso", "superficial", "previsível",
        "detestei", "odiei", "abandonei", "desisti", "frustrante",
        "medíocre", "repetitivo", "desnecessário", "forçado", "artificial",
    }

    def __init__(self):
        self.stopwords_pt = set()
        self._initialize()

    def _initialize(self):
        """Inicializa recursos de NLP."""
        try:
            _setup_nltk()
            from nltk.corpus import stopwords
            self.stopwords_pt = set(stopwords.words("portuguese"))
            logger.info(f"NLTK inicializado com {len(self.stopwords_pt)} stopwords PT-BR")
        except Exception as e:
            logger.warning(f"Erro ao inicializar NLTK: {e}. Usando stopwords mínimas.")
            self.stopwords_pt = {
                "a", "o", "e", "é", "de", "do", "da", "em", "um", "uma", "para",
                "com", "não", "que", "os", "as", "dos", "das", "no", "na", "nos",
                "nas", "por", "mais", "mas", "como", "se", "seu", "sua", "este",
                "esse", "esta", "essa", "são", "foi", "ser", "ter", "que", "muito",
                "me", "te", "lhe", "eu", "ele", "ela", "nós", "eles", "elas",
            }

    def analyze_sentiment(self, text: str) -> float:
        """
        Análise de sentimento de uma descrição de livro.

        Conta ocorrências de palavras positivas vs negativas
        num léxico curado em português.

        Returns:
            Score entre -1.0 (muito negativo) e 1.0 (muito positivo).
            0.0 quando não há palavras de sentimento reconhecidas.
        """
        if not text:
            return 0.0

        words = set(re.findall(r"\b\w+\b", text.lower()))

        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return round((positive_count - negative_count) / total, 3)

    def extract_keywords(self, text: str, top_n: int = 10) -> str:
        """
        Extrai as palavras-chave mais relevantes de uma descrição.

        Pipeline:
        1. Tokenização (regex, apenas palavras com 3+ letras)
        2. Remoção de stopwords PT-BR (NLTK ou fallback manual)
        3. Contagem de frequência
        4. Retorna os top_n termos mais frequentes

        Returns:
            String com keywords separadas por vírgula.
        """
        if not text:
            return ""

        # Tokenizar — apenas palavras com 3+ letras, incluindo acentos
        words = re.findall(r"\b[a-záàâãéèêíïóôõöúçñ]{3,}\b", text.lower())

        # Remover stopwords
        filtered = [w for w in words if w not in self.stopwords_pt]

        # Contar frequência e retornar top N
        from collections import Counter
        freq = Counter(filtered)
        top = freq.most_common(top_n)
        return ", ".join([word for word, _ in top])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nlp = NLPProcessor()
    sample = "Um livro incrível e emocionante sobre a vida e o amor, muito envolvente."
    print(f"Sentimento: {nlp.analyze_sentiment(sample)}")
    print(f"Keywords:   {nlp.extract_keywords(sample)}")
