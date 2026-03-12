"""
Módulo de processamento de linguagem natural (NLP) para resenhas.
Análise de sentimento, extração de keywords e temas recorrentes.
"""

import re
import logging
import os
import pandas as pd
import numpy as np

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
    resources = [
        "punkt", "punkt_tab", "stopwords", "vader_lexicon",
        "rslp",  # Stemmer para português
    ]
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                try:
                    nltk.data.find(f"sentiment/{resource}")
                except LookupError:
                    logger.info(f"Baixando recurso NLTK: {resource}")
                    nltk.download(resource, quiet=True)

    _nltk_ready = True


class NLPProcessor:
    """
    Processador de NLP para resenhas de livros do Skoob.

    Funcionalidades:
    - Tokenização e limpeza de texto
    - Remoção de stopwords (português)
    - Análise de sentimento simplificada
    - Extração de palavras-chave mais frequentes
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

    def process_reviews(
        self, input_file: str = None, output_file: str = None
    ) -> pd.DataFrame:
        """
        Processa resenhas com NLP: sentimento, keywords, etc.

        Args:
            input_file: CSV de resenhas limpas
            output_file: CSV com resultados do NLP

        Returns:
            DataFrame com dados de NLP adicionados
        """
        if input_file is None:
            input_file = config.CLEANED_REVIEWS_FILE
        if output_file is None:
            output_file = config.NLP_RESULTS_FILE

        logger.info(f"Processando resenhas com NLP: {input_file}")

        df = pd.read_csv(input_file, encoding="utf-8")

        # Análise de sentimento
        df["sentimento_score"] = df["texto"].fillna("").apply(self.analyze_sentiment)
        df["sentimento_label"] = df["sentimento_score"].apply(
            lambda x: "Positivo" if x > 0.1 else ("Negativo" if x < -0.1 else "Neutro")
        )

        # Extração de keywords
        df["keywords"] = df["texto"].fillna("").apply(self.extract_keywords)

        # Comprimento do texto
        df["texto_comprimento"] = df["texto"].fillna("").str.len()
        df["texto_palavras"] = df["texto"].fillna("").str.split().str.len()

        # Salvar
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(
            f"  NLP concluído: {len(df)} resenhas processadas -> {output_file}"
        )

        return df

    def analyze_sentiment(self, text: str) -> float:
        """
        Análise de sentimento simplificada em português.

        Conta ocorrências de palavras positivas vs negativas.

        Returns:
            Score entre -1.0 (negativo) e 1.0 (positivo)
        """
        if not text:
            return 0.0

        words = set(re.findall(r"\b\w+\b", text.lower()))

        positive_count = len(words & self.POSITIVE_WORDS)
        negative_count = len(words & self.NEGATIVE_WORDS)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        score = (positive_count - negative_count) / total
        return round(score, 3)

    def extract_keywords(self, text: str, top_n: int = 10) -> str:
        """
        Extrai as palavras-chave mais relevantes de um texto.

        Remove stopwords e retorna as palavras mais frequentes.

        Returns:
            String com keywords separadas por vírgula
        """
        if not text:
            return ""

        # Tokenizar
        words = re.findall(r"\b[a-záàâãéèêíïóôõöúçñ]{3,}\b", text.lower())

        # Remover stopwords
        filtered = [w for w in words if w not in self.stopwords_pt]

        # Contar frequência
        from collections import Counter
        freq = Counter(filtered)

        # Top N
        top = freq.most_common(top_n)
        return ", ".join([word for word, count in top])

    def get_sentiment_by_book(self, nlp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrupa sentimento por livro.

        Returns:
            DataFrame com sentimento médio por book_id
        """
        grouped = nlp_df.groupby("book_id").agg(
            sentimento_medio=("sentimento_score", "mean"),
            sentimento_mediana=("sentimento_score", "median"),
            sentimento_std=("sentimento_score", "std"),
            total_resenhas_nlp=("sentimento_score", "count"),
            prop_positivas=("sentimento_label", lambda x: (x == "Positivo").mean()),
            prop_negativas=("sentimento_label", lambda x: (x == "Negativo").mean()),
            media_comprimento=("texto_comprimento", "mean"),
        ).reset_index()

        return grouped


if __name__ == "__main__":
    print("NLPProcessor - Módulo de processamento de linguagem natural")
    print("Use main.py para executar o pipeline completo.")
