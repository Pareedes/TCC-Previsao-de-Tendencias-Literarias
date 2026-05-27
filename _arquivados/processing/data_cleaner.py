"""
Módulo de limpeza e tratamento de dados coletados.
Remove duplicatas, trata valores nulos e normaliza campos.
"""

import pandas as pd
import numpy as np
import re
import logging
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("DataCleaner")


class DataCleaner:
    """
    Classe para limpeza e estruturação dos dados brutos do Skoob.

    Opera sobre os CSVs em data/raw/ e gera CSVs limpos em data/processed/.
    """

    # Mapeamento de normalização de gêneros
    GENRE_NORMALIZATION = {
        "sci-fi": "Ficção Científica",
        "ficção científica": "Ficção Científica",
        "ficcao cientifica": "Ficção Científica",
        "science fiction": "Ficção Científica",
        "ficção": "Ficção",
        "fiction": "Ficção",
        "romance": "Romance",
        "romance literário": "Romance",
        "fantasia": "Fantasia",
        "fantasy": "Fantasia",
        "terror": "Terror",
        "horror": "Terror",
        "suspense": "Suspense",
        "thriller": "Suspense",
        "mistério": "Mistério",
        "mystery": "Mistério",
        "aventura": "Aventura",
        "adventure": "Aventura",
        "drama": "Drama",
        "infantojuvenil": "Infantojuvenil",
        "infantil": "Infantojuvenil",
        "jovem adulto": "Jovem Adulto",
        "young adult": "Jovem Adulto",
        "ya": "Jovem Adulto",
        "poesia": "Poesia",
        "poetry": "Poesia",
        "biografia": "Biografia",
        "biography": "Biografia",
        "autoajuda": "Autoajuda",
        "self-help": "Autoajuda",
        "auto-ajuda": "Autoajuda",
        "história": "História",
        "history": "História",
        "humor": "Humor",
        "comédia": "Humor",
        "clássicos": "Clássicos",
        "classics": "Clássicos",
        "distopia": "Distopia",
        "dystopia": "Distopia",
        "hq": "HQ",
        "quadrinhos": "HQ",
        "comics": "HQ",
        "mangá": "Mangá",
        "manga": "Mangá",
        "religião": "Religião",
        "religion": "Religião",
        "policial": "Policial",
        "crime": "Policial",
        "crônica": "Crônica",
        "conto": "Conto",
        "short stories": "Conto",
        "filosofia": "Filosofia",
        "philosophy": "Filosofia",
        "psicologia": "Psicologia",
        "psychology": "Psicologia",
        "negócios": "Negócios",
        "business": "Negócios",
        "erótico": "Erótico",
        "erotica": "Erótico",
    }

    def clean_books(self, input_file: str = None, output_file: str = None) -> pd.DataFrame:
        """
        Limpa e estrutura o CSV de livros.

        Args:
            input_file: Caminho do CSV bruto de livros
            output_file: Caminho do CSV limpo de saída

        Returns:
            DataFrame com os dados limpos
        """
        if input_file is None:
            input_file = config.RAW_BOOKS_FILE
        if output_file is None:
            output_file = config.CLEANED_BOOKS_FILE

        logger.info(f"Limpando dados de livros: {input_file}")

        # Carregar dados
        df = pd.read_csv(input_file, encoding="utf-8")
        initial_count = len(df)
        logger.info(f"  Registros carregados: {initial_count}")

        # 1. Remover duplicatas por book_id
        df = df.drop_duplicates(subset=["book_id"], keep="first")
        logger.info(f"  Após remover duplicatas: {len(df)} (-{initial_count - len(df)})")

        # 2. Remover linhas sem título
        df = df.dropna(subset=["titulo"])
        df = df[df["titulo"].str.strip() != ""]
        logger.info(f"  Após remover sem título: {len(df)}")

        # 3. Converter colunas numéricas
        numeric_cols = [
            "nota_media", "total_avaliacoes", "total_leitores",
            "total_resenhas", "total_edicoes", "num_paginas",
            "leitores_leram", "leitores_lendo", "leitores_quero_ler",
            "leitores_relendo", "leitores_abandonaram",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                if col == "nota_media":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 4. Garantir colunas essenciais
        if "generos" not in df.columns:
            df["generos"] = ""
            
        # 5. Normalizar gêneros
        df["generos"] = df["generos"].fillna("").apply(self._normalize_genres)

        # 6. Limpar campos de texto
        text_cols = ["titulo", "autor", "sinopse", "editora"]
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").str.strip()

        # 7. Extrair ano de publicação
        if "data_publicacao" not in df.columns:
            df["data_publicacao"] = ""
        df["ano_publicacao"] = df["data_publicacao"].apply(self._extract_year)

        # 8. Remover livros com dados insuficientes (sem avaliações)
        if "total_avaliacoes" not in df.columns:
            df["total_avaliacoes"] = 0
            
        min_avaliacoes = 5
        df_filtered = df[df["total_avaliacoes"] >= min_avaliacoes]
        logger.info(
            f"  Livros com >= {min_avaliacoes} avaliações: {len(df_filtered)} "
            f"(removidos: {len(df) - len(df_filtered)})"
        )
        df = df_filtered

        # Salvar
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"  Dados limpos salvos: {output_file} ({len(df)} registros)")

        return df

    def clean_reviews(self, input_file: str = None, output_file: str = None) -> pd.DataFrame:
        """Limpa e estrutura o CSV de resenhas."""
        if input_file is None:
            input_file = config.RAW_REVIEWS_FILE
        if output_file is None:
            output_file = config.CLEANED_REVIEWS_FILE

        logger.info(f"Limpando dados de resenhas: {input_file}")

        df = pd.read_csv(input_file, encoding="utf-8")
        initial_count = len(df)

        # 1. Remover duplicatas
        df = df.drop_duplicates(subset=["book_id", "usuario", "texto"], keep="first")

        # 2. Remover resenhas vazias ou muito curtas
        df["texto"] = df["texto"].fillna("").str.strip()
        df = df[df["texto"].str.len() >= 30]

        # 3. Converter nota para numérico
        df["nota"] = pd.to_numeric(df["nota"], errors="coerce").fillna(0)

        # 4. Limpar texto (remover HTML tags residuais)
        df["texto"] = df["texto"].apply(self._clean_html)

        logger.info(
            f"  Resenhas: {initial_count} -> {len(df)} "
            f"(removidas: {initial_count - len(df)})"
        )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"  Resenhas limpas salvas: {output_file}")

        return df

    def _normalize_genres(self, genres_str: str) -> str:
        """Normaliza nomes de gêneros para um padrão consistente."""
        if not genres_str:
            return ""

        genres = [g.strip() for g in genres_str.split(",")]
        normalized = []

        for genre in genres:
            lower = genre.lower().strip()
            norm = self.GENRE_NORMALIZATION.get(lower, genre.strip().title())
            if norm and norm not in normalized:
                normalized.append(norm)

        return ", ".join(normalized)

    def _extract_year(self, date_str) -> int:
        """Extrai o ano de uma string de data."""
        if pd.isna(date_str) or not str(date_str).strip():
            return 0
        match = re.search(r"(\d{4})", str(date_str))
        return int(match.group(1)) if match else 0

    def _clean_html(self, text: str) -> str:
        """Remove tags HTML residuais de um texto."""
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean


if __name__ == "__main__":
    print("DataCleaner - Módulo de limpeza de dados")
    print("Use main.py para executar a limpeza no pipeline completo.")
