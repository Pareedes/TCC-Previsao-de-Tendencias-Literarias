"""
Módulo de enriquecimento de dados.
Adiciona métricas derivadas, categorização hierárquica e agrupamento temporal.
"""

import pandas as pd
import numpy as np
import logging
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("DataEnricher")


class DataEnricher:
    """
    Enriquece os dados limpos com métricas derivadas e categorizações.

    Calcula scores de popularidade, engagement, e gera dados agregados por
    gênero e período temporal para alimentar os modelos preditivos.
    """

    # Categorização hierárquica de gêneros
    MACRO_GENRES = {
        "Ficção": ["Romance", "Ficção", "Ficção Científica", "Fantasia", "Terror",
                    "Suspense", "Mistério", "Drama", "Aventura", "Distopia",
                    "Policial", "Erótico"],
        "Jovem/Infantil": ["Infantojuvenil", "Jovem Adulto"],
        "Não-Ficção": ["Biografia", "Autoajuda", "História", "Filosofia",
                       "Psicologia", "Negócios", "Religião"],
        "Poesia/Crônica": ["Poesia", "Crônica", "Conto"],
        "Visual": ["HQ", "Mangá"],
        "Humor": ["Humor"],
        "Clássicos": ["Clássicos"],
    }

    def enrich_books(self, input_file: str = None, output_file: str = None) -> pd.DataFrame:
        """
        Enriquece os dados de livros com métricas derivadas.

        Args:
            input_file: CSV limpo de entrada
            output_file: CSV enriquecido de saída

        Returns:
            DataFrame enriquecido
        """
        if input_file is None:
            input_file = config.CLEANED_BOOKS_FILE
        if output_file is None:
            output_file = config.ENRICHED_BOOKS_FILE

        logger.info(f"Enriquecendo dados de livros: {input_file}")

        df = pd.read_csv(input_file, encoding="utf-8")

        # 1. Score de Popularidade (combinação ponderada)
        df["popularidade_score"] = self._calculate_popularity_score(df)

        # 2. Score de Engajamento
        df["engajamento_score"] = self._calculate_engagement_score(df)

        # 3. Score de Aceitação (relação lidos vs abandonados)
        df["aceitacao_score"] = self._calculate_acceptance_score(df)

        # 4. Classificar macro-gênero
        df["macro_genero"] = df["generos"].fillna("").apply(self._classify_macro_genre)

        # 5. Gênero principal (primeiro da lista)
        df["genero_principal"] = df["generos"].fillna("").apply(
            lambda x: x.split(",")[0].strip() if x else "Desconhecido"
        )

        # 6. Década de publicação
        if "ano_publicacao" in df.columns:
            df["decada_publicacao"] = df["ano_publicacao"].apply(
                lambda x: (x // 10) * 10 if x > 1800 else 0
            )

        # 7. Faixa de popularidade
        df["faixa_popularidade"] = pd.qcut(
            df["popularidade_score"].clip(lower=0.01),
            q=5,
            labels=["Baixa", "Média-Baixa", "Média", "Média-Alta", "Alta"],
            duplicates="drop"
        )

        # 8. Contagem de gêneros por livro
        df["qtd_generos"] = df["generos"].fillna("").apply(
            lambda x: len([g for g in x.split(",") if g.strip()]) if x else 0
        )

        # Salvar
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"  Dados enriquecidos salvos: {output_file} ({len(df)} registros)")

        return df

    def generate_genre_trends(
        self, books_df: pd.DataFrame, output_file: str = None
    ) -> pd.DataFrame:
        """
        Gera dados de tendências por gênero para modelagem preditiva.

        Agrega dados por gênero e calcula métricas-chave de tendência.

        Returns:
            DataFrame com métricas por gênero
        """
        if output_file is None:
            output_file = config.GENRE_TRENDS_FILE

        logger.info("Gerando dados de tendências por gênero...")

        # Explodir gêneros (um livro pode ter vários)
        df_exploded = books_df.copy()
        df_exploded["genero_list"] = df_exploded["generos"].fillna("").str.split(",")
        df_exploded = df_exploded.explode("genero_list")
        df_exploded["genero_list"] = df_exploded["genero_list"].str.strip()
        df_exploded = df_exploded[df_exploded["genero_list"] != ""]

        # Agregar por gênero
        trends = df_exploded.groupby("genero_list").agg(
            total_livros=("book_id", "count"),
            media_nota=("nota_media", "mean"),
            mediana_nota=("nota_media", "median"),
            std_nota=("nota_media", "std"),
            total_leitores=("total_leitores", "sum"),
            media_leitores_por_livro=("total_leitores", "mean"),
            total_avaliacoes=("total_avaliacoes", "sum"),
            media_avaliacoes_por_livro=("total_avaliacoes", "mean"),
            total_resenhas=("total_resenhas", "sum"),
            media_resenhas_por_livro=("total_resenhas", "mean"),
            media_popularidade=("popularidade_score", "mean"),
            media_engajamento=("engajamento_score", "mean"),
            media_aceitacao=("aceitacao_score", "mean"),
        ).reset_index()

        trends.columns.name = None
        trends = trends.rename(columns={"genero_list": "genero"})

        # Ordenar por popularidade total
        trends = trends.sort_values("total_leitores", ascending=False)

        # Adicionar ranking
        trends["ranking_popularidade"] = range(1, len(trends) + 1)

        # Proporção do total
        total_all = trends["total_leitores"].sum()
        if total_all > 0:
            trends["porcentagem_leitores"] = (
                trends["total_leitores"] / total_all * 100
            ).round(2)

        # Salvar
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        trends.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"  Tendências por gênero salvas: {output_file} ({len(trends)} gêneros)")

        return trends

    def _calculate_popularity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula score de popularidade combinando múltiplas métricas.

        Fórmula: 0.4*leitores_norm + 0.3*quero_ler_norm + 0.2*avaliacoes_norm + 0.1*resenhas_norm
        """
        cols_weights = {
            "total_leitores": 0.4,
            "leitores_quero_ler": 0.3,
            "total_avaliacoes": 0.2,
            "total_resenhas": 0.1,
        }

        score = pd.Series(0.0, index=df.index)

        for col, weight in cols_weights.items():
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").fillna(0)
                max_val = values.max()
                if max_val > 0:
                    normalized = values / max_val
                    score += normalized * weight

        return (score * 100).round(2)

    def _calculate_engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula score de engajamento (interação/leitor).

        Métrica: (resenhas + avaliações) / total_leitores
        """
        avaliacoes = pd.to_numeric(df.get("total_avaliacoes", 0), errors="coerce").fillna(0)
        resenhas = pd.to_numeric(df.get("total_resenhas", 0), errors="coerce").fillna(0)
        leitores = pd.to_numeric(df.get("total_leitores", 0), errors="coerce").fillna(1)
        leitores = leitores.replace(0, 1)  # Evitar divisão por zero

        engagement = ((resenhas + avaliacoes) / leitores * 100).round(2)
        return engagement

    def _calculate_acceptance_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula score de aceitação (quem terminou vs abandonou).

        Métrica: leram / (leram + abandonaram) * 100
        """
        leram = pd.to_numeric(df.get("leitores_leram", 0), errors="coerce").fillna(0)
        abandonaram = pd.to_numeric(
            df.get("leitores_abandonaram", 0), errors="coerce"
        ).fillna(0)

        total = leram + abandonaram
        total = total.replace(0, 1)  # Evitar divisão por zero

        return ((leram / total) * 100).round(2)

    def _classify_macro_genre(self, genres_str: str) -> str:
        """Classifica o macro-gênero baseado nos gêneros do livro."""
        if not genres_str:
            return "Desconhecido"

        genres = [g.strip() for g in genres_str.split(",")]

        for macro, sub_genres in self.MACRO_GENRES.items():
            for genre in genres:
                if genre in sub_genres:
                    return macro

        return "Outros"


if __name__ == "__main__":
    print("DataEnricher - Módulo de enriquecimento de dados")
    print("Use main.py para executar o pipeline completo.")
