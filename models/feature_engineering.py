"""
Módulo de engenharia de features para os modelos de ML.
Transforma os dados enriquecidos em features prontas para treinamento.
"""

import pandas as pd
import numpy as np
import logging
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("FeatureEngineering")


class FeatureEngineer:
    """
    Engenharia de features para modelos preditivos de tendências literárias.

    Transforma dados de livros/gêneros em features numéricas para ML:
    - Métricas agregadas por gênero
    - Proporções e razões
    - Encodings categóricos
    """

    def create_genre_features(
        self, books_file: str = None, nlp_file: str = None
    ) -> pd.DataFrame:
        """
        Cria um dataset de features por gênero para os modelos preditivos.

        A variável alvo é o 'popularidade_score_medio' de cada gênero.

        Returns:
            DataFrame com features e target por gênero
        """
        if books_file is None:
            books_file = config.ENRICHED_BOOKS_FILE

        logger.info("Criando features por gênero para ML...")

        df = pd.read_csv(books_file, encoding="utf-8")

        # Explodir gêneros
        df["genero_list"] = df["generos"].fillna("").str.split(",")
        df_exploded = df.explode("genero_list")
        df_exploded["genero_list"] = df_exploded["genero_list"].str.strip()
        df_exploded = df_exploded[df_exploded["genero_list"] != ""]

        # Features agregadas por gênero
        features = df_exploded.groupby("genero_list").agg(
            # Contagens
            total_livros=("book_id", "count"),

            # Notas
            nota_media=("nota_media", "mean"),
            nota_mediana=("nota_media", "median"),
            nota_std=("nota_media", "std"),
            nota_min=("nota_media", "min"),
            nota_max=("nota_media", "max"),

            # Popularidade
            popularidade_media=("popularidade_score", "mean"),
            popularidade_mediana=("popularidade_score", "median"),
            popularidade_std=("popularidade_score", "std"),
            popularidade_max=("popularidade_score", "max"),

            # Leitores
            total_leitores_soma=("total_leitores", "sum"),
            leitores_media=("total_leitores", "mean"),
            leitores_mediana=("total_leitores", "median"),

            # Engajamento
            engajamento_medio=("engajamento_score", "mean"),
            engajamento_std=("engajamento_score", "std"),

            # Aceitação
            aceitacao_media=("aceitacao_score", "mean"),

            # Avaliações
            avaliacoes_media=("total_avaliacoes", "mean"),
            avaliacoes_soma=("total_avaliacoes", "sum"),

            # Resenhas
            resenhas_media=("total_resenhas", "mean"),
            resenhas_soma=("total_resenhas", "sum"),
        ).reset_index()

        features = features.rename(columns={"genero_list": "genero"})

        # Features derivadas
        features["leitores_por_livro"] = (
            features["total_leitores_soma"] / features["total_livros"].clip(lower=1)
        ).round(2)

        features["resenhas_por_leitor"] = (
            features["resenhas_soma"] / features["total_leitores_soma"].clip(lower=1)
        ).round(4)

        features["avaliacoes_por_leitor"] = (
            features["avaliacoes_soma"] / features["total_leitores_soma"].clip(lower=1)
        ).round(4)

        features["engagement_ratio"] = (
            (features["resenhas_soma"] + features["avaliacoes_soma"])
            / features["total_leitores_soma"].clip(lower=1)
        ).round(4)

        # Proporção do mercado (market share)
        total_market = features["total_leitores_soma"].sum()
        features["market_share"] = (
            features["total_leitores_soma"] / total_market * 100
        ).round(2) if total_market > 0 else 0

        # Tratar NaN
        features = features.fillna(0)

        # Target: popularidade média do gênero
        features["target_popularidade"] = features["popularidade_media"]

        # Adicionar NLP features se disponíveis
        if nlp_file and os.path.exists(nlp_file):
            features = self._add_nlp_features(features, nlp_file, df)

        logger.info(
            f"  Features criadas: {features.shape[1]} colunas x "
            f"{features.shape[0]} gêneros"
        )

        return features

    def _add_nlp_features(
        self, features: pd.DataFrame, nlp_file: str, books_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Adiciona features de NLP agregadas por gênero."""
        try:
            nlp_df = pd.read_csv(nlp_file, encoding="utf-8")

            # Juntar com books para ter gênero
            nlp_merged = nlp_df.merge(
                books_df[["book_id", "generos"]].drop_duplicates(),
                on="book_id",
                how="left"
            )

            # Explodir gêneros
            nlp_merged["genero_list"] = nlp_merged["generos"].fillna("").str.split(",")
            nlp_exploded = nlp_merged.explode("genero_list")
            nlp_exploded["genero_list"] = nlp_exploded["genero_list"].str.strip()

            # Agregar por gênero
            nlp_features = nlp_exploded.groupby("genero_list").agg(
                sentimento_medio=("sentimento_score", "mean"),
                sentimento_std=("sentimento_score", "std"),
                prop_positivas=("sentimento_label", lambda x: (x == "Positivo").mean()),
                prop_negativas=("sentimento_label", lambda x: (x == "Negativo").mean()),
            ).reset_index()

            nlp_features = nlp_features.rename(columns={"genero_list": "genero"})

            features = features.merge(nlp_features, on="genero", how="left")
            features = features.fillna(0)

            logger.info("  Features de NLP adicionadas")

        except Exception as e:
            logger.warning(f"  Erro ao adicionar features NLP: {e}")

        return features

    def prepare_train_test(
        self, features_df: pd.DataFrame, target_col: str = "target_popularidade"
    ) -> tuple:
        """
        Prepara os dados para treinamento e teste.

        Returns:
            (X, y, feature_names, genre_names) onde X e y são arrays numpy
        """
        from sklearn.model_selection import train_test_split

        # Remover colunas não-feature (evitar Data Leakage)
        drop_cols = [
            "genero", 
            target_col,
            "popularidade_media",
            "popularidade_mediana",
            "popularidade_std",
            "popularidade_max"
        ]
        
        # Garantir que ignoramos colunas que possam não existir
        feature_cols = [c for c in features_df.columns if c not in drop_cols]

        X = features_df[feature_cols].values
        y = features_df[target_col].values
        feature_names = feature_cols
        genre_names = features_df["genero"].tolist()

        logger.info(
            f"  Dataset preparado: {X.shape[0]} amostras, "
            f"{X.shape[1]} features, target='{target_col}'"
        )

        return X, y, feature_names, genre_names


if __name__ == "__main__":
    print("FeatureEngineer - Módulo de engenharia de features")
    print("Use main.py para executar o pipeline completo.")
