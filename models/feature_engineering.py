"""
Módulo de engenharia de features para o modelo de ML.

Transforma o dataset limpo (books_clean.csv) em matriz de features (X)
e vetor alvo (y) prontos para treino/avaliação dos modelos.

Features utilizadas:
- Numéricas: paginas, ano, rating, female (%), taxa_abandono, razao_desejo,
             tamanho_descricao, sentimento_descricao
- Categóricas: Top-N gêneros (One-Hot), Editora (Label Encoded Top-K)
- Holdout temporal: treino até ANO_CORTE_TREINO, teste após
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("FeatureEngineering")


class FeatureEngineer:
    """
    Transforma o dataset limpo em features para ML.

    Suporta dois modos:
    - Random split (20% teste aleatório)
    - Temporal split (treino ≤ ANO_CORTE_TREINO, teste > ANO_CORTE_TREINO)
    """

    def __init__(self):
        self.feature_names = []
        self.top_generos = []
        self.top_editoras = []
        self.scaler = MinMaxScaler()

    def preparar_features(
        self,
        input_file: str = None,
        temporal_split: bool = True,
        output_file: str = None,
    ) -> tuple:
        """
        Carrega o dataset limpo e prepara X, y para ML.

        Returns:
            (X_train, X_test, y_train, y_test, feature_names, df_train, df_test)
        """
        if input_file is None:
            input_file = config.CLEANED_BOOKS_FILE
        if output_file is None:
            output_file = config.FEATURES_FILE

        logger.info(f"Carregando dataset limpo: {input_file}")
        df = pd.read_csv(input_file, encoding="utf-8")
        logger.info(f"  {len(df):,} livros")

        df = df[df["popularidade_score"].notna()].copy()

        df = self._features_numericas(df)
        df = self._features_genero(df)
        df = self._features_editora(df)
        df = self._feature_sentimento(df)

        feature_cols = self._get_feature_columns(df)
        self.feature_names = feature_cols

        X = df[feature_cols].values.astype(float)
        y = df["popularidade_score"].values.astype(float)

        logger.info(f"  Features preparadas: {len(feature_cols)} colunas × {len(X)} amostras")

        n_nan = np.isnan(X).sum()
        if n_nan > 0:
            logger.warning(f"  {n_nan} NaNs encontrados — substituindo por 0")
            X = np.nan_to_num(X, nan=0.0)

        if temporal_split:
            X_train, X_test, y_train, y_test, df_train, df_test = self._temporal_split(X, y, df)
        else:
            X_train, X_test, y_train, y_test, df_train, df_test = self._random_split(X, y, df)

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_out = df[["titulo", "autor", "ano", "genero_principal", "popularidade_score"] + feature_cols].copy()
            df_out.to_csv(output_file, index=False, encoding="utf-8")
            logger.info(f"  Features salvas: {output_file}")

        logger.info(f"  Treino: {len(X_train):,} amostras | Teste: {len(X_test):,} amostras")
        return (X_train, X_test, y_train, y_test, feature_cols, df_train, df_test)

    # ------------------------------------------------------------------
    # Construção de features individuais
    # ------------------------------------------------------------------

    def _features_numericas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza features numéricas contínuas para [0, 1]."""
        cols_numericas = {
            "paginas":           (0, 5000),
            "ano":               (1900, 2024),
            "rating":            (0, 5),
            "female":            (0, 100),
            "taxa_abandono":     (0, 1),
            "taxa_conclusao":    (0, 1),
            "razao_desejo":      (0, None),
            "tamanho_descricao": (0, None),
        }

        for col, (vmin, vmax) in cols_numericas.items():
            if col not in df.columns:
                df[col] = 0.0
                df[col + "_norm"] = 0.0
                continue

            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].clip(lower=vmin)

            if vmax is not None:
                df[col] = df[col].clip(upper=vmax)
            else:
                p99 = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=p99 if p99 > 0 else 1)

            col_min = df[col].min()
            col_max = df[col].max()
            df[col + "_norm"] = (
                (df[col] - col_min) / (col_max - col_min)
                if col_max > col_min else 0.0
            )

        logger.info("  Features numéricas normalizadas")
        return df

    def _features_genero(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-Hot Encoding dos Top-N gêneros."""
        n = config.TOP_GENEROS_FEATURES
        col_lista = "generos_lista" if "generos_lista" in df.columns else "genero_principal"

        todas = df[col_lista].fillna("").str.split("|").explode()
        todas = todas.str.strip().replace("", np.nan).dropna()
        top = todas.value_counts().head(n).index.tolist()
        self.top_generos = top

        logger.info(f"  Top {n} gêneros para OHE: {top[:5]}...")

        for genero in top:
            col_name = f"genero_{genero.lower().replace(' ', '_').replace('-', '_')}"
            df[col_name] = df[col_lista].fillna("").str.contains(
                re.escape(genero), case=False, regex=True
            ).astype(int)

        return df

    def _features_editora(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label encoding das editoras: top-K + 'Outras'."""
        if "editora" not in df.columns:
            df["editora_id"] = 0
            return df

        k = config.TOP_EDITORAS_FEATURES
        top_ed = df["editora"].value_counts().head(k).index.tolist()
        self.top_editoras = top_ed

        editora_map = {ed: i + 1 for i, ed in enumerate(top_ed)}
        df["editora_id"] = df["editora"].map(editora_map).fillna(0).astype(int)

        logger.info(f"  Top {k} editoras encodadas")
        return df

    def _feature_sentimento(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score de sentimento da descrição (NLP ou fallback inline)."""
        nlp_file = config.NLP_RESULTS_FILE
        if os.path.exists(nlp_file):
            try:
                nlp_df = pd.read_csv(nlp_file, encoding="utf-8")
                if "sentimento_score" in nlp_df.columns and "titulo" in nlp_df.columns:
                    df = df.merge(nlp_df[["titulo", "sentimento_score"]], on="titulo", how="left")
                    df["sentimento_score"] = df["sentimento_score"].fillna(0.0)
                    logger.info("  Score de sentimento NLP integrado")
                    return df
            except Exception as e:
                logger.warning(f"  NLP file error: {e}")

        if "descricao" in df.columns:
            df["sentimento_score"] = df["descricao"].fillna("").apply(self._sentimento_basico)
            logger.info("  Score de sentimento calculado inline (descrição)")
        else:
            df["sentimento_score"] = 0.0

        return df

    @staticmethod
    def _sentimento_basico(texto: str) -> float:
        """Análise de sentimento simplificada sem NLTK."""
        palavras_pos = {
            "bom", "otimo", "excelente", "incrivel", "maravilhoso",
            "lindo", "perfeito", "adorei", "amei", "recomendo",
            "fascinante", "envolvente", "cativante", "emocionante",
            "genial", "imperdivel", "sensacional", "apaixonante",
        }
        palavras_neg = {
            "ruim", "pessimo", "horrivel", "terrivel", "chato",
            "entediante", "decepcionante", "fraco", "confuso",
            "superficial", "previsivel", "mediocre", "frustrante",
        }
        import unicodedata
        def _normalizar(s):
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

        palavras = set(re.findall(r"\b\w+\b", _normalizar(texto.lower())))
        pos = len(palavras & palavras_pos)
        neg = len(palavras & palavras_neg)
        total = pos + neg
        if total == 0:
            return 0.0
        return round((pos - neg) / total, 3)

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Retorna lista das colunas de feature construídas."""
        cols = []

        for base in ["paginas", "ano", "rating", "female", "taxa_abandono",
                     "taxa_conclusao", "razao_desejo", "tamanho_descricao"]:
            norm_col = base + "_norm"
            if norm_col in df.columns:
                cols.append(norm_col)

        genero_cols = [
            c for c in df.columns
            if c.startswith("genero_") and c not in ("genero_principal", "generos_lista")
        ]
        cols.extend(genero_cols)

        if "editora_id" in df.columns:
            cols.append("editora_id")

        if "sentimento_score" in df.columns:
            cols.append("sentimento_score")

        return cols

    def _random_split(self, X, y, df):
        """Split aleatório 80/20."""
        from sklearn.model_selection import train_test_split
        idx_train, idx_test = train_test_split(
            np.arange(len(X)), test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        logger.info("  Split: aleatório 80/20")
        return (
            X[idx_train], X[idx_test],
            y[idx_train], y[idx_test],
            df.iloc[idx_train].reset_index(drop=True),
            df.iloc[idx_test].reset_index(drop=True),
        )

    def _temporal_split(self, X, y, df):
        """Holdout temporal: treino ≤ ANO_CORTE_TREINO, teste > ANO_CORTE_TREINO."""
        corte = config.ANO_CORTE_TREINO
        mask_treino = df["ano"] <= corte
        mask_teste = df["ano"] > corte

        if mask_teste.sum() < 50:
            logger.warning(f"  Poucos dados no teste temporal: {mask_teste.sum()}. Usando split aleatório.")
            return self._random_split(X, y, df)

        logger.info(
            f"  Split temporal: treino ≤ {corte} ({mask_treino.sum():,}) "
            f"| teste >{corte} ({mask_teste.sum():,})"
        )
        return (
            X[mask_treino.values], X[mask_teste.values],
            y[mask_treino.values], y[mask_teste.values],
            df[mask_treino].reset_index(drop=True),
            df[mask_teste].reset_index(drop=True),
        )

    # ------------------------------------------------------------------
    # Previsão de livro hipotético (CORRIGIDA)
    # ------------------------------------------------------------------

    def prever_livro_hipotetico(
        self,
        trainer,
        genero: str,
        paginas: int = 300,
        ano: int = 2024,
        rating: float = 4.0,
        editora: str = "",
        descricao: str = "",
        perc_female: float = 70.0,
        taxa_conclusao: float = 0.55,
        razao_desejo: float = 0.5,
        df_referencia: pd.DataFrame = None,
    ) -> dict:
        """
        Prevê o score de popularidade de um livro hipotético.

        Retorna o score, sua posição no percentil do dataset real,
        classificação calibrada e 3 livros reais com score similar.

        Args:
            trainer:        ModelTrainer já treinado com save_models()
            genero:         Gênero principal exato (ex: "Romance", "Fantasia")
            paginas:        Número de páginas
            ano:            Ano de publicação
            rating:         Nota média esperada (0-5)
            editora:        Nome da editora (deve estar no top-10 para ter efeito)
            descricao:      Sinopse/descrição (usado para sentimento NLP)
            perc_female:    % esperado de leitoras mulheres (0-100)
            taxa_conclusao: Fração que vai concluir o livro (0-1).
                            0.3=nicho difícil, 0.55=típico, 0.8=muito acessível
            razao_desejo:   Razão desejo/leitores (0-1).
                            0.3=pouco antecipado, 0.5=normal, 1.0=muito aguardado
            df_referencia:  DataFrame do dataset real (para cálculo de percentil).
                            Se None, carrega automaticamente de CLEANED_BOOKS_FILE.

        Returns:
            Dict com previsões por modelo + _livros_similares + _score_medio
        """
        row = {}

        # --- Features numéricas (normalizadas manualmente no intervalo do treino) ---
        num_features = {
            "paginas_norm":           (float(paginas),        0.0,  5000.0),
            "ano_norm":               (float(ano),         1900.0,  2024.0),
            "rating_norm":            (float(rating),         0.0,     5.0),
            "female_norm":            (float(perc_female),    0.0,   100.0),
            "taxa_abandono_norm":     (1.0 - taxa_conclusao,  0.0,     1.0),
            "taxa_conclusao_norm":    (float(taxa_conclusao), 0.0,     1.0),
            "razao_desejo_norm":      (float(razao_desejo),   0.0,     1.0),
            "tamanho_descricao_norm": (min(float(len(descricao)), 2000.0), 0.0, 2000.0),
        }
        for col, (val, vmin, vmax) in num_features.items():
            if col in self.feature_names:
                row[col] = float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))

        # --- OHE de gênero: matching EXATO (case-insensitive) ---
        # Bug fix: matching anterior era substring ("Romance" ativava "Romance Policial")
        genero_norm = genero.strip().lower()
        for g in self.top_generos:
            col = f"genero_{g.lower().replace(' ', '_').replace('-', '_')}"
            if col in self.feature_names:
                row[col] = 1.0 if g.strip().lower() == genero_norm else 0.0

        # --- Editora ---
        if "editora_id" in self.feature_names:
            editora_map = {ed: float(i + 1) for i, ed in enumerate(self.top_editoras)}
            row["editora_id"] = editora_map.get(editora, 0.0)

        # --- Sentimento NLP ---
        if "sentimento_score" in self.feature_names:
            row["sentimento_score"] = self._sentimento_basico(descricao)

        # --- Montar vetor X (garante ordem correta e sem NaN) ---
        X_hypo = np.array(
            [row.get(c, 0.0) for c in self.feature_names], dtype=float
        ).reshape(1, -1)

        # --- Carregar distribuição real para percentil ---
        if df_referencia is None:
            try:
                df_referencia = pd.read_csv(config.CLEANED_BOOKS_FILE, encoding="utf-8")
            except Exception:
                df_referencia = None

        scores_reais = None
        if df_referencia is not None and "popularidade_score" in df_referencia.columns:
            scores_reais = df_referencia["popularidade_score"].dropna().values

        # Percentis do dataset real (calibração de thresholds)
        if scores_reais is not None:
            p50 = float(np.percentile(scores_reais, 50))
            p75 = float(np.percentile(scores_reais, 75))
            p90 = float(np.percentile(scores_reais, 90))
            p99 = float(np.percentile(scores_reais, 99))
        else:
            p50, p75, p90, p99 = 0.038, 0.145, 0.311, 0.652

        # --- Prever com cada modelo ---
        resultados = {}
        scores_lista = []

        for nome_modelo in trainer.models:
            score_raw = trainer.predict(nome_modelo, X_hypo)[0]
            score = float(np.clip(score_raw, 0.0, 1.0))
            scores_lista.append(score)

            pct_abaixo = float((scores_reais <= score).mean() * 100) if scores_reais is not None else score * 100

            if score >= p99:
                classif = "Bestseller Absoluto (top 1% do dataset)"
            elif score >= p90:
                classif = "Alta Popularidade (top 10%)"
            elif score >= p75:
                classif = "Acima da Media (top 25%)"
            elif score >= p50:
                classif = "Popularidade Media (acima da mediana)"
            else:
                classif = "Popularidade de Nicho (abaixo da mediana)"

            resultados[nome_modelo] = {
                "popularidade_score": round(score, 4),
                "percentil":          round(pct_abaixo, 1),
                "top_pct":            round(100.0 - pct_abaixo, 1),
                "classificacao":      classif,
            }

        score_medio = float(np.mean(scores_lista))
        resultados["_score_medio"] = round(score_medio, 4)

        # --- Livros reais similares (mesmo gênero, score próximo ao previsto) ---
        livros_similares = []
        if df_referencia is not None and "genero_principal" in df_referencia.columns:
            df_sim = df_referencia[
                df_referencia["genero_principal"].str.lower() == genero_norm
            ].copy()
            if len(df_sim) < 3:
                df_sim = df_referencia.copy()

            df_sim = df_sim[df_sim["popularidade_score"].notna()].copy()
            df_sim["_dist"] = (df_sim["popularidade_score"] - score_medio).abs()

            top_sim = df_sim.nsmallest(3, "_dist")[
                ["titulo", "autor", "ano", "rating", "leram", "popularidade_score"]
            ]
            for _, r in top_sim.iterrows():
                livros_similares.append({
                    "titulo":     str(r["titulo"]),
                    "autor":      str(r.get("autor", "")),
                    "ano":        int(r["ano"]),
                    "rating":     round(float(r["rating"]), 1),
                    "leram":      int(r["leram"]),
                    "score_real": round(float(r["popularidade_score"]), 4),
                })

        resultados["_livros_similares"] = livros_similares
        return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fe = FeatureEngineer()
    resultado = fe.preparar_features()
    print(f"Features: {resultado[4]}")
