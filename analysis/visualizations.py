"""
Módulo de visualizações para análise de tendências literárias.
Gera gráficos profissionais com matplotlib e seaborn.

Novos gráficos (vs versão anterior):
- plot_genre_timeline: Evolução temporal de gêneros (2000→2020)
- plot_growth_heatmap: Heatmap gênero × ano de popularidade
- plot_genre_growth_bar: Barras de crescimento por gênero
- plot_feature_importance: Importância das features do modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import logging
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("Visualizations")

matplotlib.use("Agg")
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
sns.set_theme(style="whitegrid", palette="husl")


class Visualizations:
    """Gerador de visualizações para o projeto de tendências literárias."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all(self, books_file: str = None, trends_file: str = None):
        """Gera todas as visualizações do pipeline."""
        if books_file is None:
            books_file = config.CLEANED_BOOKS_FILE
        if trends_file is None:
            trends_file = config.GENRE_TRENDS_FILE

        df = pd.read_csv(books_file, encoding="utf-8")
        logger.info(f"Gerando visualizações para {len(df):,} livros")

        # Gráficos do dataset
        self.plot_genre_popularity(df)
        self.plot_rating_distribution(df)
        self.plot_genre_ratings(df)
        self.plot_correlation_heatmap(df)
        self.plot_popularity_vs_rating(df)
        self.plot_readers_by_year(df)

        # Gráficos temporais de gêneros
        if os.path.exists(trends_file):
            df_anual = pd.read_csv(trends_file, encoding="utf-8")
            self.plot_genre_timeline(df_anual)
            self.plot_growth_heatmap(df_anual)

        logger.info(f"Visualizações salvas em: {self.output_dir}")

    # ------------------------------------------------------------------
    # Gráficos do dataset
    # ------------------------------------------------------------------

    def plot_genre_popularity(self, df: pd.DataFrame):
        """Barras horizontais: top 15 gêneros por total de leitores."""
        col_genero = "genero_principal"
        if col_genero not in df.columns:
            return

        genre_pop = df.groupby(col_genero)["leram"].sum()
        genre_pop = genre_pop.sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(genre_pop))
        genre_pop.plot(kind="barh", ax=ax, color=colors)

        ax.set_title("Top 15 Gêneros por Total de Leitores (Skoob)", fontweight="bold", pad=15)
        ax.set_xlabel("Total de Leitores")
        ax.set_ylabel("Gênero")

        for i, (val, name) in enumerate(zip(genre_pop.values, genre_pop.index)):
            ax.text(val + genre_pop.max() * 0.01, i, f"{val:,.0f}", va="center", fontsize=9)

        plt.tight_layout()
        self._save("generos_popularidade.png")

    def plot_rating_distribution(self, df: pd.DataFrame):
        """Histograma da distribuição de ratings."""
        if "rating" not in df.columns:
            return

        df_validos = df[df["rating"] > 0]
        fig, ax = plt.subplots(figsize=(10, 6))
        df_validos["rating"].hist(bins=25, ax=ax, color="#4C72B0", edgecolor="white", alpha=0.85)

        mean_val = df_validos["rating"].mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                   label=f"Média: {mean_val:.2f}")

        ax.set_title("Distribuição de Ratings dos Livros (Skoob)", fontweight="bold", pad=15)
        ax.set_xlabel("Rating (0–5)")
        ax.set_ylabel("Frequência")
        ax.legend()

        plt.tight_layout()
        self._save("distribuicao_notas.png")

    def plot_genre_ratings(self, df: pd.DataFrame):
        """Boxplot de ratings por gênero (top 12)."""
        if "genero_principal" not in df.columns or "rating" not in df.columns:
            return

        top_genres = df["genero_principal"].value_counts().head(12).index
        df_f = df[df["genero_principal"].isin(top_genres) & (df["rating"] > 0)]

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(data=df_f, x="genero_principal", y="rating",
                    ax=ax, palette="Set2", order=top_genres)
        ax.set_title("Ratings por Gênero (Top 12)", fontweight="bold", pad=15)
        ax.set_xlabel("Gênero")
        ax.set_ylabel("Rating")
        plt.xticks(rotation=40, ha="right")

        plt.tight_layout()
        self._save("notas_por_genero.png")

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Heatmap de correlações entre variáveis numéricas relevantes."""
        cols = [c for c in [
            "rating", "avaliacao", "resenha", "leram",
            "querem_ler", "abandonos", "popularidade_score",
            "taxa_conclusao", "taxa_abandono",
        ] if c in df.columns]

        if len(cols) < 2:
            return

        corr = df[cols].corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                    ax=ax, vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
        ax.set_title("Correlações entre Variáveis do Dataset (Skoob)",
                     fontweight="bold", pad=15)

        plt.tight_layout()
        self._save("heatmap_correlacoes.png")

    def plot_popularity_vs_rating(self, df: pd.DataFrame):
        """Scatter: score de popularidade vs rating por gênero."""
        if "rating" not in df.columns or "popularidade_score" not in df.columns:
            return

        df_f = df[df["rating"] > 0].copy()
        top_genres = df_f["genero_principal"].value_counts().head(8).index
        df_plot = df_f[df_f["genero_principal"].isin(top_genres)]

        fig, ax = plt.subplots(figsize=(11, 7))
        sns.scatterplot(data=df_plot, x="rating", y="popularidade_score",
                        hue="genero_principal", alpha=0.55, ax=ax, s=35)
        ax.set_title("Popularidade vs Rating por Gênero", fontweight="bold", pad=15)
        ax.set_xlabel("Rating (0–5)")
        ax.set_ylabel("Score de Popularidade")

        plt.tight_layout()
        self._save("popularidade_vs_nota.png")

    def plot_readers_by_year(self, df: pd.DataFrame):
        """Linha: total de leitores por ano de publicação (2000-2020)."""
        if "ano" not in df.columns or "leram" not in df.columns:
            return

        df_f = df[(df["ano"] >= 2000) & (df["ano"] <= 2020)]
        anual = df_f.groupby("ano").agg(
            total_leitores=("leram", "sum"),
            qtd_livros=("titulo", "count"),
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(13, 6))
        ax2 = ax1.twinx()

        color1, color2 = "#2196F3", "#FF5722"
        ax1.fill_between(anual["ano"], anual["total_leitores"], alpha=0.3, color=color1)
        ax1.plot(anual["ano"], anual["total_leitores"], marker="o", color=color1,
                 linewidth=2, label="Total de Leitores")
        ax2.bar(anual["ano"], anual["qtd_livros"], alpha=0.2, color=color2,
                label="Qtd. Livros")

        ax1.set_title("Leitores e Publicações por Ano (2000–2020)", fontweight="bold", pad=15)
        ax1.set_xlabel("Ano")
        ax1.set_ylabel("Total de Leitores", color=color1)
        ax2.set_ylabel("Nº de Livros Publicados", color=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        self._save("leitores_por_ano.png")

    # ------------------------------------------------------------------
    # Gráficos temporais de gêneros (NOVOS)
    # ------------------------------------------------------------------

    def plot_genre_timeline(self, df_anual: pd.DataFrame, top_n: int = 10):
        """
        Linha de evolução temporal dos top-N gêneros.
        Mostra como a popularidade (leram_total) de cada gênero
        variou ao longo dos anos.
        """
        if "genero" not in df_anual.columns or "leram_total" not in df_anual.columns:
            return

        # Selecionar top-N gêneros por total absoluto
        top_generos = (
            df_anual.groupby("genero")["leram_total"].sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        df_f = df_anual[df_anual["genero"].isin(top_generos)]

        fig, ax = plt.subplots(figsize=(15, 8))
        palette = sns.color_palette("tab10", n_colors=top_n)

        for i, genero in enumerate(top_generos):
            serie = df_f[df_f["genero"] == genero].sort_values("ano")
            ax.plot(serie["ano"], serie["leram_total"],
                    marker="o", markersize=4, linewidth=2,
                    label=genero, color=palette[i])

        ax.set_title(
            f"Evolução Temporal dos Top {top_n} Gêneros (2000–2020)\n"
            "Total de Leitores por Gênero ao Longo dos Anos",
            fontweight="bold", pad=15
        )
        ax.set_xlabel("Ano de Publicação")
        ax.set_ylabel("Total de Leitores (leram)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        ax.set_xlim(df_anual["ano"].min() - 0.5, df_anual["ano"].max() + 0.5)

        plt.tight_layout()
        self._save("evolucao_temporal_generos.png")

    def plot_growth_heatmap(self, df_anual: pd.DataFrame, top_n: int = 15):
        """
        Heatmap: gênero (eixo Y) × ano (eixo X) com popularidade_media como cor.
        Permite ver visualmente quais gêneros cresceram ou declinaram por período.
        """
        if not all(c in df_anual.columns for c in ["genero", "ano", "popularidade_media"]):
            return

        top_generos = (
            df_anual.groupby("genero")["leram_total"].sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        df_f = df_anual[df_anual["genero"].isin(top_generos)].copy()
        pivot = df_f.pivot_table(
            index="genero", columns="ano",
            values="popularidade_media", aggfunc="mean"
        )
        # Selecionar colunas de ano par para não poluir
        anos_disp = [a for a in pivot.columns if a % 2 == 0]
        pivot = pivot[anos_disp]

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
            annot=False, fmt=".2f", cbar_kws={"label": "Popularidade Média"},
        )
        ax.set_title(
            f"Heatmap de Popularidade por Gênero e Ano (Top {top_n})\n"
            "Cores mais quentes = maior popularidade no período",
            fontweight="bold", pad=15
        )
        ax.set_xlabel("Ano")
        ax.set_ylabel("Gênero")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        self._save("heatmap_genero_ano.png")

    def plot_genre_growth_bar(self, df_tendencias: pd.DataFrame, n: int = 15):
        """
        Barras horizontais dos gêneros com maior crescimento absoluto.
        Colorido por tipo de tendência (Ascensão/Declínio/Emergente).
        """
        if "genero" not in df_tendencias.columns:
            return

        # Filtrar top N crescimento positivo e 5 maiores declínios
        ascensao = df_tendencias[
            df_tendencias["crescimento_absoluto"] > 0
        ].head(n)
        declinio = df_tendencias[
            df_tendencias["crescimento_absoluto"] < 0
        ].tail(5)

        df_plot = pd.concat([ascensao, declinio]).sort_values("crescimento_absoluto")

        colors = df_plot["tendencia"].map({
            "Ascensão": "#2ECC71",
            "Emergente": "#3498DB",
            "Estagnação": "#95A5A6",
            "Declínio": "#E74C3C",
        }).fillna("#95A5A6")

        fig, ax = plt.subplots(figsize=(12, 9))
        bars = ax.barh(df_plot["genero"], df_plot["crescimento_absoluto"],
                       color=colors, edgecolor="white", alpha=0.9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        ax.set_title(
            "Crescimento de Leitores por Gênero (2000–2020)\n"
            "Diferença entre período tardio e inicial",
            fontweight="bold", pad=15
        )
        ax.set_xlabel("Crescimento Absoluto (Nº de Leitores)")
        ax.set_ylabel("Gênero")

        # Legenda manual
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ECC71", label="Ascensão"),
            Patch(facecolor="#3498DB", label="Emergente"),
            Patch(facecolor="#95A5A6", label="Estagnação"),
            Patch(facecolor="#E74C3C", label="Declínio"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        self._save("crescimento_generos.png")

    # ------------------------------------------------------------------
    # Gráficos de modelos de ML (mantidos)
    # ------------------------------------------------------------------

    def plot_model_comparison(self, results: dict):
        """Gráfico de comparação entre modelos de ML (MAE, RMSE, R²)."""
        models = list(results.keys())
        metrics = ["MAE", "RMSE", "R2"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in models]
            colors = sns.color_palette("Set2", len(models))
            bars = axes[i].bar(models, values, color=colors, edgecolor="white")
            axes[i].set_title(metric, fontweight="bold")
            axes[i].set_ylabel("Valor")
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=15)

            for bar, val in zip(bars, values):
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                )

        plt.suptitle("Comparação de Modelos Preditivos — Popularidade de Livros",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        self._save("comparacao_modelos.png")

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name: str):
        """Scatter: valores preditos vs reais."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.45, s=25, color="#4C72B0")

        lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
        ax.plot(lims, lims, "r--", alpha=0.8, label="Ideal (y=x)")
        ax.set_title(f"Predição vs Real — {model_name}", fontweight="bold", pad=15)
        ax.set_xlabel("Score Real de Popularidade")
        ax.set_ylabel("Score Predito de Popularidade")
        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        self._save(f"predicao_vs_real_{model_name.lower().replace(' ', '_')}.png")

    def plot_feature_importance(self, importances: dict):
        """
        Barras horizontais de importância de features para todos os modelos
        que suportam feature_importances_ (Árvore de Decisão, Random Forest).
        """
        for model_name, imp_df in importances.items():
            if imp_df is None or len(imp_df) == 0:
                continue

            col_val = "importance" if "importance" in imp_df.columns else "coefficient"
            top = imp_df.nlargest(15, col_val)

            fig, ax = plt.subplots(figsize=(11, 7))
            colors = sns.color_palette("Blues_d", len(top))
            top.plot.barh(x="feature", y=col_val, ax=ax, color=colors, legend=False)
            ax.set_title(
                f"Top 15 Features Mais Importantes — {model_name}",
                fontweight="bold", pad=15
            )
            ax.set_xlabel("Importância")
            ax.set_ylabel("Feature")
            ax.invert_yaxis()

            plt.tight_layout()
            self._save(f"feature_importance_{model_name.lower().replace(' ', '_')}.png")

    def _save(self, filename: str):
        """Salva a figura atual e fecha."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"  Gráfico salvo: {filename}")


if __name__ == "__main__":
    print("Visualizations — Use main.py para executar o pipeline completo.")
