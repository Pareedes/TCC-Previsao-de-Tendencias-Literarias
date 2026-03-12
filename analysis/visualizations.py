"""
Módulo de visualizações para análise de tendências literárias.
Gera gráficos profissionais com matplotlib e seaborn.
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

# Configuração global de estilo
matplotlib.use("Agg")  # Backend não-interativo para salvar arquivos
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
    """
    Gerador de visualizações para o projeto de tendências literárias.

    Gera gráficos de barras, séries temporais, heatmaps, e mais.
    Todos os gráficos são salvos em data/results/.
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.RESULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all(self, books_file: str = None, trends_file: str = None):
        """Gera todas as visualizações."""
        if books_file is None:
            books_file = config.ENRICHED_BOOKS_FILE
        if trends_file is None:
            trends_file = config.GENRE_TRENDS_FILE

        df = pd.read_csv(books_file, encoding="utf-8")

        logger.info(f"Gerando visualizações para {len(df)} livros")

        self.plot_genre_popularity(df)
        self.plot_rating_distribution(df)
        self.plot_genre_ratings(df)
        self.plot_correlation_heatmap(df)
        self.plot_popularity_vs_rating(df)
        self.plot_macro_genre_distribution(df)
        self.plot_engagement_by_genre(df)
        self.plot_acceptance_vs_popularity(df)

        if os.path.exists(trends_file):
            trends_df = pd.read_csv(trends_file, encoding="utf-8")
            self.plot_genre_trends_bar(trends_df)

        logger.info(f"Todas as visualizações salvas em: {self.output_dir}")

    def plot_genre_popularity(self, df: pd.DataFrame):
        """Gráfico de barras: gêneros mais populares por total de leitores."""
        if "genero_principal" not in df.columns or "total_leitores" not in df.columns:
            return

        genre_pop = df.groupby("genero_principal")["total_leitores"].sum()
        genre_pop = genre_pop.sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(genre_pop))
        genre_pop.plot(kind="barh", ax=ax, color=colors)

        ax.set_title("Top 15 Gêneros Literários por Total de Leitores (Skoob)",
                     fontweight="bold", pad=15)
        ax.set_xlabel("Total de Leitores")
        ax.set_ylabel("Gênero")
        ax.ticklabel_format(style="plain", axis="x")

        # Adicionar valores nas barras
        for i, (val, name) in enumerate(zip(genre_pop.values, genre_pop.index)):
            ax.text(val + genre_pop.max() * 0.01, i, f"{val:,.0f}",
                   va="center", fontsize=9)

        plt.tight_layout()
        self._save("generos_popularidade.png")

    def plot_rating_distribution(self, df: pd.DataFrame):
        """Histograma da distribuição de notas."""
        if "nota_media" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        df["nota_media"].hist(bins=20, ax=ax, color="#4C72B0", edgecolor="white",
                              alpha=0.8)

        mean_val = df["nota_media"].mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                   label=f"Média: {mean_val:.2f}")

        ax.set_title("Distribuição de Notas Médias dos Livros (Skoob)",
                     fontweight="bold", pad=15)
        ax.set_xlabel("Nota Média")
        ax.set_ylabel("Frequência")
        ax.legend()

        plt.tight_layout()
        self._save("distribuicao_notas.png")

    def plot_genre_ratings(self, df: pd.DataFrame):
        """Boxplot de notas por gênero."""
        if "genero_principal" not in df.columns or "nota_media" not in df.columns:
            return

        # Top 12 gêneros por quantidade
        top_genres = df["genero_principal"].value_counts().head(12).index
        df_filtered = df[df["genero_principal"].isin(top_genres)]

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(data=df_filtered, x="genero_principal", y="nota_media",
                   ax=ax, palette="Set2", order=top_genres)

        ax.set_title("Distribuição de Notas por Gênero (Top 12)",
                     fontweight="bold", pad=15)
        ax.set_xlabel("Gênero")
        ax.set_ylabel("Nota Média")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        self._save("notas_por_genero.png")

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Heatmap de correlações entre variáveis numéricas."""
        numeric_cols = [
            "nota_media", "total_avaliacoes", "total_leitores",
            "total_resenhas", "popularidade_score", "engajamento_score",
            "aceitacao_score",
        ]
        available = [c for c in numeric_cols if c in df.columns]

        if len(available) < 2:
            return

        corr = df[available].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                   ax=ax, vmin=-1, vmax=1, center=0,
                   square=True, linewidths=0.5)

        ax.set_title("Correlações entre Variáveis do Dataset",
                     fontweight="bold", pad=15)

        plt.tight_layout()
        self._save("heatmap_correlacoes.png")

    def plot_popularity_vs_rating(self, df: pd.DataFrame):
        """Scatter plot: popularidade vs nota média."""
        if "nota_media" not in df.columns or "popularidade_score" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 7))

        if "genero_principal" in df.columns:
            top_genres = df["genero_principal"].value_counts().head(8).index
            df_plot = df[df["genero_principal"].isin(top_genres)]
            sns.scatterplot(data=df_plot, x="nota_media", y="popularidade_score",
                          hue="genero_principal", alpha=0.6, ax=ax, s=30)
        else:
            ax.scatter(df["nota_media"], df["popularidade_score"], alpha=0.4, s=20)

        ax.set_title("Popularidade vs Nota Média dos Livros",
                     fontweight="bold", pad=15)
        ax.set_xlabel("Nota Média")
        ax.set_ylabel("Score de Popularidade")

        plt.tight_layout()
        self._save("popularidade_vs_nota.png")

    def plot_macro_genre_distribution(self, df: pd.DataFrame):
        """Gráfico de pizza dos macro-gêneros."""
        if "macro_genero" not in df.columns:
            return

        counts = df["macro_genero"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("pastel", len(counts))
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, pctdistance=0.85
        )

        for autotext in autotexts:
            autotext.set_fontsize(9)

        ax.set_title("Distribuição por Macro-Gênero",
                     fontweight="bold", pad=15)

        plt.tight_layout()
        self._save("macro_generos_pizza.png")

    def plot_engagement_by_genre(self, df: pd.DataFrame):
        """Barras: engajamento médio por gênero."""
        if "genero_principal" not in df.columns or "engajamento_score" not in df.columns:
            return

        engagement = df.groupby("genero_principal")["engajamento_score"].mean()
        engagement = engagement.sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = sns.color_palette("coolwarm", len(engagement))
        engagement.plot(kind="barh", ax=ax, color=colors)

        ax.set_title("Engajamento Médio por Gênero (Top 15)",
                     fontweight="bold", pad=15)
        ax.set_xlabel("Score de Engajamento")
        ax.set_ylabel("Gênero")

        plt.tight_layout()
        self._save("engajamento_por_genero.png")

    def plot_acceptance_vs_popularity(self, df: pd.DataFrame):
        """Scatter: aceitação vs popularidade por gênero."""
        if not all(c in df.columns for c in ["aceitacao_score", "popularidade_score", "genero_principal"]):
            return

        genre_stats = df.groupby("genero_principal").agg(
            aceitacao=("aceitacao_score", "mean"),
            popularidade=("popularidade_score", "mean"),
            total=("book_id", "count"),
        ).reset_index()

        # Filtrar gêneros com poucos livros
        genre_stats = genre_stats[genre_stats["total"] >= 5]

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            genre_stats["aceitacao"], genre_stats["popularidade"],
            s=genre_stats["total"] * 3, alpha=0.7, c=range(len(genre_stats)),
            cmap="viridis", edgecolors="white", linewidth=0.5
        )

        for _, row in genre_stats.iterrows():
            ax.annotate(row["genero_principal"],
                       (row["aceitacao"], row["popularidade"]),
                       fontsize=8, ha="center", va="bottom")

        ax.set_title("Aceitação vs Popularidade por Gênero",
                     fontweight="bold", pad=15)
        ax.set_xlabel("Score de Aceitação (% que concluíram)")
        ax.set_ylabel("Score de Popularidade")

        plt.tight_layout()
        self._save("aceitacao_vs_popularidade.png")

    def plot_genre_trends_bar(self, trends_df: pd.DataFrame):
        """Barras comparativas das tendências por gênero."""
        top = trends_df.head(15)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Esquerda: total de leitores
        sns.barplot(data=top, x="total_leitores", y="genero",
                   ax=axes[0], palette="Blues_d")
        axes[0].set_title("Total de Leitores por Gênero", fontweight="bold")
        axes[0].set_xlabel("Total de Leitores")

        # Direita: nota média
        sns.barplot(data=top, x="media_nota", y="genero",
                   ax=axes[1], palette="Greens_d")
        axes[1].set_title("Nota Média por Gênero", fontweight="bold")
        axes[1].set_xlabel("Nota Média")

        plt.suptitle("Tendências Literárias - Análise por Gênero",
                    fontweight="bold", fontsize=16, y=1.02)
        plt.tight_layout()
        self._save("tendencias_generos.png")

    def plot_model_comparison(self, results: dict):
        """Gráfico de comparação entre modelos de ML."""
        models = list(results.keys())
        metrics = ["MAE", "RMSE", "R2"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in models]
            colors = sns.color_palette("Set2", len(models))
            bars = axes[i].bar(models, values, color=colors, edgecolor="white")
            axes[i].set_title(metric, fontweight="bold")
            axes[i].set_ylabel("Valor")

            for bar, val in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle("Comparação de Modelos Preditivos",
                    fontweight="bold", fontsize=14)
        plt.tight_layout()
        self._save("comparacao_modelos.png")

    def plot_predictions_vs_actual(self, y_true, y_pred, model_name: str):
        """Gráfico de predições vs valores reais."""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(y_true, y_pred, alpha=0.5, s=30, color="#4C72B0")

        # Linha ideal (y=x)
        lims = [
            min(min(y_true), min(y_pred)),
            max(max(y_true), max(y_pred)),
        ]
        ax.plot(lims, lims, "r--", alpha=0.8, label="Ideal (y=x)")

        ax.set_title(f"Predição vs Real - {model_name}",
                     fontweight="bold", pad=15)
        ax.set_xlabel("Valor Real")
        ax.set_ylabel("Valor Predito")
        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        self._save(f"predicao_vs_real_{model_name.lower().replace(' ', '_')}.png")

    def _save(self, filename: str):
        """Salva a figura atual."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"  Gráfico salvo: {filename}")


if __name__ == "__main__":
    print("Visualizations - Módulo de geração de gráficos")
    print("Use main.py para executar o pipeline completo.")
