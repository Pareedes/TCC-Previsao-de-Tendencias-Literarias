"""
Módulo de análise temporal de tendências por gênero literário.

Analisa a evolução da popularidade de gêneros ao longo dos anos (2000-2020),
identificando gêneros em ascensão, estagnação e declínio.

Este módulo responde à pergunta central do TCC:
"Quais gêneros tiveram crescimento significativo ao longo dos anos?"
"""

import os
import logging
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("GenreTrends")


class GenreTrendAnalyzer:
    """
    Analisador de tendências temporais de gêneros literários.

    Agrupa livros por gênero e ano, calcula métricas de popularidade
    acumulada, e classifica gêneros por ritmo de crescimento.
    """

    def __init__(self):
        self.df_anual = None    # Dados agregados por gênero × ano
        self.df_tendencias = None  # Ranking final de tendências

    def analisar(
        self,
        input_file: str = None,
        output_file: str = None,
        ano_inicio: int = None,
        ano_fim: int = None,
    ) -> pd.DataFrame:
        """
        Executa a análise temporal completa.

        Args:
            input_file: CSV com livros limpos (books_clean.csv)
            output_file: CSV de saída com tendências por gênero e ano
            ano_inicio: Primeiro ano da janela temporal
            ano_fim: Último ano da janela temporal

        Returns:
            DataFrame com tendências calculadas
        """
        if input_file is None:
            input_file = config.CLEANED_BOOKS_FILE
        if output_file is None:
            output_file = config.GENRE_TRENDS_FILE
        if ano_inicio is None:
            ano_inicio = config.ANO_INICIO_ANALISE
        if ano_fim is None:
            ano_fim = config.ANO_FIM_ANALISE

        logger.info(f"Analisando tendências temporais: {ano_inicio}–{ano_fim}")

        df = pd.read_csv(input_file, encoding="utf-8")
        logger.info(f"  {len(df):,} livros carregados")

        # Filtrar janela temporal
        df_janela = df[
            (df["ano"] >= ano_inicio) & (df["ano"] <= ano_fim)
        ].copy()
        logger.info(
            f"  {len(df_janela):,} livros na janela {ano_inicio}–{ano_fim}"
        )

        # Explodir livros com múltiplos gêneros
        df_generos = self._explodir_generos(df_janela)

        # Agregar por gênero e ano
        self.df_anual = self._agregar_por_ano(df_generos)

        # Calcular métricas de crescimento
        self.df_tendencias = self._calcular_crescimento(self.df_anual)

        # Salvar resultado final
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.df_anual.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"  ✓ Tendências salvas: {output_file}")

        self._imprimir_top_tendencias(self.df_tendencias)

        return self.df_tendencias

    def _explodir_generos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explode a coluna generos_lista para ter uma linha por
        combinação livro × gênero.
        """
        col_genero = "generos_lista" if "generos_lista" in df.columns else "genero_principal"

        if col_genero == "generos_lista":
            df = df.copy()
            df[col_genero] = df[col_genero].fillna("").astype(str)
            # Explodir
            df_exp = df.assign(
                genero_tag=df[col_genero].str.split("|")
            ).explode("genero_tag")
            df_exp["genero_tag"] = df_exp["genero_tag"].str.strip()
        else:
            df_exp = df.copy()
            df_exp["genero_tag"] = df_exp["genero_principal"]

        # Remover vazios e gêneros curtos demais
        df_exp = df_exp[df_exp["genero_tag"].str.len() > 2]
        
        # Limpar sujeira (títulos de livros/autores que caíram na coluna de gênero)
        # Um gênero regular não deve ter vírgulas, exclamações, parênteses, etc.
        df_exp = df_exp[
            (df_exp["genero_tag"].str.len() <= 30) &
            (~df_exp["genero_tag"].str.contains(r'[,\!\?\";:\(\)\[\]]', regex=True, na=False)) &
            (df_exp["genero_tag"].str.contains(r'^[a-zA-ZÀ-ÿ]', regex=True, na=False)) &
            (df_exp["genero_tag"].str.count(' ') <= 3)
        ]
        
        # Filtro final: remover gêneros extremamente raros (ruído residual como nomes de autores)
        # Gêneros reais devem aparecer em pelo menos 3 livros diferentes na janela temporal
        contagem = df_exp["genero_tag"].value_counts()
        generos_validos = contagem[contagem >= 3].index
        df_exp = df_exp[df_exp["genero_tag"].isin(generos_validos)]
        
        return df_exp

    def _agregar_por_ano(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega métricas de popularidade por gênero × ano.

        Métricas calculadas:
        - qtd_livros: número de livros publicados
        - leram_total: total de leitores que concluíram
        - rating_medio: nota média dos livros do gênero naquele ano
        - avaliacao_total: total de avaliações registradas
        - popularidade_media: média do score de popularidade
        """
        agg = df.groupby(["genero_tag", "ano"]).agg(
            qtd_livros=("titulo", "count"),
            leram_total=("leram", "sum"),
            rating_medio=("rating", "mean"),
            avaliacao_total=("avaliacao", "sum"),
            popularidade_media=("popularidade_score", "mean"),
        ).reset_index()

        agg.columns = [
            "genero", "ano", "qtd_livros", "leram_total",
            "rating_medio", "avaliacao_total", "popularidade_media",
        ]

        agg["rating_medio"] = agg["rating_medio"].round(2)
        agg["popularidade_media"] = agg["popularidade_media"].round(4)

        logger.info(
            f"  Agregação temporal: {agg['genero'].nunique()} gêneros × "
            f"{agg['ano'].nunique()} anos"
        )
        return agg

    def _calcular_crescimento(self, df_anual: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o crescimento de cada gênero comparando dois períodos:
        - Período Early:  primeiros 5 anos da janela (2000–2004)
        - Período Late:   últimos 5 anos da janela (2016–2020)

        Métricas de crescimento:
        - crescimento_absoluto: diferença de leram_total entre períodos
        - crescimento_percentual: variação % de leram_total
        - tendencia: Ascensão / Estagnação / Declínio
        """
        anos = sorted(df_anual["ano"].unique())
        mid = anos[len(anos) // 2]

        periodo_early = anos[:max(1, len(anos) // 3)]
        periodo_late = anos[-max(1, len(anos) // 3):]

        def _metrica_periodo(df, anos_lista, prefixo):
            subset = df[df["ano"].isin(anos_lista)]
            return subset.groupby("genero").agg(
                **{f"{prefixo}_leram": ("leram_total", "sum")},
                **{f"{prefixo}_rating": ("rating_medio", "mean")},
                **{f"{prefixo}_livros": ("qtd_livros", "sum")},
            ).reset_index()

        early = _metrica_periodo(df_anual, periodo_early, "early")
        late = _metrica_periodo(df_anual, periodo_late, "late")

        merged = early.merge(late, on="genero", how="outer").fillna(0)

        merged["crescimento_absoluto"] = (
            merged["late_leram"] - merged["early_leram"]
        ).astype(int)

        merged["crescimento_percentual"] = merged.apply(
            lambda r: (
                round((r["late_leram"] - r["early_leram"]) / r["early_leram"] * 100, 1)
                if r["early_leram"] > 0
                else 999.0  # Gênero surgiu no período late
            ),
            axis=1,
        )

        # Classificação qualitativa
        def _classificar(row):
            if row["early_leram"] == 0 and row["late_leram"] > 0:
                return "Emergente"
            if row["crescimento_percentual"] >= 50:
                return "Ascensão"
            if row["crescimento_percentual"] <= -20:
                return "Declínio"
            return "Estagnação"

        merged["tendencia"] = merged.apply(_classificar, axis=1)

        # Totais absolutos (para ranking geral)
        total = df_anual.groupby("genero").agg(
            leram_total=("leram_total", "sum"),
            rating_medio=("rating_medio", "mean"),
            total_livros=("qtd_livros", "sum"),
            popularidade_media=("popularidade_media", "mean"),
        ).reset_index()
        total["rating_medio"] = total["rating_medio"].round(2)
        total["popularidade_media"] = total["popularidade_media"].round(4)

        resultado = merged.merge(total, on="genero", how="left")
        resultado = resultado.sort_values("crescimento_absoluto", ascending=False)

        logger.info(
            f"  Crescimento calculado: "
            f"{(resultado['tendencia']=='Ascensão').sum()} em Ascensão, "
            f"{(resultado['tendencia']=='Declínio').sum()} em Declínio, "
            f"{(resultado['tendencia']=='Emergente').sum()} Emergentes"
        )
        return resultado

    def _imprimir_top_tendencias(self, df: pd.DataFrame, n: int = 15):
        """Imprime o ranking de tendências no log."""
        logger.info("\n" + "=" * 60)
        logger.info("  TOP GÊNEROS EM ASCENSÃO (2000→2020)")
        logger.info("=" * 60)

        ascensao = df[df["tendencia"].isin(["Ascensão", "Emergente"])].head(n)
        for _, row in ascensao.iterrows():
            logger.info(
                f"  {row['tendencia']:10} | {row['genero']:<30} | "
                f"+{row['crescimento_absoluto']:,} leitores | "
                f"{row['crescimento_percentual']:+.0f}%"
            )

        logger.info("\n  TOP GÊNEROS EM DECLÍNIO")
        declinio = df[df["tendencia"] == "Declínio"].sort_values(
            "crescimento_absoluto"
        ).head(5)
        for _, row in declinio.iterrows():
            logger.info(
                f"  {row['tendencia']:10} | {row['genero']:<30} | "
                f"{row['crescimento_absoluto']:,} leitores | "
                f"{row['crescimento_percentual']:+.0f}%"
            )

    def get_serie_temporal(self, genero: str) -> pd.DataFrame:
        """
        Retorna a série temporal de um gênero específico.

        Útil para gerar gráfico de linha de um único gênero.
        """
        if self.df_anual is None:
            raise RuntimeError("Execute analisar() primeiro.")

        return self.df_anual[
            self.df_anual["genero"] == genero
        ].sort_values("ano")

    def get_top_generos_por_ano(self, ano: int, n: int = 10) -> pd.DataFrame:
        """Retorna os N gêneros mais populares em um ano específico."""
        if self.df_anual is None:
            raise RuntimeError("Execute analisar() primeiro.")

        return (
            self.df_anual[self.df_anual["ano"] == ano]
            .sort_values("leram_total", ascending=False)
            .head(n)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = GenreTrendAnalyzer()
    df = analyzer.analisar()
    print(f"\nTop 10 em Ascensão:\n{df[df['tendencia']=='Ascensão'].head(10)[['genero','crescimento_percentual','leram_total']].to_string()}")
