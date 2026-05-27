"""
Módulo de análise descritiva dos dados literários.
Gera estatísticas e insights sobre os dados coletados do Skoob.
"""

import pandas as pd
import numpy as np
import logging
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("DescriptiveAnalysis")


class DescriptiveAnalysis:
    """
    Análise descritiva dos dados de livros e tendências literárias.

    Gera:
    - Rankings de gêneros mais populares
    - Distribuições de notas e leitores
    - Correlações entre variáveis
    - Métricas de tendência por gênero
    """

    def run_full_analysis(
        self, books_file: str = None, trends_file: str = None
    ) -> dict:
        """
        Executa a análise descritiva completa e retorna um relatório.

        Returns:
            Dicionário com seções do relatório
        """
        if books_file is None:
            books_file = config.ENRICHED_BOOKS_FILE
        if trends_file is None:
            trends_file = config.GENRE_TRENDS_FILE

        df = pd.read_csv(books_file, encoding="utf-8")
        report = {}

        logger.info(f"Executando análise descritiva com {len(df)} livros")

        # 1. Visão geral do dataset
        report["visao_geral"] = self._overview(df)

        # 2. Análise de gêneros
        report["generos"] = self._genre_analysis(df)

        # 3. Análise de notas
        report["notas"] = self._rating_analysis(df)

        # 4. Análise de popularidade
        report["popularidade"] = self._popularity_analysis(df)

        # 5. Correlações
        report["correlacoes"] = self._correlation_analysis(df)

        # 6. Top livros
        report["top_livros"] = self._top_books(df)

        # Imprimir relatório resumido
        self._print_report(report)

        # Salvar relatório
        report_file = os.path.join(config.RESULTS_DIR, "relatorio_descritivo.txt")
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        self._save_report(report, report_file)

        return report

    def _overview(self, df: pd.DataFrame) -> dict:
        """Visão geral do dataset."""
        overview = {
            "total_livros": len(df),
            "total_autores": df["autor"].nunique() if "autor" in df.columns else 0,
            "total_generos": 0,
            "nota_media_geral": df["nota_media"].mean() if "nota_media" in df.columns else 0,
            "total_leitores_geral": df["total_leitores"].sum() if "total_leitores" in df.columns else 0,
            "total_avaliacoes_geral": df["total_avaliacoes"].sum() if "total_avaliacoes" in df.columns else 0,
        }

        if "generos" in df.columns:
            all_genres = set()
            for genres in df["generos"].dropna():
                for g in genres.split(","):
                    g = g.strip()
                    if g:
                        all_genres.add(g)
            overview["total_generos"] = len(all_genres)

        return overview

    def _genre_analysis(self, df: pd.DataFrame) -> dict:
        """Análise detalhada por gênero."""
        if "genero_principal" not in df.columns:
            return {}

        genre_stats = df.groupby("genero_principal").agg(
            total_livros=("book_id", "count"),
            nota_media=("nota_media", "mean"),
            media_leitores=("total_leitores", "mean"),
            media_avaliacoes=("total_avaliacoes", "mean"),
        ).reset_index().sort_values("total_livros", ascending=False)

        return {
            "top_generos_por_livros": genre_stats.head(15).to_dict("records"),
            "top_generos_por_leitores": genre_stats.sort_values(
                "media_leitores", ascending=False
            ).head(15).to_dict("records"),
            "top_generos_por_nota": genre_stats.sort_values(
                "nota_media", ascending=False
            ).head(15).to_dict("records"),
        }

    def _rating_analysis(self, df: pd.DataFrame) -> dict:
        """Análise de distribuição de notas."""
        if "nota_media" not in df.columns:
            return {}

        notas = df["nota_media"]
        return {
            "media": round(notas.mean(), 2),
            "mediana": round(notas.median(), 2),
            "desvio_padrao": round(notas.std(), 2),
            "minimo": round(notas.min(), 2),
            "maximo": round(notas.max(), 2),
            "distribuicao": {
                "1-2 estrelas": int((notas < 2).sum()),
                "2-3 estrelas": int(((notas >= 2) & (notas < 3)).sum()),
                "3-4 estrelas": int(((notas >= 3) & (notas < 4)).sum()),
                "4-5 estrelas": int((notas >= 4).sum()),
            },
        }

    def _popularity_analysis(self, df: pd.DataFrame) -> dict:
        """Análise de popularidade."""
        if "popularidade_score" not in df.columns:
            return {}

        return {
            "media_score": round(df["popularidade_score"].mean(), 2),
            "mediana_score": round(df["popularidade_score"].median(), 2),
            "top10_populares": df.nlargest(10, "popularidade_score")[
                ["titulo", "autor", "popularidade_score", "genero_principal"]
            ].to_dict("records"),
        }

    def _correlation_analysis(self, df: pd.DataFrame) -> dict:
        """Análise de correlações entre variáveis numéricas."""
        numeric_cols = [
            "nota_media", "total_avaliacoes", "total_leitores",
            "total_resenhas", "popularidade_score", "engajamento_score",
        ]
        available = [c for c in numeric_cols if c in df.columns]

        if len(available) < 2:
            return {}

        corr = df[available].corr()

        # Pares com correlação mais forte
        pairs = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                pairs.append({
                    "var1": available[i],
                    "var2": available[j],
                    "correlacao": round(corr.iloc[i, j], 3),
                })

        pairs.sort(key=lambda x: abs(x["correlacao"]), reverse=True)

        return {
            "maiores_correlacoes": pairs[:10],
            "matrix": corr.round(3).to_dict(),
        }

    def _top_books(self, df: pd.DataFrame) -> dict:
        """Top livros por diferentes critérios."""
        result = {}

        if "nota_media" in df.columns:
            result["melhor_avaliados"] = df.nlargest(10, "nota_media")[
                ["titulo", "autor", "nota_media", "total_avaliacoes"]
            ].to_dict("records")

        if "total_leitores" in df.columns:
            result["mais_lidos"] = df.nlargest(10, "total_leitores")[
                ["titulo", "autor", "total_leitores", "nota_media"]
            ].to_dict("records")

        return result

    def _print_report(self, report: dict):
        """Imprime relatório formatado no console."""
        print("\n" + "=" * 70)
        print("  RELATÓRIO DE ANÁLISE DESCRITIVA")
        print("  Previsão de Tendências Literárias - Skoob")
        print("=" * 70)

        # Visão geral
        ov = report.get("visao_geral", {})
        print(f"\n📊 VISÃO GERAL")
        print(f"  Total de livros: {ov.get('total_livros', 0):,}")
        print(f"  Total de autores: {ov.get('total_autores', 0):,}")
        print(f"  Total de gêneros: {ov.get('total_generos', 0)}")
        print(f"  Nota média geral: {ov.get('nota_media_geral', 0):.2f}")
        print(f"  Total de leitores: {ov.get('total_leitores_geral', 0):,}")

        # Top gêneros
        generos = report.get("generos", {})
        if "top_generos_por_livros" in generos:
            print(f"\n📚 TOP GÊNEROS (por nº de livros)")
            for g in generos["top_generos_por_livros"][:10]:
                print(
                    f"  {g['genero_principal']:20s} | "
                    f"{g['total_livros']:5d} livros | "
                    f"Nota média: {g['nota_media']:.2f}"
                )

        # Notas
        notas = report.get("notas", {})
        if notas:
            print(f"\n⭐ DISTRIBUIÇÃO DE NOTAS")
            print(f"  Média: {notas['media']} | Mediana: {notas['mediana']}")
            for faixa, count in notas.get("distribuicao", {}).items():
                print(f"  {faixa}: {count} livros")

        print("\n" + "=" * 70)

    def _save_report(self, report: dict, filepath: str):
        """Salva relatório em arquivo texto."""
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Relatório salvo em: {filepath}")


if __name__ == "__main__":
    print("DescriptiveAnalysis - Análise descritiva dos dados")
    print("Use main.py para executar o pipeline completo.")
