"""
Módulo de carregamento e limpeza do dataset público do Skoob.

Substitui o web scraper como fonte de dados do projeto.
Dataset: ~12.000 livros com licença Public Domain.

Responsabilidades:
- Carregar dados.csv
- Limpeza e normalização
- Cálculo do Score de Popularidade (variável alvo do ML)
- Normalização de gêneros
"""

import os
import re
import logging
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("DataLoader")


class DataLoader:
    """
    Carregador e limpador do dataset público do Skoob.

    Gera o arquivo books_clean.csv pronto para uso no pipeline de ML
    e análise de tendências temporais.
    """

    def load_and_clean(
        self,
        input_file: str = None,
        output_file: str = None,
    ) -> pd.DataFrame:
        """
        Carrega o CSV bruto, realiza limpeza completa e calcula o score
        de popularidade (variável alvo do modelo de ML).

        Args:
            input_file: Caminho do CSV bruto (dados.csv)
            output_file: Caminho do CSV limpo de saída

        Returns:
            DataFrame limpo e enriquecido
        """
        if input_file is None:
            input_file = config.DADOS_CSV_PATH
        if output_file is None:
            output_file = config.CLEANED_BOOKS_FILE

        logger.info(f"Carregando dataset: {input_file}")
        df = pd.read_csv(input_file, encoding="utf-8")
        logger.info(f"  {len(df):,} registros carregados | {df.shape[1]} colunas")

        df = self._remover_duplicatas(df)
        df = self._limpar_colunas_basicas(df)
        df = self._filtrar_anos(df)
        df = self._normalizar_generos(df)
        df = self._calcular_popularidade_score(df)
        df = self._calcular_features_auxiliares(df)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")

        logger.info(f"  ✓ Dataset limpo: {len(df):,} livros → {output_file}")
        logger.info(
            f"  Score de popularidade: min={df['popularidade_score'].min():.3f} "
            f"max={df['popularidade_score'].max():.3f} "
            f"média={df['popularidade_score'].mean():.3f}"
        )

        return df

    # ------------------------------------------------------------------
    # Etapas privadas de limpeza
    # ------------------------------------------------------------------

    def _remover_duplicatas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove livros com ISBN duplicado e títulos idênticos de mesmo autor."""
        antes = len(df)

        # Duplicatas por ISBN_13 (mantém primeira ocorrência)
        df = df.drop_duplicates(subset=["ISBN_13"], keep="first")

        # Duplicatas por título + autor (para livros sem ISBN)
        df = df.drop_duplicates(subset=["titulo", "autor"], keep="first")

        removidos = antes - len(df)
        logger.info(f"  Duplicatas removidas: {removidos:,}")
        return df.reset_index(drop=True)

    def _limpar_colunas_basicas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata nulos, tipos e valores fora de domínio."""

        # Colunas de texto
        for col in ["titulo", "autor", "editora", "idioma", "genero", "descricao"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()

        # Colunas numéricas de engajamento — valores devem ser >= 0
        cols_engajamento = [
            "rating", "avaliacao", "resenha",
            "abandonos", "relendo", "querem_ler", "lendo", "leram",
        ]
        for col in cols_engajamento:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

        # Rating normalizado: alguns valores estão em escala 0-10, outros 0-5
        # Máximo histórico observado é 10.0. Normalizamos para 0-5.
        if "rating" in df.columns:
            df["rating"] = df["rating"].clip(0, 10)
            # Se maioria dos valores está acima de 5, escala é 0–10
            if df.loc[df["rating"] > 0, "rating"].median() > 5:
                df["rating"] = df["rating"] / 2.0

        # Percentuais male/female (0-100)
        for col in ["male", "female"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(50).clip(0, 100)

        # Páginas — limpar outliers (0 ou > 5000 são suspeitos)
        if "paginas" in df.columns:
            df["paginas"] = pd.to_numeric(df["paginas"], errors="coerce").fillna(0)
            df["paginas"] = df["paginas"].clip(0, 5000)

        # Ano
        if "ano" in df.columns:
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)

        logger.info("  Colunas básicas limpas")
        return df

    def _filtrar_anos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros com ano inválido (< 1900 ou > 2024)."""
        antes = len(df)
        df = df[(df["ano"] >= 1900) & (df["ano"] <= 2024)]
        removidos = antes - len(df)
        logger.info(f"  Registros com ano inválido removidos: {removidos:,}")
        return df.reset_index(drop=True)

    def _normalizar_generos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza a coluna gênero:
        - Extrai o primeiro gênero como `genero_principal`
        - Cria lista normalizada de todos os gêneros p/ encoding
        """
        if "genero" not in df.columns:
            df["genero_principal"] = "Desconhecido"
            df["generos_lista"] = ""
            return df

        def _limpar_genero(g: str) -> str:
            """Remove espaços extras, normaliza capitalização."""
            return g.strip().title() if g.strip() else ""

        def _extrair_primeiro(g: str) -> str:
            if not g:
                return "Desconhecido"
            partes = [_limpar_genero(p) for p in g.split("/") if p.strip()]
            # Ignora rótulos que parecem descrição (muito longos ou numéricos)
            partes = [p for p in partes if len(p) > 2 and len(p) < 50]
            return partes[0] if partes else "Desconhecido"

        def _listar_generos(g: str) -> str:
            if not g:
                return ""
            partes = [_limpar_genero(p) for p in g.split("/") if p.strip()]
            partes = [p for p in partes if len(p) > 2 and len(p) < 50]
            return "|".join(partes)

        df["genero_principal"] = df["genero"].apply(_extrair_primeiro)
        df["generos_lista"] = df["genero"].apply(_listar_generos)

        # Normalizar alguns nomes comuns
        _mapa = {
            "Ficção": "Ficção",
            "Literatura Estrangeira": "Literatura Estrangeira",
            "Romance": "Romance",
            "Jovem Adulto": "Jovem Adulto",
            "Infantojuvenil": "Infantojuvenil",
            "Não-Ficção": "Não-Ficção",
        }
        df["genero_principal"] = df["genero_principal"].replace(_mapa)

        logger.info(
            f"  Gêneros normalizados: {df['genero_principal'].nunique()} únicos"
        )
        return df

    def _calcular_popularidade_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o Score de Popularidade (variável alvo do modelo de ML).

        Fórmula (pesos configuráveis em config.py):
            popularidade_score = w1*norm(leram) + w2*norm(avaliacao) + w3*norm(resenha)

        Resultado normalizado entre 0 e 1.
        """
        from sklearn.preprocessing import MinMaxScaler

        componentes = {
            "leram": config.PESO_LERAM,
            "avaliacao": config.PESO_AVALIACAO,
            "resenha": config.PESO_RESENHA,
        }

        score = pd.Series(np.zeros(len(df)), index=df.index)
        scaler = MinMaxScaler()

        for col, peso in componentes.items():
            if col in df.columns:
                vals = df[[col]].values.astype(float)
                norm = scaler.fit_transform(vals).flatten()
                score += peso * norm

        df["popularidade_score"] = score.round(4)
        logger.info("  Score de popularidade calculado (0-1)")
        return df

    def _calcular_features_auxiliares(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features auxiliares derivadas."""

        # Taxa de abandono em relação a leitores totais
        total_leitores = df["leram"] + df["lendo"] + df["abandonos"]
        total_leitores = total_leitores.replace(0, 1)  # evitar divisão por zero
        df["taxa_abandono"] = (df["abandonos"] / total_leitores).round(4)
        df["taxa_conclusao"] = (df["leram"] / total_leitores).round(4)

        # Razão desejo/leitores (indicador de antecipação)
        df["razao_desejo"] = (
            df["querem_ler"] / (df["leram"] + 1)
        ).round(4)

        # Tamanho da descrição como feature de texto
        if "descricao" in df.columns:
            df["tamanho_descricao"] = df["descricao"].str.len()
        else:
            df["tamanho_descricao"] = 0

        logger.info("  Features auxiliares calculadas")
        return df

    def get_summary(self, df: pd.DataFrame) -> dict:
        """Retorna um resumo estatístico do dataset limpo."""
        return {
            "total_livros": len(df),
            "generos_unicos": df["genero_principal"].nunique(),
            "intervalo_anos": f"{df['ano'].min()}–{df['ano'].max()}",
            "editoras_unicas": df["editora"].nunique(),
            "media_popularidade": round(df["popularidade_score"].mean(), 3),
            "media_rating": round(df["rating"].mean(), 2),
            "total_leitores": int(df["leram"].sum()),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    df = loader.load_and_clean()
    print(f"\nResumo:\n{loader.get_summary(df)}")
