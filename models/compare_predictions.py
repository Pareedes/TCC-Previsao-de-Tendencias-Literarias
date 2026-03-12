"""
Módulo para comparar previsões históricas com a realidade atual.
Permite validar se as tendências previstas pelo modelo se concretizaram.
"""

import os
import glob
import pandas as pd
import logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("ComparePredictions")


class PredictionComparator:
    """
    Compara previsões passadas com os dados coletados atualmente.
    """

    def listar_previsoes_historicas(self):
        """Lista todos os arquivos de previsão no histórico."""
        history_dir = os.path.join(config.RESULTS_DIR, "history")
        if not os.path.exists(history_dir):
            return []
        
        arquivos = glob.glob(os.path.join(history_dir, "previsao_*.csv"))
        arquivos.sort(reverse=True) # Mais recentes primeiro
        return arquivos

    def comparar(self, prediction_file: str, current_trends_file: str = None):
        """
        Compara um arquivo de previsão passado com as tendências atuais.

        Args:
            prediction_file: Caminho para o CSV de previsão histórica.
            current_trends_file: Caminho para o CSV de tendências atuais (default: config.GENRE_TRENDS_FILE)
        """
        if current_trends_file is None:
            current_trends_file = config.GENRE_TRENDS_FILE
            
        if not os.path.exists(prediction_file) or not os.path.exists(current_trends_file):
            logger.error("Arquivos necessários não encontrados para comparação.")
            print(f"\n❌ Erro: Certifique-se de que existem previsões passadas e dados atuais processados.")
            return None

        # Carregar dados
        df_prev = pd.read_csv(prediction_file, encoding="utf-8")
        df_atual = pd.read_csv(current_trends_file, encoding="utf-8")

        # Selecionar coluna de previsão (pode ser ensemble ou o modelo específico)
        col_prev = "prev_ensemble" if "prev_ensemble" in df_prev.columns else \
                  ([c for c in df_prev.columns if c.startswith("prev_")][0] if any(c.startswith("prev_") for c in df_prev.columns) else "popularidade_prevista")
        
        # A realidade atual é ditada pelo `popularidade_score` mais recente
        if "popularidade_score" not in df_atual.columns:
            logger.error("Coluna 'popularidade_score' não encontrada nos dados atuais.")
            return None

        # Preparar DataFrames para o merge
        df_prev_clean = df_prev[["genero", col_prev, "tendencia", "data_previsao"]].copy()
        df_prev_clean = df_prev_clean.rename(columns={col_prev: "score_previsto"})
        
        df_atual_clean = df_atual[["genero", "popularidade_score"]].copy()
        df_atual_clean = df_atual_clean.rename(columns={"popularidade_score": "score_real"})

        # Calcular ranking da previsão e ranking da realidade
        df_prev_clean["ranking_previsto"] = df_prev_clean["score_previsto"].rank(ascending=False, method="min")
        df_atual_clean["ranking_real"] = df_atual_clean["score_real"].rank(ascending=False, method="min")

        # Realizar merge
        df_comp = pd.merge(df_prev_clean, df_atual_clean, on="genero", how="inner")

        if df_comp.empty:
            logger.warning("Nenhum gênero em comum para comparar.")
            return None

        # Calcular diferença de posições
        df_comp["mudanca_posicao"] = df_comp["ranking_previsto"] - df_comp["ranking_real"]
        df_comp["acerto_direcao"] = df_comp["mudanca_posicao"].apply(
            lambda x: "Exato" if x == 0 else ("Quase (" + str(int(x)) + ")" if abs(x) <= 2 else "Errou Mto")
        )

        # Ordenar pelo ranking real
        df_comp = df_comp.sort_values("ranking_real")
        
        self._gerar_relatorio(df_comp, prediction_file)
        return df_comp

    def _gerar_relatorio(self, df_comp: pd.DataFrame, prediction_file: str):
        """Gera um relatório formatado da comparação."""
        data_prev = df_comp["data_previsao"].iloc[0] if "data_previsao" in df_comp.columns else "Desconhecida"
        
        import datetime
        hoje = datetime.datetime.now().strftime("%Y-%m-%d")
        
        relatorio_txt = os.path.join(config.RESULTS_DIR, f"comparacao_previsao_{data_prev}_vs_{hoje}.txt")
        csv_comp_file = os.path.join(config.RESULTS_DIR, f"comparacao_previsao_{data_prev}_vs_{hoje}.csv")
        
        # Salvar csv
        df_comp.to_csv(csv_comp_file, index=False, encoding="utf-8")

        # Calcular métricas básicas
        mae_posicoes = abs(df_comp["ranking_previsto"] - df_comp["ranking_real"]).mean()
        exatos = len(df_comp[df_comp["ranking_previsto"] == df_comp["ranking_real"]])
        no_top_5_previsto = set(df_comp.nsmallest(5, "ranking_previsto")["genero"])
        no_top_5_real = set(df_comp.nsmallest(5, "ranking_real")["genero"])
        acertos_top_5 = len(no_top_5_previsto.intersection(no_top_5_real))

        with open(relatorio_txt, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"  🎯 VALIDAÇÃO DE PREVISÕES - BACKTEST\n")
            f.write(f"  Previsão feita em: {data_prev} | Realidade em: {hoje}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"  📈 DESEMPENHO GERAL GÊNEROS LITERÁRIOS:\n")
            f.write(f"  • Diferença Média de Posição (MAE): {mae_posicoes:.2f} posições\n")
            f.write(f"  • Acertos exatos de posição: {exatos} / {len(df_comp)}\n")
            f.write(f"  • Acertos de Gêneros no Top 5: {acertos_top_5} de 5\n\n")

            f.write(f"  {'Gênero':<25} {'Rank Previsto':<15} {'Rank Real':<15} {'Status':<15}\n")
            f.write("  " + "-" * 65 + "\n")
            
            for _, row in df_comp.head(30).iterrows():
                rank_p = int(row['ranking_previsto'])
                rank_r = int(row['ranking_real'])
                f.write(
                    f"  {row['genero']:<25} {rank_p:<15} {rank_r:<15} {row['acerto_direcao']}\n"
                )
            
            f.write("\n" + "=" * 70 + "\n")

        print("\n" + "=" * 70)
        print("  🎯 VALIDAÇÃO DE PREVISÕES CONCLUÍDA")
        print("=" * 70)
        with open(relatorio_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Pular cabeçalhos repetitivos para console
            print("".join(lines[5:]))

        print(f"  📁 Tabela CSV de Comparação salva em: {csv_comp_file}")
        print(f"  📄 Relatório Txt Completo salvo em: {relatorio_txt}")
        print("=" * 70)

if __name__ == "__main__":
    comp = PredictionComparator()
    arquivos = comp.listar_previsoes_historicas()
    if arquivos:
        print(f"Testando com o arquivo mais recente: {arquivos[0]}")
        comp.comparar(arquivos[0])
    else:
        print("Não há arquivos de previsão no histórico para comparar.")
