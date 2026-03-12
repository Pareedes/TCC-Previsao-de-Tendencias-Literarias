"""
Módulo de predição de tendências literárias.
Utiliza os modelos treinados para gerar previsões futuras.
"""

import numpy as np
import pandas as pd
import logging
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.train import ModelTrainer

logger = logging.getLogger("Predictor")


class TrendPredictor:
    """
    Preditor de tendências literárias.

    Carrega o melhor modelo treinado e gera previsões de popularidade
    futura por gênero, auxiliando autores e editoras em decisões estratégicas.
    """

    def __init__(self):
        self.trainer = ModelTrainer()
        self.best_model_name = None

    def load_models(self, model_dir: str = None):
        """Carrega modelos salvos."""
        self.trainer.load_models(model_dir)
        logger.info("Modelos carregados para predição")

    def predict_genre_trends(
        self,
        features_df: pd.DataFrame,
        model_name: str = None,
    ) -> pd.DataFrame:
        """
        Gera previsões de popularidade para cada gênero.

        Args:
            features_df: DataFrame com features por gênero
            model_name: Nome do modelo a usar (usa todos se None)

        Returns:
            DataFrame com previsões por gênero
        """
        genres = features_df["genero"].tolist()

        # Features (excluir colunas não-feature)
        drop_cols = ["genero", "target_popularidade"]
        feature_cols = [c for c in features_df.columns if c not in drop_cols]
        X = features_df[feature_cols].values

        if model_name:
            predictions = self.trainer.predict(model_name, X)
            result = pd.DataFrame({
                "genero": genres,
                "popularidade_prevista": predictions.round(2),
            })
        else:
            # Prever com todos os modelos
            all_preds = self.trainer.predict_all(X)
            result = pd.DataFrame({"genero": genres})
            for name, preds in all_preds.items():
                col_name = f"prev_{name.lower().replace(' ', '_')}"
                result[col_name] = preds.round(2)

            # Média ensemble (média de todos os modelos)
            pred_cols = [c for c in result.columns if c.startswith("prev_")]
            result["prev_ensemble"] = result[pred_cols].mean(axis=1).round(2)

        # Ordenar por previsão
        sort_col = "prev_ensemble" if "prev_ensemble" in result.columns else result.columns[-1]
        result = result.sort_values(sort_col, ascending=False)

        # Ranking
        result["ranking_previsto"] = range(1, len(result) + 1)

        # Classificação de tendência
        median_val = result[sort_col].median()
        result["tendencia"] = result[sort_col].apply(
            lambda x: "🔥 Em Alta" if x > median_val * 1.2
            else ("📈 Crescendo" if x > median_val
                  else ("📉 Estável" if x > median_val * 0.8
                        else "⬇️ Em Queda"))
        )

        return result

    def generate_report(
        self, predictions: pd.DataFrame, output_file: str = None
    ):
        """
        Gera relatório de previsões de tendências com histórico.

        Args:
            predictions: DataFrame com previsões
            output_file: Caminho do arquivo de saída principal
        """
        import datetime
        hoje = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Adicionar data da previsão
        predictions["data_previsao"] = hoje

        if output_file is None:
            output_file = os.path.join(config.RESULTS_DIR, "previsao_tendencias.csv")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Salvar o arquivo atual sempre sobrescrevendo
        predictions.to_csv(output_file, index=False, encoding="utf-8")
        
        # Salvar também no histórico
        history_dir = os.path.join(config.RESULTS_DIR, "history")
        os.makedirs(history_dir, exist_ok=True)
        historico_file = os.path.join(history_dir, f"previsao_{hoje}.csv")
        predictions.to_csv(historico_file, index=False, encoding="utf-8")

        # Gerar um relatório em texto amigável
        relatorio_txt = os.path.join(config.RESULTS_DIR, f"relatorio_previsao_{hoje}.txt")
        sort_col = "prev_ensemble" if "prev_ensemble" in predictions.columns else predictions.columns[1]
        
        with open(relatorio_txt, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"  🔮 RELATÓRIO DE PREVISÃO DE TENDÊNCIAS LITERÁRIAS - {hoje}\n")
            f.write("  Baseado em modelos de Machine Learning sobre dados do Skoob\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"  {'Rank':<6} {'Gênero':<25} {'Score Previsto':<15} {'Tendência':<15}\n")
            f.write("  " + "-" * 65 + "\n")
            
            for _, row in predictions.head(30).iterrows():
                score = row.get(sort_col, 0)
                f.write(
                    f"  {row['ranking_previsto']:<6} {row['genero']:<25} "
                    f"{score:<15.2f} {row['tendencia']}\n"
                )
            
            f.write("\n  📊 RESUMO DE DESTAQUES:\n")
            top_5 = predictions.head(5)["genero"].tolist()
            bottom_5 = predictions.tail(5)["genero"].tolist()
            f.write(f"  • Top Gênero Absoluto: {top_5[0]}\n")
            f.write(f"  • Principais tendências de ALTA: {', '.join(top_5)}\n")
            f.write(f"  • Principais tendências de QUEDA: {', '.join(bottom_5)}\n")
            f.write("\n" + "=" * 70 + "\n")

        # Imprimir no console
        with open(relatorio_txt, "r", encoding="utf-8") as f:
            print(f.read())

        print(f"\n  📁 CSV Principal: {output_file}")
        print(f"  📁 CSV Histórico: {historico_file}")
        print(f"  📄 Relatório Txt: {relatorio_txt}")
        print("=" * 70)

        logger.info(f"Relatórios de previsão salvos em: {config.RESULTS_DIR}")


if __name__ == "__main__":
    print("TrendPredictor - Módulo de predição de tendências")
    print("Use main.py para executar o pipeline completo.")
