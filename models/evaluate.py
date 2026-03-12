"""
Módulo de avaliação e comparação de modelos de ML.
Calcula métricas de performance e gera relatórios comparativos.
"""

import numpy as np
import pandas as pd
import logging
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("ModelEvaluator")


class ModelEvaluator:
    """
    Avaliador de modelos preditivos.

    Métricas:
    - MAE (Mean Absolute Error) - erro médio absoluto
    - RMSE (Root Mean Squared Error) - raiz do erro quadrático médio
    - R² (Coeficiente de Determinação) - explicabilidade do modelo
    """

    def evaluate_model(
        self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """
        Avalia um modelo com as principais métricas de regressão.

        Returns:
            Dicionário com MAE, RMSE e R²
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        results = {
            "modelo": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }

        logger.info(
            f"  {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
        )

        return results

    def evaluate_all(
        self, trainer, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """
        Avalia todos os modelos treinados.

        Args:
            trainer: Instância de ModelTrainer com modelos treinados
            X_test: Features de teste
            y_test: Target de teste

        Returns:
            Dicionário {nome_modelo: métricas}
        """
        logger.info("Avaliando modelos...")
        all_results = {}

        predictions = trainer.predict_all(X_test)

        for name, y_pred in predictions.items():
            results = self.evaluate_model(name, y_test, y_pred)
            all_results[name] = results

        # Identificar melhor modelo
        best = min(all_results.values(), key=lambda x: x["MAE"])
        logger.info(f"\n  🏆 Melhor modelo (menor MAE): {best['modelo']}")

        return all_results

    def cross_validate(
        self, trainer, X: np.ndarray, y: np.ndarray, cv: int = None
    ) -> dict:
        """
        Executa validação cruzada em todos os modelos.

        Returns:
            Dicionário com scores de validação cruzada
        """
        if cv is None:
            cv = min(config.CV_FOLDS, len(X))  # Ajustar folds se poucos dados

        if cv < 2:
            logger.warning("Dados insuficientes para validação cruzada")
            return {}

        logger.info(f"Validação cruzada ({cv}-fold)...")
        cv_results = {}

        X_scaled = trainer.scaler.transform(X)

        for name, model in trainer.models.items():
            scores = cross_val_score(
                model, X_scaled, y, cv=cv, scoring="r2"
            )
            cv_results[name] = {
                "R2_medio": round(scores.mean(), 4),
                "R2_std": round(scores.std(), 4),
                "R2_scores": scores.round(4).tolist(),
            }
            logger.info(
                f"  {name}: R² = {scores.mean():.4f} (±{scores.std():.4f})"
            )

        return cv_results

    def get_best_model(self, results: dict) -> str:
        """Retorna o nome do melhor modelo baseado no MAE."""
        best = min(results.items(), key=lambda x: x[1]["MAE"])
        return best[0]

    def generate_comparison_table(self, results: dict) -> pd.DataFrame:
        """
        Gera uma tabela comparativa dos modelos.

        Returns:
            DataFrame com a comparação
        """
        rows = []
        for name, metrics in results.items():
            rows.append({
                "Modelo": name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R²": metrics["R2"],
            })

        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values("MAE")

        # Marcar melhor modelo
        comparison["Ranking"] = range(1, len(comparison) + 1)

        return comparison

    def print_comparison(self, results: dict, cv_results: dict = None):
        """Imprime uma comparação formatada dos modelos."""
        print("\n" + "=" * 70)
        print("  COMPARAÇÃO DE MODELOS PREDITIVOS")
        print("=" * 70)

        table = self.generate_comparison_table(results)

        print(f"\n{'Ranking':<10} {'Modelo':<25} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
        print("-" * 70)

        for _, row in table.iterrows():
            emoji = "🥇" if row["Ranking"] == 1 else ("🥈" if row["Ranking"] == 2 else "🥉")
            print(
                f"  {emoji} {row['Ranking']:<5} {row['Modelo']:<25} "
                f"{row['MAE']:<12.4f} {row['RMSE']:<12.4f} {row['R²']:<12.4f}"
            )

        if cv_results:
            print(f"\n{'':─<70}")
            print("  VALIDAÇÃO CRUZADA")
            print(f"{'':─<70}")
            for name, cv in cv_results.items():
                print(f"  {name:<25} R² = {cv['R2_medio']:.4f} (±{cv['R2_std']:.4f})")

        print("\n" + "=" * 70)

    def save_results(self, results: dict, cv_results: dict = None, filepath: str = None):
        """Salva resultados em CSV."""
        if filepath is None:
            filepath = os.path.join(config.RESULTS_DIR, "resultados_modelos.csv")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        table = self.generate_comparison_table(results)

        if cv_results:
            table["CV_R2_medio"] = table["Modelo"].map(
                lambda m: cv_results.get(m, {}).get("R2_medio", "N/A")
            )
            table["CV_R2_std"] = table["Modelo"].map(
                lambda m: cv_results.get(m, {}).get("R2_std", "N/A")
            )

        table.to_csv(filepath, index=False, encoding="utf-8")
        logger.info(f"  Resultados salvos: {filepath}")


if __name__ == "__main__":
    print("ModelEvaluator - Módulo de avaliação de modelos")
    print("Use main.py para executar o pipeline completo.")
