"""
Módulo de treinamento de modelos de Machine Learning.
Implementa 3 modelos conforme o TCC: Regressão Linear, Árvore de Decisão, Random Forest.
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("ModelTrainer")


class ModelTrainer:
    """
    Treinador de modelos preditivos para tendências literárias.

    Modelos implementados (conforme metodologia do TCC):
    1. Regressão Linear - baseline, relações lineares
    2. Árvore de Decisão - captura não-linearidades
    3. Random Forest - ensemble, maior robustez
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def train_all_models(
        self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None
    ) -> dict:
        """
        Treina os 3 modelos com os dados de treinamento.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            feature_names: Nomes das features

        Returns:
            Dicionário com os modelos treinados
        """
        if feature_names:
            self.feature_names = feature_names

        # Normalizar features
        X_scaled = self.scaler.fit_transform(X_train)

        logger.info(
            f"Treinando modelos com {X_train.shape[0]} amostras e "
            f"{X_train.shape[1]} features"
        )

        # 1. Regressão Linear
        logger.info("  Treinando: Regressão Linear...")
        lr = LinearRegression()
        lr.fit(X_scaled, y_train)
        self.models["Regressão Linear"] = lr
        logger.info("    Regressão Linear treinada!")

        # 2. Árvore de Decisão
        logger.info("  Treinando: Árvore de Decisão...")
        dt = DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
        )
        dt.fit(X_scaled, y_train)
        self.models["Árvore de Decisão"] = dt
        logger.info("    Árvore de Decisão treinada!")

        # 3. Random Forest
        logger.info("  Treinando: Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_scaled, y_train)
        self.models["Random Forest"] = rf
        logger.info("    Random Forest treinada!")

        self.is_trained = True
        logger.info(f"  Todos os {len(self.models)} modelos treinados com sucesso!")

        return self.models

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Gera previsões com um modelo específico.

        Args:
            model_name: Nome do modelo
            X: Features para predição

        Returns:
            Array com as previsões
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' não encontrado. "
                           f"Disponíveis: {list(self.models.keys())}")

        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict(X_scaled)

    def predict_all(self, X: np.ndarray) -> dict:
        """
        Gera previsões com todos os modelos.

        Returns:
            Dicionário {nome_modelo: previsões}
        """
        predictions = {}
        X_scaled = self.scaler.transform(X)

        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)

        return predictions

    def get_feature_importance(self) -> dict:
        """
        Retorna a importância das features para cada modelo.

        Returns:
            Dicionário {modelo: DataFrame com importâncias}
        """
        importances = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({
                    "feature": self.feature_names,
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False)
                importances[name] = imp
            elif hasattr(model, "coef_"):
                imp = pd.DataFrame({
                    "feature": self.feature_names,
                    "coefficient": model.coef_
                }).sort_values("coefficient", ascending=False, key=abs)
                importances[name] = imp

        return importances

    def save_models(self, directory: str = None):
        """Salva todos os modelos treinados em disco."""
        if directory is None:
            directory = config.MODELS_DIR

        os.makedirs(directory, exist_ok=True)

        # Salvar scaler
        scaler_path = os.path.join(directory, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Salvar modelos
        for name, model in self.models.items():
            filename = name.lower().replace(" ", "_") + ".pkl"
            filepath = os.path.join(directory, filename)
            with open(filepath, "wb") as f:
                pickle.dump(model, f)

        # Salvar feature names
        meta_path = os.path.join(directory, "model_metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_names": self.feature_names,
                "model_names": list(self.models.keys()),
            }, f)

        logger.info(f"  Modelos salvos em: {directory}")

    def load_models(self, directory: str = None):
        """Carrega modelos salvos do disco."""
        if directory is None:
            directory = config.MODELS_DIR

        # Carregar metadata
        meta_path = os.path.join(directory, "model_metadata.pkl")
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        self.feature_names = metadata["feature_names"]

        # Carregar scaler
        scaler_path = os.path.join(directory, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Carregar modelos
        for name in metadata["model_names"]:
            filename = name.lower().replace(" ", "_") + ".pkl"
            filepath = os.path.join(directory, filename)
            with open(filepath, "rb") as f:
                self.models[name] = pickle.load(f)

        self.is_trained = True
        logger.info(f"  {len(self.models)} modelos carregados de: {directory}")


if __name__ == "__main__":
    print("ModelTrainer - Módulo de treinamento de modelos")
    print("Use main.py para executar o pipeline completo.")
