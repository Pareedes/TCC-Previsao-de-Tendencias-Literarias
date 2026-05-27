"""
Pipeline principal — Previsão de Tendências Literárias (Skoob)
IFTM — Engenharia de Computação | TCC — Gabriel Paredes Ferreira

Fonte de dados: Dataset Público do Skoob (~12.000 livros, Public Domain)
Objetivos:
  1. Prever a popularidade de livros com base em características e gênero
  2. Identificar gêneros literários em crescimento ao longo de 2000–2020
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Configurar logging antes de qualquer import do projeto
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("Main")

# ---------------------------------------------------------------------------
# Imports do projeto
# ---------------------------------------------------------------------------
import config

from processing.data_loader import DataLoader
from processing.nlp_processor import NLPProcessor
from models.feature_engineering import FeatureEngineer
from models.train import ModelTrainer
from models.evaluate import ModelEvaluator
from analysis.genre_trends import GenreTrendAnalyzer
from analysis.visualizations import Visualizations


# ===========================================================================
# FUNÇÕES DE ETAPA
# ===========================================================================

def etapa_carregar(args):
    """Etapa 1: Carrega e limpa o dataset CSV público."""
    logger.info("=" * 60)
    logger.info("ETAPA 1 — CARGA E LIMPEZA DO DATASET")
    logger.info("=" * 60)

    loader = DataLoader()
    df = loader.load_and_clean()
    resumo = loader.get_summary(df)

    logger.info("\nRESUMO DO DATASET:")
    for k, v in resumo.items():
        logger.info(f"  {k}: {v}")

    return df


def etapa_nlp(args):
    """Etapa 2 (opcional): NLP nas descrições dos livros."""
    logger.info("=" * 60)
    logger.info("ETAPA 2 — PROCESSAMENTO NLP DAS DESCRIÇÕES")
    logger.info("=" * 60)

    import pandas as pd
    import re

    df = pd.read_csv(config.CLEANED_BOOKS_FILE, encoding="utf-8")

    if "descricao" not in df.columns or df["descricao"].dropna().empty:
        logger.warning("Coluna 'descricao' não encontrada ou vazia. Pulando NLP.")
        return

    nlp = NLPProcessor()

    logger.info(f"Calculando sentimento para {len(df):,} descrições...")
    df["sentimento_score"] = df["descricao"].fillna("").apply(nlp.analyze_sentiment)
    df["descricao_keywords"] = df["descricao"].fillna("").apply(nlp.extract_keywords)

    out = df[["titulo", "autor", "sentimento_score", "descricao_keywords"]]
    os.makedirs(os.path.dirname(config.NLP_RESULTS_FILE), exist_ok=True)
    out.to_csv(config.NLP_RESULTS_FILE, index=False, encoding="utf-8")

    logger.info(f"  NLP concluído → {config.NLP_RESULTS_FILE}")
    logger.info(f"  Sentimento médio: {df['sentimento_score'].mean():.3f}")


def etapa_analise(args):
    """Etapa 3: Análise descritiva e visualizações."""
    logger.info("=" * 60)
    logger.info("ETAPA 3 — ANÁLISE DESCRITIVA E VISUALIZAÇÕES")
    logger.info("=" * 60)

    viz = Visualizations()
    viz.generate_all()

    logger.info(f"Gráficos salvos em: {config.RESULTS_DIR}")


def etapa_tendencias(args):
    """Etapa 4: Análise temporal de tendências de gêneros (2000–2020)."""
    logger.info("=" * 60)
    logger.info("ETAPA 4 — ANÁLISE TEMPORAL DE TENDÊNCIAS DE GÊNEROS")
    logger.info("=" * 60)

    analyzer = GenreTrendAnalyzer()
    df_tend = analyzer.analisar()

    # Gerar gráficos de tendências
    viz = Visualizations()

    if analyzer.df_anual is not None:
        viz.plot_genre_timeline(analyzer.df_anual)
        viz.plot_growth_heatmap(analyzer.df_anual)

    viz.plot_genre_growth_bar(df_tend)

    logger.info("\nRESUMO DAS TENDÊNCIAS:")
    for tendencia in ["Ascensão", "Emergente", "Estagnação", "Declínio"]:
        count = (df_tend.get("tendencia", []) == tendencia).sum()
        logger.info(f"  {tendencia}: {count} gêneros")

    return df_tend


def etapa_modelo(args):
    """Etapa 5: Treinamento e avaliação dos modelos de ML."""
    logger.info("=" * 60)
    logger.info("ETAPA 5 — TREINAMENTO E AVALIAÇÃO DOS MODELOS DE ML")
    logger.info("=" * 60)

    # Feature Engineering
    fe = FeatureEngineer()
    temporal = not getattr(args, "sem_holdout_temporal", False)

    X_train, X_test, y_train, y_test, feature_names, df_train, df_test = (
        fe.preparar_features(temporal_split=temporal)
    )

    logger.info(
        f"Features: {len(feature_names)} | "
        f"Treino: {len(X_train):,} | Teste: {len(X_test):,}"
    )

    # Treino
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, feature_names=feature_names)
    trainer.save_models()

    # Avaliação
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(trainer, X_test, y_test)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS DOS MODELOS")
    logger.info("=" * 60)
    logger.info(f"{'Modelo':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R²':>8}")
    logger.info("-" * 60)

    for nome, metricas in sorted(results.items(), key=lambda x: -x[1].get("R2", 0)):
        logger.info(
            f"{nome:<25} "
            f"{metricas.get('MAE', 0):>8.4f} "
            f"{metricas.get('RMSE', 0):>8.4f} "
            f"{metricas.get('R2', 0):>8.4f} "
            f"{metricas.get('CV_R2_mean', 0):>8.4f}"
        )

    # Gráficos dos modelos
    viz = Visualizations()
    viz.plot_model_comparison(results)

    # Gráfico predição vs real para o melhor modelo
    melhor = max(results, key=lambda m: results[m].get("R2", 0))
    y_pred_melhor = trainer.predict(melhor, X_test)
    viz.plot_predictions_vs_actual(y_test, y_pred_melhor, melhor)

    # Feature importance
    importances = trainer.get_feature_importance()
    viz.plot_feature_importance(importances)

    # Salvar resultados como CSV
    import pandas as pd
    rows = [{"Modelo": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(config.MODEL_RESULTS_FILE, index=False)
    logger.info(f"\nResultados salvos: {config.MODEL_RESULTS_FILE}")

    return trainer, fe, results


def etapa_prever_hipotetico(trainer=None, fe=None, input_data=None):
    """Etapa 6 (interativa): Prevê popularidade de um livro hipotético."""
    logger.info("=" * 60)
    logger.info("ETAPA 6 — PREVISÃO DE LIVRO HIPOTÉTICO")
    logger.info("=" * 60)

    # Carregar modelos se não fornecidos
    if trainer is None:
        trainer = ModelTrainer()
        try:
            trainer.load_models()
        except Exception:
            logger.error("Modelos não encontrados. Execute a etapa de treinamento primeiro.")
            return

    if fe is None:
        fe = FeatureEngineer()
        try:
            fe.preparar_features()  # Reprocessa para recriar mapeamentos internos
        except Exception:
            logger.error("Erro ao carregar features. Execute a etapa de treinamento primeiro.")
            return

    if input_data:
        genero = input_data.get("genero", "Romance")
        paginas = input_data.get("paginas", 300)
        ano = input_data.get("ano", 2024)
        rating = input_data.get("rating", 4.0)
        descricao = input_data.get("descricao", "")
        editora = input_data.get("editora", "")
    else:
        print("\n" + "=" * 60)
        print("  PREVISÃO DE POPULARIDADE — LIVRO HIPOTÉTICO")
        print("=" * 60)
        print("Informe as características do livro para prever sua popularidade:")
        print(f"  Generos disponíveis: {', '.join(fe.top_generos[:10])}")
        print()

        genero = input("  Gênero principal: ").strip() or "Romance"
        paginas_str = input("  Número de páginas [300]: ").strip()
        paginas = int(paginas_str) if paginas_str.isdigit() else 300
        ano_str = input("  Ano de publicação [2024]: ").strip()
        ano = int(ano_str) if ano_str.isdigit() else 2024
        rating_str = input("  Rating esperado 0-5 [4.0]: ").strip()
        try:
            rating = float(rating_str) if rating_str else 4.0
        except ValueError:
            rating = 4.0
        descricao = input("  Breve descrição/sinopse (opcional): ").strip()
        editora = input("  Editora (Rocco/Intrínseca/Sextante/Planeta - Enter p/ pular): ").strip()



    resultados = fe.prever_livro_hipotetico(
        trainer=trainer,
        genero=genero,
        paginas=paginas,
        ano=ano,
        rating=rating,
        editora=editora,
        descricao=descricao,
    )

    # Separar metadados das previsões por modelo
    livros_similares = resultados.pop("_livros_similares", [])
    score_medio = resultados.pop("_score_medio", 0)

    print("\n" + "─" * 65)
    print(f"  LIVRO: {genero} | {paginas}p | {ano} | Rating {rating:.1f}"
          + (f" | {editora}" if editora else ""))
    print("─" * 65)
    print(f"  {'Modelo':<25} {'Score':>8}  {'Pos. no dataset':>15}  Classificação")
    print("  " + "-" * 62)
    for modelo, pred in resultados.items():
        print(
            f"  {modelo:<25} "
            f"{pred['popularidade_score']:>8.4f}  "
            f"Top {pred['top_pct']:>5.1f}% ({pred['percentil']:.0f}° pct)  "
            f"{pred['classificacao']}"
        )
    print("─" * 65)
    print(f"  Score medio previsto: {score_medio:.4f}")

    if livros_similares:
        print(f"\n  Livros reais do Skoob com popularidade similar:")
        for liv in livros_similares:
            print(f"    - {liv['titulo'][:48]:<48} | score real: {liv['score_real']:.4f} | {liv['leram']} leram")
    print()


def etapa_pipeline_completo(args):
    """Executa todas as etapas em sequência."""
    inicio = time.time()
    logger.info("\n" + "█" * 60)
    logger.info("  PIPELINE COMPLETO — PREVISÃO DE TENDÊNCIAS LITERÁRIAS")
    logger.info("█" * 60 + "\n")

    etapa_carregar(args)
    etapa_nlp(args)
    etapa_analise(args)
    etapa_tendencias(args)
    trainer, fe, results = etapa_modelo(args)

    elapsed = time.time() - inicio
    logger.info(f"\n✓ Pipeline concluído em {elapsed:.1f}s")
    return trainer, fe


# ===========================================================================
# MENU INTERATIVO
# ===========================================================================

MENU = """
╔══════════════════════════════════════════════════════════════╗
║   PREVISÃO DE TENDÊNCIAS LITERÁRIAS — SKOOB DATASET         ║
║   IFTM | Engenharia de Computação | TCC 2026                ║
╠══════════════════════════════════════════════════════════════╣
║  [1] Carregar e limpar dataset (dados.csv)                  ║
║  [2] Processamento NLP das descrições                       ║
║  [3] Análise descritiva e gráficos                          ║
║  [4] Análise temporal de tendências (2000–2020)             ║
║  [5] Treinar e avaliar modelos de ML                        ║
║  [6] Prever popularidade de livro hipotético                ║
║  [7] Executar pipeline completo                             ║
║  [0] Sair                                                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def menu_interativo():
    """Loop do menu interativo."""
    trainer_cache = None
    fe_cache = None

    while True:
        print(MENU)
        opcao = input("  Opção: ").strip()

        args = argparse.Namespace(sem_holdout_temporal=False)

        if opcao == "0":
            print("\n  Encerrando. Até logo!\n")
            break
        elif opcao == "1":
            etapa_carregar(args)
        elif opcao == "2":
            etapa_nlp(args)
        elif opcao == "3":
            etapa_analise(args)
        elif opcao == "4":
            etapa_tendencias(args)
        elif opcao == "5":
            trainer_cache, fe_cache, _ = etapa_modelo(args)
        elif opcao == "6":
            etapa_prever_hipotetico(trainer_cache, fe_cache)
        elif opcao == "7":
            trainer_cache, fe_cache = etapa_pipeline_completo(args)
        else:
            print("  Opção inválida. Tente novamente.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de Previsão de Tendências Literárias — Skoob Dataset"
    )
    parser.add_argument("--carregar", action="store_true",
                        help="Etapa 1: Carregar e limpar dataset")
    parser.add_argument("--nlp", action="store_true",
                        help="Etapa 2: Processamento NLP")
    parser.add_argument("--analise", action="store_true",
                        help="Etapa 3: Análise descritiva e gráficos")
    parser.add_argument("--tendencias", action="store_true",
                        help="Etapa 4: Análise temporal de gêneros")
    parser.add_argument("--modelo", action="store_true",
                        help="Etapa 5: Treinar e avaliar modelos")
    parser.add_argument("--prever", action="store_true",
                        help="Etapa 6: Prever livro hipotético (interativo)")
    parser.add_argument("--pipeline", action="store_true",
                        help="Executar pipeline completo")
    parser.add_argument("--sem-holdout-temporal", action="store_true",
                        dest="sem_holdout_temporal",
                        help="Usar split aleatório em vez do holdout temporal")
    return parser.parse_args()


def main():
    args = parse_args()

    # Se nenhum argumento CLI, abre menu interativo
    qualquer_flag = any([
        args.carregar, args.nlp, args.analise,
        args.tendencias, args.modelo, args.prever, args.pipeline,
    ])

    if not qualquer_flag:
        menu_interativo()
        return

    # Modo CLI
    if args.pipeline:
        etapa_pipeline_completo(args)
        return

    trainer_ref = None
    fe_ref = None

    if args.carregar:
        etapa_carregar(args)
    if args.nlp:
        etapa_nlp(args)
    if args.analise:
        etapa_analise(args)
    if args.tendencias:
        etapa_tendencias(args)
    if args.modelo:
        trainer_ref, fe_ref, _ = etapa_modelo(args)
    if args.prever:
        etapa_prever_hipotetico(trainer_ref, fe_ref)


if __name__ == "__main__":
    main()
