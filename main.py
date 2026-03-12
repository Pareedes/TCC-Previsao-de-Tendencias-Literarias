"""
Pipeline principal do projeto TCC.
Previsão de Tendências Literárias com Base em Dados do Skoob.

Autor: Gabriel Paredes Ferreira
IFTM - Campus Avançado Uberaba Parque Tecnológico
Engenharia de Computação

Uso:
    python main.py              # Menu interativo
    python main.py --coleta     # Executar apenas coleta
    python main.py --pipeline   # Pipeline completo
    python main.py --analise    # Apenas análise e modelos (requer dados coletados)
"""

import sys
import os
import logging
import argparse

# Configuração base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(config.BASE_DIR, "pipeline.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("Main")


def banner():
    """Exibe banner do projeto."""
    print("\n" + "=" * 70)
    print("  📚 PREVISÃO DE TENDÊNCIAS LITERÁRIAS COM BASE EM DADOS DO SKOOB")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  Gabriel Paredes Ferreira | IFTM | Engenharia de Computação")
    print("=" * 70)


def etapa_coleta(start_id: int = 1, end_id: int = 5000):
    """
    Etapa 1: Coleta de dados do Skoob via web scraping.

    Coleta dados de livros e resenhas, salvando em data/raw/.
    """
    from scraper.book_scraper import BookScraper
    from scraper.review_scraper import ReviewScraper

    print("\n📥 ETAPA 1: COLETA DE DADOS DO SKOOB")
    print("-" * 50)

    # 1.1 Coleta de livros
    print(f"\n🔍 Coletando dados de livros (IDs {start_id} a {end_id})...")
    print("   Isso pode levar várias horas dependendo do range de IDs.")
    print("   Os dados são salvos periodicamente em data/raw/books_raw.csv")
    print()

    with BookScraper() as scraper:
        books = scraper.scrape_range(start_id, end_id)

    print(f"\n✅ {len(books)} livros coletados e salvos!")

    # 1.2 Coleta de resenhas (top livros)
    if books:
        # Pegar IDs dos livros com mais leitores para coletar resenhas
        import pandas as pd
        df = pd.read_csv(config.RAW_BOOKS_FILE, encoding="utf-8")
        if "total_avaliacoes" in df.columns:
            top_ids = df.nlargest(200, "total_avaliacoes")["book_id"].tolist()
        else:
            top_ids = df["book_id"].head(200).tolist()

        print(f"\n📝 Coletando resenhas para {len(top_ids)} livros mais populares...")
        with ReviewScraper() as rscraper:
            reviews = rscraper.scrape_reviews_for_books(top_ids, max_reviews_per_book=20)
        print(f"✅ {len(reviews)} resenhas coletadas!")

    return books


def etapa_processamento():
    """
    Etapa 2: Tratamento e processamento dos dados.

    Limpeza, enriquecimento e NLP nas resenhas.
    """
    from processing.data_cleaner import DataCleaner
    from processing.data_enricher import DataEnricher
    from processing.nlp_processor import NLPProcessor

    print("\n🔧 ETAPA 2: PROCESSAMENTO E TRATAMENTO DE DADOS")
    print("-" * 50)

    # 2.1 Limpeza de dados
    print("\n🧹 Limpando dados de livros...")
    cleaner = DataCleaner()
    df_books = cleaner.clean_books()
    print(f"   ✅ {len(df_books)} livros após limpeza")

    # 2.2 Limpeza de resenhas
    if os.path.exists(config.RAW_REVIEWS_FILE):
        print("\n🧹 Limpando resenhas...")
        df_reviews = cleaner.clean_reviews()
        print(f"   ✅ {len(df_reviews)} resenhas após limpeza")

        # 2.3 NLP nas resenhas
        print("\n🧠 Processando resenhas com NLP...")
        nlp = NLPProcessor()
        df_nlp = nlp.process_reviews()
        print(f"   ✅ {len(df_nlp)} resenhas processadas com NLP")

    # 2.4 Enriquecimento de dados
    print("\n📊 Enriquecendo dados com métricas derivadas...")
    enricher = DataEnricher()
    df_enriched = enricher.enrich_books()
    print(f"   ✅ {len(df_enriched)} livros enriquecidos")

    # 2.5 Gerar tendências por gênero
    print("\n📈 Gerando tendências por gênero...")
    trends = enricher.generate_genre_trends(df_enriched)
    print(f"   ✅ {len(trends)} gêneros analisados")

    return df_enriched


def etapa_analise():
    """
    Etapa 3: Análise descritiva e visualizações.
    """
    from analysis.descriptive import DescriptiveAnalysis
    from analysis.visualizations import Visualizations

    print("\n📊 ETAPA 3: ANÁLISE DESCRITIVA E VISUALIZAÇÕES")
    print("-" * 50)

    # 3.1 Análise descritiva
    print("\n📋 Gerando relatório descritivo...")
    analyzer = DescriptiveAnalysis()
    report = analyzer.run_full_analysis()

    # 3.2 Visualizações
    print("\n🎨 Gerando gráficos...")
    viz = Visualizations()
    viz.generate_all()

    print(f"   ✅ Gráficos salvos em: {config.RESULTS_DIR}")

    return report


def etapa_modelagem():
    """
    Etapa 4: Modelagem preditiva com Machine Learning.
    """
    from models.feature_engineering import FeatureEngineer
    from models.train import ModelTrainer
    from models.evaluate import ModelEvaluator
    from models.predict import TrendPredictor
    from analysis.visualizations import Visualizations
    from sklearn.model_selection import train_test_split

    print("\n🤖 ETAPA 4: MODELAGEM PREDITIVA")
    print("-" * 50)

    # 4.1 Engenharia de features
    print("\n⚙️  Criando features para ML...")
    fe = FeatureEngineer()
    nlp_file = config.NLP_RESULTS_FILE if os.path.exists(config.NLP_RESULTS_FILE) else None
    features_df = fe.create_genre_features(nlp_file=nlp_file)
    X, y, feature_names, genre_names = fe.prepare_train_test(features_df)

    print(f"   Features: {X.shape[1]} | Amostras: {X.shape[0]}")

    # 4.2 Split treino/teste
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
    else:
        # Poucos dados: usar todos para treino e teste
        X_train, X_test = X, X
        y_train, y_test = y, y
        print("   ⚠️  Poucos dados: usando mesmo conjunto para treino e teste")

    # 4.3 Treinamento
    print("\n🏋️  Treinando modelos...")
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, feature_names)

    # 4.4 Avaliação
    print("\n📏 Avaliando modelos...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(trainer, X_test, y_test)

    # 4.5 Validação cruzada (se dados suficientes)
    cv_results = {}
    if len(X) >= 5:
        cv_results = evaluator.cross_validate(trainer, X, y)

    # 4.6 Imprimir comparação
    evaluator.print_comparison(results, cv_results)
    evaluator.save_results(results, cv_results)

    # 4.7 Feature importance
    print("\n📊 Importância das features:")
    importances = trainer.get_feature_importance()
    for model_name, imp_df in importances.items():
        print(f"\n  {model_name}:")
        for _, row in imp_df.head(5).iterrows():
            col = "importance" if "importance" in row else "coefficient"
            print(f"    {row['feature']:<30} {row[col]:.4f}")

    # 4.8 Salvar modelos
    trainer.save_models()
    print(f"\n   💾 Modelos salvos em: {config.MODELS_DIR}")

    # 4.9 Gerar visualizações de modelos
    viz = Visualizations()
    viz.plot_model_comparison(results)

    predictions_all = trainer.predict_all(X_test)
    best_model = evaluator.get_best_model(results)
    viz.plot_predictions_vs_actual(y_test, predictions_all[best_model], best_model)

    # 4.10 Predição de tendências
    print("\n🔮 Gerando previsão de tendências...")
    predictor = TrendPredictor()
    predictor.trainer = trainer
    predictions = predictor.predict_genre_trends(features_df)
    predictor.generate_report(predictions)

    return results, predictions


def etapa_comparacao():
    """
    Etapa Extra: Validação de previsões com dados atuais (Backtesting).
    """
    from models.compare_predictions import PredictionComparator
    print("\n🎯 ETAPA: VALIDAÇÃO DE PREVISÕES")
    print("-" * 50)
    
    comp = PredictionComparator()
    arquivos = comp.listar_previsoes_historicas()
    
    if not arquivos:
        print("\n  ❌ Nenhum histórico de previsão encontrado em data/results/history/")
        print("  Execute a etapa de modelagem pelo menos uma vez para gerar histórico.")
        return
        
    print("\n  Previsões históricas disponíveis:")
    for i, arq in enumerate(arquivos[:10]): # Mostrar até 10
        nome = os.path.basename(arq)
        print(f"  [{i+1}] {nome}")
        
    print(f"  [{len(arquivos[:10])+1}] Voltar")
    
    try:
        esc = int(input("\n  Escolha a previsão para comparar com a realidade ATUAL: ").strip())
        if esc == len(arquivos[:10]) + 1:
            return
        if 1 <= esc <= len(arquivos[:10]):
            arquivo_escolhido = arquivos[esc-1]
            print(f"\n  Comparando {os.path.basename(arquivo_escolhido)} com os gêneros atuais...\n")
            comp.comparar(arquivo_escolhido)
        else:
            print("  ❌ Opção inválida.")
    except (ValueError, EOFError, KeyboardInterrupt):
        print("  Operação cancelada.")


def menu_interativo():
    """Menu interativo para execução do pipeline."""
    banner()

    print("\n  Escolha uma opção:\n")
    print("  [1] 📥 Coleta de dados (web scraping do Skoob)")
    print("  [2] 🔧 Processamento de dados")
    print("  [3] 📊 Análise descritiva + visualizações")
    print("  [4] 🤖 Modelagem preditiva (ML)")
    print("  [5] 🚀 Pipeline completo (todas as etapas)")
    print("  [6] 📈 Apenas análise + modelos (requer dados já coletados)")
    print("  [7] 🎯 Validar previsões passadas (Backtesting)")
    print("  [0] ❌ Sair")
    print()

    try:
        opcao = input("  Opção: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Saindo...")
        return

    if opcao == "1":
        try:
            start = int(input("  ID inicial (padrão: 1): ").strip() or "1")
            end = int(input("  ID final (padrão: 5000): ").strip() or "5000")
        except ValueError:
            start, end = 1, 5000
        etapa_coleta(start, end)

    elif opcao == "2":
        etapa_processamento()

    elif opcao == "3":
        etapa_analise()

    elif opcao == "4":
        etapa_modelagem()

    elif opcao == "5":
        print("\n🚀 Executando pipeline completo...")
        etapa_coleta(1, 5000)
        etapa_processamento()
        etapa_analise()
        etapa_modelagem()
        print("\n🎉 Pipeline completo finalizado!")

    elif opcao == "6":
        print("\n📈 Executando análise + modelos com dados existentes...")
        etapa_processamento()
        etapa_analise()
        etapa_modelagem()
        print("\n🎉 Análise e modelagem finalizadas!")

    elif opcao == "0":
        print("\n  Saindo...")
    else:
        print("\n  ❌ Opção inválida!")


def main():
    """Entry point do pipeline."""
    parser = argparse.ArgumentParser(
        description="Previsão de Tendências Literárias - Pipeline"
    )
    parser.add_argument("--coleta", action="store_true", help="Executar coleta de dados")
    parser.add_argument("--processo", action="store_true", help="Executar processamento")
    parser.add_argument("--analise", action="store_true", help="Executar análise + modelos")
    parser.add_argument("--pipeline", action="store_true", help="Pipeline completo")
    parser.add_argument("--start-id", type=int, default=1, help="ID inicial para coleta")
    parser.add_argument("--end-id", type=int, default=config.TARGET_BOOKS_COUNT, help="ID final para coleta")

    args = parser.parse_args()

    banner()

    if args.pipeline:
        etapa_coleta(args.start_id, args.end_id)
        etapa_processamento()
        etapa_analise()
        etapa_modelagem()
    elif args.coleta:
        etapa_coleta(args.start_id, args.end_id)
    elif args.processo:
        etapa_processamento()
    elif args.analise:
        etapa_processamento()
        etapa_analise()
        etapa_modelagem()
    else:
        menu_interativo()


if __name__ == "__main__":
    main()
