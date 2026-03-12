# TCC - Previsão de Tendências Literárias com Base em Dados do Skoob

## Planejamento
- [x] Leitura e análise do documento do TCC
- [x] Pesquisa sobre APIs e scraping do Skoob
- [x] Criação do plano de implementação
- [x] Aprovação do plano pelo usuário

## Etapa 1 — Estrutura do Projeto
- [x] Criar estrutura de pastas do projeto Python
- [x] Configurar dependências ([requirements.txt](file:///c:/Users/gabri/OneDrive/Documentos/TCC/requirements.txt))
- [x] Criar [config.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/config.py) com constantes e configurações

## Etapa 2 — Coleta de Dados (Web Scraping)
- [x] Módulo [scraper/skoob_client.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/scraper/skoob_client.py) — cliente HTTP
- [x] Analyze the [config.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/config.py) and [book_scraper.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/scraper/book_scraper.py) code to identify scraping speed issues.
- [x] Configure and structure concurrent scraping via `ThreadPoolExecutor` to drastically improve scraping speed.
- [x] Understand why the downloaded book page HTML text is not parsing the genre tags. Fix the HTML parsing, specifically locating the genres array embedded in Next.js Flight Data (`__next_f`).
- [x] Test the scraping on a small batch of pages to ensure it cleanly connects to the Data Processing steps.
- [x] Validate the Data processing pipeline, ensure `generos` dataset has data and that Machine Learning pipelines (StandardScaler) are able to train correctly!
- [x] Reset pipeline bounds to 5000 records so the user can easily run the massive scraping.
- [x] Módulo [scraper/book_scraper.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/scraper/book_scraper.py) — scraping de dados de livros
- [x] Módulo [scraper/ranking_scraper.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/scraper/ranking_scraper.py) — coleta de rankings por gênero
- [x] Módulo [scraper/review_scraper.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/scraper/review_scraper.py) — coleta de resenhas

## Etapa 3 — Tratamento de Dados
- [x] Módulo [processing/data_cleaner.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/processing/data_cleaner.py) — limpeza e deduplicação
- [x] Módulo [processing/data_enricher.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/processing/data_enricher.py) — enriquecimento com métricas
- [x] Módulo [processing/nlp_processor.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/processing/nlp_processor.py) — NLP nas resenhas

## Etapa 4 — Análise Descritiva
- [x] Módulo [analysis/descriptive.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/analysis/descriptive.py) — estatísticas e relatório
- [x] Módulo [analysis/visualizations.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/analysis/visualizations.py) — gráficos (matplotlib/seaborn)

## Etapa 5 — Modelagem Preditiva
- [x] Módulo [models/feature_engineering.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/models/feature_engineering.py) — criação de features
- [x] Módulo [models/train.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/models/train.py) — 3 modelos (Reg. Linear, Decision Tree, Random Forest)
- [x] Módulo [models/evaluate.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/models/evaluate.py) — avaliação (MAE, RMSE, R²)
- [x] Módulo [models/predict.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/models/predict.py) — predição de tendências

## Etapa 6 — Orquestração
- [x] Módulo [main.py](file:///c:/Users/gabri/OneDrive/Documentos/TCC/main.py) — pipeline CLI com menu interativo

## Verificação
- [x] Todos os imports validados (14/14)
- [x] Scraper testado com dados reais do Skoob (livro ID=1 → Ensaio sobre a Cegueira)
- [x] Fix do bug de encoding brotli no Accept-Encoding
