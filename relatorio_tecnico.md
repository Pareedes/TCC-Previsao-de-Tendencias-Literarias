# Relatório Técnico: Previsão de Tendências Literárias (Skoob)

Este documento detalha tecnicamente o funcionamento completo do pipeline de dados, incluindo o fluxo de execução, as variáveis coletadas e tratadas, e a fundamentação legal e técnica da coleta de dados online (Web Scraping).

---

## 1. Funcionamento do Programa (Pipeline Ponto a Ponto)

O sistema (orquestrado pelo `main.py`) opera em um pipeline de 4 estágios lineares:

### Etapa 1: Coleta de Dados (Scraping)
1. O script inicializa o `BookScraper` com um pool de 5 threads concorrentes (via `ThreadPoolExecutor`), otimizando o tempo de processamento.
2. Cada thread solicita a página de um livro específico (`https://www.skoob.com.br/pt/book/{id}`).
3. A extração intercepta tanto os metadados visíveis na página HTML (usando BeautifulSoup) quanto os dados estruturados trafegados nos bastidores do Next.js (Flight Data JSON).
4. Em seguida, o `ReviewScraper` busca as resenhas textuais dos livros mais populares utilizando os endpoints públicos (ex: `.../books/{id}/reviews`).
5. Os dados extraídos são salvos em `data/raw/books_raw.csv` e `data/raw/reviews_raw.csv`.

### Etapa 2: Processamento e Tratamento (Data Cleaning & NLP)
1. **Limpeza:** O módulo `DataCleaner` remove livros sem informações básicas cruciais e padroniza dados incorretos, como datas e tipos numéricos.
2. **Processamento NLP:** O módulo `NLPProcessor` atua nos textos das resenhas (Natural Language Processing). Usando a biblioteca `nltk`, ele tokeniza os textos, remove "stopwords" (palavras neutras do português) e analisa se o sentimento da resenha é Positivo, Negativo ou Neutro.
3. **Enriquecimento:** O `DataEnricher` funde esses dados textuais normalizados às estatísticas frias, calculando Média Móvel de Notas, Taxa de Engajamento, market_share (%) e um *Score Absoluto de Popularidade* para cada um dos gêneros contidos no banco.
4. Gera-se o `data/processed/genre_trends.csv`.

### Etapa 3: Análises Descritivas e Visualização
1. O Módulo `DescriptiveAnalysis` calcula coeficientes e médias percentuais.
2. Usando `matplotlib` e `seaborn`, os scripts geram ~10 gráficos salvos na pasta `data/results/`, incluindo Correlação de Spearman, Distribuição de Sentimentos (NLP) e Evolução do Score dos gêneros literários.

### Etapa 4: Modelagem Preditiva (Machine Learning)
1. Através da biblioteca `Scikit-Learn`, a classe `FeatureEngineer` agrupa as métricas de Engajamento, Sentimento e Popularidade como "Features" independentes.
2. O sistema divide a base entre "Treino" (80%) e "Teste" (20%).
3. Três modelos matemáticos competem para aprender as relações causais na base de Treino:
   - **Regressão Linear Múltipla**
   - **Árvore de Decisão** (Decision Tree Regressor)
   - **Floresta Aleatória** (Random Forest Regressor)
4. O `ModelEvaluator` testa-os contra a base de Teste que eles nunca viram. O modelo avalia usando erro médio ($MAE$) e R-quadrado ($R^2$). O modelo com melhor performance (geralmente Random Forest) é salvo em disco (`data/models/`).
5. O `TrendPredictor` infere o "Popularidade Score" virtual de todos os gêneros usando as anomalias atuais do passo de treinamento, ranqueando quais Gêneros estão em Tendência Alta. Estes resultados são gerados no `data/results/previsao_tendencias.csv` e no relatório txt.

---

## 2. Variáveis Coletadas e Geradas

Durante a vida do pipeline, os seguintes DataPoints principais são mapeados:

### Váriaveis Coletadas (Skoob)
- `book_id` (Inteiro): Identificador do livro no Skoob.
- `titulo` (String): Título oficial na página.
- `autor` (String): Autor extraído das tags ou HTML de dados.
- `generos` (String Multivalorada): Extraído do Flight Data Javascript (Next.js).
- `nota_media` (Float): Nota geral da rede para o livro (1.0 a 5.0).
- `total_avaliacoes` (Inteiro): Total de reviews simples.
- `total_leitores` (Inteiro): Soma de usuários que marcaram (Leu, Lendo, Quero Ler).
- `total_resenhas` (Inteiro): Total de contribuições textuais gravadas.
- `resenha_texto` (String): O conteúdo do tipo resenha escrito pelos leitores (públicas).
- `resenha_data` (Data): Registra o dia/hora que o usuário redigiu a resenha.

### Variáveis Tratadas e Calculadas Matematicamente
- `sentimento_score` (Float [-1 a 1]): Calculado via Léxicos NLP provando se a resenha teve um tom agressivo/insatisfeito (-x) ou positivo (+x).
- `engajamento_score` (Float): Razão combinada entre ($Resenhas + Avaliações) / Leitores Totais$. Mede se os leitores apenas clicam ou se realmente sentem a necessidade falar sobre o livro.
- `popularidade_score` (Float): O produto entre `nota_media`, `total_leitores` e `engajamento`. É a "medida métrica de Força Base do Gênero".
- `popularidade_prevista` (Float): O Output puro do modelo de Machine Learning acusando o tamanho da popularidade latente para um futuro próximo (após detecção de anomalias na Base Oculta de Features).

---

## 3. Legitimidade da Coleta de Dados (Web Scraping Legal)

A legalidade do processo de *Web Scraping* realizado por este script obedece a limites técnicos e normas éticas estritas de computação, adequando-se perfeitamente a uma pesquisa acadêmica legítima (TCC), embasada nos seguintes pilares:

### 1. Ausência de Quebra de Segurança / Dados Públicos
O script interage exclusivamente com a interface pública do Skoob. Não há injeção SQL, invasão, quebra de senhas (brute forcing) ou uso de credenciais indevidas. Ele consome unicamente os dados aos quais qualquer navegador de usuário normal tem acesso quando visita um perfil de livro de forma logoff (sem login na plataforma). No Brasil, o entendimento da Lei Geral de Proteção de Dados (LGPD) e o Direito Digital apontam que dados deliberadamente abertos e não sensíveis podem ser catalogados dentro de bom senso, especialmente se dissociados de PII (Informações de Identificação Pessoal).

### 2. Anonimização Total (Estatística Agregada)
Em nenhum momento a coleta armazena dados de Identificação de Usuários (nomes de usuário, e-mails ou fotos que deixaram resenhas). A pesquisa opera no espectro Agregado, o que extingue o conflito com a LGPD (Lei 13.709/18).

### 3. Rate Limiting Ético Constritivo
Diferente de ataques "DDoS", o código em `config.py` e `skoob_client.py` instrui expressamente o software a:
- Implementar `REQUEST_DELAY_MIN = 0.5` a `1.5` segundos por instância.
- Assumir comportamentos limpos via *User-Agent* padronizados evitando burlar Cloudflares.
- Conduzir uma carga concorrida de no máximo `MAX_WORKERS = 5`. 
As políticas protegem integralmente a infraestrutura de servidores da plataforma alvo, não afetando sua disponibilidade nem onerando suas bases.

### Conclusão Legal
Visto que a finalidade desta pesquisa é estritamente **Acadêmica e Não-Comercial** (Citar § Artigos de Educação Constitucional ou uso "Fair Use"), processada com limitação agressiva de requests e calcada sob anonimização Pessoal, o *pipeline* da coleta de dados constitui um estudo exploratório plenamente documentável e legítimo.
