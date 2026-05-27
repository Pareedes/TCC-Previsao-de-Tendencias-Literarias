# Relatório Técnico: Previsão de Tendências Literárias (Skoob)

Este documento detalha o funcionamento técnico completo do pipeline de dados — o fluxo de execução, as variáveis utilizadas, a metodologia de modelagem e as decisões de engenharia que fundamentam o projeto.

---

## 1. Fonte de Dados

O projeto utiliza um **dataset público com licença Public Domain** contendo dados do Skoob — a maior rede social para leitores do Brasil. O dataset está armazenado localmente em `dados.csv` (raiz do projeto) e **não requer nenhuma etapa de coleta online**.

| Atributo | Valor |
|---|---|
| **Licença** | Public Domain |
| **Volume bruto** | ~12.000 livros |
| **Após limpeza** | 9.742 livros |
| **Período coberto** | Publicações de 1900 a 2020 |
| **Janela de análise temporal** | 2000–2020 |

Esta escolha metodológica garante **reprodutibilidade total** do projeto: qualquer pessoa com o arquivo `dados.csv` pode executar o pipeline do zero e obter os mesmos resultados.

---

## 2. Pipeline End-to-End

O sistema é orquestrado pelo `main.py`, que expõe tanto um menu interativo quanto uma interface de linha de comando (CLI). O pipeline opera em 5 etapas lineares:

### Etapa 1 — Carga e Limpeza do Dataset (`processing/data_loader.py`)

1. O módulo `DataLoader` lê o arquivo `dados.csv` com `pandas`.
2. São removidos registros sem informações essenciais (`titulo`, `genero`, `leram`).
3. Colunas numéricas são padronizadas (tipos corretos, tratamento de nulos com medianas).
4. A coluna `genero` — que contém múltiplos gêneros separados por vírgula — é explodida em listas normalizadas.
5. O dataset limpo é salvo em `data/processed/books_clean.csv` (**9.742 livros**).

### Etapa 2 — Processamento NLP das Descrições (`processing/nlp_processor.py`)

1. O módulo `NLPProcessor` atua na coluna `descricao` (sinopses dos livros) usando a biblioteca `nltk`.
2. Os textos são tokenizados e stopwords do português são removidas.
3. É calculado um **score de sentimento** por descrição (escala −1 a +1), que depois é usado como uma das features do modelo de ML.
4. São extraídas as **palavras-chave** mais relevantes de cada sinopse.
5. Resultados salvos em `data/processed/nlp_descricoes.csv`.

> **Nota:** Esta etapa é opcional. Caso a coluna `descricao` esteja ausente ou vazia, o pipeline continua normalmente sem a feature de sentimento.

### Etapa 3 — Análise Descritiva e Visualizações (`analysis/visualizations.py`)

1. O módulo `Visualizations` gera automaticamente o conjunto completo de gráficos do projeto usando `matplotlib` e `seaborn`.
2. Principais gráficos gerados:

| Arquivo | Conteúdo |
|---|---|
| `distribuicao_score_popularidade.png` | Distribuição assimétrica do score alvo |
| `top15_livros_popularidade.png` | Validação qualitativa: os 15 livros com maior score |
| `heatmap_correlacoes.png` | Correlações de Spearman entre variáveis numéricas |
| `comparacao_modelos_defesa.png` | Comparação de MAE, RMSE e R² entre os três modelos |
| `predicao_vs_real_random_forest.png` | Predição × valor real do melhor modelo |
| `feature_importance.png` | Importância relativa das features no Random Forest |

3. Todos os gráficos são salvos em `data/results/`.

### Etapa 4 — Análise Temporal de Gêneros (`analysis/genre_trends.py`)

1. O módulo `GenreTrendAnalyzer` filtra o dataset para o período **2000–2020** e agrega leitores por gênero e ano.
2. Cada gênero é classificado em uma de quatro categorias de tendência:

| Categoria | Critério |
|---|---|
| 🟢 **Ascensão** | Crescimento ≥ 50% entre os períodos inicial e final |
| 🔵 **Emergente** | Surgiu apenas no período 2013–2020 (ausente no início) |
| ⚪ **Estagnação** | Variação entre −20% e +50% |
| 🔴 **Declínio** | Queda > 20% |

3. São gerados os gráficos de linha (`evolucao_generos_defesa.png`), heatmap (`heatmap_genero_ano.png`) e barras de crescimento (`crescimento_generos.png`).
4. Resultados salvos em `data/processed/genre_trends_temporal.csv`.

### Etapa 5 — Treinamento e Avaliação dos Modelos de ML

#### 5a. Engenharia de Features (`models/feature_engineering.py`)

A classe `FeatureEngineer` constrói um vetor de **30 features** por livro:

| Categoria | Features | Quantidade |
|---|---|---|
| **Numéricas** | `paginas`, `ano`, `rating`, % leitoras mulheres, taxa de abandono, taxa de conclusão, razão desejo/leitores, tamanho da descrição | 8 |
| **Gêneros (OHE)** | Top 20 gêneros como colunas binárias (0 ou 1) | 20 |
| **Editora** | Label encoding — top 10 editoras + "Outras" | 1 |
| **NLP** | Score de sentimento da sinopse | 1 |

**Holdout Temporal** — em vez de um split aleatório, o conjunto é dividido por data de publicação:

```
Treino: livros publicados até 2017  →  8.021 amostras (82%)
Teste : livros publicados em 2018–2020  →  1.721 amostras (18%)
```

Essa abordagem simula o uso real do modelo: treinado com histórico, testado em livros que ele nunca viu — metodologicamente mais rigoroso para dados com dimensão temporal.

#### 5b. Variável Alvo — Score de Popularidade

O modelo não prediz a nota do livro, mas sim um **Score de Popularidade composto** (escala 0–1) que captura três dimensões de engajamento do leitor:

```
popularidade_score = 0.5 × norm(leram) + 0.3 × norm(avaliacao) + 0.2 × norm(resenha)
```

| Componente | Peso | Justificativa |
|---|---|---|
| `leram` | 50% | Alcance — quantas pessoas efetivamente concluíram o livro |
| `avaliacao` | 30% | Engajamento ativo — quem se importou o suficiente para avaliar |
| `resenha` | 20% | Engajamento qualitativo — quem escreveu sobre o livro |

#### 5c. Treinamento (`models/train.py`)

Três modelos de regressão supervisionada competem sobre o mesmo conjunto de treino:

- **Regressão Linear Múltipla** — baseline linear
- **Árvore de Decisão** (Decision Tree Regressor)
- **Random Forest Regressor** — ensemble de múltiplas árvores com bagging

Todos os modelos treinados são persistidos em `data/models/` via `joblib`.

#### 5d. Avaliação (`models/evaluate.py`)

O `ModelEvaluator` avalia os modelos no conjunto de teste usando:

| Métrica | Descrição |
|---|---|
| **MAE** | Erro médio absoluto na escala do score (0–1) |
| **RMSE** | Raiz do erro quadrático médio — penaliza erros grandes |
| **R²** | Coeficiente de determinação — % da variância explicada |
| **CV R² (5-fold)** | R² médio em validação cruzada — robustez do modelo |

---

## 3. Resultados

### Modelos de Machine Learning

| Modelo | MAE | RMSE | R² | CV R² |
|---|---|---|---|---|
| **Random Forest** ← melhor | **0.0591** | **0.0938** | **0.6986** | — |
| Árvore de Decisão | 0.0595 | 0.0956 | 0.6870 | — |
| Regressão Linear | 0.1327 | 0.1640 | 0.0785 | — |

O **Random Forest** alcançou R² = 0.70, explicando aproximadamente 70% da variância do score de popularidade. O R² baixo da Regressão Linear (0.08) confirma empiricamente que a relação entre as características do livro e sua popularidade é **não-linear** — validando a escolha dos modelos baseados em árvores.

Os 30% de variância não explicados pelo melhor modelo correspondem a fatores externos ao dataset: ações de marketing editorial, adaptações para cinema/TV e fenômenos virais em redes sociais — fatores genuinamente impossíveis de capturar com os dados disponíveis.

### Análise Temporal de Gêneros (2000–2020)

| Categoria | Quantidade de Gêneros |
|---|---|
| 🟢 Ascensão | 60 |
| 🔵 Emergente | 221 |
| ⚪ Estagnação | 2 |
| 🔴 Declínio | 69 |

**Principais gêneros em ascensão:**

| Gênero | Crescimento de Leitores | Crescimento % |
|---|---|---|
| Jovem Adulto | +82.967 | +14.581% |
| Fantasia | +67.130 | +4.889% |
| Não-Ficção | +42.431 | +4.041% |
| Ficção Científica | +21.958 | +1.646% |
| Romance | +199.180 | +1.922% |

---

## 4. Reprodutibilidade

Todos os parâmetros do pipeline são centralizados em `config.py`:

| Parâmetro | Valor | Descrição |
|---|---|---|
| `RANDOM_STATE` | 42 | Seed global para reprodutibilidade |
| `CV_FOLDS` | 5 | Folds para validação cruzada |
| `ANO_CORTE_TREINO` | 2017 | Divisão temporal treino/teste |
| `ANO_INICIO_ANALISE` | 2000 | Início da análise de gêneros |
| `ANO_FIM_ANALISE` | 2020 | Fim da análise de gêneros |
| `PESO_LERAM` | 0.5 | Peso do score de popularidade |
| `TOP_GENEROS_FEATURES` | 20 | Nº de gêneros usados como features |
| `TOP_EDITORAS_FEATURES` | 10 | Nº de editoras categorizadas |

---

*IFTM — Campus Avançado Uberaba Parque Tecnológico | Engenharia de Computação | TCC 2026*
*Gabriel Paredes Ferreira*
