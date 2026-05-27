# Previsão de Tendências Literárias — Skoob Dataset

**Previsão de Tendências Literárias baseado em dados do Skoob**: produto de um Trabalho de Conclusão de Curso (TCC) em Engenharia de Computação, focado em Processamento de Linguagem Natural (NLP), Engenharia de Features e Machine Learning (ML) aplicados ao mercado editorial.

## Objetivo

Analisar e modelar estatisticamente quais **gêneros literários** estão ganhando tração e popularidade no Brasil, e construir um modelo capaz de **prever a popularidade** de um livro com base em suas características. O projeto utiliza um dataset público com licença *Public Domain* contendo dados do Skoob — a maior rede social para leitores do Brasil.

A capacidade preditiva do projeto fornece insights acionáveis para autores, editores e influenciadores do mercado editorial digital.

---

## Fonte de Dados

O projeto utiliza um **dataset público com licença Public Domain** extraído do Skoob, disponível abertamente para fins acadêmicos e de pesquisa. O uso desse dataset é **eticamente rigoroso e academicamente válido**, em plena conformidade com os Termos de Uso da plataforma.

| Atributo                     | Valor                                              |
| ---------------------------- | -------------------------------------------------- |
| **Fonte**              | Dataset Público do Skoob — Public Domain License |
| **Volume bruto**       | ~12.000 livros                                     |
| **Após limpeza**      | 9.742 livros                                       |
| **Período coberto**   | Publicações de 1900 a 2020                       |
| **Janela de análise** | 2000–2020                                         |
| **Arquivo local**      | `dados.csv` (raiz do projeto)                    |

**Principais colunas utilizadas:**

- `leram` — nº de usuários que concluíram o livro
- `avaliacao` — nº de avaliações na plataforma
- `resenha` — nº de resenhas escritas
- `rating` — nota média (0–5)
- `genero` — categorias literárias (até 5 por livro)
- `paginas`, `ano`, `editora` — metadados estruturais
- `descricao` — sinopse do livro (usada no módulo NLP)

---

## O que foi Implementado

Este projeto compõe um **Pipeline de Análise de Dados End-to-End**, organizado em 5 módulos Python e orquestrado por um CLI interativo:

| Módulo                                     | Pacotes                      | Responsabilidade                                                                               |
| ------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------- |
| **`processing/data_loader.py`**     | `pandas`, `re`           | Carga do `dados.csv`, limpeza de nulos, padronização de tipos e geração do dataset limpo |
| **`processing/nlp_processor.py`**   | `nltk`                     | Análise de sentimento e extração de palavras-chave das descrições dos livros              |
| **`models/feature_engineering.py`** | `pandas`, `scikit-learn` | Construção das 30 features do modelo; holdout temporal (treino ≤ 2017 / teste 2018–2020)   |
| **`models/train.py`**               | `scikit-learn`             | Treinamento dos três modelos de ML: Regressão Linear, Árvore de Decisão e Random Forest    |
| **`models/evaluate.py`**            | `scikit-learn`             | Avaliação com MAE, RMSE, R² e validação cruzada (5-fold)                                  |
| **`analysis/genre_trends.py`**      | `pandas`                   | Análise temporal de gêneros no período 2000–2020, com classificação por tendência       |
| **`analysis/visualizations.py`**    | `matplotlib`, `seaborn`  | Geração automática de todos os gráficos do projeto                                         |

### Variável Alvo — Score de Popularidade

O modelo prediz um **Score de Popularidade composto** (escala 0–1) que combina três dimensões de engajamento:

```
popularidade_score = 0.5 × norm(leram) + 0.3 × norm(avaliacao) + 0.2 × norm(resenha)
```

---

## Resultados

### Modelos de Machine Learning

| Modelo                            | MAE              | RMSE             | R²              |
| --------------------------------- | ---------------- | ---------------- | ---------------- |
| **Random Forest** ← melhor | **0.0591** | **0.0938** | **0.6986** |
| Árvore de Decisão               | 0.0595           | 0.0956           | 0.6870           |
| Regressão Linear                 | 0.1327           | 0.1640           | 0.0785           |

O **Random Forest** alcançou R² = 0.70, explicando ~70% da variância do score de popularidade. A Regressão Linear com R² = 0.08 confirma empiricamente que a relação entre características do livro e popularidade é **não-linear**.

### Gêneros em Ascensão (2000–2020)

| Gênero              | Crescimento de Leitores | Crescimento % |
| -------------------- | ----------------------- | ------------- |
| Jovem Adulto         | +82.967                 | +14.581%      |
| Fantasia             | +67.130                 | +4.889%       |
| Não-Ficção        | +42.431                 | +4.041%       |
| Ficção Científica | +21.958                 | +1.646%       |
| Romance              | +199.180                | +1.922%       |

---

## Estrutura do Projeto

```
TCC/
├── dados.csv                    # Dataset público do Skoob (~12.000 livros)
├── main.py                      # Orquestrador: menu interativo + CLI
├── config.py                    # Constantes, caminhos e parâmetros globais
├── requirements.txt             # Dependências do projeto
│
├── processing/
│   ├── data_loader.py           # Carga e limpeza do dataset
│   └── nlp_processor.py        # Sentimento e keywords via NLTK
│
├── models/
│   ├── feature_engineering.py  # Engenharia de features (30 features, holdout temporal)
│   ├── train.py                 # Treinamento dos modelos ML
│   └── evaluate.py              # Métricas e validação cruzada
│
├── analysis/
│   ├── genre_trends.py          # Análise temporal de gêneros
│   └── visualizations.py        # Geração de todos os gráficos
│
└── data/
    ├── processed/               # CSVs intermediários gerados pelo pipeline
    ├── results/                 # Gráficos (.png) e resultados (.csv)
    └── models/                  # Modelos treinados persistidos (.pkl)
```

---

## Como Instalar e Rodar

### Pré-requisitos

- **Python 3.11+**
- Recomenda-se usar `venv` ou Conda para isolamento do ambiente

### 1. Clonar e Instalar Dependências

```bash
git clone https://github.com/SEU-USUARIO/previsao-tendencias-literarias.git
cd previsao-tendencias-literarias
pip install -r requirements.txt
```

### 2. Executar o Menu Interativo

```bash
python main.py
```

Isso abrirá o menu que guia pelas etapas do pipeline:

```
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
```

### 3. Modo CLI (sem menu)

Cada etapa também pode ser chamada diretamente via argumentos:

```bash
# Carregar e limpar o dataset
python main.py --carregar

# Gerar análise descritiva e gráficos
python main.py --analise

# Análise temporal de gêneros
python main.py --tendencias

# Treinar e avaliar os modelos
python main.py --modelo

# Prever popularidade de um livro hipotético (interativo)
python main.py --prever

# Executar todo o pipeline de uma vez
python main.py --pipeline
```

### Outputs Gerados

Todos os arquivos gerados são salvos automaticamente dentro de `data/`:

| Tipo                              | Localização                           |
| --------------------------------- | --------------------------------------- |
| Datasets processados              | `data/processed/`                     |
| Gráficos (`.png`)              | `data/results/`                       |
| Modelos treinados (`.pkl`)      | `data/models/`                        |
| Resultados dos modelos (`.csv`) | `data/results/resultados_modelos.csv` |

---

## Documentação Adicional

- **[`relatorio_tecnico.md`](./relatorio_tecnico.md)** — Relatório técnico completo: engenharia de features, metodologia dos modelos e fluxo de dados
- **[`resultados.md`](./resultados_defesa_tcc.md)** — Resultados detalhados, interpretação das métricas e guia de apresentação para a banca

---

## Autor

**Gabriel Paredes Ferreira**
*IFTM — Campus Avançado Uberaba Parque Tecnológico*
*Curso: Engenharia de Computação | TCC 2026*
