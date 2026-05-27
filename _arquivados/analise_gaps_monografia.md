# ANÁLISE DE LACUNAS DA MONOGRAFIA
**TCC: Previsão de Tendências Literárias com Base em Dados do Skoob**
**Gabriel Paredes Ferreira | IFTM | 2026**

> Esta análise cruza o conteúdo atual do seu DOCX com tudo o que foi efetivamente implementado e validado no projeto Python. Para cada lacuna, indico: o que falta, onde inserir na monografia, e quais evidências (prints, arquivos, dados) usar.

---

## 📋 ESTRUTURA ATUAL (O QUE JÁ EXISTE)

| Capítulo | Status |
|----------|--------|
| 1 – Introdução | ✅ Redigido (bom) |
| 1.1 – Objetivo Geral | ✅ Redigido |
| 1.2 – Objetivos Específicos | ✅ Redigido (6 objetivos) |
| 2 – Referencial Teórico | ✅ Redigido (3 seções) |
| 3 – Metodologia | ⚠️ Parcial – escrito no futuro, não no passado |
| 4 – Recursos | ⚠️ Muito raso |
| 5 – Cronograma | ❌ Vazio |
| **6 – Resultados e Discussão** | ❌ **COMPLETAMENTE AUSENTE** |
| **7 – Conclusão** | ❌ **COMPLETAMENTE AUSENTE** |
| Referências | ✅ Básico OK |

---

## 🚨 LACUNA CRÍTICA #1: O CAPÍTULO DE RESULTADOS (INEXISTENTE)

Este é o maior gap. Toda a execução do projeto — coleta real de dados, treinamento dos modelos, os erros encontrados, as métricas finais, e a validação empírica de 30 dias — não está documentada em lugar nenhum da monografia.

### O que escrever: Capítulo 4 – RESULTADOS E DISCUSSÃO

**Seção 4.1 – Coleta de Dados Realizada**
- Descreva que foram coletados **5.000 livros** do Skoob em execução real via Web Scraping concorrente com `ThreadPoolExecutor` (5 workers).
- Mencione o Rate Limiting ético (0,5 a 1,5 segundos por requisição) implementado.
- Mencione que os dados foram salvos em `data/raw/books_raw.csv`.
- 📌 **EVIDÊNCIA:** Print do terminal mostrando o progresso da coleta (a barra de progresso do scraper rodando).

**Seção 4.2 – Tratamento dos Dados**
- Descreva a limpeza realizada (deduplicação, remoção de nulos, normalização de strings de gênero).
- Descreva o enriquecimento: criação das métricas derivadas (`media_engajamento`, `media_aceitacao`, `porcentagem_leitores`, `ranking_popularidade`).
- Descreva o processamento NLP das resenhas (tokenização, remoção de stopwords, classificação de sentimento entre -1 e +1).
- 📌 **EVIDÊNCIA:** Tabela dos primeiros registros de `data/processed/genre_trends.csv` (você pode mostrar as primeiras 10 linhas com as colunas de engajamento, popularidade e sentimento).

**Seção 4.3 – Análise Descritiva**
- Apresente os gráficos gerados e o que eles revelam.
- 📌 **EVIDÊNCIAS (todos em `data/results/`):**
  - `generos_popularidade.png` → Gráfico de barras dos gêneros mais populares
  - `distribuicao_notas.png` → Prova que as notas têm viés positivo (média > 3.5)
  - `engajamento_por_genero.png` → Gêneros de nicho podem ter público mais fiel
  - `heatmap_correlacoes.png` → Correlações entre as features (essencial para justificar o Feature Engineering)
  - `notas_por_genero.png` e `popularidade_vs_nota.png` → Análises visuais complementares

**Seção 4.4 – Modelagem e Treinamento**
- Explique o Feature Engineering: quais colunas foram usadas como Features (X) e qual era o Target (y = `ranking_popularidade`).
- Explique a divisão treino/teste (80/20) e a Validação Cruzada (k-fold = 5).
- **⚠️ PONTO CRUCIAL:** Documente o erro de **Data Leakage** encontrado e como foi corrigido (veja seção abaixo).

**Seção 4.5 – Resultados da Comparação de Modelos**
Apresente a tabela de resultados dos 3 modelos:

| Modelo | MAE | RMSE | R² | CV R² Médio |
|---|---|---|---|---|
| Árvore de Decisão | 0.0349 | — | — | 0.9706 |
| Random Forest | 0.0367 | — | — | 0.9615 |
| Regressão Linear | 0.0497 | — | — | 0.9345 |

- 📌 **EVIDÊNCIAS:**
  - `data/results/comparacao_modelos.png` → Gráfico comparativo dos modelos
  - `data/results/resultados_modelos.csv` → CSV com as métricas completas
  - `data/results/predicao_vs_real_random_forest.png` → Gráfico Previsto × Real do vencedor

**Seção 4.6 – Validação Empírica (Backtesting)**
Este é o "pulo do gato" mais impressionante do seu TCC.
- Explique que as previsões geradas em **12/03/2026** foram arquivadas em `data/results/history/`.
- Explique que um mês depois (**12/04/2026**), o pipeline foi executado novamente coletando os dados reais do Skoob.
- Apresente os resultados do confronto:

**Métricas finais do Backtest:**
- Previsão feita em: 12/03/2026
- Realidade medida em: 12/04/2026
- **MAE de Posição:** 12,52 posições (num universo de 320 gêneros)
- **Precisão Top 5:** 5 de 5 (100%) ← Os 5 gêneros mais em alta foram previstos corretamente
- Acertos exatos de posição: 25 / 320

**Tabela Top 10 do Backtest:**

| Gênero | Rank Previsto | Rank Real | Status |
|---|---|---|---|
| Distopia | 1 | 1 | ✅ Exato |
| Fábula | 2 | 2 | ✅ Exato |
| Culinária E Gastronomia | 4 | 3 | ⚡ Quase (+1) |
| Animais De Estimação | 3 | 4 | ⚡ Quase (-1) |
| Ficção | 5 | 5 | ✅ Exato |
| Fantasia | 6 | 6 | ✅ Exato |
| Jovem Adulto | 8 | 7 | ⚡ Quase (+1) |
| Literatura Estrangeira | 7 | 8 | ⚡ Quase (-1) |
| Romance | 11 | 9 | ⚡ Quase (+2) |
| Drama | 9 | 10 | ⚡ Quase (-1) |

- 📌 **EVIDÊNCIA:** Arquivo `data/results/comparacao_previsao_2026-03-12_vs_2026-04-12.txt` — cole a tabela completa como Apêndice ou Figura na monografia.

---

## 🚨 LACUNA CRÍTICA #2: O DATA LEAKAGE DEVE SER DOCUMENTADO

Este é um dos pontos mais ricos do TCC do ponto de vista acadêmico — você DEVE documentar isso.

### Onde inserir: Seção 4.4 da Metodologia/Resultados

**O que ocorreu:**
No primeiro ciclo de treinamento, o modelo de Regressão Linear apresentou R² = 1.0000 e MAE = 0.0000 — resultados impossíveis na realidade. A investigação revelou um **vazamento de dados (*Data Leakage*)**: variáveis derivadas da popularidade (`popularidade_media`, `popularidade_std`) estavam incluídas nas *Features* de entrada, apesar de serem diretamente correlacionadas ao *Target* (`ranking_popularidade`). O modelo literalmente "recebia a resposta antes da prova".

**Como foi corrigido:**
As colunas vazantes foram identificadas e explicitamente removidas do array de *Features* nos módulos `models/feature_engineering.py` e `models/predict.py`. Após a correção, os modelos passaram a competir de forma justa.

**Resultado antes → depois:**
| Modelo | R² (com leakage) | R² (corrigido) |
|---|---|---|
| Regressão Linear | 1.0000 ← inválido | 0.9345 |
| Árvore de Decisão | — | 0.9706 |
| Random Forest | — | 0.9615 |

---

## ⚠️ LACUNA #3: CAPÍTULO DE CONCLUSÃO (INEXISTENTE)

### O que escrever: Capítulo 5 – CONCLUSÃO

Estrutura recomendada:

1. **Retomada dos objetivos:** Mostre como cada um dos 6 objetivos específicos foi alcançado (coleta ✅, tratamento ✅, análise descritiva ✅, modelagem ✅, validação ✅, implicações práticas ✅).

2. **Principal resultado:** O sistema de ML foi capaz de prever, com 100% de precisão, os 5 gêneros literários que estariam em alta no mês seguinte, com um erro médio de posicionamento de apenas 12,52 posições em um ecossistema de 320 gêneros.

3. **Limitações:**
   - O modelo opera sobre Janela de Curto Prazo (~30 dias), pois o Web Scraping extrai apenas o estado presente da plataforma.
   - A ausência de um banco de dados histórico longitudinal impede o uso de Séries Temporais (LSTM/ARIMA), que seriam mais precisas para previsões trimestrais.
   - O Skoob não disponibiliza API oficial, tornando a coleta dependente de engenharia reversa da estrutura HTML.

4. **Trabalhos futuros:**
   - Implementar coleta agendada mensal (cron jobs) para construir base histórica longitudinal.
   - Aplicar modelos de Séries Temporais (LSTM) com dados de 12+ meses.
   - Usar LLMs fine-tuned em português (ex: BERTimbau) para análise de sentimento mais precisa nas resenhas.
   - Integrar com banco de dados em nuvem (AWS/GCP) para operação contínua.

---

## ⚠️ LACUNA #4: METODOLOGIA ESCRITA NO FUTURO (DEVE SER ATUALIZADA)

O Capítulo 3 atual usa linguagem de **proposta** (*"será realizada"*, *"será utilizado"*, *"passarão por"*). Como o projeto **já foi executado**, você deve reescrever no passado:

**Exemplos de correções necessárias:**
- *"A coleta será realizada por meio de web scraping"* → *"A coleta foi realizada por meio de web scraping"*
- *"Serão utilizados scripts em Python"* → *"Foram utilizados scripts em Python"*
- *"Os dados coletados passarão por limpeza"* → *"Os dados coletados passaram por limpeza"*

Além disso, adicione informações que concretizem o que foi feito:
- Coleta paralela com 5 threads (`ThreadPoolExecutor`)
- Volume exato: **5.000 livros**
- Gerou **3 arquivos de processado**: `books_cleaned.csv`, `books_enriched.csv`, `genre_trends.csv`
- **322 gêneros literários** identificados e ranqueados

---

## ⚠️ LACUNA #5: REFERENCIAL TEÓRICO INCOMPLETO

O referencial cobre Big Data e Skoob, mas **não menciona** os conceitos centrais implementados:

| Conceito faltante | Onde inserir |
|---|---|
| Web Scraping, BeautifulSoup e ética da raspagem | Seção 2.2 ou nova seção 2.4 |
| Aprendizado de Máquina Supervisionado | Nova seção 2.4 ou 2.5 |
| Regressão Linear, Árvores de Decisão, Random Forest | Nova seção 2.5 |
| Métricas de avaliação: MAE, RMSE, R² | Nova seção 2.5 |
| Backtesting como técnica de validação | Nova seção 2.6 |

---

## 📌 RESUMO: PRIORIDADE DE AÇÃO

| Prioridade | Ação | Capítulo |
|---|---|---|
| 🔴 Urgente | Escrever capítulo de **Resultados e Discussão** | Cap. 4 |
| 🔴 Urgente | Escrever capítulo de **Conclusão** | Cap. 5 |
| 🟠 Alta | Documentar o **Data Leakage** e sua correção | Seção 4.4 |
| 🟠 Alta | Documentar o **Backtest com tabela e métricas** | Seção 4.6 |
| 🟡 Média | Reescrever Metodologia do futuro para o **passado** | Cap. 3 |
| 🟡 Média | Adicionar referencial teórico sobre **ML e métricas** | Cap. 2 |
| 🟢 Baixa | Completar **Cronograma** (Cap. 5 atual) | Cap. 5 |
| 🟢 Baixa | Adicionar imagens e gráficos ao longo do texto | Todos |

---

## 📁 EVIDÊNCIAS DISPONÍVEIS EM `data/results/`

| Arquivo | Usar em |
|---|---|
| `generos_popularidade.png` | Seção 4.3 – Análise Descritiva |
| `distribuicao_notas.png` | Seção 4.3 |
| `engajamento_por_genero.png` | Seção 4.3 |
| `heatmap_correlacoes.png` | Seção 4.4 – Feature Engineering |
| `comparacao_modelos.png` | Seção 4.5 – Comparação de Modelos |
| `predicao_vs_real_random_forest.png` | Seção 4.5 |
| `resultados_modelos.csv` | Tabela na Seção 4.5 |
| `comparacao_previsao_2026-03-12_vs_2026-04-12.txt` | Seção 4.6 – Backtest |
| `relatorio_previsao_2026-03-12.txt` | Seção 4.6 como contexto |
