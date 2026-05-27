# Resultados e Análise — Previsão de Tendências Literárias

**Previsão de Tendências Literárias com Dados do Skoob**
IFTM — Engenharia de Computação | Gabriel Paredes Ferreira | 2026

---

## 1. Visão Geral do Projeto

Este projeto utiliza um **dataset público (Public Domain License)** extraído do Skoob — a maior rede social de leitores do Brasil — contendo dados de ~12.000 livros para responder a duas perguntas centrais:

1. **É possível prever a popularidade de um livro com base em suas características?**
2. **Quais gêneros literários apresentaram maior crescimento entre 2000 e 2020?**

> **Nota sobre a fonte de dados:** O projeto utiliza um dataset com licença Public Domain, tornando a abordagem **eticamente rigorosa e academicamente válida**, em plena conformidade com os Termos de Uso do Skoob.

---

## 2. Dataset

| Atributo | Valor |
|---|---|
| **Fonte** | Dataset Público do Skoob — Public Domain License |
| **Volume bruto** | ~12.000 livros |
| **Após limpeza** | **9.742 livros** |
| **Período coberto** | Publicações de 1900 a 2020 |
| **Janela de análise temporal** | 2000–2020 |
| **Idioma predominante** | Português (98,8%) |

### Colunas principais utilizadas:
- `leram` — nº de usuários que concluíram o livro
- `avaliacao` — nº de avaliações na plataforma
- `resenha` — nº de resenhas escritas
- `rating` — nota média (0–5)
- `genero` — categorias literárias (até 5 por livro)
- `paginas`, `ano`, `editora` — metadados estruturais

---

## 3. Variável Alvo — Score de Popularidade

O modelo prediz um **Score de Popularidade composto** (escala 0–1), que combina três dimensões de engajamento do leitor:

```
popularidade_score = 0.5 × norm(leram) + 0.3 × norm(avaliacao) + 0.2 × norm(resenha)
```

| Componente | Peso | Justificativa |
|---|---|---|
| `leram` | 50% | Alcance — quantas pessoas efetivamente concluíram o livro |
| `avaliacao` | 30% | Engajamento ativo — quem avaliou o livro |
| `resenha` | 20% | Engajamento qualitativo — quem escreveu sobre o livro |

### Distribuição do Score

| Percentil | Score | Interpretação |
|---|---|---|
| 50% (mediana) | 0.038 | Metade dos livros tem score abaixo disso |
| 75% | 0.145 | Top 25% |
| 90% | 0.311 | Top 10% |
| 95% | 0.458 | Top 5% |
| 99% | 0.652 | Top 1% |
| Máximo | 0.815 | Bestsellers absolutos |

A distribuição é fortemente assimétrica à direita — a grande maioria dos livros tem popularidade baixa, e apenas ~1% atinge score acima de 0.65. Isso reflete fielmente a realidade do mercado editorial, onde poucos títulos concentram a maior parte dos leitores.

---

## 4. Features do Modelo (30 features)

| Categoria | Features | Nº |
|---|---|---|
| **Numéricas** | páginas, ano, rating, % leitoras mulheres, taxa de abandono, taxa de conclusão, razão desejo/leitores, tamanho da descrição | 8 |
| **Gêneros (OHE)** | Top 20 gêneros como features binárias (0/1) | 20 |
| **Editora** | Label encoding — top 10 editoras + "Outras" | 1 |
| **NLP** | Score de sentimento da descrição | 1 |

### Holdout Temporal

Em vez de um split aleatório, o conjunto de dados é dividido por data de publicação:

```
Treino: livros publicados até 2017  →  8.021 amostras (82%)
Teste : livros publicados em 2018–2020  →  1.721 amostras (18%)
```

Essa abordagem simula o cenário real de uso: o modelo é treinado com dados históricos e avaliado em livros que nunca viu — metodologicamente mais rigoroso para dados com dimensão temporal.

---

## 5. Resultados dos Modelos de Machine Learning

| Modelo | MAE | RMSE | R² | Interpretação |
|---|---|---|---|---|
| **Random Forest** ← **Melhor** | **0.0591** | **0.0938** | **0.6986** | Explica ~70% da variância |
| Árvore de Decisão | 0.0595 | 0.0956 | 0.6870 | Muito próximo do RF |
| Regressão Linear | 0.1327 | 0.1640 | 0.0785 | Relação não é linear |

### Interpretação das Métricas

**MAE = 0.0591:** Em média, o modelo erra ±0.059 no score de popularidade (escala 0–1). Dado que 75% dos livros têm score < 0.145, este erro é proporcionalmente pequeno.

**R² = 0.70:** O modelo explica **70% da variância** do score de popularidade com as features disponíveis. Os 30% restantes são atribuídos a fatores externos ao dataset — marketing editorial, adaptações para cinema/TV, fenômenos virais em redes sociais — que não são capturáveis com os dados disponíveis.

**Regressão Linear R² = 0.08:** Este resultado confirma a escolha por modelos não-lineares. A popularidade literária não apresenta relação linear simples com as características do livro.

### Por que o Random Forest vence?

O Random Forest combina múltiplas Árvores de Decisão e reduz o overfitting por meio da técnica de *bagging*. Como a relação entre features e popularidade envolve interações não-lineares (ex: gênero × ano × editora), o ensemble captura essas interações melhor que modelos mais simples.

---

## 6. Previsão de Livro Hipotético

O modelo treinado pode receber as características de um livro e estimar seu score de popularidade. Exemplo:

```
Gênero: Romance    |  páginas: 350  |  ano: 2024  |  rating esperado: 4.5  |  Editora: Rocco

  Random Forest  →  Score: 0.0726  |  Percentil ~63%  |  Acima da mediana
  Árvore de Decisão  →  Score: 0.0691  |  Percentil ~62%

  Score médio previsto: 0.071  →  Livro no top 37% do dataset
```

Um score de 0.07 representa o **percentil 63%** do dataset — o modelo prevê que o livro superará 63% de todos os livros do Skoob em popularidade. Scores acima de 0.65 (top 1%) correspondem apenas a bestsellers absolutos.

---

## 7. Análise Temporal de Gêneros (2000→2020)

### Classificação dos 352 gêneros analisados

| Categoria | Quantidade | Critério |
|---|---|---|
| 🟢 Ascensão | 60 | Crescimento ≥ 50% entre os períodos |
| 🔵 Emergente | 221 | Surgiu apenas no período tardio (2013–2020) |
| ⚪ Estagnação | 2 | Variação entre -20% e +50% |
| 🔴 Declínio | 69 | Queda > 20% |

### Gêneros em Ascensão (Top resultados)

| Gênero | Crescimento de Leitores | Crescimento % | Tendência |
|---|---|---|---|
| **Jovem Adulto** | +82.967 | **+14.581%** | 🟢 Ascensão explosiva |
| **Fantasia** | +67.130 | **+4.889%** | 🟢 Ascensão forte |
| **Não-Ficção** | +42.431 | **+4.041%** | 🟢 Ascensão forte |
| **Ficção Científica** | +21.958 | **+1.646%** | 🟢 Ascensão |
| **Romance** | +199.180 | **+1.922%** | 🟢 Maior volume absoluto |

### Contextualização dos resultados

**Jovem Adulto +14.581%** é justificado historicamente:
- 2012: Jogos Vorazes / Divergente no auge no Brasil
- 2014: A Culpa é das Estrelas (best-seller mundial)
- 2016–2019: consolidação do segmento YA nas editoras brasileiras

**Fantasia +4.889%:** Crescimento da série Game of Thrones (HBO, 2011–2019) e expansão das editoras Galera/Planeta no segmento.

**Não-Ficção +4.041%:** Crescimento do interesse em autoconhecimento, negócios e comportamento pós-2015.

---

*Documentação gerada em: Maio/2026 | IFTM — Engenharia de Computação*
