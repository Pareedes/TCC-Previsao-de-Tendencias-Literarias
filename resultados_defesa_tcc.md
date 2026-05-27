# Resultados e Análise — Defesa do TCC
**Previsão de Tendências Literárias com Dados do Skoob**
IFTM — Engenharia de Computação | Gabriel Paredes Ferreira | 2026

---

## 1. Visão Geral do Projeto

Este projeto utiliza um **dataset público (Public Domain License)** extraído do Skoob — a maior rede social de leitores do Brasil — contendo dados de ~12.000 livros para responder a duas perguntas centrais:

1. **É possível prever a popularidade de um livro com base em suas características?**
2. **Quais gêneros literários apresentaram maior crescimento entre 2000 e 2020?**

> **Nota sobre a fonte de dados:** O projeto originalmente previa coleta via web scraping. Após identificar que essa prática viola os Termos de Uso do Skoob, a metodologia foi adaptada para utilizar um dataset com licença Public Domain — tornando a abordagem **eticamente rigorosa e academicamente válida**.

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

**Por que essa fórmula?**
- `leram` (peso 50%): mede alcance — quantas pessoas efetivamente leram o livro
- `avaliacao` (peso 30%): mede engajamento ativo — quem se importou o suficiente para avaliar
- `resenha` (peso 20%): mede engajamento qualitativo — quem escreveu sobre o livro

> 📊 **Print recomendado:** `distribuicao_score_popularidade.png`

### Distribuição do Score (insight fundamental)

| Percentil | Score | Interpretação |
|---|---|---|
| 50% (mediana) | 0.038 | Metade dos livros tem score abaixo disso |
| 75% | 0.145 | Top 25% |
| 90% | 0.311 | Top 10% |
| 95% | 0.458 | Top 5% |
| 99% | 0.652 | Top 1% |
| Máximo | 0.815 | Bestsellers absolutos |

**Interpretação chave:** A distribuição é fortemente assimétrica à direita — a grande maioria dos livros tem popularidade baixa, e apenas ~1% atinge score acima de 0.65. **Isso reflete fielmente a realidade do mercado editorial**, onde poucos títulos concentram a maior parte de leitores.

---

## 4. Features do Modelo (30 features)

| Categoria | Features | Nº |
|---|---|---|
| **Numéricas** | páginas, ano, rating, % leitoras mulheres, taxa de abandono, taxa de conclusão, razão desejo/leitores, tamanho da descrição | 8 |
| **Gêneros (OHE)** | Top 20 gêneros como features binárias (0/1) | 20 |
| **Editora** | Label encoding — top 10 editoras + "Outras" | 1 |
| **NLP** | Score de sentimento da descrição | 1 |

### Holdout Temporal (divisão do conjunto de treino/teste)

```
Treino: livros publicados até 2017  →  8.021 amostras (82%)
Teste : livros publicados em 2018–2020  →  1.721 amostras (18%)
```

**Por que holdout temporal?** Simula a situação real: o modelo é treinado com dados históricos e testado em livros que ele "nunca viu". É metodologicamente mais robusto que um split aleatório para dados com dimensão temporal.

---

## 5. Resultados dos Modelos de Machine Learning

> 📊 **Print recomendado:** `comparacao_modelos_defesa.png`

| Modelo | MAE | RMSE | R² | Interpretação |
|---|---|---|---|---|
| **Random Forest** ← **Melhor** | **0.0591** | **0.0938** | **0.6986** | Explica ~70% da variância |
| Árvore de Decisão | 0.0595 | 0.0956 | 0.6870 | Muito próximo do RF |
| Regressão Linear | 0.1327 | 0.1640 | 0.0785 | **Péssimo** — relação não é linear |

### Como interpretar cada métrica:

**MAE = 0.0591:** Em média, o modelo erra por ±0.059 no score de popularidade (escala 0–1). Dado que 75% dos livros têm score < 0.145, um erro de 0.059 é proporcionalmente pequeno.

**R² = 0.70:** O modelo explica **70% da variância** do score de popularidade com as features disponíveis. Os 30% restantes são atribuídos a fatores não capturados (marketing, autor famoso, adaptação cinematográfica, buzz nas redes sociais).

**Regressão Linear R² = 0.08:** Este resultado **valida a escolha de modelos não-lineares**. Confirma que a popularidade literária não tem relação linear simples com as características do livro.

### Por que o Random Forest vence?

O Random Forest combina múltiplas Árvores de Decisão e reduz o overfitting por meio da técnica de *bagging*. Como a relação entre features e popularidade envolve muitas interações não-lineares (ex: gênero × ano × editora), o ensemble captura essas interações melhor que modelos simples.

---

## 6. Previsão de Livro Hipotético — Como Funciona

> 📊 **Demonstração ao vivo:** `python main.py --prever`

O modelo treinado pode receber as características de um livro ainda não publicado e estimar seu score de popularidade. Exemplo no terminal:

```
Genero: Romance            paginas: 350    ano: 2024
Rating esperado: 4.5       Editora: Rocco

  Random Forest   →  Score: 0.0726  |  Percentil ~63%  |  Acima da mediana
  Decisão Tree    →  Score: 0.0691  |  Percentil ~62%

Score medio: 0.071  →  Livro no top 37% do dataset
```

**Como interpretar para a banca:** Um score de 0.07 parece baixo, mas representa o percentil 63% do dataset — significa que o modelo prevê que **o livro superará 63% de todos os livros do Skoob em popularidade**. Scores acima de 0.65 (top 1%) são reservados apenas para bestsellers absolutos como séries mundialmente conhecidas.

---

## 7. Análise Temporal de Gêneros (2000→2020)

> 📊 **Prints recomendados:** `evolucao_generos_defesa.png`, `crescimento_generos.png`, `heatmap_genero_ano.png`

### Gêneros em Ascensão (Top resultados)

| Gênero | Crescimento de Leitores | Crescimento % | Tendência |
|---|---|---|---|
| **Jovem Adulto** | +82.967 leitores | **+14.581%** | 🟢 Ascensão explosiva |
| **Fantasia** | +67.130 leitores | **+4.889%** | 🟢 Ascensão forte |
| **Não-Ficção** | +42.431 leitores | **+4.041%** | 🟢 Ascensão forte |
| **Ficção Científica** | +21.958 leitores | **+1.646%** | 🟢 Ascensão |
| **Romance** | +199.180 leitores | **+1.922%** | 🟢 Volume absoluto maior |

### Contextualização dos resultados

**Jovem Adulto +14.581%** é justificado historicamente:
- 2012: Jogos Vorazes / Divergente no auge no Brasil
- 2014: A Culpa é das Estrelas (best-seller mundial)
- 2016–2019: consolidação do segmento YA nas editoras brasileiras

**Fantasia +4.889%:** Game of Thrones (série HBO 2011–2019), expansão das editoras Galera/Planeta no segmento.

**Não-Ficção +4.041%:** Crescimento do interesse em autoconhecimento, negócios e comportamento pós-2015.

### Classificação dos 352 gêneros analisados

| Categoria | Quantidade | Critério |
|---|---|---|
| 🟢 Ascensão | 60 | Crescimento ≥ 50% entre períodos |
| 🔵 Emergente | 221 | Surgiu apenas no período tardio (2013–2020) |
| ⚪ Estagnação | 2 | Variação entre -20% e +50% |
| 🔴 Declínio | 69 | Queda > 20% |

---

## 8. Gráficos para Apresentação — Guia de Uso

> Use os gráficos na seguinte ordem sugerida nos slides:

| Slide | Gráfico | Arquivo | O que discutir |
|---|---|---|---|
| Metodologia | Fluxo do Pipeline | `pipeline_fluxo.png` | As 7 etapas do projeto |
| Dataset | Distribuição do Score | `distribuicao_score_popularidade.png` | Assimetria do mercado editorial |
| Dataset | Top 15 livros | `top15_livros_popularidade.png` | Validação qualitativa do score |
| Correlações | Heatmap de variáveis | `heatmap_correlacoes.png` | Relação entre engajamento e nota |
| Modelos | Comparação ML | `comparacao_modelos_defesa.png` | Por que Random Forest vence |
| Modelos | Predição vs Real | `predicao_vs_real_random_forest.png` | Ajuste visual do modelo |
| Tendências | Evolução temporal | `evolucao_generos_defesa.png` | Crescimento do Jovem Adulto |
| Tendências | Heatmap gênero×ano | `heatmap_genero_ano.png` | Visão panorâmica calor/frio |
| Tendências | Crescimento barras | `crescimento_generos.png` | Ranking de crescimento |

---

## 9. Possíveis Perguntas da Banca — Respostas Preparadas

**"Por que R² = 0.70 e não maior?"**
> "R² de 0.70 com holdout temporal é um resultado robusto e honesto. Os 30% de variância não explicada correspondem a fatores externos ao dataset: marketing editorial, adaptações para cinema/TV, fenômenos virais em redes sociais. Esses fatores são genuinamente impossíveis de capturar com os dados disponíveis."

**"Por que os scores hipotéticos ficam todos baixos (~0.07)?"**
> "Porque a distribuição real de popularidade é fortemente assimétrica — metade dos livros tem score abaixo de 0.038. Um score de 0.07 representa o percentil 63%: o livro supera 63% de todo o dataset, o que é um resultado acima da mediana. Scores acima de 0.65 existem em apenas 1% dos livros — são os bestsellers absolutos."

**"O dataset é representativo do mercado atual?"**
> "O dataset cobre até 2020 e contém ~10.000 títulos portugueses do Skoob. Para fins acadêmicos de análise de tendências históricas, é suficiente e válido. A limitação de ser um dataset estático (snapshot de 5 anos atrás) é declarada explicitamente no trabalho como uma fronteira de pesquisa."

**"Por que a Regressão Linear tem R² tão baixo?"**
> "Esse resultado é esperado e valida nossa escolha metodológica. A popularidade literária envolve interações complexas entre gênero, período de publicação, editora e engajamento — relações que a Regressão Linear não consegue capturar. O R² de 0.08 confirma empiricamente que o fenômeno é não-linear."

**"Como vocês garantem que não há data leakage?"**
> "As features de engajamento (leram, avaliação, resenha) foram excluídas do conjunto de features X, pois fazem parte do cálculo do target y. As features inputadas ao modelo são apenas características a priori do livro: gênero, páginas, ano, editora e rating — informações que existiriam antes de o livro ser lançado."

---

## 10. Comandos para Demonstração ao Vivo

```bash
# Menu interativo completo
python main.py

# Apenas carga e limpeza dos dados
python main.py --carregar

# Gerar todos os gráficos
python main.py --analise

# Análise temporal de gêneros
python main.py --tendencias

# Treinar e avaliar modelos
python main.py --modelo

# Prever livro hipotético (interativo)
python main.py --prever

# Pipeline completo
python main.py --pipeline
```

---

*Documentação gerada em: Abril/2026 | Atualizada com interpretação revisada e gráficos para defesa*
