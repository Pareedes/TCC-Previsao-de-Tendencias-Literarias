# ESTRUTURA PARA A TESE DO TCC: ANÁLISES E RESULTADOS OBTIDOS

---

## 4. METODOLOGIA E PROCEDIMENTOS (DADOS E PROCESSAMENTO)

### 4.1. Automação da Coleta (Web Scraping Distribuído)
A extração dos metadados foi formulada em Python utilizando técnicas de programação paralela (`ThreadPoolExecutor` com 5 workers). Para varrer o ecossistema do Skoob, construímos robôs capazes de interceptar não apenas a marcação estática (HTML parsing via BeautifulSoup), mas também strings encadeadas injetadas em componentes Next.js (Flight Data).  
A requisição seguiu protocolos de conformidade ética, estabelecendo limites restritivos de requisições por segundo (Rate Limiting dinâmico de 0,5 a 1,5 segundos) e cabeçalhos não-ameaçadores. Esse cuidado minimizou as chances de recusa do servidor por negação de serviço (DDoS) ou *Cloudflare Captchas*. O volume alvo alcançou 5.000 livros, propiciando um *Data Lake* primário com dezenas de milhares de resenhas textuais.

### 4.2. Processamento e Tratamento (NLP e Enriquecimento)
A segunda fase da arquitetura converteu dados não estruturados brutos em variáveis de inteligência para o modelo matemático. Duas frentes principais foram processadas:
1. **Linguagem Natural (NLP):** Utilizando o módulo `nltk`, implementou-se a normalização dos textos de review (remoção de *stopwords*, pontuação e emojis). Aplicou-se a quantificação da polaridade semântica da resenha (Sentimento Positivo, Negativo e Neutro) atribuindo a elas um peso contínuo entre $-1$ e $1$.
2. **Razões Qualitativas:** Além de totais numéricos (Leitores e Páginas), construímos métricas profundas como a Taxa de Engajamento, Rácio de Aceitação e Popularidade Global (Score Absoluto). Esses cruzamentos evidenciaram se gêneros de nichos pequenos possuem fãs com laços desproporcionalmente leais.

> **📌 PRINT INDICADO 1:** 
> - **Arquivo:** `data/results/distribuicao_notas.png` e `data/results/engajamento_por_genero.png`.
> - **Onde colocar:** Coloque nesta seção para provar à banca como a maioria das notas no Skoob não segue uma distribuição gaussiana (Bell curve) comum, possuindo viés para a positividade.

---

## 5. TREINAMENTO, ERROS E MODELAGEM DE MACHINE LEARNING

### 5.1. Construção Paramétrica (Feature Engineering)
Após a redução de dimensionalidade e enriquecimento da matriz estatística, os campos foram pivotados para compor um array de *Features Numéricas Contínuas*, agregadas em torno do Eixo de **Gêneros Literários**. O projeto objetiva que o classificador aprenda, por indução vetorial supervisionada, como a Popularidade cresce dada as métricas observadas hoje.

### 5.2. Tratamento de Vazamento de Dados (Data Leakage)
*Nota de Defesa: Este é o seu Ponto de Erro/Acerto chave para a Banca.*

Durante as fases iniciais de treinamento, os dados coligidos sofreram uma falha de validação clássica no ambiente do aprendizado de máquina conhecida como **Vazamento de Dados (Data Leakage)**. No primeiro ciclo gerado, o modelo de Regressão Linear apresentou uma métrica impensável: Erro Algébrico nulo ($MAE = 0.00$) e Correlação Exata ($R^2 = 1.0000$). 

Ao investigar as equações sob as quais a rede fora exposta, rastreamos que variáveis intrínsecas ao peso da Popularidade ($popularidade\_media$, $popularidade\_std$) escoaram para a esteira de `Features (X)` da rede de inferência, além da coluna `Target (y)`. Como o Classificador possuía a resposta em mãos antes mesmo da avaliação, o teste se tornou corrompido, e a Regressão Linear anulou qualquer variável externa como o Engajamento e a Análise de Sentimentos.

A remoção completa dessas *leaked features* foi devidamente arquitetada do código de Data Modeling e do Módulo de Predição da Aplicação. Com as colunas isoladas formadas por mais de 20 *features* puras e estritas (ex: `media_leitores_por_livro`, `total_resenhas_soma`, `sentimento_medio`), o aprendizado prosseguiu sadio para a etapa de Benchmarking dos Algoritmos.

> **📌 PRINT INDICADO 2:**
> - **Arquivo:** `data/results/heatmap_correlacoes.png`.
> - **Onde colocar:** Para ilustrar o quão densa era a associação dos vetores de features que sobrou na matriz para a rede preditiva analisar.

### 5.3. Resultados da Otimização Múltipla
O modelo foi exposto a Validação Cruzada (*k-fold de 5 estratos*) com um *split* $80/20$ em três algoritmos paramétricos e ensacados:
1. **Regressão Linear Múltipla** (Modelo de Relação Tradicional Estrita)
2. **Árvore de Decisão** (Decision Tree Regressor)
3. **Floresta Aleatória** (Random Forest Regressor - Método de Ensemble)

Os resultados justos das métricas apontaram que a natureza humana nas mídias sociais segue padrões profundamente não-lineares. Modelos de "Bagging" (Árvore e Florestas Aleatórias) sobressaíram na performance com sobras, garantindo o título e o direito a ser a Carga (Peso) Mestra responsável por prever a tendência global:
- **1º Lugar: Árvore de Decisão** $(R^2 = 0.9706 | MAE = 0.0349)$
- **2º Lugar: Random Forest** $(R^2 = 0.9615 | MAE = 0.0367)$
- **3º Lugar: Regressão Linear** $(R^2 = 0.9345 | MAE = 0.0497)$

> **📌 PRINT INDICADO 3:**
> - **Arquivos:** `data/results/comparacao_modelos.png` e a `Tabela Completa de Resultados do models.txt` (que geramos no prompt ou em CSV).
> - **Onde colocar:** Coloque a Tabela ou Imagem provando aos jurados do seu TCC quais modelos ganharam a disputa!

---

## 6. VALIDAÇÃO EMPÍRICA (BACKTESTING NO TEMPO) E EXATIDÃO

Como este modelo atua na intersecção com inferência transversal (deduzindo a popularidade intrínseca latente via volume de interações), a prova final materializou-se sobre a construção de um script de retro-validação temporal autoral (*Backtesting*).

Nós confrontamos uma listagem de predição cega (gerada pela Inteligência Artificial e depositada no servidor congelada no dia $12/03/2026$) contra uma raspagem de dados reais populacionais completa, com os milhões de eleitores do Skoob, lida *exatamente um mês depois*, em $12/04/2026$.

Os desdobramentos atestam a escalabilidade da Ciência de Dados na predição comportamental, apresentando sucesso irrestrito num ecossistema denso habitado por centenas de categorias literárias:

* **Top 5 Precisão (Recall Global Ouro):** O algoritmo de Machine Learning cravou, categoricamente, e com perfeição global **$(5 / 5)$**, os gêneros "Distopia", "Fábula", "Gastronomia", "Animais" e "Ficção" no Top 5 definitivo do ranking ao longo daquele intervalo orgânico no aplicativo.
* **Erro Médio de Alocação de Ranking (MAE Position):** Num quadro avaliando mais de $320$ Gêneros em oscilação, o erro médio contínuo de oscilacão preditiva global do Algoritmo foi reduzidíssimo, caindo na marca de **$\approx 12,52$ posições de erro**.

> **📌 PRINT INDICADO 4:**
> - **Arquivo:** Cópia textual da Tabela de "VALIDAÇÃO DE PREVISÕES - BACKTEST" que eu extraí do comparador (o bloco de texto que citei Top 5 = 100%). O texto de relatório está salvo em `data/results/comparacao_previsao_2026-03-12_vs_2026-04-12.txt`.
> - **Onde colocar:** Na Conclusão Prática Oficial. Isso é O Pulo do Gato do TCC inteiro, materializa o valor social em um mês de uso do seu Software de Machine Learning em prever o mercado.

---

## 7. MELHORIAS E TRABALHOS FUTUROS

Encerre a Defesa Escrita citando o que poderia melhorar, sob a óptica Acadêmica e Operacional:

1. **Aprimoramento Dinâmico de NLP Baseado em LLM (Deep Learning):**
O atual estágio utiliza Modelos Preditivos Clássicos (Regressão Estatística) sobre a técnica base de Natural Language Processing via léxico estrito. Expandir os *Scores de Sentimento* para classificadores de ponta finos, que envolvem Transformers baseados em Redes Neurais Bi-direcionais (Ex: BERTimbau treinado em gírias brasileiras do Skoob), melhoraria drasticamente os cálculos orgânicos sensíveis (Sarcasmo e Ironia em resenhas literárias).

2. **Banco de Dados em Nuvem Contínuo (Data Warehousing):**
Por limitações do escopo, o arquivamento repousou em artefatos tabulares e estáticos locais (`.csv`). Evoluir o ecossistema com orquestradores de fluxo ETL assíncronos que enviam os subprodutos com metadados temporais diretamente à Nuvem (Ex: *AWS Redshift*, *GCP BigQuery*) fomentaria uma esteira viva de métricas para a leitura num painel (Dashboard de Business Intelligence via Tableau) vitalício.

3. **Time-Series Modelings Avançadas na Raiz:**
No atual espectro, a inferência preditiva baseia-se em regressão cruzada de traços rápidos (engajamento da amostra vigente). Escalonar um sistema com execução programada agendada (Crons mensais) transformaria a matriz linear em matriz longitudinal (dados no tempo 3D), providenciando margem substancial para o emprego de LSTMs (*Long Short-Term Memory Neural Networks*) capazes de prever sazonalidade e épocas do ano (Ex: Livros de romance jovem escalando sempre nas Férias de Verão).
