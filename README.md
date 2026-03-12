# Previsão de Tendências Literárias - Skoob

**Previsão de Tendências Literárias baseado em dados do Skoob**: Este projeto é o produto de um Trabalho de Conclusão de Curso (TCC) em Engenharia de Computação focado em Web Scraping, Processamento de Linguagem Natural (NLP) e Machine Learning (ML).

## Objetivo
Coletar dados públicos da maior rede social para leitores do Brasil (Skoob) para modelar, processar e prever estatisticamente quais **gêneros literários** estão ganhando tração e popularidade. A capacidade preditiva deste projeto ajuda a fornecer insights acionáveis para autores, editores e influenciadores do mercado editorial digital.

---

## Relatório Técnico Oficial
Se você deseja ler a documentação que abrange *Engenharia de Features*, *Adoção de Modelos ML*, *Fluxo de Dados* e os *Aspectos Legais e Técnicos* do Web Scraping sob a LGPD:
**[Relatório Técnico de Desenvolvimento](./relatorio_tecnico.md)**

---

## O que foi Implementado?
Este projeto compõe um Pipeline de Engenharia de Dados *End-to-End* (Ponta-a-Ponta), com **16 arquivos Python** organizados em 4 camadas principais:

| Camada | Stack / Módulos | Resumo das Funções |
|--------|---------|--------|
| **Data Collection** | `requests`, `beautifulsoup4`, `concurrent.futures` | Scraper multithread concorrente para varredura do Next.js App Router (Flight Data) do Skoob para livros e reviews. |
| **Data Processing** | `pandas`, `nltk`, `re` | Limpeza robusta de nulos, agregações em cascata e extração de Sentimento em resenhas de usuários via **Linguagem Natural**. |
| **Analytics & BI** | `matplotlib`, `seaborn` | Análise descritiva da população coletada. Correlações de Spearman (Ex: Nota Média x Engajamento) exportados automaticamente em dezenas de gráficos paramétricos. |
| **Machine Learning** | `scikit-learn` | Três algoritmos de regressão supervisionados para detecção de tendências: *Regressão Multipla*, *Decision Tree*, e *Random Forest Regressor*. Utiliza o histórico dinâmico para previsão a curto prazo do ranking de Gêneros Literários! |

---

## Como Instalar e Rodar

### Pré-requisitos
O projeto utiliza bibliotecas científicas de dados pesadas. Recomendamos usar um ecossistema como `venv` ou Conda com a versão **Python 3.9+**.

### 1. Clonar e Instalar
```bash
git clone https://github.com/SEU-USUARIO/previsao-tendencias-literarias.git
cd previsao-tendencias-literarias
pip install -r requirements.txt
```

### 2. Executar o Módulo Interativo
A arquitetura `main.py` incorpora um CLI dinâmico. Basta chamar:
```bash
python main.py
```
Isso abrirá o menu que o guiará pelas 7 Funcionalidades do sistema

```text
======================================================================
  📚 PREVISÃO DE TENDÊNCIAS LITERÁRIAS COM BASE EM DADOS DO SKOOB
  ─────────────────────────────────────────────────────────────────
  Gabriel Paredes Ferreira | IFTM | Engenharia de Computação
======================================================================

  Escolha uma opção:

  [1] 📥 Coleta de dados (web scraping do Skoob)
  [2] 🔧 Processamento de dados
  [3] 📊 Análise descritiva + visualizações
  [4] 🤖 Modelagem preditiva (ML)
  [5] 🚀 Pipeline completo (todas as etapas)
  [6] 📈 Apenas análise + modelos (requer dados já coletados)
  [7] 🎯 Validar previsões passadas (Backtesting)
  [0] ❌ Sair
```

*(Outra forma é chamar via parâmetros do shell `python main.py --pipeline` ou `--coleta`...)*

### Info de Execução:
- **Rate Limiting e Ética:** A etapa de raspagem (Coleta/Scraping) foi planejada e codificada contendo um forte *Backoff* Ético, demorando até 4 segundos entre as extrações por thread. Por causa disso, rodar o Dataset alvo de **5.000 Livros** tem uma demora programada de aproximadamente **25 minutos**.
- **Outputs e Modelos:** Todos os CSVs, Modelos ML Persistidos e Relatórios em TXT e PNG são sempre gerados dentro do diretório `/data/`.

---

## Autor
- **Gabriel Paredes Ferreira** 
- *IFTM - Campus Avançado Uberaba Parque Tecnológico*
- *Curso*: Engenharia de Computação

---
*Para contato e envio de pull requests sinta-se à vontade para navegar nos arquivos*
