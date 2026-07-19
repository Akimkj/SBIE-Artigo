# Avaliação de LLMs como Recurso de Apoio ao Aprendizado de Programação Básica

Projeto de Iniciação Científica submetido ao **SBIE 2026 (Simpósio Brasileiro de Informática na Educação)**.

O objetivo é avaliar semanticamente respostas geradas por Large Language Models (LLMs) em perguntas de programação Python, comparando-as com respostas humanas de alta qualidade extraídas do Stack Overflow. O foco do projeto é a comparação direta entre respostas de LLMs e respostas humanas.

A principal contribuição é evidênciar que as LLMs ainda não são capazes de oferecer a qualidade semântica das respostas humanas. Além disso, um diferencial é a investigação do impacto do tamanho das respostas no comportamento do BERTScore, e a proposta de controle de tokens para reduzir esse viés.

---

## Metodologia

### 1. Montagem do Golden Set
- Coleta de questões do Stack Overflow via Stack Exchange API com a tag `python`
- Ordenação por score líquido (votes) em ordem decrescente
- Seleção dos **90 primeiros pares** pergunta-resposta
- Estrutura dos dados em JSON: `id`, `question`, `answer`

### 2. Geração dos Datasets pelas LLMs
Foram utilizados quatro modelos:

| Modelo | Provedor |
|---|---|
| GPT-4.1 mini | OpenAI |
| Gemini 2.5 Flash | Google |
| Claude Sonnet 4.6 | Anthropic |
| DeepSeek-V3 | DeepSeek |

Foram gerados **8 datasets** no total:
- **4 sem controle de tamanho** — resposta livre do modelo
- **4 com controle de tamanho** — resposta limitada a ±10% do número de tokens da resposta humana correspondente

O controle de tamanho foi introduzido para reduzir o viés do BERTScore em respostas de tamanhos muito diferentes. Os tokenizadores utilizados foram:
- `tiktoken` com encoding `cl100k_base` → GPT-4.1 mini e DeepSeek-V3
- APIs nativas de contagem de tokens → Gemini e Claude

### 3. Avaliação Semântica com BERTScore
- **Métrica principal**: BERTScore (Precision, Recall, F1-score)
- Encoder inicial: `roBERTa-large` — descartado por limite de 512 tokens
- Encoder final: **`ModernBERT-large`** — janela de contexto de até 8192 tokens, treinado com dados de código, mais adequado ao domínio de programação

### 4. Investigação do Viés por Tamanho
- Identificação de concentração de scores baixos em respostas curtas
- Reordenação das questões por tamanho de resposta
- Geração de heatmaps para visualização do comportamento do F1-score
- Proposta e aplicação do controle de tokens (±10%) como mitigação

---

## Prompts utilizados

### Versão sem controle de tamanho
```
You are a Computer Science and Python documentation expert. Your task is to answer the given question completely, technically, and in detail. Return ONLY a valid JSON with the following keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the given question; 'answer': a string that will be the returned answer. Rules: 1. DO NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input. If not provided, use 0; 4. The 'question' must exactly match the given question; 5. In the 'answer' field, provide a clear and complete response to the question, using appropriate technical details where relevant; 6. Escape all special characters correctly within the JSON strings; 7. DO NOT include Markdown; 8. DO NOT omit any required fields.
```

### Versão com controle de tamanho
```
You are a Computer Science and Python documentation expert. Your task is to answer the given question completely, technically, and in detail. Return ONLY a valid JSON with the following keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the given question; 'answer': a string that will be the returned answer. Rules: 1. DO NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input. If not provided, use 0; 4. The 'question' must exactly match the given question; 5. In the 'answer' field, provide a clear and complete response to the question, using appropriate technical details where relevant; 6. The response should be as close as possible to the specified number of tokens, staying within a tolerance margin between {int(numTokens * (1 - 0.10))} tokens and {int(numTokens * (1 + 0.10))} tokens. Avoid exceeding this margin unless strictly necessary.; 7. Escape all special characters correctly within the JSON strings; 8. DO NOT include Markdown; 9. DO NOT omit any required fields.
```

---

## Estrutura do Projeto

```
├── data/
│   ├── raw/                    # Golden Set (Stack Overflow)
│   └── processed/
│       ├── base/               # Datasets gerados pelas LLMs (sem controle de tamanho)
│       └── limited/            # Datasets gerados com controle de tamanho (±10% tokens)
├── results/
│   ├── csv/
│   │   ├── base/               # Resultados BERTScore dos datasets base
│   │   ├── debug/              # Resultados de execuções de depuração
│   │   └── limited/            # Resultados BERTScore dos datasets com controle
│   ├── graphs/
│   │   ├── base/               # Gráficos gerados a partir dos datasets base
│   │   ├── debug/              # Gráficos de depuração
│   │   └── limited/            # Gráficos dos datasets com controle de tamanho
│   └── stats/
│       ├── base/               # Estatísticas dos datasets base (bert_base_statistics.csv)
│       └── limited/            # Estatísticas dos datasets com controle (bert_limited_statistics.csv)
├── src/
│   ├── services/               # Integrações com APIs (Claude, OpenAI, Gemini, DeepSeek)
│   ├── evaluators/             # Avaliação com BERTScore (ModernBERT e RoBERTa)
│   ├── generator.py            # Geração dos datasets via LLMs
│   ├── dataFormat.py           # Validação de dados com Pydantic
│   ├── count_tokens.py         # Contagem de tokens por modelo
│   ├── bertStatistics.py       # Cálculo de estatísticas agregadas
│   ├── charts.py               # Geração de gráficos e heatmaps
│   └── utils.py                # Funções utilitárias
├── main.py                     # Menu interativo principal
└── requirements.txt
```

---

## Como Executar

### 1. Pré-requisitos
- Python 3.10 ou superior

### 2. Configuração do ambiente

```bash
python -m venv .venv
```

Ative o ambiente virtual:
- Windows: `.\.venv\Scripts\activate`
- Linux/macOS: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

> **macOS:** os pacotes `nvidia-*` e `triton` são exclusivos de Linux com GPU e foram removidos do `requirements.txt`. No macOS o `torch` é instalado automaticamente na versão compatível.

### 3. Variáveis de ambiente
Crie um arquivo `.env` na raiz com as chaves de API:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

### 4. Execução

```bash
python main.py
```

O menu oferece as seguintes opções:
1. Gerar novo dataset com um dos modelos
2. Avaliar dataset com BERTScore (ModernBERT ou RoBERTa)
3. Gerar gráficos comparativos (base ou limited)
4. Calcular estatísticas BERT — média, mediana, desvio padrão (base ou limited)

---
