# Avaliação Semântica de Respostas Geradas por LLMs no Contexto de Programação em Python

Projeto de Iniciação Científica submetido ao **SBIE (Simpósio Brasileiro de Informática na Educação)**.

O objetivo é avaliar semanticamente respostas geradas por Large Language Models (LLMs) em perguntas de programação Python, comparando-as com respostas humanas de alta qualidade extraídas do Stack Overflow. O projeto **não utiliza RAG** — o foco é a comparação direta entre respostas de LLMs e respostas humanas.

O principal diferencial é a investigação do impacto do tamanho das respostas no comportamento do BERTScore, e a proposta de controle de tokens para reduzir esse viés.

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

## Evolução dos Prompts

### Versão 1
```
You are an expert in Computer Science and Python documentation.
Answer the given question completely, technically, and directly, but without introductions like 'Sure', 'okay', 'certainly', etc.
Return ONLY a valid JSON with keys: 'id': integer; 'expectedQuestion': string; 'expectedAnswer': string.
```

### Versão 2
```
You are a Senior Computer Science Professor and Python Core Developer specializing in technical documentation.
Your goal is to provide a comprehensive, thorough, academic, and complete technical explanation of the proposed question. The answer should be at least 300 words to ensure depth.
CONSTRAINTS: 1. DO NOT use Markdown formatting. Use plain text only. 2. For code examples, write them inline or in plain text blocks without backticks. 3. DO NOT use introductory phrases or conversational fillers. 4. Structure the response with clear logical paragraphs instead of bullet points. 5. Focus on the internal mechanics of Python whenever applicable.
Return ONLY a valid JSON with keys: 'id': integer; 'question': string (exactly as provided); 'answer': string.
```

### Versão 3
```
You are an expert in Computer Science and Python documentation. Your task is to answer the provided question in a complete, technical, and detailed manner.
Return ONLY a valid JSON with keys: 'id', 'question', 'answer'.
Rules: 1. Do NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input; 4. The 'question' must exactly match the input question; 5. The 'answer' must be detailed, technical, and well-structured; 6. Escape all special characters properly; 7. Do NOT include markdown; 8. Do NOT omit any required field.
```

### Versão 4 — Com controle de tamanho
```
You are a Computer Science and Python documentation expert. Your task is to answer the given question completely, technically, and in detail.
Return ONLY a valid JSON with keys: 'id', 'question', 'answer'.
Rules: 1. DO NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input; 4. The 'question' must exactly match the given question; 5. Provide a clear and complete response using appropriate technical details; 6. The response should stay within a tolerance margin between {min_tokens} and {max_tokens} tokens; 7. Escape all special characters correctly; 8. DO NOT include Markdown; 9. DO NOT omit any required fields.
```

---

## Estrutura do Projeto

```
├── data/
│   ├── raw/                    # Golden Set (Stack Overflow)
│   └── processed/              # Datasets gerados pelas LLMs
│       └── limited/            # Datasets com controle de tamanho
├── results/
│   ├── csv/                    # Resultados BERTScore por modelo
│   │   └── limited/            # Resultados dos datasets com controle
│   ├── stats/                  # Estatísticas agregadas (média, mediana, desvio)
│   └── graphs/                 # Visualizações geradas
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
3. Gerar gráficos comparativos
4. Calcular estatísticas BERT (média, mediana, desvio padrão)

---

## Referências

```bibtex
@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}
```
