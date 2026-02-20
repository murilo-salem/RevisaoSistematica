# Arquitetura do Sistema — Revisão Sistemática Automatizada

> Pipeline completo para conduzir revisões sistemáticas da literatura usando
> LLMs locais (Ollama), embeddings semânticos e análise de conteúdo.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Modos de Operação](#modos-de-operação)
- [Diagrama de Fluxo](#diagrama-de-fluxo)
- [Componentes Principais](#componentes-principais)
  - [Entrada e CLI](#1-entrada-e-cli--mainpy)
  - [Orquestrador](#2-orquestrador--orchestratorpy)
  - [Conversão PDF→Texto](#3-conversão-pdftexto--pdf_converterpy)
  - [Carregamento de Artigos](#4-carregamento-de-artigos--local_loaderpy)
  - [Deduplicação](#5-deduplicação--deduplicationpy)
  - [Triagem](#6-triagem--screeningpy)
  - [Análise de Conteúdo](#7-análise-de-conteúdo--content_analyzerpy)
  - [Organização em Pastas](#8-organização-em-pastas--organizerpy)
  - [Escrita da Revisão](#9-escrita-da-revisão--review_writerpy)
  - [Pós-Processamento](#10-pós-processamento--post_processorpy)
  - [Extração de Dados](#11-extração-de-dados--extractionpy)
  - [Avaliação de Risco de Viés](#12-avaliação-de-risco-de-viés--risk_of_biaspy)
  - [Síntese](#13-síntese--synthesispy)
  - [Geração de Manuscrito](#14-geração-de-manuscrito--manuscriptpy)
  - [Conversão Markdown→LaTeX](#15-conversão-markdownlatex--md2latexpy)
  - [Utilitários](#16-utilitários--utilspy)
  - [Construção de Query](#17-construção-de-query--query_builderpy)
  - [Recuperação do PubMed](#18-recuperação-do-pubmed--retrievalpy)
- [Configuração](#configuração)
- [Modelos de Dados](#modelos-de-dados)
- [Sistema de Qualidade Textual](#sistema-de-qualidade-textual)

---

## Visão Geral

O sistema automatiza todas as etapas de uma revisão sistemática da literatura:

1. **Coleta** de artigos (PubMed online ou arquivos locais)
2. **Deduplicação** semântica ou por título
3. **Triagem** via LLM ou taxonomia de palavras-chave
4. **Análise de conteúdo** — chunking + embeddings + mapeamento por seção
5. **Escrita** da revisão com prompts encadeados (chain-of-thought)
6. **Pós-processamento** — refinamento narrativo, coerência e limpeza textual
7. **Exportação** para Markdown, LaTeX e PDF

A infraestrutura LLM é fornecida pelo **Ollama** (modelos locais como
`qwen2.5:14b`, `gemma3:4b`). Embeddings semânticos são gerados por
**SentenceTransformers** (`all-MiniLM-L6-v2` ou `all-mpnet-base-v2`).

---

## Estrutura de Diretórios

```
RevisaoSistematica/
├── config/
│   ├── config.yaml              # Configuração padrão
│   ├── config_5090.yaml         # Perfil otimizado para GPU de alto desempenho
│   ├── taxonomia.json           # Taxonomia genérica (modo keyword)
│   └── biodiesel_prompts.json   # Taxonomia tipo outline (define capítulos/seções)
│
├── data/
│   ├── pdfs/                    # PDFs de entrada (convertidos automaticamente)
│   ├── raw/                     # Artigos em .txt / .json / .csv / .bib
│   ├── processed/               # Dados intermediários (pico.json, extracted_data.json, ...)
│   ├── organized/               # Artigos organizados por seção da taxonomia
│   └── results/                 # Saídas finais (review, manuscrito, gráficos)
│
├── src/
│   ├── main.py                  # CLI — ponto de entrada
│   ├── orchestrator.py          # Coordena todos os estágios do pipeline
│   ├── pdf_converter.py         # PDF → texto (PyMuPDF / pdfminer)
│   ├── local_loader.py          # Carrega artigos locais + taxonomia
│   ├── deduplication.py         # Deduplicação semântica / exata
│   ├── screening.py             # Triagem via LLM ou taxonomia
│   ├── content_analyzer.py      # Chunking + embedding + tag mapping
│   ├── organizer.py             # Organiza artigos em pastas por seção
│   ├── review_writer.py         # Escrita da revisão (multi-estágio)
│   ├── post_processor.py        # Refinamento narrativo + limpeza textual
│   ├── extraction.py            # Extração estruturada de dados (LLM)
│   ├── risk_of_bias.py          # Avaliação de risco de viés (LLM)
│   ├── synthesis.py             # Meta-análise ou síntese temática
│   ├── manuscript.py            # Gera manuscrito LaTeX via Jinja2
│   ├── md2latex.py              # Converte Markdown → LaTeX
│   ├── query_builder.py         # Gera PICO + query booleana (LLM)
│   ├── retrieval.py             # Busca no PubMed via Entrez
│   └── utils.py                 # Config, LLM, DB, modelos Pydantic
│
├── templates/
│   └── manuscript.tex.jinja     # Template LaTeX (Jinja2)
│
├── requirements.txt
└── README.md
```

---

## Modos de Operação

O sistema possui **três modos**, todos coordenados por `orchestrator.py`:

| Modo | Comando | Descrição |
|------|---------|-----------|
| **Online** | `python src/main.py --topic "..."` | LLM gera PICO → busca PubMed → Triagem → Extração → Síntese → Manuscrito LaTeX |
| **Local Outline** | `python src/main.py --local --taxonomy config/biodiesel_prompts.json` | Carrega artigos locais → Análise de conteúdo → Escrita por seções → Markdown |
| **Local Keyword** | `python src/main.py --local` | Carrega artigos → Triagem por taxonomia → Extração → Síntese |

Opções adicionais:
- `--profile 5090` — carrega `config_5090.yaml` (otimizado para GPUs de 32 GB)
- `--resume` — retoma da última execução salva
- `--cpu` — força modo CPU

---

## Diagrama de Fluxo

### Pipeline Online (7 estágios)

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ 1. Query    │───▶│ 2. PubMed   │───▶│ 3. Dedup     │───▶│ 4. Screen  │
│    Builder  │    │    Retrieval │    │   (semântica) │    │   (LLM)    │
└─────────────┘    └─────────────┘    └──────────────┘    └─────┬──────┘
                                                                │
                   ┌─────────────┐    ┌──────────────┐    ┌─────▼──────┐
                   │ 7. Manuscr. │◀───│ 6. Risk of   │◀───│ 5. Extract │
                   │    (LaTeX)  │    │    Bias (LLM) │    │   (LLM)    │
                   └──────┬──────┘    └──────────────┘    └────────────┘
                          │
                   ┌──────▼──────┐
                   │  Synthesis  │
                   │  (meta/qual)│
                   └─────────────┘
```

### Pipeline Local — Modo Outline (9 estágios)

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────────────┐
│ 0. PDF   │──▶│ 1. Load  │──▶│ 2. Dedup │──▶│ 3–6. Content Analysis    │
│  Convert │   │  Articles│   │          │   │  (chunk + embed + tag +  │
└──────────┘   └──────────┘   └──────────┘   │   coverage report)       │
                                              └────────────┬─────────────┘
                                                           │
┌───────────────────┐   ┌────────────────┐   ┌─────────────▼───────────┐
│ 9. Post-Process   │◀──│ 8. Section     │◀──│ 7. Organize into        │
│ (refine+coherence │   │    Writing     │   │    Folders               │
│  +textual cleanup)│   │  (LLM multi-  │   └─────────────────────────┘
└────────┬──────────┘   │   stage)       │
         │              └────────────────┘
         ▼
   systematic_review_v2.md
```

---

## Componentes Principais

### 1. Entrada e CLI — `main.py`

**Responsabilidade:** Ponto de entrada único do sistema. Analisa argumentos
da linha de comando, carrega a configuração e delega ao orquestrador.

| Argumento | Efeito |
|-----------|--------|
| `--topic "..."` | Modo online (requer Ollama + internet) |
| `--local` | Modo offline — artigos em `data/raw/` |
| `--taxonomy PATH` | Arquivo de taxonomia (JSON ou MD) |
| `--profile 5090` | Carrega `config_5090.yaml` |
| `--resume` | Retoma execução anterior |
| `--cpu` | Força CPU (ignora GPU) |

**Fluxo:**
1. Resolve qual `config.yaml` usar (padrão / perfil / custom)
2. Executa `check_system_capabilities()` (detecta PyTorch, CUDA, SentenceTransformers)
3. Delega para `run_pipeline()` (online) ou `run_pipeline_local()` (local)

---

### 2. Orquestrador — `orchestrator.py`

**Responsabilidade:** Coordena a execução sequencial de todos os estágios
do pipeline. Registra timing, estado intermediário e logs de auditoria.

**Funções principais:**

| Função | Descrição |
|--------|-----------|
| `run_pipeline(topic, cfg)` | Pipeline online completo (7 estágios) |
| `run_pipeline_local(taxonomy_path, cfg)` | Pipeline local (outline ou keyword) |
| `_init(cfg, label, topic)` | Inicializa config, logging, DB e estado |
| `_save_state(state, cfg)` | Persiste estado em `pipeline_state.json` |

**Mecanismo de estado:** Cada estágio registra `elapsed_s` e métricas
relevantes em um dicionário `state["stages"]`, permitindo auditoria
e retomada.

---

### 3. Conversão PDF→Texto — `pdf_converter.py`

**Responsabilidade:** Converte PDFs em `data/pdfs/` para `.txt` em `data/raw/`.

**Backends (fallback chain):**
1. **PyMuPDF** (`fitz`) — rápido, boa qualidade *(preferido)*
2. **pdfminer.six** — fallback caso PyMuPDF não esteja disponível

**Características:**
- Conversão em lote com barra de progresso (`tqdm`)
- Pula arquivos já convertidos (a menos que `--force` seja usado)
- Executável como CLI independente: `python src/pdf_converter.py --input data/pdfs`

---

### 4. Carregamento de Artigos — `local_loader.py`

**Responsabilidade:** Lê artigos de `data/raw/` em múltiplos formatos e os
converte para `StudyRecord`. Também carrega a taxonomia.

**Formatos suportados:**

| Formato | Função | Detalhes |
|---------|--------|----------|
| `.txt` | `_load_txt_files()` | Um arquivo = um estudo. Metadados extraídos do nome do arquivo |
| `.json` | `_load_json_file()` | Lista de objetos `{id, title, abstract, ...}` |
| `.csv` | `_load_csv_file()` | Cabeçalhos: id, title, abstract, ... |
| `.bib` | `_load_bib_file()` | BibTeX — extrai title, author, year, abstract |

**Parser de metadados do nome de arquivo:**
`_parse_filename_metadata("A._Saravanan_2019")` →
`{'author': 'A. Saravanan', 'year': 2019}`

**Taxonomia:** Suporta JSON e Markdown. Dois tipos:
- **Outline** (`"type": "outline"`) — define capítulos/seções com prompts
  individuais para escrita
- **Keyword** — define palavras-chave, critérios de inclusão/exclusão para
  triagem

---

### 5. Deduplicação — `deduplication.py`

**Responsabilidade:** Remove estudos duplicados antes da triagem.

**Dois modos:**

| Modo | Condição | Método |
|------|----------|--------|
| **Semântico** | SentenceTransformers disponível | Embeddings de título+abstract → similaridade cosseno > threshold |
| **Exato** | Fallback | Títulos normalizados (lowercase, strip) |

**Parâmetros (config):**
- `deduplication.model`: modelo de embedding (default: `all-MiniLM-L6-v2`)
- `deduplication.similarity_threshold`: limiar de similaridade (default: `0.95`)

Todas as decisões de deduplicação são registradas em `dedup_log` no SQLite.

---

### 6. Triagem — `screening.py`

**Responsabilidade:** Decide quais estudos devem ser incluídos na revisão.

**Dois modos:**

| Modo | Função | Requer LLM? |
|------|--------|-------------|
| **LLM** | `screen_studies()` | Sim — gera decisão JSON `{decision, confidence, justification}` |
| **Taxonomia** | `screen_studies_by_taxonomy()` | Não — verifica palavras-chave, inclusão e exclusão |

**Triagem LLM (dois passes):**
1. **Pass 1:** Prompt com critérios PICO → JSON de decisão
2. **Pass 2:** Filtragem por limiar de confiança

**Limiares (config):**
- `screening.threshold_include`: 0.75 (acima = incluído)
- `screening.threshold_exclude`: 0.25 (abaixo = excluído)
- Entre ambos = **ambíguo** (sinalizado para revisão humana)

Registra decisões no SQLite e gera `prisma.json` (diagrama PRISMA).

---

### 7. Análise de Conteúdo — `content_analyzer.py`

**Responsabilidade:** O módulo central do modo outline. Executa os
estágios 3–6 em uma só chamada:

1. **Chunking** — divide artigos em trechos semânticos sobrepostos
   (`_chunk_text()`, max 400 tokens, overlap 50)
2. **Embedding** — codifica chunks e prompts da taxonomia usando
   SentenceTransformers
3. **Tag Mapping** — calcula similaridade cosseno entre cada chunk e
   cada seção da taxonomia. Atribui as top-k tags
4. **Coverage Report** — gera estatísticas de cobertura por seção

**Parâmetros (config_5090):**
- `content_analyzer.chunk_max_tokens`: 600
- `content_analyzer.chunk_overlap`: 100
- `content_analyzer.similarity_threshold`: 0.25
- `content_analyzer.top_k_tags`: 5

**Saída:** `(chunks: List[Chunk], tags: List[ChunkTag], coverage: dict)`

---

### 8. Organização em Pastas — `organizer.py`

**Responsabilidade:** Cria uma estrutura de diretórios que espelha a
taxonomia, com metadados de artigos relevantes por seção.

**Hierarquia gerada:**
```
data/organized/
  └── {capítulo}/
      └── {seção}/
          └── section_articles.json
```

Cada `section_articles.json` contém: PMID, autor, ano, citação formatada,
similaridade máxima, quantidade de chunks e trecho representativo.

---

### 9. Escrita da Revisão — `review_writer.py`

**Responsabilidade:** Componente mais complexo. Gera o texto da revisão
para cada seção da taxonomia usando LLM com raciocínio encadeado.

**Pipeline multi-estágio por seção:**

```
Evidência → [Pré-Sumarização] → Escrita (chain-of-thought) → [Polimento]
```

#### 9.1. Coleta de Evidência (`_gather_evidence`)
Seleciona os top-k chunks mais relevantes para cada seção (por similaridade),
formata com citação `(Autor, Ano)`.

#### 9.2. Pré-Sumarização (opcional) (`_pre_summarize_evidence`)
Quando `evidence_pre_summarize: true`, cada chunk é resumido em um *brief*
estruturado antes de ser enviado ao prompt de escrita.

#### 9.3. Escrita com Chain-of-Thought (`_SECTION_PROMPT`)
Prompt de 3+1 estágios internos (invisíveis ao output):

| Estágio | Descrição |
|---------|-----------|
| **Stage 1 — Author Analysis** | Identifica contribuição de cada autor |
| **Stage 2 — Synthesis Map** | Mapeia consenso, divergências e lacunas |
| **Stage 2.5 — Critical Analysis** | Qualifica resultados (laboratório vs. campo, escalabilidade, custo) |
| **Stage 3 — Write Section** | Produz o texto final com regras rígidas |

**Regras de escrita:**
- Sujeito de cada frase = fenômeno/resultado, não o autor
- Citações apenas parentéticas `(Autor, Ano)`
- Proibidos conectores genéricos ("Além disso", "No entanto", etc.)
- Transições narrativas substanivas
- Terminologia consistente
- Todo o texto no idioma configurado (`language`)
- Linguagem de hedging para resultados não verificados em escala

#### 9.4. Polimento (opcional) (`_POLISH_PROMPT`)
Quando `two_pass_writing: true`, o rascunho passa por um segundo prompt
focado em coerência, redundância e fluência.

#### 9.5. Montagem do Documento (`_assemble_markdown`)
Combina todas as seções em um documento Markdown estruturado com hierarquia
de capítulos (`##`) e seções (`###`).

**Paralelismo:** Suporta escrita paralela via `ThreadPoolExecutor`
(configurável em `review_writer.parallel_workers`).

---

### 10. Pós-Processamento — `post_processor.py`

**Responsabilidade:** Refina o documento gerado pelo `review_writer`,
transformando parágrafos centrados em autores em uma narrativa acadêmica
orientada por consenso.

**Pipeline de refinamento:**

```
v1 (review_writer) → Parse → Refine seções → Coerência por capítulo
                                                    → Reassembly → Cleanup textual → v2
```

#### 10.1. Parsing (`_split_sections`)
Divide o Markdown em preamble + lista de `_Section` (heading, body, level,
parent).

#### 10.2. Refinamento por Seção (`_REFINE_PROMPT`)
Cada seção é enviada ao LLM com 18 regras:
- Reorganizar por tema (não por autor)
- Explicitar consenso e divergência
- Hedging para conclusões não verificadas
- Eliminação de redundâncias
- Transições narrativas
- Terminologia consistente

#### 10.3. Coerência por Capítulo (`_COHERENCE_PROMPT`)
Agrupa seções pelo `##` heading (capítulo) e envia o texto completo do
capítulo ao LLM para:
- Remover informação redundante entre seções
- Garantir transições lógicas
- Unificar formatação de citações

#### 10.4. Limpeza Textual Programática (`_textual_cleanup`)
Função determinística (sem LLM) que aplica regex para:
- Deduplicar citações dentro do mesmo grupo parentético
- Remover frases em inglês isoladas em texto português
- Normalizar erros ortográficos comuns ("catalise" → "catálise")
- Remover linhas em branco excessivas
- Remover frases meta ("This section discusses...")

---

### 11. Extração de Dados — `extraction.py`

**Responsabilidade:** Extrai dados estruturados de cada
estudo incluído (modo online e keyword).

**Dados extraídos (via LLM → JSON):**
`study_design`, `sample_size`, `population`, `intervention`, `comparison`,
`outcome`, `effect_size`, `ci_lower`, `ci_upper`, `p_value`, `notes`

Validados contra o schema Pydantic `ExtractionResult`.

---

### 12. Avaliação de Risco de Viés — `risk_of_bias.py`

**Responsabilidade:** Avalia cada estudo em 5 domínios de viés (modo online).

| Domínio | Rating |
|---------|--------|
| Selection | low / unclear / high |
| Performance | low / unclear / high |
| Detection | low / unclear / high |
| Attrition | low / unclear / high |
| Reporting | low / unclear / high |

Fallback: se o parse do JSON falhar, todos os domínios recebem "unclear".

---

### 13. Síntese — `synthesis.py`

**Responsabilidade:** Combina os dados extraídos em uma síntese quantitativa
ou qualitativa.

| Tipo | Condição | Método |
|------|----------|--------|
| **Meta-análise** | ≥ 3 estudos com `effect_size` + CIs | Inverse-variance weighting, Cochran's Q, I², τ² |
| **Temática** | Insuficiente dados numéricos | LLM identifica temas e gera síntese narrativa |

**Saídas:**
- `meta_analysis.json` — efeito poolado, IC, heterogeneidade
- `forest_plot.png` — gráfico forest plot (matplotlib)
- `thematic_analysis.json` — síntese temática

---

### 14. Geração de Manuscrito — `manuscript.py`

**Responsabilidade:** Gera um manuscrito LaTeX completo com 4 seções
(Introduction, Methods, Results, Discussion), cada uma produzida por um
prompt LLM especializado que recebe dados concretos.

Usa **Jinja2** com delimitadores customizados (`<<`, `>>`, `<%`, `%>`) para
renderizar o template `manuscript.tex.jinja`.

---

### 15. Conversão Markdown→LaTeX — `md2latex.py`

**Responsabilidade:** Converte o documento Markdown gerado pelo
review_writer/post_processor em LaTeX.

**Conversões:**
- `#` → `\section`, `##` → `\subsection`, `###` → `\subsubsection`
- `**bold**` → `\textbf{}`, `*italic*` → `\textit{}`
- Listas com `- ` → `\begin{itemize}`
- Escape de caracteres especiais LaTeX (`&`, `%`, `$`, `#`, `_`, etc.)
- Detecção e exclusão de seção "Referências"

Suporta idiomas (`pt` → `brazilian`, `en` → `english`) via pacote `babel`.

Executável como CLI: `python src/md2latex.py input.md -o output.tex`

---

### 16. Utilitários — `utils.py`

**Responsabilidade:** Módulo compartilhado com infraestrutura transversal.

#### Funções de Infra

| Função | Descrição |
|--------|-----------|
| `load_config(path)` | Carrega YAML e injeta `config_hash` |
| `setup_logging(cfg)` | Configura logging (arquivo + console) |
| `call_llm(prompt, cfg)` | Envia prompt ao Ollama via `/api/generate` (streaming) |
| `get_db_connection(cfg)` | Retorna conexão SQLite |
| `init_database(cfg)` | Cria schema (tabelas) se não existirem |
| `save_json(data, path)` / `load_json(path)` | I/O JSON |
| `_resolve(rel)` | Resolve caminho relativo ao projeto |
| `check_system_capabilities(cfg)` | Detecta PyTorch, CUDA, GPU e SentenceTransformers |

#### Comunicação com LLM (`call_llm`)

- Endpoint: `POST {base_url}/api/generate` (streaming)
- Timeout: `connect=30s`, `read=cfg[llm.timeout]` (gap entre chunks)
- Parâmetros opcionais passados: `temperature`, `seed`, `num_ctx`, `top_p`,
  `repeat_penalty`
- Retry com backoff em 500/502/503/504

#### Modelos de Dados (Pydantic)

| Modelo | Descrição |
|--------|-----------|
| `PICOModel` | Pergunta PICO estruturada |
| `StudyRecord` | Um registro bibliográfico (PMID, título, abstract, DOI, ...) |
| `ScreeningDecision` | Decisão de triagem (incluir/excluir + confiança) |
| `ExtractionResult` | Dados extraídos (design, amostra, efeito, IC, p) |
| `RiskOfBiasItem` | Avaliação de um domínio de viés |
| `RiskOfBiasResult` | Conjunto de 5 domínios para um estudo |
| `TaxonomyEntry` | Uma entrada da taxonomia (prompt, folder, parent) |
| `Chunk` | Trecho semântico de um artigo |
| `ChunkTag` | Mapeamento chunk → seção da taxonomia (com similaridade) |

#### Schema do Banco SQLite

Tabelas: `raw_studies`, `screening_log`, `dedup_log`, `extraction_log`,
`pipeline_runs`.

---

### 17. Construção de Query — `query_builder.py`

**Responsabilidade:** Gera a pergunta PICO e uma query booleana para PubMed
a partir de um tópico em texto livre (modo online).

**Fluxo:**
1. Prompt ao LLM pedindo JSON com `{population, intervention, comparison,
   outcome, query}`
2. Parse com fallback: JSON direto → regex → texto bruto
3. Persiste em `pico.json`

---

### 18. Recuperação do PubMed — `retrieval.py`

**Responsabilidade:** Busca artigos no PubMed usando a API NCBI Entrez
(modo online).

**Fluxo:**
1. `Entrez.esearch()` → lista de PMIDs
2. `Entrez.efetch()` em lotes → XML → parse para `StudyRecord`
3. Persiste cada registro no SQLite e salva log em `retrieval_log.json`

**Parâmetros:**
- `retrieval.max_results`: máximo de IDs (default: 500, 5090: 2000)
- `retrieval.batch_size`: IDs por chamada efetch (default: 50, 5090: 100)

---

## Configuração

O sistema é parametrizado por arquivos YAML em `config/`.

### `config.yaml` — Perfil Padrão

| Seção | Parâmetros-chave |
|-------|------------------|
| `llm` | `model: qwen2.5:14b`, `temperature: 0.2`, `timeout: 120` |
| `retrieval` | `max_results: 500`, `batch_size: 50` |
| `deduplication` | `model: all-MiniLM-L6-v2`, `threshold: 0.95` |
| `screening` | `threshold_include: 0.75`, `threshold_exclude: 0.25` |
| `review_writer` | `top_k_evidence: 10`, `language: pt`, `two_pass: false` |
| `post_processing` | `enabled: true`, `preserve_v1: true` |

### `config_5090.yaml` — Perfil de Alto Desempenho

Otimizado para 32 GB VRAM:

| Diferença | Padrão → 5090 |
|-----------|---------------|
| Modelo LLM | `qwen2.5:14b` → `gemma3:4b` |
| Contexto | (padrão) → `32768` tokens |
| Timeout | 120s → 900s |
| Embedding | `all-MiniLM-L6-v2` → `all-mpnet-base-v2` |
| Chunks | 400 tokens → 600 tokens |
| Evidência top-k | 10 → 25 |
| Paralelismo | 1 → 3 workers |
| Two-pass writing | false → true |
| Pré-sumarização | false → true |

---

## Modelos de Dados

```
StudyRecord ──▶ Chunk ──▶ ChunkTag
     │                        │
     │                        ▼
     │              TaxonomyEntry
     │              (prompt, folder, parent)
     │
     ├──▶ ScreeningDecision
     ├──▶ ExtractionResult ──▶ Synthesis
     └──▶ RiskOfBiasResult
                                  │
                                  ▼
                           Manuscript (LaTeX)
```

---

## Sistema de Qualidade Textual

A qualidade do texto gerado é garantida por **5 camadas**:

| Camada | Componente | Tipo |
|--------|-----------|------|
| **1. Chain-of-Thought** | `_SECTION_PROMPT` (Stages 1–3) | Prompt LLM |
| **2. Two-Pass Writing** | `_POLISH_PROMPT` | Prompt LLM |
| **3. Refinamento** | `_REFINE_PROMPT` | Prompt LLM |
| **4. Coerência** | `_COHERENCE_PROMPT` | Prompt LLM |
| **5. Cleanup** | `_textual_cleanup()` | Regex/Programático |

### Regras transversais (presentes em todos os prompts):
- ❌ Conectores genéricos ("Além disso", "No entanto", "Adicionalmente")
- ❌ Redundâncias e citações duplicadas
- ❌ Frases em inglês em texto português
- ❌ Erros ortográficos ("catalise" → "catálise")
- ✅ Transições narrativas substantivas
- ✅ Linguagem de hedging para resultados preliminares
- ✅ Análise crítica (laboratório vs. campo, escalabilidade)
- ✅ Terminologia consistente
