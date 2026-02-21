# Arquitetura do Sistema — Revisão Sistemática Automatizada

> Pipeline completo para conduzir revisões sistemáticas da literatura usando
> LLMs locais (Ollama), embeddings semânticos, análise de conteúdo e um
> **sistema multi-agente** com análise crítica, detecção de tabelas e
> consolidação de agenda de pesquisa.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Modos de Operação](#modos-de-operação)
- [Diagrama de Fluxo — Pipeline Clássica](#diagrama-de-fluxo--pipeline-clássica)
- [Componentes Clássicos](#componentes-clássicos)
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
- [**Sistema Multi-Agente**](#sistema-multi-agente)
  - [Visão Geral do Sistema](#visão-geral-do-sistema)
  - [Arquitetura de Agentes](#arquitetura-de-agentes)
  - [Protocolo de Mensagens](#protocolo-de-mensagens)
  - [Blackboard (Memória Compartilhada)](#blackboard--memória-compartilhada)
  - [Coordinator Agent](#coordinator-agent)
  - [Extraction Agent](#extraction-agent)
  - [Mapping Agent](#mapping-agent)
  - [Critical Agent](#critical-agent)
  - [Synthesis Agent](#synthesis-agent)
  - [Writing Agent](#writing-agent)
  - [Review Agent](#review-agent)
  - [Debate Agent](#debate-agent)
  - [Formatting Agent](#formatting-agent)
  - [Módulos de Suporte](#módulos-de-suporte)
  - [Diagrama de Fluxo — Multi-Agente](#diagrama-de-fluxo--multi-agente)
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
5. **Sistema Multi-Agente** — extração, análise crítica comparativa, síntese
   com tese e lacunas, escrita iterativa com revisão, debate de controvérsias
6. **Pós-processamento** — refinamento narrativo, coerência e limpeza textual
7. **Integração automática** — tabelas comparativas, agenda de pesquisa consolidada
8. **Exportação** para Markdown, LaTeX e PDF

A infraestrutura LLM é fornecida pelo **Ollama** com modelos configuráveis
por agente: `qwen3:8b` para tarefas gerais e `mixtral:8x22b` para raciocínio
complexo (análise crítica, síntese, debate). Embeddings semânticos são gerados
por **SentenceTransformers** (`all-mpnet-base-v2`).

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
│   ├── orchestrator.py          # Coordena pipeline clássica (sem multi-agente)
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
│   ├── utils.py                 # Config, LLM, DB, modelos Pydantic
│   │
│   ├── concept_registry.py      # Registro de conceitos já explicados (anti-redundância)
│   ├── evidence_synthesizer.py  # Análise consenso / contradição / lacunas por tema
│   ├── table_generator.py       # Detecção e geração automática de tabelas comparativas
│   │
│   └── agents/                  # ★ SISTEMA MULTI-AGENTE ★
│       ├── __init__.py
│       ├── base_agent.py        # Classes base (Message, AgentResult, BaseAgent)
│       ├── blackboard.py        # Memória compartilhada entre agentes
│       ├── pipeline.py          # Ponto de entrada do modo multi-agente
│       ├── coordinator_agent.py # Orquestra todos os agentes (8 fases)
│       ├── extraction_agent.py  # Extração de dados estruturados (PICO, metadata)
│       ├── mapping_agent.py     # Mapeamento evidência → taxonomia
│       ├── critical_agent.py    # Análise crítica + comparativa entre estudos
│       ├── synthesis_agent.py   # Síntese: tese + lacunas agrupadas + agenda
│       ├── writing_agent.py     # Escrita de seções com ConceptRegistry
│       ├── review_agent.py      # Revisão com redundância cross-section
│       ├── debate_agent.py      # Debate estruturado para temas controversos
│       ├── formatting_agent.py  # Montagem final + tabelas + agenda
│       └── models.py            # Modelos Pydantic dos agentes
│
├── templates/
│   └── manuscript.tex.jinja     # Template LaTeX (Jinja2)
│
├── requirements.txt
└── README.md
```

---

## Modos de Operação

O sistema possui **quatro modos**, coordenados por `orchestrator.py` (clássico) ou
`agents/pipeline.py` (multi-agente):

| Modo | Comando | Descrição |
|------|---------|-----------|
| **Online** | `python src/main.py --topic "..."` | LLM gera PICO → busca PubMed → Triagem → Extração → Síntese → Manuscrito LaTeX |
| **Local Outline** | `python src/main.py --local --taxonomy config/biodiesel_prompts.json` | Carrega artigos locais → Análise de conteúdo → Escrita por seções → Markdown |
| **Local Keyword** | `python src/main.py --local` | Carrega artigos → Triagem por taxonomia → Extração → Síntese |
| **★ Multi-Agente** | `python src/main.py --local --multi-agent --taxonomy ...` | Sistema de 8 agentes com análise crítica, síntese, revisão iterativa e debate |

Opções adicionais:
- `--profile 5090` — carrega `config_5090.yaml` (otimizado para GPUs de alto desempenho)
- `--resume` — retoma da última execução salva
- `--cpu` — força modo CPU

---

## Diagrama de Fluxo — Pipeline Clássica

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

## Componentes Clássicos

### 1. Entrada e CLI — `main.py`

**Responsabilidade:** Ponto de entrada único do sistema. Analisa argumentos
da linha de comando, carrega a configuração e delega ao orquestrador.

| Argumento | Efeito |
|-----------|--------|
| `--topic "..."` | Modo online (requer Ollama + internet) |
| `--local` | Modo offline — artigos em `data/raw/` |
| `--multi-agent` | Ativa o sistema multi-agente em vez do pipeline clássico |
| `--taxonomy PATH` | Arquivo de taxonomia (JSON ou MD) |
| `--profile 5090` | Carrega `config_5090.yaml` |
| `--resume` | Retoma execução anterior |
| `--cpu` | Força CPU (ignora GPU) |

**Fluxo:**
1. Resolve qual `config.yaml` usar (padrão / perfil / custom)
2. Executa `check_system_capabilities()` (detecta PyTorch, CUDA, SentenceTransformers)
3. Se `--multi-agent`: delega para `run_multi_agent_pipeline()` em `agents/pipeline.py`
4. Caso contrário: delega para `run_pipeline()` (online) ou `run_pipeline_local()` (local)

---

### 2. Orquestrador — `orchestrator.py`

**Responsabilidade:** Coordena a execução sequencial de todos os estágios
do pipeline clássico. Registra timing, estado intermediário e logs de auditoria.

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

| Formato | Função | Detalhes |
|---------|--------|----------|
| `.txt` | `_load_txt_files()` | Um arquivo = um estudo. Metadados extraídos do nome do arquivo |
| `.json` | `_load_json_file()` | Lista de objetos `{id, title, abstract, ...}` |
| `.csv` | `_load_csv_file()` | Cabeçalhos: id, title, abstract, ... |
| `.bib` | `_load_bib_file()` | BibTeX — extrai title, author, year, abstract |

**Taxonomia:** Suporta JSON e Markdown. Dois tipos:
- **Outline** (`"type": "outline"`) — define capítulos/seções com prompts
  individuais para escrita
- **Keyword** — define palavras-chave, critérios de inclusão/exclusão para
  triagem

---

### 5. Deduplicação — `deduplication.py`

**Responsabilidade:** Remove estudos duplicados antes da triagem.

| Modo | Condição | Método |
|------|----------|--------|
| **Semântico** | SentenceTransformers disponível | Embeddings de título+abstract → similaridade cosseno > threshold |
| **Exato** | Fallback | Títulos normalizados (lowercase, strip) |

**Parâmetros:**
- `deduplication.model`: modelo de embedding (default: `all-MiniLM-L6-v2`)
- `deduplication.similarity_threshold`: limiar de similaridade (default: `0.95`)

---

### 6. Triagem — `screening.py`

**Responsabilidade:** Decide quais estudos devem ser incluídos na revisão.

| Modo | Função | Requer LLM? |
|------|--------|-------------|
| **LLM** | `screen_studies()` | Sim — gera decisão JSON `{decision, confidence, justification}` |
| **Taxonomia** | `screen_studies_by_taxonomy()` | Não — verifica palavras-chave, inclusão e exclusão |

**Triagem LLM (dois passes):**
1. **Pass 1:** Prompt com critérios PICO → JSON de decisão
2. **Pass 2:** Filtragem por limiar de confiança

---

### 7. Análise de Conteúdo — `content_analyzer.py`

**Responsabilidade:** O módulo central do modo outline. Executa os
estágios 3–6 em uma só chamada:

1. **Chunking** — divide artigos em trechos semânticos sobrepostos
   (`_chunk_text()`, max 600 tokens, overlap 100)
2. **Embedding** — codifica chunks e prompts da taxonomia usando
   SentenceTransformers (`all-mpnet-base-v2`)
3. **Tag Mapping** — calcula similaridade cosseno entre cada chunk e
   cada seção da taxonomia. Atribui as top-k tags
4. **Coverage Report** — gera estatísticas de cobertura por seção

**Saída:** `(chunks: List[Chunk], tags: List[ChunkTag], coverage: dict)`

---

### 8. Organização em Pastas — `organizer.py`

**Responsabilidade:** Cria uma estrutura de diretórios que espelha a
taxonomia, com metadados de artigos relevantes por seção.

---

### 9. Escrita da Revisão — `review_writer.py`

**Responsabilidade:** Gera o texto da revisão para cada seção da taxonomia
usando LLM com raciocínio encadeado (usado pelo modo clássico e como backend
pelo `WritingAgent` no modo multi-agente).

**Pipeline por seção:**
```
Evidência → [Pré-Sumarização] → Escrita (chain-of-thought) → [Polimento]
```

**Regras de escrita:**
- Sujeito de cada frase = fenômeno/resultado, não o autor
- Citações apenas parentéticas `(Autor, Ano)`
- Proibidos conectores genéricos ("Além disso", "No entanto", etc.)
- Transições narrativas substantivas
- Terminologia consistente
- Linguagem de hedging para resultados não verificados em escala

---

### 10. Pós-Processamento — `post_processor.py`

**Responsabilidade:** Refina o documento gerado, transformando parágrafos
centrados em autores em uma narrativa acadêmica orientada por consenso.

**Pipeline:**
```
v1 → Parse → Refine seções → Coerência por capítulo → Cleanup textual → v2
```

---

### 11–18. Componentes Adicionais

| # | Componente | Responsabilidade |
|---|-----------|-----------------|
| 11 | `extraction.py` | Extração estruturada de dados via LLM → `ExtractionResult` |
| 12 | `risk_of_bias.py` | Avaliação de risco de viés em 5 domínios |
| 13 | `synthesis.py` | Meta-análise (inverse-variance) ou síntese temática |
| 14 | `manuscript.py` | Geração de manuscrito LaTeX via Jinja2 |
| 15 | `md2latex.py` | Conversão Markdown → LaTeX |
| 16 | `utils.py` | Config, LLM, DB, modelos Pydantic, check_system_capabilities |
| 17 | `query_builder.py` | Geração PICO + query booleana (modo online) |
| 18 | `retrieval.py` | Busca PubMed via NCBI Entrez |

---

## Sistema Multi-Agente

### Visão Geral do Sistema

O sistema multi-agente é a evolução central do pipeline, substituindo a
execução linear clássica por um sistema de **8 agentes especializados**
coordenados por um `CoordinatorAgent`, comunicando-se via **mensagens
tipadas** e compartilhando estado via um **Blackboard** (memória
compartilhada).

**Ativação:** `python src/main.py --local --multi-agent --profile 5090 --taxonomy config/biodiesel_prompts.json`

**Vantagens sobre o pipeline clássico:**
- **Análise crítica comparativa** — avalia robustez de claims com comparação
  estruturada entre estudos (design, escala, resultados)
- **Eliminação de redundância** — `ConceptRegistry` rastreia conceitos já
  explicados; `ReviewAgent` valida redundância cross-section
- **Síntese profunda** — tese falsificável por tema, lacunas agrupadas com
  prioridade, agenda de pesquisa consolidada
- **Iteração write→review** — loop de até N iterações com feedback
  estruturado e threshold de qualidade
- **Debate estruturado** — para temas controversos, advocacia pro/contra
  seguida de síntese moderada
- **Tabelas automáticas** — detecção de oportunidades de tabela comparativa
  e geração em Markdown
- **Modelos por agente** — cada agente pode usar um modelo LLM diferente
  (ex: `qwen3:8b` para escrita, `mixtral:8x22b` para análise crítica)

---

### Arquitetura de Agentes

Todos os agentes herdam de `BaseAgent` (`base_agent.py`), que fornece:

```python
class BaseAgent(ABC):
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None
    def process(self, message: Message) -> AgentResult    # abstrato
    def call_llm(self, prompt: str, model_override=None) -> str
    def timed_process(self, message: Message) -> AgentResult
```

**Seleção de modelo por agente:**
O `call_llm()` verifica `cfg.multi_agent.agent_models.<name>`. Se houver
um override, usa esse modelo; senão, usa o modelo global (`llm.model`).

```yaml
# config_5090.yaml
multi_agent:
  agent_models:
    extraction: ~                  # usa qwen3:8b (global)
    mapping: ~                     # usa qwen3:8b (global)
    critical: "mixtral:8x22b"      # raciocínio complexo
    synthesis: "mixtral:8x22b"     # tese + lacunas + agenda
    writing: ~                     # usa qwen3:8b (global)
    review: ~                      # usa qwen3:8b (global)
    debate: "mixtral:8x22b"        # raciocínio complexo
```

---

### Protocolo de Mensagens

A comunicação inter-agente usa a classe `Message`:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `task` | `str` | Identificador da tarefa (`"extract"`, `"write_section"`, `"review"`, ...) |
| `payload` | `Dict[str, Any]` | Dados da tarefa — conteúdo varia por agente |
| `source` | `str` | Agente de origem (ou `"coordinator"`) |
| `timestamp` | `str` | ISO-8601 |
| `iteration` | `int` | Número da iteração atual (para loops write→review) |
| `feedback` | `str` | Feedback da iteração anterior |

**Retorno padronizado** (`AgentResult`):

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `success` | `bool` | Se o agente completou sem erros fatais |
| `data` | `Dict[str, Any]` | Output específico do agente |
| `errors` | `List[str]` | Mensagens de erro |
| `metrics` | `Dict[str, float]` | Métricas de desempenho (`elapsed_s`, etc.) |

---

### Blackboard — Memória Compartilhada

O `Blackboard` (`blackboard.py`) é o estado compartilhado entre todos os
agentes. Ele armazena:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `extractions` | `Dict[str, ExtractionResult]` | Dados extraídos por PMID |
| `synthesis_maps` | `Dict[str, SynthesisMap]` | Mapa de consenso/contradição/lacunas por tema |
| `chapter_theses` | `Dict[str, str]` | Tese de cada capítulo/seção |
| `section_drafts` | `Dict[str, List[str]]` | Histórico de rascunhos por seção |
| `approved_sections` | `Dict[str, str]` | Seções aprovadas pelo ReviewAgent |
| `review_scores` | `Dict[str, float]` | Nota da última revisão por seção |
| `debate_sections` | `Dict[str, str]` | Seções de debate geradas |
| `concept_registry` | `ConceptRegistry` | Registro de conceitos já explicados |
| `research_agenda` | `List[Dict]` | Agenda consolidada de pesquisa |
| `table_count` | `int` | Número de tabelas geradas |

**Persistência:** Os métodos `save()` e `load()` serializam o blackboard em
JSON, incluindo o `ConceptRegistry` (exportado via `to_json()`).

---

### Coordinator Agent

**Arquivo:** `coordinator_agent.py`
**Responsabilidade:** Orquestra a execução dos 8 agentes em **6 fases**
(com sub-fases).

#### Fases de Execução

| Fase | Nome | Agente | Descrição |
|------|------|--------|-----------|
| **1** | Extraction | `ExtractionAgent` | Extrai dados PICO + metadados críticos de cada estudo |
| **2** | Mapping | `MappingAgent` | Mapeia evidências para seções da taxonomia |
| **3** | Theme Loop | Múltiplos | Para **cada tema** da taxonomia: |
| 3a | Critical Analysis | `CriticalAgent` | Análise crítica com comparação estruturada |
| 3b | Synthesis | `SynthesisAgent` | Gera tese + lacunas agrupadas |
| 3c | Write | `WritingAgent` | Escreve seção com `ConceptRegistry` |
| 3d | Review | `ReviewAgent` | Avalia qualidade + redundância cross-section |
| 3c-d | *Loop* | Write + Review | Itera até `quality_threshold` (default: 7.0) ou `max_iterations` (default: 3) |
| **3.5** | Table Detection | (interno) | Detecta oportunidades de tabela comparativa por tema |
| **3.6** | Agenda Consolidation | `SynthesisAgent` | Consolida lacunas de todos os temas em agenda final |
| **4** | Debate (opcional) | `DebateAgent` | Se houver temas com >N contradições e robustez baixa |
| **5** | Formatting | `FormattingAgent` | Monta documento final + tabelas + agenda de pesquisa |

#### Mecanismos Internos do Coordinator

**ConceptRegistry Integration:**
Após cada seção ser aprovada, o coordinator registra os conceitos cobertos
no `ConceptRegistry`. Na próxima seção, os conceitos já registrados são
enviados ao `WritingAgent` com informação de onde foram introduzidos:
```
"catálise heterogênea (ver: Alkaline Catalysis)"
```

**Build Approved Summary:**
O método `_build_approved_summary()` gera resumos de até 300 caracteres de
cada seção aprovada, que são passados ao `ReviewAgent` para validação de
redundância cross-section.

**Table Detection:**
O método `_detect_tables_for_themes()` usa `table_generator.detect_table_opportunity()`
para identificar temas com dados quantitativos comparáveis e gerar
`TableSpec` objects, registrados no `TableRegistry`.

---

### Extraction Agent

**Arquivo:** `extraction_agent.py`
**Modelo:** `qwen3:8b` (padrão)

Extrai dados estruturados de cada estudo via LLM → JSON → `ExtractionResult`.

**Campos extraídos:**
`study_design`, `sample_size`, `population`, `intervention`, `comparison`,
`outcome`, `effect_size`, `ci_lower`, `ci_upper`, `p_value`, `notes`,
`study_scale`, `geographic_scope`, `funding_source`, `conflict_of_interest`,
`limitations`

**Tolerância a nulls:** Um `@model_validator` converte automaticamente
campos `None` (retornados pelo LLM) em strings vazias.

---

### Mapping Agent

**Arquivo:** `mapping_agent.py`
**Modelo:** `qwen3:8b` (padrão)

Mapeia estudos extraídos para seções da taxonomia usando a evidência
pré-mapeada (chunk→tag). Gera o `SynthesisMap` por tema usando o
`evidence_synthesizer`.

---

### Critical Agent

**Arquivo:** `critical_agent.py`
**Modelo:** `mixtral:8x22b` (complexo)

Produz uma análise crítica **estruturada** para cada tema com os seguintes
componentes:

| Componente | Descrição |
|-----------|-----------|
| **Methodological Quality** | Avaliação de design, tamanho amostral, reprodutibilidade |
| **Contradictions** | Identificação de resultados contraditórios com causa possível |
| **Comparative Analysis** | Comparação estruturada entre 2–4 estudos principais |
| **Robustness Rating** | Classificação: `alta`, `média` ou `baixa` |
| **Contextual Factors** | Influência de geografia, financiamento, escala |

**Análise Comparativa (novo):**
Para cada par de estudos comparado, o agente produz um `ComparativeItem`:
```json
{
  "study_a": "...", "study_b": "...",
  "methodology_diff": "lab-scale com KOH vs. pilot-scale com NaOH",
  "result_diff": "98% rendimento vs. 82% rendimento",
  "possible_cause": "concentração de catalisador e tempo de reação diferentes",
  "robustness_note": "moderate — poucos estudos corroborando"
}
```

**Janela de evidência:** Top 20 tags mais relevantes por tema (ampliado de 15).

---

### Synthesis Agent

**Arquivo:** `synthesis_agent.py`
**Modelo:** `mixtral:8x22b` (complexo)

Produz dois outputs fundamentais:

#### 1. Tese + Lacunas Agrupadas (por tema)
O prompt produz um JSON com:
- **Thesis:** Declaração falsificável de 1-2 sentenças
- **Grouped Gaps:** Lacunas organizadas por grupo temático, cada uma com
  prioridade (`alta`/`média`) e justificativa
- **Research Priorities:** 3-5 prioridades ordenadas por importância

```json
{
  "thesis": "Embora a catálise alcalina domine...",
  "grouped_gaps": [
    {
      "group": "Escalabilidade",
      "gaps": [
        {"description": "...", "priority": "alta", "justification": "..."}
      ]
    }
  ],
  "research_priorities": ["...", "..."]
}
```

#### 2. Consolidação de Agenda (cross-tema)
O método `consolidate_agenda()` recebe **todas** as lacunas de todos os
temas e usa um prompt dedicado (`_AGENDA_PROMPT`) para:
- Deduplicar lacunas semanticamente equivalentes
- Priorizar por impacto científico
- Gerar uma agenda final de até 30 itens

---

### Writing Agent

**Arquivo:** `writing_agent.py`
**Modelo:** `qwen3:8b` (padrão)

Escreve cada seção da revisão usando:
1. **Evidência** coletada por `_gather_evidence()` do `review_writer.py`
2. **Conceitos cobertos** do `ConceptRegistry` com indicação da seção
   onde foram introduzidos
3. **Tese** e `SynthesisMap` do tema

**Anti-redundância:**
Quando o `ConceptRegistry` fornece dados estruturados, o agente gera
strings como:
```
"Já cobertos: catálise heterogênea (ver: Alkaline Catalysis), esterificação (ver: Acid Process)"
```
Isso instrui o LLM a referenciar — não re-explicar — esses conceitos.

---

### Review Agent

**Arquivo:** `review_agent.py`
**Modelo:** `qwen3:8b` (padrão)

Avalia cada seção em **6 critérios** (escala 1-10):

| # | Critério | Descrição |
|---|---------|-----------|
| 1 | Thesis Clarity | Clareza e presença da tese |
| 2 | Redundancy | Repetições internas na seção |
| 3 | **Cross-Section Redundancy** | Sobreposição com seções já aprovadas |
| 4 | Citation Usage | Uso correto e consistente de citações |
| 5 | Hedging | Linguagem de hedging apropriada |
| 6 | Critical Depth | Análise além de mera descrição |

**Cross-Section Redundancy (novo):**
O coordinator passa ao reviewer um resumo das seções já aprovadas
(`approved_sections_summary`). O reviewer verifica se a seção atual
repete explicações ou conceitos já cobertos em outras seções, listando
os overlaps específicos encontrados.

**Output JSON:**
```json
{
  "thesis_clarity": {"score": 8.0, "comment": "..."},
  "redundancy": {"score": 7.5, "comment": "..."},
  "cross_section_redundancy": {"score": 9.0, "comment": "..."},
  "citation_usage": {"score": 8.0, "comment": "..."},
  "hedging": {"score": 7.0, "comment": "..."},
  "critical_depth": {"score": 6.5, "comment": "..."},
  "overall_score": 7.5,
  "overall_feedback": "Sugestões específicas..."
}
```

**Loop iterativo:** Se `overall_score < quality_threshold` (default 7.0),
o coordinator re-envia a seção ao `WritingAgent` com o feedback como
contexto, até `max_iterations` (default 3).

---

### Debate Agent

**Arquivo:** `debate_agent.py`
**Modelo:** `mixtral:8x22b` (complexo)

Ativado quando um tema tem:
- Robustez ≤ `"média"` na análise crítica
- Número de contradições ≥ `debate_controversy_threshold` (default: 2)

**Fluxo:**
1. **Position A (Supportive):** LLM argumenta que os resultados são
   positivos e as contradições são explicáveis por condições experimentais
2. **Position B (Critical):** LLM argumenta que a metodologia é inadequada
   e as contradições minam as conclusões
3. **Moderator Synthesis:** LLM modera o debate, produzindo 3-4 parágrafos
   balanceados com perspectivas justas, explicações das contradições e
   lacunas a resolver

O resultado é inserido como subseção `#### Debate: {tema}` na seção aprovada.

---

### Formatting Agent

**Arquivo:** `formatting_agent.py`
**Modelo:** `qwen3:8b` (padrão)

Responsável pela montagem final do documento:

1. **Assemblagem Markdown:** Combina todas as seções aprovadas em documento
   estruturado via `_assemble_markdown()` do `review_writer.py`
2. **Pós-processamento:** Aplica o pipeline completo de refinamento
   (`_refine_section`, `_check_chapter_coherence`, `_dedup_chapters`,
   `_validate_argumentation`, `_textual_cleanup`)
3. **Agenda de Pesquisa (novo):** Insere a seção
   `## Agenda de Pesquisas Futuras` com lacunas agrupadas por prioridade
   (Alta / Média), cada uma com abordagem sugerida
4. **Tabelas:** Log de tabelas incluídas no documento

---

### Módulos de Suporte

#### `concept_registry.py` — Registro Anti-Redundância

Registra conceitos já explicados na revisão, thread-safe, com normalização
Unicode para matching fuzzy:

| Método | Descrição |
|--------|-----------|
| `register(concept, section)` | Marca conceito como coberto na seção |
| `register_many(concepts, section)` | Registra lista de conceitos |
| `already_covered(concept)` | Verifica se conceito já foi coberto |
| `get_covered_concepts()` | Lista de conceitos (originais, sorted) |
| `get_covered_with_sections()` | Lista `[{concept, section}]` |
| `to_json(path)` / `from_json(path)` | Persistência |

**Normalização:** lowercase → strip accents (NFD) → collapse whitespace
```python
"Catálise Heterogênea" → "catalise heterogenea"
```

#### `evidence_synthesizer.py` — Análise de Consenso

Analisa chunks tagados por tema e produz um `SynthesisMap`:

| Componente | Modelo Pydantic | Campos |
|-----------|----------------|--------|
| Consensus | `ConsensusPoint` | `statement`, `supporting_studies`, `strength` |
| Contradiction | `Contradiction` | `point`, `study_a/b`, `finding_a/b`, `possible_reason` |
| Gap | `KnowledgeGap` | `description`, `priority`, `suggested_approach` |

#### `table_generator.py` — Tabelas Comparativas Automáticas

Detecta oportunidades de tabelas comparativas e gera Markdown:

| Função | Descrição |
|--------|-----------|
| `detect_table_opportunity()` | LLM analisa evidência e decide se tabela é apropriada |
| `generate_markdown_table()` | Converte `TableSpec` em Markdown formatado |
| `TableRegistry.register()` | Registra tabela gerada |
| `TableRegistry.replace_markers()` | Substitui marcadores `[TABLE: id]` no texto |

**Critérios de detecção:**
- Múltiplos estudos reportando a mesma métrica
- Comparações entre catalisadores, matérias-primas, temperaturas
- Dados numéricos que beneficiam apresentação lado a lado

---

### Diagrama de Fluxo — Multi-Agente

```
                    ┌──────────────────────────────────────────────────┐
                    │           PIPELINE PRÉ-PROCESSAMENTO            │
                    │  Load → Dedup → PDF Convert → Content Analysis  │
                    └──────────────────────┬───────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COORDINATOR AGENT                                    │
│                                                                             │
│   Phase 1: ExtractionAgent ─────────────────────────────────────────────── │
│   │  Extrai PICO + metadados de cada estudo                                │
│   ▼                                                                         │
│   Phase 2: MappingAgent ────────────────────────────────────────────────── │
│   │  Mapeia evidências → SynthesisMap por tema                             │
│   ▼                                                                         │
│   Phase 3: Theme Loop (para cada tema) ─────────────────────────────────── │
│   │                                                                         │
│   │  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐        │
│   │  │ 3a. Critical │───▶│ 3b. Synthesis │───▶│ 3c. Write        │        │
│   │  │   Agent      │    │    Agent      │    │    Agent         │        │
│   │  │ (comparativa)│    │ (tese+gaps)   │    │ (ConceptRegistry)│        │
│   │  └──────────────┘    └───────────────┘    └────────┬─────────┘        │
│   │                                                     │                   │
│   │                                           ┌─────────▼─────────┐        │
│   │                                           │ 3d. Review Agent  │        │
│   │                                           │ (6 critérios +    │        │
│   │                                           │  cross-section)   │        │
│   │                                           └─────────┬─────────┘        │
│   │                                                     │                   │
│   │                                    score ≥ 7.0? ────┤                   │
│   │                                    ┌──YES──┐   ┌──NO──┐                │
│   │                                    │Approve│   │Retry │                │
│   │                                    │+ Reg. │   │(→3c) │                │
│   │                                    └───────┘   └──────┘                │
│   │                                                                         │
│   Phase 3.5: Table Detection ─────────────────────────────────────────── │
│   │  Detecta oportunidades de tabela comparativa por tema                  │
│   ▼                                                                         │
│   Phase 3.6: Agenda Consolidation ────────────────────────────────────── │
│   │  SynthesisAgent consolida lacunas de todos os temas                    │
│   ▼                                                                         │
│   Phase 4: DebateAgent (opcional) ────────────────────────────────────── │
│   │  Debate pro/contra + síntese moderada para temas controversos          │
│   ▼                                                                         │
│   Phase 5: FormattingAgent ──────────────────────────────────────────── │
│      Montagem final + pós-processamento + agenda + tabelas                 │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
                        systematic_review.md
                     (+ agenda de pesquisa + tabelas)
```

---

### Estado Compartilhado — Fluxo de Dados

```
                    ┌─────────────┐
                    │  Blackboard │
                    ├─────────────┤
  ExtractionAgent ─▶│ extractions │
                    │             │
    MappingAgent ──▶│ synthesis_  │
                    │    maps     │
                    │             │
  SynthesisAgent ──▶│ chapter_    │──▶ WritingAgent
                    │    theses   │
                    │             │
                    │ concept_    │◀── CoordinatorAgent
                    │    registry │──▶ WritingAgent
                    │             │
    ReviewAgent ───▶│ approved_   │──▶ FormattingAgent
                    │    sections │
                    │             │
    ReviewAgent ───▶│ review_     │──▶ CoordinatorAgent (loop decision)
                    │    scores   │
                    │             │
  SynthesisAgent ──▶│ research_   │──▶ FormattingAgent
                    │    agenda   │
                    │             │
  CoordinatorAgent ▶│ table_count │──▶ FormattingAgent
                    └─────────────┘
```

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

Otimizado para GPU de alto desempenho:

| Parâmetro | Valor |
|-----------|-------|
| Modelo global | `qwen3:8b` |
| Modelos complexos | `mixtral:8x22b` (critical, synthesis, debate) |
| Contexto | `32768` tokens |
| Timeout | 900s |
| Embedding | `all-mpnet-base-v2` |
| Chunks | 600 tokens, overlap 100 |
| Evidência top-k | 25 |
| Paralelismo | 3 workers |
| Two-pass writing | `true` |
| Pré-sumarização | `true` |
| Quality threshold | 7.0 |
| Max iterations | 3 |
| Debate | enabled, threshold 2 contradições |

---

## Modelos de Dados

```
StudyRecord ──▶ Chunk ──▶ ChunkTag
     │                        │
     │                        ▼
     │              TaxonomyEntry
     │              (prompt, folder, parent)
     │
     ├──▶ ExtractionResult ──────────────▶ CriticalAnalysis
     │      (+study_scale, geographic_      (+comparative_analysis:
     │       scope, funding_source,          [ComparativeItem])
     │       conflict_of_interest,
     │       limitations)
     │
     ├──▶ ScreeningDecision
     ├──▶ RiskOfBiasResult
     │
     └──▶ SynthesisMap ──▶ ConsensusPoint
              │          ──▶ Contradiction
              │          ──▶ KnowledgeGap
              │
              └──▶ Thesis + Grouped Gaps ──▶ Research Agenda
```

**Modelos Pydantic do sistema multi-agente (em `agents/models.py`):**

| Modelo | Campos Principais |
|--------|------------------|
| `CriticalAnalysis` | `methodological_quality_summary`, `contradictions`, `comparative_analysis`, `robustness_rating`, `contextual_factors` |
| `ComparativeItem` | `study_a`, `study_b`, `methodology_diff`, `result_diff`, `possible_cause`, `robustness_note` |
| `ContradictionDetail` | `point`, `study_a`, `finding_a`, `study_b`, `finding_b`, `possible_cause` |

---

## Sistema de Qualidade Textual

A qualidade do texto gerado é garantida por **7 camadas** (expandido no
modo multi-agente):

| # | Camada | Componente | Tipo |
|---|--------|-----------|------|
| 1 | Chain-of-Thought | `_SECTION_PROMPT` (Stages 1–3) | Prompt LLM |
| 2 | Two-Pass Writing | `_POLISH_PROMPT` | Prompt LLM |
| 3 | **Review Agent** | 6 critérios + cross-section redundancy | Prompt LLM (iterativo) |
| 4 | Refinamento | `_REFINE_PROMPT` (post-processing) | Prompt LLM |
| 5 | Coerência | `_COHERENCE_PROMPT` (por capítulo) | Prompt LLM |
| 6 | Argumentation | `_validate_argumentation` (tese → seção) | Prompt LLM |
| 7 | Cleanup | `_textual_cleanup()` | Regex/Programático |

### Anti-Redundância (3 níveis)

| Nível | Mecanismo | Escopo |
|-------|----------|--------|
| **1. ConceptRegistry** | Rastreia conceitos cobertos + seção de origem | Cross-section (escrita) |
| **2. Cross-section Review** | Resumos de seções aprovadas → ReviewAgent | Cross-section (revisão) |
| **3. Post-processing Dedup** | `_dedup_chapters()` programático | Cross-chapter (final) |

### Regras transversais (presentes em todos os prompts):
- ❌ Conectores genéricos ("Além disso", "No entanto", "Adicionalmente")
- ❌ Redundâncias e citações duplicadas
- ❌ Frases em inglês em texto português
- ❌ Erros ortográficos ("catalise" → "catálise")
- ✅ Transições narrativas substantivas
- ✅ Linguagem de hedging para resultados preliminares
- ✅ Análise crítica comparativa (laboratório vs. campo, escalabilidade)
- ✅ Terminologia consistente
