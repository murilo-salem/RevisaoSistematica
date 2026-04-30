# Arquitetura do Sistema — Revisão Sistemática Automatizada

Documento técnico completo da arquitetura implementada em `/home/murilo/Documentos/RevisaoSistematica`, cobrindo camadas, processos, metodologia, contratos de dados, funções e operação.

## Sumário

1. [Objetivo e Escopo](#1-objetivo-e-escopo)
2. [Visão Arquitetural](#2-visão-arquitetural)
3. [Estrutura do Repositório](#3-estrutura-do-repositório)
4. [Modos de Operação e Execução](#4-modos-de-operação-e-execução)
5. [Arquitetura de Dados e Persistência](#5-arquitetura-de-dados-e-persistência)
6. [Pipeline Clássica (Monolítica)](#6-pipeline-clássica-monolítica)
7. [Metodologia por Etapa](#7-metodologia-por-etapa)
8. [Sistema Multi-Agente](#8-sistema-multi-agente)
9. [Configuração e Perfis](#9-configuração-e-perfis)
10. [Observabilidade, Estado e Recuperação](#10-observabilidade-estado-e-recuperação)
11. [Catálogo de Funções e Classes](#11-catálogo-de-funções-e-classes)
12. [Dependências Técnicas](#12-dependências-técnicas)
13. [Extensibilidade e Pontos de Evolução](#13-extensibilidade-e-pontos-de-evolução)

---

## 1. Objetivo e Escopo

O sistema automatiza revisões sistemáticas de literatura com duas arquiteturas complementares:

- Arquitetura clássica (pipeline procedural em estágios), voltada a execução direta e rastreável.
- Arquitetura multi-agente (coordenação por Blackboard), voltada a refino iterativo de qualidade textual e argumentativa.

Ele cobre o ciclo completo:

- Definição de pergunta (PICO) e estratégia de busca
- Coleta/ingestão de estudos
- Deduplicação
- Triagem
- Extração estruturada
- Avaliação de risco de viés
- Síntese quantitativa/qualitativa
- Escrita e refinamento do texto final
- Geração de artefatos (Markdown/LaTeX, logs, estado, relatórios)

---

## 2. Visão Arquitetural

### 2.1 Camadas

1. **Interface de execução (CLI)**
- `src/main.py`
- Resolve modo de operação, perfil, retomada e configuração.

2. **Orquestração**
- Clássica: `src/orchestrator.py`
- Multi-agente: `src/agents/pipeline.py` + `src/agents/coordinator_agent.py`

3. **Processamento de evidência**
- Busca, deduplicação, triagem, extração, síntese, chunking/tagging.

4. **Geração de documento**
- Escrita por seções, pós-processamento, montagem final, conversão Markdown→LaTeX.

5. **Persistência e auditoria**
- SQLite (`data/raw/studies.db`), JSON intermediários/finais, audit log, estado de pipeline, blackboard.

### 2.2 Princípios de projeto observados no código

- **Determinismo estrutural**: prompts pedem JSON sempre que possível; há parsing com fallback.
- **Fallbacks explícitos**: quando LLM/dependência falha, o pipeline tenta alternativas (ex.: dedup exata, síntese básica).
- **Rastreabilidade**: cada etapa registra tempo e métricas em arquivos de estado/log.
- **Componentização**: módulos separados por responsabilidade única.
- **Compatibilidade local**: fluxo offline com taxonomia e dados locais sem depender de PubMed.

---

## 3. Estrutura do Repositório

```text
RevisaoSistematica/
├── config/
│   ├── config.yaml
│   ├── config_5090.yaml
│   ├── taxonomia.json
│   └── biodiesel_prompts.json
├── data/
│   ├── pdfs/
│   ├── raw/
│   │   └── studies.db
│   ├── processed/
│   ├── organized/
│   └── results/
├── src/
│   ├── main.py
│   ├── orchestrator.py
│   ├── query_builder.py
│   ├── retrieval.py
│   ├── deduplication.py
│   ├── screening.py
│   ├── extraction.py
│   ├── risk_of_bias.py
│   ├── synthesis.py
│   ├── manuscript.py
│   ├── local_loader.py
│   ├── content_analyzer.py
│   ├── evidence_synthesizer.py
│   ├── review_writer.py
│   ├── post_processor.py
│   ├── organizer.py
│   ├── table_generator.py
│   ├── concept_registry.py
│   ├── md2latex.py
│   ├── pdf_converter.py
│   ├── utils.py
│   └── agents/
│       ├── base_agent.py
│       ├── blackboard.py
│       ├── coordinator_agent.py
│       ├── extraction_agent.py
│       ├── mapping_agent.py
│       ├── critical_agent.py
│       ├── synthesis_agent.py
│       ├── writing_agent.py
│       ├── review_agent.py
│       ├── debate_agent.py
│       ├── formatting_agent.py
│       ├── models.py
│       └── pipeline.py
├── templates/
│   └── manuscript.tex.jinja
├── requirements.txt
├── README.md
└── Arquitetura.md
```

---

## 4. Modos de Operação e Execução

### 4.1 Comandos principais

- Modo online (PubMed + LLM):
```bash
python src/main.py --topic "..."
```

- Modo local keyword (taxonomia por palavras-chave):
```bash
python src/main.py --local
```

- Modo local outline (escrita por seções):
```bash
python src/main.py --local --taxonomy config/biodiesel_prompts.json
```

- Modo local multi-agente:
```bash
python src/main.py --local --multi-agent --taxonomy config/biodiesel_prompts.json
```

- Retomar execução anterior:
```bash
python src/main.py --resume
```

- Perfil de hardware:
```bash
python src/main.py --local --profile 5090 --taxonomy config/biodiesel_prompts.json
```

- Forçar CPU:
```bash
python src/main.py --cpu
```

### 4.2 Resolução de modo no `main.py`

1. Carrega config (`--profile` sobrescreve `--config`).
2. Aplica `--cpu` em `cfg.system.force_cpu`.
3. Detecta capacidades de runtime (PyTorch/CUDA/SentenceTransformers).
4. Se `--local --multi-agent`: chama `run_multi_agent_pipeline`.
5. Se `--local`: chama `run_pipeline_local`.
6. Se `--resume`: lê `pipeline_state` e retoma modo gravado.
7. Caso contrário: fluxo online (`run_pipeline`).

---

## 5. Arquitetura de Dados e Persistência

### 5.1 Modelos base (`utils.py`)

- `PICOModel`: population, intervention, comparison, outcome, query, topic.
- `StudyRecord`: pmid, title, authors, journal, year, abstract, doi, raw_text.
- `ScreeningDecision`: decisão de triagem com confiança e justificativa.
- `ExtractionResult`: campos clínicos/metodológicos + metadados críticos.
- `RiskOfBiasItem` e `RiskOfBiasResult`.
- `TaxonomyEntry`: prompt/folder/parent.
- `Chunk` e `ChunkTag`: unidade semântica e associação temática.

### 5.2 Banco SQLite (`data/raw/studies.db`)

Tabelas criadas em `init_database`:

- `raw_studies`
- `dedup_log`
- `screening_decisions`
- `chunks`
- `chunk_tags`

### 5.3 Artefatos JSON mais importantes

- `data/processed/pico.json`
- `data/results/prisma.json`
- `data/processed/extracted_data.json`
- `data/processed/risk_of_bias.json`
- `data/results/meta_analysis.json` ou `data/results/thematic_analysis.json`
- `data/processed/chunks.json`
- `data/processed/chunk_tags.json`
- `data/processed/coverage_report.json`
- `data/processed/synthesis_map.json`
- `data/results/review_sections.json`
- `data/results/systematic_review.md`
- `data/results/pipeline_state.json`
- `data/results/audit.log`
- `data/processed/blackboard.json` (modo multi-agente)

---

## 6. Pipeline Clássica (Monolítica)

### 6.1 Fluxo online (`orchestrator.run_pipeline`)

Estágios implementados:

1. Query Builder (`query_builder.build_query`)
2. Retrieval PubMed (`retrieval.search_pubmed`)
3. Dedup (`deduplication.deduplicate`)
4. Screening (`screening.screen_studies`)
5. Extraction (`extraction.extract_data`)
6. Risk of Bias (`risk_of_bias.assess_risk_of_bias`)
7. Synthesis + Manuscript (`synthesis.run_synthesis` + `manuscript.generate_manuscript`)

Comportamento de controle:

- Se `retrieval` não retorna estudos: encerra com `status = no_results`.
- Se `screening` não inclui estudos: encerra com `status = no_included`.
- Cada estágio atualiza `pipeline_state` com tempo e contagens.

### 6.2 Fluxo local (`orchestrator.run_pipeline_local`)

O modo local bifurca por tipo de taxonomia.

#### 6.2.1 Outline mode (`taxonomy.type == "outline"`)

Fluxo implementado:

0. Conversão opcional de PDF (`pdf_converter.convert_pdfs`)
1. Carga local (`local_loader.load_local_studies`)
2. Dedup
3. Content analysis (`content_analyzer.analyze_and_chunk`)
4. Chunking (interno ao estágio 3)
5. Tag mapping (interno ao estágio 3)
6. Correlation/coverage report
7. Evidence synthesis por tema (`evidence_synthesizer.synthesize_all_themes`)
8. Organização por pastas (`organizer.organize_by_taxonomy`)
9. Escrita por seções (`review_writer.write_review`)
10. Assembly (já incluída no writer)

#### 6.2.2 Keyword mode

Fluxo implementado:

1. Carga local
2. Dedup
3. Screening por taxonomia (`screen_studies_by_taxonomy`)
4. Extraction
5. Synthesis

Fallbacks no keyword mode:

- Se extração LLM falha: cria `ExtractionResult` mínimo automático.
- Se síntese falha: salva resumo local básico.

---

## 7. Metodologia por Etapa

### 7.1 Formulação de pergunta e busca (`query_builder.py`)

- Prompt orienta o LLM a gerar JSON com PICO + query booleana para PubMed.
- Parsing em 3 níveis:
  - JSON direto
  - Regex de chaves
  - fallback final usando resposta bruta como query
- Saída persistida em `pico.json`.

### 7.2 Recuperação de literatura (`retrieval.py`)

- Usa `Bio.Entrez` (`esearch` + `efetch` XML).
- Parsing estruturado de artigo para `StudyRecord`.
- Persistência imediata em SQLite (`raw_studies`).
- Gera `retrieval_log.json` com query e contagens.

### 7.3 Deduplicação (`deduplication.py`)

- Preferencial: embeddings de `title + abstract` e similaridade cosseno.
- Limiar configurável: `deduplication.similarity_threshold`.
- Fallback robusto: dedup exata por título normalizado.
- Logs de remoção em `dedup_log`.

### 7.4 Triagem (`screening.py`)

Dois métodos:

1. **LLM + limiares**
- Gera JSON `{decision, confidence, justification}`.
- Regras:
  - include se `decision=include` e confiança >= `threshold_include`
  - exclude se confiança <= `threshold_exclude` ou `decision=exclude`
  - ambíguo: mantém incluído para revisão humana

2. **Taxonomia por regras**
- Score por cobertura de keywords.
- Regras de inclusão/exclusão (`classification_rules`).

Sempre salva dados PRISMA em JSON.

### 7.5 Extração estruturada (`extraction.py`)

- Prompt exige JSON com desenho de estudo, amostra, efeitos, CI, p-value e metadados críticos.
- Parsing para `ExtractionResult` (Pydantic).
- Falhas de parse são registradas em `extraction_errors.json`.

### 7.6 Risco de viés (`risk_of_bias.py`)

- Prompt em cinco domínios: selection, performance, detection, attrition, reporting.
- Saída esperada: array JSON de itens.
- Fallback: todos domínios como `unclear` se parse falhar.
- Rating global heurístico:
  - qualquer `high` -> `high`
  - >=2 `unclear` -> `moderate`
  - caso contrário -> `low`

### 7.7 Síntese quantitativa/qualitativa (`synthesis.py`)

**Meta-análise** quando há dados numéricos suficientes:

- Critério: pelo menos `synthesis.min_studies_for_meta` com `effect_size + CI`.
- Método: efeito fixo por variância inversa.
- Estimativas: efeito combinado, IC, erro-padrão, Q, I², tau².
- Gera `forest_plot.png`.

**Síntese temática** quando dados insuficientes:

- Prompt qualitativo para temas, convergências e divergências.
- Salva `thematic_analysis.json`.

### 7.8 Geração de manuscrito LaTeX (`manuscript.py`)

- Quatro prompts dedicados: Introduction, Methods, Results, Discussion.
- Template Jinja2: `templates/manuscript.tex.jinja`.
- Se meta-análise existe, inclui figura de forest plot.
- Salva seções separadas em `manuscript_sections.json`.

### 7.9 Ingestão local e taxonomia (`local_loader.py`)

Suporte a:

- `.txt` (arquivo por estudo)
- `.json`
- `.csv`
- `.bib`

Taxonomia:

- JSON e Markdown.
- Suporta formato outline (lista de prompts por seção) e keyword.

### 7.10 Análise de conteúdo e mapeamento (`content_analyzer.py`)

- Chunking com sobreposição por sentenças.
- Embeddings para chunks e prompts.
- Similaridade cosseno chunk→prompt.
- Atribuição top-k tags acima de threshold.
- Relatório de cobertura por seção com alertas de lacuna.
- Persistência em DB e JSON.

### 7.11 Síntese de evidência por tema (`evidence_synthesizer.py`)

Produz `SynthesisMap` com:

- `consensus_points`
- `contradictions`
- `knowledge_gaps`

Metodologia:

- Reúne trechos mais relevantes por tema (`top_k`).
- Prompt exige JSON estruturado.
- Fallback para lacuna explícita quando não há evidência.

### 7.12 Escrita por seções (`review_writer.py`)

Processo implementado:

1. Coleta de evidências top-k por seção
2. Pré-sumarização opcional de chunks
3. Escrita por prompt multi-estágio
4. Polimento opcional (2º passe)
5. Extração de conceitos para reduzir redundância cross-section
6. Detecção opcional de tabela comparativa
7. Montagem do documento final em Markdown
8. Geração de resumo executivo, limitações e agenda de pesquisa

Recursos de metodologia textual:

- Tese por capítulo (`_generate_chapter_thesis`)
- Registro de conceitos (`ConceptRegistry`)
- Tabelas comparativas (`TableRegistry` + `table_generator`)

### 7.13 Pós-processamento (`post_processor.py`)

Pipeline de refino:

1. Refino seção a seção via LLM
2. Deduplicação intra-capítulo
3. Coerência entre seções do capítulo
4. Validação de argumentação frente à tese do capítulo
5. Limpeza programática determinística (regex)

Saídas:

- Pode preservar versão v1 (`.v1.md`)
- Reescreve arquivo refinado

### 7.14 Geração de tabelas (`table_generator.py`)

- Detecta oportunidade de tabela comparativa por tema.
- Prompt retorna `should_create_table` + especificação de colunas/linhas.
- Gera Markdown tabular com caption e fontes.

### 7.15 Conversões de formato

- `pdf_converter.py`: PDF→TXT com fallback PyMuPDF→pdfminer.
- `md2latex.py`: Markdown→LaTeX com escape de caracteres, headings, listas e idioma via `babel`.

---

## 8. Sistema Multi-Agente

### 8.1 Objetivo

Substituir a escrita monolítica por um fluxo iterativo com agentes especializados, feedback explícito e critério de qualidade por seção.

### 8.2 Protocolo comum (`agents/base_agent.py`)

- `Message`: envelope de tarefa, payload, origem, iteração e feedback.
- `AgentResult`: resposta padrão com sucesso, dados, erros e métricas.
- `BaseAgent`: API `process`, helper `call_llm` com override por agente e medição de tempo.

### 8.3 Blackboard (`agents/blackboard.py`)

Memória compartilhada com:

- Contadores globais (artigos/chunks/tags)
- Saídas por fase (extração, mapeamento, análise crítica, síntese)
- Drafts, revisões, iterações e seções aprovadas
- Documento final, agenda de pesquisa, contagem de tabelas
- Registro de eventos e `ConceptRegistry`

Persistência parcial em `data/processed/blackboard.json`.

### 8.4 Orquestração central (`CoordinatorAgent`)

Fases implementadas em `run`:

1. Extraction
2. Mapping
3. Loop por tema: Critical -> Synthesis -> Writing -> Review (iterativo)
4. Detecção de tabelas
5. Consolidação de agenda de pesquisa
6. Debate opcional para temas controversos
7. Formatting & assembly

Critério de aprovação por seção:

- Se score da revisão >= `multi_agent.quality_threshold`, seção é aprovada.
- Caso contrário, feedback alimenta nova iteração de escrita.
- Ao atingir `max_iterations`, aceita o último draft.

### 8.5 Agentes e responsabilidades

- `ExtractionAgent`: orquestra `extract_data` + `assess_risk_of_bias`.
- `MappingAgent`: usa chunk/tag (ou recalcula) e detecta contradições preliminares.
- `CriticalAgent`: análise metodológica profunda e robustez por tema.
- `SynthesisAgent`: síntese tema-a-tema + tese + prioridades de pesquisa.
- `WritingAgent`: escrita da seção com contexto de conceitos cobertos e feedback.
- `ReviewAgent`: avaliação de qualidade em critérios múltiplos e feedback acionável.
- `DebateAgent`: seção de debate para controvérsias de alta intensidade.
- `FormattingAgent`: montagem final + pós-processamento + agenda consolidada.

### 8.6 Fluxo multi-agente em alto nível

```text
preprocessamento (load/dedup/chunk/tag)
          |
          v
  [CoordinatorAgent]
          |
          +--> ExtractionAgent
          +--> MappingAgent
          +--> para cada tema:
                  CriticalAgent
                  SynthesisAgent
                  WritingAgent <-> ReviewAgent (loop)
          +--> DebateAgent (opcional)
          +--> FormattingAgent
          |
          v
   documento final + blackboard + logs
```

---

## 9. Configuração e Perfis

### 9.1 `config/config.yaml`

Blocos principais:

- `llm`: endpoint/modelo/timeout/seed
- `retrieval`: email/api_key/max_results/batch_size
- `deduplication`: modelo e threshold
- `screening`: limiares include/exclude
- `synthesis`: parâmetros de meta-análise
- `paths`: localizações de DB/JSON/artefatos
- `review_writer`: escrita, paralelismo, idioma, passes
- `post_processing`: habilitação e paralelismo
- `version`: versão de pipeline + hash da config

### 9.2 `config/config_5090.yaml`

Perfil de alto desempenho:

- Contexto LLM maior (`num_ctx`), timeout estendido.
- Batch sizes maiores para embeddings.
- Escrita e pós-processamento paralelos.
- Parâmetros específicos para multi-agente.

### 9.3 Seleção de modelo por agente

No multi-agente, `BaseAgent.call_llm` verifica:

- `cfg.multi_agent.agent_models.<agent_name>`
- Se definido, substitui temporariamente `cfg.llm.model` durante a chamada.

---

## 10. Observabilidade, Estado e Recuperação

### 10.1 Logging

- Logger raiz: `systematic_review`.
- Arquivo de auditoria: `data/results/audit.log`.
- Console em nível INFO.

### 10.2 Estado incremental do pipeline clássico

- `pipeline_state.json` contém:
  - tópico, modo, timestamps
  - hash da config
  - métricas por estágio (`elapsed_s`, contagens, outputs)
  - status final (`completed`, `no_results`, `no_included`, etc.)

### 10.3 Retomada (`--resume`)

- Lê `pipeline_state`.
- Recupera modo (`LOCAL`/`ONLINE`) e tópico.
- Reexecuta fluxo correspondente.

### 10.4 Recuperação no multi-agente

- Blackboard persistido com resumo operacional e log de eventos.
- Permite inspeção pós-execução do processo de decisão.

---

## 11. Catálogo de Funções e Classes

### 11.1 Núcleo e utilitários

- `src/main.py`
  - `main`

- `src/utils.py`
  - `_resolve`
  - `load_config`
  - `setup_logging`
  - `call_llm`
  - `get_db_connection`
  - `init_database`
  - `save_json`
  - `load_json`
  - `now_iso`
  - `check_system_capabilities`
  - classes: `PICOModel`, `StudyRecord`, `ScreeningDecision`, `ExtractionResult`, `RiskOfBiasItem`, `RiskOfBiasResult`, `TaxonomyEntry`, `Chunk`, `ChunkTag`

- `src/orchestrator.py`
  - `_init`
  - `_save_state`
  - `run_pipeline`
  - `run_pipeline_local`

### 11.2 Pipeline online

- `src/query_builder.py`
  - `_parse_llm_response`
  - `build_query`

- `src/retrieval.py`
  - `_text`
  - `_parse_article`
  - `search_pubmed`

- `src/deduplication.py`
  - `_deduplicate_exact`
  - `deduplicate`

- `src/screening.py`
  - `_parse_screening`
  - `screen_studies`
  - `screen_studies_by_taxonomy`

- `src/extraction.py`
  - `_parse_extraction`
  - `extract_data`

- `src/risk_of_bias.py`
  - `_parse_rob`
  - `_compute_overall_rating`
  - `assess_risk_of_bias`

- `src/synthesis.py`
  - `_inverse_variance_meta`
  - `_generate_forest_plot`
  - `_thematic_analysis`
  - `run_synthesis`

- `src/manuscript.py`
  - `_gen_introduction`
  - `_gen_methods`
  - `_gen_results`
  - `_gen_discussion`
  - `generate_manuscript`

### 11.3 Pipeline local e escrita

- `src/local_loader.py`
  - `_parse_filename_metadata`, `_format_citation`
  - `_load_txt_files`, `_load_json_file`, `_load_csv_file`, `_load_bib_file`
  - `_parse_taxonomy_md`
  - `load_taxonomy`
  - `load_local_studies`

- `src/content_analyzer.py`
  - `_chunk_text`
  - `_make_chunk_id`
  - `_load_model`
  - `_embed_texts`
  - `_cosine_matrix`
  - `analyze_and_chunk`

- `src/evidence_synthesizer.py`
  - classes: `ConsensusPoint`, `Contradiction`, `KnowledgeGap`, `SynthesisMap`
  - `_build_evidence_for_theme`
  - `_parse_synthesis`
  - `synthesize_theme`
  - `synthesize_all_themes`

- `src/review_writer.py`
  - `_make_citation`
  - `_generate_chapter_thesis`
  - `_gather_evidence`
  - `_pre_summarize_evidence`
  - `_write_section`
  - `_polish_section`
  - `_extract_concepts`
  - `_write_single_entry`
  - `_assemble_markdown`
  - `_generate_executive_summary`
  - `_generate_limitations`
  - `_generate_research_agenda`
  - `write_review`

- `src/post_processor.py`
  - `_validate_argumentation`
  - `_split_sections`
  - `_refine_section`
  - `_reassemble_markdown`
  - `_dedup_citations_in_group`
  - `_textual_cleanup`
  - `_check_chapter_coherence`
  - `_dedup_chapters`
  - `post_process_review`

- `src/organizer.py`
  - `_sanitize_dirname`
  - `organize_by_taxonomy`

- `src/table_generator.py`
  - class: `TableSpec`
  - `_gather_table_evidence`
  - `detect_table_opportunity`
  - `generate_markdown_table`
  - class: `TableRegistry`

- `src/concept_registry.py`
  - `_normalise`
  - class: `ConceptRegistry`

- `src/md2latex.py`
  - `_escape_latex`
  - `_convert_inline`
  - `_is_heading`
  - `_is_bullet`
  - `_is_reference_header`
  - `_convert_md_to_latex`
  - `convert`
  - `main`

- `src/pdf_converter.py`
  - `_extract_text_pymupdf`
  - `_extract_text_pdfminer`
  - `extract_text`
  - `convert_pdfs`
  - `main`

### 11.4 Multi-agente

- `src/agents/base_agent.py`
  - classes: `Message`, `AgentResult`, `BaseAgent`

- `src/agents/blackboard.py`
  - class: `Blackboard`
  - métodos principais: `log_event`, `save`, `load`, `get_theme_key`, `summary`

- `src/agents/models.py`
  - `ContradictionDetail`, `ThemeEvidence`, `ComparativeItem`, `CriticalAnalysis`, `CriterionScore`, `ReviewReport`

- `src/agents/pipeline.py`
  - `run_multi_agent_pipeline`

- `src/agents/coordinator_agent.py`
  - class: `CoordinatorAgent`
  - métodos principais: `register_agent`, `_dispatch`, `run`, `_process_single_theme`, `_process_themes_sequential`, `_process_themes_parallel`, `_run_debates`, `_build_approved_summary`, `_detect_tables_for_themes`, `_consolidate_research_agenda`, `_save_blackboard`

- `src/agents/extraction_agent.py`
  - class: `ExtractionAgent.process`

- `src/agents/mapping_agent.py`
  - class: `MappingAgent.process`, `_detect_contradictions`

- `src/agents/critical_agent.py`
  - class: `CriticalAgent.process`, `_format_studies`, `_parse_response`

- `src/agents/synthesis_agent.py`
  - class: `SynthesisAgent.process`, `consolidate_agenda`

- `src/agents/writing_agent.py`
  - class: `WritingAgent.process`

- `src/agents/review_agent.py`
  - class: `ReviewAgent.process`, `_parse_response`

- `src/agents/debate_agent.py`
  - class: `DebateAgent.process`

- `src/agents/formatting_agent.py`
  - class: `FormattingAgent.process`, `_convert_synthesis_maps`, `_format_research_agenda`

---

## 12. Dependências Técnicas

Arquivo `requirements.txt`:

- `pandas`, `numpy`, `scikit-learn`
- `sentence-transformers`, `faiss-cpu`
- `requests`, `pydantic`, `jinja2`, `pyyaml`, `tqdm`
- `matplotlib`, `scipy`
- `biopython`
- `PyMuPDF`, `pdfminer.six`

Dependências opcionais por capacidade:

- GPU/CUDA via PyTorch (detectado em runtime)
- SentenceTransformers para dedup/chunk mapping semântico

---

## 13. Extensibilidade e Pontos de Evolução

### 13.1 Onde extender com baixo acoplamento

- Novas fontes de busca: criar módulo análogo a `retrieval.py`.
- Novo classificador de triagem: adicionar estratégia em `screening.py`.
- Novos critérios de qualidade textual: evoluir `review_agent` e `post_processor`.
- Novos formatos de saída: reaproveitar `review_sections.json` e camada de montagem.
- Novos agentes: registrar no `CoordinatorAgent` sem alterar o protocolo base.

### 13.2 Riscos técnicos atuais (conforme implementação)

- Forte dependência da qualidade de resposta JSON do LLM.
- Latência acumulada em fluxos com muitos prompts por seção.
- Divergência potencial entre caminhos de saída (`manuscript` pode apontar para `.tex` e receber Markdown no fluxo multi-agente).
- Paralelismo exige ajuste de capacidade do backend Ollama.

### 13.3 Boas práticas para operação

- Fixar `retrieval.email` real para PubMed.
- Validar limiares de triagem por domínio.
- Revisar sempre itens ambíguos e seções sem evidência.
- Inspecionar `audit.log`, `pipeline_state.json` e `blackboard.json` em execuções longas.

---

Documento atualizado para refletir a implementação atual dos módulos em `src/` e `src/agents/`.
