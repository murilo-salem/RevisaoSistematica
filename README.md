# 📚 Systematic Review Automation Pipeline

Automated, versionable, and auditable pipeline for systematic literature reviews.

```
tema → busca → triagem → extração → síntese → manuscrito.tex
```

## 🏗️ Architecture

```
systematic-review/
├── config/
│   └── config.yaml          # Central configuration
├── data/
│   ├── raw/                  # SQLite database with raw studies
│   ├── processed/            # PICO, extracted data, risk-of-bias
│   └── results/              # Manuscript, PRISMA, forest plot, audit log
├── models/
│   └── embeddings/           # Cached sentence-transformer models
├── src/
│   ├── main.py               # CLI entry point
│   ├── orchestrator.py       # Pipeline orchestration
│   ├── query_builder.py      # PICO + Boolean query generation
│   ├── retrieval.py          # PubMed search via Entrez
│   ├── deduplication.py      # Embedding-based duplicate removal
│   ├── screening.py          # LLM-assisted study screening
│   ├── extraction.py         # Structured data extraction
│   ├── synthesis.py          # Meta-analysis / thematic synthesis
│   ├── risk_of_bias.py       # Risk-of-bias assessment
│   ├── manuscript.py         # LaTeX manuscript generation
│   └── utils.py              # Shared utilities, models, logging
├── templates/
│   └── manuscript.tex.jinja  # Jinja2 LaTeX template
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
₢₢₢₢
### Prerequisites

1. **Python 3.10+**
2. **Ollama** with `llama3:8b`:
   ```bash
   ollama pull llama3:8
   ```

### Setup

```bash
cd systematic-review
pip install -r requirements.txt
```

### Configure

Edit `config/config.yaml`:
- Set `retrieval.email` to your real email (required by NCBI)
- Optionally set `retrieval.api_key` for higher rate limits
- Adjust `llm.model` if using a different Ollama model
- source env/bin/activate

### Run

```bash
python src/main.py --topic "effect of exercise on depression in older adults"
```

Or interactively:
```bash
python src/main.py
```

### Resume a previous run

```bash
python src/main.py --resume
```

## 📊 Outputs

| File | Description |
|------|-------------|
| `data/results/manuscript.tex` | Full LaTeX manuscript |
| `data/results/prisma.json` | PRISMA flow diagram data |
| `data/results/forest_plot.png` | Forest plot (if meta-analysis) |
| `data/results/audit.log` | Complete audit trail |
| `data/processed/pico.json` | Structured PICO question |
| `data/processed/extracted_data.json` | Extracted study data |
| `data/raw/studies.db` | SQLite database with all records |

## 🔁 Reproducibility

Every run stores:
- Configuration snapshot (with hash)
- LLM model version and seed
- All intermediate results
- Timestamped audit log
- Pipeline state for resumption

## 📝 License

MIT
