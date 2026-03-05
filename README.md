# Summer: An Interactive System for Traceable Medical Aspect-Based Summarization

**ACL 2026 Demo Track Submission** | [Paper (arXiv: 2601.03418)](https://arxiv.org/abs/2601.03418)

> Chu, Damm, Pakull, Frihat, Li, Muhabbek, Lodde, Fuhr — University of Duisburg-Essen

## Overview

**Summer** is an interactive system demo for **traceable medical aspect-based summarization with phrase-level context attribution**. Given a randomized controlled trial (RCT) article, the system generates aspect-based summaries across 16 medical aspects with explicit links to their phrase-level contextual evidence.

### Pipeline

The system implements three context attribution strategies from the paper:

| Strategy | Steps | Description |
|----------|-------|-------------|
| **Prior** (default) | 2 steps | Retrieves sentences & phrases first, then summarizes |
| **Intrinsic** | 1 step | Generates summary, citations, and phrases together |
| **Post-Hoc** | 2 steps | Generates summary first, then retrieves evidence |

### 16 Medical Aspects

`OB` Objective · `P` Participants · `I` Intervention · `C` Comparator · `O` Outcomes · `F` Findings · `M` Medicines · `TD` Treatment Duration · `PE` Primary Endpoints · `SE` Secondary Endpoints · `FD` Follow-Up Duration · `AE` Adverse Events · `R` Randomization · `B` Blinding · `FU` Funding · `RE` Registration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
NCBI_EMAIL=your@email.com
NCBI_API_KEY=your_ncbi_api_key
```

### 3. Run the Demo

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Docker

```bash
docker compose up --build
```

## Architecture

```
pcoa/
├── app.py                    # FastAPI backend (REST API + file upload endpoints)
├── static/
│   └── index.html            # Single-page frontend
├── pcoa/
│   ├── config.py             # 16 aspect definitions & environment config
│   ├── pubmed.py             # PubMed E-utilities (esearch + efetch)
│   ├── fulltext.py           # Fulltext acquisition (PMC BioC XML + PDF upload)
│   ├── text_processing.py    # NLTK sentence splitting & indexing
│   ├── prompts.py            # Original prompts from paper (Tables 6-11, all 3 strategies)
│   ├── prompt_prior.py       # Per-aspect guidance configs for Prior strategy
│   ├── prompt_intrinsic.py   # Prompt builder for Intrinsic strategy
│   ├── llm.py                # LLM client (OpenAI / Anthropic / xAI) & response parsing
│   └── pipeline.py           # 3 attribution strategies orchestration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## How It Works

### 1. Article Input

Four ways to load an article:

| Method | API | Description |
|--------|-----|-------------|
| PubMed keyword search | `POST /api/search` | Queries PubMed E-utilities (esearch + efetch) |
| Fetch by PMID | `POST /api/fetch` | Retrieves a single article by PubMed ID |
| PMC fulltext | `POST /api/fetch-pmc` | Fetches full article text via PMC BioC XML |
| PDF upload | `POST /api/upload-pdf` | Extracts text from an uploaded PDF (max 20 MB) |

### 2. Sentence Indexing

The abstract or fulltext is split into individually indexed sentences using NLTK (`text_processing.py`). Each sentence receives a zero-based index used throughout for citation tracking.

### 3. Summarization and Attribution (`POST /api/analyze`)

The user selects one or more of the 16 medical aspects and a strategy. `analyze_article()` in `pipeline.py` iterates over each aspect and runs the chosen strategy:

**Prior** (default, 2 LLM calls per aspect)
1. `prompt_prior_step1` → LLM identifies relevant sentence indices and key phrases from the indexed abstract
2. `prompt_prior_step2` → LLM generates the summary using only the retrieved sentences and phrases

**Intrinsic** (1 LLM call per aspect)
- `prompt_intrinsic` → LLM jointly generates the summary, sentence indices, and key phrases in a single pass

**Post-Hoc** (2 LLM calls per aspect)
1. `build_posthoc_step1_prompt` → LLM generates the summary from the full abstract
2. `build_posthoc_step2_prompt` → LLM retrieves supporting sentence indices and key phrases for the generated summary

Each strategy returns an `AspectResult` containing `summary`, `sentence_indices`, `cited_sentences`, and `key_phrases`.

### 4. Evaluation (`POST /api/evaluate`, optional)

The summary, indexed sentences, and key phrases are sent to the external evaluation service (`EVAL_API_URL`), which returns three metrics:

- **ECR** (Evidence Coverage Ratio)
- **SSR** (Summary Support Ratio)
- **CPR** (Contributory Phrase Ratio)

Along with per-claim citation mappings and contributory key phrase lists.

### 5. Visualization

Results are rendered in the frontend with color-coded highlights: cited sentences in red and contributory key phrases in blue, linked back to their positions in the original abstract.

## Citation

```bibtex
@article{chu2026pcoa,
  title={PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution},
  author={Chu, Bohao and Frihat, Sameh and Pakull, Tabea MG and Damm, Hendrik and Li, Meijie and Muhabbek, Ula and Lodde, Georg and Fuhr, Norbert},
  journal={arXiv preprint arXiv:2601.03418},
  year={2026}
}
```

