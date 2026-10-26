# GraphRAG — AI Copyright & Governance Knowledge Graph

An end-to-end GraphRAG pipeline that scrapes web content about AI intellectual property and copyright, extracts entities and relationships using LLMs, builds a queryable knowledge graph, and renders it as an interactive visualization.

## Overview

This project combines web scraping, LLM-powered entity extraction, graph analysis, and interactive visualization to build a domain-specific Retrieval-Augmented Generation (RAG) system focused on AI copyright and governance topics.

## Tech Stack

| Layer | Tools |
|-------|-------|
| LLM / RAG | LlamaIndex, OpenAI (gpt-4o-mini, gpt-4o) |
| Web scraping | SerpAPI, Trafilatura, YouTube Transcript API |
| Graph analysis | NetworkX, Graspologic (Louvain community detection) |
| Visualization | D3.js v7, vis-network 9.1.2 |
| Data | Pandas, Pydantic |


## Quick start (V2, GitHub Codespaces)

```bash
bash setup_env.sh                                                         # Python 3.12 venv + Jupyter kernel
.venv/bin/python run_pipeline.py --config topics/ai_copyright.json --dry-run   # no OpenAI calls
.venv/bin/python run_pipeline.py --config topics/ai_copyright.json             # full pipeline
```

Or open [graphrag_ai_copyright_v2.ipynb](./graphrag_ai_copyright_v2.ipynb) and select the kernel
**GraphRAG (Python 3.12)**. To research another topic, copy [topics/TEMPLATE.json](./topics/TEMPLATE.json);
every input is explained in [PIPELINE_INPUTS.md](./PIPELINE_INPUTS.md).

## Project Structure

```
graphrag/
├── graphrag_ai_copyright_v2.ipynb # V2 notebook: step-by-step GraphRAG for any topic config
├── graphrag_pipeline.py           # V2 pipeline module (data, extraction, graph, communities, queries)
├── run_pipeline.py                # V2 command-line runner (--dry-run, --ask, --refresh-data)
├── PIPELINE_INPUTS.md             # Meaning of every topic input + credentials + outputs
├── topics/                        # One JSON config per topic (ai_copyright, ev_battery_recycling, TEMPLATE)
├── outputs/<output_name>/         # V2 results: caches, graph, reports, HTML, answers.md
├── graph_template_v2.html         # Topic-agnostic, XSS-safe D3 template used by V2
├── setup_env.sh                   # Creates .venv (Python 3.12) and the Jupyter kernel
├── requirements.txt               # Dependencies (requirements.lock.txt = exact tested versions)
├── tests/                         # Offline tests (fake LLM, no API calls)
├── .devcontainer/                 # Codespaces: runs setup_env.sh, recommends API-key secrets
├── scrape_info.ipynb              # V1 web scraping & text enrichment
├── graphrag_ai_copyright.ipynb    # V1 GraphRAG pipeline, queries & visualization
├── graphrag_ai_copyright_colab.ipynb # Original-style pipeline for Google Colab
├── ai_copyright_dataset.csv       # Scraped dataset (6 on-topic article records)
├── graph_data.json                # V1 extracted knowledge graph (nodes + edges)
├── ai_copyright_graph.html        # V1 interactive visualization (generated output)
├── graph_template.html            # V1 HTML/D3.js template for visualization
└── .env                           # Optional local API keys (git-ignored)
```

## Prerequisites

- **Python 3.12** (not 3.13+: `graspologic` requires Python < 3.13). The default Codespaces
  Python is 3.14, which is why installing the packages into it fails — `setup_env.sh` handles this.
- An [OpenAI](https://platform.openai.com/) API key
- A [SerpAPI](https://serpapi.com/) key (Google Search) — only for collecting new data

## Setup

1. **Create the environment** (uses [uv](https://docs.astral.sh/uv/) to get Python 3.12, installs
   `requirements.txt` pinned by `requirements.lock.txt`, registers the Jupyter kernel
   *GraphRAG (Python 3.12)*). New codespaces run this automatically via `.devcontainer/`.
   ```bash
   bash setup_env.sh
   ```

2. **Configure API keys** — preferably as GitHub Codespaces secrets (Settings → Codespaces → Secrets,
   grant access to this repository, then restart the codespace): `OPENAI_API_KEY` and `SERPAPI`
   (`SERPAPI_API_KEY` / `SERPAPI_KEY` also work). Alternatively create a git-ignored `.env` file:
   ```
   SERPAPI_KEY=your_serpapi_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Test (optional, no API calls):** `.venv/bin/python -m pytest -q tests`

## Usage

### V2 — any topic (recommended)

Run [graphrag_ai_copyright_v2.ipynb](./graphrag_ai_copyright_v2.ipynb) top to bottom, or use the CLI
[run_pipeline.py](./run_pipeline.py). The topic is chosen by a config file in [topics/](./topics)
(`input_mode: "csv"` reuses a dataset; `"search"` collects articles with SerpAPI + Trafilatura).
Extraction and community summaries are cached per chunk/community in `outputs/<output_name>/`, so
re-runs and follow-up questions (`--ask "..."`) do not pay for the graph again. See
[PIPELINE_INPUTS.md](./PIPELINE_INPUTS.md) for all inputs.

### V1 notebooks

### Google Colab

Upload [graphrag_ai_copyright_colab.ipynb](./graphrag_ai_copyright_colab.ipynb)
to Colab and run its cells in order using a CPU runtime. It preserves the
original notebook's sections 0–15, LlamaIndex extraction, hierarchical Leiden,
community summaries, D3 visualization, and two-stage query engine. No helper
modules or separate HTML template are needed.

The default `INPUT_MODE = "csv"` prompts for the provided
[ai_copyright_dataset.csv](./ai_copyright_dataset.csv); no SerpAPI key is needed.
Set `OPENAI_API_KEY` in Colab Secrets with notebook access enabled, or use the
hidden prompt. Defaults process three articles and up to three chunks per
article. Extraction, summaries, and questions make paid OpenAI calls.

For a new topic, edit `TOPIC`, `SEARCH_QUERIES`, `QUESTIONS`, `YOUR_QUESTION`,
and `OUTPUT_NAME` in section 2, and both ontology lists in section 3.
Choose `INPUT_MODE = "search"` and supply `SERPAPI_API_KEY` to discover URLs
with SerpAPI and fetch article text with Trafilatura. This optional collection
mode handles articles, not YouTube transcripts. Alternatively upload your own
CSV with a `full_text` column. Changing only the topic label does not change
the underlying evidence. Download the results ZIP before the Colab runtime ends.

The provided dataset was collected by the separate scraping workflow below:
SerpAPI supplies URLs/search metadata, while Trafilatura and the YouTube
Transcript API supply article text and video transcripts respectively.
The checked-in CSV currently contains 6 on-topic article records; quoted multiline
article text means CSV line count is not record count.

### Step 1 — Scrape content (`scrape_info.ipynb`)

Searches Google for AI copyright/IP articles and YouTube videos, then enriches them with full article text (via Trafilatura) and video transcripts (via YouTube Transcript API).
You can swap out the search queries with any other search terms relevant to your project.
Pages with fewer than 500 characters (login walls, bot checks) or fewer than 5 topic-term mentions per 1,000 words are dropped; update `TOPIC_TERMS` in the notebook when researching a different topic.

Output: `ai_copyright_dataset.csv`

### Step 2 — Build the knowledge graph (`graphrag_ai_copyright.ipynb`)

Loads the scraped dataset and runs the full GraphRAG pipeline:

- **Entity extraction** — uses GPT4o to extract entities and relationships (under the types defined in ontology).
- **Relationship extraction** — maps typed relationships between entities using Pydantic-validated schemas
- **Community detection** — runs Louvain clustering to group related entities into thematic communities
- **Community summarization** — generates LLM summaries for each cluster
- **Query engine** — answers complex questions by synthesizing context across communities using `gpt-4o`
- **Export** — writes `graph_data.json` and renders `ai_copyright_graph.html`

### Step 3 — Explore the visualization

Open `ai_copyright_graph.html` in a browser. Features:

- Force-directed graph layout (D3.js)
- Filter nodes by entity type via the sidebar legend
- Search nodes by name
- Click a node to highlight its direct connections
- Hover for entity details in a tooltip
- Adjust link distance with the slider
