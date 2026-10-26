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


## Project Structure

```
graphrag/
├── scrape_info.ipynb              # Web scraping & text enrichment
├── graphrag_ai_copyright.ipynb    # GraphRAG pipeline, queries & visualization
├── ai_copyright_dataset.csv       # Scraped dataset (6 on-topic article records)
├── graphrag_ai_copyright_colab.ipynb # Original-style pipeline for Google Colab
├── graph_data.json                # Extracted knowledge graph (nodes + edges)
├── ai_copyright_graph.html        # Interactive visualization (generated output)
├── graph_template.html            # HTML/D3.js template for visualization
├── .env                           # API keys (not committed)
└── lib/
    ├── bindings/utils.js          # Graph interaction utilities
    ├── vis-9.1.2/                 # vis-network library
    └── tom-select/                # Dropdown UI component
```

## Prerequisites

- Python 3.10+
- A [SerpAPI](https://serpapi.com/) key (Google Search)
- An [OpenAI](https://platform.openai.com/) API key
- API key variables added in the `.env` file

## Setup

1. **Clone the repo and create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install llama-index llama-index-llms-openai graspologic \
               pandas nest-asyncio python-dotenv \
               google-search-results trafilatura youtube_transcript_api requests
   ```

3. **Configure API keys** — create a `.env` file in the project root:
   ```
   SERPAPI_KEY=your_serpapi_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

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
