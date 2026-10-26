# Pipeline inputs — running GraphRAG V2 on any topic

Everything topic-specific is in **one JSON file** in [`topics/`](./topics). The code
([graphrag_pipeline.py](./graphrag_pipeline.py)) never needs to change.

## Quick start for a new topic

1. Copy [`topics/TEMPLATE.json`](./topics/TEMPLATE.json) to `topics/<my_topic>.json`
   ([`topics/ev_battery_recycling.json`](./topics/ev_battery_recycling.json) is a filled-in example).
2. Fill in the **required** inputs (table below), at least `topic`, `output_name`, `search_queries`,
   `topic_terms`, `entity_types`, `relation_types`, `questions`.
3. Check it without spending OpenAI credit (loads/scrapes data, chunks, estimates calls):
   ```bash
   .venv/bin/python run_pipeline.py --config topics/<my_topic>.json --dry-run
   ```
4. Run it — either in the notebook ([graphrag_ai_copyright_v2.ipynb](./graphrag_ai_copyright_v2.ipynb), set
   `CONFIG_FILE = "topics/<my_topic>.json"`) or in the terminal:
   ```bash
   .venv/bin/python run_pipeline.py --config topics/<my_topic>.json
   .venv/bin/python run_pipeline.py --config topics/<my_topic>.json --ask "another question"
   ```
5. Results appear in `outputs/<output_name>/` (see [Outputs](#outputs)).

Keys that start with `_` (e.g. `_comment`) are ignored, so you can leave notes in the file.
Unknown keys or invalid values stop the run with a message listing every problem.

## Credentials (not in the config)

| Variable | Needed for | Accepted names |
|---|---|---|
| OpenAI key | extraction, summaries, answers (every full run) | `OPENAI_API_KEY`, `OPENAI_KEY`, `OPENAI` |
| SerpAPI key | only `input_mode: "search"` | `SERPAPI_API_KEY`, `SERPAPI_KEY`, `SERPAPI` |

Recommended: add them as **GitHub Codespaces secrets** — github.com → *Settings → Codespaces → Secrets → New secret*,
give the secret access to this repository, then **restart the codespace** (secrets are injected only at start).
Alternatively create a local `.env` file (git-ignored) with `OPENAI_API_KEY=...`. Codespaces secrets win over `.env`.
Keys are never printed or written to outputs.

## Inputs

### 1. Topic & output

| Key | Required | Meaning | Example |
|---|---|---|---|
| `topic` | ✅ | One-line description of the research topic. It is inserted into every LLM prompt (extraction, summaries, answers) to keep the model focused, and used as the graph title. | `"Electric vehicle battery recycling"` |
| `output_name` | ✅ | Folder name for results: `outputs/<output_name>/`. Letters, digits, `_`, `-` only. Use a new name per topic so caches do not mix. | `"ev_battery_recycling"` |
| `output_dir` | | Parent folder for all outputs. | `"outputs"` |

### 2. Where the documents come from

| Key | Required | Meaning | Example |
|---|---|---|---|
| `input_mode` | ✅ | `"csv"` = use an existing dataset file; `"search"` = collect new documents with SerpAPI (Google) + Trafilatura. | `"search"` |
| `dataset_file` | csv mode | Path (relative to the repo) of a CSV with a **`full_text`** column. Optional columns `title`, `url`, `source`, `date` are used for citations. | `"ai_copyright_dataset.csv"` |
| `search_queries` | search mode | Google queries. Each query returns up to `results_per_query` organic results. Use several angles of the topic (actors, regulation, technology…). | `["EV battery recycling companies", "lithium-ion battery recycling regulation"]` |
| `results_per_query` | | Organic results kept per query (1–100). One SerpAPI search credit per query. | `10` |
| `include_videos` | | Also fetch transcripts of YouTube videos shown in the results. | `false` |
| `topic_terms` | search mode (recommended) | Keywords or regular expressions that an on-topic page should mention. Pages mentioning them less than `min_topic_mentions_per_1k_words` times per 1,000 words are dropped (removes Wikipedia/home pages/unrelated hits). Case-insensitive regex; write `\\b` in JSON for a word boundary. Empty list = no filter. | `["recycl", "battery", "lithium"]` |
| `min_topic_mentions_per_1k_words` | | Threshold for the filter above. Lower it if too many good pages are rejected (the dry run prints why each page was dropped). | `5` |
| `min_text_chars` | | Pages with less text are treated as login walls/bot checks and dropped. | `500` |

In search mode the scraped data is saved to `outputs/<output_name>/dataset.csv` and **reused** on later runs (no new
SerpAPI cost). Use `--refresh-data` (CLI) or `pipeline.load_documents(refresh=True)` (notebook) to scrape again.
You can also edit that CSV by hand, or point a csv-mode config at it.

### 3. Ontology — the schema of the knowledge graph

| Key | Required | Meaning |
|---|---|---|
| `entity_types` | ✅ | Object of `"TYPE_NAME": "description"`. The LLM may **only** create entities of these types (enforced by structured output). Names in `UPPER_SNAKE_CASE`. The descriptions are shown to the LLM — include examples. Typically 5–9 types; always consider `ORGANIZATION`, `PERSON`, `CONCEPT`. |
| `relation_types` | ✅ | Object of `"RELATION_NAME": "description"`. The only relationship labels the LLM may use. Write descriptions as `SOURCE_TYPE does X to TARGET_TYPE` so the direction is clear. Typically 6–10 relations. |

Tips: too few types → everything becomes `CONCEPT`; too many overlapping types → inconsistent labels. Ask yourself
which *questions* you want answered and make sure the relations needed to answer them exist.

### 4. Questions

| Key | Required | Meaning |
|---|---|---|
| `questions` | ✅ | Questions answered at the end of a full run and saved to `answers.md`. GraphRAG is strongest on **big-picture / cross-document** questions ("What are the main…", "How do X and Y differ…", "Which actors… and what are their positions?"). You can ask more later with `--ask` or in the notebook. |

### 5. Models

| Key | Default | Meaning |
|---|---|---|
| `extraction_model` | `gpt-4o-mini` | Used for high-volume steps: extraction (1 call per chunk), community reports (1 per community) and the per-community "map" step of each question. |
| `query_model` | `gpt-4o` | Used once per question to write the final answer from the most relevant community answers. |

Changing `extraction_model`, the ontology or `topic` invalidates the extraction cache (a new extraction is needed).

### 6. Size, cost and quality knobs

| Key | Default | Meaning |
|---|---|---|
| `max_articles` | `50` | Max documents used (first N after removing empty/duplicate texts). Main cost control. |
| `chunk_size` | `1024` | Chunk length in **tokens** (~750 words). Smaller = more detailed extraction but more calls. |
| `chunk_overlap` | `100` | Tokens shared by consecutive chunks so facts on a boundary are not lost. Must be `< chunk_size`. |
| `max_chunks_per_article` | `0` | Only the first N chunks of each document are processed (`0` = all). Caps cost for very long pages. |
| `max_paths_per_chunk` | `20` | Upper bound on relationships the LLM extracts per chunk. |
| `num_workers` | `4` | Parallel OpenAI requests. Lower it if you hit rate limits. |
| `max_cluster_size` | `10` | Max entities per community (hierarchical Leiden). Smaller → more, more specific communities (more summary calls); larger → fewer, broader ones. |
| `summary_context_chars` | `20000` | Max characters of evidence sent per community report. |
| `top_communities` | `10` | How many of the most relevant community answers are combined into the final answer. |

Rough cost: `extraction_calls` (printed by `--dry-run`) + number of communities + per question (communities + 1).
With `gpt-4o-mini` the AI-copyright example (21 chunks) costs on the order of a few US cents to build.

## Outputs

All in `outputs/<output_name>/`:

| File | Content |
|---|---|
| `dataset.csv` | Scraped documents (search mode only). |
| `extraction_cache.jsonl` | Raw extraction per chunk — the cache that makes re-runs free. Delete it to force re-extraction. |
| `graph.json` | Full merged graph + community ids + source list (used by `--ask` / `load_saved()`). |
| `community_reports.json` | Title + summary per community (cached by evidence). |
| `graph_data.json` | Nodes/links for the visualization. |
| `<output_name>_graph.html` | Interactive D3 graph — open in a browser. |
| `answers.md` | Questions with answers and cited sources (`[S1]` = first document of the dataset, etc.). |

Answers are research aids generated from the collected pages — verify important claims in the cited sources.
