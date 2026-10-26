#!/usr/bin/env python
"""Command-line runner for the GraphRAG V2 pipeline.

Examples (run with the project venv: .venv/bin/python run_pipeline.py ...):

    # Check config, credentials, dataset and chunking — makes no OpenAI calls
    python run_pipeline.py --config topics/ai_copyright.json --dry-run

    # Full pipeline: extract → graph → communities → summaries → HTML → answers
    python run_pipeline.py --config topics/ai_copyright.json

    # Ask a new question using the saved graph (no re-extraction)
    python run_pipeline.py --config topics/ai_copyright.json --ask "What is fair use?"

    # Re-scrape a search-mode topic (otherwise outputs/<name>/dataset.csv is reused)
    python run_pipeline.py --config topics/ev_battery_recycling.json --refresh-data
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

from graphrag_pipeline import (  # noqa: E402
    GraphRAGPipeline, MissingCredentialError, config_summary, load_config, load_credentials,
)


async def main(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(config_summary(config), "\n")
    creds = load_credentials()
    print(f"Credentials: OpenAI={'found' if creds['openai'] else 'MISSING'}, "
          f"SerpAPI={'found' if creds['serpapi'] else 'missing (only needed for input_mode=search)'}\n")
    pipeline = GraphRAGPipeline(config)

    if args.dry_run:
        pipeline.load_documents(refresh=args.refresh_data)
        pipeline.chunk()
        print("\nEstimated OpenAI work:", json.dumps(pipeline.estimate_calls(), indent=2))
        print("Plus one summary call per community and, per question, one call per community "
              "(map) + one final call (reduce).")
        return 0

    if args.ask:
        pipeline.load_saved()
        for question in args.ask:
            print(f"\n❓ {question}")
            print(pipeline.format_answer(await pipeline.aquery(question, verbose=True)))
        pipeline.save_answers()
        return 0

    load_credentials(require_openai=True)
    await pipeline.arun(refresh_data=args.refresh_data, ask_questions=not args.no_questions)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="topics/ai_copyright.json", help="topic config JSON")
    parser.add_argument("--dry-run", action="store_true", help="load data + chunk only; no OpenAI calls")
    parser.add_argument("--refresh-data", action="store_true", help="re-run SerpAPI scraping (search mode)")
    parser.add_argument("--no-questions", action="store_true", help="build the graph but skip the questions")
    parser.add_argument("--ask", action="append", metavar="QUESTION", help="query the saved graph (repeatable)")
    try:
        sys.exit(asyncio.run(main(parser.parse_args())))
    except (MissingCredentialError, FileNotFoundError, ValueError) as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)
