"""Offline tests for graphrag_pipeline (no network, no OpenAI calls).

Run: .venv/bin/python -m pytest -q tests
"""
import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import graphrag_pipeline as gp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

VOCAB = ["OpenAI", "New York Times", "Getty Images", "Stability AI", "EU AI Act",
         "Fair Use", "US Copyright Office", "Anthropic", "Midjourney"]


class FakeLLM:
    """Mimics the two LlamaIndex LLM methods the pipeline uses."""

    def __init__(self):
        self.calls = 0

    async def astructured_predict(self, output_cls, prompt, llm_kwargs=None, **kwargs):
        self.calls += 1
        name = output_cls.__name__
        if name == "ExtractionResult":
            found = [v for v in VOCAB if v.lower() in kwargs["text"].lower()]
            ent_types = list(output_cls.model_fields["entities"].annotation.__args__[0]
                             .model_fields["type"].annotation.__args__)
            rel_type = output_cls.model_fields["relationships"].annotation.__args__[0] \
                .model_fields["relation"].annotation.__args__[0]
            entities = [{"name": v.upper() if i % 2 else v, "type": ent_types[i % len(ent_types)],
                         "description": f"{v} appears in the text."} for i, v in enumerate(found)]
            rels = [{"source": a, "target": b, "relation": rel_type, "description": f"{a} relates to {b}."}
                    for a, b in zip(found, found[1:])]
            rels.append({"source": "Unknown Entity", "target": found[0] if found else "x",
                         "relation": rel_type, "description": "dangling"})
            return output_cls(entities=entities, relationships=rels)
        if name == "CommunityReport":
            ids = sorted(set(re.findall(r"S\d+", kwargs["evidence"])))
            return output_cls(title="Theme", summary=f"Summary citing [{ids[0]}].")
        if name == "CommunityAnswer":
            return output_cls(relevance=7, answer=f"Partial answer from {kwargs['title']} [S1].")
        raise AssertionError(name)

    async def acomplete(self, prompt, **kwargs):
        self.calls += 1
        return SimpleNamespace(text="Final answer [S1] and [S99].")


@pytest.fixture()
def config(tmp_path):
    text = ("OpenAI is sued by the New York Times. Getty Images sued Stability AI. "
            "The EU AI Act and Fair Use matter. ") * 60
    df = pd.DataFrame({
        "title": ["A", "B", "<script>alert(1)</script>", "Dup"],
        "url": ["https://a", "https://b", "https://c", "https://d"],
        "full_text": [text, "Anthropic and Midjourney face the US Copyright Office. Anthropic Midjourney.",
                      "OpenAI </script><script>alert('x')</script> Anthropic", text],
    })
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    raw = json.loads((ROOT / "topics" / "ai_copyright.json").read_text())
    raw.update(dataset_file=str(csv), output_dir=str(tmp_path / "out"), chunk_size=256,
               chunk_overlap=20, max_cluster_size=3)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))
    return gp.load_config(path)


def test_all_topic_configs_are_valid():
    for path in (ROOT / "topics").glob("*.json"):
        if path.name != "TEMPLATE.json":
            gp.load_config(path)


def test_config_validation_errors(tmp_path):
    raw = json.loads((ROOT / "topics" / "ai_copyright.json").read_text())
    raw.update(output_name="bad name", chunk_overlap=5000, typo_key=1)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown config keys"):
        gp.load_config(path)
    raw.pop("typo_key")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="output_name.*\n.*chunk_overlap"):
        gp.load_config(path)


def test_credentials_aliases(monkeypatch):
    for name in gp.OPENAI_ENV_NAMES + gp.SERPAPI_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gp, "ROOT", Path("/nonexistent"))
    monkeypatch.setenv("SERPAPI", "serp-secret")
    assert gp.load_credentials(require_serpapi=True) == {"openai": False, "serpapi": True}
    assert gp.os.environ["SERPAPI_API_KEY"] == "serp-secret"
    with pytest.raises(gp.MissingCredentialError, match="OPENAI_API_KEY"):
        gp.load_credentials(require_openai=True)


def test_normalize_name():
    assert gp.normalize_name("  The  OpenAI. ") == gp.normalize_name("openai") == "openai"


def test_full_pipeline_offline(config):
    llm = FakeLLM()
    pipe = gp.GraphRAGPipeline(config, extraction_llm=llm, query_llm=llm)
    df = pipe.load_documents()
    assert len(df) == 3  # duplicate full_text removed
    assert len(pipe.chunk()) > 3  # long article was split
    asyncio.run(pipe.aextract(show_progress=False))
    extraction_calls = llm.calls
    assert extraction_calls == len(pipe.chunks)

    graph = pipe.build_graph()
    # "OpenAI" and "OPENAI" from different chunks are merged into one node
    assert "openai" in graph and sum(1 for n in graph if n == "openai") == 1
    assert set(graph.nodes["openai"]["sources"]) == {"S1", "S3"}
    assert "unknown entity" not in graph

    pipe.detect_communities()
    assert any(c >= 0 for c in pipe.community_of.values())
    asyncio.run(pipe.asummarize())
    assert pipe.reports and all("[S" in r["summary"] for r in pipe.reports.values())

    html_path = pipe.visualize()
    page = html_path.read_text()
    assert "GRAPH_DATA_PLACEHOLDER" not in page and "GRAPH_TITLE_PLACEHOLDER" not in page
    assert "<script>alert" not in page  # scraped text cannot inject markup

    result = asyncio.run(pipe.aquery("Who is involved?"))
    assert list(result["sources"]) == ["S1"]  # S99 is not a real source → dropped
    pipe.save_answers()
    assert (config.out_dir / "answers.md").read_text().count("https://a") == 1

    # Second run: everything cached → zero new LLM calls for extraction/summaries
    llm2 = FakeLLM()
    pipe2 = gp.GraphRAGPipeline(config, extraction_llm=llm2, query_llm=llm2)
    pipe2.load_documents()
    pipe2.chunk()
    asyncio.run(pipe2.aextract(show_progress=False))
    pipe2.build_graph()
    pipe2.detect_communities()
    asyncio.run(pipe2.asummarize())
    assert llm2.calls == 0
    assert pipe2.community_of == pipe.community_of  # seeded Leiden is deterministic

    # Query-only resume from saved artifacts
    pipe3 = gp.GraphRAGPipeline(config, extraction_llm=FakeLLM(), query_llm=FakeLLM()).load_saved()
    assert pipe3.graph.number_of_nodes() == graph.number_of_nodes()
    assert asyncio.run(pipe3.aquery("Q?"))["answer"]


def test_search_mode_offline(config, monkeypatch, tmp_path):
    config.input_mode, config.search_queries, config.topic_terms = "search", ["q"], ["copyright"]
    config.min_topic_mentions_per_1k_words = 1
    monkeypatch.setenv("SERPAPI_API_KEY", "x")
    monkeypatch.setattr(gp, "_serpapi_search", lambda q, k, n: {"organic_results": [
        {"link": "https://ok", "title": "ok"}, {"link": "https://off", "title": "off"},
        {"link": "https://ok", "title": "duplicate"}]})
    texts = {"https://ok": ("copyright " * 100, "success"), "https://off": ("weather " * 100, "success")}
    monkeypatch.setattr(gp, "_scrape_article", lambda url, n: texts[url])
    df = gp.GraphRAGPipeline(config).load_documents()
    assert df["url"].tolist() == ["https://ok"]
    assert config.dataset_path.exists()
