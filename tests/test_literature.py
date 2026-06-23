from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bioml_agent.literature import (
    ArxivLiteratureProvider,
    CachedLiteratureProvider,
    FixtureLiteratureProvider,
    FrozenLiteratureProvider,
    PubMedLiteratureProvider,
    SemanticScholarLiteratureProvider,
    export_cache_snapshot,
    parse_arxiv_atom,
    parse_pubmed_esearch,
    parse_pubmed_xml,
    parse_semantic_scholar_search,
)


ARXIV_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.01234v1</id>
    <published>2025-01-01T00:00:00Z</published>
    <title>Retrieval Graph Models for Gene Expression Prediction</title>
    <summary>
      We study retrieval and graph neural models for promoter-aware gene expression prediction.
      The benchmark includes multiple tissue splits.
    </summary>
    <category term="cs.LG"/>
    <category term="q-bio.GN"/>
  </entry>
</feed>
"""

S2_SAMPLE = """{
  "total": 1,
  "data": [
    {
      "paperId": "abc123",
      "title": "Graph retrieval for promoter gene expression prediction",
      "year": 2026,
      "abstract": "We combine graph retrieval and promoter context for gene expression prediction. Results are evaluated on tissue benchmarks.",
      "citationCount": 12,
      "url": "https://www.semanticscholar.org/paper/abc123",
      "fieldsOfStudy": ["Computer Science", "Biology"],
      "venue": "ExampleConf",
      "externalIds": {"ArXiv": "2601.00001"}
    }
  ]
}"""

S2_RERANK_SAMPLE = """{
  "total": 2,
  "data": [
    {
      "paperId": "low-cite",
      "title": "Graph retrieval for promoter expression",
      "year": 2024,
      "abstract": "Graph retrieval is evaluated for promoter gene expression.",
      "citationCount": 2,
      "url": "https://www.semanticscholar.org/paper/low-cite",
      "fieldsOfStudy": ["Computer Science", "Biology"],
      "venue": "Workshop",
      "externalIds": {}
    },
    {
      "paperId": "high-cite",
      "title": "Graph retrieval for promoter expression",
      "year": 2024,
      "abstract": "Graph retrieval is evaluated for promoter gene expression.",
      "citationCount": 500,
      "url": "https://www.semanticscholar.org/paper/high-cite",
      "fieldsOfStudy": ["Computer Science", "Biology"],
      "venue": "TopConf",
      "externalIds": {"DOI": "10.0000/example"}
    }
  ]
}"""

PUBMED_ESEARCH_SAMPLE = """{
  "esearchresult": {
    "count": "2",
    "retmax": "2",
    "idlist": ["111", "222"]
  }
}"""

PUBMED_EFETCH_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <Journal>
          <Title>Genome Biology</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Promoter regulatory graphs for gene expression prediction</ArticleTitle>
        <Abstract>
          <AbstractText>We evaluate regulatory graph features for promoter gene expression prediction across tissues.</AbstractText>
        </Abstract>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Gene Expression Profiling</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/pubmed-example</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class LiteratureProviderTest(unittest.TestCase):
    def test_fixture_provider_ranks_retrieval_query(self) -> None:
        papers = FixtureLiteratureProvider().search("retrieval-aware conditioning", top_k=2)
        self.assertEqual(papers[0].paper_id, "tabr-2023")

    def test_parse_arxiv_atom_maps_candidate_fields(self) -> None:
        papers = parse_arxiv_atom(ARXIV_SAMPLE)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.paper_id, "arxiv:2501.01234v1")
        self.assertEqual(paper.year, 2025)
        self.assertEqual(paper.source_url, "http://arxiv.org/abs/2501.01234v1")
        self.assertEqual(paper.venue, "arXiv")
        self.assertIn("retrieval", paper.tags)
        self.assertIn("graph", paper.tags)
        self.assertIn("gene_expression", paper.tags)
        self.assertIn("q-bio.GN", paper.tags)

    def test_arxiv_provider_uses_fetcher_and_top_k(self) -> None:
        seen_urls: list[str] = []

        def fake_fetcher(url: str) -> bytes:
            seen_urls.append(url)
            return ARXIV_SAMPLE.encode("utf-8")

        provider = ArxivLiteratureProvider(fetcher=fake_fetcher)
        papers = provider.search("gene expression graph retrieval", top_k=1)
        self.assertEqual(len(papers), 1)
        self.assertIn("max_results=1", seen_urls[0])
        self.assertIn("search_query=all%3Agene+expression+graph+retrieval", seen_urls[0])

    def test_parse_semantic_scholar_search_maps_candidate_fields(self) -> None:
        papers = parse_semantic_scholar_search(S2_SAMPLE)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.paper_id, "s2:abc123")
        self.assertEqual(paper.year, 2026)
        self.assertIn("graph", paper.tags)
        self.assertIn("retrieval", paper.tags)
        self.assertIn("computer_science", paper.tags)
        self.assertEqual(paper.source_url, "https://www.semanticscholar.org/paper/abc123")
        self.assertEqual(paper.venue, "ExampleConf")
        self.assertEqual(paper.citation_count, 12)

    def test_semantic_scholar_provider_uses_fetcher_and_fields(self) -> None:
        seen_urls: list[str] = []

        def fake_fetcher(url: str) -> bytes:
            seen_urls.append(url)
            return S2_SAMPLE.encode("utf-8")

        provider = SemanticScholarLiteratureProvider(fetcher=fake_fetcher)
        papers = provider.search("promoter graph retrieval", top_k=2)
        self.assertEqual(len(papers), 1)
        self.assertIn("limit=2", seen_urls[0])
        self.assertIn("citationCount", seen_urls[0])

    def test_semantic_scholar_provider_reranks_by_citation_when_relevance_ties(self) -> None:
        def fake_fetcher(url: str) -> bytes:
            return S2_RERANK_SAMPLE.encode("utf-8")

        provider = SemanticScholarLiteratureProvider(fetcher=fake_fetcher)
        papers = provider.search("graph retrieval promoter expression", top_k=2)
        self.assertEqual([paper.paper_id for paper in papers], ["s2:high-cite", "s2:low-cite"])
        self.assertEqual(papers[0].doi, "10.0000/example")

    def test_parse_pubmed_esearch_returns_ids(self) -> None:
        self.assertEqual(parse_pubmed_esearch(PUBMED_ESEARCH_SAMPLE), ["111", "222"])

    def test_parse_pubmed_xml_maps_candidate_fields(self) -> None:
        papers = parse_pubmed_xml(PUBMED_EFETCH_SAMPLE)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.paper_id, "pmid:111")
        self.assertEqual(paper.year, 2024)
        self.assertEqual(paper.source_url, "https://pubmed.ncbi.nlm.nih.gov/111/")
        self.assertEqual(paper.doi, "10.1000/pubmed-example")
        self.assertEqual(paper.venue, "Genome Biology")
        self.assertIn("gene_expression", paper.tags)
        self.assertIn("graph", paper.tags)

    def test_pubmed_provider_runs_esearch_then_efetch(self) -> None:
        seen_urls: list[str] = []

        def fake_fetcher(url: str) -> bytes:
            seen_urls.append(url)
            if "esearch.fcgi" in url:
                return PUBMED_ESEARCH_SAMPLE.encode("utf-8")
            return PUBMED_EFETCH_SAMPLE.encode("utf-8")

        provider = PubMedLiteratureProvider(fetcher=fake_fetcher, email="test@example.com")
        papers = provider.search("promoter gene expression graph", top_k=2)
        self.assertEqual([paper.paper_id for paper in papers], ["pmid:111"])
        self.assertIn("db=pubmed", seen_urls[0])
        self.assertIn("retmode=json", seen_urls[0])
        self.assertIn("email=test%40example.com", seen_urls[0])
        self.assertIn("efetch.fcgi", seen_urls[1])
        self.assertIn("id=111%2C222", seen_urls[1])

    def test_cached_provider_reuses_stored_results(self) -> None:
        calls = 0

        class CountingProvider:
            def search(self, query: str, top_k: int = 4):
                nonlocal calls
                calls += 1
                return FixtureLiteratureProvider().search(query, top_k)

        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = CachedLiteratureProvider(
                provider=CountingProvider(),
                cache_dir=Path(tmp_dir),
                namespace="fixture-test",
            )
            first = provider.search("retrieval-aware conditioning", top_k=1)
            second = provider.search("retrieval-aware conditioning", top_k=1)
        self.assertEqual(calls, 1)
        self.assertEqual(first[0].paper_id, second[0].paper_id)

    def test_cache_snapshot_exports_and_replays_cached_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "cache"
            snapshot = Path(tmp_dir) / "snapshot.jsonl"
            provider = CachedLiteratureProvider(
                provider=FixtureLiteratureProvider(),
                cache_dir=cache_dir,
                namespace="fixture-test",
            )

            provider.search("retrieval-aware conditioning", top_k=1)
            cache_files = list(cache_dir.glob("fixture-test/*.json"))
            self.assertEqual(len(cache_files), 1)
            raw_cache = json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertEqual(raw_cache["query"], "retrieval-aware conditioning")
            self.assertEqual(raw_cache["top_k"], 1)

            count = export_cache_snapshot(cache_dir, snapshot)
            replay = FrozenLiteratureProvider(snapshot)

        self.assertEqual(count, 1)
        self.assertEqual(replay.search("retrieval-aware conditioning", top_k=1)[0].paper_id, "tabr-2023")
        self.assertEqual(replay.search("unseen query"), [])


if __name__ == "__main__":
    unittest.main()
