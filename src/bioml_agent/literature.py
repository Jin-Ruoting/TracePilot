from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Protocol

from .fixtures import PAPER_FIXTURES
from .schemas import PaperCandidate

ATOM = "{http://www.w3.org/2005/Atom}"


class LiteratureProvider(Protocol):
    def search(self, query: str, top_k: int = 4) -> list[PaperCandidate]:
        ...


class FixtureLiteratureProvider:
    def search(self, query: str, top_k: int = 4) -> list[PaperCandidate]:
        tokens = _tokens(query)
        scored: list[tuple[int, PaperCandidate]] = []
        for paper in PAPER_FIXTURES:
            haystack = _tokens(" ".join((paper.title, paper.claim, " ".join(paper.tags))))
            score = len(tokens & haystack)
            if score:
                scored.append((score, paper))
        if not scored:
            scored = [(1, paper) for paper in PAPER_FIXTURES]
        scored.sort(key=lambda item: (-item[0], -item[1].year, item[1].paper_id))
        return [paper for _, paper in scored[:top_k]]


class ArxivLiteratureProvider:
    def __init__(
        self,
        fetcher: Callable[[str], bytes] | None = None,
        base_url: str = "https://export.arxiv.org/api/query",
    ) -> None:
        self.fetcher = fetcher or _fetch_url
        self.base_url = base_url

    def search(self, query: str, top_k: int = 4) -> list[PaperCandidate]:
        url = self._build_url(query=query, max_results=top_k)
        payload = self.fetcher(url)
        return rerank_candidates(parse_arxiv_atom(payload), query=query)[:top_k]

    def _build_url(self, query: str, max_results: int) -> str:
        clean_query = " ".join(query.strip().split())
        params = {
            "search_query": f"all:{clean_query}",
            "start": "0",
            "max_results": str(max(1, min(max_results, 25))),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        return f"{self.base_url}?{urllib.parse.urlencode(params)}"


class SemanticScholarLiteratureProvider:
    def __init__(
        self,
        fetcher: Callable[[str], bytes] | None = None,
        base_url: str = "https://api.semanticscholar.org/graph/v1/paper/search",
        api_key: str | None = None,
    ) -> None:
        self.fetcher = fetcher or _fetch_url
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("S2_API_KEY")

    def search(self, query: str, top_k: int = 4) -> list[PaperCandidate]:
        url = self._build_url(query=query, limit=top_k)
        payload = self._fetch(url)
        return rerank_candidates(parse_semantic_scholar_search(payload), query=query)[:top_k]

    def _build_url(self, query: str, limit: int) -> str:
        fields = ",".join(
            (
                "paperId",
                "title",
                "year",
                "abstract",
                "citationCount",
                "url",
                "fieldsOfStudy",
                "venue",
                "externalIds",
            )
        )
        params = {
            "query": " ".join(query.strip().split()),
            "limit": str(max(1, min(limit, 25))),
            "fields": fields,
        }
        return f"{self.base_url}?{urllib.parse.urlencode(params)}"

    def _fetch(self, url: str) -> bytes:
        if self.api_key and self.fetcher is _fetch_url:
            return _fetch_url(url, headers={"x-api-key": self.api_key})
        return self.fetcher(url)


class PubMedLiteratureProvider:
    def __init__(
        self,
        fetcher: Callable[[str], bytes] | None = None,
        search_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        fetch_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        api_key: str | None = None,
        email: str | None = None,
    ) -> None:
        self.fetcher = fetcher or _fetch_url
        self.search_base_url = search_base_url
        self.fetch_base_url = fetch_base_url
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL")

    def search(self, query: str, top_k: int = 4) -> list[PaperCandidate]:
        ids = parse_pubmed_esearch(self.fetcher(self._build_search_url(query=query, retmax=top_k)))
        if not ids:
            return []
        payload = self.fetcher(self._build_fetch_url(ids=ids[:top_k]))
        return rerank_candidates(parse_pubmed_xml(payload), query=query)[:top_k]

    def _build_search_url(self, query: str, retmax: int) -> str:
        params = {
            **self._common_params(),
            "term": " ".join(query.strip().split()),
            "retmax": str(max(1, min(retmax, 25))),
            "retmode": "json",
            "sort": "relevance",
        }
        return f"{self.search_base_url}?{urllib.parse.urlencode(params)}"

    def _build_fetch_url(self, ids: list[str]) -> str:
        params = {
            **self._common_params(),
            "id": ",".join(ids),
            "retmode": "xml",
        }
        return f"{self.fetch_base_url}?{urllib.parse.urlencode(params)}"

    def _common_params(self) -> dict[str, str]:
        params = {
            "db": "pubmed",
            "tool": "bioml_research_agent",
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params


class CachedLiteratureProvider:
    def __init__(
        self,
        provider: LiteratureProvider,
        cache_dir: str | Path,
        namespace: str,
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace

    def search(self, query: str, top_k: int = 4) -> list[PaperCandidate]:
        cache_path = self._cache_path(query=query, top_k=top_k)
        if cache_path.exists():
            return _papers_from_json(cache_path.read_text(encoding="utf-8"))
        papers = self.provider.search(query=query, top_k=top_k)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            _cache_record_to_json(source=self.namespace, query=query, top_k=top_k, papers=papers),
            encoding="utf-8",
        )
        return papers

    def _cache_path(self, query: str, top_k: int) -> Path:
        key = json.dumps(
            {"source": self.namespace, "query": query, "top_k": top_k},
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / self.namespace / f"{digest}.json"


class FrozenLiteratureProvider:
    def __init__(self, snapshot_path: str | Path) -> None:
        self.snapshot_path = Path(snapshot_path)
        self._records = _load_snapshot_records(self.snapshot_path)

    def search(self, query: str, top_k: int = 4) -> list[PaperCandidate]:
        normalized = _normalize_query(query)
        for record in self._records:
            if _normalize_query(str(record.get("query", ""))) == normalized:
                candidates = record["candidates"]
                return list(candidates[:top_k])
        return []


def export_cache_snapshot(
    cache_dir: str | Path,
    output_path: str | Path,
    namespaces: tuple[str, ...] | None = None,
) -> int:
    root = Path(cache_dir)
    records = []
    allowed = set(namespaces or ())
    for path in sorted(root.glob("*/*.json")):
        record = _cache_record_from_json(
            path.read_text(encoding="utf-8"),
            fallback_source=path.parent.name,
            cache_file=str(path),
        )
        if allowed and record["source"] not in allowed:
            continue
        if not record["query"]:
            continue
        records.append(record)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(json.dumps(_record_to_jsonable(record), ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    return len(records)


def build_literature_provider(
    source: str,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> LiteratureProvider:
    if source == "fixture":
        return FixtureLiteratureProvider()
    if source.startswith("frozen:"):
        return FrozenLiteratureProvider(source.removeprefix("frozen:"))
    if source == "arxiv":
        return _maybe_cache(
            provider=ArxivLiteratureProvider(),
            source=source,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
    if source == "semantic_scholar":
        return _maybe_cache(
            provider=SemanticScholarLiteratureProvider(),
            source=source,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
    if source == "pubmed":
        return _maybe_cache(
            provider=PubMedLiteratureProvider(),
            source=source,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
    raise ValueError(f"unknown literature source: {source}")


def _maybe_cache(
    provider: LiteratureProvider,
    source: str,
    cache_dir: str | Path | None,
    use_cache: bool,
) -> LiteratureProvider:
    if not use_cache:
        return provider
    return CachedLiteratureProvider(
        provider=provider,
        cache_dir=cache_dir or ".cache/literature",
        namespace=source,
    )


def parse_arxiv_atom(payload: bytes | str) -> list[PaperCandidate]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    root = ET.fromstring(text)
    papers: list[PaperCandidate] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = _text(entry, "title")
        summary = _normalize_whitespace(_text(entry, "summary"))
        published = _text(entry, "published")
        entry_id = _text(entry, "id")
        paper_id = _paper_id_from_url(entry_id)
        categories = tuple(
            category.attrib.get("term", "")
            for category in entry.findall(f"{ATOM}category")
            if category.attrib.get("term")
        )
        papers.append(
            PaperCandidate(
                paper_id=paper_id,
                title=_normalize_whitespace(title),
                year=_year_from_date(published),
                tags=_infer_tags(title=title, summary=summary, categories=categories),
                claim=_first_sentence(summary),
                risk=_risk_hint(summary),
                source_url=entry_id,
                venue="arXiv",
            )
        )
    return papers


def parse_semantic_scholar_search(payload: bytes | str) -> list[PaperCandidate]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    raw = json.loads(text)
    papers: list[PaperCandidate] = []
    for item in raw.get("data", []):
        title = item.get("title") or "Untitled Semantic Scholar result"
        abstract = item.get("abstract") or ""
        fields = tuple(item.get("fieldsOfStudy") or ())
        external_ids = item.get("externalIds") or {}
        paper_id = item.get("paperId") or external_ids.get("DOI") or "unknown"
        citation_count = item.get("citationCount")
        papers.append(
            PaperCandidate(
                paper_id=f"s2:{paper_id}",
                title=_normalize_whitespace(title),
                year=int(item.get("year") or 0),
                tags=_infer_tags(title=title, summary=abstract, categories=_field_tags(fields)),
                claim=_first_sentence(_normalize_whitespace(abstract)) or "No abstract returned by Semantic Scholar.",
                risk=_risk_hint(abstract),
                source_url=item.get("url") or "",
                doi=str(external_ids.get("DOI") or ""),
                venue=item.get("venue") or "",
                citation_count=int(citation_count) if citation_count is not None else None,
            )
        )
    return papers


def parse_pubmed_esearch(payload: bytes | str) -> list[str]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    raw = json.loads(text)
    return [str(uid) for uid in raw.get("esearchresult", {}).get("idlist", [])]


def parse_pubmed_xml(payload: bytes | str) -> list[PaperCandidate]:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    root = ET.fromstring(text)
    papers: list[PaperCandidate] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _first_element_text(article, ".//PMID") or "unknown"
        title = _first_element_text(article, ".//ArticleTitle") or "Untitled PubMed result"
        abstract = _pubmed_abstract(article)
        venue = _first_element_text(article, ".//Journal/Title")
        doi = _pubmed_article_id(article, "doi")
        mesh_terms = tuple(
            _normalize_whitespace(" ".join(descriptor.itertext()))
            for descriptor in article.findall(".//MeshHeading/DescriptorName")
        )
        papers.append(
            PaperCandidate(
                paper_id=f"pmid:{pmid}",
                title=_normalize_whitespace(title),
                year=_pubmed_year(article),
                tags=_infer_tags(title=title, summary=abstract, categories=_field_tags(mesh_terms)),
                claim=_first_sentence(abstract) or "No abstract returned by PubMed.",
                risk=_risk_hint(abstract),
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=doi,
                venue=venue,
                citation_count=None,
            )
        )
    return papers


def rerank_candidates(candidates: list[PaperCandidate], query: str) -> list[PaperCandidate]:
    query_tokens = _tokens(query)
    return sorted(
        candidates,
        key=lambda paper: _rerank_score(paper=paper, query_tokens=query_tokens),
        reverse=True,
    )


def _fetch_url(url: str, headers: dict[str, str] | None = None) -> bytes:
    request_headers = {"User-Agent": "bioml-research-agent/0.1"}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def _text(entry: ET.Element, name: str) -> str:
    value = entry.findtext(f"{ATOM}{name}")
    return value or ""


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _paper_id_from_url(url: str) -> str:
    match = re.search(r"/abs/([^/?#]+)", url)
    if match:
        return f"arxiv:{match.group(1)}"
    return f"arxiv:{url.rsplit('/', 1)[-1]}" if url else "arxiv:unknown"


def _year_from_date(value: str) -> int:
    match = re.match(r"(\d{4})", value)
    return int(match.group(1)) if match else 0


def _first_sentence(summary: str) -> str:
    for marker in (". ", "? ", "! "):
        if marker in summary:
            return summary.split(marker, 1)[0].strip() + marker.strip()
    return summary[:240].strip()


def _first_element_text(root: ET.Element, path: str) -> str:
    element = root.find(path)
    if element is None:
        return ""
    return _normalize_whitespace(" ".join(element.itertext()))


def _pubmed_abstract(article: ET.Element) -> str:
    parts = [
        _normalize_whitespace(" ".join(node.itertext()))
        for node in article.findall(".//Abstract/AbstractText")
    ]
    return " ".join(part for part in parts if part)


def _pubmed_article_id(article: ET.Element, id_type: str) -> str:
    for node in article.findall(".//ArticleIdList/ArticleId"):
        if node.attrib.get("IdType") == id_type:
            return _normalize_whitespace(" ".join(node.itertext()))
    return ""


def _pubmed_year(article: ET.Element) -> int:
    for path in (
        ".//ArticleDate/Year",
        ".//JournalIssue/PubDate/Year",
        ".//PubDate/Year",
    ):
        value = _first_element_text(article, path)
        if value:
            return int(value)
    medline_date = _first_element_text(article, ".//PubDate/MedlineDate")
    match = re.search(r"\d{4}", medline_date)
    return int(match.group(0)) if match else 0


def _risk_hint(summary: str) -> str:
    lowered = summary.lower()
    if "benchmark" in lowered or "dataset" in lowered:
        return "Check whether reported gains transfer across data splits and dataset boundaries."
    if "graph" in lowered:
        return "Graph construction quality may dominate the observed gain."
    if "retrieval" in lowered or "nearest" in lowered:
        return "Neighbor retrieval must be constrained to the training side of the split."
    return "Transfer risk is unknown until tested against the trusted baseline and controls."


def _field_tags(fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalize_whitespace(field).lower().replace(" ", "_") for field in fields)


def _rerank_score(paper: PaperCandidate, query_tokens: set[str]) -> float:
    text_tokens = _tokens(" ".join((paper.title, paper.claim, " ".join(paper.tags))))
    lexical = len(query_tokens & text_tokens)
    citation = math.log10((paper.citation_count or 0) + 1) * 0.25
    recency = max(0, paper.year - 2020) * 0.03
    domain = 0.2 if {"gene_expression", "enhancer_promoter"} & set(paper.tags) else 0.0
    return lexical + citation + recency + domain


def _infer_tags(title: str, summary: str, categories: tuple[str, ...]) -> tuple[str, ...]:
    text = f"{title} {summary}".lower()
    tags: list[str] = []
    keyword_tags = (
        ("retrieval", ("retrieval", "nearest neighbor", "nearest-neighbor")),
        ("graph", ("graph", "network", "edges")),
        ("gene_expression", ("gene expression", "transcription", "promoter")),
        ("enhancer_promoter", ("enhancer", "promoter", "regulatory")),
        ("moe", ("mixture-of-experts", "mixture of experts", "routing")),
        ("contrastive_learning", ("contrastive", "ranking")),
        ("foundation_model", ("foundation model", "pretrained", "pre-trained")),
    )
    for tag, keywords in keyword_tags:
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    tags.extend(
        category
        for category in categories
        if category.startswith(("cs.", "q-bio.", "stat."))
        or category in {"computer_science", "biology", "medicine", "mathematics"}
    )
    return tuple(dict.fromkeys(tags or ["literature_candidate"]))


def _papers_to_json(papers: list[PaperCandidate]) -> str:
    return json.dumps([asdict(paper) for paper in papers], indent=2, ensure_ascii=False)


def _cache_record_to_json(source: str, query: str, top_k: int, papers: list[PaperCandidate]) -> str:
    return json.dumps(
        {
            "source": source,
            "query": query,
            "top_k": top_k,
            "candidates": [asdict(paper) for paper in papers],
        },
        indent=2,
        ensure_ascii=False,
    )


def _papers_from_json(text: str) -> list[PaperCandidate]:
    raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("candidates", [])
    return _papers_from_raw(raw)


def _papers_from_raw(raw: object) -> list[PaperCandidate]:
    if not isinstance(raw, list):
        raise ValueError("paper cache payload must be a list or contain a candidates list")
    return [
        PaperCandidate(
            paper_id=str(item["paper_id"]),
            title=str(item["title"]),
            year=int(item["year"]),
            tags=tuple(item["tags"]),
            claim=str(item["claim"]),
            risk=str(item["risk"]),
            source_url=str(item.get("source_url", "")),
            doi=str(item.get("doi", "")),
            venue=str(item.get("venue", "")),
            citation_count=_optional_int(item.get("citation_count")),
        )
        for item in raw
    ]


def _cache_record_from_json(text: str, fallback_source: str = "", cache_file: str = "") -> dict[str, Any]:
    raw = json.loads(text)
    if isinstance(raw, list):
        return {
            "source": fallback_source,
            "query": "",
            "top_k": len(raw),
            "cache_file": cache_file,
            "candidates": _papers_from_raw(raw),
        }
    if not isinstance(raw, dict):
        raise ValueError("cache record must be an object or candidate list")
    return {
        "source": str(raw.get("source") or fallback_source),
        "query": str(raw.get("query") or ""),
        "top_k": int(raw.get("top_k") or len(raw.get("candidates", []))),
        "cache_file": cache_file,
        "candidates": _papers_from_raw(raw.get("candidates", [])),
    }


def _record_to_jsonable(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": record["source"],
        "query": record["query"],
        "top_k": record["top_k"],
        "cache_file": record.get("cache_file", ""),
        "candidates": [asdict(paper) for paper in record["candidates"]],
    }


def _load_snapshot_records(snapshot_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(snapshot_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(_cache_record_from_json(stripped, cache_file=str(snapshot_path)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid frozen literature snapshot at line {line_number}") from exc
    return records


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


STOP_WORDS = frozenset(
    {
        "and",
        "are",
        "aware",
        "for",
        "from",
        "into",
        "should",
        "the",
        "try",
        "with",
    }
)


def _tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in normalized.split() if len(token) > 2 and token not in STOP_WORDS}
