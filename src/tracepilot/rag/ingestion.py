from __future__ import annotations

import hashlib
import ipaddress
import io
import json
import re
import socket
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree

from tracepilot.models import SourceDocument

from .safety import findings_to_metadata, scan_prompt_injection


DEFAULT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".css",
        ".csv",
        ".docx",
        ".go",
        ".h",
        ".hpp",
        ".htm",
        ".html",
        ".ini",
        ".ipynb",
        ".java",
        ".js",
        ".json",
        ".jsonl",
        ".md",
        ".pdf",
        ".php",
        ".properties",
        ".py",
        ".rb",
        ".rst",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
DEFAULT_FILENAMES = frozenset(
    {
        ".dockerignore",
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "containerfile",
        "dockerfile",
        "license",
        "makefile",
        "procfile",
        "readme",
    }
)


class SourceProvider(Protocol):
    def load(self) -> list[SourceDocument]:
        ...


@dataclass(frozen=True)
class FetchResponse:
    body: bytes | str
    final_url: str
    content_type: str = ""
    status: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


URLFetcher = Callable[[str], FetchResponse | bytes | str | tuple[Any, ...]]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id_for_uri(uri: str) -> str:
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:24]
    return f"doc-{digest}"


class LocalSourceProvider:
    def __init__(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        suffixes: Iterable[str] | None = None,
        filenames: Iterable[str] = DEFAULT_FILENAMES,
        excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
        max_file_bytes: int = 5_000_000,
    ) -> None:
        self.path = Path(path).expanduser()
        self.recursive = recursive
        self.suffixes = frozenset(
            suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}"
            for suffix in (suffixes or DEFAULT_SUFFIXES)
        )
        self.excluded_dirs = frozenset(excluded_dirs)
        self.filenames = frozenset(name.casefold() for name in filenames)
        self.max_file_bytes = max_file_bytes

    def load(self) -> list[SourceDocument]:
        root = self.path.resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        paths = [root] if root.is_file() else list(self._iter_files(root))
        documents: list[SourceDocument] = []
        for path in paths:
            document = self._read_document(path)
            if document is not None:
                documents.append(document)
        return _deduplicate_documents(documents)

    def _iter_files(self, root: Path) -> Iterable[Path]:
        iterator = root.rglob("*") if self.recursive else root.glob("*")
        for path in sorted(iterator):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                path.resolve().relative_to(root)
                relative_parts = path.relative_to(root).parts
            except ValueError:
                continue
            if any(part in self.excluded_dirs for part in relative_parts[:-1]):
                continue
            if (
                path.suffix.casefold() not in self.suffixes
                and path.name.casefold() not in self.filenames
            ):
                continue
            yield path

    def _read_document(self, path: Path) -> SourceDocument | None:
        stat = path.stat()
        if stat.st_size > self.max_file_bytes:
            return None
        suffix = path.suffix.casefold()
        try:
            if suffix == ".pdf":
                content, extra_metadata = _read_pdf(path)
            elif suffix == ".docx":
                content, extra_metadata = _read_docx(path)
            elif suffix == ".ipynb":
                content, extra_metadata = _read_notebook(path)
            else:
                raw = path.read_bytes()
                if _looks_binary(raw):
                    return None
                content = _decode_bytes(raw)
                extra_metadata = {}
        except (OSError, UnicodeError, ValueError, zipfile.BadZipFile):
            return None

        content = _normalize_content(content, preserve_form_feed=suffix == ".pdf")
        if not content.strip():
            return None
        resolved = path.resolve()
        uri = resolved.as_uri()
        content_hash = sha256_text(content)
        findings = scan_prompt_injection(content)
        metadata: dict[str, Any] = {
            "path": str(resolved),
            "suffix": suffix,
            "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
            "prompt_injection_detected": bool(findings),
            "prompt_injection_findings": findings_to_metadata(findings),
            **extra_metadata,
        }
        return SourceDocument(
            document_id=document_id_for_uri(uri),
            source_type="local",
            uri=uri,
            title=_infer_title(content, fallback=path.name),
            content=content,
            content_hash=content_hash,
            version=f"{stat.st_mtime_ns}:{stat.st_size}",
            metadata=metadata,
        )


class UrlSourceProvider:
    def __init__(
        self,
        urls: str | Iterable[str],
        *,
        fetcher: URLFetcher | None = None,
        timeout: float = 15.0,
        max_response_bytes: int = 5_000_000,
    ) -> None:
        self.urls = (urls,) if isinstance(urls, str) else tuple(urls)
        self.fetcher = fetcher
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    def load(self) -> list[SourceDocument]:
        documents = [self._load_url(url) for url in self.urls]
        return _deduplicate_documents(documents)

    def _load_url(self, url: str) -> SourceDocument:
        _validate_remote_url(url, resolve=False)
        response = (
            _coerce_fetch_response(url, self.fetcher(url))
            if self.fetcher is not None
            else self._fetch(url)
        )
        if not 200 <= response.status < 400:
            raise ValueError(f"URL returned HTTP {response.status}: {url}")
        raw = (
            response.body
            if isinstance(response.body, bytes)
            else response.body.encode("utf-8")
        )
        if len(raw) > self.max_response_bytes:
            raise ValueError(f"URL response exceeds byte limit: {url}")

        content_type = response.content_type.casefold()
        decoded = (
            response.body
            if isinstance(response.body, str)
            else _decode_bytes(response.body, content_type=content_type)
        )
        is_pdf = (
            "application/pdf" in content_type
            or urllib.parse.urlparse(response.final_url or url).path.casefold().endswith(".pdf")
        )
        extra_metadata: dict[str, Any] = {}
        if is_pdf:
            content, extra_metadata = _read_pdf_bytes(raw)
            title = ""
        elif "html" in content_type or _looks_like_html(decoded):
            extractor = _HTMLTextExtractor()
            extractor.feed(decoded)
            content = extractor.text()
            title = extractor.title.strip()
        else:
            content = decoded
            title = ""
        content = _normalize_content(content, preserve_form_feed=is_pdf)
        if not content.strip():
            raise ValueError(f"URL did not contain indexable text: {url}")

        final_url = response.final_url or url
        _validate_remote_url(final_url, resolve=False)
        content_hash = sha256_text(content)
        headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
        version = headers.get("etag") or headers.get("last-modified") or content_hash
        findings = scan_prompt_injection(content)
        return SourceDocument(
            document_id=document_id_for_uri(final_url),
            source_type="url",
            uri=final_url,
            title=title or _url_title(final_url),
            content=content,
            content_hash=content_hash,
            version=version,
            metadata={
                "requested_url": url,
                "http_status": response.status,
                "content_type": response.content_type,
                "etag": headers.get("etag", ""),
                "last_modified": headers.get("last-modified", ""),
                "prompt_injection_detected": bool(findings),
                "prompt_injection_findings": findings_to_metadata(findings),
                **extra_metadata,
            },
        )

    def _fetch(self, url: str) -> FetchResponse:
        _validate_remote_url(url, resolve=True)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html,text/plain,application/json,application/pdf;q=0.8,*/*;q=0.2",
                "User-Agent": "TracePilot/0.1",
            },
        )
        opener = urllib.request.build_opener(_SafeRedirectHandler())
        with opener.open(request, timeout=self.timeout) as response:
            body = response.read(self.max_response_bytes + 1)
            if len(body) > self.max_response_bytes:
                raise ValueError(f"URL response exceeds byte limit: {url}")
            return FetchResponse(
                body=body,
                final_url=response.geturl(),
                content_type=response.headers.get_content_type(),
                status=int(response.status),
                headers=dict(response.headers.items()),
            )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_remote_url(newurl, resolve=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_remote_url(url: str, *, resolve: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme or '<missing>'}")
    hostname = (parsed.hostname or "").strip().casefold()
    if not hostname:
        raise ValueError("URL hostname is required")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("local and private network URLs are not allowed")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None and not _is_public_ip(literal):
        raise ValueError("local and private network URLs are not allowed")
    if not resolve or literal is not None:
        return
    try:
        addresses = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme.casefold() == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"URL hostname could not be resolved: {hostname}") from exc
    for address in addresses:
        resolved = ipaddress.ip_address(address[4][0])
        if not _is_public_ip(resolved):
            raise ValueError("URL resolves to a local or private network address")


def _is_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


URLSourceProvider = UrlSourceProvider


class FrozenSourceProvider:
    def __init__(
        self,
        snapshot: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        verify_hashes: bool = True,
    ) -> None:
        self.snapshot = snapshot
        self.verify_hashes = verify_hashes

    def load(self) -> list[SourceDocument]:
        records, snapshot_ref = self._records()
        if not all(isinstance(record, Mapping) for record in records):
            raise ValueError("each frozen record must be an object")
        documents: list[SourceDocument] = []
        for ordinal, record in enumerate(records):
            document = self._document_from_record(record, ordinal, snapshot_ref)
            documents.append(document)
        return _deduplicate_documents(documents)

    def _records(self) -> tuple[list[Mapping[str, Any]], str]:
        if not isinstance(self.snapshot, (str, Path)):
            if isinstance(self.snapshot, Mapping):
                candidate = self.snapshot.get(
                    "documents",
                    self.snapshot.get("records"),
                )
                if isinstance(candidate, list):
                    return candidate, "in-memory"
                return [self.snapshot], "in-memory"
            return list(self.snapshot), "in-memory"
        path = Path(self.snapshot).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.casefold() == ".jsonl":
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(text)
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                candidate = payload.get("documents", payload.get("records", [payload]))
                records = candidate if isinstance(candidate, list) else [candidate]
            else:
                raise ValueError("frozen snapshot must contain objects")
        if not all(isinstance(record, Mapping) for record in records):
            raise ValueError("each frozen record must be an object")
        return records, str(path)

    def _document_from_record(
        self,
        record: Mapping[str, Any],
        ordinal: int,
        snapshot_ref: str,
    ) -> SourceDocument:
        content = record.get("content", record.get("text", ""))
        pages = record.get("pages")
        page_numbers: list[int] = []
        if not content and isinstance(pages, list):
            page_texts: list[str] = []
            for index, page in enumerate(pages, start=1):
                if isinstance(page, Mapping):
                    page_texts.append(str(page.get("text", page.get("content", ""))))
                    page_numbers.append(int(page.get("page", index)))
                else:
                    page_texts.append(str(page))
                    page_numbers.append(index)
            content = "\f".join(page_texts)
        content = _normalize_content(str(content), preserve_form_feed=True)
        if not content.strip():
            raise ValueError(f"frozen record {ordinal} has no content")

        uri = str(
            record.get(
                "uri",
                record.get("url", f"frozen://{sha256_text(snapshot_ref)[:12]}/{ordinal}"),
            )
        )
        actual_hash = sha256_text(content)
        expected_hash = str(record.get("content_hash", ""))
        if (
            self.verify_hashes
            and expected_hash
            and re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash)
            and expected_hash.casefold() != actual_hash
        ):
            raise ValueError(f"frozen content_hash mismatch for {uri}")
        findings = scan_prompt_injection(content)
        raw_metadata = record.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata.update(
            {
                "snapshot": snapshot_ref,
                "snapshot_ordinal": ordinal,
                "prompt_injection_detected": bool(findings),
                "prompt_injection_findings": findings_to_metadata(findings),
            }
        )
        if page_numbers:
            metadata["page_numbers"] = page_numbers
        return SourceDocument(
            document_id=str(record.get("document_id") or document_id_for_uri(uri)),
            source_type="frozen",
            uri=uri,
            title=str(record.get("title") or _infer_title(content, fallback=uri)),
            content=content,
            content_hash=actual_hash,
            version=str(record.get("version") or actual_hash),
            metadata=metadata,
        )


class Ingestor:
    def __init__(
        self,
        *,
        url_fetcher: URLFetcher | None = None,
        max_file_bytes: int = 5_000_000,
        max_response_bytes: int = 5_000_000,
    ) -> None:
        self.url_fetcher = url_fetcher
        self.max_file_bytes = max_file_bytes
        self.max_response_bytes = max_response_bytes

    def ingest(self, provider: SourceProvider) -> list[SourceDocument]:
        return _deduplicate_documents(provider.load())

    def ingest_path(
        self,
        path: str | Path,
        *,
        recursive: bool = True,
        suffixes: Iterable[str] | None = None,
    ) -> list[SourceDocument]:
        return self.ingest(
            LocalSourceProvider(
                path,
                recursive=recursive,
                suffixes=suffixes,
                max_file_bytes=self.max_file_bytes,
            )
        )

    def ingest_url(self, urls: str | Iterable[str]) -> list[SourceDocument]:
        return self.ingest(
            UrlSourceProvider(
                urls,
                fetcher=self.url_fetcher,
                max_response_bytes=self.max_response_bytes,
            )
        )

    def ingest_frozen(
        self,
        snapshot: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        verify_hashes: bool = True,
    ) -> list[SourceDocument]:
        return self.ingest(
            FrozenSourceProvider(snapshot, verify_hashes=verify_hashes)
        )


class _HTMLTextExtractor(HTMLParser):
    _BLOCK_TAGS = frozenset(
        {
            "article",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "figcaption",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "li",
            "main",
            "nav",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "td",
            "th",
            "tr",
            "ul",
        }
    )
    _IGNORED_TAGS = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    @property
    def title(self) -> str:
        return " ".join(self.title_parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if not self._ignored_depth and re.fullmatch(r"h[1-6]", tag):
            self.parts.append(f"\n{'#' * int(tag[1])} ")
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self._in_title = False
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        if not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data.strip())
            return
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def _coerce_fetch_response(
    requested_url: str,
    value: FetchResponse | bytes | str | tuple[Any, ...],
) -> FetchResponse:
    if isinstance(value, FetchResponse):
        return value
    if isinstance(value, (bytes, str)):
        return FetchResponse(body=value, final_url=requested_url)
    if not isinstance(value, tuple):
        raise TypeError("URL fetcher must return FetchResponse, bytes, str, or tuple")
    if len(value) == 2:
        body, metadata = value
        if isinstance(metadata, Mapping):
            return FetchResponse(
                body=body,
                final_url=str(metadata.get("final_url", requested_url)),
                content_type=str(metadata.get("content_type", "")),
                status=int(metadata.get("status", 200)),
                headers={
                    str(key): str(item)
                    for key, item in metadata.get("headers", {}).items()
                },
            )
        return FetchResponse(
            body=body,
            final_url=requested_url,
            content_type=str(metadata),
        )
    if len(value) == 3:
        status, headers, body = value
        normalized_headers = (
            {str(key): str(item) for key, item in headers.items()}
            if isinstance(headers, Mapping)
            else {}
        )
        return FetchResponse(
            body=body,
            final_url=requested_url,
            content_type=normalized_headers.get(
                "content-type",
                normalized_headers.get("Content-Type", ""),
            ),
            status=int(status),
            headers=normalized_headers,
        )
    raise TypeError("unsupported URL fetcher tuple")


def _deduplicate_documents(documents: Iterable[SourceDocument]) -> list[SourceDocument]:
    unique: list[SourceDocument] = []
    seen_hashes: set[str] = set()
    for document in documents:
        content_hash = document.content_hash or sha256_text(document.content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        unique.append(document)
    return unique


def _decode_bytes(raw: bytes, *, content_type: str = "") -> str:
    charset_match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
    encodings = [charset_match.group(1)] if charset_match else []
    encodings.extend(["utf-8-sig", "utf-8"])
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _looks_binary(raw: bytes) -> bool:
    if not raw:
        return False
    if b"\x00" in raw[:4096]:
        return True
    control_count = sum(
        byte < 9 or (13 < byte < 32)
        for byte in raw[:4096]
    )
    return control_count / min(len(raw), 4096) > 0.1


def _normalize_content(text: str, *, preserve_form_feed: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pages = text.split("\f") if preserve_form_feed else [text.replace("\f", "\n")]
    normalized_pages: list[str] = []
    for page in pages:
        lines = [re.sub(r"[ \t]+$", "", line) for line in page.split("\n")]
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
        normalized_pages.append(normalized.strip())
    separator = "\f" if preserve_form_feed else "\n"
    return separator.join(normalized_pages)


def _infer_title(content: str, *, fallback: str) -> str:
    for line in content.replace("\f", "\n").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading:
            return heading.group(1).strip()[:200]
        if len(stripped) <= 120:
            return stripped[:200]
        break
    return fallback


def _url_title(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    return name or parsed.netloc or url


def _looks_like_html(text: str) -> bool:
    prefix = text.lstrip()[:500].casefold()
    return prefix.startswith("<!doctype html") or "<html" in prefix or "<body" in prefix


def _read_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    return _read_pdf_bytes(path.read_bytes())


def _read_pdf_bytes(raw: bytes) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("PDF ingestion requires optional pypdf") from exc
    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\f".join(pages), {"page_count": len(pages)}


def _read_docx(path: Path) -> tuple[str, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("word/document.xml")
    root = ElementTree.fromstring(payload)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs), {"format": "docx"}


def _read_notebook(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells: list[str] = []
    for index, cell in enumerate(payload.get("cells", []), start=1):
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        if text.strip():
            cells.append(f"## Cell {index} ({cell.get('cell_type', 'unknown')})\n{text}")
    return "\n\n".join(cells), {"cell_count": len(cells), "format": "ipynb"}
