from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tracepilot.models import EvidenceChunk, SourceDocument

from .ingestion import sha256_text
from .safety import findings_to_metadata, scan_prompt_injection


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_CODE_SECTION_RE = re.compile(
    r"^\s*(?:(?:async\s+)?def|class|function)\s+([A-Za-z_][\w.]*)"
)
_CONFIG_SECTION_RE = re.compile(r"^\s*\[([A-Za-z0-9_.:-]+)\]\s*$")


@dataclass(frozen=True)
class _Block:
    text: str
    start_line: int
    end_line: int
    page: int
    section: str


class StructuredChunker:
    """Split documents on pages, headings, and paragraph boundaries."""

    def __init__(
        self,
        *,
        max_chars: int = 1200,
        min_chars: int = 80,
        overlap_lines: int = 1,
    ) -> None:
        if max_chars < 16:
            raise ValueError("max_chars must be at least 16")
        if min_chars < 0 or min_chars > max_chars:
            raise ValueError("min_chars must be between 0 and max_chars")
        if overlap_lines < 0:
            raise ValueError("overlap_lines must not be negative")
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_lines = overlap_lines

    def chunk(self, document: SourceDocument) -> list[EvidenceChunk]:
        return self.chunk_document(document)

    def chunk_document(self, document: SourceDocument) -> list[EvidenceChunk]:
        blocks = self._blocks(document)
        grouped = self._pack_blocks(blocks)
        chunks: list[EvidenceChunk] = []
        document_hash = document.content_hash or sha256_text(document.content)
        for ordinal, block in enumerate(grouped):
            text = block.text.strip()
            if not text:
                continue
            content_hash = sha256_text(text)
            identity = (
                f"{document.document_id}\0{document_hash}\0{block.page}\0"
                f"{block.start_line}\0{block.end_line}\0{ordinal}\0{content_hash}"
            )
            chunk_id = f"chunk-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:28]}"
            findings = scan_prompt_injection(text)
            metadata: dict[str, Any] = {
                "ordinal": ordinal,
                "section": block.section,
                "document_version": document.version,
                "prompt_injection_detected": bool(findings),
                "prompt_injection_findings": findings_to_metadata(findings),
            }
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=text,
                    content_hash=content_hash,
                    source_uri=document.uri,
                    title=document.title,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    page=block.page,
                    score=0.0,
                    metadata=metadata,
                )
            )
        return chunks

    def chunk_documents(
        self,
        documents: Iterable[SourceDocument],
    ) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def _blocks(self, document: SourceDocument) -> list[_Block]:
        pages = document.content.split("\f")
        raw_page_numbers = document.metadata.get("page_numbers", [])
        page_numbers = (
            [int(number) for number in raw_page_numbers]
            if isinstance(raw_page_numbers, (list, tuple))
            else []
        )
        blocks: list[_Block] = []
        global_line = 1
        current_section = ""
        for page_index, page_text in enumerate(pages, start=1):
            page_number = (
                page_numbers[page_index - 1]
                if page_index <= len(page_numbers)
                else page_index
            )
            lines = page_text.split("\n")
            paragraph_lines: list[tuple[int, str]] = []

            def flush_paragraph() -> None:
                if not paragraph_lines:
                    return
                text = "\n".join(line for _, line in paragraph_lines).strip()
                if text:
                    blocks.extend(
                        self._split_large_block(
                            _Block(
                                text=text,
                                start_line=paragraph_lines[0][0],
                                end_line=paragraph_lines[-1][0],
                                page=page_number,
                                section=current_section,
                            )
                        )
                    )
                paragraph_lines.clear()

            for offset, line in enumerate(lines):
                line_number = global_line + offset
                section = _section_title(line)
                if section is not None:
                    flush_paragraph()
                    current_section = section
                    blocks.extend(
                        self._split_large_block(
                            _Block(
                                text=line.strip(),
                                start_line=line_number,
                                end_line=line_number,
                                page=page_number,
                                section=current_section,
                            )
                        )
                    )
                elif not line.strip():
                    flush_paragraph()
                else:
                    paragraph_lines.append((line_number, line))
            flush_paragraph()
            global_line += max(1, len(lines))
        return blocks

    def _split_large_block(self, block: _Block) -> list[_Block]:
        if len(block.text) <= self.max_chars:
            return [block]
        lines = block.text.split("\n")
        if len(lines) == 1:
            return [
                _Block(
                    text=block.text[start : start + self.max_chars],
                    start_line=block.start_line,
                    end_line=block.end_line,
                    page=block.page,
                    section=block.section,
                )
                for start in range(0, len(block.text), self.max_chars)
            ]

        pieces: list[_Block] = []
        current: list[str] = []
        current_start = block.start_line
        for index, line in enumerate(lines):
            line_number = block.start_line + index
            if current and len("\n".join((*current, line))) > self.max_chars:
                pieces.append(
                    _Block(
                        text="\n".join(current),
                        start_line=current_start,
                        end_line=line_number - 1,
                        page=block.page,
                        section=block.section,
                    )
                )
                current = []
                current_start = line_number
            if len(line) > self.max_chars:
                if current:
                    pieces.append(
                        _Block(
                            text="\n".join(current),
                            start_line=current_start,
                            end_line=line_number - 1,
                            page=block.page,
                            section=block.section,
                        )
                    )
                    current = []
                pieces.extend(
                    _Block(
                        text=line[start : start + self.max_chars],
                        start_line=line_number,
                        end_line=line_number,
                        page=block.page,
                        section=block.section,
                    )
                    for start in range(0, len(line), self.max_chars)
                )
                current_start = line_number + 1
            else:
                current.append(line)
        if current:
            pieces.append(
                _Block(
                    text="\n".join(current),
                    start_line=current_start,
                    end_line=block.end_line,
                    page=block.page,
                    section=block.section,
                )
            )
        return pieces

    def _pack_blocks(self, blocks: list[_Block]) -> list[_Block]:
        packed: list[_Block] = []
        current: list[_Block] = []

        def flush() -> None:
            if not current:
                return
            text = "\n\n".join(block.text for block in current)
            packed.append(
                _Block(
                    text=text,
                    start_line=current[0].start_line,
                    end_line=current[-1].end_line,
                    page=current[0].page,
                    section=current[-1].section or current[0].section,
                )
            )
            current.clear()

        for block in blocks:
            boundary_changed = bool(
                current
                and (
                    block.page != current[-1].page
                    or (
                        block.section
                        and current[-1].section
                        and block.section != current[-1].section
                    )
                )
            )
            candidate_length = len(block.text)
            if current:
                candidate_length += len("\n\n".join(item.text for item in current)) + 2
            if current and (boundary_changed or candidate_length > self.max_chars):
                previous = current[-1]
                flush()
                if (
                    self.overlap_lines
                    and previous.page == block.page
                    and previous.section == block.section
                ):
                    overlap = previous.text.splitlines()[-self.overlap_lines :]
                    if overlap and len("\n".join(overlap)) + len(block.text) + 2 <= self.max_chars:
                        overlap_start = max(
                            previous.start_line,
                            previous.end_line - len(overlap) + 1,
                        )
                        current.append(
                            _Block(
                                text="\n".join(overlap),
                                start_line=overlap_start,
                                end_line=previous.end_line,
                                page=previous.page,
                                section=previous.section,
                            )
                        )
            current.append(block)
        flush()

        if len(packed) > 1 and len(packed[-1].text) < self.min_chars:
            previous = packed[-2]
            tail = packed[-1]
            if (
                previous.page == tail.page
                and previous.section == tail.section
                and len(previous.text) + len(tail.text) + 2 <= self.max_chars
            ):
                packed[-2:] = [
                    _Block(
                        text=f"{previous.text}\n\n{tail.text}",
                        start_line=previous.start_line,
                        end_line=tail.end_line,
                        page=previous.page,
                        section=tail.section,
                    )
                ]
        return packed


def _section_title(line: str) -> str | None:
    heading = _HEADING_RE.match(line)
    if heading:
        return heading.group(1).strip()
    code_section = _CODE_SECTION_RE.match(line)
    if code_section:
        return code_section.group(1)
    config_section = _CONFIG_SECTION_RE.match(line)
    if config_section:
        return config_section.group(1)
    return None
