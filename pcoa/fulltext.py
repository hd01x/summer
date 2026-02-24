"""
PCoA — Fulltext acquisition (PMC BioC XML + PDF upload).

Provides two entry points that return a FulltextResult whose `.text`
field can be fed directly into the existing pipeline.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple
from xml.etree import ElementTree as ET

import requests

# ---------------------------------------------------------------------------
#  Data model
# ---------------------------------------------------------------------------

@dataclass
class FulltextResult:
    text: str
    pmc_id: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    doi: str = ""
    sections_found: List[str] = field(default_factory=list)
    is_truncated: bool = False
    source: str = ""          # "pmc" | "pdf"
    word_count: int = 0


# ---------------------------------------------------------------------------
#  Section filtering (shared by both sources)
# ---------------------------------------------------------------------------

_KEEP_PREFIXES = (
    "abstract", "introduction", "background",
    "method", "material",                          # catches "methods", "materials and methods", etc.
    "result", "finding",                           # catches "results", "results and discussion", "findings"
    "discussion", "conclusion",
)

_DROP_SECTIONS = {
    "references", "bibliography", "acknowledgments", "acknowledgements",
    "supplementary", "supplementary material", "supplementary materials",
    "author contributions", "authors' contributions",
    "funding", "financial support",
    "figures", "tables", "figure legends",
    "ethics", "ethics statement", "ethical approval",
    "data availability", "data sharing", "competing interests",
    "conflict of interest", "conflicts of interest",
    "abbreviations", "appendix", "appendices",
}

_MAX_WORDS = 8_000

# Patterns for garbage lines that slip through section filtering
_GARBAGE_LINE_RE = re.compile(
    r"\{abstract\}|\{sentences\}|\{phrases\}|\{summary\}"
    r"|#\s*Instructions:|#\s*Abstract:|#\s*Summary:"
    r"|^Table\s*\d+\s*[:.]\s",
    re.IGNORECASE,
)


def _is_kept_section(heading: str) -> bool:
    """Return True if the section heading matches the keep-list (prefix match)."""
    norm = heading.strip().lower()
    return any(norm.startswith(prefix) for prefix in _KEEP_PREFIXES)


def _clean_text(text: str) -> str:
    """Remove garbage lines (template placeholders, OCR junk, table captions)."""
    lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        # Drop template / placeholder lines
        if _GARBAGE_LINE_RE.search(stripped):
            continue
        # Drop lines that are long concatenated words without spaces (OCR garbage)
        if len(stripped) > 50 and " " not in stripped:
            continue
        lines.append(line)
    return "\n".join(lines)


def _filter_sections(sections: List[Tuple[str, str]]) -> str:
    """Keep only allowlisted sections, drop everything else, cap at _MAX_WORDS."""
    kept: List[str] = []
    for heading, body in sections:
        norm = heading.strip().lower()
        # Explicitly dropped sections
        if norm in _DROP_SECTIONS:
            continue
        # Only keep sections on the allowlist; unknown sections are dropped
        if not _is_kept_section(norm):
            continue
        kept.append(body)

    text = _clean_text("\n\n".join(kept))
    words = text.split()
    if len(words) > _MAX_WORDS:
        text = " ".join(words[:_MAX_WORDS])
    return text


# ---------------------------------------------------------------------------
#  PMC BioC XML fetch
# ---------------------------------------------------------------------------

_PMC_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
    "pmcoa.cgi/BioC_xml/{pmc_id}/unicode"
)


def _normalize_pmc_id(raw: str) -> str:
    raw = raw.strip()
    if raw.upper().startswith("PMC"):
        return "PMC" + raw[3:]
    if raw.isdigit():
        return "PMC" + raw
    return raw


def _parse_bioc_xml(xml_bytes: bytes) -> FulltextResult:
    """Parse BioC XML and return a FulltextResult with filtered sections."""
    root = ET.fromstring(xml_bytes)

    # Metadata from the first <document>
    title = ""
    authors: List[str] = []
    journal = ""
    pub_date = ""
    doi = ""
    pmc_id = ""

    doc = root.find(".//document")
    if doc is not None:
        id_el = doc.find("id")
        if id_el is not None and id_el.text:
            pmc_id = id_el.text.strip()
        for infon in doc.findall("infon"):
            key = infon.get("key", "")
            val = (infon.text or "").strip()
            if key == "title":
                title = val
            elif key == "journal":
                journal = val
            elif key == "year":
                pub_date = val
            elif key == "doi":
                doi = val
            elif key == "authors":
                authors = [a.strip() for a in val.split(";") if a.strip()]

    # Iterate passages — build (section_type, text) pairs
    sections: List[Tuple[str, str]] = []
    section_names: List[str] = []
    current_section = "unknown"

    for passage in root.iter("passage"):
        sec_type = ""
        for infon in passage.findall("infon"):
            key = infon.get("key", "")
            if key == "section_type":
                sec_type = (infon.text or "").strip().lower()
            elif key == "type" and not sec_type:
                sec_type = (infon.text or "").strip().lower()

        if sec_type:
            current_section = sec_type

        text_el = passage.find("text")
        if text_el is not None and text_el.text and text_el.text.strip():
            sections.append((current_section, text_el.text.strip()))
            if current_section not in section_names:
                section_names.append(current_section)

        # Also pick up title from the first "front" / "title" passage
        if not title and sec_type in ("front", "title", "article-title"):
            if text_el is not None and text_el.text:
                title = text_el.text.strip()

    filtered_text = _filter_sections(sections)
    word_count = len(filtered_text.split())

    return FulltextResult(
        text=filtered_text,
        pmc_id=pmc_id,
        title=title,
        authors=authors,
        journal=journal,
        pub_date=pub_date,
        doi=doi,
        sections_found=section_names,
        is_truncated=word_count >= _MAX_WORDS,
        source="pmc",
        word_count=word_count,
    )


def fetch_pmc_fulltext(pmc_id: str) -> FulltextResult:
    """Fetch fulltext from PubMed Central via BioC XML API.

    Raises ``ValueError`` for invalid / not-found PMC IDs.
    """
    pmc_id = _normalize_pmc_id(pmc_id)
    url = _PMC_URL.format(pmc_id=pmc_id)

    resp = requests.get(url, timeout=30)

    if resp.status_code == 404 or b"<error>" in resp.content[:500]:
        raise ValueError(
            f"{pmc_id} not found — ensure it is a valid Open Access PMC ID."
        )
    resp.raise_for_status()

    result = _parse_bioc_xml(resp.content)
    if not result.pmc_id:
        result.pmc_id = pmc_id
    return result


# ---------------------------------------------------------------------------
#  PDF text extraction
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^(abstract|introduction|background|methods|materials|results|"
    r"findings|discussion|conclusion|conclusions|references|"
    r"acknowledgments|acknowledgements|supplementary|"
    r"author contributions|funding|ethics|appendix)\b",
    re.IGNORECASE,
)

_MAX_UNSTRUCTURED_WORDS = 4_000


def _detect_pdf_sections(pages_text: List[str]) -> List[Tuple[str, str]]:
    """Heuristic section detection in extracted PDF text.

    Returns (heading, body) pairs. If fewer than 3 headings are found
    the text is treated as unstructured.
    """
    full_text = "\n\n".join(pages_text)
    lines = full_text.split("\n")

    sections: List[Tuple[str, str]] = []
    current_heading = "unknown"
    current_lines: List[str] = []
    heading_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        # Heuristic: ALL-CAPS short line or known section name
        is_heading = False
        if len(stripped) < 80 and stripped.isupper() and len(stripped.split()) <= 8:
            is_heading = True
        elif _SECTION_RE.match(stripped):
            is_heading = True

        if is_heading:
            # Flush previous section
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = stripped.lower()
            current_lines = []
            heading_count += 1
        else:
            current_lines.append(stripped)

    # Flush last section
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    # If too few headings, treat as unstructured
    if heading_count < 3:
        return [("unknown", full_text)]

    return sections


def extract_pdf_text(file_bytes: bytes) -> FulltextResult:
    """Extract text from a PDF using pdfplumber.

    Raises ``ValueError`` when no text can be extracted.
    """
    import pdfplumber  # imported here so the dep is only needed for PDF uploads
    import io

    pages_text: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    if not pages_text:
        raise ValueError("Could not extract text — the PDF may be image-based or empty.")

    sections = _detect_pdf_sections(pages_text)

    # Structured → filter sections; unstructured → truncate
    is_truncated = False
    if len(sections) == 1 and sections[0][0] == "unknown":
        words = sections[0][1].split()
        if len(words) > _MAX_UNSTRUCTURED_WORDS:
            filtered = " ".join(words[:_MAX_UNSTRUCTURED_WORDS])
            is_truncated = True
        else:
            filtered = sections[0][1]
        section_names = []
    else:
        filtered = _filter_sections(sections)
        section_names = [h for h, _ in sections]
        if len(filtered.split()) >= _MAX_WORDS:
            is_truncated = True

    # Guess title: first short non-empty line
    title = "Uploaded PDF"
    for page in pages_text[:1]:
        for line in page.split("\n"):
            line = line.strip()
            if line and len(line) < 200:
                title = line
                break
        break

    word_count = len(filtered.split())

    return FulltextResult(
        text=filtered,
        title=title,
        sections_found=section_names,
        is_truncated=is_truncated,
        source="pdf",
        word_count=word_count,
    )
