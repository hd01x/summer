"""
PubMed API integration using NCBI E-utilities.
Searches for and fetches RCT article abstracts from PubMed.
Based on the selection criteria described in §3.1 of the PCoA paper.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from pcoa.config import NCBI_EMAIL, NCBI_API_KEY
from pcoa.text_processing import split_into_sentences

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class PubMedArticle:
    """Represents a PubMed article with its metadata and abstract."""
    pmid: str
    title: str
    abstract: List[str]
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    doi: str = ""


def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: PubMed search query string
        max_results: Maximum number of results to return

    Returns:
        List of PMID strings
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    id_list = data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        raise ValueError(f"No PubMed results found for query: {query}")
    return id_list


def fetch_articles(pmids: List[str]) -> List[PubMedArticle]:
    """
    Fetch full article details from PubMed for given PMIDs.

    Args:
        pmids: List of PubMed IDs

    Returns:
        List of PubMedArticle objects
    """
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    response = requests.get(EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()

    return _parse_pubmed_xml(response.text)


def fetch_single_article(pmid: str) -> PubMedArticle:
    """Fetch a single article by PMID."""
    articles = fetch_articles([pmid])
    if not articles:
        raise ValueError(f"No article found for PMID: {pmid}")
    return articles[0]


def _parse_pubmed_xml(xml_text: str) -> List[PubMedArticle]:
    """Parse PubMed XML response into PubMedArticle objects."""
    root = ET.fromstring(xml_text)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        title_elem = article_elem.find(".//ArticleTitle")
        title = _get_text_content(title_elem) if title_elem is not None else ""

        # Extract abstract - handle structured abstracts with labels
        abstract_elem = article_elem.find(".//Abstract")
        abstract_text = ""
        if abstract_elem is not None:
            abstract_parts = []
            for abs_text in abstract_elem.findall("AbstractText"):
                label = abs_text.get("Label", "")
                text = _get_text_content(abs_text)
                if text:
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            abstract_text = " ".join(abstract_parts)

        if not abstract_text:
            continue

        abstract_sentences = split_into_sentences(abstract_text)

        # Authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and fore_name is not None:
                authors.append(f"{fore_name.text} {last_name.text}")
            elif last_name is not None:
                authors.append(last_name.text)

        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Publication date
        pub_date = _extract_pub_date(article_elem)

        # DOI
        doi = ""
        for article_id in article_elem.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text or ""
                break

        articles.append(PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract_sentences,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            doi=doi,
        ))

    return articles


def _get_text_content(element) -> str:
    """Extract all text content from an XML element, including nested tags."""
    return "".join(element.itertext()).strip()


def _extract_pub_date(article_elem) -> str:
    """Extract publication date from article element."""
    # Try PubDate first
    pub_date_elem = article_elem.find(".//PubDate")
    if pub_date_elem is not None:
        year = pub_date_elem.find("Year")
        month = pub_date_elem.find("Month")
        day = pub_date_elem.find("Day")
        parts = []
        if year is not None:
            parts.append(year.text)
        if month is not None:
            parts.append(month.text)
        if day is not None:
            parts.append(day.text)
        if parts:
            return " ".join(parts)

    # Try MedlineDate
    medline_date = article_elem.find(".//MedlineDate")
    if medline_date is not None and medline_date.text:
        return medline_date.text

    return ""


def search_and_fetch(query: str, max_results: int = 20) -> List[PubMedArticle]:
    """
    Combined search and fetch: search PubMed and return full article details.

    Args:
        query: PubMed search query
        max_results: Maximum number of results

    Returns:
        List of PubMedArticle objects with abstracts
    """
    pmids = search_pubmed(query, max_results)
    return fetch_articles(pmids)
