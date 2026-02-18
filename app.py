"""
PCoA Demo — FastAPI Backend
Medical Aspect-Based Summarization with Phrase-Level Context Attribution
ACL 2026 Demo Track
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pcoa.config import MEDICAL_ASPECTS, ASPECT_ORDER
from pcoa.pubmed import search_and_fetch, fetch_single_article, PubMedArticle
from pcoa.text_processing import index_sentences
from pcoa.pipeline import analyze_article, AspectResult

app = FastAPI(title="PCoA", version="1.0.0")

# --- Pydantic models ---

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

class PMIDRequest(BaseModel):
    pmid: str

class ArticleOut(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    doi: str

class AnalyzeRequest(BaseModel):
    pmid: str
    title: str
    abstract: str
    aspects: List[str]
    strategy: str = "prior"

class SentenceOut(BaseModel):
    index: int
    text: str

class AspectResultOut(BaseModel):
    aspect_code: str
    aspect_name: str
    strategy: str
    summary: str
    sentence_indices: List[int]
    cited_sentences: List[SentenceOut]
    key_phrases: List[str]
    error: str

class AnalysisOut(BaseModel):
    pmid: str
    title: str
    strategy: str
    sentences: List[SentenceOut]
    results: List[AspectResultOut]

class AspectInfo(BaseModel):
    code: str
    name: str
    full_name: str
    description: str
    example_summary: str

# --- Routes ---

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/aspects")
async def get_aspects() -> List[AspectInfo]:
    return [
        AspectInfo(
            code=a.code,
            name=a.name,
            full_name=a.full_name,
            description=a.description,
            example_summary=a.example_summary,
        )
        for a in [MEDICAL_ASPECTS[c] for c in ASPECT_ORDER]
    ]

@app.post("/api/search")
async def search(req: SearchRequest) -> List[ArticleOut]:
    try:
        articles = search_and_fetch(req.query, req.max_results)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return [
        ArticleOut(
            pmid=a.pmid, title=a.title, abstract=a.abstract,
            authors=a.authors, journal=a.journal, pub_date=a.pub_date, doi=a.doi,
        )
        for a in articles
    ]

@app.post("/api/fetch")
async def fetch_pmid(req: PMIDRequest) -> ArticleOut:
    try:
        a = fetch_single_article(req.pmid.strip())
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return ArticleOut(
        pmid=a.pmid, title=a.title, abstract=a.abstract,
        authors=a.authors, journal=a.journal, pub_date=a.pub_date, doi=a.doi,
    )

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> AnalysisOut:
    if not req.aspects:
        raise HTTPException(400, "Select at least one aspect")
    if req.strategy not in ("prior", "intrinsic", "post-hoc"):
        raise HTTPException(400, "Invalid strategy")
    try:
        analysis = analyze_article(
            abstract=req.abstract,
            pmid=req.pmid,
            title=req.title,
            aspect_codes=req.aspects,
            strategy=req.strategy,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sentences = [SentenceOut(index=i, text=t) for i, t in analysis.indexed_sentences]
    results = []
    for code in req.aspects:
        r = analysis.aspect_results.get(code)
        if not r:
            continue
        results.append(AspectResultOut(
            aspect_code=r.aspect_code,
            aspect_name=r.aspect_name,
            strategy=r.strategy,
            summary=r.summary,
            sentence_indices=r.sentence_indices,
            cited_sentences=[SentenceOut(index=i, text=t) for i, t in r.cited_sentences],
            key_phrases=r.key_phrases,
            error=r.error,
        ))
    return AnalysisOut(
        pmid=analysis.pmid,
        title=analysis.title,
        strategy=analysis.strategy,
        sentences=sentences,
        results=results,
    )

# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")
