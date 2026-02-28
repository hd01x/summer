"""
PCoA Demo — FastAPI Backend
Medical Aspect-Based Summarization with Phrase-Level Context Attribution
ACL 2026 Demo Track
"""

import os
import time
from typing import List, Optional

import sys
# Startup check: verify OPENAI_API_KEY is available
_key = os.getenv("OPENAI_API_KEY", "")
print(f"[STARTUP] OPENAI_API_KEY set: {bool(_key)}, length: {len(_key)}", flush=True)

import requests as http_requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pcoa.config import MEDICAL_ASPECTS, ASPECT_ORDER
from pcoa.pubmed import search_and_fetch, fetch_single_article, PubMedArticle
from pcoa.text_processing import index_sentences
from pcoa.pipeline import analyze_article, AspectResult
from pcoa.fulltext import fetch_pmc_fulltext, extract_pdf_text

app = FastAPI(title="PCoA", version="1.0.0")

# --- Pydantic models ---

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

class PMIDRequest(BaseModel):
    pmid: str

class PMCRequest(BaseModel):
    pmc_id: str

class ArticleOut(BaseModel):
    pmid: str
    title: str
    abstract: List[str]
    authors: List[str]
    journal: str
    pub_date: str
    doi: str
    is_fulltext: bool = False
    is_truncated: bool = False
    word_count: int = 0

class AnalyzeRequest(BaseModel):
    pmid: str
    title: str
    abstract: List[str]
    aspects: List[str]
    strategy: str = "prior"
    model: str = ""
    api_key: str = ""

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

class EvalSentence(BaseModel):
    id: str
    text: str

class EvaluateRequest(BaseModel):
    summary: str
    sentences: List[EvalSentence]
    kps: List[str]

class ClaimCitation(BaseModel):
    claim: str
    sentences: List[int]

class CitationClaim(BaseModel):
    sentence_index: int
    claims: List[str]

class EvaluateResponse(BaseModel):
    ecr: str
    ssr: str
    cpr: str
    claims_citations: List[ClaimCitation] = []
    citations_claims: List[CitationClaim] = []
    contributory_kps: List[str] = []

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
            model=req.model,
            api_key=req.api_key,
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

@app.post("/api/fetch-pmc")
async def fetch_pmc(req: PMCRequest) -> ArticleOut:
    try:
        ft = fetch_pmc_fulltext(req.pmc_id.strip())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return ArticleOut(
        pmid=ft.pmc_id,
        title=ft.title,
        abstract=ft.text,
        authors=ft.authors,
        journal=ft.journal,
        pub_date=ft.pub_date,
        doi=ft.doi,
        is_fulltext=True,
        is_truncated=ft.is_truncated,
        word_count=ft.word_count,
    )

_MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 MB

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> ArticleOut:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")
    contents = await file.read()
    if len(contents) > _MAX_PDF_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit")
    try:
        ft = extract_pdf_text(contents)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF extraction failed: {e}")
    if not any(s.strip() for s in ft.text):
        raise HTTPException(status_code=422, detail="No text could be extracted from this PDF")
    return ArticleOut(
        pmid=f"pdf-{file.filename}",
        title=ft.title,
        abstract=ft.text,
        authors=ft.authors,
        journal=ft.journal,
        pub_date=ft.pub_date,
        doi=ft.doi,
        is_fulltext=True,
        is_truncated=ft.is_truncated,
        word_count=ft.word_count,
    )

EVAL_API_URL = os.environ.get("EVAL_API_URL", "http://34.32.24.46:8000")

_EVAL_MAX_RETRIES = 3
_EVAL_RETRY_DELAY = 1.5  # seconds

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    if not req.summary or not req.summary.strip():
        raise HTTPException(status_code=422, detail="Cannot evaluate an empty summary")

    payload = {
        "summary": req.summary,
        "sentences": [{"id": s.id, "text": s.text} for s in req.sentences],
        "kps": req.kps,
    }
    last_error = None

    for attempt in range(_EVAL_MAX_RETRIES):
        try:
            resp = http_requests.post(EVAL_API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return EvaluateResponse(
                ecr=data.get("ecr", "0/0"),
                ssr=data.get("ssr", "0/0"),
                cpr=data.get("cpr", "0/0"),
                claims_citations=[
                    ClaimCitation(claim=d.get("claim", ""), sentences=d.get("sentences", []))
                    for d in data.get("claims_citations", [])
                ],
                citations_claims=[
                    CitationClaim(sentence_index=d.get("sentence_index", 0), claims=d.get("claims", []))
                    for d in data.get("citations_claims", [])
                ],
                contributory_kps=data.get("contributory_kps", []),
            )
        except http_requests.exceptions.ConnectionError:
            last_error = "Evaluation service unavailable"
        except http_requests.exceptions.Timeout:
            last_error = "Evaluation service timed out"
        except http_requests.exceptions.HTTPError as e:
            last_error = f"Evaluation service error: {e}"
        # Wait before retrying
        if attempt < _EVAL_MAX_RETRIES - 1:
            time.sleep(_EVAL_RETRY_DELAY)

    raise HTTPException(status_code=502, detail=last_error)

# Mount static files last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")
