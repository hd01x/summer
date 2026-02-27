"""
Core PCoA pipeline implementing the three context attribution strategies
from the paper (§2.2, §5.1, §5.2):
1. Intrinsic Context Attribution
2. Prior Context Attribution (best performing)
3. Post-Hoc Context Attribution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pcoa.config import MedicalAspect, MEDICAL_ASPECTS, ASPECT_ORDER
from pcoa.text_processing import (
    format_indexed_abstract,
    get_sentences_by_indices,
    format_sentences_for_prompt,
)
from pcoa.prompts import (
    build_posthoc_step1_prompt,
    build_posthoc_step2_prompt,
    build_subclaim_decomposition_prompt,
)
from pcoa.prompt_prior import (
    prompt_prior_step1,
    prompt_prior_step2
)
from pcoa.prompt_intrinsic import prompt_intrinsic
from pcoa.llm import (
    call_llm,
    parse_intrinsic_response,
    parse_prior_step1_response,
    parse_prior_step2_response,
    parse_posthoc_step1_response,
    parse_posthoc_step2_response,
    parse_subclaim_response,
)


@dataclass
class AspectResult:
    """Result of processing a single aspect for an article."""
    aspect_code: str
    aspect_name: str
    strategy: str
    summary: str = ""
    sentence_indices: List[int] = field(default_factory=list)
    cited_sentences: List[Tuple[int, str]] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    raw_responses: List[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ArticleAnalysis:
    """Complete analysis of an article across all requested aspects."""
    pmid: str
    title: str
    abstract: List[str]
    indexed_sentences: List[Tuple[int, str]] = field(default_factory=list)
    aspect_results: Dict[str, AspectResult] = field(default_factory=dict)
    strategy: str = ""


def run_intrinsic(abstract: List[str], aspect: MedicalAspect, **llm_kwargs) -> AspectResult:
    """
    Run intrinsic context attribution strategy (§C.1).
    Single-step: generates summary, citations, and phrases together.

    Args:
        abstract: List of sentence strings
        aspect: Target medical aspect
        **llm_kwargs: Forwarded to call_llm (model, api_key)

    Returns:
        AspectResult with all three components
    """
    result = AspectResult(
        aspect_code=aspect.code,
        aspect_name=aspect.name,
        strategy="intrinsic",
    )
    formatted = format_indexed_abstract(abstract)
    prompt = prompt_intrinsic(aspect.code, formatted)
    raw = call_llm(prompt, **llm_kwargs)
    result.raw_responses.append(raw)

    parsed = parse_intrinsic_response(raw)
    result.sentence_indices = parsed.sentence_indices
    result.key_phrases = parsed.key_phrases
    result.summary = parsed.summary
    result.cited_sentences = get_sentences_by_indices(abstract, result.sentence_indices)

    if not result.summary:
        result.error = "Failed to extract summary from LLM response"
        
    return result


def run_prior(abstract: List[str], aspect: MedicalAspect, **llm_kwargs) -> AspectResult:
    result = AspectResult(
        aspect_code=aspect.code,
        aspect_name=aspect.name,
        strategy="prior",
    )

    formatted = format_indexed_abstract(abstract)

    # Step 1: Retrieve relevant sentences and extract key phrases
    step1_prompt = prompt_prior_step1(aspect.code, formatted)
    step1_raw = call_llm(step1_prompt, **llm_kwargs)
    result.raw_responses.append(step1_raw)

    indices, phrases = parse_prior_step1_response(step1_raw)
    result.sentence_indices = indices
    result.key_phrases = phrases
    result.cited_sentences = get_sentences_by_indices(abstract, indices)

    if not result.cited_sentences:
        result.error = "No relevant sentences identified in step 1"
        return result

    # Step 2: Generate summary from retrieved sentences and phrases
    formatted_sents = format_sentences_for_prompt(result.cited_sentences)
    phrases_str = ", ".join(f'"{p}"' for p in phrases) if phrases else "N/A"

    step2_prompt = prompt_prior_step2(aspect.code, formatted_sents, phrases_str)
    step2_raw = call_llm(step2_prompt, **llm_kwargs)
    result.raw_responses.append(step2_raw)

    result.summary = parse_prior_step2_response(step2_raw)

    if not result.summary:
        result.error = "Failed to extract summary from step 2 response"

    return result


def run_posthoc(abstract: List[str], aspect: MedicalAspect, **llm_kwargs) -> AspectResult:
    """
    Run post-hoc context attribution strategy (§C.4).
    Two-step: first summarize, then retrieve sentences & phrases.

    Args:
        abstract: List of sentence strings
        aspect: Target medical aspect
        **llm_kwargs: Forwarded to call_llm (model, api_key)

    Returns:
        AspectResult with all three components
    """
    result = AspectResult(
        aspect_code=aspect.code,
        aspect_name=aspect.name,
        strategy="post-hoc",
    )

    formatted = format_indexed_abstract(abstract)

    # Step 1: Generate summary
    step1_prompt = build_posthoc_step1_prompt(aspect, formatted)
    step1_raw = call_llm(step1_prompt, **llm_kwargs)
    result.raw_responses.append(step1_raw)

    summary = parse_posthoc_step1_response(step1_raw)
    result.summary = summary

    if not result.summary:
        result.error = "Failed to generate summary in step 1"
        return result

    # Step 2: Retrieve sentences and phrases for the summary
    step2_prompt = build_posthoc_step2_prompt(aspect, formatted, summary)
    step2_raw = call_llm(step2_raw, **llm_kwargs)
    result.raw_responses.append(step2_raw)

    indices, phrases = parse_posthoc_step2_response(step2_raw)
    result.sentence_indices = indices
    result.key_phrases = phrases
    result.cited_sentences = get_sentences_by_indices(abstract, indices)

    return result


# Strategy dispatcher
STRATEGIES = {
    "intrinsic": run_intrinsic,
    "prior": run_prior,
    "post-hoc": run_posthoc,
}


def analyze_article(
    abstract: List[str],
    pmid: str = "",
    title: str = "",
    aspect_codes: Optional[List[str]] = None,
    strategy: str = "prior",
    progress_callback=None,
    model: str = "",
    api_key: str = "",
) -> ArticleAnalysis:
    """
    Run the full PCoA pipeline on an article.

    Args:
        abstract: List of sentence strings
        pmid: PubMed ID
        title: Article title
        aspect_codes: List of aspect codes to analyze (default: all 16)
        strategy: Attribution strategy ("intrinsic", "prior", or "post-hoc")
        progress_callback: Optional callback(aspect_code, i, total) for progress
        model: UI model name (e.g. "GPT-5.2") forwarded to call_llm
        api_key: User-provided API key forwarded to call_llm

    Returns:
        Complete ArticleAnalysis
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(STRATEGIES.keys())}")

    strategy_fn = STRATEGIES[strategy]
    codes = aspect_codes or ASPECT_ORDER

    llm_kwargs = {}
    if model:
        llm_kwargs["model"] = model
    else:
        llm_kwargs["model"] = 'gpt-4o-mini'
    if api_key:
        llm_kwargs["api_key"] = api_key
        

    analysis = ArticleAnalysis(
        pmid=pmid,
        title=title,
        abstract=abstract,
        strategy=strategy,
    )

    for i, code in enumerate(codes):
        aspect = MEDICAL_ASPECTS[code]

        if progress_callback:
            progress_callback(code, i, len(codes))

        try:
            aspect_result = strategy_fn(abstract, aspect, **llm_kwargs)
        except Exception as e:
            aspect_result = AspectResult(
                aspect_code=code,
                aspect_name=aspect.name,
                strategy=strategy,
                error=str(e),
            )

        analysis.aspect_results[code] = aspect_result

    return analysis
