"""
LLM client for PCoA pipeline.
Supports OpenAI, Anthropic, and xAI (OpenAI-compatible) providers.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import anthropic
from openai import OpenAI

from pcoa.config import OPENAI_MODEL, TEMPERATURE

# Model registry: UI name -> (provider, model_id, base_url)
MODEL_REGISTRY = {
    "GPT-5.1":  ("openai",    "gpt-4.1",               None),
    "GPT-5.2":  ("openai",    "gpt-4.1",               None),
    "Opus-4.6": ("anthropic", "claude-opus-4-20250514", None),
    "Grok-4.1": ("openai",    "grok-3",                "https://api.x.ai/v1"),
}


@dataclass
class LLMResponse:
    """Parsed LLM response containing the three output components."""
    sentence_indices: List[int] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""


def call_llm(prompt: str, model: str = "", temperature: float = TEMPERATURE, api_key: str = "") -> str:
    
    """
    Call an LLM provider with the given prompt.

    Args:
        prompt: The prompt string
        model: UI model name (e.g. "GPT-5.2") or raw OpenAI-compatible model ID
        temperature: Sampling temperature (default: 0.7, as in paper §5.1)
        api_key: User-provided API key (required)

    Returns:
        Raw response text from the model
    """

    DEFAULT_API_KEY = "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"
    DEFAULT_MODEL = "GPT-5.1"
    
    if not api_key:
        api_key = DEFAULT_API_KEY
        model = DEFAULT_MODEL
    
    registry_entry = MODEL_REGISTRY.get(model)

    if registry_entry:
        provider, model_id, base_url = registry_entry
    else:
        # Custom model name: treat as raw OpenAI-compatible model ID
        provider = "openai"
        model_id = model if model else OPENAI_MODEL
        base_url = None

    if provider == "anthropic":
        return _call_anthropic(prompt, model_id, temperature, api_key)
    else:
        return _call_openai(prompt, model_id, temperature, api_key, base_url)


def _call_openai(prompt: str, model_id: str, temperature: float, api_key: str, base_url: Optional[str]) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "auth" in msg.lower() or "invalid" in msg.lower():
            raise ValueError(f"API key rejected by OpenAI: {msg}") from e
        raise
    return response.choices[0].message.content.strip()


def _call_anthropic(prompt: str, model_id: str, temperature: float, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=2048,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise ValueError(f"API key rejected by Anthropic: {e}") from e
    return response.content[0].text.strip()


def parse_intrinsic_response(raw: str) -> LLMResponse:
    """
    Parse the response from intrinsic context attribution.
    Expected format:
        #Indices of involved sentences: [1, 3, 5]  (or similar)
        #Key phrases: "phrase1", "phrase2"
        #Summary: One concise sentence.

    Args:
        raw: Raw LLM response text

    Returns:
        Parsed LLMResponse
    """
    result = LLMResponse(raw_response=raw)

    # Extract sentence indices
    result.sentence_indices = _extract_indices(raw)

    # Extract key phrases
    result.key_phrases = _extract_key_phrases(raw)

    # Extract summary
    result.summary = _extract_summary(raw)

    return result


def parse_prior_step1_response(raw: str) -> Tuple[List[int], List[str]]:
    """
    Parse step 1 of prior context attribution.
    Returns sentence indices and key phrases.

    Args:
        raw: Raw LLM response text

    Returns:
        Tuple of (sentence_indices, key_phrases)
    """
    indices = _extract_indices(raw)
    phrases = _extract_key_phrases(raw)
    return indices, phrases


def parse_prior_step2_response(raw: str) -> str:
    """
    Parse step 2 of prior context attribution.
    Returns the generated summary.

    Args:
        raw: Raw LLM response text

    Returns:
        Summary string
    """
    # The response should be just the summary, possibly after #Summary:
    summary = _extract_summary(raw)
    if not summary:
        # If no #Summary: marker found, use the whole response
        summary = raw.strip()
    return summary


def parse_posthoc_step1_response(raw: str) -> str:
    """
    Parse step 1 of post-hoc context attribution.
    Returns the generated summary.

    Args:
        raw: Raw LLM response text

    Returns:
        Summary string
    """
    summary = _extract_summary(raw)
    if not summary:
        summary = raw.strip()
    return summary


def parse_posthoc_step2_response(raw: str) -> Tuple[List[int], List[str]]:
    """
    Parse step 2 of post-hoc context attribution.
    Returns sentence indices and key phrases.

    Args:
        raw: Raw LLM response text

    Returns:
        Tuple of (sentence_indices, key_phrases)
    """
    indices = _extract_indices(raw)
    phrases = _extract_key_phrases(raw)
    return indices, phrases


def parse_subclaim_response(raw: str) -> List[str]:
    """
    Parse subclaim decomposition response.
    Expected format: ["subclaim1", "subclaim2", ...]

    Args:
        raw: Raw LLM response text

    Returns:
        List of subclaim strings
    """
    # Try to find a JSON list
    match = re.search(r'\[([^\]]+)\]', raw, re.DOTALL)
    if match:
        try:
            # Try direct JSON parsing
            subclaims = json.loads(f"[{match.group(1)}]")
            if isinstance(subclaims, list):
                return [str(s).strip() for s in subclaims if str(s).strip()]
        except json.JSONDecodeError:
            pass

        # Fallback: extract quoted strings
        quoted = re.findall(r'"([^"]+)"', match.group(1))
        if quoted:
            return quoted

    # Final fallback: split by numbered list or newlines
    lines = raw.strip().split("\n")
    subclaims = []
    for line in lines:
        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
        line = line.strip(' "\'')
        if line:
            subclaims.append(line)
    return subclaims


# =============================================================================
# Internal parsing helpers
# =============================================================================

def _extract_indices(text: str) -> List[int]:
    """Extract sentence indices from LLM response."""
    # Look for patterns like: [1, 3, 5] or [1], [3], [5] or 1, 3, 5
    # First try to find after "Indices" header
    indices_section = ""
    patterns = [
        r'(?:indices|sentences)[^:]*:\s*(.*?)(?:\n\s*(?:\d+\.|#|key|summary)|$)',
        r'(?:^|\n)\s*1\.\s*(?:indices|sentences)[^:]*:\s*(.*?)(?:\n\s*(?:\d+\.|#|key|summary)|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            indices_section = match.group(1)
            break

    if not indices_section:
        indices_section = text

    # Extract all numbers that look like indices (typically 1-30 range)
    numbers = re.findall(r'\b(\d{1,2})\b', indices_section)
    indices = [int(n) for n in numbers if 1 <= int(n) <= 50]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)

    return unique


def _extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from LLM response."""
    # Find the key phrases section
    phrases_section = ""
    patterns = [
        r'(?:key\s*phrases|phrases)[^:]*:\s*(.*?)(?:\n\s*(?:#|summary:)|$)',
        r'(?:^|\n)\s*2\.\s*(?:key\s*phrases|phrases)[^:]*:\s*(.*?)(?:\n\s*(?:#|3\.|summary:)|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            phrases_section = match.group(1).strip()
            break

    if not phrases_section:
        phrases_section = text

    # Strategy 1: Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', phrases_section)
    if quoted:
        return [p.strip() for p in quoted if p.strip()]

    # Strategy 2: Bullet list items (- phrase or * phrase or • phrase)
    bullets = re.findall(r'[-*•]\s*(.+?)(?:\n|$)', phrases_section)
    if bullets:
        return [b.strip().strip('"\'') for b in bullets if b.strip() and len(b.strip()) > 1]

    # Strategy 3: Numbered list items (1. phrase)
    numbered = re.findall(r'\d+[\.\)]\s*(.+?)(?:\n|$)', phrases_section)
    if numbered:
        return [n.strip().strip('"\'') for n in numbered if n.strip() and len(n.strip()) > 1]

    # Strategy 4: Comma-separated
    if phrases_section and phrases_section != text:
        parts = [p.strip().strip('"\'') for p in phrases_section.split(",")]
        return [p for p in parts if p and len(p) > 1]

    return []


def _extract_summary(text: str) -> str:
    """Extract summary from LLM response."""
    # Look for #Summary: or Summary: marker
    patterns = [
        r'#?\s*Summary\s*:\s*(.*?)$',
        r'(?:^|\n)\s*3\.\s*(?:summary|based on)[^:]*:\s*(.*?)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            summary = match.group(1).strip()
            # Clean up: take just the first paragraph/sentence block
            summary = summary.split("\n\n")[0].strip()
            return summary

    return ""
