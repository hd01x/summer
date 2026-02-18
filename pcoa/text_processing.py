"""
Text processing utilities for PCoA pipeline.
Handles sentence splitting and indexing as described in the paper.
"""

import re
from typing import List, Tuple

import nltk


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK sentence tokenizer.
    As described in §3.5, NLTK is used for tokenization.

    Args:
        text: Input text (abstract)

    Returns:
        List of sentences
    """
    ensure_nltk_data()
    sentences = nltk.sent_tokenize(text)
    # Clean up whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def index_sentences(text: str) -> List[Tuple[int, str]]:
    """
    Split text into indexed sentences (1-based indexing).

    Args:
        text: Input text (abstract)

    Returns:
        List of (index, sentence) tuples with 1-based indices
    """
    sentences = split_into_sentences(text)
    return [(i + 1, sent) for i, sent in enumerate(sentences)]


def format_indexed_abstract(text: str) -> str:
    """
    Format an abstract as a list of indexed sentences for prompt input.
    This matches the format expected by the LLM prompts from the paper.

    Example output:
        [1] First sentence of the abstract.
        [2] Second sentence of the abstract.

    Args:
        text: Raw abstract text

    Returns:
        Formatted string with indexed sentences
    """
    indexed = index_sentences(text)
    lines = [f"[{idx}] {sent}" for idx, sent in indexed]
    return "\n".join(lines)


def get_sentences_by_indices(text: str, indices: List[int]) -> List[Tuple[int, str]]:
    """
    Get specific sentences by their 1-based indices.

    Args:
        text: Input text (abstract)
        indices: List of 1-based sentence indices

    Returns:
        List of (index, sentence) tuples for the requested indices
    """
    all_sentences = index_sentences(text)
    result = []
    for idx in indices:
        for sent_idx, sent in all_sentences:
            if sent_idx == idx:
                result.append((sent_idx, sent))
                break
    return result


def format_sentences_for_prompt(sentences: List[Tuple[int, str]]) -> str:
    """
    Format selected sentences for use in the prior attribution step 2 prompt.

    Args:
        sentences: List of (index, sentence) tuples

    Returns:
        Formatted string
    """
    lines = [f"[{idx}] {sent}" for idx, sent in sentences]
    return "\n".join(lines)


def tokenize_phrase(phrase: str) -> List[str]:
    """
    Tokenize a phrase into individual words using NLTK.
    As described in §4.3, NLTK is used as tokenizer T for contributory phrases.

    Args:
        phrase: Input phrase

    Returns:
        List of word tokens
    """
    ensure_nltk_data()
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return nltk.word_tokenize(phrase)
