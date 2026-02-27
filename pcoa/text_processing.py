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
    Split text into sentences using NLTK sentence tokenizer,
    with protection for decimal numbers and common abbreviations.
    """
    ensure_nltk_data()
    
    # Protect decimal points (e.g., 9.79, 83.87%, 0.001)
    protected = re.sub(r'(\d)\.\s*(\d)', r'\1<DOT>\2', text)
    
    # Protect common URL-like patterns (e.g., ClinicalTrials.gov)
    protected = re.sub(r'(\w)\.(gov|org|com|edu|net|io)\b', r'\1<DOT>\2', protected)
    
    # Protect common abbreviations (e.g., vs., etc., i.e., e.g., al., no.)
    protected = re.sub(r'\b(vs|etc|i\.e|e\.g|al|no|Dr|Mr|Mrs|Fig|vol)\.\s', 
                       lambda m: m.group(0).replace('.', '<DOT>'), protected)
    
    # Protect periods before closing parentheses (e.g., "NCT03470922.).")
    protected = re.sub(r'\.(\))', r'<DOT>\1', protected)
    
    sentences = nltk.sent_tokenize(protected)
    
    # Restore protected dots and clean up whitespace
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
    
    # Post-process: merge sentences that are just ")." or similar trailing punctuation
    merged_sentences = []
    for sentence in sentences:
        if sentence in [').',  ').', ').)', ')', '.)']:
            # Merge with previous sentence if it exists
            if merged_sentences:
                merged_sentences[-1] += ' ' + sentence
            else:
                merged_sentences.append(sentence)
        else:
            merged_sentences.append(sentence)
    
    return merged_sentences


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


def format_indexed_abstract(sentences: List[str]) -> str:
    """
    Format a list of sentences as indexed sentences for prompt input.
    This matches the format expected by the LLM prompts from the paper.

    Example output:
        [1] First sentence of the abstract.
        [2] Second sentence of the abstract.

    Args:
        sentences: List of sentence strings

    Returns:
        Formatted string with indexed sentences
    """
    lines = [f"[{i + 1}] {sent}" for i, sent in enumerate(sentences)]
    return "\n".join(lines)


def get_sentences_by_indices(sentences: List[str], indices: List[int]) -> List[Tuple[int, str]]:
    """
    Get specific sentences by their 1-based indices.

    Args:
        sentences: List of sentence strings
        indices: List of 1-based sentence indices

    Returns:
        List of (index, sentence) tuples for the requested indices
    """
    result = []
    for idx in indices:
        if 1 <= idx <= len(sentences):
            result.append((idx, sentences[idx - 1]))
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