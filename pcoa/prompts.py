"""
Original prompts from the PCoA paper (Appendix C).
Each prompt template is faithfully reproduced from Tables 6-11.

Three context attribution strategies:
- Intrinsic (C.1, Table 6): Single-step generation of summary + citations + phrases
- Prior (C.3, Tables 8-9): Two-step — first retrieve sentences & phrases, then summarize
- Post-Hoc (C.4, Tables 10-11): Two-step — first summarize, then retrieve sentences & phrases

Plus subclaim decomposition (C.2, Table 7) for evaluation.
"""

from pcoa.config import MedicalAspect


# =============================================================================
# C.1 — Intrinsic Context Attribution (Table 6)
# =============================================================================

def build_intrinsic_prompt(aspect: MedicalAspect, formatted_abstract: str) -> str:
    """
    Build the intrinsic context attribution prompt (Table 6).
    Single-step: generates summary, cited sentences, and key phrases together.

    Args:
        aspect: The target medical aspect
        formatted_abstract: Abstract formatted as indexed sentences

    Returns:
        Complete prompt string
    """
    return f"""#Instruction:
Given a medical abstract presented as a list of sentences, perform the following tasks:
1. Provide a list of indices of the sentences that contain information about the study's {aspect.description}.
2. From those sentences, extract a list of key phrases (e.g., {aspect.key_phrase_examples}) that are relevant to the {aspect.description}.
3. Based on the relevant sentences and extracted key phrases, summarize the {aspect.description} of the study in one concise sentence.

#Abstract:
{formatted_abstract}

#Indices of involved sentences:
#Key phrases:
#Summary:"""


# =============================================================================
# C.3 — Prior Context Attribution (Tables 8 and 9)
# =============================================================================

def build_prior_step1_prompt(aspect: MedicalAspect, formatted_abstract: str) -> str:
    """
    Build the prior context attribution step 1 prompt (Table 8).
    Retrieves relevant sentences and extracts contributory phrases.

    Args:
        aspect: The target medical aspect
        formatted_abstract: Abstract formatted as indexed sentences

    Returns:
        Complete prompt string
    """
    return f"""#Instructions:
Given a medical abstract presented as a list of sentences, perform the following tasks:
1. Provide a list of indices of the sentences that contain or contribute to the {aspect.description}.
2. From those sentences, extract a list of key phrases (e.g., {aspect.key_phrase_examples}) that are relevant to the {aspect.description}.

#Abstract:
{formatted_abstract}

1. Indices of involved sentences:
2. Key phrases:"""


def build_prior_step2_prompt(aspect: MedicalAspect, formatted_sentences: str, key_phrases: str) -> str:
    """
    Build the prior context attribution step 2 prompt (Table 9).
    Generates a summary from retrieved sentences and extracted phrases.

    Args:
        aspect: The target medical aspect
        formatted_sentences: The retrieved sentences formatted as indexed list
        key_phrases: The extracted key phrases as a string

    Returns:
        Complete prompt string
    """
    return f"""#Instructions:
Given a list of sentences containing information about the study's {aspect.description}, and a corresponding list of key phrases (e.g., {aspect.key_phrase_examples}), summarize the study's {aspect.description} in one concise and coherent sentence and try to include as much information as possible from the key phrases.

#Input Sentences:
{formatted_sentences}

#Input Key Phrases:
{key_phrases}

#Summary:"""


# =============================================================================
# C.4 — Post-Hoc Context Attribution (Tables 10 and 11)
# =============================================================================

def build_posthoc_step1_prompt(aspect: MedicalAspect, formatted_abstract: str) -> str:
    """
    Build the post-hoc context attribution step 1 prompt (Table 10).
    Generates a summary directly from the abstract.

    Args:
        aspect: The target medical aspect
        formatted_abstract: Abstract formatted as indexed sentences

    Returns:
        Complete prompt string
    """
    return f"""#Instructions:
Given a medical abstract presented as a list of sentences, summarize the {aspect.description} of the study in one concise sentence.

#Abstract:
{formatted_abstract}

#Summary:"""


def build_posthoc_step2_prompt(aspect: MedicalAspect, formatted_abstract: str, summary: str) -> str:
    """
    Build the post-hoc context attribution step 2 prompt (Table 11).
    Retrieves sentences and phrases related to the generated summary.

    Args:
        aspect: The target medical aspect
        formatted_abstract: Abstract formatted as indexed sentences
        summary: The generated summary from step 1

    Returns:
        Complete prompt string
    """
    return f"""#Instructions:
Given a medical abstract provided as a list of sentences, along with a summary describing the study's {aspect.description}, perform the following tasks:
1. Identify and return the indices of sentences in the abstract that are relevant to the given summary of the study's {aspect.description}.
2. From the identified sentences, extract a list of key phrases that are pertinent to the study {aspect.description} (e.g., {aspect.key_phrase_examples}).

#Abstract:
{formatted_abstract}

#Summary:
{summary}

1. Indices of involved sentences:
2. Key phrases:"""


# =============================================================================
# C.2 — Subclaim Decomposition (Table 7)
# =============================================================================

def build_subclaim_decomposition_prompt(summary: str) -> str:
    """
    Build the subclaim decomposition prompt (Table 7).
    Used in the evaluation framework to decompose summaries into atomic subclaims.

    Args:
        summary: The summary to decompose

    Returns:
        Complete prompt string
    """
    return f"""Instructions:
Please decompose the given summary into a list of atomic subclaims, where each subclaim represents a single factual statement from the summary. Output only a list of sentences, and do not output additional information.

Demonstration:

Summary:
Long-term follow-up showed no new safety concerns, and results were consistent with the known tolerability profile of encorafenib plus binimetinib.

List of subclaims:
["Long-term follow-up was conducted.", "No new safety concerns were found.", "Results were consistent with the known tolerability profile of encorafenib plus binimetinib."]

Summary:
{summary}

List of subclaims:"""
