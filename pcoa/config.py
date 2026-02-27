"""
Configuration module for PCoA pipeline.
Defines the 16 medical aspects from the paper (Table 3) with their
descriptions, prompt-specific wordings, and example key phrases.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Default OpenAI model (paper uses GPT-4o)
OPENAI_MODEL = "gpt-4o"
TEMPERATURE = 0


@dataclass
class MedicalAspect:
    """Definition of a single medical aspect from Table 3 of the PCoA paper."""
    code: str
    name: str
    full_name: str
    description: str  # Used in prompts as the aspect description
    key_phrase_examples: str  # Used in prompts as example key phrases
    example_summary: str  # Example summary from the paper


# All 16 medical aspects as defined in Table 3 of the paper
MEDICAL_ASPECTS: Dict[str, MedicalAspect] = {
    "OB": MedicalAspect(
        code="OB",
        name="Objective",
        full_name="Objective",
        description="objective",
        key_phrase_examples="indicators to be measured, target diseases, treatments",
        example_summary="To determine the 5-year survival rates of patients with unresectable or metastatic melanoma who derive long-term benefit from combination therapy with BRAF and MEK inhibitors.",
    ),
    "P": MedicalAspect(
        code="P",
        name="Participants",
        full_name="Participants",
        description="participants",
        key_phrase_examples="number of participants, diseases, inclusion criteria",
        example_summary="A total of 563 previously untreated patients with unresectable or metastatic melanoma and a BRAF V600E or V600K mutation participated in the study.",
    ),
    "I": MedicalAspect(
        code="I",
        name="Intervention",
        full_name="Intervention",
        description="intervention",
        key_phrase_examples="mode of administration, drug name, dosage",
        example_summary="Patients received first-line treatment with the BRAF inhibitor dabrafenib (150 mg twice daily) plus the MEK inhibitor trametinib (2 mg once daily).",
    ),
    "C": MedicalAspect(
        code="C",
        name="Comparator",
        full_name="Comparator",
        description="control group description",
        key_phrase_examples="control group name, mode of administration",
        example_summary="The comparator was ipilimumab monotherapy, administered at 3 mg/kg every three weeks for four doses.",
    ),
    "O": MedicalAspect(
        code="O",
        name="Outcomes",
        full_name="Outcomes",
        description="outcomes",
        key_phrase_examples="measured endpoints, values, statistical results",
        example_summary="Approximately one-third of patients experienced long-term benefits, with a 5-year OS rate of 34%.",
    ),
    "F": MedicalAspect(
        code="F",
        name="Findings",
        full_name="Findings",
        description="findings",
        key_phrase_examples="key results, conclusions, statistical significance",
        example_summary="Nivolumab plus ipilimumab or nivolumab alone demonstrated durable, improved clinical outcomes over ipilimumab monotherapy in advanced melanoma.",
    ),
    "M": MedicalAspect(
        code="M",
        name="Medicines",
        full_name="Medicines",
        description="medicines used",
        key_phrase_examples="medicine names, drug types, pharmacological agents",
        example_summary="Used medicines were dabrafenib and trametinib.",
    ),
    "TD": MedicalAspect(
        code="TD",
        name="Treatment Duration",
        full_name="Treatment Duration",
        description="treatment duration",
        key_phrase_examples="duration of treatment, conditions for stopping treatment",
        example_summary="Weekly administration of ontuxizumab without dose change until disease progression.",
    ),
    "PE": MedicalAspect(
        code="PE",
        name="Primary Endpoints",
        full_name="Primary Endpoints",
        description="primary endpoints",
        key_phrase_examples="primary outcome measures, primary efficacy endpoints",
        example_summary="Primary endpoint was progression-free survival.",
    ),
    "SE": MedicalAspect(
        code="SE",
        name="Secondary Endpoints",
        full_name="Secondary Endpoints",
        description="secondary endpoints",
        key_phrase_examples="secondary outcome measures, secondary efficacy endpoints",
        example_summary="Secondary endpoint was long-term survival rates.",
    ),
    "FD": MedicalAspect(
        code="FD",
        name="Follow-Up Duration",
        full_name="Follow-Up Duration",
        description="follow-up duration",
        key_phrase_examples="follow-up period, median follow-up duration",
        example_summary="The median follow-up duration was 22 months.",
    ),
    "AE": MedicalAspect(
        code="AE",
        name="Adverse Events",
        full_name="Adverse Events",
        description="adverse events",
        key_phrase_examples="adverse event types, frequencies, side effects",
        example_summary="The most common adverse events overall were headache (55.3%).",
    ),
    "R": MedicalAspect(
        code="R",
        name="Randomization",
        full_name="Randomization",
        description="randomization",
        key_phrase_examples="randomization ratio, treatment groups, allocation",
        example_summary="Patients were randomized in a 2:1 to receive either binimetinib or dacarbazine.",
    ),
    "B": MedicalAspect(
        code="B",
        name="Blinding",
        full_name="Blinding",
        description="blinding",
        key_phrase_examples="blinding type, study design (single-blind, double-blind, open-label)",
        example_summary="The study was double-blind.",
    ),
    "FU": MedicalAspect(
        code="FU",
        name="Funding",
        full_name="Funding",
        description="funding",
        key_phrase_examples="sponsor, funding source, financial support",
        example_summary="The study was funded by Bristol-Myers Squibb.",
    ),
    "RE": MedicalAspect(
        code="RE",
        name="Registration",
        full_name="Registration",
        description="registration",
        key_phrase_examples="trial registration number, registry name",
        example_summary="The ClinicalTrials.gov number is NCT03698019.",
    ),
}

# Ordered list of aspect codes matching the paper's presentation order
ASPECT_ORDER = ["OB", "P", "I", "C", "O", "F", "M", "TD", "PE", "SE", "FD", "AE", "R", "B", "FU", "RE"]

# Default PubMed search query (matches paper's criteria from §3.1)
DEFAULT_PUBMED_QUERY = 'melanoma[MeSH Terms] AND "randomized controlled trial"[Publication Type] AND english[Language]'
