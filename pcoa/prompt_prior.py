ASPECT_CONFIG = {
    'OB': {
        'description': 'the study objective',
        'step1': 'Prefer sentences containing phrases such as "aim", "objective", "purpose", "to evaluate", "to assess", "to determine". Exclude methods, results, or conclusions unless they directly state the objective.',
        'step2': 'Include the target disease or population, comparison (if any), measured indicators, and any specific hypothesis being tested.',
    },
    'P': {
        'description': 'the participating patients',
        'step1': 'Prefer sentences reporting the number of participants, diagnosis or condition, age or age range, and geographic region or study setting.',
        'step2': 'Include number of participants, diagnosis, age, and region.',
    },
    'I': {
        'description': 'the intervention',
        'step1': 'Prefer sentences describing the type of intervention or treatment, mode of administration, dosage, and duration or treatment schedule.',
        'step2': 'Include mode of administration, dosage, and duration or treatment schedule.',
    },
    'C': {
        'description': 'the control group',
        'step1': 'Prefer sentences that clearly state the control group\'s name, treatment, administration mode, or schedule.',
        'step2': 'Include control group name, administration mode, and duration or treatment schedule.',
    },
    'O': {
        'description': 'the study outcomes',
        'step1': 'Prioritize sentences reporting measured indicators, endpoints, or results. Avoid interpretive or conclusion-only statements.',
        'step2': 'Include measured indicators (e.g., biomarkers, clinical scores, survival rates) and reported values or observed effects (e.g., percentage change, mean difference, hazard ratio).',
    },
    'F': {
        'description': 'the study conclusion',
        'step1': 'Prioritize sentences that clearly state the final conclusion (e.g., beginning with "In conclusion," "We conclude that"). Avoid detailed outcome reporting or general interpretations.',
        'step2': 'Include key phrases that represent the main conclusion of the study.',
    },
    'M': {
        'description': 'the medicines used in the treatment',
        'step1': 'Prioritize sentences that clearly name the medicines involved in the treatment.',
        'step2': 'Include the names of the medicines.',
    },
    'TD': {
        'description': 'the treatment duration',
        'step1': 'Prioritize sentences that clearly state duration, period, treatment frequency, or termination conditions.',
        'step2': 'Include duration, period, frequency, and termination conditions.',
    },
    'PE': {
        'description': 'the primary endpoints',
        'step1': 'Prioritize sentences that clearly introduce or define the primary endpoints. Avoid secondary endpoints or unrelated methodological details.',
        'step2': 'Include the name of primary endpoints.',
    },
    'SE': {
        'description': 'the secondary endpoints',
        'step1': 'Prioritize sentences that clearly introduce or define the secondary endpoints. Avoid primary endpoints or unrelated methodological details.',
        'step2': 'Include the name of secondary endpoints.',
    },
    'FD': {
        'description': 'the follow-up duration',
        'step1': 'Prioritize sentences that clearly state the follow-up duration (e.g., "patients were followed for 12 months", "median follow-up of 24 weeks").',
        'step2': 'Include follow-up length, time frame, or assessment period.',
    },
    'AE': {
        'description': 'the adverse events',
        'step1': 'Prioritize sentences reporting adverse event names, incidence rates or frequency, and severity or grade.',
        'step2': 'Include adverse event names, rates, and severity.',
    },
    'R': {
        'description': 'the randomization method',
        'step1': 'Prioritize sentences that clearly state how participants were randomized (e.g., random sequence generation, allocation method, stratification, block size).',
        'step2': 'Include randomization technique, allocation procedure, concealment method, and stratification.',
    },
    'B': {
        'description': 'the blinding method',
        'step1': 'Prioritize sentences that clearly state the blinding design (e.g., double-blind, single-blind, open-label) and who was blinded.',
        'step2': 'Include the name or type of blinding method and who was blinded.',
    },
    'FU': {
        'description': 'the funding',
        'step1': 'Prioritize sentences that clearly state the funding source (e.g., name of funder, grant support). Avoid unrelated acknowledgments or conflict-of-interest statements.',
        'step2': 'Include funder name(s) and grant numbers if explicitly stated.',
    },
    'RE': {
        'description': 'the registration details',
        'step1': 'Prioritize sentences that clearly state the trial registry name and/or registration number (e.g., ClinicalTrials.gov identifier). Avoid ethical approval statements.',
        'step2': 'Include registration platform name and registration number or identifier.',
    },
}


def _build_prior_step1_prompt(aspect_description, step1_guidance, step2_guidance, abstract):
    """Step 1: retrieve relevant sentences and extract key phrases."""
    return f"""# Instruction:
You are given a medical abstract formatted as a list of indexed sentences.
Your task is to identify supporting sentences and extract contributory phrases about {aspect_description} by performing the following steps:

Step 1 - Identify supporting sentences:
    - Select the sentence indices that most explicitly describe {aspect_description}.
    - {step1_guidance}
    - Avoid redundant, indirect, or background statements.

Step 2 - Extract contributory phrases from the identified sentences:
    - Extract short, meaningful noun or verb phrases that are directly relevant to {aspect_description}. Exclude phrases related to other aspects.
    - {step2_guidance}
    - Extract phrases exactly as they appear in the original text. Do not rephrase, rewrite, or paraphrase.


Output Format (strictly follow this structure):
# Indices:
[index1, index2, ...]

# Key Phrases:
['phrase1', 'phrase2', ...]


# Abstract:
{abstract}

# Indices:
[]

# Key Phrases:
[]
"""


def _build_prior_step2_prompt(aspect_description, key_phrases, sentences):
    """Step 2: generate summary from retrieved sentences and key phrases."""
    return f"""# Instruction:
You are given a list of sentences and key phrases about {aspect_description} extracted from these sentences.
Based solely on the provided sentences and key phrases, write ONE concise sentence summarizing {aspect_description}.
The summary should incorporate the extracted key phrases as much as possible.


# Input Sentences:
{sentences}

# Input Key Phrases:
{key_phrases}


Output Format (strictly follow this structure):
# Summary:
One concise sentence summarizing {aspect_description}.


# Summary:
"""


def prompt_prior_step1(aspect, abstract):
    config = ASPECT_CONFIG.get(aspect)
    if config is None:
        raise ValueError(f"Unknown aspect: {aspect}")
    return _build_prior_step1_prompt(
        aspect_description=config['description'],
        step1_guidance=config['step1'],
        step2_guidance=config['step2'],
        abstract=abstract,
    )


def prompt_prior_step2(aspect, sentences, key_phrases):
    config = ASPECT_CONFIG.get(aspect)
    if config is None:
        raise ValueError(f"Unknown aspect: {aspect}")
    return _build_prior_step2_prompt(
        aspect_description=config['description'],
        key_phrases=key_phrases,
        sentences=sentences,
    )