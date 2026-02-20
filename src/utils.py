"""
Shared utilities for the SLM de-identification evaluation harness.

Provides: ASQ-PHI dataset loading, comprehensive tag detection (14 patterns),
tag normalization, Levenshtein distance, bootstrap CI computation, prompt hashing,
output parsers, and logging setup.
"""

import json
import re
import hashlib
import logging
import sys
from enum import Enum
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"

# PHI type mapping: ASQ-PHI identifier_type -> canonical redaction tag
PHI_TYPE_TO_TAG = {
    "NAME": "NAME",
    "GEOGRAPHIC_LOCATION": "LOCATION",
    "DATE": "DATE",
    "MEDICAL_RECORD_NUMBER": "MRN",
    "HEALTH_PLAN_BENEFICIARY_NUMBER": "ID",
    "PHONE_NUMBER": "PHONE",
    "SOCIAL_SECURITY_NUMBER": "ID",
    "EMAIL_ADDRESS": "EMAIL",
    "UNIQUE_IDENTIFIER": "ID",
    "ACCOUNT_NUMBER": "ID",
    "FAX_NUMBER": "PHONE",
    "CERTIFICATE_LICENSE_NUMBER": "ID",
    "IP_ADDRESS": "ID",
}

# Tag variant normalization: model-generated variants -> canonical types
TAG_VARIANT_MAP = {
    # NAME variants
    "PATIENT_NAME": "NAME", "PERSON": "NAME", "PATIENT": "NAME",
    "FIRST_NAME": "NAME", "LAST_NAME": "NAME", "FULL_NAME": "NAME",
    "DOCTOR": "NAME", "PHYSICIAN": "NAME", "CLINICIAN": "NAME",
    # LOCATION variants
    "PLACE": "LOCATION", "FACILITY": "LOCATION", "HOSPITAL": "LOCATION",
    "ADDRESS": "LOCATION", "CITY": "LOCATION", "STATE": "LOCATION",
    "COUNTRY": "LOCATION", "CLINIC": "LOCATION", "INSTITUTION": "LOCATION",
    "GEOGRAPHIC_LOCATION": "LOCATION",
    # DATE variants
    "DOB": "DATE", "VISIT_DATE": "DATE", "ADMISSION_DATE": "DATE",
    "DATE_OF_BIRTH": "DATE", "DISCHARGE_DATE": "DATE",
    # MRN variants
    "RECORD_NUMBER": "MRN", "MEDICAL_RECORD": "MRN", "CHART_NUMBER": "MRN",
    "MEDICAL_RECORD_NUMBER": "MRN",
    # PHONE variants
    "TELEPHONE": "PHONE", "FAX": "PHONE", "PHONE_NUMBER": "PHONE",
    "FAX_NUMBER": "PHONE",
    # EMAIL variants
    "EMAIL_ADDRESS": "EMAIL",
    # ID variants
    "SSN": "ID", "IDENTIFIER": "ID", "SOCIAL_SECURITY": "ID",
    "ACCOUNT": "ID", "LICENSE": "ID", "SOCIAL_SECURITY_NUMBER": "ID",
    "HEALTH_PLAN": "ID", "DEVICE_ID": "ID", "ACCOUNT_NUMBER": "ID",
    # Direct canonical mappings
    "NAME": "NAME", "LOCATION": "LOCATION", "DATE": "DATE",
    "MRN": "MRN", "PHONE": "PHONE", "EMAIL": "EMAIL", "ID": "ID",
}

# Stopwords for text similarity checks
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "and", "but", "or", "nor",
    "not", "so", "if", "than", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "we", "you", "his", "her", "their", "our", "your",
    "what", "which", "who", "whom", "when", "where", "why", "how",
}


# ---------------------------------------------------------------------------
# Parse failure taxonomy for chain-of-thought output extraction
# ---------------------------------------------------------------------------

class ParseResult(Enum):
    CLEAN = "clean"
    MINOR_FORMAT = "minor_format"
    OUTPUT_ONLY = "output_only"
    SECTION_BLEED = "section_bleed"
    REPETITION_COLLAPSE = "repetition"
    TRUNCATED = "truncated"
    GARBAGE = "garbage"
    EMPTY = "empty"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path=None):
    """Load the ASQ-PHI combined dataset (all_queries.json).

    Returns:
        dict with 'metadata' and 'queries' keys.
    """
    if path is None:
        path = DATA_DIR / "all_queries.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(
        "Loaded dataset: %d queries (%d positive, %d negative, %d PHI elements)",
        data["metadata"]["total_queries"],
        data["metadata"]["positive_queries"],
        data["metadata"]["negative_queries"],
        data["metadata"]["total_phi_elements"],
    )
    return data


def load_prompt_templates(path=None):
    """Load prompt templates config."""
    if path is None:
        path = CONFIGS_DIR / "prompt_templates.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_configs(path=None):
    """Load model configs."""
    if path is None:
        path = CONFIGS_DIR / "model_configs.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_inference_params(path=None):
    """Load inference parameters."""
    if path is None:
        path = CONFIGS_DIR / "inference_params.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt hashing for version control and reproducibility
# ---------------------------------------------------------------------------

def compute_prompt_hash(system_prompt, user_template):
    """Compute SHA-256 hash of a prompt template for versioning."""
    content = system_prompt + "|||" + user_template
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Comprehensive tag detection: 14-pattern detector covering all known SLM output formats
# ---------------------------------------------------------------------------

# Extended type names for regex matching
_EXTENDED_TYPES = (
    "NAME|PATIENT_NAME|PERSON|PATIENT|FIRST_NAME|LAST_NAME|FULL_NAME|"
    "DOCTOR|PHYSICIAN|CLINICIAN|"
    "LOCATION|PLACE|FACILITY|HOSPITAL|ADDRESS|CITY|STATE|COUNTRY|CLINIC|INSTITUTION|"
    "GEOGRAPHIC_LOCATION|"
    "DATE|DOB|VISIT_DATE|ADMISSION_DATE|DATE_OF_BIRTH|DISCHARGE_DATE|"
    "MRN|RECORD_NUMBER|MEDICAL_RECORD|CHART_NUMBER|MEDICAL_RECORD_NUMBER|"
    "PHONE|TELEPHONE|FAX|PHONE_NUMBER|FAX_NUMBER|"
    "EMAIL|EMAIL_ADDRESS|"
    "ID|SSN|IDENTIFIER|SOCIAL_SECURITY|SOCIAL_SECURITY_NUMBER|"
    "ACCOUNT|LICENSE|HEALTH_PLAN|DEVICE_ID|ACCOUNT_NUMBER|"
    "AGE|URL|REDACTED|REMOVED|PHI"
)

# Compiled patterns ordered from most specific to most general
_TAG_PATTERNS = [
    # Pattern 1: Standard bracket tags [TYPE] (case-insensitive)
    re.compile(
        r'\[(?:REDACTED[-_]?)?(' + _EXTENDED_TYPES + r')\]',
        re.IGNORECASE
    ),
    # Pattern 2: Patient-prefixed bracket tags [PATIENT_NAME]
    re.compile(
        r'\[(?:PATIENT[-_])(' + _EXTENDED_TYPES + r')\]',
        re.IGNORECASE
    ),
    # Pattern 3: Angle bracket tags <TYPE>
    re.compile(
        r'<(?:REDACTED[-_]?)?(' + _EXTENDED_TYPES + r')>',
        re.IGNORECASE
    ),
    # Pattern 4: Double angle brackets <<TYPE>>
    re.compile(
        r'<<(' + _EXTENDED_TYPES + r')>>',
        re.IGNORECASE
    ),
    # Pattern 5: Double square brackets [[TYPE]]
    re.compile(
        r'\[\[(' + _EXTENDED_TYPES + r')\]\]',
        re.IGNORECASE
    ),
    # Pattern 6: Generic [REDACTED]
    re.compile(r'\[REDACTED\]', re.IGNORECASE),
    # Pattern 7: Generic [REMOVED]
    re.compile(r'\[REMOVED\]', re.IGNORECASE),
    # Pattern 8: Generic [PHI]
    re.compile(r'\[PHI\]', re.IGNORECASE),
    # Pattern 9: Generic <REDACTED>
    re.compile(r'<REDACTED>', re.IGNORECASE),
    # Pattern 10: Asterisk redaction ***
    re.compile(r'\*{3,}'),
    # Pattern 11: X-sequence redaction XXXX
    re.compile(r'X{4,}'),
    # Pattern 12: Ellipsis bracket [...]
    re.compile(r'\[\.{2,}\]'),
    # Pattern 13: Underscore redaction ____
    re.compile(r'_{4,}'),
    # Pattern 14: Curly brace tags {TYPE}
    re.compile(
        r'\{(' + _EXTENDED_TYPES + r')\}',
        re.IGNORECASE
    ),
]


def find_all_tags(text):
    """Find all redaction tags in text regardless of format.

    Returns:
        list of dicts with keys: tag_text, tag_type, position, format
    """
    results = []
    seen_positions = set()

    # Phase 1: Typed tags (patterns 1-5, 14)
    for i, pattern in enumerate(_TAG_PATTERNS[:5]):
        for m in pattern.finditer(text):
            if m.start() not in seen_positions:
                raw_type = m.group(1) if m.lastindex else "UNKNOWN"
                results.append({
                    "tag_text": m.group(0),
                    "tag_type": normalize_tag_type(raw_type),
                    "position": m.start(),
                    "format": ["bracket", "patient_bracket", "angle",
                               "double_angle", "double_bracket"][i],
                })
                seen_positions.add(m.start())

    # Pattern 14: curly brace
    for m in _TAG_PATTERNS[13].finditer(text):
        if m.start() not in seen_positions:
            raw_type = m.group(1) if m.lastindex else "UNKNOWN"
            results.append({
                "tag_text": m.group(0),
                "tag_type": normalize_tag_type(raw_type),
                "position": m.start(),
                "format": "curly_brace",
            })
            seen_positions.add(m.start())

    # Phase 2: Generic redaction markers (patterns 6-13)
    for i, pattern in enumerate(_TAG_PATTERNS[5:13], start=5):
        for m in pattern.finditer(text):
            if m.start() not in seen_positions:
                fmt_names = [
                    "generic_redacted", "generic_removed", "generic_phi",
                    "angle_redacted", "asterisk", "x_sequence",
                    "ellipsis_bracket", "underscore",
                ]
                results.append({
                    "tag_text": m.group(0),
                    "tag_type": "UNKNOWN",
                    "position": m.start(),
                    "format": fmt_names[i - 5],
                })
                seen_positions.add(m.start())

    results.sort(key=lambda r: r["position"])
    return results


def normalize_tag_type(raw_type):
    """Map variant tag type names to canonical types."""
    upper = raw_type.strip().upper()
    return TAG_VARIANT_MAP.get(upper, upper)


def count_tags_in_text(text):
    """Count total redaction tags in text (any format)."""
    return len(find_all_tags(text))


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_direct_output(raw_output):
    """Parse output for P1, P2, P3: strip prefixes and whitespace."""
    text = raw_output.strip()
    prefixes = [
        "Output:", "Result:", "De-identified:", "Redacted:",
        "Answer:", "De-identified text:", "Here is the de-identified text:",
        "Here's the de-identified text:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text


def parse_phi_json(raw_output):
    """Parse P4 Pass 1 output into list of PHI dicts.

    Returns:
        list of dicts with 'text' and 'type' keys, or None on parse failure.
    """
    raw = raw_output.strip()

    # Handle empty / no-PHI responses
    if raw in ("[]", "", "None", "No PHI found", "No PHI found.",
                "No PHI", "none", "NONE"):
        return []

    # Try parsing as JSON array
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)
                    and "text" in p and "type" in p]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try wrapping in brackets (model may have omitted them)
    try:
        parsed = json.loads("[" + raw + "]")
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)
                    and "text" in p and "type" in p]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to line-by-line parsing
    results = []
    for line in raw.split("\n"):
        line = line.strip().rstrip(",").strip()
        if not line or line in ("[]", "[", "]", "```", "```json"):
            continue
        # Strip markdown code fence markers
        if line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "text" in obj and "type" in obj:
                results.append(obj)
        except (json.JSONDecodeError, TypeError):
            # Try extracting JSON from the line
            json_match = re.search(r'\{[^}]+\}', line)
            if json_match:
                try:
                    obj = json.loads(json_match.group(0))
                    if "text" in obj and "type" in obj:
                        results.append(obj)
                except (json.JSONDecodeError, TypeError):
                    continue

    return results if results else None  # None signals parse failure


def apply_redactions(original_text, phi_list):
    """Programmatic Pass 2: replace identified PHI spans with tags.

    Args:
        original_text: The original query text.
        phi_list: List of dicts with 'text' and 'type' keys.

    Returns:
        De-identified text with PHI replaced by tags.
    """
    result = original_text
    # Sort by length descending to handle overlapping spans correctly
    sorted_phi = sorted(phi_list, key=lambda x: len(x["text"]), reverse=True)
    for phi in sorted_phi:
        tag = f"[{phi['type'].upper()}]"
        result = result.replace(phi["text"], tag)
    return result


def extract_cot_output(raw_output, original_query=""):
    """Extract de-identified text from P5 chain-of-thought output.

    Implements cascading fallback for malformed or truncated CoT responses.

    Returns:
        tuple of (extracted_text, ParseResult)
    """
    if not raw_output or not raw_output.strip():
        return "", ParseResult.EMPTY

    text = raw_output.strip()

    # Level 1: Find OUTPUT: section header
    output_markers = [
        "OUTPUT:", "Output:", "output:",
        "RESULT:", "Result:", "result:",
        "DE-IDENTIFIED TEXT:", "De-identified text:",
        "STEP 3:", "Step 3:",
    ]
    for marker in output_markers:
        idx = text.rfind(marker)
        if idx != -1:
            candidate = text[idx + len(marker):].strip()
            if candidate and _resembles_query(candidate, original_query):
                return candidate, ParseResult.CLEAN
            elif candidate:
                return candidate, ParseResult.MINOR_FORMAT

    # Level 2: No section header; try last paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for p in reversed(paragraphs):
        if _resembles_query(p, original_query):
            return p, ParseResult.OUTPUT_ONLY

    # Level 3: Detect repetition collapse
    if _detect_repetition(text, threshold=3):
        return "", ParseResult.REPETITION_COLLAPSE

    # Level 4: Detect truncation
    words = text.split()
    if len(words) > 200 and not text.rstrip().endswith((".", "?", "!", "]", ">")):
        return "", ParseResult.TRUNCATED

    # Level 5: Garbage detection
    if len(text) > 0:
        non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
        if non_ascii_ratio > 0.1:
            return "", ParseResult.GARBAGE

    # Last resort: return full output flagged as section bleed
    return text, ParseResult.SECTION_BLEED


def _resembles_query(candidate, original):
    """Check if candidate text structurally resembles the original query."""
    if not original:
        # Without original to compare, check if it has redaction tags
        return bool(re.search(r'[\[<](?:' + _EXTENDED_TYPES + r')[\]>]',
                              candidate, re.IGNORECASE))
    orig_tokens = set(original.lower().split()) - STOPWORDS
    cand_tokens = set(candidate.lower().split()) - STOPWORDS
    if not orig_tokens:
        return False
    overlap = len(orig_tokens & cand_tokens) / len(orig_tokens)
    has_tags = bool(find_all_tags(candidate))
    return overlap > 0.4 and (has_tags or overlap > 0.85)


def _detect_repetition(text, threshold=3):
    """Detect repetition collapse in model output."""
    lines = text.strip().split("\n")
    if len(lines) < threshold:
        return False
    # Check for consecutive identical lines
    for i in range(len(lines) - threshold + 1):
        window = lines[i:i + threshold]
        if len(set(window)) == 1 and len(window[0].strip()) > 5:
            return True
    # Check for repeated n-gram phrases
    tokens = text.split()
    for ngram_size in [10, 20, 30]:
        if len(tokens) < ngram_size * 3:
            continue
        ngrams = [
            " ".join(tokens[i:i + ngram_size])
            for i in range(len(tokens) - ngram_size)
        ]
        counts = Counter(ngrams)
        if counts and counts.most_common(1)[0][1] >= threshold:
            return True
    return False


def is_runaway_output(output, input_query):
    """Detect if the model generated irrelevant verbose output."""
    if len(output) > len(input_query) * 3:
        return True
    input_tokens = set(input_query.lower().split())
    output_tokens = set(output.lower().split())
    if input_tokens and len(input_tokens & output_tokens) / len(input_tokens) < 0.3:
        return True
    return False


# ---------------------------------------------------------------------------
# Levenshtein distance and similarity
# ---------------------------------------------------------------------------

def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def levenshtein_similarity(s1, s2):
    """Compute Levenshtein similarity ratio (0.0 to 1.0)."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (BCa method)
# ---------------------------------------------------------------------------

def bootstrap_bca_ci(data, stat_func, n_iterations=10000, alpha=0.05,
                     random_state=42):
    """Compute bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Args:
        data: array-like of observations (resampling units).
        stat_func: callable that takes a resampled array and returns a scalar.
        n_iterations: number of bootstrap iterations.
        alpha: significance level (default 0.05 for 95% CI).
        random_state: random seed for reproducibility.

    Returns:
        tuple of (ci_lower, ci_upper, point_estimate)
    """
    rng = np.random.RandomState(random_state)
    data = np.asarray(data)
    n = len(data)

    # Point estimate
    theta_hat = stat_func(data)

    # Bootstrap distribution
    boot_stats = np.empty(n_iterations)
    for i in range(n_iterations):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = stat_func(sample)

    # Bias correction factor (z0)
    z0 = _norm_ppf(np.mean(boot_stats < theta_hat))

    # Acceleration factor (a) via jackknife
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.concatenate([data[:i], data[i + 1:]])
        jackknife_stats[i] = stat_func(jack_sample)

    jack_mean = np.mean(jackknife_stats)
    num = np.sum((jack_mean - jackknife_stats) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    # BCa quantiles
    z_alpha_lo = _norm_ppf(alpha / 2)
    z_alpha_hi = _norm_ppf(1 - alpha / 2)

    def _bca_quantile(z_alpha):
        numer = z0 + z_alpha
        adjusted = z0 + numer / (1 - a * numer)
        return _norm_cdf(adjusted)

    q_lo = _bca_quantile(z_alpha_lo)
    q_hi = _bca_quantile(z_alpha_hi)

    # Clamp quantiles to valid range
    q_lo = max(0.0, min(1.0, q_lo))
    q_hi = max(0.0, min(1.0, q_hi))

    ci_lower = np.percentile(boot_stats, 100 * q_lo)
    ci_upper = np.percentile(boot_stats, 100 * q_hi)

    return (float(ci_lower), float(ci_upper), float(theta_hat))


def _norm_ppf(p):
    """Inverse CDF of standard normal (using scipy-free approximation)."""
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        t = np.sqrt(-2.0 * np.log(p))
    else:
        t = np.sqrt(-2.0 * np.log(1.0 - p))

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    if p < 0.5:
        return -result
    return result


def _norm_cdf(x):
    """CDF of standard normal (using error function approximation)."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Stratified bootstrap for recall/precision/F1/specificity
# Resamples positive and negative queries separately to preserve class ratio
# ---------------------------------------------------------------------------

def stratified_bootstrap_metrics(positive_queries, negative_queries,
                                 n_iterations=10000, random_state=42):
    """Compute bootstrap CIs for recall, precision, F1, specificity.

    Resamples positive and negative queries separately (stratified bootstrap)
    to preserve the positive/negative ratio per statistics review.

    Args:
        positive_queries: list of dicts, each with 'phi_detected', 'phi_total', 'fp_count'
        negative_queries: list of dicts, each with 'tags_found' (int)
        n_iterations: bootstrap iterations
        random_state: random seed

    Returns:
        dict with keys: recall, precision, f1, specificity; each a dict with
        'mean', 'ci_lower', 'ci_upper'.
    """
    rng = np.random.RandomState(random_state)
    n_pos = len(positive_queries)
    n_neg = len(negative_queries)

    recalls = np.empty(n_iterations)
    precisions = np.empty(n_iterations)
    f1s = np.empty(n_iterations)
    specificities = np.empty(n_iterations)

    for i in range(n_iterations):
        # Resample positive queries with replacement
        pos_idx = rng.randint(0, n_pos, size=n_pos)
        pos_sample = [positive_queries[j] for j in pos_idx]

        # Resample negative queries with replacement
        neg_idx = rng.randint(0, n_neg, size=n_neg)
        neg_sample = [negative_queries[j] for j in neg_idx]

        # Compute metrics from resampled queries
        total_tp = sum(q["phi_detected"] for q in pos_sample)
        total_phi = sum(q["phi_total"] for q in pos_sample)
        total_fp_pos = sum(q.get("fp_count", 0) for q in pos_sample)

        recalls[i] = total_tp / total_phi if total_phi > 0 else 0.0
        precisions[i] = (total_tp / (total_tp + total_fp_pos)
                         if (total_tp + total_fp_pos) > 0 else 0.0)

        if recalls[i] + precisions[i] > 0:
            f1s[i] = 2 * recalls[i] * precisions[i] / (recalls[i] + precisions[i])
        else:
            f1s[i] = 0.0

        preserved = sum(1 for q in neg_sample if q["tags_found"] == 0)
        specificities[i] = preserved / n_neg if n_neg > 0 else 0.0

    results = {}
    for name, arr in [("recall", recalls), ("precision", precisions),
                      ("f1", f1s), ("specificity", specificities)]:
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        results[name] = {
            "mean": float(np.mean(arr)),
            "ci_lower": lo,
            "ci_upper": hi,
        }

    return results


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging for the evaluation harness."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


def utc_timestamp():
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
