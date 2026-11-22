"""
Purpose: Provides clinical dictionary utilities for prompt suffixes, coverage scoring, and audits.
Inputs: Generated model answers, question strings, and cluster group identifiers.
Outputs: Audit dictionaries, coverage metrics, safe boundary violations, and prompt suffix strings.
"""
import json, re
from pathlib import Path

_DICT_PATH = Path(__file__).resolve().parents[2] / "configs" / "cluster_dictionaries.json"
with open(_DICT_PATH, encoding="utf-8") as f:
    CLUSTER_DICTS = json.load(f)

# ── 1. Prompt suffix builder ──────────────────────────────────────────────────
def build_prompt_suffix(group_id: str, subset: str) -> str:
    """
    Returns a prompt suffix string for the given cluster group
    calibrated to the answer length register of the subset.
    Terse register: Amh_Eth, Eng_Eth
    Verbose register: Eng_Uga, Aka_Gha, Lug_Uga, Swa_Ken, Eng_Gha, Eng_Ken
    """
    TERSE_SUBSETS = {"Amh_Eth", "Eng_Eth"}
    d = CLUSTER_DICTS.get(group_id, {})
    must_haves = d.get("must_have_terms", [])
    flags = d.get("cross_cutting_flags", [])
    safe_boundaries = d.get("must_not_claim", [])
    is_terse = subset in TERSE_SUBSETS

    key = "terse_form" if is_terse else "verbose_form"
    terms = [t[key] for t in must_haves if t.get("clinical_weight", 1) >= 2]

    suffix_parts = []
    if terms:
        suffix_parts.append("Your answer must include: " + "; ".join(terms[:4]) + ".")
    if safe_boundaries:
        never = safe_boundaries[0]["safe_alternative"]
        suffix_parts.append(f"Important: {never}")
    if flags:
        for flag in flags[:1]:
            suffix_parts.append(flag["implementation"].split(".")[0] + ".")

    return " ".join(suffix_parts)

# ── 2. Coverage scorer (local ROUGE proxy) ────────────────────────────────────
def coverage_score(generated_answer: str, group_id: str) -> dict:
    """
    Scores a generated answer against must-have terms.
    Returns: {score: float, hits: list, misses: list, weighted_score: float}
    """
    d = CLUSTER_DICTS.get(group_id, {})
    must_haves = d.get("must_have_terms", [])
    text = generated_answer.lower()
    hits, misses = [], []
    total_weight = sum(t.get("clinical_weight", 1) for t in must_haves)
    hit_weight = 0

    for t in must_haves:
        term = t["term"].lower()
        # check term or any word from terse/verbose form
        term_words = set(re.findall(r"\b\w+\b", term))
        terse_words = set(re.findall(r"\b\w+\b", t.get("terse_form","").lower()))
        check_words = term_words | terse_words
        found = any(w in text for w in check_words if len(w) > 3)
        if found:
            hits.append(term)
            hit_weight += t.get("clinical_weight", 1)
        else:
            misses.append(term)

    return {
        "score": len(hits) / len(must_haves) if must_haves else 0.0,
        "weighted_score": hit_weight / total_weight if total_weight else 0.0,
        "hits": hits,
        "misses": misses,
        "n_terms": len(must_haves),
    }

# ── 3. Diverge term detector ──────────────────────────────────────────────────
def detect_diverge_terms(text: str, group_id: str) -> list:
    """
    Returns list of diverge terms found in text for this group.
    Use to flag generated answers before submission.
    """
    d = CLUSTER_DICTS.get(group_id, {})
    diverge = d.get("diverge_terms", [])
    found = []
    text_lower = text.lower()
    for t in diverge:
        if t["term"].lower() in text_lower:
            found.append({
                "term": t["term"],
                "register": t["register_triggered"],
                "use_instead": t["safer_alternative"],
            })
    return found

# ── 4. Myth framing detector ──────────────────────────────────────────────────
MYTH_SIGNALS = [
    "prayer", "pray", "praying", "fasting", "fast",
    "holy water", "traditional healer", "healer", "pastor",
    "church", "mosque", "faith healing", "herbs", "herbal",
    "miracle", "spiritual", "god will", "allah will",
]

def is_myth_framing(question: str) -> bool:
    """
    Returns True if the question contains cultural/religious cure framing.
    Routes to G13_CULTURAL_MYTHS suffix regardless of disease group.
    """
    q = question.lower()
    return any(signal in q for signal in MYTH_SIGNALS)

# ── 5. Must-not-claim filter ──────────────────────────────────────────────────
def check_safe_boundaries(generated_answer: str, group_id: str) -> list:
    """
    Returns list of potential safe boundary violations found in generated answer.
    These are heuristic checks — not exhaustive.
    """
    d = CLUSTER_DICTS.get(group_id, {})
    violations = []
    text = generated_answer.lower()

    # Universal check: dosage claims
    dosage_pattern = r"\b\d+\s*(mg|ml|g|mcg|iu|units?)\b"
    if re.search(dosage_pattern, text):
        violations.append("Possible specific dosage claim detected — remove or replace with clinical referral")

    # Universal check: cure claim for incurable conditions
    incurable = {"G02_HIV_GENERAL", "G04_HERPES_HSV", "G03_HIV_TRANSMISSION"}
    if group_id in incurable and "cured" in text and "cannot be cured" not in text:
        violations.append("Possible cure claim for condition that cannot be cured — verify phrasing")

    # Group-specific checks
    mnc = d.get("must_not_claim", [])
    for item in mnc:
        forbidden_words = set(re.findall(r"\b\w+\b", item["forbidden_claim"].lower()))
        if len(forbidden_words) >= 3:
            if sum(1 for w in forbidden_words if w in text and len(w) > 4) >= 2:
                violations.append(f"Possible violation: {item['forbidden_claim'][:80]}")

    return violations

# ── 6. Anchor quality pre-filter ─────────────────────────────────────────────
def anchor_quality_score(answer: str, group_id: str) -> str:
    """
    Fast pre-filter for RAG anchor quality.
    Returns: GOOD / WEAK / POISONED
    """
    cov = coverage_score(answer, group_id)
    violations = check_safe_boundaries(answer, group_id)
    diverge = detect_diverge_terms(answer, group_id)

    if violations:
        return "POISONED"
    if cov["weighted_score"] < 0.35:
        return "POISONED"
    if cov["weighted_score"] < 0.60 or len(diverge) > 1:
        return "WEAK"
    return "GOOD"

# ── 7. Full answer audit ──────────────────────────────────────────────────────
def audit_answer(question: str, answer: str, group_id: str, subset: str) -> dict:
    """
    Full audit of a generated answer. Use in your evaluation loop.
    """
    myth = is_myth_framing(question)
    effective_group = "G13_CULTURAL_MYTHS" if myth else group_id
    cov = coverage_score(answer, effective_group)
    violations = check_safe_boundaries(answer, effective_group)
    diverge = detect_diverge_terms(answer, effective_group)
    quality = anchor_quality_score(answer, effective_group)

    return {
        "group_id": group_id,
        "effective_group": effective_group,
        "is_myth_framing": myth,
        "subset": subset,
        "coverage_score": round(cov["score"], 3),
        "weighted_coverage": round(cov["weighted_score"], 3),
        "missing_terms": cov["misses"],
        "safe_boundary_violations": violations,
        "diverge_terms_found": [d["term"] for d in diverge],
        "anchor_quality": quality,
    }

if __name__ == "__main__":
    # Quick smoke test
    test_q = "Can prayer cure HIV?"
    test_a = "Prayer cannot cure HIV. HIV is a virus that requires antiretroviral treatment (ART). Please visit a health facility for testing and treatment. Many people combine their faith with medical care."
    result = audit_answer(test_q, test_a, "G02_HIV_GENERAL", "Eng_Uga")
    print(json.dumps(result, indent=2))
