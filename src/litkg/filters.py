"""Entity noise filtering: gazetteer correction, plausibility checks,
structural noise, and automatic frequency-based noise detection."""

import re
from collections import defaultdict

import pycountry

_COUNTRY_NAMES = {c.name for c in pycountry.countries}
_COUNTRY_NAMES |= {c.common_name for c in pycountry.countries if hasattr(c, "common_name")}

_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# Ships with the package, not tied to any query — about academic paper
# MECHANICS (tables, common stats tests, common data repositories), not
# paper CONTENT, so it's genuinely topic-independent. Extend this list as
# new cross-domain mechanics-noise is spotted; it's not meant to hold
# domain-specific terms — those are handled by find_frequent_org_noise instead.
GENERIC_ACADEMIC_STOPLIST = [
    "Table", "Tables", "Fig", "Figs", "Figure", "Figures",
    "Appendix", "Appendices", "Supplementary",
    "Wilcoxon", "ANOVA", "Kruskal-Wallis", "Mann-Whitney", "Tukey",
    "Chi-square", "t-test", "MANOVA",
    "Dryad", "GitHub", "Zenodo", "Figshare", "OSF",
]


def reclassify_geo(text, label):
    """Correct type mislabels using a country gazetteer, e.g. 'Maldives'
    tagged ORG should route as GPE like any other place name."""
    if text.strip() in _COUNTRY_NAMES and label != "GPE":
        return "GPE"
    return label


def is_structural_noise(text, stoplist):
    """Filters recurring document-structure artifacts (Table/Figure
    references, generic method-section headings) that repeat many times per
    paper and aren't meaningful real-world entities."""
    t = text.strip().lower()
    return any(t == s.lower() or t.startswith(s.lower() + " ") for s in stoplist)


def is_plausible_date(text):
    """Real dates almost always contain a digit (year) or a month name.
    Distinguishes genuine years from measurements/decimals/page numbers
    that also happen to contain digits."""
    t = text.strip()
    if re.search(r"[˚°]|(?:°?[CF]\b)", t):
        return False
    if re.fullmatch(r"\d*\.\d+", t):  # decimal, e.g. "0.67", "10.14"
        return False
    if re.fullmatch(r"(19|20)\d{2}", t):  # clean 4-digit year
        return True
    if re.fullmatch(r"\d+", t):  # bare integer that isn't a 4-digit year
        return False
    if re.search(r"\d", t):
        return True
    words = re.findall(r"[a-zA-Z]+", t.lower())
    return any(w in _MONTH_WORDS for w in words)


def is_plausible_proper_noun(text):
    """Genuine ORG/GPE/LOC names are virtually always capitalized somewhere.
    'reef fishes', 'macroalgae' etc. are lowercase common nouns misfired by
    the general-purpose model."""
    return any(c.isupper() for c in text)


def is_likely_formula_fragment(text):
    """Chemical formulas (CO2, CaCl2, CaCO3) often get their subscript
    numbers split off as separate tokens with a space during text
    extraction — 'CO 2', 'CaCl 2' — which spaCy then misreads as an
    organization name. Generic across any chemistry-adjacent paper, not
    tied to a specific query/topic."""
    t = text.strip()
    return bool(re.search(r"\b[A-Za-z]{1,4}\s+\d\b", t)) and len(t) < 30


def is_noise(entity_text):
    """Drops obviously-bad entities that survive NER: pure numbers (PMC/PMID
    IDs, page numbers), citation fragments, and near-empty strings."""
    t = entity_text.strip()
    if len(t) < 3:
        return True
    if re.fullmatch(r"[\d\s]+", t):
        return True
    if re.search(r"\bet al\b\.?", t, re.IGNORECASE):
        return True
    if t.lower() in ("al.", "al"):
        return True
    return False


def find_frequent_org_noise(all_papers, threshold=0.2, min_papers=5):
    """Automatically detects ORG entities that recur across a suspiciously
    high fraction of papers in THIS run's corpus — computed fresh per run,
    so it adapts to whatever query/topic is currently configured without
    manual stoplist editing. Scoped to ORG only: real domain entities
    (species names, locations) legitimately recur often because they're
    central to the topic, but genuine organizations are rarely tagged ORG
    this consistently — recurring ORG hits are usually stats tests, data
    repositories, or table/heading artifacts instead.
    Skipped entirely below min_papers, since frequency is meaningless on a
    tiny pilot batch (a term in the only paper is trivially "in 100%")."""
    n_papers = len(all_papers)
    if n_papers < min_papers:
        return set()

    entity_paper_sets = defaultdict(set)
    for paper in all_papers:
        seen_in_paper = set()
        for m in paper["mentions"]:
            if m["type"] == "ORG":
                key = m["text"].strip().lower()
                if key not in seen_in_paper:
                    entity_paper_sets[key].add(paper["paper_id"])
                    seen_in_paper.add(key)

    return {
        key for key, papers in entity_paper_sets.items()
        if len(papers) / n_papers >= threshold
    }
