"""Automatic keyphrase extraction (YAKE): discovers important recurring
phrases directly from the corpus text, independent of the query's own
keyword terms. This is genuine unsupervised keyphrase extraction, distinct
from the KEYWORD entity type elsewhere in litkg — that type is just a
substring match against phrases YOU put in the query; this discovers new
candidate phrases statistically from the text itself."""

import pandas as pd
import yake


def extract_keyphrases(text, top_n=15, max_ngram_size=3, language="en"):
    """Returns [(phrase, score), ...] for one paper's text.

    Note YAKE's score convention: LOWER score = more relevant/important —
    the opposite of most ranking scores (like Wikidata's match score
    elsewhere in this package). This is YAKE's own convention, not a bug
    here; keep it in mind when reading keyphrases.csv."""
    if not text or not text.strip():
        return []
    try:
        kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=top_n)
        return kw_extractor.extract_keywords(text)
    except Exception:
        # A keyphrase extraction hiccup on one paper shouldn't crash the
        # whole pipeline run — keyphrases are a bonus output, not required
        # for the graph itself to be usable.
        return []


def aggregate_keyphrases(per_paper_keyphrases):
    """per_paper_keyphrases: list of {"paper_id": ..., "keyphrases": [(phrase, score), ...]}

    Returns a DataFrame ranked by how many distinct papers each phrase
    appears in (doc_frequency) first, then by average YAKE score (lower is
    better) as a tiebreaker — a phrase appearing across many papers is more
    likely to represent a genuine corpus-level theme than one paper's
    single best-scored phrase."""
    rows = []
    for entry in per_paper_keyphrases:
        for phrase, score in entry["keyphrases"]:
            rows.append({
                "paper_id": entry["paper_id"],
                "phrase": phrase.strip().lower(),
                "score": score,
            })
    if not rows:
        return pd.DataFrame(columns=["phrase", "doc_frequency", "avg_score"])

    df = pd.DataFrame(rows)
    agg = df.groupby("phrase").agg(
        doc_frequency=("paper_id", "nunique"),
        avg_score=("score", "mean"),
    ).reset_index()
    agg = agg.sort_values(["doc_frequency", "avg_score"], ascending=[False, True])
    return agg.reset_index(drop=True)
