"""Named entity extraction: spaCy NER + keyword matching + noise filters."""

from litkg.filters import (
    is_likely_formula_fragment,
    is_noise,
    is_plausible_date,
    is_plausible_proper_noun,
    is_structural_noise,
    reclassify_geo,
)


def extract_mentions(text, nlp, keyword_terms, entity_relation_map, structural_stoplist=None):
    """Returns a list of mentions, each tagged with its containing sentence
    index so co-occurring entities can be linked automatically."""
    structural_stoplist = structural_stoplist or []
    doc = nlp(text)
    mentions = []

    for sent_idx, sent in enumerate(doc.sents):
        for ent in sent.ents:
            label = reclassify_geo(ent.text.strip(), ent.label_)
            if label not in entity_relation_map or is_noise(ent.text):
                continue
            if is_structural_noise(ent.text, structural_stoplist):
                continue
            if label in ("GPE", "ORG", "LOC") and not is_plausible_proper_noun(ent.text):
                continue
            if label == "ORG" and is_likely_formula_fragment(ent.text):
                continue
            if label == "DATE" and not is_plausible_date(ent.text):
                continue
            mentions.append(
                {
                    "text": ent.text.strip(),
                    "type": label,
                    "sent_idx": sent_idx,
                    "sentence": sent.text.strip(),
                }
            )
        sent_lower = sent.text.lower()
        for term in keyword_terms:
            if term.lower() in sent_lower:
                mentions.append(
                    {
                        "text": term,
                        "type": "KEYWORD",
                        "sent_idx": sent_idx,
                        "sentence": sent.text.strip(),
                    }
                )
    return mentions
