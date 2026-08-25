"""RDF knowledge graph construction: entity identity resolution, provenance,
FAIR-compliant triples using standard vocabularies."""

import re
from datetime import datetime, timezone

import pandas as pd
from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DCTERMS, OWL, PROV, SKOS, XSD


def slugify(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")


def strip_leading_article(text):
    """'The Red Sea', 'the Red Sea', and 'Red Sea' all refer to the same
    real-world entity — without this, they'd slugify to different URIs
    (the_red_sea vs red_sea) and fragment into separate graph nodes purely
    because of how one particular sentence happened to phrase it."""
    return re.sub(r"^(the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)


ACRONYM_PATTERN = re.compile(r"^(.*\S)\s*\(([A-Z]{2,10})\)?$")


def resolve_acronym_alias(text):
    """Detects 'Full Descriptive Name (ACRONYM' patterns — with or without
    a closing paren, since spaCy's entity span often truncates right
    before it (e.g. 'The Intergovernmental Panel on Climate Change (IPCC'
    with no trailing ')'). Domain-agnostic: works for any 'X (ACRONYM)'
    pattern, not just IPCC — 'World Health Organization (WHO)', 'Federal
    Reserve (Fed)', etc. Treats the acronym as canonical (shorter forms
    tend to be the dominant, most-repeated mention of an entity) so 'The
    Intergovernmental Panel on Climate Change (IPCC' merges into the same
    graph node as bare 'IPCC' mentions elsewhere, instead of fragmenting.
    Returns (canonical_text, alt_label_or_None)."""
    match = ACRONYM_PATTERN.match(text.strip())
    if match:
        full_name, acronym = match.groups()
        if len(full_name) > len(acronym) + 3:  # sanity: full name should be meaningfully longer
            return acronym, full_name
    return text, None


def build_graph(all_papers, cfg, wikidata_links=None):
    """all_papers: list of dicts with keys paper_id, metadata, mentions."""
    SC = Namespace(cfg["kg"]["namespace"])
    g = Graph()
    g.bind("sc", SC)
    g.bind("dcterms", DCTERMS)
    g.bind("prov", PROV)

    run_time = Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime)
    relation_map = cfg["entity_relation_map"]
    flat_rows = []  # for the human-validation CSV

    for paper in all_papers:
        paper_uri = URIRef(SC[f"paper/{slugify(paper['paper_id'])}"])
        g.add((paper_uri, RDF.type, SC.Paper))
        g.add((paper_uri, PROV.generatedAtTime, run_time))
        g.add((paper_uri, DCTERMS.license, URIRef(cfg["kg"]["license"])))
        if paper["metadata"].get("title"):
            g.add((paper_uri, DCTERMS.title, Literal(paper["metadata"]["title"])))
        if paper["metadata"].get("doi"):
            g.add((paper_uri, DCTERMS.identifier, Literal(paper["metadata"]["doi"])))

        # Track entities per sentence to link co-occurring mentions
        by_sentence = {}
        for m in paper["mentions"]:
            by_sentence.setdefault(m["sent_idx"], []).append(m)

        for sent_idx, mentions_in_sentence in by_sentence.items():
            entity_uris = []
            for m in mentions_in_sentence:
                canonical_text, alt_label = resolve_acronym_alias(m["text"])
                entity_uri = URIRef(SC[f"entity/{slugify(strip_leading_article(canonical_text))}"])
                g.add((entity_uri, RDF.type, SC.Entity))
                if not list(g.objects(entity_uri, RDFS.label)):
                    g.add((entity_uri, RDFS.label, Literal(canonical_text)))
                if alt_label:
                    g.add((entity_uri, SKOS.altLabel, Literal(alt_label)))
                qid = (wikidata_links or {}).get(m["text"]) or (wikidata_links or {}).get(canonical_text)
                if qid:
                    g.add((entity_uri, OWL.sameAs, URIRef(f"http://www.wikidata.org/entity/{qid}")))

                predicate = SC[relation_map.get(m["type"], "mentions")]
                g.add((paper_uri, predicate, entity_uri))

                # Provenance: this specific mention, in this sentence, in this paper
                mention_node = BNode()
                g.add((mention_node, RDF.type, SC.Mention))
                g.add((mention_node, SC.ofEntity, entity_uri))
                g.add((mention_node, PROV.wasDerivedFrom, paper_uri))
                g.add((mention_node, SC.context, Literal(m["sentence"][:300])))

                entity_uris.append(entity_uri)
                flat_rows.append(
                    {
                        "paper_id": paper["paper_id"],
                        "relation": relation_map.get(m["type"], "mentions"),
                        "entity": m["text"],
                        "entity_type": m["type"],
                        "wikidata_qid": qid or "",
                        "sentence": m["sentence"][:200],
                    }
                )

            # Co-occurrence edges between distinct entities in the same sentence
            for i in range(len(entity_uris)):
                for j in range(i + 1, len(entity_uris)):
                    if entity_uris[i] != entity_uris[j]:
                        g.add((entity_uris[i], SC.co_occurs_with, entity_uris[j]))

    return g, pd.DataFrame(flat_rows)
