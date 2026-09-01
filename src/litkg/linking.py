"""Standalone Wikidata linking: separated from extraction so a slow,
network-dependent step can be run (and re-run) independently of the fast,
local extraction step. Updates an already-built graph and CSV in place
rather than requiring a full pipeline re-run."""

import os

import pandas as pd
from rdflib import Graph, Namespace, OWL, RDFS, URIRef

from litkg.wikidata import link_entities_to_wikidata


def link_topic(topic_dir, namespace, log, cache_path="wikidata_cache.json"):
    """Reads validation_report.csv from topic_dir, finds GPE/ORG/LOC
    entities that don't yet have a Wikidata QID, links them, then updates
    BOTH the .ttl graph and the CSV in place with any new links found —
    so extraction and linking stay consistent with each other even when
    run as separate steps at different times.

    Returns the number of newly-linked entities."""
    csv_path = os.path.join(topic_dir, "validation_report.csv")
    ttl_path = os.path.join(topic_dir, "knowledge_graph.ttl")

    if not os.path.exists(csv_path) or not os.path.exists(ttl_path):
        raise FileNotFoundError(
            f"Could not find validation_report.csv and knowledge_graph.ttl "
            f"in {topic_dir} — run 'litkg run' first to produce them."
        )

    df = pd.read_csv(csv_path, keep_default_na=False)
    linkable = df[df["entity_type"].isin(["GPE", "ORG", "LOC"])]
    unique_entities = sorted(linkable["entity"].unique())

    log.info("Linking %d unique GPE/ORG/LOC entities for this topic.", len(unique_entities))
    wikidata_links = link_entities_to_wikidata(unique_entities, log, cache_path=cache_path)

    n_new = sum(1 for v in wikidata_links.values() if v)
    log.info(
        "Entity linking done: %d of %d entities matched to a Wikidata QID.",
        n_new, len(unique_entities),
    )

    # Update the CSV's wikidata_qid column for linkable rows
    def _lookup(row):
        if row["entity_type"] in ("GPE", "ORG", "LOC"):
            qid = wikidata_links.get(row["entity"])
            return qid or row.get("wikidata_qid", "")
        return row.get("wikidata_qid", "")

    df["wikidata_qid"] = df.apply(_lookup, axis=1)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("Updated %s with new Wikidata links.", csv_path)

    # Update the graph: add owl:sameAs for any entity whose label now has a
    # resolved QID and doesn't already carry a sameAs link.
    SC = Namespace(namespace)
    g = Graph()
    g.parse(ttl_path, format="turtle")

    added = 0
    for s, p, label in list(g.triples((None, RDFS.label, None))):
        text = str(label)
        qid = wikidata_links.get(text)
        if qid and (s, OWL.sameAs, None) not in g:
            g.add((s, OWL.sameAs, URIRef(f"http://www.wikidata.org/entity/{qid}")))
            added += 1

    g.serialize(destination=ttl_path, format="turtle")
    log.info("Added %d new owl:sameAs links to %s.", added, ttl_path)

    return n_new
