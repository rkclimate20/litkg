"""Querying the knowledge graph for WP6-style answers: find an entity,
pull every paper that mentions it, deduplicate excerpts, group by paper."""

import glob
import os
import re
import sys

import yaml
from rdflib import Graph, Namespace, RDF, RDFS
from rdflib.namespace import DCTERMS, OWL, PROV


def resolve_ttl_path(topic, base_dir="kg_output"):
    """If topic is given, use that folder explicitly (raising with a
    helpful list of available topics on a typo). Otherwise falls back to
    the most recently modified topic folder."""
    if topic:
        path = os.path.join(base_dir, topic, "knowledge_graph.ttl")
        if os.path.exists(path):
            return path
        available = [
            os.path.basename(os.path.dirname(p))
            for p in glob.glob(os.path.join(base_dir, "*", "knowledge_graph.ttl"))
        ]
        raise FileNotFoundError(
            f"No topic folder '{base_dir}/{topic}/' found. "
            f"Available topics: {available}" if available
            else f"No topic folder '{base_dir}/{topic}/' found and no topics under {base_dir}/."
        )

    topic_dirs = sorted(
        glob.glob(os.path.join(base_dir, "*", "knowledge_graph.ttl")),
        key=os.path.getmtime, reverse=True,
    )
    if topic_dirs:
        return topic_dirs[0]
    flat = os.path.join(base_dir, "knowledge_graph.ttl")
    if os.path.exists(flat):
        return flat
    raise FileNotFoundError(f"Could not find knowledge_graph.ttl under {base_dir}/.")


def load_namespace(config_path="config.yaml"):
    """Reads kg.namespace from config.yaml instead of hardcoding it."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("kg", {}).get("namespace", "https://example.org/semanticclimate/kg/")
    except (FileNotFoundError, yaml.YAMLError):
        return "https://example.org/semanticclimate/kg/"


def _normalize_for_dedup(sentence):
    """Two sentences that differ only in leading capitalization or
    whitespace are the same excerpt for display purposes."""
    return re.sub(r"\s+", " ", sentence.strip().lower())


def build_answer(entity_search_text, ttl_path, namespace):
    """Returns a list of entity result dicts: entity_label, entity_uri,
    wikidata_qid, paper_count, and papers (each with title, doi, and
    deduplicated excerpt sentences) — the structured, no-synthesis WP6
    answer format."""
    SC = Namespace(namespace)
    g = Graph()
    g.parse(ttl_path, format="turtle")

    raw_matches = [
        (s, str(o)) for s, p, o in g.triples((None, RDFS.label, None))
        if entity_search_text.lower() in str(o).lower()
    ]
    # Defensive dedup by entity_uri: a graph can have the same entity_uri
    # carrying multiple label variants — collapse into one result, keeping
    # the cleanest-looking label (letters/spaces/standard punctuation only).
    by_uri = {}
    for entity_uri, label in raw_matches:
        if entity_uri not in by_uri:
            by_uri[entity_uri] = label
            continue
        current = by_uri[entity_uri]
        current_clean = bool(re.fullmatch(r"[A-Za-z0-9\s\-'.,()]+", current))
        candidate_clean = bool(re.fullmatch(r"[A-Za-z0-9\s\-'.,()]+", label))
        if candidate_clean and not current_clean:
            by_uri[entity_uri] = label
        elif candidate_clean == current_clean and len(label) > len(current):
            by_uri[entity_uri] = label
    matches = list(by_uri.items())

    results = []
    for entity_uri, label in matches:
        qid_matches = list(g.objects(entity_uri, OWL.sameAs))
        wikidata_qid = str(qid_matches[0]) if qid_matches else None

        papers_mentioning = set()
        for s, p, o in g.triples((None, None, entity_uri)):
            if str(p).startswith(str(SC)) and (s, RDF.type, SC.Paper) in g:
                papers_mentioning.add(s)

        mention_nodes = list(g.subjects(SC.ofEntity, entity_uri))

        papers_out = []
        for paper_uri in papers_mentioning:
            title = g.value(paper_uri, DCTERMS.title)
            doi = g.value(paper_uri, DCTERMS.identifier)

            seen_normalized = set()
            excerpts = []
            for mention_node in mention_nodes:
                derived_from = g.value(mention_node, PROV.wasDerivedFrom)
                if derived_from != paper_uri:
                    continue
                context = g.value(mention_node, SC.context)
                if not context:
                    continue
                norm = _normalize_for_dedup(str(context))
                if norm in seen_normalized:
                    continue
                seen_normalized.add(norm)
                excerpts.append(str(context))

            papers_out.append({
                "title": str(title) if title else None,
                "doi": str(doi) if doi else None,
                "excerpts": excerpts,
            })

        papers_out.sort(key=lambda p: len(p["excerpts"]), reverse=True)

        results.append({
            "entity_label": label,
            "entity_uri": str(entity_uri),
            "wikidata_qid": wikidata_qid,
            "paper_count": len(papers_out),
            "papers": papers_out,
        })

    return results


def print_readable(results, search_term):
    if not results:
        print(f"No entity found matching '{search_term}'.")
        return
    for r in results:
        print("=" * 70)
        print(f"TOPIC: {r['entity_label']}")
        print(f"Wikidata: {r['wikidata_qid'] or '(not linked)'}")
        print(f"Mentioned in {r['paper_count']} paper(s)\n")
        for p in r["papers"]:
            print(f"  {p['title'] or '(untitled)'}")
            if p["doi"]:
                print(f"  {p['doi']}")
            for ex in p["excerpts"]:
                print(f'    "{ex}"')
            print()
