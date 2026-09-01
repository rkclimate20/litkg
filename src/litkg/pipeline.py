"""Pipeline orchestration: retrieval -> extraction -> NER -> linking -> graph."""

import os

import spacy

from litkg.config import load_config, setup_logging
from litkg.extraction import load_paper_metadata, load_paper_text
from litkg.filters import GENERIC_ACADEMIC_STOPLIST, find_frequent_org_noise
from litkg.graph import build_graph
from litkg.keyphrases import extract_corpus_keyphrases
from litkg.ner import extract_mentions
from litkg.retrieval import (
    derive_keywords_from_query,
    derive_topic_slug,
    find_paper_units,
    retrieve_papers,
)
from litkg.review import flag_for_review, write_manifest
from litkg.wikidata import link_entities_to_wikidata


def run_pipeline(config_path="config.yaml", query_override=None, limit_override=None, skip_wikidata=False):
    """Runs the full pipeline end to end. Returns the output directory
    containing knowledge_graph.ttl, validation_report.csv, and manifest.json.

    This is the same function both the CLI (`litkg run`) and any Python
    code importing litkg directly should call — the CLI is a thin wrapper
    around this, not a separate code path.

    skip_wikidata: if True, skips entity linking entirely — extraction
    (retrieval, NER, graph-building) is fast and mostly local, while
    Wikidata linking is the slowest, most network-failure-prone step.
    Run `litkg link --topic ...` afterward to add links without redoing
    extraction.
    """
    cfg = load_config(config_path)
    log = setup_logging(cfg)

    if query_override is not None:
        cfg["retrieval"]["query"] = query_override
        log.info("Query overridden: %s", query_override)
    if limit_override is not None:
        cfg["retrieval"]["limit"] = limit_override
        log.info("Limit overridden: %d", limit_override)

    # Auto-namespace corpus and kg_output by topic, derived from the query's
    # quoted phrases — each distinct topic gets its own subfolder, so
    # switching queries can never silently mix papers from different topics
    # into the same folder. output_dir/kg.output_dir in config act as the
    # ROOT under which per-topic subfolders are created.
    topic_slug = derive_topic_slug(cfg["retrieval"]["query"])
    cfg["retrieval"]["output_dir"] = os.path.join(cfg["retrieval"]["output_dir"], topic_slug)
    cfg["kg"]["output_dir"] = os.path.join(cfg["kg"]["output_dir"], topic_slug)
    log.info(
        "Topic slug '%s' — using corpus dir '%s', kg_output dir '%s'.",
        topic_slug, cfg["retrieval"]["output_dir"], cfg["kg"]["output_dir"],
    )
    os.makedirs(cfg["kg"]["output_dir"], exist_ok=True)

    corpus_dir = retrieve_papers(cfg, log)
    paper_units = find_paper_units(corpus_dir, log)

    nlp = spacy.load("en_core_web_sm")
    query_keywords = derive_keywords_from_query(cfg["retrieval"]["query"])
    extra_keywords = cfg.get("keyword_dictionary", [])
    keyword_terms = list(dict.fromkeys(query_keywords + extra_keywords))
    log.info("Keyword dictionary (auto + config): %s", keyword_terms)

    structural_stoplist = GENERIC_ACADEMIC_STOPLIST + cfg.get("structural_stoplist", [])
    relation_map = cfg["entity_relation_map"]

    all_papers = []
    papers_with_text = []
    for paper_dir in paper_units:
        text = load_paper_text(paper_dir)
        if not text:
            log.warning("No text found in %s — skipping.", paper_dir)
            continue
        metadata = load_paper_metadata(paper_dir)
        paper_id = metadata.get("doi") or os.path.basename(paper_dir)
        log.info("Extracted %d characters of text from %s", len(text), paper_dir)

        papers_with_text.append({"paper_id": paper_id, "text": text})

        mentions = extract_mentions(
            text, nlp, keyword_terms, relation_map,
            structural_stoplist=structural_stoplist,
        )
        if len(mentions) == 0:
            log.warning(
                "ZERO mentions extracted from %s despite %d chars of text — "
                "check whether JATS structure differs for this paper.",
                paper_dir, len(text),
            )
        all_papers.append(
            {
                "paper_id": paper_id,
                "metadata": metadata,
                "mentions": mentions,
            }
        )
        log.info("Processed %s — %d mentions.", paper_dir, len(mentions))

    freq_noise = find_frequent_org_noise(all_papers)
    if freq_noise:
        log.info(
            "Auto-detected %d frequently-recurring ORG terms — flagging for "
            "review rather than deleting: %s",
            len(freq_noise), sorted(freq_noise),
        )

    unique_entities = sorted({
        m["text"] for paper in all_papers for m in paper["mentions"]
        if m["type"] in ("GPE", "ORG", "LOC")
    })
    wikidata_cache_path = cfg.get("kg", {}).get("wikidata_cache", "./wikidata_cache.json")

    if skip_wikidata:
        log.info(
            "Skipping Wikidata linking (--skip-wikidata). %d entities left "
            "unlinked — run 'litkg link --topic %s' later to add links "
            "without redoing extraction.",
            len(unique_entities), topic_slug,
        )
        wikidata_links = {}
    else:
        wikidata_links = link_entities_to_wikidata(unique_entities, log, cache_path=wikidata_cache_path)
        log.info(
            "Entity linking done: %d of %d entities matched to a Wikidata QID.",
            sum(1 for v in wikidata_links.values() if v), len(unique_entities),
        )

    graph, flat_df = build_graph(all_papers, cfg, wikidata_links=wikidata_links)
    flat_df = flag_for_review(flat_df, log, freq_noise=freq_noise)

    ttl_path = os.path.join(cfg["kg"]["output_dir"], "knowledge_graph.ttl")
    graph.serialize(destination=ttl_path, format="turtle")
    log.info("Wrote RDF graph: %s (%d triples)", ttl_path, len(graph))

    csv_path = os.path.join(cfg["kg"]["output_dir"], "validation_report.csv")
    flat_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("Wrote validation CSV: %s", csv_path)

    keyphrase_path = extract_corpus_keyphrases(papers_with_text, cfg["kg"]["output_dir"], log=log)
    if keyphrase_path:
        log.info("Keyphrases available at: %s", keyphrase_path)

    manifest_path = write_manifest(
        cfg, config_path, len(all_papers), len(graph), cfg["kg"]["output_dir"]
    )
    log.info("Wrote manifest: %s", manifest_path)

    log.info(
        "Done. %d papers processed, %d RDF triples, %d rows need human review.",
        len(all_papers), len(graph),
        int(flat_df["needs_review"].sum()) if not flat_df.empty else 0,
    )

    return cfg["kg"]["output_dir"]
