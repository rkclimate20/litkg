"""Command-line interface: litkg run / litkg query / litkg analyze."""

import argparse
import glob
import json
import os
import sys

import pandas as pd

from litkg.query import build_answer, print_readable, load_namespace, resolve_ttl_path


def _cmd_run(args):
    from litkg.pipeline import run_pipeline  # lazy: only 'run' needs spacy/pygetpapers
    run_pipeline(
        config_path=args.config,
        query_override=args.query,
        limit_override=args.limit,
        skip_wikidata=args.skip_wikidata,
    )


def _cmd_link(args):
    """Standalone Wikidata linking on an already-extracted topic — doesn't
    need spacy/pygetpapers, only rdflib/pandas/requests, and can be run
    independently of (and repeatedly after) 'litkg run --skip-wikidata'."""
    from litkg.linking import link_topic
    from litkg.query import load_namespace
    from litkg.config import load_config, setup_logging

    if args.topic:
        topic_dir = os.path.join("kg_output", args.topic)
    else:
        candidates = sorted(
            glob.glob("kg_output/*/validation_report.csv"),
            key=os.path.getmtime, reverse=True,
        )
        if not candidates:
            print("No topic found under kg_output/. Run 'litkg run' first.")
            sys.exit(1)
        topic_dir = os.path.dirname(candidates[0])
        print(f"Using most recently modified topic folder: {topic_dir}")

    cfg = load_config(args.config)
    log = setup_logging(cfg)
    namespace = load_namespace(args.config)
    cache_path = cfg.get("kg", {}).get("wikidata_cache", "./wikidata_cache.json")

    try:
        n_new = link_topic(topic_dir, namespace, log, cache_path=cache_path)
        print(f"\nDone. {n_new} entities newly linked to Wikidata.")
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)


def _cmd_query(args):
    try:
        ttl_path = resolve_ttl_path(args.topic)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    namespace = load_namespace(args.config)
    print(f"Loading {ttl_path} ...")
    results = build_answer(args.entity, ttl_path, namespace)
    print()
    print_readable(results, args.entity)

    if results:
        topic_dir = os.path.dirname(ttl_path)
        answers_dir = os.path.join(topic_dir, "answers")
        os.makedirs(answers_dir, exist_ok=True)
        import re
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.entity.strip().lower()).strip("_")
        out_path = os.path.join(answers_dir, f"{slug}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote structured answer data to {out_path}")


def _cmd_analyze(args):
    if args.topic:
        kg_dir = os.path.join("kg_output", args.topic)
        if not os.path.exists(os.path.join(kg_dir, "validation_report.csv")):
            available = [
                os.path.basename(os.path.dirname(p))
                for p in glob.glob("kg_output/*/validation_report.csv")
            ]
            print(f"No topic folder 'kg_output/{args.topic}/' found.")
            print(f"Available topics: {available}" if available else "No topics found under kg_output/.")
            sys.exit(1)
    else:
        topic_dirs = sorted(
            glob.glob("kg_output/*/validation_report.csv"),
            key=os.path.getmtime, reverse=True,
        )
        if topic_dirs:
            kg_dir = os.path.dirname(topic_dirs[0])
            print(f"Using most recently modified topic folder: {kg_dir}")
        else:
            kg_dir = "kg_output"

    csv_path = os.path.join(kg_dir, "validation_report.csv")
    log_path = "pipeline.log"
    ttl_path = os.path.join(kg_dir, "knowledge_graph.ttl")

    print("=" * 60)
    print("1. LOG WARNINGS / ERRORS")
    print("=" * 60)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
        problems = [l.strip() for l in log_lines if "WARNING" in l or "ERROR" in l]
        print("\n".join(problems) if problems else "No warnings or errors found.")
    else:
        print(f"Could not find {log_path}.")

    df = pd.read_csv(csv_path, keep_default_na=False)

    print("\n" + "=" * 60)
    print("2. MENTIONS PER PAPER")
    print("=" * 60)
    print(df.groupby("paper_id").size().sort_values().to_string())

    print("\n" + "=" * 60)
    print("3. NEEDS_REVIEW RATE BY ENTITY TYPE")
    print("=" * 60)
    summary = df.groupby("entity_type")["needs_review"].agg(["count", "sum"])
    summary["review_rate_%"] = (summary["sum"] / summary["count"] * 100).round(1)
    summary.columns = ["total", "flagged_for_review", "review_rate_%"]
    print(summary.to_string())

    if "wikidata_qid" in df.columns:
        print("\n" + "=" * 60)
        print("3b. WIKIDATA LINKING RATE BY ENTITY TYPE")
        print("=" * 60)
        linkable = df[df["entity_type"].isin(["GPE", "ORG", "LOC"])]
        link_summary = linkable.groupby("entity_type").apply(
            lambda g: pd.Series({
                "total": len(g),
                "linked": (g["wikidata_qid"] != "").sum(),
                "linked_%": round((g["wikidata_qid"] != "").mean() * 100, 1),
            }),
            include_groups=False,
        )
        print(link_summary.to_string())

    print("\n" + "=" * 60)
    print(f"4. RANDOM SAMPLE OF {min(5, df['paper_id'].nunique())} PAPERS")
    print("=" * 60)
    sample_papers = pd.Series(df["paper_id"].unique()).sample(
        min(5, df["paper_id"].nunique()), random_state=42
    )
    for pid in sample_papers:
        print(f"\n--- {pid} ---")
        rows = df[df["paper_id"] == pid][["relation", "entity", "entity_type", "needs_review"]]
        print(rows.to_string(index=False))


def _cmd_keyphrases(args):
    """Views keyphrases.csv from an existing run — pure viewer, doesn't
    need yake/spacy installed since it's just reading an already-generated
    CSV. Note YAKE's score convention: LOWER avg_score = more relevant."""
    if args.topic:
        kg_dir = os.path.join("kg_output", args.topic)
        path = os.path.join(kg_dir, "keyphrases.csv")
        if not os.path.exists(path):
            available = [
                os.path.basename(os.path.dirname(p))
                for p in glob.glob("kg_output/*/keyphrases.csv")
            ]
            print(f"No keyphrases.csv found at 'kg_output/{args.topic}/'.")
            print(f"Topics with keyphrases: {available}" if available else "No topics have keyphrases.csv — was this run with an older litkg version?")
            sys.exit(1)
    else:
        topic_dirs = sorted(
            glob.glob("kg_output/*/keyphrases.csv"),
            key=os.path.getmtime, reverse=True,
        )
        if not topic_dirs:
            print("No keyphrases.csv found under any kg_output/ topic.")
            sys.exit(1)
        path = topic_dirs[0]
        print(f"Using most recently modified topic folder: {os.path.dirname(path)}")

    df = pd.read_csv(path)
    print(f"\nTop {args.top} keyphrases by document frequency (lower avg_score = more relevant within a paper):\n")
    print(df.head(args.top).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(prog="litkg", description="Literature -> knowledge graph pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline")
    p_run.add_argument("--config", default="config.yaml")
    p_run.add_argument("--query", default=None, help='Override retrieval.query, e.g. --query \'"carbon budget"\'')
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--skip-wikidata", action="store_true", help="Skip Wikidata linking (fast, no network dependency for that step). Run 'litkg link' afterward to add links.")
    p_run.set_defaults(func=_cmd_run)

    p_link = sub.add_parser("link", help="Run (or re-run) Wikidata linking on an already-extracted topic")
    p_link.add_argument("--topic", default=None, help="Topic slug under kg_output/. Defaults to most recently modified.")
    p_link.add_argument("--config", default="config.yaml")
    p_link.set_defaults(func=_cmd_link)

    p_query = sub.add_parser("query", help="Query the graph for one entity")
    p_query.add_argument("entity", help="Entity name to search for")
    p_query.add_argument("--topic", default=None, help="Topic slug under kg_output/. Defaults to most recently modified.")
    p_query.add_argument("--config", default="config.yaml")
    p_query.set_defaults(func=_cmd_query)

    p_analyze = sub.add_parser("analyze", help="Analyze a pipeline run's output")
    p_analyze.add_argument("--topic", default=None)
    p_analyze.set_defaults(func=_cmd_analyze)

    p_keyphrases = sub.add_parser("keyphrases", help="View extracted keyphrases for a topic")
    p_keyphrases.add_argument("--topic", default=None)
    p_keyphrases.add_argument("--top", type=int, default=25)
    p_keyphrases.set_defaults(func=_cmd_keyphrases)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
