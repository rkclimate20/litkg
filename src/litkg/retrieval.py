"""Paper retrieval via pygetpapers, with caching and topic namespacing."""

import glob
import json
import os
import re
import subprocess
import time


def derive_keywords_from_query(query):
    """Auto-derives the keyword dictionary from quoted phrases in the query
    string, so changing retrieval.query automatically updates what counts as
    a relevant KEYWORD match — no manual list maintenance per topic."""
    return re.findall(r'"([^"]+)"', query)


def derive_topic_slug(query):
    """Derives a filesystem-safe folder name from the query's quoted
    phrases, so each distinct topic automatically gets its own corpus and
    kg_output subfolder — switching queries can never silently mix a new
    topic's papers into an old topic's folder."""
    phrases = re.findall(r'"([^"]+)"', query)
    base = "_".join(phrases) if phrases else query
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base.strip().lower()).strip("_")
    return slug[:80] or "default"


def retrieve_papers(cfg, log):
    """Run pygetpapers. Query is single-quoted at the shell level since it
    contains embedded double quotes for phrase search. Skips re-downloading
    only if the output folder already has papers AND they were downloaded
    for THIS exact query (tracked via a small marker file) — a query change
    correctly triggers a fresh retrieval rather than silently reusing a
    stale, mismatched corpus. Set retrieval.force_refresh: true in config
    to always fetch fresh regardless."""
    r = cfg["retrieval"]
    os.makedirs(r["output_dir"], exist_ok=True)
    marker_path = os.path.join(r["output_dir"], ".last_query.json")

    if not r.get("force_refresh", False):
        existing = [
            d for d in os.listdir(r["output_dir"])
            if os.path.isdir(os.path.join(r["output_dir"], d))
        ]
        last_query_info = None
        if os.path.exists(marker_path):
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    last_query_info = json.load(f)
            except (json.JSONDecodeError, OSError):
                last_query_info = None

        query_matches = (
            last_query_info is not None
            and last_query_info.get("query") == r["query"]
            and last_query_info.get("limit") == r["limit"]
        )

        if existing and query_matches:
            log.info(
                "Corpus already has %d paper folders in %s for this exact "
                "query — skipping re-download.",
                len(existing), r["output_dir"],
            )
            return r["output_dir"]
        elif existing and not query_matches:
            log.warning(
                "Corpus folder %s has existing papers, but they were "
                "downloaded for a different query (or no record of the "
                "query is available) — re-running pygetpapers for the "
                "current query. Note: new papers will be added alongside "
                "old ones in the same folder; use a different "
                "retrieval.output_dir per topic to keep corpora separate.",
                r["output_dir"],
            )

    cmd = [
        "pygetpapers",
        "--query", r["query"],
        "--xml", "--pdf",
        "--limit", str(r["limit"]),
        "--output", r["output_dir"],
        "--save_query",
    ]
    log.info("Running: %s", " ".join(cmd))
    result = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            break
        if attempt < max_retries:
            wait = 10 * (attempt + 1)
            log.warning(
                "pygetpapers exited with code %d on attempt %d/%d — "
                "retrying in %ds (transient network issues are common on "
                "long downloads).",
                result.returncode, attempt + 1, max_retries + 1, wait,
            )
            time.sleep(wait)

    downloaded_count = len([
        d for d in os.listdir(r["output_dir"])
        if os.path.isdir(os.path.join(r["output_dir"], d))
    ]) if os.path.exists(r["output_dir"]) else 0

    if result.returncode != 0:
        if downloaded_count > 0:
            log.warning(
                "pygetpapers exited with code %d (likely a network timeout "
                "partway through) but %d paper folders were already "
                "downloaded successfully before the failure — continuing "
                "with what's on disk rather than discarding that work. "
                "Not marking this query as complete, so a future run will "
                "retry retrieval to try to fill in the rest.",
                result.returncode, downloaded_count,
            )
            return r["output_dir"]
        else:
            log.error("pygetpapers failed: %s", result.stderr)
            raise RuntimeError(
                f"pygetpapers exited with code {result.returncode} and no "
                "papers were downloaded — nothing to work with."
            )

    log.info("Retrieval complete.")

    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump({"query": r["query"], "limit": r["limit"]}, f)

    return r["output_dir"]


def find_paper_units(corpus_dir, log):
    """Robustly find per-paper folders. Falls back to treating corpus_dir
    itself as one paper folder if pygetpapers wrote files flat."""
    subdirs = [
        os.path.join(corpus_dir, d)
        for d in os.listdir(corpus_dir)
        if os.path.isdir(os.path.join(corpus_dir, d))
    ]
    if subdirs:
        log.info("Found %d per-paper subfolders.", len(subdirs))
        return subdirs

    flat_files = (
        glob.glob(os.path.join(corpus_dir, "*.xml"))
        + glob.glob(os.path.join(corpus_dir, "*.pdf"))
    )
    if flat_files:
        log.warning("No subfolders found — treating corpus_dir as one paper unit.")
        return [corpus_dir]

    raise FileNotFoundError(f"No paper folders or files found under {corpus_dir}")
