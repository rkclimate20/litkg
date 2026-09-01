"""Automatic keyphrase extraction via txt2phrases (semanticClimate org
tool): AI-model-based keyword extraction, with TF-IDF classification into
'specific' vs 'general' terms. Replaces an earlier YAKE-based approach,
which produced too many generic single-word/overlapping terms (gene, cell,
editing, gene editing, crispr gene editing all cluttering the same list) —
the specific/general split directly addresses that.

Operates on a directory of .txt files (txt2phrases' own API), not raw
in-memory strings, so this module writes each paper's text to a temp
directory, calls txt2phrases, then cleans up.
"""

import glob
import os
import shutil
import tempfile

import pandas as pd


def _sanitize_filename(paper_id):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in paper_id)[:100]


def extract_corpus_keyphrases(papers_with_text, output_dir, top_n=200, classify_threshold=0.7, log=None):
    """papers_with_text: list of {"paper_id": ..., "text": ...}.

    Writes keyphrases.csv into output_dir (the raw txt2phrases keyword/count
    table), and — if the TF-IDF specific/general classification step
    succeeds — keyphrases_specific.csv and keyphrases_general.csv alongside
    it. Returns the path to keyphrases.csv, or None if extraction produced
    no output. Any failure here is logged and skipped rather than crashing
    the pipeline — keyphrases are a bonus output, not required for the
    knowledge graph itself to be usable."""
    from txt2phrases import KeywordExtraction

    tmp_dir = tempfile.mkdtemp(prefix="litkg_txt2phrases_")
    try:
        n_written = 0
        for p in papers_with_text:
            if not p.get("text"):
                continue
            fname = _sanitize_filename(p["paper_id"]) + ".txt"
            with open(os.path.join(tmp_dir, fname), "w", encoding="utf-8") as f:
                f.write(p["text"])
            n_written += 1

        if n_written == 0:
            if log:
                log.warning("No paper text available for keyphrase extraction — skipping.")
            return None

        raw_output_dir = os.path.join(output_dir, "_txt2phrases_raw")
        os.makedirs(raw_output_dir, exist_ok=True)

        extractor = KeywordExtraction(
            input_path=tmp_dir,
            output_folder=raw_output_dir,
            top_n=top_n,
        )
        extractor.extract()

        produced_csvs = glob.glob(os.path.join(raw_output_dir, "*.csv"))
        if not produced_csvs:
            if log:
                log.warning("txt2phrases produced no CSV output — skipping keyphrases.")
            return None

        # txt2phrases may write one CSV per input file or one aggregate
        # file depending on version/mode — handle either by combining
        # whatever it produced into a single corpus-level table.
        if len(produced_csvs) == 1:
            combined = pd.read_csv(produced_csvs[0])
        else:
            combined = pd.concat([pd.read_csv(p) for p in produced_csvs], ignore_index=True)

        keyphrases_path = os.path.join(output_dir, "keyphrases.csv")
        combined.to_csv(keyphrases_path, index=False, encoding="utf-8-sig")
        if log:
            log.info(
                "Wrote keyphrase table: %s (%d rows from %d papers)",
                keyphrases_path, len(combined), n_written,
            )

        # TF-IDF specific/general split — this is what actually fixes the
        # "too many generic single-word terms" complaint, since it
        # separates domain-specific phrases from broadly-generic ones.
        try:
            from txt2phrases import classify_keywords_split_files
            classified_dir = os.path.join(output_dir, "_txt2phrases_classified")
            os.makedirs(classified_dir, exist_ok=True)
            classify_keywords_split_files(raw_output_dir, classified_dir, threshold=classify_threshold)
            for f in glob.glob(os.path.join(classified_dir, "*.csv")):
                dest_name = "keyphrases_" + os.path.basename(f)
                shutil.copy(f, os.path.join(output_dir, dest_name))
            if log:
                log.info("Wrote specific/general keyphrase split alongside keyphrases.csv.")
        except Exception as e:
            if log:
                log.warning("Keyphrase specific/general classification skipped (%s).", e)

        return keyphrases_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
