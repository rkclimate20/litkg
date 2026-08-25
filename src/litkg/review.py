"""Human-review flagging and reproducibility manifest."""

import json
import os
from datetime import datetime, timezone

import spacy


def flag_for_review(df, log, freq_noise=None, audit_fraction=0.15, random_state=42):
    """Confidence flag, designed to scale: KEYWORD (substring-based, weaker
    evidence) is always flagged. GPE/ORG/LOC/DATE get a random audit_fraction
    sampled for review (catches filter blind spots without re-reading 100%
    of rows). Additionally, any entity text matched by the automatic
    ORG frequency detector is ALWAYS flagged — these are candidates the
    algorithm found suspicious but can't safely auto-delete (frequency alone
    can't distinguish 'boilerplate' from 'a genuinely important recurring
    term like a real agency name'), so a human makes the actual call."""
    if df.empty:
        return df
    freq_noise = freq_noise or set()
    df["needs_review"] = df["entity_type"] == "KEYWORD"
    other_idx = df.index[df["entity_type"] != "KEYWORD"]
    audit_idx = []
    if len(other_idx) > 0:
        audit_idx = df.loc[other_idx].sample(
            frac=audit_fraction, random_state=random_state
        ).index
        df.loc[audit_idx, "needs_review"] = True
    freq_mask = df["entity"].str.strip().str.lower().isin(freq_noise)
    df.loc[freq_mask, "needs_review"] = True
    log.info(
        "%d of %d triples flagged for human review (%d KEYWORD + %d audit sample "
        "+ %d frequency-detected).",
        df["needs_review"].sum(),
        len(df),
        (df["entity_type"] == "KEYWORD").sum(),
        len(audit_idx),
        freq_mask.sum(),
    )
    return df


def write_manifest(cfg, config_path, n_papers, n_triples, out_dir):
    manifest = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config_file": config_path,
        "config": cfg,
        "package_versions": {
            "spacy": spacy.__version__,
        },
        "papers_processed": n_papers,
        "triples_generated": n_triples,
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
