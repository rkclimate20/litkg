"""JATS-aware paper text extraction and citation stripping."""

import glob
import json
import os
import re
import xml.etree.ElementTree as ET

CITATION_PATTERN = re.compile(r"\([^()]*\b(?:19|20)\d{2}[a-z]?\b[^()]*\)")

# Catches "Author (Year)" / "Author et al. (Year)" style citations where the
# name sits outside the parenthesis — a different citation convention than
# CITATION_PATTERN above, common in some journals (e.g. numbered-reference
# style papers that still narrate "Padfield (2016) showed...").
AUTHOR_YEAR_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?(?:\s+(?:&|and)\s+[A-Za-z\-]+)?\s*\((?:19|20)\d{2}[a-z]?\)"
)


def _strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _clean_text(text):
    text = AUTHOR_YEAR_PATTERN.sub(" ", text)
    text = CITATION_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_paper_text(paper_dir):
    """JATS-aware extraction: pulls only <abstract> and <body><p> text,
    explicitly excluding front-matter (journal/PMC/DOI metadata, license,
    keywords) and back-matter (references) rather than blanket tag-stripping
    the whole document. Falls back to a regex strip if the XML doesn't parse.
    Always reads as UTF-8 to avoid mojibake from platform-default encodings."""
    xml_files = glob.glob(os.path.join(paper_dir, "*.xml"))
    if not xml_files:
        txt_files = glob.glob(os.path.join(paper_dir, "*.txt"))
        if not txt_files:
            return None
        with open(txt_files[0], "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return _clean_text(text)

    path = xml_files[0]
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        text = re.sub("<[^<]+?>", " ", raw)
        return _clean_text(text)

    paragraphs = []
    for el in root.iter():
        if _strip_ns(el.tag) == "abstract":
            paragraphs.append(" ".join(el.itertext()))
    for el in root.iter():
        if _strip_ns(el.tag) == "body":
            for p in el.iter():
                if _strip_ns(p.tag) == "p":
                    ptext = " ".join(p.itertext()).strip()
                    if ptext.lower().startswith("keywords"):
                        continue
                    paragraphs.append(ptext)

    text = " ".join(paragraphs)
    return _clean_text(text) if text.strip() else None


def load_paper_metadata(paper_dir):
    """Lenient loader — pygetpapers' metadata JSON field names can vary by
    version/repository, so pull whatever common keys exist and default
    everything else to None rather than crashing."""
    candidates = glob.glob(os.path.join(paper_dir, "*.json"))
    meta = {"title": None, "doi": None, "year": None}
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        meta["title"] = meta["title"] or data.get("title")
        meta["doi"] = meta["doi"] or data.get("doi")
        meta["year"] = meta["year"] or data.get("pubYear") or data.get("year")
    return meta
