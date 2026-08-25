# litkg

Literature → knowledge graph pipeline. Retrieves scientific papers on any
topic, extracts named entities, links them to Wikidata, and builds a
FAIR-compliant RDF knowledge graph — fully config-driven and topic-agnostic
(validated on both coral-bleaching/ocean-acidification and carbon-budget/
IPCC literature).

## Install

From GitHub (before this is published to PyPI):

```bash
pip install git+https://github.com/<your-org>/litkg.git
```

Then install the spaCy model separately (not installable via pip metadata):

```bash
python -m spacy download en_core_web_sm
```

## Quickstart

1. Copy `config.yaml` into your working directory and edit `retrieval.query`.
2. Run the pipeline:

```bash
litkg run --config config.yaml
```

Or override the query/limit without editing the file:

```bash
litkg run --config config.yaml --query "\"carbon budget\" AND \"IPCC\"" --limit 30
```

3. Query the resulting graph for one entity:

```bash
litkg query "IPCC" --topic carbon_budget_ipcc
```

4. Check the health of a run:

```bash
litkg analyze --topic carbon_budget_ipcc
```

## What each command does

- **`litkg run`** — pygetpapers retrieval (cached, retry-resilient) → JATS-aware
  text extraction → spaCy NER with automatic noise filtering → Wikidata
  entity linking (batched, cached) → RDF graph output (`knowledge_graph.ttl`,
  `validation_report.csv`, `manifest.json`).
- **`litkg query <entity>`** — pulls every paper mentioning an entity,
  deduplicates excerpts, groups by paper. No synthesis — every excerpt is
  verbatim from the source. Writes structured JSON to
  `kg_output/<topic>/answers/<entity>.json`.
- **`litkg analyze`** — validation report: log warnings, per-paper mention
  counts, review-flag rates by entity type, Wikidata linking rates, and a
  random sample of papers to spot-check.

## Design principles

- **No manual intervention required** to run — the only human step is
  reviewing `validation_report.csv`'s flagged rows.
- **Topic-agnostic** — change `retrieval.query`, everything else (keywords,
  noise filtering, output folders) adapts automatically. No code edits, no
  per-topic manual tuning.
- **FAIR** — stable URIs, Dublin Core + PROV-O + SKOS vocabularies, explicit
  license, and a reproducibility manifest recording exact config and package
  versions per run.

## Using it as a library

```python
from litkg import run_pipeline, build_answer
from litkg.query import resolve_ttl_path, load_namespace

output_dir = run_pipeline(config_path="config.yaml", query_override='"deforestation"')

ttl_path = resolve_ttl_path(topic="deforestation")
namespace = load_namespace("config.yaml")
results = build_answer("Amazon", ttl_path, namespace)
```

## Known limitations

- Some citation styles (bare "Author and Author Year" without wrapping
  parentheses) aren't stripped and may occasionally surface as mislabeled
  entities — caught by the review-flag system, not silently invisible.
- Taxonomic/genus names can be mislabeled as locations by the general-purpose
  NER model (no domain-specific model swap has been done yet).
- Sub-document aliasing (e.g. two different phrasings of the same guidance
  document) isn't resolved — only the broader "Full Name (ACRONYM)" pattern is.

## License

MIT (package code). Retrieved paper text is subject to each paper's own
license — the pipeline records `dcterms:license` per paper where available
and defaults to CC-BY-4.0 in the graph metadata; verify against the actual
source license before redistribution.
