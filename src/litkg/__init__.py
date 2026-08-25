"""litkg — literature to knowledge graph pipeline.

Retrieves scientific literature, extracts named entities, links them to
Wikidata, and builds a FAIR-compliant RDF knowledge graph — with a
config-driven, topic-agnostic design (works on any domain, not just
climate literature).
"""

__version__ = "0.1.0"

__all__ = ["run_pipeline", "build_answer", "__version__"]


def __getattr__(name):
    """Lazy imports (PEP 562): `import litkg` alone, or `from litkg.query
    import build_answer`, shouldn't require spacy/pygetpapers to be
    installed — only run_pipeline actually needs the full dependency
    stack, and only at the point it's called, not at import time."""
    if name == "run_pipeline":
        from litkg.pipeline import run_pipeline
        return run_pipeline
    if name == "build_answer":
        from litkg.query import build_answer
        return build_answer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

