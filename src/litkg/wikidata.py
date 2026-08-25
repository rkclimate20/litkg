"""Wikidata reconciliation: batched, cached, confidence-aware entity linking."""

import json
import os
import time

import requests

RECONCILIATION_URL = "https://wikidata.reconci.link/en/api"


def _load_wikidata_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_wikidata_cache(cache_path, cache):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _pick_best_candidate(candidates, min_score):
    """Prefers a genuine match=true candidate. Falls back to a single
    unambiguous top-scorer (score above threshold AND clearly ahead of the
    runner-up) only if no candidate is explicitly marked match=true.
    Returns None rather than guessing among tied, ambiguous candidates."""
    if not candidates:
        return None
    for c in candidates:
        if c.get("match") is True:
            return c["id"]
    top = candidates[0]
    if top.get("score", 0) < min_score:
        return None
    if len(candidates) > 1 and candidates[1].get("score", 0) == top.get("score", 0):
        return None  # tied with runner-up, genuinely ambiguous
    return top["id"]


def link_entities_to_wikidata(
    unique_entities, log, batch_size=5, delay_seconds=1.0,
    cache_path="wikidata_cache.json", min_score=70, timeout=45, max_retries=2,
):
    """Looks up each unique entity text against Wikidata's reconciliation
    API and returns {text: qid_or_None}. Deliberately conservative:
    - Batched (not one request per entity, not all at once) — polite to a
      shared public service and resilient to a single bad batch. Batch size
      kept small (5) since real-world testing showed this API can take
      15-20+ seconds even for a 2-entity request.
    - Cached to disk — re-runs only query entities not already resolved.
    - Prefers Wikidata's own "match" flag over raw score: when several
      candidates tie at the same high score (e.g. 'NOAA' the agency vs
      'NOAA' the satellite series, both scoring 100), the API itself marks
      match=false for all of them because IT isn't confident which one is
      right — blindly taking the first-listed candidate in that case is a
      coin flip, not a real match. Only a genuine match=true candidate, or
      a single unambiguous top-scorer with a real gap over the runner-up,
      gets accepted.
    - Retries a failed batch (with backoff) before giving up on it.
    - Any network failure is logged and skipped, never crashes the run."""
    cache = _load_wikidata_cache(cache_path)
    to_query = [e for e in unique_entities if e not in cache]
    log.info(
        "Entity linking: %d entities cached, %d to query against Wikidata.",
        len(unique_entities) - len(to_query), len(to_query),
    )

    for i in range(0, len(to_query), batch_size):
        batch = to_query[i:i + batch_size]
        queries = {f"q{j}": {"query": text} for j, text in enumerate(batch)}
        succeeded = False
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    RECONCILIATION_URL,
                    data={"queries": json.dumps(queries)},
                    timeout=timeout,
                )
                resp.raise_for_status()
                results = resp.json()
                for j, text in enumerate(batch):
                    candidates = results.get(f"q{j}", {}).get("result", [])
                    cache[text] = _pick_best_candidate(candidates, min_score)
                log.info(
                    "Wikidata batch %d-%d done (%d matched)%s.",
                    i, i + len(batch),
                    sum(1 for t in batch if cache.get(t)),
                    f" on retry {attempt}" if attempt else "",
                )
                succeeded = True
                break
            except (requests.RequestException, ValueError) as e:
                if attempt < max_retries:
                    wait = 3 * (attempt + 1)
                    log.warning(
                        "Wikidata batch %d-%d failed (%s) — retrying in %ds "
                        "(attempt %d/%d).",
                        i, i + len(batch), e, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                else:
                    log.warning(
                        "Wikidata batch %d-%d failed after %d attempts (%s) — "
                        "leaving these entities unlinked for THIS run, but "
                        "NOT caching them as 'no match' — a network failure "
                        "isn't evidence the entity has no Wikidata page, so "
                        "a future run will retry them rather than silently "
                        "treating this as a permanent negative result.",
                        i, i + len(batch), max_retries + 1, e,
                    )
        # Deliberately no cache.setdefault(text, None) here on failure —
        # only a genuine API response (success path above) writes a result
        # to the cache. A network failure leaves the entity absent from the
        # cache entirely, so `to_query = [e for e in unique_entities if e
        # not in cache]` on the next run will correctly retry it.

        if i + batch_size < len(to_query):
            time.sleep(delay_seconds)

    _save_wikidata_cache(cache_path, cache)
    return {e: cache.get(e) for e in unique_entities}
