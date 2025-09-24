import os
import json
import time
import pathlib
import pandas as pd
import pytest
import trending

pytestmark = pytest.mark.integration

# ---------- Helpers ----------

def _require_file(path_relative: str) -> str:
    """Resolve and assert the existence of a file relative to project root."""
    p = pathlib.Path(path_relative).resolve()
    if not p.exists():
        pytest.skip(f"Required file not found: {p}")
    return str(p)

def _is_http_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _save_artifact(keyword: str, payload: dict):
    outdir = pathlib.Path("test_artifacts/live_runs")
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_kw = "".join(ch if ch.isalnum() else "_" for ch in keyword)[:40]
    path = outdir / f"trending_{safe_kw}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)

# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def categories_real_path():
    # Adjust if your Categories.xlsx lives elsewhere
    return _require_file("Categories.xlsx")

@pytest.fixture(scope="session")
def categories_df(categories_real_path):
    return pd.read_excel(categories_real_path)

# ---------- Core integration tests ----------

@pytest.mark.timeout(60)
@pytest.mark.parametrize("kw", [
    "AI chips",
    "NFL season",
    "Bitcoin price",
])
def test_live_generate_trending_element_real_world(kw, categories_real_path, categories_df):
    """
    Live end-to-end: real Categories.xlsx + real LLM/news/image calls.
    Validates schema via validate_trending_output and basic invariants.
    Saves an artifact JSON for client evidence.
    """
    t0 = time.time()
    out = trending.generate_trending_element(
        keywords=kw,
        categories_file=categories_real_path,
        prompt_version="v02",      # use your real version here
    )
    elapsed = time.time() - t0

    # Basic existence
    assert out is not None, f"generate_trending_element returned None for '{kw}'"

    # Validate with your own validator (should be strict)
    ok, errors = trending.validate_trending_output(out)
    assert ok, f"Schema invalid for '{kw}': {errors}"

    # Picture URL should be http(s)
    assert _is_http_url(out.get("picture_url", "")), "picture_url must be http/https"

    # Category mapping should be resolved to a string
    assert isinstance(out.get("category", ""), str) and out["category"], "category must be non-empty string"

    # If subcategory exists in Excel, category should match mapping; otherwise General is fine
    subcat = str(out.get("subcategory", "")).strip()
    if subcat:
        in_excel = not categories_df[categories_df["Subcategory"] == subcat].empty
        if in_excel:
            mapped = categories_df.loc[categories_df["Subcategory"] == subcat, "Category"].iloc[0]
            assert out["category"] == mapped, f"Category mismatch for subcategory '{subcat}'"
        else:
            assert out["category"] in {"General", out["category"]}  # allow General fallback

    # Save artifact for client
    artifact_path = _save_artifact(kw, out)
    # Soft performance check (tune as you wish)
    assert elapsed < 60, f"Call took too long: {elapsed:.1f}s (artifact: {artifact_path})"

@pytest.mark.timeout(90)
def test_live_generate_trending_set_with_keywords(categories_real_path):
    """
    Live: build a small set with multiple keywords.
    Ensures None results are filtered and all outputs validate.
    """
    keywords = ["OpenAI news", "Champions League", "Electric vehicles"]
    results = trending.generate_trending_set(
        keywords=keywords,
        categories_file=categories_real_path,
        prompt_version="v02",
    )
    # Must return as many valid objects as could be created (skipping Nones is okay)
    assert isinstance(results, list) and len(results) >= 1

    for item in results:
        ok, errors = trending.validate_trending_output(item)
        assert ok, f"Invalid item in set: {errors}"
        assert _is_http_url(item.get("picture_url", "")), "picture_url must be http/https"

@pytest.mark.timeout(90)
def test_live_generate_trending_set_without_keywords(categories_real_path):
    """
    Live: no keywords → relies on real get_trending_topics().
    Ensures outputs validate and artifact is captured.
    """
    results = trending.generate_trending_set(
        keywords=None,
        categories_file=categories_real_path,
        prompt_version="v02",
    )
    assert isinstance(results, list) and len(results) >= 1

    for i, item in enumerate(results):
        ok, errors = trending.validate_trending_output(item)
        assert ok, f"Invalid item at index {i}: {errors}"
        assert _is_http_url(item.get("picture_url", "")), "picture_url must be http/https"
        _save_artifact(f"auto_{i}", item)

def test_generate_trending_element_picture_url_fallback(monkeypatch, tiny_categories_xlsx):
    import json, trending
    payload = {
        "topic": "Fallback Image Case",
        "subcategory": "AI",  # in tiny excel → maps to Tech
        "description": "d",
        "why_is_it_trending": {},
        "key_points": {},
        "overlook_what_might_happen_next": {},
        "picture_url": "placeholder",
    }
    monkeypatch.setattr(trending, "generate_perplexity_output",
                        lambda sp, up: {"choices": [{"message": {"content": json.dumps(payload)}}]})
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, json.loads(s)))
    # Force a bad image return to trigger fallback
    monkeypatch.setattr(trending, "get_image", lambda q: None)

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is not None
    assert out["category"] == "Tech"
    assert isinstance(out["picture_url"], str) and out["picture_url"].startswith(("http://", "https://"))

def test_generate_trending_element_invalid_parse_type(monkeypatch, tiny_categories_xlsx):
    import trending
    # LLM returns something that the parser claims is 'valid' but it's not a dict
    monkeypatch.setattr(trending, "generate_perplexity_output",
                        lambda sp, up: {"choices": [{"message": {"content": "42"}}]})
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, 42))  # not a dict
    # get_image won't be called, but stub anyway
    monkeypatch.setattr(trending, "get_image", lambda q: "http://img.test/pic.png")

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is None

def test_generate_trending_element_normalizes_scalar_fields(monkeypatch, tiny_categories_xlsx):
    import json, trending
    payload = {
        "topic": "Scalar Fields Case",
        "subcategory": "AI",
        "description": "d",
        "why_is_it_trending": "scalar",    # → []
        "key_points": 123,                 # → []
        "overlook_what_might_happen_next": None,  # → []
        "picture_url": "placeholder",
    }
    monkeypatch.setattr(trending, "generate_perplexity_output",
                        lambda sp, up: {"choices": [{"message": {"content": json.dumps(payload)}}]})
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, json.loads(s)))
    monkeypatch.setattr(trending, "get_image", lambda q: "http://img.test/pic.png")

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is not None
    assert out["why_is_it_trending"] == [] and out["key_points"] == [] and out["overlook_what_might_happen_next"] == []

def test_generate_trending_element_picture_url_fallback_exact(monkeypatch, tiny_categories_xlsx):
    import json, trending
    payload = {
        "topic": "Exact Fallback Hit",
        "subcategory": "AI",
        "description": "d",
        "why_is_it_trending": {},
        "key_points": {},
        "overlook_what_might_happen_next": {},
        "picture_url": "placeholder",
    }
    # Valid parse
    monkeypatch.setattr(trending, "generate_perplexity_output",
                        lambda sp, up: {"choices": [{"message": {"content": json.dumps(payload)}}]})
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, json.loads(s)))
    # Return an explicitly BAD URL that fails http(s) check to ensure the fallback line executes
    monkeypatch.setattr(trending, "get_image", lambda q: "ftp://not-http.example/image.png")

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is not None
    assert out["picture_url"].startswith(("http://", "https://"))
    assert "placehold.co" in out["picture_url"]
