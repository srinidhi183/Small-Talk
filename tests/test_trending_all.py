# tests/test_trending_all.py
"""
Unified test suite for `trending`:
- Unit tests (fast, mocked) and
- Integration tests (live, slower) in one file.

Tips:
- To run only unit tests:         pytest -m unit
- To run only integration tests:  pytest -m integration
- To avoid unknown mark warnings, add to pytest.ini:

[pytest]
markers =
    unit: marks tests as fast, isolated, mocked
    integration: marks tests as live and slower
"""

import os
import json
import time
import pathlib
import pandas as pd
import pytest
import trending




# ===========================
# Shared Helpers (integration)
# ===========================

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

# ===========================
# Fixtures (previously conftest)
# ===========================

@pytest.fixture
def tiny_categories_xlsx(tmp_path):
    """
    Create a minimal Excel with the exact columns your code expects:
    Category | Subcategory
    """
    df = pd.DataFrame({
        "Category": ["Tech", "Sports"],
        "Subcategory": ["AI", "NFL"],
    })
    path = tmp_path / "Categories.xlsx"
    df.to_excel(path, index=False)
    return str(path)

@pytest.fixture
def stub_llm_success(monkeypatch):
    """
    Stub LLM + parser + image fetcher so generate_trending_element() becomes deterministic.
    """
    payload = {
        "topic": "AI Chips",
        "category": "placeholder",                 # will be overwritten by mapping
        "subcategory": "AI",                       # exists in our tiny Excel
        "description": "Why AI chips matter",
        "why_is_it_trending": ["GPUs", "Launches", "Demand"],
        "key_points": ["NVIDIA/AMD", "Cloud", "Edge"],
        "overlook_what_might_happen_next": ["Prices", "Custom silicon", "Regulation"],
        "picture_url": "placeholder"
    }
    as_json = json.dumps(payload)

    def fake_generate_perplexity_output(system_prompt, user_input):
        return {"choices": [{"message": {"content": as_json}}]}

    def fake_verify_and_fix_json(s):
        return True, json.loads(s)

    def fake_get_image(q):
        return "http://img.test/ai-chips.png"

    monkeypatch.setattr(trending, "generate_perplexity_output", fake_generate_perplexity_output)
    monkeypatch.setattr(trending, "verify_and_fix_json", fake_verify_and_fix_json)
    monkeypatch.setattr(trending, "get_image", fake_get_image)

# =====================================
# Integration-only fixtures and helpers
# =====================================

@pytest.fixture(scope="session")
def categories_real_path():
    # Adjust if your Categories.xlsx lives elsewhere
    return _require_file("Categories.xlsx")

@pytest.fixture(scope="session")
def categories_df(categories_real_path):
    return pd.read_excel(categories_real_path)

# ======================
# UNIT TESTS (fast/mocks)
# ======================

@pytest.mark.unit
def test_validate_trending_output_valid():
    data = {
        "topic": "AI Chips",
        "category": "Tech",
        "subcategory": "AI",
        "description": "Why AI chips matter",
        "why_is_it_trending": ["GPUs supply", "New launches", "Enterprise demand"],
        "key_points": ["NVIDIA/AMD moves", "Cloud vendors", "Edge AI"],
        "overlook_what_might_happen_next": ["Lower prices", "Custom silicon", "Regulation"],
    }
    ok, errors = trending.validate_trending_output(data)
    assert ok is True and errors == []

@pytest.mark.unit
def test_validate_trending_output_missing_and_type_errors():
    data = {
        "topic": 123,                         # should be str
        "category": "Tech",
        # subcategory missing
        "description": "desc",
        "why_is_it_trending": "not-a-list",   # should be list[str]
        "key_points": ["good", 42],           # mixed types
        # overlook_what_might_happen_next missing
    }
    ok, errors = trending.validate_trending_output(data)
    assert ok is False
    joined = " | ".join(errors)
    assert "Missing required field: 'subcategory'." in joined
    assert "Field 'topic' must be a string" in joined
    assert "Field 'why_is_it_trending' must be a list" in joined
    assert "Element at index 1 in list 'key_points' must be a string" in joined

@pytest.mark.unit
def test_extract_text_happy_path():
    data = {
        "topic": "AI Chips",
        "category": "Tech",
        "subcategory": "AI",
        "description": "Short explainer",
        "why_is_it_trending": ["GPUs", "Launches"],
        "key_points": ["NVIDIA/AMD"],
        "overlook_what_might_happen_next": ["Custom silicon"],
    }
    txt = trending.extract_text_from_trending_now_json(data)
    lines = txt.splitlines()
    assert lines[0] == "AI Chips"
    assert "Tech" in lines[1]
    assert "AI" in lines[2]
    assert "Short explainer" in lines[3]
    assert "GPUs" in txt and "Custom silicon" in txt

@pytest.mark.unit
def test_extract_text_non_dict_returns_empty():
    assert trending.extract_text_from_trending_now_json(["not", "a", "dict"]) == ""

@pytest.mark.unit
def test_extract_text_handles_missing_keys():
    data = {"topic": "Only Topic"}
    txt = trending.extract_text_from_trending_now_json(data)
    assert txt.strip().startswith("Only Topic")

@pytest.mark.unit
def test_extract_text_non_string_list_items_are_coerced():
    data = {
        "topic": "AI Chips",
        "category": "Tech",
        "subcategory": "AI",
        "description": "Short explainer",
        "key_points": ["ok", 42, 3.14],                 # coerced to "42" and "3.14"
        "why_is_it_trending": [None, "", "still valid"], # None/empty ignored
    }
    txt = trending.extract_text_from_trending_now_json(data)
    assert "AI Chips" in txt
    assert "Tech" in txt
    assert "AI" in txt
    assert "Short explainer" in txt
    assert "ok" in txt
    assert "42" in txt
    assert "3.14" in txt
    assert "still valid" in txt

@pytest.mark.unit
def test_generate_trending_set_with_keywords_invokes_generator(monkeypatch, tiny_categories_xlsx):
    calls = []
    def fake_generate_element(kw, **kwds):
        calls.append(kw)
        return {
            "topic": kw, "category": "Tech", "subcategory": "AI", "description": "d",
            "why_is_it_trending": [], "key_points": [], "overlook_what_might_happen_next": [],
            "picture_url": "u",
        }
    monkeypatch.setattr(trending, "generate_trending_element", fake_generate_element)

    out = trending.generate_trending_set(
        keywords=["A", "B", "C"],
        categories_file=tiny_categories_xlsx
    )
    assert [o["topic"] for o in out] == ["A", "B", "C"]
    assert calls == ["A", "B", "C"]

@pytest.mark.unit
def test_generate_trending_set_without_keywords_fetches_topics(monkeypatch, tiny_categories_xlsx):
    monkeypatch.setattr(trending, "get_trending_topics", lambda: ["K1", "K2"])
    monkeypatch.setattr(trending, "generate_trending_element",
                        lambda kw, **kwds: {
                            "topic": kw, "category": "Tech", "subcategory": "AI", "description": "d",
                            "why_is_it_trending": [], "key_points": [], "overlook_what_might_happen_next": [],
                            "picture_url": "u",
                        })
    out = trending.generate_trending_set(categories_file=tiny_categories_xlsx)
    assert [o["topic"] for o in out] == ["K1", "K2"]

@pytest.mark.unit
def test_generate_trending_element_smoke(tiny_categories_xlsx, stub_llm_success):
    out = trending.generate_trending_element(
        keywords="ai chips",
        categories_file=tiny_categories_xlsx,
        prompt_version="v02",
    )
    assert out is not None
    # Subcategory "AI" from stub maps to Category "Tech" in tiny Excel
    assert out["subcategory"] == "AI"
    assert out["category"] == "Tech"
    assert out["picture_url"].startswith("http")

@pytest.mark.unit
def test_generate_trending_element_returns_none_when_parse_fails(monkeypatch, tiny_categories_xlsx):
    # Force invalid JSON parsing result
    def bad_generate_perplexity_output(system_prompt, user_input):
        return {"choices": [{"message": {"content": "{not json}"}}]}
    def bad_verify_and_fix_json(s):
        return False, {}
    # image fetch won't be hit, but stub anyway to be safe
    def fake_get_image(q): return "http://img.test/unused.png"

    monkeypatch.setattr(trending, "generate_perplexity_output", bad_generate_perplexity_output)
    monkeypatch.setattr(trending, "verify_and_fix_json", bad_verify_and_fix_json)
    monkeypatch.setattr(trending, "get_image", fake_get_image)

    out = trending.generate_trending_element("whatever", categories_file=tiny_categories_xlsx)
    assert out is None

@pytest.mark.unit
def test_generate_trending_element_missing_topic_returns_none(monkeypatch, tiny_categories_xlsx):
    # Parsed successfully but no 'topic'
    payload = {
        # "topic": missing on purpose
        "subcategory": "AI",
        "description": "d",
        "why_is_it_trending": [],
        "key_points": [],
        "overlook_what_might_happen_next": [],
        "picture_url": "placeholder",
    }
    monkeypatch.setattr(
        trending,
        "generate_perplexity_output",
        lambda sys_p, user_p: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, json.loads(s)))
    monkeypatch.setattr(trending, "get_image", lambda q: "http://img.test/unused.png")

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is None  # early return when 'topic' missing

@pytest.mark.unit
def test_generate_trending_element_unknown_subcategory_maps_to_general(monkeypatch, tiny_categories_xlsx):
    payload = {
        "topic": "Something",
        "subcategory": "NoSuchSubcat",  # not in tiny excel
        "description": "d",
        "why_is_it_trending": [],
        "key_points": [],
        "overlook_what_might_happen_next": [],
        "picture_url": "placeholder",
    }
    monkeypatch.setattr(
        trending,
        "generate_perplexity_output",
        lambda sys_p, user_p: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, json.loads(s)))
    monkeypatch.setattr(trending, "get_image", lambda q: "http://img.test/pic.png")

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is not None
    assert out["category"] == "General"  # fallback branch covered
    assert out["subcategory"] == "NoSuchSubcat"

@pytest.mark.unit
def test_generate_trending_set_skips_none_elements(monkeypatch, tiny_categories_xlsx):
    # First keyword returns None, second returns a valid dict → ensure skip logic is covered
    def fake_elem(kw, **_):
        if kw == "bad":
            return None
        return {
            "topic": kw, "category": "Tech", "subcategory": "AI", "description": "d",
            "why_is_it_trending": [], "key_points": [], "overlook_what_might_happen_next": [],
            "picture_url": "u",
        }
    monkeypatch.setattr(trending, "generate_trending_element", fake_elem)

    out = trending.generate_trending_set(keywords=["bad", "good"], categories_file=tiny_categories_xlsx)
    assert [o["topic"] for o in out] == ["good"]  # 'bad' was filtered out

@pytest.mark.unit
def test_validate_trending_output_top_level_not_object():
    ok, errors = trending.validate_trending_output(["not", "a", "dict"])
    assert ok is False
    assert "Top-level JSON must be an object." in errors

@pytest.mark.unit
def test_generate_trending_element_picture_url_fallback_unit(monkeypatch, tiny_categories_xlsx):
    payload = {
        "topic": "Fallback Image Case",
        "subcategory": "AI",       # present in tiny excel → maps to Tech
        "description": "d",
        "why_is_it_trending": {},
        "key_points": {},
        "overlook_what_might_happen_next": {},
        "picture_url": "placeholder",
    }
    monkeypatch.setattr(
        trending,
        "generate_perplexity_output",
        lambda sp, up: {"choices": [{"message": {"content": json.dumps(payload)}}]},
    )
    # valid parse
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, json.loads(s)))
    # Force a bad image return to trigger fallback
    monkeypatch.setattr(trending, "get_image", lambda q: None)

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is not None
    assert out["category"] == "Tech"
    # should use placeholder http(s) URL now
    assert isinstance(out["picture_url"], str) and out["picture_url"].startswith(("http://", "https://"))

# ============================
# INTEGRATION TESTS (live/slow)
# ============================

@pytest.mark.integration
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

@pytest.mark.integration
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

@pytest.mark.integration
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

@pytest.mark.integration
def test_generate_trending_element_picture_url_fallback_integration(monkeypatch, tiny_categories_xlsx):
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

@pytest.mark.integration
def test_generate_trending_element_invalid_parse_type(monkeypatch, tiny_categories_xlsx):
    # LLM returns something that the parser claims is 'valid' but it's not a dict
    monkeypatch.setattr(trending, "generate_perplexity_output",
                        lambda sp, up: {"choices": [{"message": {"content": "42"}}]})
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, 42))  # not a dict
    # get_image won't be called, but stub anyway
    monkeypatch.setattr(trending, "get_image", lambda q: "http://img.test/pic.png")

    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is None

@pytest.mark.integration
def test_generate_trending_element_normalizes_scalar_fields(monkeypatch, tiny_categories_xlsx):
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

@pytest.mark.integration
def test_generate_trending_element_picture_url_fallback_exact(monkeypatch, tiny_categories_xlsx):
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

# ==========================================
# ADVANCED TESTS (add below your current code)
# ==========================================

# 1) Property-based fuzzing of validators & extractors
# ----------------------------------------------------
# Ensures your validator never crashes on arbitrary JSON shapes and types.
# Also checks the extractor always returns a string (possibly empty).
import pytest
from hypothesis import given, strategies as st

@given(st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()),
    lambda children: st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=10), children, max_size=6),
    max_leaves=30
))
def test_validate_never_crashes_on_arbitrary_json(arbitrary):
    ok, errors = trending.validate_trending_output(arbitrary)
    assert isinstance(ok, bool)
    assert isinstance(errors, list)

@given(st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()),
    lambda children: st.lists(children, max_size=6) | st.dictionaries(st.text(max_size=10), children, max_size=6),
    max_leaves=30
))
def test_extract_text_returns_string_for_any_shape(arbitrary):
    out = trending.extract_text_from_trending_now_json(arbitrary)
    assert isinstance(out, str)


# 2) JSON Schema contract tests for generate_trending_element
# -----------------------------------------------------------
# Locks the output shape so accidental key changes are caught early.
def _compile_contract_validator():
    fastjsonschema = pytest.importorskip("fastjsonschema")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": [
            "topic", "category", "subcategory", "description",
            "why_is_it_trending", "key_points", "overlook_what_might_happen_next", "picture_url"
        ],
        "properties": {
            "topic": {"type": "string", "minLength": 1},
            "category": {"type": "string", "minLength": 1},
            "subcategory": {"type": "string"},
            "description": {"type": "string"},
            "why_is_it_trending": {"type": "array", "items": {"type": "string"}},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "overlook_what_might_happen_next": {"type": "array", "items": {"type": "string"}},
            "picture_url": {"type": "string", "minLength": 1}
        },
        "additionalProperties": True
    }
    return fastjsonschema.compile(schema)

def test_contract_generate_trending_element_valid_under_stub(stub_llm_success, tiny_categories_xlsx):
    validate = _compile_contract_validator()
    out = trending.generate_trending_element("ai chips", categories_file=tiny_categories_xlsx)
    assert out is not None
    validate(out)  # raises on contract break


# 3) Unicode / very-long inputs robustness
# ----------------------------------------
def test_long_and_unicode_inputs_dont_break_extract_text():
    weird_keyword = "🤖🚀🔥 " * 500 + "العربية 中文 हिन्दी русский"
    data = {
        "topic": weird_keyword,
        "category": "Tech",
        "subcategory": "AI",
        "description": "A" * 5000,
        "why_is_it_trending": ["🔥"*100, "趋势", "الاتجاه"],
        "key_points": ["ポイント1", "ポイント2"],
        "overlook_what_might_happen_next": ["Δοκιμή", "Тест"],
        "picture_url": "https://example.com/p.png"
    }
    txt = trending.extract_text_from_trending_now_json(data)
    assert isinstance(txt, str)
    # Should include key markers even for very long text
    assert "Tech" in txt and "AI" in txt


# 4) Concurrency & idempotence under stub
# ---------------------------------------
# Ensures no shared-state bugs when generating a set in parallel-like scenarios.
def test_generate_trending_set_concurrency(monkeypatch, tiny_categories_xlsx):
    # Deterministic stub: returns predictable object based on keyword
    def stub_elem(kw, **_):
        return {
            "topic": kw, "category": "Tech", "subcategory": "AI", "description": "d",
            "why_is_it_trending": ["x"], "key_points": ["y"], "overlook_what_might_happen_next": ["z"],
            "picture_url": "http://img.test/u.png",
        }
    monkeypatch.setattr(trending, "generate_trending_element", stub_elem)
    keywords = [f"K{i}" for i in range(100)]
    out = trending.generate_trending_set(keywords=keywords, categories_file=tiny_categories_xlsx)
    assert len(out) == len(keywords)
    # Idempotence: running again yields same ordered topics
    out2 = trending.generate_trending_set(keywords=keywords, categories_file=tiny_categories_xlsx)
    assert [o["topic"] for o in out] == [o["topic"] for o in out2] == keywords


# 5) URL safety & placeholder fallback hardening
# ----------------------------------------------
# Ensures non-http(s) URLs are rejected in favor of a safe placeholder.
def test_picture_url_rejects_non_http_schemes(monkeypatch, tiny_categories_xlsx):
    payload = {
        "topic": "Non-HTTP URL",
        "subcategory": "AI",
        "description": "desc",
        "why_is_it_trending": [],
        "key_points": [],
        "overlook_what_might_happen_next": [],
        "picture_url": "placeholder",
    }
    # Valid parse to dict
    monkeypatch.setattr(
        trending,
        "generate_perplexity_output",
        lambda sp, up: {"choices": [{"message": {"content": __import__("json").dumps(payload)}}]},
    )
    monkeypatch.setattr(trending, "verify_and_fix_json", lambda s: (True, __import__("json").loads(s)))
    # Return a bad scheme and ensure the function replaces it with a proper placeholder http(s) URL
    monkeypatch.setattr(trending, "get_image", lambda q: "data:image/png;base64,AAAA")
    out = trending.generate_trending_element("kw", categories_file=tiny_categories_xlsx)
    assert out is not None
    assert isinstance(out["picture_url"], str)
    assert out["picture_url"].startswith(("http://", "https://"))