"""
Advanced test suite for update_me.py

Covers:
1) Property-based tests (Hypothesis) for validators and news-branch invariants
2) Contract/Golden tests for category mapping stability
3) Concurrency safety (no shared global state issues)
4) Live Integration tests (no mocks) hitting real dependencies

Usage:
- Run everything (unit-style, property, contract, concurrency):
    pytest tests/test_update_me_advanced.py -q

- Run only integration (no mocks; real APIs/files):
    pytest tests/test_update_me_advanced.py -m integration -v

- With coverage:
    pytest tests/test_update_me_advanced.py --cov=update_me --cov-report=term-missing -v
"""

import json
import re
import concurrent.futures as cf
from datetime import datetime, timedelta
from pytest import MonkeyPatch
from hypothesis import given, strategies as st

import pandas as pd
import pytest
from hypothesis import given, strategies as st

import update_me as mod


# =========================
# Helpers
# =========================
def _fake_categories_df(subcats=("AI", "Geopolitics", "Football")):
    # Minimal realistic category sheet for unit/property tests
    return pd.DataFrame({
        "Category": ["Tech", "World", "Sports"][:len(subcats)],
        "Subcategory": list(subcats),
    })


def _assert_valid_schema(payload):
    ok, errs = mod.validate_update_me_output(payload)
    assert ok, f"Output schema invalid. Errors: {errs}"


def _assert_reasonable_picture_url(url):
    # Allow empty (service might return none), else must look URL-like
    if url:
        assert re.match(r"^https?://", url), f"picture_url not URL-like: {url}"


# =========================
# 1) Property-based tests
# =========================

# Free-form keywords; must contain at least one alphanumeric char to avoid all-symbol noise
kw_strat = st.text(min_size=1).filter(lambda s: any(c.isalnum() for c in s))

# Valid minimal parsed JSON (schema-happy) for news branch
valid_json_strategy = st.fixed_dictionaries({
    "category": st.text(min_size=0),  # overwritten by function
    "subcategory": st.sampled_from(["AI", "Geopolitics", "Football"]),
    "topic": st.text(min_size=1),
    "short_info": st.lists(st.text(min_size=1), min_size=1, max_size=5),
    "background_story": st.fixed_dictionaries({
        "context_history": st.text(min_size=0, max_size=200),
        "current_developments": st.text(min_size=0, max_size=200),
        "relevance": st.text(min_size=0, max_size=200),
    }),
})


@given(valid_json_strategy)
def test_validate_update_me_output_always_true_on_valid(valid_json):
    ok, errs = mod.validate_update_me_output(valid_json)
    assert ok and errs == []


@given(
    st.one_of(
        st.none(),
        st.integers(),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers()),
    )
)
def test_extract_text_is_total_function_on_garbage_inputs(x):
    # Should never throw; always returns a string
    out = mod.extract_text_from_update_element_json(x)
    assert isinstance(out, str)

# ... keep your existing helpers/strategies (kw_strat, valid_json_strategy, etc.)
def test_generate_update_me_element_news_branch_invariants_deterministic():
    # deterministic “property-like” checks without Hypothesis
    from pytest import MonkeyPatch
    mp = MonkeyPatch()
    try:
        mp.setattr(mod.pd, "read_excel", lambda *_a, **_k: _fake_categories_df())
        mp.setattr(mod, "build_prompt", lambda **_kw: "PROMPT")

        # Two representative parsed outputs
        samples = [
            {  # known subcategory -> 'Tech'
                "category": "",
                "subcategory": "AI",
                "topic": "Transformers",
                "short_info": ["pt1", "pt2"],
                "background_story": {
                    "context_history": "c",
                    "current_developments": "d",
                    "relevance": "r",
                },
            },
            {  # unknown subcategory -> 'General'
                "category": "",
                "subcategory": "NotInSheet",
                "topic": "X",
                "short_info": ["a"],
                "background_story": {
                    "context_history": "",
                    "current_developments": "",
                    "relevance": "",
                },
            },
        ]

        for parsed in samples:
            mp.setattr(mod, "get_llm_response", lambda _p, _j=json.dumps(parsed): _j)
            mp.setattr(mod, "verify_and_fix_json", lambda r: (True, json.loads(r)))
            mp.setattr(mod, "get_image", lambda s: f"https://img/{s}.png" if s else "")

            out = mod.generate_update_me_element(
                keywords="ai news",
                prompt_type="update_me_news",
                categories_file="Categories.xlsx",
            )
            _assert_valid_schema(out)

            df = _fake_categories_df()
            subcat = out.get("subcategory", "")
            expected_cat = (
                df[df["Subcategory"] == subcat]["Category"].values[0]
                if not df[df["Subcategory"] == subcat].empty
                else "General"
            )
            assert out["category"] == expected_cat
            _assert_reasonable_picture_url(out.get("picture_url", ""))
    finally:
        mp.undo()





# =========================
# 2) Contract/Golden tests
# =========================

GOLDEN_CASES = [
    # (keywords, parsed_subcat, expected_category)
    ("AI breakthroughs", "AI", "Tech"),
    ("Middle East diplomacy", "Geopolitics", "World"),
    ("UEFA fixtures", "Football", "Sports"),
    ("Unknown topic", "DoesNotExist", "General"),
]


@pytest.mark.parametrize("kw,subcat,expected_cat", GOLDEN_CASES, ids=[
    "AI->Tech", "Geo->World", "Football->Sports", "Fallback->General"
])
def test_category_mapping_is_stable(monkeypatch, kw, subcat, expected_cat):
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: _fake_categories_df())
    monkeypatch.setattr(mod, "build_prompt", lambda **_kw: "PROMPT")

    parsed = {
        "category": "",
        "subcategory": subcat,
        "topic": "T",
        "short_info": ["a"],
        "background_story": {
            "context_history": "c",
            "current_developments": "d",
            "relevance": "r",
        },
    }
    monkeypatch.setattr(mod, "get_llm_response", lambda _p: json.dumps(parsed))
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda r: (True, json.loads(r)))
    monkeypatch.setattr(mod, "get_image", lambda s: "https://img/T.png" if s else "")

    out = mod.generate_update_me_element(
        keywords=kw,
        prompt_type="update_me_news",
        categories_file="Categories.xlsx",
    )
    assert out["category"] == expected_cat
    _assert_valid_schema(out)


# =========================
# 3) Concurrency safety
# =========================

def _setup_concurrency_monkey(monkeypatch):
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: pd.DataFrame({
        "Category": ["Tech"], "Subcategory": ["AI"]
    }))
    monkeypatch.setattr(mod, "build_prompt", lambda **_kw: "PROMPT")
    parsed = {
        "category": "", "subcategory": "AI", "topic": "T",
        "short_info": ["a"], "background_story": {
            "context_history": "c", "current_developments": "d", "relevance": "r"
        }
    }
    monkeypatch.setattr(mod, "get_llm_response", lambda _p: json.dumps(parsed))
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda r: (True, json.loads(r)))
    monkeypatch.setattr(mod, "get_image", lambda s: "https://img/T.png")


def _call_once():
    return mod.generate_update_me_element(
        keywords="k", prompt_type="update_me_news", categories_file="Categories.xlsx"
    )


def test_concurrent_calls_are_isolated(monkeypatch):
    _setup_concurrency_monkey(monkeypatch)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: _call_once(), range(16)))
    for r in results:
        _assert_valid_schema(r)


# =========================
# 4) Live Integration (no mocks)
# =========================

pytestmark_integration = pytest.mark.integration  # convenience alias


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_generate_update_me_element_live_news_branch():
    """
    Full live call to the news branch:
    - build_prompt -> get_llm_response -> verify_and_fix_json -> get_image
    - requires Categories.xlsx and working API credentials behind llm_utils/api_utils.
    """
    keywords = "Artificial Intelligence latest breakthroughs"
    out = mod.generate_update_me_element(
        keywords=keywords,
        link=None,
        news_text="Recent developments and analysis about AI breakthroughs",
        prompt_type="update_me_news",
        categories_file="Categories.xlsx",
        prompt_version="v01",
    )
    assert out is not None
    _assert_valid_schema(out)
    assert isinstance(out["category"], str) and out["category"]
    assert isinstance(out["subcategory"], str) and out["subcategory"]
    assert isinstance(out["topic"], str) and out["topic"]
    assert isinstance(out["short_info"], list) and len(out["short_info"]) >= 1
    _assert_reasonable_picture_url(out.get("picture_url", ""))


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_generate_update_me_element_live_general_branch():
    """
    Live call to the general branch (Perplexity path).
    """
    keywords = "AI"
    out = mod.generate_update_me_element(
        keywords=keywords,
        prompt_type="update_me_general",
        categories_file="Categories.xlsx",
        prompt_version="v01",
    )
    assert out is not None
    _assert_valid_schema(out)
    assert out["subcategory"] == keywords
    assert isinstance(out["category"], str) and out["category"]
    _assert_reasonable_picture_url(out.get("picture_url", ""))


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_generate_update_me_news_set_live():
    """
    End-to-end: fetch top news -> iterate generate_update_me_element() for each item.
    """
    update_set = mod.generate_update_me_news_set(
        use_tfidf=True,
        sort_by_date=False,
        categories_file="Categories.xlsx",
        prompt_version="v01",
    )
    assert isinstance(update_set, list) and len(update_set) >= 1
    for item in update_set:
        assert item is not None
        _assert_valid_schema(item)
        _assert_reasonable_picture_url(item.get("picture_url", ""))
