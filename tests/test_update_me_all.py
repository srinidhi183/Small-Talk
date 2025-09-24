"""
Unified test suite for update_me.py
Includes:
1. Unit tests (mocked, fast, deterministic)
2. Integration tests (no mocks, real data)
3. Advanced tests (contracts, concurrency, resilience)

Run everything:
    pytest tests/test_update_me_all.py -v --cov=update_me --cov-report=term-missing

Run only integration (live calls):
    pytest tests/test_update_me_all.py -v -m integration
"""
# Single entry point that pulls in unit, integration, and advanced tests
from tests.test_update_me_unit import *           # unit (mocked)
from tests.test_update_me_integration import *    # integration (live)
from tests.test_update_me_advanced import *       # advanced (contracts, concurrency, some live)
import json
import re
import pytest
import pandas as pd
import concurrent.futures as cf
from pytest import MonkeyPatch
from hypothesis import given, strategies as st

import update_me as mod

# ============================================================
# Helpers
# ============================================================

def _fake_categories_df(subcats=("AI", "Geopolitics", "Football")):
    return pd.DataFrame({
        "Category": ["Tech", "World", "Sports"][:len(subcats)],
        "Subcategory": list(subcats),
    })

def _assert_valid_schema(payload):
    ok, errs = mod.validate_update_me_output(payload)
    assert ok, f"Schema invalid: {errs}"

def _assert_reasonable_picture_url(url):
    if url:
        assert re.match(r"^https?://", url)

# ============================================================
# UNIT TESTS
# ============================================================

def test_generate_update_me_element_general_branch(monkeypatch):
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: _fake_categories_df())
    monkeypatch.setattr(mod, "build_prompt", lambda **kw: "PROMPT")
    monkeypatch.setattr(mod, "generate_perplexity_output",
        lambda **kw: {"choices":[{"message":{"content":json.dumps({
            "category": "", "subcategory":"AI", "topic":"LLMs",
            "short_info":["a","b"],
            "background_story":{"context_history":"c","current_developments":"d","relevance":"r"}
        })}}]}
    )
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda r: (True, json.loads(r)))
    monkeypatch.setattr(mod, "get_image", lambda s: "https://img/T.png")
    out = mod.generate_update_me_element("AI", prompt_type="update_me_general", categories_file="Categories.xlsx")
    assert out["subcategory"] == "AI"
    assert out["category"] == "Tech"

def test_generate_update_me_element_news_branch(monkeypatch):
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: _fake_categories_df())
    monkeypatch.setattr(mod, "build_prompt", lambda **kw: "PROMPT")
    parsed = {
        "category":"", "subcategory":"AI", "topic":"LLMs",
        "short_info":["a"], "background_story":{"context_history":"c","current_developments":"d","relevance":"r"}
    }
    monkeypatch.setattr(mod, "get_llm_response", lambda p: json.dumps(parsed))
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda r: (True, json.loads(r)))
    monkeypatch.setattr(mod, "get_image", lambda s: "https://img/T.png")
    out = mod.generate_update_me_element("AI news", prompt_type="update_me_news", categories_file="Categories.xlsx")
    _assert_valid_schema(out)

def test_validate_update_me_output_reports_errors():
    bad = {"category":123,"topic":"t","short_info":["a",1],
           "background_story":{"context_history":1}}
    ok, errs = mod.validate_update_me_output(bad)
    assert not ok and errs

def test_extract_text_from_update_element_json_handles_non_strs():
    j = {"category":1,"subcategory":"AI","topic":"X","short_info":[1,2],
         "background_story":{"context_history":1,"current_developments":2,"relevance":3}}
    out = mod.extract_text_from_update_element_json(j)
    assert isinstance(out, str)  # should not crash

# ============================================================
# INTEGRATION TESTS (live)
# ============================================================

@pytest.mark.integration
@pytest.mark.timeout(120)
def test_generate_update_me_element_live_news_branch():
    out = mod.generate_update_me_element(
        keywords="Artificial Intelligence latest breakthroughs",
        news_text="Recent AI developments",
        prompt_type="update_me_news",
        categories_file="Categories.xlsx"
    )
    assert out is not None
    _assert_valid_schema(out)

@pytest.mark.integration
@pytest.mark.timeout(120)
def test_generate_update_me_element_live_general_branch():
    out = mod.generate_update_me_element(
        keywords="AI",
        prompt_type="update_me_general",
        categories_file="Categories.xlsx"
    )
    assert out is not None
    _assert_valid_schema(out)
    assert out["subcategory"] == "AI"

@pytest.mark.integration
@pytest.mark.timeout(180)
def test_generate_update_me_news_set_live():
    update_set = mod.generate_update_me_news_set(
        use_tfidf=True, sort_by_date=False, categories_file="Categories.xlsx"
    )
    assert isinstance(update_set, list) and update_set
    for item in update_set:
        _assert_valid_schema(item)

# ============================================================
# ADVANCED TESTS (contracts, concurrency)
# ============================================================

GOLDEN_CASES = [
    ("AI breakthroughs","AI","Tech"),
    ("Middle East","Geopolitics","World"),
    ("UEFA","Football","Sports"),
    ("Unknown","Nope","General"),
]

@pytest.mark.parametrize("kw,subcat,expected", GOLDEN_CASES)
def test_category_mapping_is_stable(monkeypatch, kw, subcat, expected):
    monkeypatch.setattr(mod.pd,"read_excel",lambda *_a,**_k:_fake_categories_df())
    monkeypatch.setattr(mod,"build_prompt",lambda **_k:"PROMPT")
    parsed={"category":"","subcategory":subcat,"topic":"T",
            "short_info":["a"],"background_story":{"context_history":"c","current_developments":"d","relevance":"r"}}
    monkeypatch.setattr(mod,"get_llm_response",lambda p:json.dumps(parsed))
    monkeypatch.setattr(mod,"verify_and_fix_json",lambda r:(True,json.loads(r)))
    monkeypatch.setattr(mod,"get_image",lambda s:"https://img/T.png")
    out = mod.generate_update_me_element(kw,prompt_type="update_me_news",categories_file="Categories.xlsx")
    assert out["category"] == expected

def test_concurrent_calls_are_isolated(monkeypatch):
    monkeypatch.setattr(mod.pd,"read_excel",lambda *_a,**_k:_fake_categories_df())
    monkeypatch.setattr(mod,"build_prompt",lambda **_k:"PROMPT")
    parsed={"category":"","subcategory":"AI","topic":"T",
            "short_info":["a"],"background_story":{"context_history":"c","current_developments":"d","relevance":"r"}}
    monkeypatch.setattr(mod,"get_llm_response",lambda p:json.dumps(parsed))
    monkeypatch.setattr(mod,"verify_and_fix_json",lambda r:(True,json.loads(r)))
    monkeypatch.setattr(mod,"get_image",lambda s:"https://img/T.png")
    def _call(): return mod.generate_update_me_element("k",prompt_type="update_me_news",categories_file="Categories.xlsx")
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        results=list(ex.map(lambda _: _call(), range(8)))
    for r in results: _assert_valid_schema(r)
