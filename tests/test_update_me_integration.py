# tests/test_update_me_integration.py
import re
import json
import pytest
import os
from datetime import datetime, timedelta

import update_me as mod

pytestmark = pytest.mark.integration  # mark the whole module as integration

# ---- Helper to assert your schema using your own validator ----
def _assert_valid_schema(payload):
    ok, errs = mod.validate_update_me_output(payload)
    assert ok, f"Output schema invalid. Errors: {errs}"

def _assert_reasonable_picture_url(url):
    # real call may occasionally be empty; allow empty but if present, it should look like a URL
    if url:
        assert re.match(r"^https?://", url), f"picture_url not URL-like: {url}"

@pytest.mark.timeout(120)
def test_generate_update_me_element_live_news_branch():
    """
    Full live call to the news branch:
    - Uses build_prompt -> get_llm_response -> verify_and_fix_json -> category mapping
    - Requires Categories.xlsx present and real network/API access.
    """
    # Pick a current topic that likely has recent coverage
    keywords = "Artificial Intelligence latest breakthroughs"
    # Note: link/news_text may be omitted—your prompt builder accepts these and LLM still returns a JSON.
    out = mod.generate_update_me_element(
        keywords=keywords,
        link=None,
        news_text="Recent developments and analysis about AI breakthroughs",
        prompt_type="update_me_news",
        categories_file="Categories.xlsx",
        prompt_version="v01",
    )

    assert out is not None, "Function returned None; LLM or JSON post-process failed."
    _assert_valid_schema(out)

    # basic sanity checks that should hold with live data
    assert isinstance(out["category"], str) and len(out["category"]) > 0
    assert isinstance(out["subcategory"], str) and len(out["subcategory"]) > 0
    assert isinstance(out["topic"], str) and len(out["topic"]) > 0
    assert isinstance(out["short_info"], list) and len(out["short_info"]) >= 1
    assert isinstance(out["background_story"], dict)

    _assert_reasonable_picture_url(out.get("picture_url", ""))

@pytest.mark.timeout(120)
def test_generate_update_me_element_live_general_branch():
    """
    Live call to the general branch (Perplexity path in your code).
    """
    keywords = "AI"  # should exist as a Subcategory in your Categories.xlsx, or fallback to 'General'
    out = mod.generate_update_me_element(
        keywords=keywords,
        prompt_type="update_me_general",
        categories_file="Categories.xlsx",
        prompt_version="v01",
    )

    assert out is not None, "General branch returned None."
    _assert_valid_schema(out)

    # General branch overrides subcategory = keywords
    assert out["subcategory"] == keywords
    assert isinstance(out["category"], str) and len(out["category"]) > 0
    _assert_reasonable_picture_url(out.get("picture_url", ""))

@pytest.mark.timeout(180)
def test_generate_update_me_news_set_live():
    """
    End-to-end: fetch top news -> iterate generate_update_me_element() for each item.
    """
    update_set = mod.generate_update_me_news_set(
        use_tfidf=True, sort_by_date=False, categories_file="Categories.xlsx", prompt_version="v01"
    )

    assert isinstance(update_set, list) and len(update_set) >= 1, "No update elements produced."

    # Validate every produced element
    for i, item in enumerate(update_set, start=1):
        assert item is not None, f"Item {i} is None."
        _assert_valid_schema(item)
        _assert_reasonable_picture_url(item.get("picture_url", ""))

