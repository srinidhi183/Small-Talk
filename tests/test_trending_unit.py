# tests/test_trending_unit.py
import pytest
import trending

pytestmark = pytest.mark.unit

# ----------------------------
# validate_trending_output()
# ----------------------------

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
# (Your function’s contract & messages are defined in trending.py.)  # :contentReference[oaicite:2]{index=2}

# ------------------------------------------
# extract_text_from_trending_now_json()
# ------------------------------------------

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
# Function pulls strings and joins them with newlines.  # :contentReference[oaicite:3]{index=3}

def test_extract_text_non_dict_returns_empty():
    assert trending.extract_text_from_trending_now_json(["not", "a", "dict"]) == ""

def test_extract_text_handles_missing_keys():
    data = {"topic": "Only Topic"}
    txt = trending.extract_text_from_trending_now_json(data)
    assert txt.strip().startswith("Only Topic")


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

# (If you later coerce list items to str before joining, remove xfail.)  # :contentReference[oaicite:4]{index=4}

# ------------------------------------------
# generate_trending_set() with mocks only
# ------------------------------------------

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
# generate_trending_set iterates keywords and delegates per item.  # :contentReference[oaicite:5]{index=5}

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
# If keywords is falsy, it calls get_trending_topics().  # :contentReference[oaicite:6]{index=6}

# -----------------------------------------------------
# generate_trending_element() — unit-style smoke test
# (mock LLM+image; use tiny temp Excel to map subcat→cat)
# -----------------------------------------------------

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
# Function reads Excel, parses LLM JSON, sets picture, resolves category.  # :contentReference[oaicite:7]{index=7}

def test_generate_trending_element_returns_none_when_parse_fails(monkeypatch, tiny_categories_xlsx):
    import trending
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

def test_generate_trending_element_missing_topic_returns_none(monkeypatch, tiny_categories_xlsx):
    import json, trending
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


def test_generate_trending_element_unknown_subcategory_maps_to_general(monkeypatch, tiny_categories_xlsx):
    import json, trending
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


def test_generate_trending_set_skips_none_elements(monkeypatch, tiny_categories_xlsx):
    import trending
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

def test_validate_trending_output_top_level_not_object():
    import trending
    ok, errors = trending.validate_trending_output(["not", "a", "dict"])
    assert ok is False
    assert "Top-level JSON must be an object." in errors

def test_generate_trending_element_picture_url_fallback(monkeypatch, tiny_categories_xlsx):
    import json, trending
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
