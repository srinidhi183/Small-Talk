import types
import pandas as pd
import pytest

# Import the module under test
import update_me as mod

# ---------- Fixtures & Mocks ----------

@pytest.fixture
def fake_categories_df():
    # Mimic Categories.xlsx content
    return pd.DataFrame(
        {
            "Category": ["Tech", "World", "Sports"],
            "Subcategory": ["AI", "Geopolitics", "Football"],
        }
    )

@pytest.fixture(autouse=True)
def mock_read_excel(monkeypatch, fake_categories_df):
    monkeypatch.setattr(mod.pd, "read_excel", lambda *a, **k: fake_categories_df)

@pytest.fixture
def mock_build_prompt(monkeypatch):
    monkeypatch.setattr(mod, "build_prompt", lambda **kw: f"PROMPT::{kw['prompt_type']}::{kw['prompt_version']}")

@pytest.fixture
def mock_get_image(monkeypatch):
    monkeypatch.setattr(mod, "get_image", lambda topic_slug: f"https://img/{topic_slug}.png")

@pytest.fixture
def mock_verify_and_fix_json_ok(monkeypatch):
    def _mock(resp):
        # resp shape differs by branch; we only need to return (True, parsed_json)
        return True, {
            "category": "",   # will be overwritten later
            "subcategory": "AI",
            "topic": "Large Language Models",
            "short_info": ["p1", "p2", "p3"],
            "background_story": {
                "context_history": "c",
                "current_developments": "d",
                "relevance": "r",
            },
        }
    monkeypatch.setattr(mod, "verify_and_fix_json", _mock)

@pytest.fixture
def mock_verify_and_fix_json_bad(monkeypatch):
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda resp: (False, {}))

# ---------- Happy paths ----------

def test_generate_update_me_element_general_branch(
    mock_build_prompt, mock_get_image, mock_verify_and_fix_json_ok, monkeypatch
):
    # general branch uses generate_perplexity_output
    def fake_perplexity(system_content, user_content):
        # must return structure like raw_response['choices'][0]['message']['content']
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}
    monkeypatch.setattr(mod, "generate_perplexity_output", fake_perplexity)

    out = mod.generate_update_me_element(
        keywords="AI", prompt_type="update_me_general", categories_file="Categories.xlsx"
    )

    assert out is not None
    # subcategory overridden to keywords in 'general' branch
    assert out["subcategory"] == "AI"
    # category inferred from df where Subcategory == keywords
    assert out["category"] == "Tech"
    # picture_url injected from get_image with dashed topic
    assert out["picture_url"] == "https://img/Large-Language-Models.png"

def test_generate_update_me_element_news_branch(
    mock_build_prompt, mock_get_image, mock_verify_and_fix_json_ok, monkeypatch
):
    # news branch uses get_llm_response
    monkeypatch.setattr(mod, "get_llm_response", lambda prompt: '{"ok": true}')

    out = mod.generate_update_me_element(
        keywords="Some headline",
        link="http://example.com",
        news_text="full text",
        prompt_type="update_me_news",  # not 'update_me_general'
        categories_file="Categories.xlsx",
    )

    assert out is not None
    # subcategory comes from parsed_json ('AI'), not from keywords in news branch
    assert out["subcategory"] == "AI"
    assert out["category"] == "Tech"
    assert out["picture_url"] == "https://img/Large-Language-Models.png"

# ---------- Fallbacks & error paths ----------

def test_generate_update_me_element_category_fallback_to_general(
    mock_build_prompt, mock_get_image, monkeypatch
):
    # Make parsed_json have a subcategory that doesn't exist in the sheet
    def _ok(_resp):
        return True, {
            "category": "",
            "subcategory": "NotInSheet",
            "topic": "X",
            "short_info": ["a"],
            "background_story": {
                "context_history": "",
                "current_developments": "",
                "relevance": "",
            },
        }
    monkeypatch.setattr(mod, "verify_and_fix_json", _ok)
    monkeypatch.setattr(mod, "get_llm_response", lambda p: "{}")

    out = mod.generate_update_me_element(
        keywords="whatever",
        prompt_type="update_me_news",
        categories_file="Categories.xlsx",
    )
    assert out["subcategory"] == "NotInSheet"
    assert out["category"] == "General"  # fallback

def test_generate_update_me_element_returns_none_if_invalid_json(
    mock_build_prompt, monkeypatch
):
    monkeypatch.setattr(mod, "get_llm_response", lambda p: "{}")
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda r: (False, {}))
    # image should not be called if invalid; no need to mock get_image

    out = mod.generate_update_me_element(
        keywords="k", prompt_type="update_me_news", categories_file="Categories.xlsx"
    )
    assert out is None

# ---------- Pure helpers: quick coverage ----------

def test_validate_update_me_output_happy():
    valid_json = {
        "category": "Tech",
        "subcategory": "AI",
        "topic": "LLMs",
        "short_info": ["a", "b"],
        "background_story": {
            "context_history": "c",
            "current_developments": "d",
            "relevance": "r",
        },
    }
    ok, errs = mod.validate_update_me_output(valid_json)
    assert ok is True
    assert errs == []

def test_validate_update_me_output_reports_errors():
    bad = {
        "category": 123,  # not string
        # missing 'subcategory'
        "topic": "t",
        "short_info": ["a", 1],  # not all strings
        "background_story": {"context_history": 1},  # wrong types + missing fields
    }
    ok, errs = mod.validate_update_me_output(bad)
    assert ok is False
    assert any("subcategory" in e for e in errs)
    assert any("must be a string" in e for e in errs)
    assert any("short_info" in e for e in errs)
    assert any("background_story" in e for e in errs)

def test_extract_text_from_update_element_json_basic():
    j = {
        "category": "Tech",
        "subcategory": "AI",
        "topic": "LLMs",
        "short_info": ["a", "b"],
        "background_story": {
            "context_history": "c",
            "current_developments": "d",
            "relevance": "r",
        },
    }
    text = mod.extract_text_from_update_element_json(j)
    # should include all non-empty pieces separated by newlines
    assert "Tech" in text and "AI" in text and "LLMs" in text
    assert "a" in text and "b" in text
    assert "c" in text and "d" in text and "r" in text

def test_extract_text_from_update_element_json_tolerates_non_dict():
    assert mod.extract_text_from_update_element_json(["not", "a", "dict"]) == ""

def test_picture_url_empty_when_topic_missing(monkeypatch, mock_build_prompt):
    import update_me as mod
    monkeypatch.setattr(mod, "get_llm_response", lambda p: "{}")
    # valid JSON but no topic
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda r: (True, {
        "category": "", "subcategory": "AI",
        "short_info": ["x"], "background_story": {
            "context_history":"c","current_developments":"d","relevance":"r"}
    }))
    monkeypatch.setattr(mod, "get_image", lambda s: f"https://img/{s}.png")
    out = mod.generate_update_me_element(keywords="k", prompt_type="update_me_news", categories_file="Categories.xlsx")
    assert out["picture_url"] == ""


def test_general_branch_unknown_subcat_goes_to_general(monkeypatch, mock_build_prompt):
    import update_me as mod, pandas as pd
    # categories sheet without the keyword
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: pd.DataFrame({"Category":["Tech"],"Subcategory":["AI"]}))
    monkeypatch.setattr(mod, "generate_perplexity_output", lambda **kw: {"choices":[{"message":{"content":"{}"}}]})
    monkeypatch.setattr(mod, "verify_and_fix_json", lambda r: (True, {
        "topic":"T", "subcategory":"ignored",
        "short_info":["a"], "background_story":{"context_history":"c","current_developments":"d","relevance":"r"}
    }))
    monkeypatch.setattr(mod, "get_image", lambda s: "https://img/T.png")

    out = mod.generate_update_me_element(keywords="NotInSheet", prompt_type="update_me_general", categories_file="Categories.xlsx")
    assert out["subcategory"] == "NotInSheet"
    assert out["category"] == "General"


def test_get_random_subtopics_clamps_and_returns_list(monkeypatch):
    import update_me as mod, pandas as pd
    monkeypatch.setattr(mod.pd, "read_excel", lambda *_a, **_k: pd.DataFrame({"Subcategory":["A","B"]}))
    out = mod.get_random_subtopics_for_update_me(5, categories_file="Categories.xlsx")
    assert len(out) == 2 and set(out) <= {"A","B"}

def test_generate_update_me_news_set(monkeypatch):
    import update_me as mod
    monkeypatch.setattr(mod, "get_random_subtopics_for_update_me", lambda n, **k: ["AI", "Tech"])
    monkeypatch.setattr(mod, "get_top_news_full_json", lambda **k: [
        {"title": "T1", "link": "L1", "full_text": "F1", "image": "I1"},
        {"title": "T2", "link": "L2", "full_text": "F2", "image": "I2"},
    ])
    monkeypatch.setattr(mod, "generate_update_me_element", lambda **kw: {"ok": kw["keywords"]})
    out = mod.generate_update_me_news_set()
    assert len(out) == 2
    assert {"ok": "T1"} in out and {"ok": "T2"} in out


@pytest.mark.parametrize("bad_input", [
    None,
    {"category": "c", "subcategory": "s", "topic": "t"},  # missing short_info/bg
    {"category": "c", "subcategory": "s", "topic": "t", "short_info": "notalist"},
    {"category": "c", "subcategory": "s", "topic": "t", "short_info": ["a"], "background_story": "notadict"},
])
def test_validate_update_me_output_edge_cases(bad_input):
    import update_me as mod
    ok, errs = mod.validate_update_me_output(bad_input)
    assert not ok
    assert errs
