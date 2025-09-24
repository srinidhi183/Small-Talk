# integration_test_example.py

import sys
import os
import pandas as pd
import pytest

# Add parent dir so import works if running in Testcases or similar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommendation import generate_recommendation, validate_recommendation_output, extract_text_from_recommendations_json
from trending import generate_trending_set
from update_me import generate_update_me_news_set, generate_update_me_element
from search_bar import search_bar

@pytest.fixture
def sample_categories_csv(tmp_path):
    """
    Fixture to create a sample Categories.xlsx for test repeatability.
    """
    df = pd.DataFrame({
        "Subcategory": ["Formula 1", "Home Office", "Quantum Computing"],
        "Category": ["Sports", "Lifestyle", "Science"]
    })
    file_path = tmp_path / "Categories.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)

def test_integration_example_flow(monkeypatch, sample_categories_csv):
    print("\n=== INTEGRATION TEST: example.py workflow ===")

    # Optionally patch LLM/callable responses if you want to avoid real network calls
    monkeypatch.setattr('recommendation.generate_perplexity_output',
                        lambda *a, **k: {'choices': [{'message': {'content': '{"topic": "Home Office", "introduction": "Test intro", "key_tips_and_takeaways": {"Section": ["Tip"]}, "fun_facts": ["Fact"], "conclusion": "Done", "image_url": "img", "headline": "h", "subcategory":"Home Office"}'}}]})
    monkeypatch.setattr('recommendation.verify_and_fix_json', lambda raw: (True, {
        "topic": "Home Office",
        "introduction": "Test intro",
        "key_tips_and_takeaways": {"Section": ["Tip"]},
        "fun_facts": ["Fact"],
        "conclusion": "Done",
        "image_url": "img",
        "headline": "h",
        "subcategory": "Home Office"
    }))
    monkeypatch.setattr('recommendation.get_image', lambda topic: "http://fakeimage.com/image.png")
    # Patch trending and update_me for speed if desired
    monkeypatch.setattr('trending.generate_trending_set', lambda: ["Topic1", "Topic2", "Topic3", "Topic4", "Topic5"])
    monkeypatch.setattr('update_me.generate_update_me_news_set', lambda: [{"headline": "Test News 1"}, {"headline": "Test News 2"}, {"headline": "Test News 3"}])
    monkeypatch.setattr('update_me.generate_update_me_element', lambda keywords, **k: {"element": keywords})

    # --- Simulate search_bar output ---
    output_search_true = search_bar('Formula 1', allow_options=True)
    output_search_false = search_bar('Formula 1, Racing', allow_options=False)

    print("> SearchBar Output (allow_options=True):", output_search_true)
    print("> SearchBar Output (allow_options=False):", output_search_false)

    # --- Generate one recommendation output
    recommendation = generate_recommendation('home office', prompt_version='v02', categories_file=sample_categories_csv)
    print("> Recommendation Output:", recommendation)

    # --- Generate all recommendations for first 2 subcategories
    df = pd.read_excel(sample_categories_csv)
    subcategories = df['Subcategory'].dropna().tolist()
    recommendations = []
    for subcat in subcategories[:2]:
        recommendations.append(generate_recommendation(subcat, prompt_version='v02', categories_file=sample_categories_csv))
    print("> List of Recommendations (2 subcategories):", recommendations)

    # --- Generate trending
    trending = generate_trending_set()
    print("> Trending Output (5 topics):", trending)

    # --- Generate update_me news set
    update_me_news_set = generate_update_me_news_set()
    print("> Update Me News Set:", update_me_news_set)

    # --- Generate update_me element per topic
    update_me = []
    for subtopic in subcategories[:2]:
        update_me.append(generate_update_me_element(keywords=subtopic, prompt_type='update_me_general', categories_file=sample_categories_csv, prompt_version='v01'))
    print("> Update Me Elements (2 subcategories):", update_me)

    # --- Now basic assertions (integration pass)
    assert isinstance(output_search_true, dict) or isinstance(output_search_true, list) or output_search_true is not None
    assert isinstance(recommendation, dict)
    assert len(recommendations) == 2
    assert isinstance(trending, list) and len(trending) == 5
    assert isinstance(update_me_news_set, list) and len(update_me_news_set) == 3
    assert isinstance(update_me, list) and len(update_me) == 2
    print("Integration test for example.py workflow PASSED!")

