import sys
import os
import pytest
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommendation import (
    generate_recommendation,
    validate_recommendation_output,
    extract_text_from_recommendations_json,
)

def print_test_detail(name, description, inputs, expected, actual, passed):
    print(f"\n=== TEST CASE: {name} ===")
    print(f"• What is being tested: {description}")
    print(f"• Input provided: {inputs}")
    print(f"• Expected output: {expected}")
    print(f"• Actual output:   {actual}")
    print(f"• Status: {'PASSED' if passed else 'FAILED'}\n")


# --- Fixtures For Input and Expected Output ---

@pytest.fixture
def valid_output_json():
    return {
        "introduction": "Intro",
        "key_tips_and_takeaways": {"Section": ["Tip"]},
        "fun_facts": ["Fact"],
        "conclusion": "Conclusion"
    }

@pytest.fixture
def missing_intro_json():
    return {
        "key_tips_and_takeaways": {},
        "fun_facts": [],
        "conclusion": "Conclusion"
    }

@pytest.fixture
def wrong_type_json():
    return {
        "introduction": "Intro",
        "key_tips_and_takeaways": "not a dict",
        "fun_facts": [],
        "conclusion": "Conclusion"
    }

@pytest.fixture
def nested_type_json():
    return {
        "introduction": "Intro",
        "key_tips_and_takeaways": {"Category": ["Tip", 123]},
        "fun_facts": [],
        "conclusion": "Conclusion"
    }

@pytest.fixture
def empty_json():
    return {}

@pytest.fixture
def extraction_input():
    return {
        "introduction": "Intro.",
        "key_tips_and_takeaways": {"Section 1": ["Tip 1."]},
        "fun_facts": ["Fact 1."],
        "conclusion": "Conclusion."
    }

@pytest.fixture
def extraction_missing():
    return {"introduction": "Intro.", "conclusion": "Conclusion."}

@pytest.fixture
def extraction_output():
    return "Intro.\nTip 1.\nFact 1.\nConclusion."

@pytest.fixture
def extraction_missing_output():
    return "Intro.\nConclusion."

@pytest.fixture
def empty_extraction_output():
    return ""

@pytest.fixture
def llm_output_ai():
    return {
        "topic": "AI",
        "headline": "AI",
        "category": "Tech",
        "subcategory": "AI",
        "introduction": "Intro",
        "key_tips_and_takeaways": {},
        "fun_facts": [],
        "conclusion": "Conclusion",
        "image_url": "url"
    }

@pytest.fixture
def llm_output_home_office():
    return {
        'topic': 'Home Office',
        'subcategory': 'Home Office Design',
        'introduction': 'Intro',
        'key_tips_and_takeaways': {},
        'fun_facts': [],
        'conclusion': 'Conclusion',
        'image_url': None
    }

# --- Recommender mocking fixture (parametrized) ---

@pytest.fixture
def recommender_mocks(monkeypatch, llm_output_ai):
    monkeypatch.setattr(pd, 'read_excel', lambda file: pd.DataFrame({'Subcategory': ['AI'], 'Category': ['Technology']}))
    monkeypatch.setattr('recommendation.generate_perplexity_output', lambda *_: {'choices': [{'message': {'content': ''}}]})
    monkeypatch.setattr('recommendation.verify_and_fix_json', lambda *_: (True, llm_output_ai))
    monkeypatch.setattr('recommendation.get_image', lambda *_: "http://example.com/mock_image.jpg")

@pytest.fixture
def recommender_mocks_general(monkeypatch, llm_output_ai):
    monkeypatch.setattr(pd, 'read_excel', lambda file: pd.DataFrame({'Subcategory': ['Quantum'], 'Category': ['Science']}))
    monkeypatch.setattr('recommendation.generate_perplexity_output', lambda *_: {'choices': [{'message': {'content': ''}}]})
    monkeypatch.setattr('recommendation.verify_and_fix_json', lambda *_: (True, llm_output_ai))
    monkeypatch.setattr('recommendation.get_image', lambda *_: "url")

@pytest.fixture
def recommender_mocks_home_office(monkeypatch, llm_output_home_office):
    monkeypatch.setattr(pd, 'read_excel', lambda file: pd.DataFrame({'Subcategory': ['AI'], 'Category': ['Tech']}))
    monkeypatch.setattr('recommendation.generate_perplexity_output', lambda *_: {'choices': [{'message': {'content': ''}}]})
    monkeypatch.setattr('recommendation.verify_and_fix_json', lambda *_: (True, llm_output_home_office))
    monkeypatch.setattr('recommendation.get_image', lambda *_: "http://example.com/home-office.jpg")

# --- Test Cases use fixtures, now with printouts ---

def test_validate_recommendation_output_valid(valid_output_json):
    desc = "Validates a correct JSON structure for recommendations."
    expected = (True, [])
    actual = validate_recommendation_output(valid_output_json)
    passed = actual == expected
    print_test_detail("validate_recommendation_output_valid", desc, valid_output_json, expected, actual, passed)
    assert passed

def test_validate_recommendation_output_missing_field(missing_intro_json):
    desc = "Detects missing required fields in the recommendation JSON (missing 'introduction')."
    expected_valid = False
    expected_error = "Missing required field: 'introduction'."
    is_valid, errors = validate_recommendation_output(missing_intro_json)
    passed = (is_valid == expected_valid and expected_error in errors)
    print_test_detail("validate_recommendation_output_missing_field", desc, missing_intro_json, f"{expected_valid}, '{expected_error}' in errors", (is_valid, errors), passed)
    assert passed

def test_validate_recommendation_output_incorrect_type(wrong_type_json):
    desc = "Detects when 'key_tips_and_takeaways' is the wrong data type (should be dict)."
    expected_valid = False
    expected_error = "must be an object"
    is_valid, errors = validate_recommendation_output(wrong_type_json)
    passed = (is_valid == expected_valid and any(expected_error in e for e in errors))
    print_test_detail("validate_recommendation_output_incorrect_type", desc, wrong_type_json, f"{expected_valid}, '{expected_error}' in errors", (is_valid, errors), passed)
    assert passed

def test_validate_recommendation_output_incorrect_nested_type(nested_type_json):
    desc = "Detects nested type errors: tips list should contain only strings, not numbers."
    expected_valid = False
    expected_error = "must be a string, but found int"
    is_valid, errors = validate_recommendation_output(nested_type_json)
    passed = (is_valid == expected_valid and any(expected_error in e for e in errors))
    print_test_detail("validate_recommendation_output_incorrect_nested_type", desc, nested_type_json, f"{expected_valid}, '{expected_error}' in errors", (is_valid, errors), passed)
    assert passed

def test_validate_recommendation_output_not_a_dict():
    desc = "Properly fails when the input is not a dictionary (e.g. is a list)."
    input_val = []
    expected = (False, ["Top-level JSON must be an object."])
    actual = validate_recommendation_output(input_val)
    passed = (actual == expected)
    print_test_detail("validate_recommendation_output_not_a_dict", desc, input_val, expected, actual, passed)
    assert passed

def test_extract_text_from_recommendations_json_success(extraction_input, extraction_output):
    desc = "Extracts and concatenates all available text sections from a valid recommendation."
    actual = extract_text_from_recommendations_json(extraction_input)
    passed = (actual == extraction_output)
    print_test_detail("extract_text_from_recommendations_json_success", desc, extraction_input, extraction_output, actual, passed)
    assert passed

def test_extract_text_from_recommendations_json_with_missing_keys(extraction_missing, extraction_missing_output):
    desc = "Works when optional keys are missing, extracting only what is present ('introduction' and 'conclusion')."
    actual = extract_text_from_recommendations_json(extraction_missing)
    passed = (actual == extraction_missing_output)
    print_test_detail("extract_text_from_recommendations_json_with_missing_keys", desc, extraction_missing, extraction_missing_output, actual, passed)
    assert passed

def test_extract_text_from_recommendations_json_empty_input(empty_json, empty_extraction_output):
    desc = "Handles empty input gracefully (returns empty string)."
    actual = extract_text_from_recommendations_json(empty_json)
    passed = (actual == empty_extraction_output)
    print_test_detail("extract_text_from_recommendations_json_empty_input", desc, empty_json, empty_extraction_output, actual, passed)
    assert passed

def test_generate_recommendation_success(recommender_mocks):
    desc = "Checks recommendation with 'artificial intelligence'; looks up category and sets image URL."
    input_val = "artificial intelligence"
    expected_category = 'Technology'
    expected_image = "http://example.com/mock_image.jpg"
    output = generate_recommendation(input_val)
    actual = (output['category'], output['image_url'])
    expected = (expected_category, expected_image)
    passed = (actual == expected)
    print_test_detail("generate_recommendation_success", desc, input_val, expected, actual, passed)
    assert passed

def test_generate_recommendation_category_fallback(recommender_mocks_general):
    desc = "Checks fallback logic for unknown subcategory (should set category to 'General')."
    input_val = "artificial intelligence"
    expected = "General"
    output = generate_recommendation(input_val)
    actual = output['category']
    passed = (actual == expected)
    print_test_detail("generate_recommendation_category_fallback", desc, input_val, expected, actual, passed)
    assert passed

def test_generate_recommendation_home_office_scenario(recommender_mocks_home_office):
    desc = "Checks home office scenario, ensures fallback to 'General' and special image URL is set."
    input_val = "home office"
    expected = ('General', "http://example.com/home-office.jpg")
    output = generate_recommendation(input_val)
    actual = (output['category'], output['image_url'])
    passed = (actual == expected)
    print_test_detail("generate_recommendation_home_office_scenario", desc, input_val, expected, actual, passed)
    assert passed
