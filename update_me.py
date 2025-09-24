#update_me.py
from api_utils import *
from llm_utils import *

import pandas as pd
import random
from datetime import datetime, timedelta



def generate_update_me_element(
    keywords,
    link=None,
    news_text=None,
    image_url=None,
    prompt_type="update_me_news",
    categories_file="Categories.xlsx",
    prompt_version="v01",
    **model_params,
):
    """
    Generates a structured update element in JSON format using OpenAI's API and the tailored prompt.
    """
    df = pd.read_excel(categories_file)

    if prompt_type == "update_me_general":
        user_input = build_prompt(
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            keywords=keywords,
            news="",
            article_text="",
            reference_url=link,
            subcategories=df["Subcategory"].dropna().tolist(),
        )

        date = (datetime.now() - timedelta(days=30)).strftime("%d.%m.%Y")

        system_prompt = f''' You are the assistant that will help quickly get information on a certain topic for update briefings.
           Output Rules:
              Return the full, detailed JSON object with the following structure:

              {{
                "category": "<Category, e.g., Global News, Personal Interest, Global Development>",
                "subcategory": "<Subcategory, e.g., NFL, Politics, Tech>",
                "topic": "<Short title of the update>",
                "short_info": [
                    "<Bullet point 1: main summary ( 2 sentences)>",
                    "<Bullet point 2: supporting detail (2 sentences)>",
                    "<Bullet point 3: additional development ( 2 sentences)>"
                ],
                "reference_url": "<URL of the news article, must point to the page with full text>",
                "background_story": {{
                    "context_history": "<Brief context or history (1-2 sentences)>",
                    "current_developments": "<Current developments (1-2 sentences)>",
                    "relevance": "<Why it matters (1-2 sentences)>"
                }},
                "picture_url": <valid url of the picture>
              }}

            Critical Requirements:
            - Always respond in valid JSON only (no extra commentary, no markdown).
            - Do not include any additional text or explanation outside the JSON.

            Be precise and concise, Dont use Wikipedia or Wikimedia
            The latest news can be from the {date}.
        '''

        raw_response = generate_perplexity_output(
            system_content=system_prompt, user_content=user_input
        )
        valid, parsed_json = verify_and_fix_json(
            raw_response["choices"][0]["message"]["content"]
        )
        if not valid:
            print("Could not parse/fix the JSON response.")
            return None

        parsed_json["picture_url"] = (
            get_image(parsed_json.get("topic", "").replace(" ", "-"))
            if parsed_json.get("topic")
            else ""
        )

    else:
        prompt = build_prompt(
            prompt_type=prompt_type,
            prompt_version=prompt_version,
            keywords=keywords,
            news=news_text,
            article_text="",
            reference_url=link,
            subcategories=df["Subcategory"].dropna().tolist(),
        )

        raw_response = get_llm_response(prompt)
        valid, parsed_json = verify_and_fix_json(raw_response)
        if not valid:
            print("Could not parse/fix the JSON response.")
            return None

        parsed_json["picture_url"] = (
            get_image(parsed_json.get("topic", "").replace(" ", "-"))
            if parsed_json.get("topic")
            else ""
        )

    # Handle categories and subcategories
    if prompt_type == "update_me_general":
        parsed_json["subcategory"] = keywords
        category = (
            df[df["Subcategory"] == keywords]["Category"].values[0]
            if not df[df["Subcategory"] == keywords].empty
            else "General"
        )
    else:
        category = (
            df[df["Subcategory"] == parsed_json.get("subcategory", "")]["Category"].values[0]
            if not df[df["Subcategory"] == parsed_json.get("subcategory", "")].empty
            else "General"
        )

    parsed_json["category"] = category

    return parsed_json



def get_random_subtopics_for_update_me(n, categories_file = 'Categories.xlsx'):
    """
    Returns a list of n random subcategories from the specified categories file.
    
    Args:
        n (int): Number of random subtopics to return.
        categories_file (str): Path to the Excel file containing categories.
        
    Returns:
        List[str]: A list of n random subcategories.
    """

    
    # Load the categories from the Excel file
    df = pd.read_excel(categories_file)
    
    # Assuming the subtopics are in a column named 'Subtopic'
    subtopics = df['Subcategory'].dropna().tolist()
    
    # Select n random subtopics
    return random.sample(subtopics, min(n, len(subtopics)))


def generate_update_me_news_set(use_tfidf=True, sort_by_date=False, categories_file = 'Categories.xlsx', prompt_version='v01', **model_params):
    """
    Generates a set of update elements for various subtopics.
    
    Returns:
        List[dict]: A list of dictionaries, each containing an update element.
    """
    subtopics = get_random_subtopics_for_update_me(2)
    top_news = get_top_news_full_json(top_n=3, use_tfidf=use_tfidf, sort_by_date=sort_by_date)
    
    update_me = []

    for news in top_news:
        print(news)
        update_me.append(generate_update_me_element(keywords = news['title'], 
                                                    link = news['link'], 
                                                    news_text = news['full_text'], 
                                                    image_url = news['image'], 
                                                    prompt_type = 'update_me_news', 
                                                    categories_file = categories_file,  
                                                    prompt_version=prompt_version, 
                                                    **model_params)
    )
    

    
    return update_me


def validate_update_me_output(output_json):
    """
    Validate that the UpdateMe output JSON conforms to the expected schema.

    Expected Schema:
    {
      "category": "string",
      "subcategory": "string",
      "topic": "string",
      "short_info": ["string", ...],
      "background_story": {
        "context_history": "string",
        "current_developments": "string",
        "relevance": "string"
      }
    }

    Args:
        output_json: The JSON object to validate.

    Returns:
        tuple: A tuple containing:
            - is_valid (bool): True if the JSON is valid, False otherwise.
            - errors (list): A list of strings describing any validation errors.
    """
    errors = []

    if not isinstance(output_json, dict):
        errors.append("Top-level JSON must be an object.")
        return False, errors

    # Check for required string fields
    string_fields = ["category", "subcategory", "topic"]
    for field in string_fields:
        if field not in output_json:
            errors.append(f"Missing required field: '{field}'.")
        elif not isinstance(output_json[field], str):
            errors.append(f"Field '{field}' must be a string, but found {type(output_json[field]).__name__}.")

    # Check for 'short_info' field (list of strings)
    if "short_info" not in output_json:
        errors.append("Missing required field: 'short_info'.")
    elif not isinstance(output_json["short_info"], list):
        errors.append(f"Field 'short_info' must be a list, but found {type(output_json['short_info']).__name__}.")
    else:
        for i, item in enumerate(output_json["short_info"]):
            if not isinstance(item, str):
                errors.append(f"Element at index {i} in list 'short_info' must be a string, but found {type(item).__name__}.")
                # Stop checking this list after the first error to avoid redundant messages
                break 

    # Check for 'background_story' field (dictionary with string fields)
    if "background_story" not in output_json:
        errors.append("Missing required field: 'background_story'.")
    elif not isinstance(output_json["background_story"], dict):
        errors.append(f"Field 'background_story' must be an object, but found {type(output_json['background_story']).__name__}.")
    else:
        bg_story = output_json["background_story"]
        bg_string_fields = ["context_history", "current_developments", "relevance"]
        for field in bg_string_fields:
            if field not in bg_story:
                errors.append(f"Missing required field in 'background_story': '{field}'.")
            elif not isinstance(bg_story[field], str):
                errors.append(f"Field 'background_story.{field}' must be a string, but found {type(bg_story[field]).__name__}.")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def extract_text_from_update_element_json(json_data):
    texts = []
    if not isinstance(json_data, dict):
        return ""

    def _append_str(x):
        if isinstance(x, str) and x:
            texts.append(x)

    # top-level fields
    _append_str(json_data.get("category", ""))
    _append_str(json_data.get("subcategory", ""))
    _append_str(json_data.get("topic", ""))

    # short_info list
    si = json_data.get("short_info")
    if isinstance(si, list):
        for item in si:
            _append_str(item)

    # background_story dict
    bg = json_data.get("background_story")
    if isinstance(bg, dict):
        _append_str(bg.get("context_history", ""))
        _append_str(bg.get("current_developments", ""))
        _append_str(bg.get("relevance", ""))

    return "\n".join(texts).strip()
