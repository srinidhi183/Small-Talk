#recommendation.py

from api_utils import *
from llm_utils import *

import random
from datetime import datetime, timedelta




def generate_recommendation(keywords, prompt_version="v02",  categories_file = 'Categories.xlsx', region = "us", num_results=15, lang="en", **model_params):   
    df = pd.read_excel(categories_file)
    
    #urls = get_urls_from_google(keywords, region = region, num_results=num_results, lang=lang)
    #articles = get_articles(urls)
    #print(len(" ".join(articles)))
    #articles = " ".join(articles)[0:21000]

    date = (datetime.now()- timedelta(days=30)).strftime("%d.%m.%Y")

    system_prompt  = f''' You are the assistant that will help quickly get information on a certain topic for update briefings.
           Output Rules:
              Return the full, detailed JSON object with the following structure:

             {{
              "topic": "{{ Short, high-level title for the topic (e.g., 'Remote Work', 'Electric Vehicles') }}",
              "headline": "{{ Catchy, informative headline summarizing the main point or takeaway }}",
              "category": "{{ Broad category such as Career, Global News, Health, Technology }}",

              "introduction": "{{ Brief introduction or summary paragraph (2–3 sentences) explaining the core idea or problem }}",

              "key_tips_and_takeaways": {{
                "Section 1 Title": [
                  "{{ Tip 1: concise, actionable tip }}",
                  "{{ Tip 2: additional insight or step }}",
                  "{{ Tip 3: optional third suggestion }}"
                ],
                "Section 2 Title": [
                  "{{ Tip or takeaway 1 }}",
                  "{{ Tip or takeaway 2 }}"
                ],
                "Section 3 Title": [
                  "{{ Tip or takeaway 1 }}",
                  "{{ Tip or takeaway 2 }}",
                  "{{ Tip or takeaway 3 }}"
                ],
                "Section 4 Title": [
                  "{{ Tip or takeaway 1 }}",
                  "{{ Tip or takeaway 2 }}"
                ],
                "Section 5 Title": [
                  "{{ Tip or takeaway 1 }}",
                  "{{ Tip or takeaway 2 }}"
                ],
                "Section 6 Title": [
                  "{{ Tip or takeaway 1 }}",
                  "{{ Tip or takeaway 2 }}",
                  "{{ Tip or takeaway 3 }}"
                ],
                "Section 7 Title": [
                  "{{ Tip or takeaway 1 }}",
                  "{{ Tip or takeaways 2 }}"
                ]
              }},

              "fun_facts": [
                "{{ Fun fact 1 related to the topic }}",
                "{{ Fun fact 2 that adds humor, surprise, or insight }}"
              ],

              "conclusion": "{{ Wrap-up paragraph summarizing why this matters and what to keep in mind (2–3 sentences) }}",
              "image_url": "{{ Direct link to an image (JPG/PNG) relevant to the topic }}"
            }}


            Critical Requirements:
            - Always respond in valid JSON only (no extra commentary, no markdown).
            - Do not include any additional text or explanation outside the JSON.

            Be precise and coincise, Dont use Wikipedia or Wikimedia
            The latest news can be from the {date}.

        '''
        

    user_input = build_prompt(prompt_type = 'recommendation', 
                          prompt_version = prompt_version,
                            keywords=keywords, news = '', 
                            article_text= '',#articles,
                            reference_url = '')
    
    #response = get_llm_response(prompt)
    raw_response = generate_perplexity_output(system_prompt, user_input)
    valid, parsed_json = verify_and_fix_json(raw_response['choices'][0]['message']['content'])
    parsed_json['image_url'] = get_image(parsed_json['topic'].replace(" ", "-"))

    category = df[df['Subcategory'] == parsed_json.get('subcategory', '')]['Category'].values[0] if not df[df['Subcategory'] == parsed_json.get('subcategory', '')].empty else 'General'
    parsed_json['category'] = category

    if valid:
        return parsed_json
    else:
        print("The Error is not fixed")




def validate_recommendation_output(output_json):
    """
    Validate that the recommendation JSON conforms to the expected schema.

    Expected Schema:
    {
      "introduction": "string",
      "key_tips_and_takeaways": {
        "category_name": ["string", ...],
        ...
      },
      "fun_facts": ["string", ...],
      "conclusion": "string"
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
    string_fields = ["introduction", "conclusion"]
    for field in string_fields:
        if field not in output_json:
            errors.append(f"Missing required field: '{field}'.")
        elif not isinstance(output_json[field], str):
            errors.append(f"Field '{field}' must be a string, but found {type(output_json[field]).__name__}.")

    # Check for 'key_tips_and_takeaways' field (dictionary of lists of strings)
    if "key_tips_and_takeaways" not in output_json:
        errors.append("Missing required field: 'key_tips_and_takeaways'.")
    elif not isinstance(output_json["key_tips_and_takeaways"], dict):
        errors.append(f"Field 'key_tips_and_takeaways' must be an object, but found {type(output_json['key_tips_and_takeaways']).__name__}.")
    else:
        for category, tips_list in output_json["key_tips_and_takeaways"].items():
            if not isinstance(tips_list, list):
                errors.append(f"Field 'key_tips_and_takeaways.{category}' must be a list, but found {type(tips_list).__name__}.")
            else:
                for i, tip in enumerate(tips_list):
                    if not isinstance(tip, str):
                        errors.append(f"Element at index {i} in list 'key_tips_and_takeaways.{category}' must be a string, but found {type(tip).__name__}.")
                        break # Stop checking this list after the first error

    # Check for 'fun_facts' field (list of strings)
    if "fun_facts" not in output_json:
        errors.append("Missing required field: 'fun_facts'.")
    elif not isinstance(output_json["fun_facts"], list):
        errors.append(f"Field 'fun_facts' must be a list, but found {type(output_json['fun_facts']).__name__}.")
    else:
        for i, fact in enumerate(output_json["fun_facts"]):
            if not isinstance(fact, str):
                errors.append(f"Element at index {i} in list 'fun_facts' must be a string, but found {type(fact).__name__}.")
                break # Stop checking this list after the first error

    is_valid = len(errors) == 0
    return is_valid, errors


def extract_text_from_recommendations_json(json_data):
    texts = []
    if not isinstance(json_data, dict): return ""
    texts.append(json_data.get("introduction", ""))
    key_tips = json_data.get("key_tips_and_takeaways", {})
    if isinstance(key_tips, dict):
        for _, tips_list in key_tips.items():
            if isinstance(tips_list, list): texts.extend(str(tip) for tip in tips_list)
    fun_facts = json_data.get("fun_facts", [])
    if isinstance(fun_facts, list): texts.extend(str(fact) for fact in fun_facts)
    texts.append(json_data.get("conclusion", ""))
    return "\n".join(filter(None, texts)).strip()

