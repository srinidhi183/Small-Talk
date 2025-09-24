#trending.py
from api_utils import *
from llm_utils import *
from datetime import datetime, timedelta


def generate_trending_element(keywords, prompt_version="v02", categories_file='Categories.xlsx', **model_params):
    df = pd.read_excel(categories_file)

    date = (datetime.now() - timedelta(days=30)).strftime("%d.%m.%Y")

    system_prompt = f''' You are the assistant that will help quickly get exhaustive information on a certain trending topic for update briefings.
           Output Rules:
              Return the full, detailed JSON object with the following structure:

              {{
      "topic": "<Short title or topic>",
      "category": "<Placeholder>",
      "subcategory": "<Subcategory, e.g., NFL, Politics, Tech>",
      "description": "<A brief description of the trending topic>",
      "why_is_it_trending": {{"Subheader 1": "<Explanation of why it's trending>",
                              "Subheader 2": "<Explanation of why it's trending>",
                              "Subheader 3": "<Explanation of why it's trending>"}},
      "key_points": {{"Subheader 1": "<Key detail>",
                      "Subheader 2": "<Key detail>",
                      "Subheader 3": "<Key detail>"}},
      "overlook_what_might_happen_next": {{"Subheader 1": "<Outlook or prediction>",
                                          "Subheader 2": "<Outlook or prediction>",
                                          "Subheader 3": "<Outlook or prediction>"}},
      "picture_url": "placeholder"
    }}

            Critical Requirements:
            - Always respond in valid JSON only (no extra commentary, no markdown).
            - Do not include any additional text or explanation outside the JSON.

            Be precise and coincise, Dont use Wikipedia or Wikimedia
            The latest news can be from the {date}.
        '''

    user_input = build_prompt(
        prompt_type='trending',
        prompt_version=prompt_version,
        keywords=keywords, news="",
        article_text='',
        reference_url='',
        extra_links="",
        subcategories=df['Subcategory'].dropna().tolist()
    )

    raw_response = generate_perplexity_output(system_prompt, user_input)
    valid, parsed_json = verify_and_fix_json(raw_response['choices'][0]['message']['content'])

    # ⛑️ Early bails
    if not valid or not isinstance(parsed_json, dict):
        print("Could not parse/fix the JSON response.")
        return None
    topic = str(parsed_json.get("topic", "")).strip()
    if not topic:
        print("Missing 'topic' in parsed JSON.")
        return None

    # 🔧 NORMALIZE: convert dict-with-subheaders → list[str]; coerce numbers; drop empties
    def _to_str_list(v):
        if isinstance(v, dict):
            vals = v.values()          # preserve insertion order
        elif isinstance(v, list):
            vals = v
        else:
            return []
        out = []
        for x in vals:
            if isinstance(x, (int, float)):
                x = str(x)
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out

    for fld in ("why_is_it_trending", "key_points", "overlook_what_might_happen_next"):
        parsed_json[fld] = _to_str_list(parsed_json.get(fld))

    # 🖼️ Fetch image and enforce http(s) fallback
    img_url = get_image(topic.replace(" ", "-"))
    def _is_http(u): 
         return isinstance(u, str) and (u.startswith("http://") or u.startswith("https://"))

    if not _is_http(img_url):
    # Stable, public placeholder image; includes the topic as text
       from urllib.parse import quote
       img_url = f"https://placehold.co/1200x628?text={quote(topic)}"

    parsed_json["picture_url"] = img_url


    # 🗂️ Map subcategory → category (fallback to "General")
    subcat = str(parsed_json.get("subcategory", "")).strip()
    if subcat and not df[df["Subcategory"] == subcat].empty:
        category = df.loc[df["Subcategory"] == subcat, "Category"].iloc[0]
    else:
        category = "General"
    parsed_json["category"] = category

    return parsed_json


def generate_trending_set(keywords = None , num_elements=5, prompt_version="v02", categories_file = 'Categories.xlsx', **model_params):

    if not keywords:
        keywords = get_trending_topics()
    trending = []
    print(keywords)

    for trend in keywords:
        print(f'generating trend for: {trend}')

        trend_json = generate_trending_element(trend, prompt_version=prompt_version, categories_file=categories_file, **model_params)
        if trend_json:
            trending.append(trend_json)



    return trending

def validate_trending_output(output_json):
    """
    Validate that the trending JSON conforms to the expected schema.

    Expected Schema:
    {
      "topic": "string",
      "category": "string",
      "subcategory": "string",
      "description": "string",
      "why_is_it_trending": ["string", ...],
      "key_points": ["string", ...],
      "overlook_what_might_happen_next": ["string", ...]
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
    string_fields = ["topic", "category", "subcategory", "description"]
    for field in string_fields:
        if field not in output_json:
            errors.append(f"Missing required field: '{field}'.")
        elif not isinstance(output_json[field], str):
            errors.append(f"Field '{field}' must be a string, but found {type(output_json[field]).__name__}.")

    # Check for required list of strings fields
    list_fields = ["why_is_it_trending", "key_points", "overlook_what_might_happen_next"]
    for field in list_fields:
        if field not in output_json:
            errors.append(f"Missing required field: '{field}'.")
        elif not isinstance(output_json[field], list):
            errors.append(f"Field '{field}' must be a list, but found {type(output_json[field]).__name__}.")
        else:
            # Check if all elements in the list are strings
            for i, item in enumerate(output_json[field]):
                if not isinstance(item, str):
                    errors.append(f"Element at index {i} in list '{field}' must be a string, but found {type(item).__name__}.")
                    # Stop checking this list after the first error
                    break 

    is_valid = len(errors) == 0
    return is_valid, errors


# --- replace the whole function in trending.py with this ---
def extract_text_from_trending_now_json(json_data):
    """
    Build a newline-joined summary from a trending JSON-like dict.
    - Coerces ints/floats to strings
    - Ignores None and whitespace-only entries
    - Safe with partially filled payloads
    """
    if not isinstance(json_data, dict):
        return ""

    def _coerce_keep(val):
        # return cleaned string or None to drop
        if isinstance(val, (int, float)):
            val = str(val)
        if isinstance(val, str):
            val = val.strip()
            return val if val else None
        return None

    texts = []
    # core fields in order
    texts.append(json_data.get("topic", ""))
    texts.append(json_data.get("category", ""))
    texts.append(json_data.get("subcategory", ""))
    texts.append(json_data.get("description", ""))

    # list-like sections
    for key in ("why_is_it_trending", "key_points", "overlook_what_might_happen_next"):
        v = json_data.get(key)
        if isinstance(v, list):
            texts.extend(v)

    # sanitize + join
    cleaned = []
    for t in texts:
        ct = _coerce_keep(t)
        if ct is not None:
            cleaned.append(ct)

    return "\n".join(cleaned)
