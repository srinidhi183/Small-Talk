from api_utils import *
from llm_utils import *





def search_bar(keywords,  allow_options = True,  prompt_version="v01", categories_file = 'Categories.xlsx', **model_params):
    """
    Retrieves news based on the given `input` keyword(s), constructs a prompt
    for the LLM, requests a JSON response, and verifies the JSON structure.

    Workflow Steps:
        1. Fetch news:
           - Calls `get_news(input, 3)` to retrieve three relevant news items
             related to the user's `input`.
        2. Build prompt:
           - Uses `build_prompt(input, news_text, second=second)` to create
             a context-rich prompt that includes the keywords and summarized
             news. The `second` parameter can be used to alter the prompt logic
             or style if needed.
        3. Get LLM response and fix JSON (up to 3 attempts):
           - Calls `get_llm_response(prompt)` to invoke a language model.
           - Then uses `verify_and_fix_json(llm_response)` to attempt parsing
             the string as valid JSON, applying heuristic fixes if needed.
           - Validates the fixed JSON structure with `validate_json_structure`.
           - Retries up to 3 times if the JSON is invalid.
        4. Return results or error:
           - If validation succeeds within 3 attempts, returns a tuple:
             (parse_ok, final_json), where:
               - `parse_ok` is a boolean indicating whether the original output
                 was valid or required fixes.
               - `final_json` is the parsed-and-verified JSON structure.
           - If validation is not successful after 3 tries, returns
             "The Error is not fixed".

    Args:
        keywords (str):
            The topic or keywords to pass into the prompt and fetch news for.
        second (bool, optional):
            A control flag that can alter how `build_prompt` structures the prompt.
            Defaults to False.
        OPENAI_API_KEY (str, optional):
            API key used for the LLM service. Defaults to a placeholder key.

    Returns:
        tuple or str:
            - If valid JSON is produced and passes schema checks:
                (need_choise, json_output)
                where `need_choise` (bool) indicates whether the JSON was initially valid
                or had to be fixed, and `json_output` is the resulting parsed JSON dict.
            - If the JSON could not be validated in 3 attempts: 
                "The Error is not fixed"


    """
    df = pd.read_excel(categories_file)
    if allow_options:
        prompt_type = 'search_bar'
    else:
        prompt_type = 'search_bar_no_opt'
    df = pd.read_excel(categories_file)
    news_str = generate_search_bar_news_perplexity(keywords)
    # Build the prompt using the keywords and fetched news.
    prompt = build_prompt(prompt_type = prompt_type, 
                          prompt_version = prompt_version,
                            keywords=keywords, 
                            news=news_str, 
                            article_text= "",
                            reference_url = '',
                            subcategories = df['Subcategory'].dropna().tolist(),
                            allow_options = allow_options)
    validated = False
    iterations = 0
    while not validated and iterations != 1:
        # Get the response from the LLM.
        llm_response = get_llm_response(prompt)
        need_choise, json_output = verify_and_fix_json(llm_response)
        print(json_output)
        validated = validate_search_bar_output(json_output)[0]
        iterations += 1
    if validated:
        if not need_choise:
            json_output['general_facts']['picture_url'] = get_image(json_output['topic']['name'].replace(" ", "-"))
            json_output['news']['picture_url'] = get_image(json_output['news']['news_text'][:50].replace(" ", "-"))
            json_output['interesting_trivia']['picture_url'] = get_image(json_output['interesting_trivia']['trivia_text'][:20].replace(" ", "-"))
        return need_choise, json_output
    else:
        return "The Error is not fixed"
  


def validate_search_bar_output(json_data):
    """
    Validate that `json_data` conforms to one of two possible schemas:

    1) Distinct Topic Schema (original detailed structure):
       {
         "further_key_words": str,
         "category": {
           "categorie_name": str
         },
         "subcatetgory": {
           "subcatetgory_name": str
         },
         "topic": {
           "name": str,
           "topic_tags": {
             "tag_names": str
           }
         },
         "general_facts": {
           "general_definition": str,
           "general_points": str,
           "general_fun_fact": str,
           "key_facts_text": str,
           "key_facts_fun_fact": str,
           "picture_url": str
         },
         "news": {
           "news_text": str,
           "news_fun_facts": str,
           "picture_url": str
         },
         "interesting_trivia": {
           "trivia_text": str,
           "trivia_fun_fact": str,
           "picture_url": str
         },
         "opinions": {
           "opinions_text": str,
           "opinions_fun_fact": str
         },
         "questions": {
           "questions_text": str,
           "questions_fun_fact": str
         }
       }

    2) Ambiguous/Collision Schema (multiple "searchN" sub-objects):
       {
         "search1": {
           "key_words": str,
           "category": str,
           "subcatetgory": str
         },
         "search2": {
           "key_words": str,
           "category": str,
           "subcatetgory": str
         },
         ...
       }

    Return: (is_valid, error_messages)
      - is_valid: bool
      - error_messages: a list of strings describing any validation errors.
    """

    errors = []

    # --- HELPER FUNCTIONS ---

    def has_distinct_topic_keys(obj):
        """Check if obj has all top-level keys for the distinct topic schema."""
        required_top_keys = {
            "further_key_words",
            "category",
            "subcatetgory",
            "topic",
            "general_facts",
            "news",
            "interesting_trivia",
            "opinions",
            "questions"
        }
        return required_top_keys.issubset(obj.keys())

    def validate_distinct_topic(obj):
        """Validate the detailed structure if it's a single, distinct topic.
           Return True if valid, otherwise False, and append errors to `errors`.
        """
        if not has_distinct_topic_keys(obj):
            missing = {
                "further_key_words",
                "category",
                "subcatetgory",
                "topic",
                "general_facts",
                "news",
                "interesting_trivia",
                "opinions",
                "questions"
            } - set(obj.keys())
            errors.append(f"Missing required top-level fields for distinct schema: {missing}")
            return False

        # Check data types / sub-objects:

        # category
        if not isinstance(obj["category"], dict):
            errors.append("`category` should be an object.")
            return False
        if "categorie_name" not in obj["category"]:
            errors.append("`category` is missing 'categorie_name' field.")
            return False

        # subcatetgory
        if not isinstance(obj["subcatetgory"], dict):
            errors.append("`subcatetgory` should be an object.")
            return False
        if "subcatetgory_name" not in obj["subcatetgory"]:
            errors.append("`subcatetgory` is missing 'subcatetgory_name' field.")
            return False

        # topic
        if not isinstance(obj["topic"], dict):
            errors.append("`topic` should be an object.")
            return False
        if "name" not in obj["topic"] or "topic_tags" not in obj["topic"]:
            errors.append("`topic` is missing 'name' or 'topic_tags'.")
            return False
        if not isinstance(obj["topic"]["topic_tags"], dict):
            errors.append("`topic.topic_tags` should be an object.")
            return False
        if "tag_names" not in obj["topic"]["topic_tags"]:
            errors.append("`topic.topic_tags` is missing 'tag_names'.")
            return False

        # general_facts
        gf = obj["general_facts"]
        if not isinstance(gf, dict):
            errors.append("`general_facts` should be an object.")
            return False
        gf_required = {"general_definition", "general_points", "general_fun_fact",
                       "key_facts_text", "key_facts_fun_fact", "picture_url"}
        missing_gf = gf_required - set(gf.keys())
        if missing_gf:
            errors.append(f"`general_facts` is missing fields: {missing_gf}")
            return False

        # news
        news_obj = obj["news"]
        if not isinstance(news_obj, dict):
            errors.append("`news` should be an object.")
            return False
        news_required = {"news_text", "news_fun_facts", "picture_url"}
        missing_news = news_required - set(news_obj.keys())
        if missing_news:
            errors.append(f"`news` is missing fields: {missing_news}")
            return False

        # interesting_trivia
        it = obj["interesting_trivia"]
        if not isinstance(it, dict):
            errors.append("`interesting_trivia` should be an object.")
            return False
        it_required = {"trivia_text", "trivia_fun_fact", "picture_url"}
        missing_it = it_required - set(it.keys())
        if missing_it:
            errors.append(f"`interesting_trivia` is missing fields: {missing_it}")
            return False

        # opinions
        opin = obj["opinions"]
        if not isinstance(opin, dict):
            errors.append("`opinions` should be an object.")
            return False
        opin_required = {"opinions_text", "opinions_fun_fact"}
        missing_opin = opin_required - set(opin.keys())
        if missing_opin:
            errors.append(f"`opinions` is missing fields: {missing_opin}")
            return False

        # questions
        quest = obj["questions"]
        if not isinstance(quest, dict):
            errors.append("`questions` should be an object.")
            return False
        quest_required = {"questions_text", "questions_fun_fact"}
        missing_quest = quest_required - set(quest.keys())
        if missing_quest:
            errors.append(f"`questions` is missing fields: {missing_quest}")
            return False

        # If all checks passed:
        return True

    def looks_like_ambiguous_schema(obj):
        """
        Check if the top-level keys look like 'search1', 'search2', ...
        Return True if at least one top-level key matches 'search' pattern,
        and no required top-level fields from the distinct schema are present.
        """
        top_keys = set(obj.keys())

        distinct_keys = {
            "further_key_words", "category", "subcatetgory",
            "topic", "general_facts", "news", "interesting_trivia",
            "opinions", "questions"
        }
        # If there's intersection with distinct schema keys, it's not purely ambiguous
        if distinct_keys.intersection(top_keys):
            return False

        # If no key matches 'search\d*' => not ambiguous
        if not any(re.match(r'^search\d*$', k) for k in top_keys):
            return False

        return True

    def validate_ambiguous_schema(obj):
        """
        Validate the ambiguous/collision schema:
        {
          "search1": {"key_words": ..., "category": ..., "subcatetgory": ...},
          "search2": {...}, ...
        }
        Return True if valid, otherwise False, appending errors as needed.
        """
        for key, val in obj.items():
            # Key must match "search\d*"
            if not re.match(r'^search\d*$', key):
                errors.append(f"Unexpected key at top level: '{key}' (expected 'searchN').")
                return False

            if not isinstance(val, dict):
                errors.append(f"'{key}' must map to an object.")
                return False

            required_subfields = {"key_words", "category", "subcatetgory"}
            diff = required_subfields - set(val.keys())
            if diff:
                errors.append(f"'{key}' is missing required fields: {diff}")
                return False

        return True

    # --- MAIN VALIDATION LOGIC ---

    if not isinstance(json_data, dict):
        errors.append("Top-level JSON must be an object.")
        return False, errors

    # 1) Check if it matches the distinct (single-topic) schema
    if has_distinct_topic_keys(json_data):
        is_valid = validate_distinct_topic(json_data)
        return (is_valid, errors if not is_valid else [])

    # 2) Check if it looks like ambiguous/collision schema
    if looks_like_ambiguous_schema(json_data):
        is_valid = validate_ambiguous_schema(json_data)
        return (is_valid, errors if not is_valid else [])

    # 3) If neither schema passes
    errors.append("JSON does not match single-topic or ambiguous schema.")
    return False, errors

def extract_text_from_search_bar_json(json_data):
    texts = []
    if not isinstance(json_data, dict) or "general_facts" not in json_data: # Only for distinct schema
        return ""
    if json_data.get("general_facts") and isinstance(json_data["general_facts"], dict):
        texts.append(json_data["general_facts"].get("general_definition", ""))
        texts.append(json_data["general_facts"].get("general_points", ""))
        texts.append(json_data["general_facts"].get("key_facts_text", ""))
    if json_data.get("news") and isinstance(json_data["news"], dict): texts.append(json_data["news"].get("news_text", ""))
    if json_data.get("interesting_trivia") and isinstance(json_data["interesting_trivia"], dict): texts.append(json_data["interesting_trivia"].get("trivia_text", ""))
    if json_data.get("opinions") and isinstance(json_data["opinions"], dict): texts.append(json_data["opinions"].get("opinions_text", ""))
    return "\n".join(filter(None, texts)).strip()

