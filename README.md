# Backend Folder Structure & File Descriptions


###  Repository Structure

```text
/SmallTalk
├── recommendation.py
├── trending.py
├── update_me.py
├── search_bar.py
├── prompts/
│   ├── recommendation.txt
│   ├── trending.txt
│   ├── update_me.txt
│   └── search_bar.txt
├── api_utils.py
├── llm_utils.py
├── config.py
├── schemas.py
├── example.py
└── evaluation/
    └── reports/
    ├── pipeline.py
    └── run_evaluation.py
```

### `/SmallTalk/recommendation.py`
- `generate_recommendation`  
  Build the general *Recommendations* response. Receives key words, prompt version (must have default value) and parameters of the model, returns the json response (without any metrics yet).
- `validate_recommendation_output`
  Validate that the TrenRecommendationding JSON conforms to schema.
- `extract_text_from_recommendations_json`

---

### `/SmallTalk/trending.py`
- `generate_trending`
  Build the general *Trending* response. Receives key words, prompt version (must have default value) and parameters of the model, returns the json response (without any metrics yet).
- `validate_trending_output`  
  Validate that the Trending JSON conforms to schema.
- `extract_text_from_update_element_json`

---

### `/SmallTalk/update_me.py`
- `generate_update_me`  
  Build the *UpdateMe* response. Receives key words, prompt version (must have default value) and parameters of the model, returns the json response (without any metrics yet).
- `validate_update_me_output`
  Ensure the UpdateMe output JSON matches schema/constraints.
- `extract_text_from_update_element_json`

---

### `/SmallTalk/search_bar.py`
- `search_bar`
  Builds the *SearchBar* response. Receives key words, prompt version (must have default value) and parameters of the model, returns the json response (without any metrics yet).
- `validate_search_bar_output` 
  Ensure the SearchBar output JSON matches schema constraints. (old validate_json_structure_search_bar)
- `extract_text_from_search_bar_json`


---

### `/SmallTalk/prompts/*.txt`
Plain-text prompt templates:

- `recommendation_v01.txt` – template for Recommendation
- `trending_v01.txt` – template for Trending
- `update_me_v01.txt` – template for UpdateMe
- `search_bar_v01.txt` – template for SearchBar

---

### `/SmallTalk/api_utils.py`
- `get_news` 
  Fetch and return a list of news items for the given key words.

- `get_articles` 
  Fetch and return a list of article metadata for the given query.

- `get_pictures`  
  Fetch and return image URL(s) matching the query.

- `get_urls_from_google`
  Get the urls given the key words and other parameters

- `extract_main_text_from_url`
  Extracts text given the url of the page 


---

### `/SmallTalk/llm_utils.py`

- `get_llm_response`
  Receives prompt and parameters (optional) and returns its raw JSON response.

- `build_prompt`    
  Receives prompt type, prompt vesion, key words, articles and news if needed 
  and returns the final prompt string.

- `verify_and_fix_json`  
  Generic JSON-schema validator. Fixes json if possible
---


### `/SmallTalk/config.py`
API keys

---

### `/SmallTalk/schemas.py`
output schemas

---

### `/SmallTalk/example.py`
example usage

---


### `/SmallTalk/evaluation/pipeline1.py`
All functions and calls associated with the evaluation for Variant 1. 

---


### `/SmallTalk/evaluation/run_evaluation1.ipynb`
automatical evaluation 

---

### `/SmallTalk/evaluation/pipeline2.py`
All functions and calls associated with the evaluation for Variant 2. 

---


### `/SmallTalk/evaluation/run_evaluation2.ipynb`
automatical evaluation 

---

### `/SmallTalk/evaluation/test.xlsx`
examples of the test cases

--

### `/SmallTalk/evaluation/reports/`
folder, where the repots will be saved

