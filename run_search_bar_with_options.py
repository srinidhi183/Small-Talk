# file: tests/run_search_bar_with_options.py
import argparse
import json
import re
from datetime import datetime

# 👉 Adjust this import to match your project layout
# If search_bar is defined in search_bar.py in the same folder, do:
# from search_bar import search_bar
from search_bar import search_bar

MAX_RERUNS = 5

def is_valid_image_url(url: str) -> bool:
    if not isinstance(url, str) or not url:
        return False
    pattern = r'^https?://.+\.(png|jpg|jpeg|gif|webp)(\?.*)?$'
    return re.match(pattern, url.strip(), flags=re.IGNORECASE) is not None

def _is_nonempty_str(x):
    return isinstance(x, str) and x.strip() != ""

def validate_type_a(obj: dict) -> (bool, str):
    # Required top-level keys
    req_keys = [
        "further_key_words", "category", "subcatetgory", "topic",
        "general_facts", "news", "interesting_trivia", "opinions", "questions"
    ]
    for k in req_keys:
        if k not in obj:
            return False, f"Missing top-level key: {k}"

    # further_key_words: 3–5 keywords (light check: contains 2 commas => ~3 items)
    if not _is_nonempty_str(obj["further_key_words"]):
        return False, "further_key_words must be a non-empty string"
    kw_items = [t.strip() for t in re.split(r"[,\|/;]", obj["further_key_words"]) if t.strip()]
    if not (3 <= len(kw_items) <= 5):
        return False, "further_key_words should contain 3–5 terms"

    # category / subcatetgory
    if not (isinstance(obj["category"], dict) and _is_nonempty_str(obj["category"].get("categorie_name", ""))):
        return False, "category.categorie_name missing or empty"
    if not (isinstance(obj["subcatetgory"], dict) and _is_nonempty_str(obj["subcatetgory"].get("subcatetgory_name", ""))):
        return False, "subcatetgory.subcatetgory_name missing or empty"

    # topic
    topic = obj["topic"]
    if not (isinstance(topic, dict) and _is_nonempty_str(topic.get("name", ""))):
        return False, "topic.name missing or empty"
    tags = topic.get("topic_tags", {})
    if not isinstance(tags, dict) or not _is_nonempty_str(tags.get("tag_names", "")):
        return False, "topic.topic_tags.tag_names missing or empty"
    tag_items = [t.strip() for t in re.split(r"[,\|/;]", tags["tag_names"]) if t.strip()]
    if not (3 <= len(tag_items) <= 5):
        return False, "topic.topic_tags.tag_names should contain 3–5 tags"

    # general_facts
    gf = obj["general_facts"]
    if not isinstance(gf, dict):
        return False, "general_facts must be an object"
    if not _is_nonempty_str(gf.get("general_definition", "")):
        return False, "general_facts.general_definition missing or empty"
    gp = gf.get("general_points", {})
    if not isinstance(gp, dict) or len(gp.keys()) != 3:
        return False, "general_facts.general_points must have exactly 3 subheaders"
    # each subheader ≥ 20 words (approx)
    for k, v in gp.items():
        if not _is_nonempty_str(v) or len(v.split()) < 20:
            return False, f"general_points['{k}'] must be ≥ 20 words"
    if not _is_nonempty_str(gf.get("general_fun_fact", "")):
        return False, "general_facts.general_fun_fact missing or empty"
    if not _is_nonempty_str(gf.get("key_facts_text", "")):
        return False, "general_facts.key_facts_text missing or empty"
    # crude check for 3 sentences
    if len([s for s in re.split(r"[.!?]\s", gf["key_facts_text"].strip()) if s]) < 3:
        return False, "general_facts.key_facts_text should have 3 separate sentences"
    if not _is_nonempty_str(gf.get("key_facts_fun_fact", "")):
        return False, "general_facts.key_facts_fun_fact missing or empty"
    if not is_valid_image_url(gf.get("picture_url", "")):
        return False, "general_facts.picture_url must be a valid image URL"

    # news
    nw = obj["news"]
    if not isinstance(nw, dict):
        return False, "news must be an object"
    if not _is_nonempty_str(nw.get("news_text", "")):
        return False, "news.news_text missing or empty"
    # crude check: 3 bullet-like sentences ⇒ at least 3 sentences
    if len([s for s in re.split(r"[.!?]\s", nw["news_text"].strip()) if s]) < 3:
        return False, "news.news_text should have 3 bullet-style sentences"
    if not _is_nonempty_str(nw.get("news_fun_facts", "")):
        return False, "news.news_fun_facts missing or empty"
    if not is_valid_image_url(nw.get("picture_url", "")):
        return False, "news.picture_url must be a valid image URL"

    # interesting_trivia
    it = obj["interesting_trivia"]
    if not isinstance(it, dict):
        return False, "interesting_trivia must be an object"
    if not _is_nonempty_str(it.get("trivia_text", "")):
        return False, "interesting_trivia.trivia_text missing or empty"
    if len([s for s in re.split(r"[.!?]\s", it["trivia_text"].strip()) if s]) < 3:
        return False, "interesting_trivia.trivia_text should have 3 separate sentences"
    if not _is_nonempty_str(it.get("trivia_fun_fact", "")):
        return False, "interesting_trivia.trivia_fun_fact missing or empty"
    if not is_valid_image_url(it.get("picture_url", "")):
        return False, "interesting_trivia.picture_url must be a valid image URL"

    # opinions
    op = obj["opinions"]
    if not isinstance(op, dict) or not _is_nonempty_str(op.get("opinions_text", "")) or not _is_nonempty_str(op.get("opinions_fun_fact", "")):
        return False, "opinions.opinions_text and/or opinions.opinions_fun_fact missing or empty"

    # questions
    qs = obj["questions"]
    if not isinstance(qs, dict) or not _is_nonempty_str(qs.get("questions_text", "")) or not _is_nonempty_str(qs.get("questions_fun_fact", "")):
        return False, "questions.questions_text and/or questions.questions_fun_fact missing or empty"

    return True, "Type A valid"

def validate_type_b(obj: dict) -> (bool, str):
    if not isinstance(obj, dict):
        return False, "Output must be a JSON object"
    expected_toplevel = {"search1", "search2", "search3"}
    if set(obj.keys()) != expected_toplevel:
        return False, f"Top-level keys must be exactly {sorted(expected_toplevel)}"

    for k in ["search1", "search2", "search3"]:
        entry = obj.get(k, {})
        if not isinstance(entry, dict):
            return False, f"{k} must be an object"
        # Required nested keys
        for nk in ["key_words", "category", "subcatetgory"]:
            if nk not in entry:
                return False, f"{k}.{nk} missing"
        if not _is_nonempty_str(entry["key_words"]):
            return False, f"{k}.key_words must be a non-empty string"
        if not _is_nonempty_str(entry["category"]):
            return False, f"{k}.category must be a non-empty string"
        if not _is_nonempty_str(entry["subcatetgory"]):
            return False, f"{k}.subcatetgory must be a non-empty string"
    return True, "Type B valid"

def detect_and_validate(obj: dict) -> (bool, str, str):
    """
    Returns (is_valid, detected_type, reason)
    detected_type in {"A", "B", "?"}
    """
    # Try Type A
    ok_a, reason_a = validate_type_a(obj)
    if ok_a:
        return True, "A", reason_a
    # Try Type B
    ok_b, reason_b = validate_type_b(obj)
    if ok_b:
        return True, "B", reason_b
    # Neither valid — return the most informative reason
    return False, "?", f"Not Type A: {reason_a} | Not Type B: {reason_b}"

def run_once(keywords: str):
    raw = search_bar(keywords, allow_options=True)
    # Ensure it's JSON (some implementations may already return dict)
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw)
        return parsed, None
    except Exception as e:
        return None, f"JSON parse error: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, default="Formula 1, Racing")
    args = parser.parse_args()

    attempts = 0
    passed = False
    detected_type = "?"
    reasons = []
    winning_output = None

    while attempts < MAX_RERUNS:
        attempts += 1
        print(f"\n--- Attempt {attempts} / {MAX_RERUNS} ---")
        obj, err = run_once(args.keywords)
        if err:
            print(f"❌ Output not valid JSON: {err}")
            reasons.append(f"Attempt {attempts}: {err}")
        else:
            ok, dtype, reason = detect_and_validate(obj)
            print(f"Detected Type: {dtype} | Validation: {reason}")
            if ok:
                passed = True
                detected_type = dtype
                winning_output = obj
                break
            else:
                reasons.append(f"Attempt {attempts}: {reason}")

    # Document results
    summary = {
        "keywords": args.keywords,
        "max_reruns_allowed": MAX_RERUNS,
        "attempts_made": attempts,
        "passed": passed,
        "detected_type": detected_type,
        "failure_reasons": reasons if not passed else [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    print("\n=== RERUN SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Save raw output if you want (optional, helpful for QA attachments)
    try:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        with open(f".search_bar_latest_summary_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if winning_output is not None:
            with open(f".search_bar_passing_output_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(winning_output, f, indent=2, ensure_ascii=False)
        print(f"\nFiles written: .search_bar_latest_summary_{ts}.json"
              + (f", .search_bar_passing_output_{ts}.json" if winning_output else ""))
    except Exception as e:
        print(f"(Warning) Could not write output files: {e}")

    # Exit code semantics (0 on pass, 1 on fail) if you call this script in CI
    if not passed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
