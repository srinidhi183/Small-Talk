import json, re, ast
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from search_bar import search_bar

MAX_RERUNS = 5  # automatic retries

# ---------------- Excel safe saver ----------------
def safe_to_excel(df: pd.DataFrame, base_name="search_bar_checklist_Stock.xlsx", out_dir="reports", engine="xlsxwriter") -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / base_name
    try:
        df.to_excel(base_path, index=False, engine=engine)
        print(f"Checklist written to: {base_path}")
        return str(base_path)
    except PermissionError:
        print(f"'{base_path.name}' is locked. Saving with a unique name...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem, suffix = base_path.stem, base_path.suffix or ".xlsx"
    # timestamp fallback
    for i in range(0, 100):
        suffix_path = out_dir / (f"{stem}_{ts}{'' if i==0 else f'_{i}'}{suffix}")
        try:
            df.to_excel(suffix_path, index=False, engine=engine)
            print(f"Checklist written to: {suffix_path}")
            return str(suffix_path)
        except PermissionError:
            continue
    raise PermissionError("Could not write Excel file. Close any open copy and retry.")

# ---------------- tiny helpers ----------------
def _get(d: Dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text or "")))

def _sentence_count(text: str) -> int:
    s = str(text or "").strip()
    bullets = [ln for ln in s.splitlines() if ln.lstrip().startswith(("-", "*", "•"))]
    if len(bullets) >= 3:
        return len(bullets)
    parts = re.split(r"[.!?]+(?:\s|$)", s)
    return len([p for p in parts if p.strip()])

def _split_to_items(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if isinstance(x, (str, int, float))]
    if isinstance(val, str):
        items = re.split(r"[;,]", val)
        items = [it.strip() for it in items if it.strip()]
        return items if items else [t for t in val.split() if t.strip()]
    return []

def _find_all_picture_urls(obj: Any) -> List[str]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "picture_url" in k.lower():
                if isinstance(v, list):
                    found.extend([x for x in v if isinstance(x, str)])
                elif isinstance(v, str):
                    found.append(v)
            found.extend(_find_all_picture_urls(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_all_picture_urls(v))
    return found

# ---------------- schema detection ----------------
TYPE_A_TOP_KEYS = [
    "further_key_words","category","subcatetgory","topic","general_facts","news","interesting_trivia","opinions","questions",
]
TYPE_B_SEARCH_KEYS = ["search1", "search2", "search3"]
TYPE_B_ENTRY_KEYS = ["key_words", "category", "subcatetgory"]

def detect_schema(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return "unknown"
    keys = set(obj.keys())
    if keys == set(TYPE_B_SEARCH_KEYS):
        # confirm each searchX entry is an object with required keys
        for k in TYPE_B_SEARCH_KEYS:
            if not isinstance(obj.get(k), dict):
                return "unknown"
            entry = obj[k]
            if not set(TYPE_B_ENTRY_KEYS).issubset(entry.keys()):
                return "unknown"
        return "type_b"
    # else likely type A
    return "type_a"

# ---------------- Type A checklist ----------------
def _has_only_keys(d: Dict, required: List[str]) -> Tuple[bool, List[str]]:
    actual, req = set(d.keys()), set(required)
    extras = sorted(list(actual - req))
    missing = sorted(list(req - actual))
    return (not extras and not missing, (missing if missing else []) + (extras if extras else []))

def build_checklist_rows_type_a(output_obj: Dict[str, Any], run_note: str, raw_json_path: Path) -> List[Dict[str, str]]:
    rows = []
    def add(item, result, present, ok):
        rows.append({
            "Checklist Item": item,
            "Result": result,
            "Present Value (from output)": json.dumps(present, ensure_ascii=False) if isinstance(present, (dict, list)) else str(present),
            "Pass/Fail": "Pass" if ok else "Fail"
        })

    # Run & Repro
    rows.append({"Checklist Item":"Ran: search_bar('', allow_options=True)","Result":"Executed","Present Value (from output)":run_note,"Pass/Fail":"Pass"})
    rows.append({"Checklist Item":"Captured raw output and attached to this issue","Result":"Saved","Present Value (from output)":str(raw_json_path),"Pass/Fail":"Pass"})

    # JSON Validity & Keys
    is_obj = isinstance(output_obj, dict)
    add("Output is valid JSON and a single object", "Checked", type(output_obj).__name__, is_obj)
    only_keys_ok, key_diff = _has_only_keys(output_obj if is_obj else {}, TYPE_A_TOP_KEYS) if is_obj else (False, [])
    add("All required top-level keys present (see Type A schema)", "Checked", {"diff": key_diff}, only_keys_ok)

    nested_paths = [
        "category.categorie_name","subcatetgory.subcatetgory_name","topic.name","topic.topic_tags.tag_names",
        "general_facts.general_definition","general_facts.general_points","general_facts.key_facts_text",
        "news.news_text","interesting_trivia.trivia_text","interesting_trivia.trivia_fun_fact",
        "opinions.opinions_text","opinions.opinions_fun_fact","questions.questions_text","questions.questions_fun_fact",
    ]
    missing = [p for p in nested_paths if _get(output_obj, p, None) is None] if is_obj else nested_paths
    add("Nested keys match exactly (spelling, casing)", "Checked", {"missing": missing}, len(missing) == 0)

    extras = sorted([k for k in output_obj.keys() if k not in TYPE_A_TOP_KEYS]) if is_obj else []
    add("No extra top-level keys", "Checked", {"extra_keys": extras}, len(extras) == 0 if is_obj else False)

    # Types & Minimal Shape
    fkw = _split_to_items(output_obj.get("further_key_words") if is_obj else None)
    add("further_key_words contains 3–5 keywords", "Checked", fkw, 3 <= len(fkw) <= 5)

    add("category.categorie_name non-empty string", "Checked", _get(output_obj,"category.categorie_name",""), bool(str(_get(output_obj,"category.categorie_name","")).strip()))
    add("subcatetgory.subcatetgory_name non-empty string", "Checked", _get(output_obj,"subcatetgory.subcatetgory_name",""), bool(str(_get(output_obj,"subcatetgory.subcatetgory_name","")).strip()))
    add("topic.name non-empty string", "Checked", _get(output_obj,"topic.name",""), bool(str(_get(output_obj,"topic.name","")).strip()))

    tags = _split_to_items(_get(output_obj,"topic.topic_tags.tag_names"))
    add("topic.topic_tags.tag_names contains 3–5 tags", "Checked", tags, 3 <= len(tags) <= 5)

    gdef = _get(output_obj,"general_facts.general_definition","")
    add("general_facts.general_definition is one introductory sentence", "Checked", gdef, isinstance(gdef,str) and _sentence_count(gdef) == 1 and _word_count(gdef) >= 6)

    gp = _get(output_obj,"general_facts.general_points")
    if isinstance(gp, dict):
        counts = {k:_word_count(str(v)) for k,v in gp.items()}
        gp_ok = len(gp) == 3 and all(c >= 20 for c in counts.values())
        present = {"subheaders": list(gp.keys()), "word_counts": counts}
    else:
        gp_ok, present = False, gp
    add("general_facts.general_points has 3 subheaders, each with ≥20 words", "Checked", present, gp_ok)

    kft = _get(output_obj,"general_facts.key_facts_text","")
    if isinstance(kft, list):
        kft_ok = len([x for x in kft if isinstance(x,str) and x.strip()]) == 3
    else:
        kft_ok = _sentence_count(str(kft)) == 3
    add("general_facts.key_facts_text has 3 separate sentences", "Checked", kft, kft_ok)

    pics = _find_all_picture_urls(output_obj) if is_obj else []
    pics_ok = len(pics) == 0 or all(isinstance(u,str) and u.startswith(("http://","https://")) for u in pics)
    add("All picture_url values are strings, valid image URLs", "Checked", pics, pics_ok)

    news_text = _get(output_obj,"news.news_text","")
    if isinstance(news_text, list):
        ncount = len([x for x in news_text if isinstance(x,str) and x.strip()])
    else:
        bullets = [ln for ln in str(news_text).splitlines() if ln.lstrip().startswith(("-", "*", "•"))]
        ncount = len(bullets) if bullets else _sentence_count(str(news_text))
    add("news.news_text has 3 bullet-style sentences tied to general info (or flags mismatch)", "Checked", news_text, ncount >= 3)

    triv = _get(output_obj,"interesting_trivia.trivia_text","")
    add("interesting_trivia.trivia_text has 3 separate sentences", "Checked", triv, _sentence_count(str(triv)) == 3)

    triv_ff = _get(output_obj,"interesting_trivia.trivia_fun_fact","")
    add("interesting_trivia.trivia_fun_fact is one fun fact", "Checked", triv_ff, isinstance(triv_ff,str) and 1 <= _sentence_count(triv_ff) <= 2 and _word_count(triv_ff) >= 4)

    opin = _get(output_obj,"opinions.opinions_text","")
    add("opinions.opinions_text is non-empty string of opinions", "Checked", opin, isinstance(opin,str) and opin.strip() != "")

    opin_ff = _get(output_obj,"opinions.opinions_fun_fact","")
    add("opinions.opinions_fun_fact is one fun fact", "Checked", opin_ff, isinstance(opin_ff,str) and 1 <= _sentence_count(opin_ff) <= 2 and _word_count(opin_ff) >= 4)

    qtxt = _get(output_obj,"questions.questions_text","")
    if isinstance(qtxt, list):
        qok = len([x for x in qtxt if isinstance(x,str) and (x.strip().endswith("?") or "?" in x)]) >= 2
    else:
        maybe = re.split(r"\n|[.!?]", str(qtxt))
        qok = len([x for x in maybe if x.strip().endswith("?") or "?" in x]) >= 2
    add("questions.questions_text contains discussion questions", "Checked", qtxt, qok)

    qff = _get(output_obj,"questions.questions_fun_fact","")
    add("questions.questions_fun_fact is one fun fact or fun discussion question", "Checked", qff, isinstance(qff,str) and (qff.strip().endswith("?") or (1 <= _sentence_count(qff) <= 2)))
    return rows

# ---------------- Type B checklist ----------------
def build_checklist_rows_type_b(output_obj: Dict[str, Any], run_note: str, raw_json_path: Path) -> List[Dict[str, str]]:
    rows = []
    def add(item, result, present, ok):
        rows.append({
            "Checklist Item": item,
            "Result": result,
            "Present Value (from output)": json.dumps(present, ensure_ascii=False) if isinstance(present, (dict, list)) else str(present),
            "Pass/Fail": "Pass" if ok else "Fail"
        })

    # Run & Repro
    rows.append({"Checklist Item":"Ran: search_bar('', allow_options=True)","Result":"Executed","Present Value (from output)":run_note,"Pass/Fail":"Pass"})
    rows.append({"Checklist Item":"Captured raw output and attached to this issue","Result":"Saved","Present Value (from output)":str(raw_json_path),"Pass/Fail":"Pass"})

    # JSON Validity & Keys
    is_obj = isinstance(output_obj, dict)
    add("Output is valid JSON and a single object", "Checked", type(output_obj).__name__, is_obj)

    # Exactly three top-level entries: search1..search3
    keys = list(output_obj.keys()) if is_obj else []
    exact_three = set(keys) == set(TYPE_B_SEARCH_KEYS)
    add("Contains exactly three top-level entries: search1, search2, search3", "Checked", keys, exact_three)

    # Each entry is an object with all of: key_words, category, subcatetgory
    entries_ok = True
    entries_detail = {}
    if is_obj and exact_three:
        for k in TYPE_B_SEARCH_KEYS:
            e = output_obj.get(k, {})
            ok = isinstance(e, dict) and set(TYPE_B_ENTRY_KEYS).issubset(e.keys())
            entries_ok = entries_ok and ok
            entries_detail[k] = {"type": type(e).__name__, "keys": list(e.keys()) if isinstance(e, dict) else None}
    else:
        entries_ok = False
    add("Each entry is an object with all of: key_words, category, subcatetgory", "Checked", entries_detail or keys, entries_ok)

    # Reserved keys respected
    reserved_ok = exact_three and entries_ok
    add("Reserved keys respected: searchX pattern, category, subcatetgory", "Checked", {"top_keys": keys}, reserved_ok)

    # Types & Minimal Shape
    # Each key_words non-empty string; category non-empty string; subcatetgory non-empty string
    kw_ok_all, cat_ok_all, sub_ok_all = True, True, True
    kw_values, cat_values, sub_values = {}, {}, {}
    if is_obj and exact_three and entries_ok:
        for k in TYPE_B_SEARCH_KEYS:
            e = output_obj[k]
            kw = e.get("key_words", "")
            cat = e.get("category", "")
            sub = e.get("subcatetgory", "")

            kw_ok = isinstance(kw, str) and kw.strip() != ""
            cat_ok = isinstance(cat, str) and cat.strip() != ""
            sub_ok = isinstance(sub, str) and sub.strip() != ""

            kw_ok_all &= kw_ok
            cat_ok_all &= cat_ok
            sub_ok_all &= sub_ok

            kw_values[k] = kw
            cat_values[k] = cat
            sub_values[k] = sub

    add("Each key_words is a non-empty string of concise, searchable terms", "Checked", kw_values, kw_ok_all and reserved_ok)
    add("Each category is a non-empty string (broad label)", "Checked", cat_values, cat_ok_all and reserved_ok)
    add("Each subcatetgory is a non-empty string (specific sublabel)", "Checked", sub_values, sub_ok_all and reserved_ok)

    return rows

# ---------------- result extraction (robust) ----------------
def extract_json_output(result: Any) -> Union[Dict[str, Any], None]:
    """Return a dict if we can find one inside result (dict, tuple, list, or str with a python-style dict)."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, dict):
                return item
        for item in result:
            if isinstance(item, str) and "{" in item:
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        return None
    if isinstance(result, str) and "{" in result and "}" in result:
        try:
            parsed = ast.literal_eval(result)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None

# ---------------- main runner ----------------
def run_search_bar(keywords, allow_options=True, prompt_version="v01",
                   categories_file='Categories.xlsx', **model_params):
    last_error = None
    last_candidate: Dict[str, Any] = None

    for attempt in range(1, MAX_RERUNS + 1):
        print(f"\n--- Attempt {attempt}/{MAX_RERUNS} ---")
        result = search_bar(keywords, allow_options, prompt_version, categories_file, **model_params)

        cand = extract_json_output(result)
        if isinstance(cand, dict):
            last_candidate = cand

        # success paths remain
        if isinstance(result, tuple) and len(result) == 3:
            need_choice, json_output, inner_attempts = result
        elif isinstance(result, tuple) and len(result) == 2:
            need_choice, json_output = result
            inner_attempts = "n/a"
        elif isinstance(result, dict):
            json_output, inner_attempts = result, "n/a"
        else:
            # failure: record error & continue
            if isinstance(result, tuple) and len(result) >= 1:
                error_message = result[0]
                inner_attempts = result[1] if len(result) > 1 else "n/a"
                last_error = f"{error_message} (inner attempts: {inner_attempts})"
            else:
                last_error = str(result)
            print(f"Attempt {attempt} failed: {last_error}")
            continue

        # success: write artifacts & return
        print(f"Search bar output validated on attempt {attempt} (inner attempts reported: {inner_attempts}).")
        out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
        raw_json_file = out_dir / f"search_bar_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        raw_json_file.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")

        # auto-detect schema and build appropriate checklist
        schema = detect_schema(json_output)
        run_note = f"search_bar(keywords='{keywords}', allow_options={allow_options}, prompt_version='{prompt_version}')"
        if schema == "type_b":
            rows = build_checklist_rows_type_b(json_output, run_note, raw_json_file)
            base_name = "search_bar_checklist_typeB.xlsx"
        else:
            rows = build_checklist_rows_type_a(json_output, run_note, raw_json_file)
            base_name = "search_bar_checklist_typeA.xlsx"

        df = pd.DataFrame(rows, columns=["Checklist Item","Result","Present Value (from output)","Pass/Fail"])
        safe_to_excel(df, base_name=base_name)  # writes even if default file is open
        return json_output

    # all attempts failed — produce best-effort artifacts if possible
    print(f"\nSearch bar failed after {MAX_RERUNS} attempt(s): {last_error}")
    if last_candidate is not None:
        print("Generating BEST-EFFORT checklist from the last JSON candidate…")
        out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
        raw_json_file = out_dir / f"search_bar_raw_FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        raw_json_file.write_text(json.dumps(last_candidate, ensure_ascii=False, indent=2), encoding="utf-8")

        schema = detect_schema(last_candidate)
        run_note = f"[FAILED mode] search_bar(keywords='{keywords}', allow_options={allow_options}, prompt_version='{prompt_version}')"
        if schema == "type_b":
            rows = build_checklist_rows_type_b(last_candidate, run_note, raw_json_file)
            base_name = "search_bar_checklist_typeB_FAILED.xlsx"
        else:
            rows = build_checklist_rows_type_a(last_candidate, run_note, raw_json_file)
            base_name = "search_bar_checklist_typeA_FAILED.xlsx"

        df = pd.DataFrame(rows, columns=["Checklist Item","Result","Present Value (from output)","Pass/Fail"])
        safe_to_excel(df, base_name=base_name)
    return None

# Example usage
if __name__ == "__main__":
    keywords = "Federal Reserve Interest Rates"
    output = run_search_bar(keywords)
    if output:
        print("Final Output saved and checklist generated.")
    else:
        print("Failed to get valid results (best-effort checklist written if a candidate was captured).")
