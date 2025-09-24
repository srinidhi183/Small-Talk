import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import streamlit as st

# ========= IMPORT YOUR MODULES (adjust names if needed) =========
# Make sure these Python files sit next to app.py, or are importable as a package.
try:
    import trending  # exposes: generate_trending_set(...)
except Exception as e:
    trending = None
    TRENDING_IMPORT_ERR = str(e)

try:
    import recommendation  # exposes: generate_recommendation(...)
except Exception as e:
    recommendation = None      # <-- keep variable name consistent
    RECO_IMPORT_ERR = str(e)

# ✅ Import BOTH the module and the callables from update_me
try:
    import update_me
    from update_me import (
        generate_update_me_news_set,
        generate_update_me_element,
    )
except Exception as e:
    update_me = None
    generate_update_me_news_set = None
    generate_update_me_element = None
    UPDATE_IMPORT_ERR = str(e)

try:
    from search_bar import search_bar
except Exception as e:
    search_bar = None
    SEARCH_IMPORT_ERR = str(e)

# ================================================================

st.set_page_config(
    page_title="SmallTalk | Live Demo",
    page_icon="🧠",
    layout="wide"
)

# ---------- Sidebar: Global Controls ----------
st.sidebar.title("⚙️ Global Controls")
st.sidebar.markdown("These apply across all tabs.")

with st.sidebar:
    model_name = st.text_input("LLM / Backend Model", value="gpt-4o-mini")
    max_tokens = st.number_input("Max tokens (if used)", min_value=64, max_value=8192, value=1024, step=64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.4, 0.05)
    max_reruns = st.number_input("Auto-retries (search_bar)", min_value=1, max_value=10, value=5, step=1)
    categories_file = st.text_input("Categories file (Excel)", value="Categories.xlsx")
    show_raw = st.checkbox("Show raw JSON by default", value=False)
    st.markdown("---")
    st.caption("Tip: hide this sidebar while screenshotting for your PPT (View → Hide sidebar).")

# ---------- Utilities ----------
def _pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)

def _download_button(data: Union[List, Dict, str], label: str, file_name: str):
    if isinstance(data, (list, dict)):
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        mime = "application/json"
    else:
        payload = str(data)
        mime = "text/plain"
    st.download_button(label, payload, file_name=file_name, mime=mime, use_container_width=True)

def _guard_run(fn, *args, **kwargs):
    """Run a function and surface a user-friendly error if anything blows up."""
    try:
        return {"ok": True, "data": fn(*args, **kwargs), "error": None}
    except Exception as e:
        return {"ok": False, "data": None, "error": str(e)}

# ---------- Header ----------
st.title("🧩 SmallTalk – Live Demo")
st.caption("Trending • Recommendations • Update Me • Search Bar")

# ---------- Tabs ----------
tab_trending, tab_reco, tab_update, tab_search = st.tabs(
    ["🔥 Trending", "🎯 Recommendations", "📰 Update Me", "🔎 Search Bar"]
)

# =============== Trending Tab ===============
with tab_trending:
    st.subheader("🔥 Trending Keywords / Topics")
    st.markdown("Feed keywords into downstream flows or show a ranked list for the day.")

    with st.form("trending_form", clear_on_submit=False):
        seed = st.text_input("Optional seed / domain (e.g., 'AI', 'Finance', 'Sports')", value="")
        how_many = st.number_input("How many keywords?", 1, 20, 10)
        submitted = st.form_submit_button("Run Trending")

    if submitted:
        if trending is None:
            st.error(f"Could not import `trending` module.\n\n{TRENDING_IMPORT_ERR}")
        else:
            # Parse the seed: if provided, treat as comma-separated keywords; else let your module fetch its own.
            keywords = [s.strip() for s in seed.split(",") if s.strip()] or None

            # Call your real function from trending.py
            run = _guard_run(
                getattr(trending, "generate_trending_set"),
                keywords=keywords,                 # None -> trending.get_trending_topics() inside your code
                num_elements=int(how_many),        # your function signature accepts this param
                prompt_version="v02",
                categories_file=categories_file,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if not run["ok"]:
                st.error("Trending failed.")
                st.exception(run["error"])
            else:
                items = run["data"] or []
                if not isinstance(items, list) or len(items) == 0:
                    st.warning("No trending items returned.")
                else:
                    # Optionally validate each item with your validator
                    if hasattr(trending, "validate_trending_output"):
                        bad = []
                        for i, it in enumerate(items):
                            valid, errs = trending.validate_trending_output(it)
                            if not valid:
                                bad.append((i, errs))
                        if bad:
                            with st.expander("Validation warnings"):
                                for idx, errs in bad:
                                    st.write(f"Item #{idx+1}:")
                                    for e in errs:
                                        st.write(f"- {e}")

                    # Render cards
                    for it in items[: int(how_many) ]:   # respect the requested count visually
                        with st.container(border=True):
                            st.markdown(f"**{it.get('topic','(no topic)')}**")
                            st.caption(f"{it.get('category','General')} → {it.get('subcategory','')}")
                            if it.get("description"):
                                st.write(it["description"])
                            for section in ("why_is_it_trending","key_points","overlook_what_might_happen_next"):
                                vals = it.get(section, [])
                                if isinstance(vals, list) and vals:
                                    st.markdown(f"**{section.replace('_',' ').title()}**")
                                    for v in vals:
                                        st.write(f"- {v}")
                            # image if present
                            if isinstance(it.get("picture_url"), str) and it["picture_url"].startswith(("http://","https://")):
                                st.image(it["picture_url"], use_container_width=True)

                    _download_button(items, "⬇️ Download JSON", f"trending_{datetime.now().date()}.json")
                    if show_raw:
                        with st.expander("Raw JSON", expanded=True):
                            st.code(_pretty_json(items), language="json")


# =============== Recommendations Tab ===============
with tab_reco:
    st.subheader("🎯 Personalized / Rule-based Recommendations")
    st.markdown("Takes user interests or context and returns recommended topics/articles.")

    with st.form("reco_form", clear_on_submit=False):
        user_profile = st.text_input("Interests / Profile (comma-separated)",
                                     value="AI, Climate, Football")
        top_k = st.number_input("How many recommendations?", 1, 20, 5)
        reco_btn = st.form_submit_button("Run Recommendations")

    if reco_btn:
        if recommendation is None:
            st.error(f"Could not import `recommendation` module.\n\n{RECO_IMPORT_ERR}")
        else:
            interests = [s.strip() for s in user_profile.split(",") if s.strip()]
            if not interests:
                st.warning("Please enter at least one interest.")
            else:
                recos = []
                errors = []

                # Call YOUR function multiple times to build N recs
                for i in range(int(top_k)):
                    # round-robin through the provided interests
                    kw = interests[i % len(interests)]

                    run = _guard_run(
                        recommendation.generate_recommendation,
                        keywords=kw,                      # your function accepts `keywords`
                        prompt_version="v02",
                        categories_file=categories_file,
                        region="us",
                        num_results=15,
                        lang="en",
                        model_name=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    if not run["ok"] or not isinstance(run["data"], dict):
                        errors.append(run["error"] or "Empty result")
                        continue

                    rec = run["data"]

                    # optional: validate with your validator
                    if hasattr(recommendation, "validate_recommendation_output"):
                        valid, errs = recommendation.validate_recommendation_output(rec)
                        if not valid:
                            errors.append(f"Validation: {errs}")

                    recos.append(rec)

                if errors:
                    with st.expander("Warnings / Errors"):
                        for e in errors:
                            st.write("- ", e)

                if not recos:
                    st.warning("No recommendations returned.")
                else:
                    for rec in recos:
                        with st.container(border=True):
                            title = rec.get("headline") or rec.get("topic") or "(untitled)"
                            st.markdown(f"**{title}**")
                            if rec.get("topic"):
                                st.caption(rec["topic"])
                            if rec.get("introduction"):
                                st.write(rec["introduction"])

                            # Tips sections
                            kt = rec.get("key_tips_and_takeaways", {})
                            if isinstance(kt, dict):
                                for sec, tips in kt.items():
                                    if isinstance(tips, list) and tips:
                                        st.markdown(f"**{sec}**")
                                        for t in tips:
                                            st.write(f"- {t}")

                            if isinstance(rec.get("fun_facts"), list) and rec["fun_facts"]:
                                st.markdown("**Fun facts**")
                                for f in rec["fun_facts"]:
                                    st.write(f"- {f}")

                            if rec.get("conclusion"):
                                st.write(rec["conclusion"])

                            # image if present
                            img = rec.get("image_url")
                            if isinstance(img, str) and img.startswith(("http://", "https://")):
                                st.image(img, use_container_width=True)

                    _download_button(recos, "⬇️ Download JSON",
                                     f"recommendations_{datetime.now().date()}.json")
                    if show_raw:
                        with st.expander("Raw JSON", expanded=True):
                            st.code(_pretty_json(recos), language="json")


# =============== Update Me Tab ===============
with tab_update:
    st.subheader("📰 Update Me – Structured News Set")
    st.markdown("Generate structured updates from your keywords (category, topic, short_info, URLs, background_story…).")

    with st.form("update_form", clear_on_submit=False):
        kw = st.text_input("Keywords (comma-separated)", value="AI, Climate Change, Middle East")
        limit = st.number_input("How many items?", 1, 10, 3)
        mode = st.selectbox("Mode", ["General (use keywords)", "News (auto URLs)"], index=0)
        update_btn = st.form_submit_button("Generate Updates")

    if update_btn:
        if (generate_update_me_element is None) and (generate_update_me_news_set is None):
            st.error(f"Could not import update_me functions.\n\n{UPDATE_IMPORT_ERR}")
        else:
            items, errors = [], []
            keywords = [k.strip() for k in kw.split(",") if k.strip()]

            if mode.startswith("General"):
                # ✅ call element generator per keyword (no duplicate kwargs)
                for i in range(int(limit)):
                    topic = keywords[i % len(keywords)]
                    run = _guard_run(
                        generate_update_me_element,     # direct callable import
                        keywords=topic,
                        link=None,
                        news_text=None,
                        prompt_type="update_me_general",
                        categories_file=categories_file,
                        prompt_version="v01",
                        model_name=model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if run["ok"] and isinstance(run["data"], dict):
                        items.append(run["data"])
                    else:
                        errors.append(run["error"] or "Empty result")
            else:
                # Use your news set generator as-is (it ignores manual keywords)
                run = _guard_run(
                    generate_update_me_news_set,
                    use_tfidf=True,
                    sort_by_date=False,
                    categories_file=categories_file,
                    prompt_version="v01",
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if run["ok"] and isinstance(run["data"], list):
                    items = run["data"][: int(limit)]
                else:
                    errors.append(run["error"] or "Empty result")

            if errors:
                with st.expander("Warnings / Errors"):
                    for e in errors:
                        st.write("- ", e)

            if not items:
                st.warning("No updates returned.")
            else:
                for u in items:
                    with st.container(border=True):
                        st.markdown(f"**{u.get('topic','(no topic)')}**")
                        st.caption(f"{u.get('category','')} → {u.get('subcategory','')}")
                        si = u.get("short_info")
                        if isinstance(si, list):
                            for bullet in si:
                                st.write(f"- {bullet}")
                        elif isinstance(si, str):
                            st.write(si)
                        if u.get("reference_url"):
                            st.write(u["reference_url"])
                        img = u.get("picture_url")
                        if isinstance(img, str) and img.startswith(("http://","https://")):
                            st.image(img, use_container_width=True)

                _download_button(items, "⬇️ Download JSON", f"update_me_{datetime.now().date()}.json")
                if show_raw:
                    with st.expander("Raw JSON", expanded=True):
                        st.code(_pretty_json(items), language="json")


# =============== Search Bar Tab ===============
with tab_search:
    st.subheader("🔎 Smart Search Bar")
    st.markdown("Detects Type-A (full content) vs Type-B (search options) and returns a validated JSON.")

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input("User query / keywords", value="Formula 1, Racing")
        allow_options = st.checkbox("Allow options (Type-B)", value=False)
        prompt_version = st.selectbox("Prompt version", ["v01", "v02", "v03"], index=0)
        run_btn = st.form_submit_button("Run Search")

    if run_btn:
        if search_bar is None:
            st.error(f"Could not import `search_bar` function. {SEARCH_IMPORT_ERR}")
        else:
            result = _guard_run(
                search_bar,
                query, allow_options, prompt_version, categories_file,
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
            )

            if not result["ok"]:
                st.error("Search Bar failed.")
                st.exception(result["error"])
            else:
                data = result["data"]

                # Unwrap your return types:
                # - tuple(bool, dict) -> (need_choise, json_out)
                # - str -> error string from your function
                if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], (bool,)):
                    need_choice, json_out = data
                elif isinstance(data, str):
                    st.error(data)
                    json_out, need_choice = {}, None
                elif isinstance(data, dict):
                    # If your function ever returns just a dict (unlikely), handle it
                    json_out, need_choice = data, None
                else:
                    st.warning("Unexpected return type from search_bar().")
                    json_out, need_choice = {}, None

                # Header box
                with st.container(border=True):
                    st.markdown("**Search Bar Result**")
                    if need_choice is not None:
                        st.info(f"need_choice = {need_choice}")

                # Render helpers
                def _render_type_a(obj: Dict):
                    topic = (obj.get("topic") or {}).get("name") if isinstance(obj.get("topic"), dict) else obj.get("topic")
                    cat = (obj.get("category") or {}).get("categorie_name") if isinstance(obj.get("category"), dict) else obj.get("category")
                    sub = (obj.get("subcatetgory") or {}).get("subcatetgory_name") if isinstance(obj.get("subcatetgory"), dict) else obj.get("subcatetgory")

                    if topic or cat or sub:
                        st.markdown(f"**{topic or '(no topic)'}**")
                        st.caption(f"{cat or '—'} → {sub or '—'}")

                    # general facts / points
                    gf = obj.get("general_facts")
                    if isinstance(gf, dict):
                        st.markdown("**General facts**")
                        for k in ["general_definition", "general_points", "general_fun_fact", "key_facts_text", "key_facts_fun_fact"]:
                            v = gf.get(k)
                            if isinstance(v, str) and v.strip():
                                st.write(f"- {v.strip()}")

                    # news
                    nw = obj.get("news")
                    if isinstance(nw, dict) and nw.get("news_text"):
                        st.markdown("**News**")
                        st.write(f"- {nw['news_text']}")
                        if isinstance(nw.get("picture_url"), str):
                            st.write(nw["picture_url"])

                    # interesting trivia
                    it = obj.get("interesting_trivia")
                    if isinstance(it, dict) and it.get("trivia_text"):
                        st.markdown("**Interesting trivia**")
                        st.write(f"- {it['trivia_text']}")
                        if isinstance(it.get("picture_url"), str):
                            st.write(it["picture_url"])

                    # opinions / questions
                    op = obj.get("opinions")
                    if isinstance(op, dict) and op.get("opinions_text"):
                        st.markdown("**Opinions**")
                        st.write(f"- {op['opinions_text']}")
                    qs = obj.get("questions")
                    if isinstance(qs, dict) and qs.get("questions_text"):
                        st.markdown("**Questions**")
                        st.write(f"- {qs['questions_text']}")

                def _render_type_b(obj: Dict):
                    st.markdown("**Suggested Searches (Type-B)**")
                    for key in ["search1", "search2", "search3"]:
                        item = obj.get(key)
                        if isinstance(item, dict) and item:
                            q = item.get("key_words") or item.get("query") or key
                            cat = item.get("category") or "—"
                            sub = item.get("subcatetgory") or "—"
                            st.write(f"- {q}  ({cat} → {sub})")

                # Type detection and render
                if isinstance(json_out, dict):
                    top_keys = set(json_out.keys())
                    is_type_b = top_keys == {"search1", "search2", "search3"}
                    if is_type_b:
                        _render_type_b(json_out)
                    else:
                        _render_type_a(json_out)

                _download_button(json_out, "⬇️ Download JSON", f"search_bar_{datetime.now().date()}.json")
                with st.expander("Raw JSON", expanded=show_raw):
                    st.code(_pretty_json(json_out), language="json")


# ---------- Footer ----------
st.markdown("---")
st.caption("© SmallTalk Demo • Streaming UI built with Streamlit")
