import json
import time
from datetime import datetime
from typing import Any, Dict, List, Union

import streamlit as st
import pandas as pd
import altair as alt
from string import Template

# ========= IMPORT YOUR MODULES (adjust names if needed) =========
try:
    import trending  # exposes: generate_trending_set(...)
except Exception as e:
    trending = None
    TRENDING_IMPORT_ERR = str(e)

try:
    import recommendation  # exposes: generate_recommendation(...)
except Exception as e:
    recommendation = None
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
st.set_page_config(page_title="SmallTalk | Live Demo", page_icon="🧠", layout="wide")

# ---------- Theme Controls (user-customizable) ----------
with st.sidebar:
    st.markdown("### 🎨 Theme")
    accent = st.selectbox(
        "Accent",
        ["violet", "blue", "teal", "emerald", "rose", "amber", "sky", "fuchsia"],
        index=0,
        help="Pick a brand color for highlights."
    )
# Fixed values since controls are removed
radius = 16
max_tokens = 1024

ACCENTS = {
    "violet": ("#7C3AED", "#22D3EE"),
    "blue": ("#2563EB", "#22D3EE"),
    "teal": ("#0D9488", "#14B8A6"),
    "emerald": ("#059669", "#10B981"),
    "rose": ("#E11D48", "#FB7185"),
    "amber": ("#D97706", "#F59E0B"),
    "sky": ("#0284C7", "#06B6D4"),
    "fuchsia": ("#C026D3", "#A78BFA"),
}
ACCENT, ACCENT2 = ACCENTS.get(accent, ACCENTS["violet"])
PADDING = "16px 18px"  # fixed comfortable padding

# ---------- Global CSS (uses Template to avoid f-string brace errors) ----------
THEME_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0e1117;
  --card: rgba(255,255,255,0.06);
  --text: #E6E6E6;
  --muted:#9aa0a6;
  --accent: $ACCENT;
  --accent2: $ACCENT2;
  --radius: ${RADIUS}px;
  --pad: $PADDING;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #ffffff; --card: rgba(0,0,0,0.04); --text:#111827; --muted:#6b7280;
  }
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; }
h1,h2,h3,h4,h5,h6, p, span, div, label { color: var(--text) !important; }
.block-container { padding-top: 1.1rem; }

.hero {
  border-radius: calc(var(--radius) + 6px);
  padding: 22px 26px;
  background:
    radial-gradient(80% 120% at 20% 0%, rgba(255,255,255,.08), transparent 60%),
    linear-gradient(135deg, color-mix(in oklab, var(--accent) 25%, transparent) 0%, color-mix(in oklab, var(--accent2) 18%, transparent) 100%);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 20px 40px rgba(0,0,0,.18), inset 0 0 40px color-mix(in oklab, var(--accent) 12%, transparent);
}
.hero h1 { margin: 0 0 4px 0; font-size: 30px; }
.small-muted { color: var(--muted) !important; font-size: 12px; }

.kpi {
  padding: var(--pad);
  border-radius: var(--radius);
  background: var(--card);
  border: 1px solid rgba(255,255,255,.08);
}
.kpi b { color: var(--accent); }

.card {
  border-radius: var(--radius);
  padding: var(--pad);
  background: linear-gradient(180deg, color-mix(in oklab, var(--accent) 6%, transparent), transparent),
             var(--card);
  border: 1px solid rgba(255,255,255,.08);
  transition: transform .08s ease, box-shadow .16s ease;
}
.card:hover { transform: translateY(-1px); box-shadow: 0 10px 30px rgba(0,0,0,.25); }

.pill {
  display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
  border:1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04); font-size:12px; color: var(--muted);
}
.pill .dot { width:7px; height:7px; border-radius:999px; background: var(--accent); display:inline-block; }

hr.sep { border:none; border-top:1px solid rgba(255,255,255,.08); margin: 20px 0; }

.stDownloadButton button, .stButton>button {
  border-radius: calc(var(--radius) - 2px) !important; border:1px solid rgba(255,255,255,.1) !important;
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  color: white !important; font-weight: 700; letter-spacing: .2px;
  box-shadow: 0 10px 20px color-mix(in oklab, var(--accent) 25%, transparent);
}
.stDownloadButton button:hover, .stButton>button:hover { filter: brightness(1.03); }

.skeleton {
  position: relative; overflow: hidden; background: rgba(255,255,255,.06);
  border-radius: var(--radius); height: 120px; border: 1px solid rgba(255,255,255,.05);
}
.skeleton::after {
  content:""; position:absolute; inset:0; transform: translateX(-100%);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.08), transparent);
  animation: shimmer 1.4s infinite;
}
@keyframes shimmer { 100% { transform: translateX(100%); } }
</style>
""").substitute(
    ACCENT=ACCENT,
    ACCENT2=ACCENT2,
    RADIUS=radius,
    PADDING=PADDING,
)
st.markdown(THEME_CSS, unsafe_allow_html=True)

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
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return {"ok": True, "data": out, "error": None, "ms": ms}
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return {"ok": False, "data": None, "error": str(e), "ms": ms}

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)

def card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)

def card_close():
    st.markdown('</div>', unsafe_allow_html=True)

def chip(text: str):
    st.markdown(f'<span class="pill"><span class="dot"></span>{text}</span>', unsafe_allow_html=True)

# ---------- Sidebar: Global Controls ----------
st.sidebar.title(" Global Controls")
st.sidebar.markdown("These apply across all tabs.")

AVAILABLE_GPT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
AVAILABLE_PPLX_MODELS = ["sonar-small-chat", "sonar-medium-chat", "sonar-large-chat", "pplx-70b-chat", "pplx-70b-online"]

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["openai", "perplexity"],
    format_func=lambda v: "OpenAI (GPT)" if v == "openai" else "Perplexity"
)
model_name = st.sidebar.selectbox(
    "Model",
    AVAILABLE_GPT_MODELS if provider == "openai" else AVAILABLE_PPLX_MODELS,
    index=0
)

with st.sidebar:
    # max_tokens removed
    temperature = st.slider("Temperature", 0.0, 1.5, 0.4, 0.05)
    max_reruns = st.number_input("Auto-retries (search_bar)", min_value=1, max_value=10, value=5, step=1)
    categories_file = st.text_input("Categories file (Excel)", value="Categories.xlsx")
    show_raw = st.checkbox("Show raw JSON by default", value=False)

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <h1>SmallTalk – Live Demo</h1>
  <div class="small-muted">Trending • Recommendations • Update Me • Search Bar</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- KPIs ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f'<div class="kpi"> Provider<br><b>{ "OpenAI" if provider=="openai" else "Perplexity" }</b></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"> Model<br><b>{model_name}</b></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"> Temperature<br><b>{temperature}</b></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi"> Max tokens<br><b>{max_tokens}</b></div>', unsafe_allow_html=True)
c5.markdown(f'<div class="kpi"> Categories<br><b>{categories_file}</b></div>', unsafe_allow_html=True)
st.markdown('<hr class="sep" />', unsafe_allow_html=True)

# ---------- Tabs ----------
tab_trending, tab_reco, tab_update, tab_search = st.tabs(
    [" Trending", " Recommendations", " Update Me", " Search Bar"]
)

# =============== Trending ===============
with tab_trending:
    section_header("Trending Keywords / Topics", "Feed keywords into downstream flows or show a ranked list for the day.")

    with st.form("trending_form", clear_on_submit=False):
        seed = st.text_input("Optional seed / domain (e.g., 'AI', 'Finance', 'Sports')", value="")
        how_many = st.number_input("How many keywords?", 1, 20, 10)
        submitted = st.form_submit_button("Run Trending")

    if submitted:
        if trending is None:
            st.error(f"Could not import `trending` module.\n\n{TRENDING_IMPORT_ERR}")
        else:
            keywords = [s.strip() for s in seed.split(",") if s.strip()] or None
            with st.status("Fetching trending topics...", expanded=False):
                run = _guard_run(
                    getattr(trending, "generate_trending_set"),
                    keywords=keywords,
                    num_elements=int(how_many),
                    prompt_version="v02",
                    categories_file=categories_file,
                    provider=provider,
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
                    # Summary
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Items", len(items))
                    categories = [it.get("category") or "—" for it in items]
                    cc2.metric("Unique categories", len(set(categories)))
                    cc3.metric("Latency (ms)", f"{run['ms']:.0f}")

                    # Render as 2-column grid
                    cols = st.columns(2)
                    for i, it in enumerate(items[: int(how_many) ]):
                        with cols[i % 2]:
                            card_open()
                            st.markdown(f"**{it.get('topic','(no topic)')}**  ")
                            st.caption(f"{it.get('category','General')} → {it.get('subcategory','')}")
                            if it.get("description"):
                                st.write(it["description"])
                            tags = []
                            if it.get("why_is_it_trending"): tags.append("Why it's trending")
                            if it.get("key_points"): tags.append("Key points")
                            if it.get("overlook_what_might_happen_next"): tags.append("What might happen next")
                            if tags:
                                st.markdown(" ".join([f'<span class="pill">{t}</span>' for t in tags]), unsafe_allow_html=True)
                            if isinstance(it.get("picture_url"), str) and it["picture_url"].startswith(("http://","https://")):
                                st.image(it["picture_url"], use_container_width=True)
                            card_close()

                    _download_button(items, "⬇️ Download JSON", f"trending_{datetime.now().date()}.json")
                    if show_raw:
                        with st.expander("Raw JSON", expanded=True):
                            st.code(_pretty_json(items), language="json")

# =============== Recommendations ===============
with tab_reco:
    section_header("Personalized / Rule-based Recommendations", "Takes user interests/context and returns recommended topics/articles.")

    with st.form("reco_form", clear_on_submit=False):
        user_profile = st.text_input("Interests / Profile (comma-separated)", value="AI, Climate, Football")
        top_k = st.number_input("How many recommendations?", 1, 20, 6)
        reco_btn = st.form_submit_button("Run Recommendations")

    if reco_btn:
        if recommendation is None:
            st.error(f"Could not import `recommendation` module.\n\n{RECO_IMPORT_ERR}")
        else:
            interests = [s.strip() for s in user_profile.split(",") if s.strip()]
            if not interests:
                st.warning("Please enter at least one interest.")
            else:
                recos, errors = [], []
                t0 = time.perf_counter()
                for i in range(int(top_k)):
                    kw = interests[i % len(interests)]
                    run = _guard_run(
                        recommendation.generate_recommendation,
                        keywords=kw,
                        prompt_version="v02",
                        categories_file=categories_file,
                        provider=provider,
                        model_name=model_name,
                        region="us",
                        num_results=15,
                        lang="en",
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if not run["ok"] or not isinstance(run["data"], dict):
                        errors.append(run["error"] or "Empty result")
                        continue
                    recos.append(run["data"])
                ms = (time.perf_counter() - t0) * 1000

                if errors:
                    with st.expander("Warnings / Errors"):
                        for e in errors: st.write("- ", e)

                if not recos:
                    st.warning("No recommendations returned.")
                else:
                    r1, r2 = st.columns(2)
                    r1.metric("Items", len(recos))
                    r2.metric("Latency (ms)", f"{ms:.0f}")

                    # grid
                    cols = st.columns(2)
                    for i, rec in enumerate(recos):
                        with cols[i % 2]:
                            card_open()
                            title = rec.get("headline") or rec.get("topic") or "(untitled)"
                            st.markdown(f"**{title}**")
                            if rec.get("topic"): chip(rec["topic"])
                            if rec.get("introduction"): st.write(rec["introduction"])

                            kt = rec.get("key_tips_and_takeaways", {})
                            if isinstance(kt, dict):
                                for sec, tips in kt.items():
                                    if isinstance(tips, list) and tips:
                                        st.markdown(f"**{sec}**")
                                        for t in tips: st.write(f"- {t}")

                            if isinstance(rec.get("fun_facts"), list) and rec["fun_facts"]:
                                st.markdown("**Fun facts**")
                                for f in rec["fun_facts"]: st.write(f"- {f}")

                            if rec.get("conclusion"): st.write(rec["conclusion"])
                            img = rec.get("image_url")
                            if isinstance(img, str) and img.startswith(("http://", "https://")):
                                st.image(img, use_container_width=True)
                            card_close()

                    _download_button(recos, "⬇️ Download JSON", f"recommendations_{datetime.now().date()}.json")
                    if show_raw:
                        with st.expander("Raw JSON", expanded=True):
                            st.code(_pretty_json(recos), language="json")

# =============== Update Me ===============
with tab_update:
    section_header("Update Me – Structured News Set", "Generate structured updates from keywords (category, topic, bullets, URLs…).")

    with st.form("update_form", clear_on_submit=False):
        kw = st.text_input("Keywords (comma-separated)", value="AI, Climate Change, Middle East")
        limit = st.number_input("How many items?", 1, 10, 4)
        mode = st.selectbox("Mode", ["General (use keywords)", "News (auto URLs)"], index=0)
        update_btn = st.form_submit_button("Generate Updates")

    if update_btn:
        if (generate_update_me_element is None) and (generate_update_me_news_set is None):
            st.error(f"Could not import update_me functions.\n\n{UPDATE_IMPORT_ERR}")
        else:
            items, errors = [], []
            keywords = [k.strip() for k in kw.split(",") if k.strip()]

            with st.status("Generating updates…", expanded=False):
                t0 = time.perf_counter()
                if mode.startswith("General"):
                    for i in range(int(limit)):
                        topic = keywords[i % len(keywords)]
                        run = _guard_run(
                            generate_update_me_element,
                            keywords=topic,
                            link=None, news_text=None,
                            prompt_type="update_me_general",
                            categories_file=categories_file,
                            provider=provider, model_name=model_name,
                            max_tokens=max_tokens, temperature=temperature,
                        )
                        if run["ok"] and isinstance(run["data"], dict):
                            items.append(run["data"])
                        else:
                            errors.append(run["error"] or "Empty result")
                else:
                    run = _guard_run(
                        generate_update_me_news_set,
                        use_tfidf=True, sort_by_date=False,
                        categories_file=categories_file,
                        provider=provider, model_name=model_name,
                        max_tokens=max_tokens, temperature=temperature,
                    )
                    if run["ok"] and isinstance(run["data"], list):
                        items = run["data"][: int(limit)]
                    else:
                        errors.append(run["error"] or "Empty result")
                dt_ms = (time.perf_counter() - t0) * 1000

            if errors:
                st.warning("Some items failed. See details below.")
                with st.expander("Warnings / Errors"):
                    for e in errors: st.write("- ", e)

            if not items:
                st.info("No updates returned.")
            else:
                u1, u2, u3 = st.columns(3)
                u1.metric("Items", len(items))
                u2.metric("Mode", mode.split()[0])
                u3.metric("Latency (ms)", f"{dt_ms:.0f}")

                cols = st.columns(2)
                for i, u in enumerate(items):
                    with cols[i % 2]:
                        card_open()
                        st.markdown(f"**{u.get('topic','(no topic)')}**")
                        st.caption(f"{u.get('category','')} → {u.get('subcategory','')}")
                        si = u.get("short_info")
                        if isinstance(si, list):
                            for bullet in si: st.write(f"- {bullet}")
                        elif isinstance(si, str):
                            st.write(si)
                        if u.get("reference_url"):
                            st.write(u["reference_url"])
                        img = u.get("picture_url")
                        if isinstance(img, str) and img.startswith(("http://","https://")):
                            st.image(img, use_container_width=True)
                        card_close()

                # Category chart
                try:
                    df = pd.DataFrame([{
                        "category": (x.get("category") or "—"),
                        "subcategory": (x.get("subcategory") or x.get("subcatetgory") or "—"),
                    } for x in items])
                    if not df.empty:
                        st.markdown("#### Category Distribution")
                        cat_counts = df.groupby("category").size().reset_index(name="count")
                        chart = (alt.Chart(cat_counts)
                                 .mark_bar()
                                 .encode(x=alt.X("category:N", sort="-y", title="Category"),
                                         y=alt.Y("count:Q", title="Count"),
                                         tooltip=["category", "count"]))
                        st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.info(f"(Info) Could not render category chart: {e}")

                _download_button(items, "⬇️ Download JSON", f"update_me_{datetime.now().date()}.json")
                if show_raw:
                    with st.expander("Raw JSON", expanded=True):
                        st.code(_pretty_json(items), language="json")

# =============== Search Bar ===============
with tab_search:
    section_header("Smart Search Bar", "Detects Type-A (full content) vs Type-B (search options) and returns a validated JSON.")

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input("User query / keywords", value="Formula 1, Racing")
        allow_options = st.checkbox("Allow options (Type-B)", value=False)
        prompt_version = st.selectbox("Prompt version", ["v01", "v02", "v03"], index=0)
        run_btn = st.form_submit_button("Run Search")

    if run_btn:
        if search_bar is None:
            st.error(f"Could not import `search_bar` function. {SEARCH_IMPORT_ERR}")
        else:
            t0 = time.perf_counter()
            result = _guard_run(
                search_bar,
                query, allow_options, prompt_version, categories_file,
                provider=provider, model_name=model_name,
                temperature=temperature, max_tokens=max_tokens
            )
            dt_ms = (time.perf_counter() - t0) * 1000

            if not result["ok"]:
                st.error("Search Bar failed.")
                st.exception(result["error"])
            else:
                data = result["data"]
                if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], (bool,)):
                    need_choice, json_out = data
                elif isinstance(data, str):
                    st.error(data); json_out, need_choice = {}, None
                elif isinstance(data, dict):
                    json_out, need_choice = data, None
                else:
                    st.warning("Unexpected return type from search_bar().")
                    json_out, need_choice = {}, None

                card_open()
                st.markdown("**Search Bar Result**")
                s1, s2 = st.columns(2)
                if need_choice is not None: s1.info(f"need_choice = {need_choice}")
                s2.metric("Latency (ms)", f"{dt_ms:.0f}")

                def _render_type_a(obj: Dict):
                    topic = (obj.get("topic") or {}).get("name") if isinstance(obj.get("topic"), dict) else obj.get("topic")
                    cat = (obj.get("category") or {}).get("categorie_name") if isinstance(obj.get("category"), dict) else obj.get("category")
                    sub = (obj.get("subcatetgory") or {}).get("subcatetgory_name") if isinstance(obj.get("subcatetgory"), dict) else obj.get("subcatetgory") or obj.get("subcategory")
                    if topic or cat or sub:
                        st.markdown(f"**{topic or '(no topic)'}**")
                        st.caption(f"{cat or '—'} → {sub or '—'}")
                    gf = obj.get("general_facts")
                    if isinstance(gf, dict):
                        st.markdown("**General facts**")
                        for k in ["general_definition", "general_points", "general_fun_fact", "key_facts_text", "key_facts_fun_fact"]:
                            v = gf.get(k)
                            if isinstance(v, str) and v.strip(): st.write(f"- {v.strip()}")
                    nw = obj.get("news")
                    if isinstance(nw, dict) and nw.get("news_text"):
                        st.markdown("**News**"); st.write(f"- {nw['news_text']}")
                        if isinstance(nw.get("picture_url"), str): st.write(nw["picture_url"])
                    it = obj.get("interesting_trivia")
                    if isinstance(it, dict) and it.get("trivia_text"):
                        st.markdown("**Interesting trivia**"); st.write(f"- {it['trivia_text']}")
                        if isinstance(it.get("picture_url"), str): st.write(it["picture_url"])
                    op = obj.get("opinions")
                    if isinstance(op, dict) and op.get("opinions_text"):
                        st.markdown("**Opinions**"); st.write(f"- {op['opinions_text']}")
                    qs = obj.get("questions")
                    if isinstance(qs, dict) and qs.get("questions_text"):
                        st.markdown("**Questions**"); st.write(f"- {qs['questions_text']}")

                def _render_type_b(obj: Dict):
                    st.markdown("**Suggested Searches (Type-B)**")
                    for key in ["search1", "search2", "search3"]:
                        item = obj.get(key)
                        if isinstance(item, dict) and item:
                            q = item.get("key_words") or item.get("query") or key
                            cat = item.get("category") or "—"
                            sub = item.get("subcatetgory") or item.get("subcategory") or "—"
                            st.write(f"- {q}  ({cat} → {sub})")

                if isinstance(json_out, dict):
                    top_keys = set(json_out.keys())
                    if top_keys == {"search1", "search2", "search3"}:
                        _render_type_b(json_out)
                    else:
                        _render_type_a(json_out)

                _download_button(json_out, " Download JSON", f"search_bar_{datetime.now().date()}.json")
                with st.expander("Raw JSON", expanded=show_raw):
                    st.code(_pretty_json(json_out), language="json")
                card_close()

# ---------- Footer ----------
st.markdown('<hr class="sep" />', unsafe_allow_html=True)
st.caption("© SmallTalk Demo • Streamlit UI")
