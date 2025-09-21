import os, json
import streamlit as st

try:
    from src.io_utils import load_tables
except Exception:
    load_tables = None

from src.agent.router import IntentRouter
from src.agent.handlers import definition_handler, slice_handler, explain_handler, diagnostics_handler, process_business_query

st.set_page_config(page_title="Ask TPI Agent", layout="wide")
st.title("Ask TPI Agent")
st.caption("This Ask TPI Agent is a beta feature that uses intent-based routing. Results are for guidance only and may be inaccurate or incomplete.")

def _load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def _load_text(path, default=""):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return default

def _ensure_defaults():
    defaults = {
        "agent_sector": None,
        "agent_company": None,
        "agent_region": None,
        "agent_scenarios": [],
        "agent_year_window": (2015, 2035),
        "agent_exact_region": False,
        "agent_last_applied": None,
        "agent_flash": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _read_state():
    return {
        "sector": st.session_state.get("agent_sector"),
        "company": st.session_state.get("agent_company"),
        "region": st.session_state.get("agent_region"),
        "scenarios": st.session_state.get("agent_scenarios"),
        "year_window": st.session_state.get("agent_year_window", (2015, 2035)),
        "exact_region": st.session_state.get("agent_exact_region", False),
    }

def _normalize_filters(d):
    """Return a normalized dict so equality is stable across runs."""
    out = {}
    for k, v in (d or {}).items():
        if k == "scenarios":
            out[k] = sorted(list(v)) if v is not None else []
        elif k == "year_window":
            # store as tuple of ints
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out[k] = (int(v[0]), int(v[1]))
            else:
                out[k] = (2015, 2035)
        else:
            out[k] = v
    return out

def _current_normalized():
    return _normalize_filters({
        "sector": st.session_state.get("agent_sector"),
        "company": st.session_state.get("agent_company"),
        "region": st.session_state.get("agent_region"),
        "scenarios": st.session_state.get("agent_scenarios"),
        "year_window": st.session_state.get("agent_year_window"),
        "exact_region": st.session_state.get("agent_exact_region"),
    })

cfg = _load_json("src/agent/agent_config.json", {})
router = IntentRouter(cfg)
resources = {
    "dictionary": _load_json("resources/data_dictionary.json", {}),
    "methodology": _load_text("resources/methodology.md", ""),
}

@st.cache_data(show_spinner=False)
def _load_tables_cached():
    if load_tables is None:
        return None, None
    return load_tables()

fact_company, fact_bench = _load_tables_cached()

_ensure_defaults()
state = _read_state()

if st.session_state["agent_flash"]:
    st.success(st.session_state["agent_flash"])
    st.session_state["agent_flash"] = None

colL, colR = st.columns([2, 1], gap="large")
with colL:
    q = st.text_input(
        "Question",
        placeholder="Examples: Define CBD • Which steel companies are best positioned for 1.5°C? • Compare Ryanair vs airlines sector • Executive summary",
    )
with colR:
    st.write("Current filters")
    st.json(state)
    if not any([state["sector"], state["company"], state["region"], state["scenarios"]]):
        st.caption("Filters will be set automatically when you ask for data (e.g., 'Get Airlines data for Ryanair')")

if not q:
    st.info("Try: 'Define CBD', 'Which steel companies are best positioned?', 'Risk assessment for auto sector', 'Compare Ryanair vs peers', or 'Executive summary'.")
    st.stop()

intent = router.detect(q)

rep = None

business_keywords = ['best positioned', 'top performers', 'investment', 'risk assessment', 
                    'compare', 'vs', 'versus', 'sector analysis', 'executive summary', 
                    'overview', 'outliers', 'falling behind', 'opportunities', 'dashboard']

if any(keyword in q.lower() for keyword in business_keywords):
    user_context = f"Sector: {state.get('sector', 'None')}, Company: {state.get('company', 'None')}"
    rep = process_business_query(q, user_context, fact_company, fact_bench)

elif intent == "definition":
    rep = definition_handler(q, resources)

elif intent == "slice_request":
    rep = slice_handler(q, state, fact_company, fact_bench)

    if getattr(rep, "set_filters", None):
        newf = _normalize_filters(rep.set_filters)
        curf = _current_normalized()

        subset_cur = {k: curf.get(k) for k in newf.keys()}
        if newf != subset_cur:
            for k, v in newf.items():
                key = f"agent_{k}"
                if k == "scenarios":
                    v = list(v)
                st.session_state[key] = v
            st.session_state["agent_last_applied"] = newf
            st.session_state["agent_flash"] = "Filters updated from your query."
            st.rerun()

elif intent == "explain_view":
    rep = explain_handler(q, state, fact_company, fact_bench)

else:
    rep = diagnostics_handler(q, state, fact_company, fact_bench)

st.markdown(f"**Intent:** `{rep.intent}`")
st.markdown(rep.text)
if getattr(rep, "citations", None):
    st.caption("Sources: " + ", ".join(rep.citations))
if getattr(rep, "attachments", None):
    st.subheader("Downloads")
    for name, df in rep.attachments.items():
        st.download_button(
            f"Download {name}",
            df.to_csv(index=False).encode("utf-8"),
            file_name=name,
            mime="text/csv",
        )
