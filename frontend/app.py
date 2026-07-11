"""RouteZero — landing page + ops dashboard."""

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routezero.frontend")

BACKEND_URL     = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
DEMO_MODE       = os.getenv("DEMO_MODE", "true").strip().lower() in ("1", "true", "yes")
READ_TIMEOUT    = 5
LLM_TIMEOUT     = 120
APPROVE_TIMEOUT = 30

# ── Colour tokens ─────────────────────────────────────────────────────────────
# Light theme: beige bg, white cards, black dark-cards, cyan accent
C_RED    = "#c0392b"
C_AMBER  = "#b7770d"
C_BLUE   = "#1a6bbf"
C_GREEN  = "#1a7a2e"
C_ORANGE = "#c0580a"
C_BG     = "#F2ECCF"       # beige page background
C_CARD   = "#FFFFFF"       # white card
C_CARD2  = "#000000"       # dark/black card
C_BORDER = "#000000"       # warm grey border
C_MUTED  = "#6b6250"       # muted warm brown text
C_TEXT   = "#1a1a1a"       # dark text (on light bg)
C_TEXT2  = "#ffffff"       # light text (on dark bg)
C_TEXT3  = "#ffffff"
C_ACCENT = "#0fa8b0"       # teal/cyan accent
C_ACC2   = "#0d8a91"       # darker teal
PRIORITY_COLOR = {"P1": C_RED, "P2": C_AMBER, "P3": C_BLUE}

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
*{{box-sizing:border-box;margin:0;padding:0;}}
[data-testid="stAppViewContainer"]{{background:{C_BG};}}
[data-testid="stSidebar"]{{display:none!important;}}
[data-testid="stMain"]{{background:transparent;padding:0!important;}}
[data-testid="block-container"]{{padding:0 2rem 2rem 2rem!important;max-width:1200px!important;}}
h1,h2,h3,h4,h5,h6,p,li,td,th,label,.stMarkdown{{font-family:'Inter','Segoe UI',system-ui,sans-serif!important;color:{C_TEXT};}}
div,span{{font-family:'Inter','Segoe UI',system-ui,sans-serif!important;}}
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondary"] *{{color:#ffffff!important;}}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover *{{color:#ffffff!important;}}
code,pre{{font-family:'JetBrains Mono',monospace!important;}}

/* ── NAV ── */
.hero-kicker {{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:{C_ACCENT};margin-bottom:16px;display:flex;align-items:center;gap:8px;}}
.hero-kicker::before {{content:'';width:24px;height:2px;background:{C_ACCENT};display:inline-block;}}
.hero-h1 {{font-size:52px;font-weight:900;letter-spacing:-.04em;line-height:1.05;color:{C_TEXT};margin:0 0 20px 0;}}
.hero-h1 span {{background:linear-gradient(135deg,{C_ACCENT},{C_ACC2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.hero-sub {{font-size:16px;color:{C_MUTED};line-height:1.65;margin:0 0 28px 0;max-width:480px;}}
.panel-label {{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:{C_MUTED};margin-bottom:10px;display:flex;align-items:center;gap:8px;}}
.panel-label::after {{content:'';flex:1;height:1px;background:{C_BORDER};}}
.section-eyebrow {{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:{C_ACCENT};margin-bottom:10px;}}
.section-title {{font-size:30px;font-weight:800;letter-spacing:-.03em;color:{C_TEXT};margin:0 0 8px 0;}}

/* ── PILLARS ── */
.pillars-grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:64px;}}
.pillar {{background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:22px 20px;}}
.pillar-icon {{width:36px;height:36px;border-radius:8px;background:{C_CARD2};display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:14px;}}
.pillar-title {{font-size:15px;font-weight:700;color:{C_TEXT};margin-bottom:6px;}}
.pillar-desc {{font-size:13px;color:{C_MUTED};line-height:1.6;}}

/* ── ARCH ── */
.arch-title {{font-size:28px;font-weight:800;letter-spacing:-.03em;color:{C_TEXT};margin:0 0 12px 0;}}
.arch-desc {{font-size:14px;color:{C_MUTED};line-height:1.7;margin:0 0 24px 0;}}

/* ── INTEGRATIONS ── */
.integrations-row {{display:flex;align-items:center;justify-content:center;gap:36px;padding:36px 0 28px 0;border-top:1px solid {C_BORDER};flex-wrap:wrap;}}
.int-badge {{font-size:14px;font-weight:700;color:{C_MUTED};display:flex;align-items:center;gap:7px;}}

/* ── DASHBOARD ── */
.status-bar {{display:flex;align-items:center;background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;overflow:hidden;margin-bottom:24px;}}
.status-item {{flex:1;padding:14px 20px;border-right:1px solid {C_BORDER};}}
.status-item:last-child {{border-right:none;}}
.status-value {{font-size:22px;font-weight:800;line-height:1;margin-bottom:3px;letter-spacing:-.02em;color:{C_TEXT};}}
.status-label {{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};}}

/* ── ROUTING BLOCK ── */
.routing-block {{background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:22px 26px;margin:20px 0;position:relative;overflow:hidden;}}
.routing-block::before {{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{C_ACCENT},{C_BLUE},transparent);}}

/* ── BADGES ── */
.badge {{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;}}
.badge-p1 {{background:{C_RED}18;color:{C_RED};border:1px solid {C_RED}40;}}
.badge-p2 {{background:{C_AMBER}18;color:{C_AMBER};border:1px solid {C_AMBER}40;}}
.badge-p3 {{background:{C_BLUE}18;color:{C_BLUE};border:1px solid {C_BLUE}40;}}
.badge-ok {{background:{C_GREEN}18;color:{C_GREEN};border:1px solid {C_GREEN}40;}}
.badge-warn {{background:{C_AMBER}18;color:{C_AMBER};border:1px solid {C_AMBER}40;}}
.badge-info {{background:{C_ACCENT}18;color:{C_ACCENT};border:1px solid {C_ACCENT}40;}}
.badge-demo {{background:#7c3aed18;color:#7c3aed;border:1px solid #7c3aed40;}}

/* ── MISC ── */
.rz-bar-bg {{background:{C_BORDER};border-radius:3px;height:4px;width:100%;margin-top:8px;}}
.rz-bar-fill {{height:4px;border-radius:3px;}}
.cause-callout {{background:linear-gradient(135deg,{C_AMBER}0a,transparent);border:1px solid {C_AMBER}40;border-left:3px solid {C_AMBER};border-radius:8px;padding:14px 18px;margin:16px 0;}}
.ctx-chip {{display:inline-block;background:{C_BORDER};color:{C_MUTED};font-size:10px;font-family:'JetBrains Mono',monospace;padding:2px 8px;border-radius:4px;margin:2px 3px 2px 0;border:1px solid {C_BORDER};}}
.approve-zone {{background:linear-gradient(135deg,{C_GREEN}0a,transparent);border:1px solid {C_GREEN}40;border-left:3px solid {C_GREEN};border-radius:10px;padding:20px 24px;margin-top:24px;}}
.stress-warning {{background:{C_RED}0a;border:1px solid {C_RED}30;border-left:3px solid {C_RED};border-radius:8px;padding:12px 16px;margin-bottom:10px;font-size:13px;color:{C_TEXT};}}

/* ── DARK CARDS — text must be light ── */
.flag-card {{background:{C_CARD2};border:1px solid {C_CARD2};border-left:3px solid {C_RED};border-radius:8px;padding:18px 22px;margin-bottom:14px;}}
.flag-card * {{color:{C_TEXT2}!important;}}
.rz-card {{background:{C_CARD2};border:1px solid {C_BORDER};border-radius:8px;padding:16px 20px;margin-bottom:12px;}}
.rz-card * {{color:{C_TEXT2}!important;}}
.step-card {{background:{C_CARD2};border:1px solid {C_BORDER};border-left:3px solid {C_ACCENT};border-radius:8px;padding:18px 20px;margin-bottom:4px;}}
.step-card * {{color:{C_TEXT2}!important;}}
.step-number {{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;background:{C_ACCENT}30;color:{C_ACCENT}!important;border-radius:50%;font-size:13px;font-weight:700;margin-right:10px;flex-shrink:0;}}
.step-title {{font-size:14px;font-weight:700;}}
.step-time {{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_ACCENT}!important;margin-top:2px;}}
.step-desc {{font-size:13px;margin-top:8px;margin-left:38px;line-height:1.6;}}
.code-block {{background:#060810;border:1px solid #333;border-radius:6px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#7dd3fc;margin-top:10px;margin-left:38px;overflow-x:auto;white-space:pre;line-height:1.6;}}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"]{{border-bottom:1px solid {C_BORDER}!important;gap:4px;}}
[data-testid="stTabs"] [role="tab"]{{font-size:13px;font-weight:500;color:{C_MUTED};padding:8px 16px;border-radius:6px 6px 0 0;}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{color:{C_TEXT};font-weight:700;background:{C_CARD};}}
[data-testid="stMetricValue"]{{font-size:22px!important;font-weight:700!important;color:{C_TEXT}!important;}}
[data-testid="stExpander"]{{background:{C_CARD}!important;border:1px solid {C_BORDER}!important;border-radius:8px!important;}}
div[data-testid="stDataFrame"]{{border:1px solid {C_BORDER};border-radius:8px;overflow:hidden;}}
iframe{{border:none!important;border-radius:8px;}}

/* ── INPUTS ── */
textarea{{background:{C_CARD2}!important;border:1px solid {C_BORDER}!important;border-radius:8px!important;color:{C_TEXT2}!important;font-family:'JetBrains Mono',monospace!important;font-size:13px!important;line-height:1.6!important;}}
textarea:focus{{border-color:{C_ACCENT}!important;box-shadow:0 0 0 2px {C_ACCENT}20!important;}}
textarea::placeholder{{color:#888!important;}}
[data-testid="stSelectbox"] > div > div{{background:{C_CARD2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT2}!important;border-radius:8px!important;}}
[data-testid="stTextInput"] > div > div > input{{background:{C_CARD2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT2}!important;border-radius:8px!important;}}
[data-testid="stNumberInput"] > div > div > input{{background:{C_CARD2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT2}!important;border-radius:8px!important;}}
[data-testid="stFileUploader"]{{background:{C_CARD}!important;border:1px solid {C_BORDER}!important;border-radius:8px!important;}}
div[data-baseweb="select"] ul {{background:{C_CARD2}!important;}}
div[data-baseweb="select"] li {{color:{C_TEXT2}!important;background:{C_CARD2}!important;}}
div[data-baseweb="select"] li:hover {{background:#2a2a2a!important;color:{C_TEXT2}!important;}}
div[data-baseweb="popover"] {{background:{C_CARD2}!important;}}
div[data-baseweb="menu"] {{background:{C_CARD2}!important;}}
div[data-baseweb="menu"] li {{color:{C_TEXT2}!important;}}
div[data-baseweb="select"] *{{color:{C_TEXT2}!important;}}
div[data-baseweb="popover"] *{{color:{C_TEXT2}!important;background:{C_CARD2};}}
div[data-baseweb="menu"] *{{color:{C_TEXT2}!important;}}
</style>
"""

st.set_page_config(page_title="RouteZero", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "landing"
if "prev_page" not in st.session_state:
    st.session_state["prev_page"] = "landing"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _show_err(exc):
    st.error(f"Backend not reachable at {BACKEND_URL} ({type(exc).__name__})")

def api_get(path, timeout=READ_TIMEOUT):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        r.raise_for_status(); return r.json()
    except requests.HTTPError as e:
        st.error(f"Backend error GET {path}: HTTP {e.response.status_code}"); return None
    except Exception as e:
        _show_err(e); return None

def api_post(path, body=None, timeout=READ_TIMEOUT):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=body, timeout=timeout)
        r.raise_for_status(); return r.json()
    except requests.HTTPError as e:
        try: detail = e.response.json().get("detail", e.response.status_code)
        except Exception: detail = e.response.status_code
        st.error(f"Backend error POST {path}: {detail}"); return None
    except Exception as e:
        _show_err(e); return None

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def parse_ts(v):
    if not v: return None
    try:
        ts = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except: return None

def fmt_ts(v):
    ts = parse_ts(v)
    return ts.strftime("%Y-%m-%d %H:%M UTC") if ts else "—"

def fmt_pct(v):
    try: return f"{float(v):.0%}"
    except: return "—"

def clamp01(v):
    try: return min(max(float(v), 0.0), 1.0)
    except: return 0.0

def priority_badge(p):
    cls = {"P1":"badge-p1","P2":"badge-p2","P3":"badge-p3"}.get(p,"badge-info")
    return f'<span class="badge {cls}">{p}</span>'

def output_type_label(v):
    return {"full_ticket":"Full Ticket","manager_digest":"Manager Digest",
            "cross_functional_fyi":"Cross-Functional FYI"}.get(v, str(v or "").replace("_"," ").title())

def confidence_bar(v, label=""):
    pct = int(clamp01(v)*100)
    col = C_GREEN if pct>=80 else C_AMBER if pct>=60 else C_RED
    txt = label or f"Confidence: {pct}%"
    return f"""<div style="margin:10px 0;">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:{C_MUTED};margin-bottom:5px;">
        <span>{txt}</span><span style="color:{col};font-weight:700;">{pct}%</span>
      </div>
      <div class="rz-bar-bg"><div class="rz-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{col},{col}bb);"></div></div>
    </div>"""

@st.cache_data(show_spinner=False)
def load_org_config():
    try:
        p = Path(__file__).resolve().parent.parent / "data" / "org_config.json"
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logger.warning("org_config: %s", e); return None

# ---------------------------------------------------------------------------
# NAV components
# ---------------------------------------------------------------------------

def render_nav(current_page: str):
    logo_col, gap_col, p_col, d_col, c_col = st.columns([3, 3, 1, 1, 1])

    with logo_col:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:10px 0 6px 0;">
          <div style="width:50px;height:50px;background:linear-gradient(135deg,{C_ACCENT},{C_BLUE});
                      border-radius:8px;display:flex;align-items:center;justify-content:center;
                      font-size:28px;flex-shrink:0;box-shadow:0 4px 12px {C_ACCENT}40;">⚡</div>
          <span style="font-size:18px;font-weight:900;letter-spacing:-.05em;color:{C_TEXT};">
            Route<span style="color:{C_ACCENT};">Zero</span>
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(n+3) button {{
        border-radius: 0 !important;
        border: none !important;
        border-right: 1px solid {C_BORDER} !important;
        background: {C_CARD} !important;
        color: {C_TEXT2} !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: none !important;
        padding: 8px 4px !important;
        margin-top: 8px !important;
        outline: none !important;
    }}
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(3) button {{
        border-radius: 8px 0 0 8px !important;
        border-left: 1px solid {C_BORDER} !important;
        border-top: 1px solid {C_BORDER} !important;
        border-bottom: 1px solid {C_BORDER} !important;
    }}
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(4) button {{
        border-top: 1px solid {C_BORDER} !important;
        border-bottom: 1px solid {C_BORDER} !important;
    }}
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(5) button {{
        border-radius: 0 8px 8px 0 !important;
        border-right: 1px solid {C_BORDER} !important;
        border-top: 1px solid {C_BORDER} !important;
        border-bottom: 1px solid {C_BORDER} !important;
    }}
    div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="column"]:nth-child(n+3) button:hover {{
        background: {C_CARD2} !important;
        color: {C_TEXT2} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    clicked = None
    with p_col:
        if st.button("Product", key=f"nav_p_{current_page}", use_container_width=True):
            clicked = "product"
    with d_col:
        if st.button("Docs", key=f"nav_d_{current_page}", use_container_width=True):
            clicked = "docs"
    with c_col:
        if st.button("Connect", key=f"nav_c_{current_page}", use_container_width=True):
            clicked = "connect"

    st.markdown(f'<div style="height:1px;background:{C_BORDER};margin-bottom:0;"></div>', unsafe_allow_html=True)
    return clicked


def render_dash_nav(current_page: str):
    logo_col, gap_col, h_col, d_col, c_col = st.columns([3, 3, 1, 1, 1])

    with logo_col:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:10px 0 6px 0;">
          <div style="width:36px;height:36px;background:linear-gradient(135deg,{C_ACCENT},{C_BLUE});
                      border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:18px;">⚡</div>
          <span style="font-size:17px;font-weight:900;letter-spacing:-.05em;color:{C_TEXT};">
            Route<span style="color:{C_ACCENT};">Zero</span>
          </span>
        </div>
        """, unsafe_allow_html=True)

    clicked = None
    with h_col:
        st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("← Home", key=f"dash_home_{current_page}", use_container_width=True):
            clicked = "landing"
    with d_col:
        st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Docs", key=f"dash_docs_{current_page}", use_container_width=True):
            clicked = "docs"
    with c_col:
        st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Connect", key=f"dash_conn_{current_page}", use_container_width=True):
            clicked = "connect"

    st.markdown(f'<div style="height:1px;background:{C_BORDER};margin-bottom:20px;"></div>', unsafe_allow_html=True)
    return clicked

# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph_html(graph_data):
    return None
# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def render_connect_page(back_to: str):
    nav_click = render_dash_nav("connect")
    if nav_click:
        st.session_state["page"] = nav_click; st.rerun()

    if st.button("← Back", key="conn_back"):
        st.session_state["page"] = back_to; st.rerun()

    st.markdown(f'<h2 style="font-size:28px;font-weight:800;letter-spacing:-.03em;color:{C_TEXT};margin:16px 0 4px 0;">Connect Your Stack</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{C_MUTED};font-size:14px;margin:0 0 28px 0;">Three steps, under an hour, fully automated from that point on.</p>', unsafe_allow_html=True)

    steps = [
        ("1","Generate your org config","~5 min","Run the onboarding script. It reads your Jira project structure and GitHub CODEOWNERS file and auto-generates the service-to-team ownership map.","python onboard.py --jira-url https://your-company.atlassian.net --github-repo your-org/your-repo"),
        ("2","Add your credentials","~2 min","Populate four environment variables. DEMO_MODE goes false. RouteZero creates real Jira tickets and sends real Slack notifications on approval.","JIRA_API_TOKEN=your_token\nJIRA_BASE_URL=https://your-company.atlassian.net\nJIRA_EMAIL=you@company.com\nSLACK_WEBHOOK_URL=https://hooks.slack.com/..."),
        ("3","Point your alerts here","~1 min","Add RouteZero's endpoint as a webhook in PagerDuty or OpsGenie. Every alert routes automatically.","POST https://your-routezero-instance.com/incidents\nContent-Type: application/json\n{ \"raw_error_text\": \"...\", \"service\": \"payment-service\" }"),
    ]
    for i,(num,title,est,desc,code) in enumerate(steps):
        st.markdown(f"""<div class="step-card">
          <div style="display:flex;align-items:center;margin-bottom:8px;">
            <div class="step-number">{num}</div>
            <div><div class="step-title">{title}</div><div class="step-time">{est}</div></div>
          </div>
          <div class="step-desc">{desc}</div>
          <div class="code-block">{code}</div>
        </div>""", unsafe_allow_html=True)
        if i<len(steps)-1:
            st.markdown(f'<div style="display:flex;justify-content:flex-start;margin:0 0 4px 23px;"><div style="width:2px;height:14px;background:{C_ACCENT}44;"></div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};margin-bottom:12px;">Supported Integrations</div>', unsafe_allow_html=True)

    jira_connected   = bool(os.getenv("JIRA_API_TOKEN") and os.getenv("JIRA_BASE_URL"))
    slack_connected  = bool(os.getenv("SLACK_WEBHOOK_URL"))
    github_connected = bool(os.getenv("GITHUB_TOKEN"))

    integrations = [
        ("Jira","Ticket creation",C_BLUE,jira_connected,"Connects to any Jira Cloud or Server project. Tickets with correct project, priority, labels, and assignee."),
        ("Slack","Notifications",C_GREEN,slack_connected,"Per-stakeholder messages sent to the team's channel. Role-appropriate content per recipient."),
        ("PagerDuty","Alert ingestion",C_RED,False,"Point your PagerDuty webhook at /incidents. Incidents route automatically with full alert payload."),
        ("OpsGenie","Alert ingestion",C_ORANGE,False,"Same as PagerDuty. Alert fields map to the RichContext schema automatically."),
        ("GitHub","Code graph",C_ACCENT,github_connected,"Reads CODEOWNERS for ownership mapping. Fetches code for Agent 4 architectural analysis."),
        ("GitLab","Code graph",C_ACC2,False,"Same as GitHub. CODEOWNERS + file content API. Roadmap: Q3 2026."),
    ]
    g1,g2,g3 = st.columns(3)
    for i,(name,cat,color,connected,desc) in enumerate(integrations):
        sc = C_GREEN if connected else C_MUTED
        sl = "Connected" if connected else ("Simulated" if DEMO_MODE else "Not connected")
        sd = "●" if connected else "○"
        with [g1,g2,g3][i%3]:
            st.markdown(f"""<div style="background:{C_CARD};border:1px solid {C_BORDER};border-left:3px solid {color};border-radius:8px;padding:16px 20px;margin-bottom:12px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <span style="font-size:14px;font-weight:700;color:{C_TEXT};">{name}</span>
                  <span style="font-size:10px;color:{color};background:{color}18;padding:2px 7px;border-radius:4px;font-weight:600;">{cat}</span>
                </div>
                <span style="font-size:10px;color:{sc};font-weight:600;">{sd} {sl}</span>
              </div>
              <div style="font-size:12px;color:{C_MUTED};line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};margin-bottom:12px;">API Reference</div>', unsafe_allow_html=True)
    for method,path,desc in [
        ("POST","/incidents","Route an incident through all three agents."),
        ("POST","/incidents/{id}/approve","Create the Jira ticket and send notifications."),
        ("POST","/incidents/{id}/resolve","Mark an incident as resolved."),
        ("GET","/incidents","List all incidents."),
        ("GET","/incidents/{id}/detail","Full routing decision and ticket contents."),
        ("POST","/audit","Trigger Agent 4 to run an architectural audit."),
        ("GET","/audit/latest","Return the most recent audit result."),
        ("GET","/graph/nodes","Return code graph nodes with incident counts."),
        ("GET","/stats","Return API call count and table row counts."),
    ]:
        mc = C_GREEN if method=="GET" else C_BLUE
        st.markdown(f"""<div style="display:flex;align-items:flex-start;gap:12px;padding:8px 0;border-bottom:1px solid {C_BORDER};font-size:12px;">
          <span style="background:{mc}18;color:{mc};font-weight:700;font-family:monospace;padding:2px 8px;border-radius:4px;min-width:42px;text-align:center;flex-shrink:0;">{method}</span>
          <span style="color:{C_TEXT};font-family:monospace;flex-shrink:0;min-width:260px;">{path}</span>
          <span style="color:{C_MUTED};">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="background:{C_ACCENT}0a;border:1px solid {C_ACCENT}30;border-radius:10px;padding:22px 26px;margin-top:28px;">
      <div style="font-size:14px;font-weight:700;color:{C_TEXT};margin-bottom:8px;">What you're seeing right now</div>
      <div style="font-size:13px;color:{C_MUTED};line-height:1.7;">
        Running with <code style="color:{C_ACCENT};background:{C_ACCENT}15;padding:1px 6px;border-radius:4px;">DEMO_MODE=true</code>
        and the fictional StreamCo org config. Jira and Slack are simulated.
        Agent 4 is analyzing five pre-seeded historical incidents to demonstrate pattern detection.<br><br>
        To connect a real company: follow the three steps above. The intelligence layer is identical in production.
        Only the integrations change.
      </div>
    </div>""", unsafe_allow_html=True)


def render_docs_page(back_to: str):
    nav_click = render_dash_nav("docs")
    if nav_click:
        st.session_state["page"] = nav_click; st.rerun()

    if st.button("← Back", key="docs_back"):
        st.session_state["page"] = back_to; st.rerun()

    st.markdown(f'<h2 style="font-size:28px;font-weight:800;letter-spacing:-.03em;color:{C_TEXT};margin:16px 0 4px 0;">Documentation</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{C_MUTED};font-size:14px;margin:0 0 28px 0;">Everything you need to use RouteZero effectively.</p>', unsafe_allow_html=True)

    tab_usage, tab_agents, tab_faq, tab_trouble = st.tabs(["How to Use","The Four Agents","FAQ","Troubleshooting"])

    with tab_usage:
        for step_title, step_body in [
            ("Step 1 — Paste your error", "Go to the New Incident tab and paste any error, stack trace, or alert text. Accepts Python tracebacks, Java stack traces, pytest output, plain text alerts, and JSON payloads from PagerDuty or OpsGenie."),
            ("Step 2 — Add context (optional but recommended)", f"Expand the context panel. All fields optional.<br><br><strong style='color:{C_TEXT};'>Most impactful:</strong><br>• Affected users — determines P1 vs P2<br>• Environment — production escalates priority<br>• Recent deployment — enables probable cause<br>• SLA breach minutes — triggers P1 if under 60"),
            ("Step 3 — Review the routing decision", "After clicking Route Incident you see: which team owns this service and why, what priority was assigned and which rule fired, routing confidence percentage, probable cause from deployment timing, and missing context that would improve the decision."),
            ("Step 4 — Review stakeholder notifications", "Three stakeholder cards show who gets notified: Engineer (full technical ticket, Jira assigned), Team Lead (leadership-framed summary, Slack), Manager (five-sentence plain English digest)."),
            ("Step 5 — Approve and Send", "Click Approve and Send. The ticket is created in Jira and all stakeholders are notified simultaneously. Cannot be undone."),
        ]:
            st.markdown(f"""<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;padding:20px 24px;margin-bottom:16px;">
              <div style="font-size:16px;font-weight:700;color:{C_TEXT};margin-bottom:10px;">{step_title}</div>
              <div style="font-size:13px;color:{C_MUTED};line-height:1.7;">{step_body}</div>
            </div>""", unsafe_allow_html=True)

    with tab_agents:
        for color,name,note,desc in [
            (C_BLUE,"A1: Classifier","Zero LLM calls. Fully deterministic.","Reads the raw error and classifies it using regex patterns and keyword scoring against your org config. Detects failure type, service ownership, environment, blast radius. Every decision is traceable to a specific rule."),
            (C_ACCENT,"A2: Router","Rules-first. LLM only on low confidence.","Determines team ownership, priority, stakeholder list. Priority rules are deterministic: critical path + production = P1, SLA breach under 60 min = P1. LLM only consulted when confidence falls below 65%."),
            (C_AMBER,"A3: Ticket Writer","Verified facts only. Hallucination-validated.","Uses Fireworks AI to assemble prose only from facts already verified by Agents 1 and 2. Any number or file token in AI text not present in input facts causes the text to be discarded in favour of a deterministic template."),
            (C_GREEN,"A4: Architectural Auditor","On-demand. Silent when uncertain.","Detects recurring locations (same file:line in 2+ incidents), service stress (3+ incidents, 2+ failure types), cascading failures (cross-service within 30 min). Files PLM ticket only when confidence exceeds 70%."),
        ]:
            st.markdown(f"""<div style="background:{C_CARD};border:1px solid {C_BORDER};border-left:3px solid {color};border-radius:10px;padding:20px 24px;margin-bottom:16px;">
              <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{color};margin-bottom:6px;">{note}</div>
              <div style="font-size:17px;font-weight:800;color:{C_TEXT};margin-bottom:10px;">{name}</div>
              <div style="font-size:13px;color:{C_MUTED};line-height:1.7;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with tab_faq:
        for q,a in [
            ("What error formats does RouteZero accept?","Python tracebacks, Java stack traces, pytest/JUnit output, plain text alerts, and JSON payloads from PagerDuty or OpsGenie. Agent 1 auto-detects the format."),
            ("What if the routing confidence is low?","Low confidence shows in the routing block with missing context chips. Providing those fields and re-routing will increase confidence."),
            ("What does Agent 4 need to work?","At least 2 incidents in DuckDB that share a file and line number in their stack traces. Click Run Audit in the Architectural Intelligence tab."),
            ("What happens if Jira creation fails?","The pipeline completes and ticket preview still appears. The failure is shown in the notification list after approval. No data is lost."),
            ("How is the probable cause detected?","If you provide a recent deployment, Agent 2 checks whether it happened within 60 minutes before the incident. Closer timing = higher confidence."),
            ("What is DEMO_MODE?","When DEMO_MODE=true, Jira and Slack are simulated. Set DEMO_MODE=false and add real credentials to go live."),
        ]:
            with st.expander(q):
                st.markdown(f'<div style="font-size:13px;color:{C_MUTED};line-height:1.7;">{a}</div>', unsafe_allow_html=True)

    with tab_trouble:
        for issue,solution in [
            ("AI assembly unavailable in tickets",f'Check FIREWORKS_API_KEY in .env and that FIREWORKS_MODEL is <code style="color:{C_ACCENT};">accounts/fireworks/models/gemma2-9b-it</code>. Restart the backend after changing .env.'),
            ("Routing confidence always below 70%","Add a service hint matching one of the services in your org config. This bypasses keyword scoring and routes directly."),
            ("Agent 4 finds no patterns","You need 2+ incidents with the same file and line number. The pre-seeded historical incidents should trigger detection on first audit run."),
            ("Graph is empty after running audit","The code graph builds from demo_repo/ on first audit. Check that demo_repo/payment_service/processor.py exists."),
            ("Backend not reachable error","The FastAPI backend is not running. Start it with: uvicorn main:app --port 8000"),
            ("Docker compose fails","Ensure Dockerfile.backend and Dockerfile.frontend exist in docker/. Run pip freeze > requirements.txt from your active venv before building."),
        ]:
            with st.expander(f"⚠ {issue}"):
                st.markdown(f'<div style="font-size:13px;color:{C_MUTED};line-height:1.7;">{solution}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load org config
# ---------------------------------------------------------------------------

org_config = load_org_config()

# ---------------------------------------------------------------------------
# PAGE: LANDING
# ---------------------------------------------------------------------------

if st.session_state["page"] == "landing":
    nav_click = render_nav("landing")
    if nav_click:
        st.session_state["prev_page"] = "landing"
        st.session_state["page"] = nav_click
        st.rerun()

    hero_left, hero_right = st.columns([1, 1], gap="large")

    with hero_left:
        st.markdown(f"""
        <div style="padding-top:48px;">
          <div class="hero-kicker">Zero administrative overhead</div>
          <h1 class="hero-h1">Incident Routing.<br><span>Zero Pain.</span></h1>
          <p class="hero-sub">Paste a stack trace. RouteZero classifies it, routes it to the right team,
            writes the Jira ticket, and notifies every stakeholder with exactly what they need to know.</p>
          <p style="font-size:13px;color:{C_MUTED};margin:0;">Faster routing. More accurate tickets. Less noise.</p>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        st.markdown(f"""
        <div style="padding-top:48px;">
          <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:12px;padding:20px;">
            <div class="panel-label">Paste stack trace &amp; context</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        hero_error = st.text_area("hero_err", height=130, key="hero_error_input",
            placeholder="Traceback (most recent call last):\n  File \"payment_service/processor.py\", line 31\n    ...\nAttributeError: 'NoneType' object has no attribute 'transaction_id'",
            label_visibility="collapsed")

        c1h, c2h = st.columns(2)
        hero_env = c1h.selectbox("Env", ["production","staging","development"], key="hero_env", index=None, placeholder="production", label_visibility="collapsed")
        hero_svc = c2h.text_input("Svc", placeholder="payment-service", key="hero_svc", label_visibility="collapsed")

        if st.button("Approve & Route →", type="primary", key="hero_route_btn", disabled=not hero_error.strip(), use_container_width=True):
            body = {"raw_error_text": hero_error.strip()}
            if hero_env: body["environment"] = hero_env
            if hero_svc.strip(): body["service_hint"] = hero_svc.strip()
            with st.spinner("Routing..."):
                result = api_post("/incidents", body, timeout=LLM_TIMEOUT)
            if result:
                st.session_state["last_output"] = result
                st.session_state["page"] = "dashboard"; st.rerun()

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:12px;">
          <div style="background:{C_CARD2};border:1px solid {C_BORDER};border-radius:7px;padding:8px 12px;text-align:center;">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{C_TEXT2};">Classification</div>
          </div>
          <div style="background:{C_CARD2};border:1px solid {C_BORDER};border-radius:7px;padding:8px 12px;text-align:center;">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{C_TEXT2};">Routing Decision</div>
          </div>
          <div style="background:{C_CARD2};border:1px solid {C_BORDER};border-radius:7px;padding:8px 12px;text-align:center;">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{C_TEXT2};">Jira Ticket Draft</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature pillars
    st.markdown("<div style='height:56px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div class="pillars-grid">
      <div class="pillar"><div class="pillar-icon">🔒</div><div class="pillar-title">Zero LLM for Classification</div><div class="pillar-desc">Agent 1 uses deterministic regex and rules. Every classification is fully auditable and traceable to a specific rule.</div></div>
      <div class="pillar"><div class="pillar-icon">⚡</div><div class="pillar-title">Faster, Not Just Automated</div><div class="pillar-desc">Manual routing, ticket writing, and stakeholder notification eliminated per incident. Multiply across your team.</div></div>
      <div class="pillar"><div class="pillar-icon">✅</div><div class="pillar-title">Traceable to Verified Facts</div><div class="pillar-desc">Every claim in every ticket traces to a verified input. The system never invents a number or file name.</div></div>
    </div>""", unsafe_allow_html=True)

    # Agent pipeline
    st.markdown(f"""<div style="border-top:1px solid {C_BORDER};padding-top:56px;margin-bottom:40px;">
      <div class="section-eyebrow">Four-agent pipeline</div>
      <div class="section-title">Every agent has one job.<br>Together they replace hours of manual work.</div>
    </div>""", unsafe_allow_html=True)

    ag1,arr1,ag2,arr2,ag3,arr3,ag4 = st.columns([4,1,4,1,4,1,4])
    for col,icon,color,name,note,desc in [
        (ag1,"🔍",C_BLUE,"A1: Classifier","Deterministic rules","Zero LLM calls. Every decision auditable."),
        (ag2,"🔀",C_ACCENT,"A2: Router","LLM only on low confidence","Rules-first. Org config aware. Probable cause from deployment timing."),
        (ag3,"📝",C_AMBER,"A3: Ticket Writer","Verified facts only","Role-appropriate per recipient. Hallucination-validated."),
        (ag4,"🧠",C_GREEN,"A4: Auditor","On-demand","Pattern detection + code graph traversal. Proactive PLM flags."),
    ]:
        with col:
            st.markdown(f"""<div style="background:{C_CARD};border:1px solid {C_BORDER};border-top:3px solid {color};border-radius:10px;padding:20px 16px;">
              <div style="width:38px;height:38px;background:{color}18;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:17px;margin-bottom:12px;">{icon}</div>
              <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{color};margin-bottom:4px;">{note}</div>
              <div style="font-size:15px;font-weight:700;color:{C_TEXT};margin-bottom:6px;">{name}</div>
              <div style="font-size:12px;color:{C_MUTED};line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    for arr in [arr1,arr2,arr3]:
        with arr:
            st.markdown(f'<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:20px;color:{C_BORDER};padding-top:40px;">→</div>', unsafe_allow_html=True)

    # Arch intelligence
    st.markdown("<div style='height:56px;'></div>", unsafe_allow_html=True)
    arch_left, arch_right = st.columns([1,1], gap="large")

    with arch_left:
        st.markdown(f'<div style="height:360px;background:{C_CARD2};border:1px solid {C_BORDER};border-radius:10px;display:flex;align-items:center;justify-content:center;color:{C_TEXT2};font-size:13px;">Graph available after audit — visit the dashboard</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="display:flex;gap:16px;margin-top:10px;font-size:11px;color:{C_MUTED};"><span><span style="color:{C_RED};">●</span> 2+ incidents</span><span><span style="color:{C_ORANGE};">●</span> connected</span><span><span style="color:#888;">●</span> clean</span></div>', unsafe_allow_html=True)

    with arch_right:
        st.markdown(f"""<div style="padding-top:40px;">
          <div class="section-eyebrow">Agent 4</div>
          <div class="arch-title">Architectural Intelligence</div>
          <p class="arch-desc">Agent 4 mines your incident history for patterns invisible to individual tickets. It builds a code knowledge graph and traverses it to find structural weaknesses — then files a proactive PLM ticket before the next incident happens.</p>
          <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:28px;">
            <div style="display:flex;align-items:flex-start;gap:10px;"><div style="width:6px;height:6px;border-radius:50%;background:{C_RED};margin-top:5px;flex-shrink:0;"></div><div style="font-size:13px;color:{C_MUTED};">Detects recurring code locations across multiple incidents</div></div>
            <div style="display:flex;align-items:flex-start;gap:10px;"><div style="width:6px;height:6px;border-radius:50%;background:{C_ORANGE};margin-top:5px;flex-shrink:0;"></div><div style="font-size:13px;color:{C_MUTED};">Traverses the code graph to find connected structural weaknesses</div></div>
            <div style="display:flex;align-items:flex-start;gap:10px;"><div style="width:6px;height:6px;border-radius:50%;background:{C_ACCENT};margin-top:5px;flex-shrink:0;"></div><div style="font-size:13px;color:{C_MUTED};">Files PLM tickets with developer attribution before the next incident</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<style>#arch_btn{{color:{C_TEXT2}!important;}} button[data-testid="baseButton-secondary"]{{color:{C_TEXT2}!important;}}</style>', unsafe_allow_html=True)
        if st.button("View Architectural Intelligence →", key="arch_btn"):
            st.session_state["page"] = "dashboard"; st.rerun()

    st.markdown(f"""<div class="integrations-row">
      <div class="int-badge" style="color:{C_MUTED};">Works with</div>
      <div class="int-badge">🔵 Jira</div>
      <div class="int-badge">🟢 Slack</div>
      <div class="int-badge">🔴 PagerDuty</div>
      <div class="int-badge">🟠 OpsGenie</div>
      <div class="int-badge">⚫ GitHub</div>
    </div>
    <div style="text-align:center;padding-bottom:32px;">
      <div style="font-size:11px;color:{C_MUTED};">MIT license · RouteZero</div>
    </div>""", unsafe_allow_html=True)

    st.stop()

# ---------------------------------------------------------------------------
# PAGE: DOCS
# ---------------------------------------------------------------------------

if st.session_state["page"] == "docs":
    render_docs_page(back_to=st.session_state.get("prev_page","landing"))
    st.stop()

# ---------------------------------------------------------------------------
# PAGE: CONNECT
# ---------------------------------------------------------------------------

if st.session_state["page"] == "connect":
    render_connect_page(back_to=st.session_state.get("prev_page","dashboard"))
    st.stop()

# ---------------------------------------------------------------------------
# PAGE: PRODUCT
# ---------------------------------------------------------------------------

if st.session_state["page"] == "product":
    st.session_state["page"] = "landing"
    st.rerun()
# ---------------------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------------------

nav_click = render_dash_nav("dashboard")
if nav_click:
    st.session_state["prev_page"] = "dashboard"
    st.session_state["page"] = nav_click
    st.rerun()

try:
    stats              = api_get("/stats") or {}
    all_incidents_data = api_get("/incidents") or []
    table_counts       = stats.get("table_counts") or {}
    total_routed       = table_counts.get("gold_incident_intelligence", 0)
    open_p1s           = sum(1 for i in all_incidents_data if i.get("priority")=="P1" and not i.get("resolved"))
    resolved_ct        = sum(1 for i in all_incidents_data if i.get("resolved"))
    fw_calls           = stats.get("fireworks_calls", 0)
    latest_audit_data  = api_get("/audit/latest")
    last_audit_ts      = fmt_ts(latest_audit_data.get("timestamp")) if latest_audit_data else "Never"
except Exception:
    stats = {}; all_incidents_data = []; total_routed = 0
    open_p1s = 0; resolved_ct = 0; fw_calls = 0; last_audit_ts = "—"

p1c = C_RED if open_p1s>0 else C_GREEN
st.markdown(f"""<div class="status-bar">
  <div class="status-item"><div class="status-value" style="color:{p1c};">{open_p1s}</div><div class="status-label">Open P1 incidents</div></div>
  <div class="status-item"><div class="status-value" style="color:{C_TEXT};">{total_routed}</div><div class="status-label">Total routed</div></div>
  <div class="status-item"><div class="status-value" style="color:{C_GREEN};">{resolved_ct}</div><div class="status-label">Resolved</div></div>
  <div class="status-item"><div class="status-value" style="color:{C_MUTED};">{last_audit_ts}</div><div class="status-label">Last audit</div></div>
  <div class="status-item"><div class="status-value" style="color:{C_ACCENT};">{fw_calls}</div><div class="status-label">AI calls</div></div>
</div>""", unsafe_allow_html=True)

tab_new, tab_history, tab_intel = st.tabs(["⚡  New Incident","📋  Incident History","🧠  Architectural Intelligence"])

# ── TAB 1 ────────────────────────────────────────────────────────────────────
with tab_new:
    st.markdown(f"""<div style="margin-top:8px;">
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};margin-bottom:10px;display:flex;align-items:center;gap:8px;">
        Paste error or stack trace
        <span style="flex:1;height:1px;background:{C_BORDER};display:inline-block;"></span>
      </div>
    </div>""", unsafe_allow_html=True)

    raw_error_text = st.text_area("err", height=160, key="raw_error_text",
        placeholder="Traceback (most recent call last):\n  File \"payment_service/processor.py\", line 31, in process_payment\n    ...\nAttributeError: 'NoneType' object has no attribute 'transaction_id'",
        label_visibility="collapsed")

    uploaded_files = st.file_uploader(
        "Attach files",
        accept_multiple_files=True,
        type=["txt","log","py","js","ts","java","go","rb","cpp","json","csv","md"],
        key="file_uploads",
        label_visibility="collapsed",
    )
    file_context = ""
    if uploaded_files:
        st.markdown(
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">' +
            "".join(f'<span style="background:{C_CARD2};border:1px solid {C_BORDER};border-radius:4px;padding:2px 10px;font-size:11px;color:{C_TEXT2};font-family:monospace;">{f.name}</span>' for f in uploaded_files) +
            '</div>', unsafe_allow_html=True)
        for f in uploaded_files:
            try:
                content = f.read().decode("utf-8", errors="ignore")
                if f.name.endswith(".csv"):
                    lines = content.split("\n")
                    content = "\n".join(lines[:50]) + ("\n... (truncated)" if len(lines)>50 else "")
                file_context += f"\n\n--- FILE: {f.name} ---\n{content}"
            except Exception:
                pass

    with st.expander("Add context  —  improves routing confidence", expanded=False):
        c1,c2,c3 = st.columns(3)
        service_hint  = c1.text_input("Service hint",key="ctx_sh",placeholder="payment-service")
        environment   = c2.selectbox("Environment",["production","staging","development"],key="ctx_env", index=None, placeholder="Select environment")
        occurrences   = c3.number_input("Occurrences last 4h",min_value=0,step=1,value=None,key="ctx_occ")
        c4,c5,c6 = st.columns(3)
        affected_users     = c4.number_input("Affected users",min_value=0,step=1,value=None,key="ctx_au")
        customer_tier      = c5.text_input("Customer tier",key="ctx_ct",placeholder="enterprise")
        sla_breach_minutes = c6.number_input("Minutes until SLA breach",min_value=0,step=1,value=None,key="ctx_sla")
        c7,c8 = st.columns(2)
        on_call_engineer   = c7.text_input("On-call engineer",key="ctx_oc",placeholder="marcus.webb")
        runbook_url        = c8.text_input("Runbook URL",key="ctx_ru",placeholder="https://...")
        related_tickets_raw = st.text_input("Related ticket IDs",key="ctx_rt",placeholder="PAY-2847, PAY-2901")
        st.markdown(f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-top:12px;margin-bottom:8px;">Recent deployment</div>', unsafe_allow_html=True)
        d1,d2 = st.columns(2)
        deployer             = d1.text_input("Deployed by",key="ctx_dep",placeholder="ana.rodriguez")
        commit_hash          = d2.text_input("Commit hash",key="ctx_ch",placeholder="a3f92bc")
        commit_message       = st.text_input("Commit message",key="ctx_cm",placeholder="refactor payment processor null handling")
        deployed_minutes_ago = st.number_input("Deployed how many minutes ago?",min_value=0,step=1,value=None,key="ctx_dma")

    def build_body():
        combined = raw_error_text.strip()
        if file_context: combined += file_context
        body = {"raw_error_text": combined}
        if service_hint.strip(): body["service_hint"]=service_hint.strip()
        if environment:          body["environment"]=environment
        if occurrences is not None: body["occurrences_last_4h"]=int(occurrences)
        if affected_users is not None: body["affected_users"]=int(affected_users)
        if customer_tier.strip(): body["customer_tier"]=customer_tier.strip()
        if deployer.strip() and commit_hash.strip() and commit_message.strip():
            mins = int(deployed_minutes_ago) if deployed_minutes_ago is not None else 0
            body["recent_deployment"]={"deployer":deployer.strip(),"commit_hash":commit_hash.strip(),"commit_message":commit_message.strip(),"deployed_at":(datetime.now(timezone.utc)-timedelta(minutes=mins)).isoformat()}
        if sla_breach_minutes is not None: body["sla_breach_minutes"]=int(sla_breach_minutes)
        if on_call_engineer.strip(): body["on_call_engineer"]=on_call_engineer.strip()
        ids=[t.strip() for t in related_tickets_raw.split(",") if t.strip()]
        if ids: body["related_ticket_ids"]=ids
        if runbook_url.strip(): body["runbook_url"]=runbook_url.strip()
        return body

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    if st.button("⚡  Route Incident", type="primary", disabled=not raw_error_text.strip() and not file_context):
        with st.spinner("Classifier → Router → Ticket Writer..."):
            result = api_post("/incidents", build_body(), timeout=LLM_TIMEOUT)
        if result:
            st.session_state["last_output"] = result
            st.session_state.pop("approve_result", None)

    output = st.session_state.get("last_output")
    if output:
        routing    = output.get("routing_decision") or {}
        confidence = clamp01(routing.get("routing_confidence"))
        priority   = routing.get("priority","P3")
        p_color    = PRIORITY_COLOR.get(priority,C_BLUE)
        tickets    = output.get("ticket_contents") or []
        missing    = routing.get("missing_context") or []

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="routing-block">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
            <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};">Routing Decision</div>
            {priority_badge(priority)}
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:16px;">
            <div>
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-bottom:5px;">Owning Team</div>
              <div style="font-size:17px;font-weight:800;color:{C_TEXT};letter-spacing:-.02em;">{routing.get('owning_team','—')}</div>
            </div>
            <div>
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-bottom:5px;">Priority</div>
              <div style="font-size:17px;font-weight:800;color:{p_color};letter-spacing:-.02em;">{priority}</div>
            </div>
            <div>
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-bottom:5px;">Assignee</div>
              <div style="font-size:17px;font-weight:800;color:{C_TEXT};letter-spacing:-.02em;">{routing.get('assignee','—')}</div>
            </div>
          </div>
          {confidence_bar(confidence)}
          <div style="font-size:12px;color:{C_MUTED};margin-top:12px;line-height:1.7;">
            <strong style="color:{C_TEXT};">Why {priority}:</strong> {routing.get('priority_reasoning','—')}
          </div>
        </div>""", unsafe_allow_html=True)

        if missing:
            chips="".join(f'<span class="ctx-chip">{f}</span>' for f in missing)
            st.markdown(f'<div style="padding:10px 16px;background:{C_CARD};border:1px solid {C_BORDER};border-top:none;border-radius:0 0 10px 10px;margin-top:-8px;"><span style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-right:8px;">Missing context:</span>{chips}</div>', unsafe_allow_html=True)

        cause = routing.get("probable_cause")
        if cause:
            st.markdown(f"""<div class="cause-callout">
              <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_AMBER};margin-bottom:8px;">⚠  Probable Cause — {fmt_pct(cause.get('confidence'))} confidence</div>
              <div style="font-size:13px;color:{C_TEXT};line-height:1.7;">{cause.get('description','')}</div>
              <div style="font-size:11px;color:{C_MUTED};margin-top:8px;font-family:'JetBrains Mono',monospace;background:{C_BG};padding:8px 10px;border-radius:5px;border:1px solid {C_BORDER};">
                commit <span style="color:{C_AMBER};">{cause.get('commit_hash','?')}</span> by {cause.get('deployer','?')} — "{cause.get('commit_message','')}" ({cause.get('minutes_before_incident','?')} min before)
              </div>
            </div>""", unsafe_allow_html=True)

        if tickets:
            st.markdown(f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};margin:28px 0 12px 0;">Stakeholder Notifications</div>', unsafe_allow_html=True)
            role_cfg = {
                "assignee":  (C_BLUE, "👨‍💻","Engineer",  "Full ticket + Jira assigned"),
                "team_lead": (C_ACC2, "🎯","Team Lead", "Full ticket + Slack"),
                "manager":   (C_AMBER,"📊","Manager",   "5-sentence digest + Slack"),
            }
            card_cols = st.columns(len(tickets))
            for col,ticket in zip(card_cols,tickets):
                role=ticket.get("recipient_role",""); name=ticket.get("recipient_name","?")
                ot=output_type_label(ticket.get("output_type",""))
                color,icon,rl,delivery=role_cfg.get(role,(C_MUTED,"👤",role.replace("_"," ").title(),"Notified"))
                with col:
                    st.markdown(f"""<div style="background:{C_CARD2};border:1px solid {C_BORDER};border-top:3px solid {color};border-radius:8px;padding:14px 16px;">
                      <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;">
                        <span style="font-size:14px;">{icon}</span>
                        <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{color};">{rl}</span>
                      </div>
                      <div style="font-size:14px;font-weight:700;color:{C_TEXT2};margin-bottom:4px;">{name}</div>
                      <div style="font-size:11px;color:#aaa;margin-bottom:10px;">{ot}</div>
                      <div style="height:1px;background:#333;margin-bottom:10px;"></div>
                      <div style="font-size:10px;color:#aaa;"><span style="color:{color};">→</span> {delivery}</div>
                    </div>""", unsafe_allow_html=True)

            eng = next((t for t in tickets if t.get("recipient_role")=="assignee"),tickets[0])
            jira_proj = routing.get("jira_project_key") or routing.get("owning_team","PROJ")
            assignee  = routing.get("assignee","—")
            st.markdown(f"""<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;margin-top:16px;overflow:hidden;">
              <div style="background:{C_CARD2};padding:14px 20px;border-bottom:1px solid {C_BORDER};display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#aaa;">Jira Ticket Preview</span>
                <span style="background:{C_BLUE}20;color:{C_BLUE};font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;font-family:monospace;">{jira_proj}</span>
                {priority_badge(priority)}
                <span style="margin-left:auto;font-size:11px;color:#aaa;">Assigned to <strong style="color:{C_TEXT2};">{assignee}</strong></span>
              </div>
              <div style="padding:20px 24px;">
                <div style="font-size:15px;font-weight:700;color:{C_TEXT};margin-bottom:16px;line-height:1.4;">{eng.get('title','')}</div>
              </div>
            </div>""", unsafe_allow_html=True)
            with st.expander("View full ticket content", expanded=False):
                st.markdown(eng.get("body",""))
            for ticket in [t for t in tickets if t.get("recipient_role")!="assignee"]:
                role=ticket.get("recipient_role",""); name=ticket.get("recipient_name","?")
                color,icon,rl,_=role_cfg.get(role,(C_MUTED,"👤",role.replace("_"," ").title(),""))
                with st.expander(f"{icon} {rl} message — {name}", expanded=False):
                    st.markdown(ticket.get("body",""))

        approve_result = st.session_state.get("approve_result")
        if approve_result is None and tickets:
            jira_proj = routing.get("jira_project_key") or routing.get("owning_team","")
            assignee  = routing.get("assignee","—")
            st.markdown(f"""<div class="approve-zone" style="margin-top:20px;">
              <div style="display:flex;align-items:flex-start;gap:14px;">
                <span style="font-size:22px;margin-top:2px;">✅</span>
                <div style="flex:1;">
                  <div style="font-size:15px;font-weight:700;color:{C_TEXT};margin-bottom:6px;">Ready to send</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:14px;">
                    <div><div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-bottom:3px;">Jira project</div><div style="font-size:13px;font-weight:600;color:{C_TEXT};">{jira_proj}</div></div>
                    <div><div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-bottom:3px;">Assigned to</div><div style="font-size:13px;font-weight:600;color:{C_TEXT};">{assignee}</div></div>
                    <div><div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-bottom:3px;">Notifying</div><div style="font-size:13px;font-weight:600;color:{C_TEXT};">{len(tickets)} stakeholders</div></div>
                  </div>
                  <div style="font-size:11px;color:{C_MUTED};">Each stakeholder receives role-appropriate content. <span style="color:{C_AMBER};">This action cannot be undone.</span></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button("✅  Approve and Send", type="primary", key="approve_btn", use_container_width=True):
                inc_id = output.get("incident_id","")
                with st.spinner("Creating Jira ticket and sending notifications..."):
                    approved = api_post(f"/incidents/{inc_id}/approve", timeout=APPROVE_TIMEOUT)
                if approved:
                    st.session_state["approve_result"] = approved; st.rerun()
        elif approve_result:
            jira_id  = approve_result.get("jira_ticket_id") or "ticket"
            jira_url = approve_result.get("jira_url")
            link     = f"[{jira_id}]({jira_url})" if jira_url else jira_id
            st.success(f"Jira ticket **{link}** created and all stakeholders notified.")
            for n in (approve_result.get("notifications") or []):
                icon="✅" if n.get("success") else "⚠️"
                st.markdown(f"- {icon} **{n.get('recipient','?')}** via `{n.get('channel','?')}` — {output_type_label(n.get('output_type',''))}")

# ── TAB 2 ────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown(f'<h2 style="margin:16px 0 4px 0;font-size:24px;font-weight:800;letter-spacing:-.03em;color:{C_TEXT};">Incident History</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{C_MUTED};font-size:13px;margin:0 0 20px 0;">All routed incidents. Expand any row for full routing decision and ticket drafts.</p>', unsafe_allow_html=True)

    incidents = all_incidents_data
    now_utc = datetime.now(timezone.utc)
    rc2 = {}
    for inc in incidents:
        ts=parse_ts(inc.get("timestamp"))
        if ts and (now_utc-ts)<=timedelta(days=7):
            s=inc.get("service","unknown"); rc2[s]=rc2.get(s,0)+1
    for s,c in sorted(rc2.items()):
        if c>=3: st.markdown(f'<div class="stress-warning">🔴 <strong>Service stress:</strong> <code>{s}</code> — {c} incidents in 7 days.</div>', unsafe_allow_html=True)

    svc_opts=(["All"]+sorted(org_config["services"].keys())) if org_config and org_config.get("services") else ["All"]+sorted({i.get("service") for i in incidents if i.get("service")})
    f1,f2,f3=st.columns(3)
    sf=f1.selectbox("Service",svc_opts,key="hs")
    pf=f2.selectbox("Priority",["All","P1","P2","P3"],key="hp")
    dr=f3.date_input("Date range",value=(date.today()-timedelta(days=30),date.today()),key="hd")

    filtered=[]
    for inc in incidents:
        if sf!="All" and inc.get("service")!=sf: continue
        if pf!="All" and inc.get("priority")!=pf: continue
        if isinstance(dr,(tuple,list)) and len(dr)==2:
            ts=parse_ts(inc.get("timestamp"))
            if ts:
                s2,e2=dr
                if not(s2<=ts.date()<=e2): continue
        filtered.append(inc)

    if not incidents: st.markdown(f'<div style="text-align:center;padding:40px;color:{C_MUTED};font-size:14px;">No incidents yet.</div>', unsafe_allow_html=True)
    elif not filtered: st.markdown(f'<div style="text-align:center;padding:40px;color:{C_MUTED};font-size:14px;">No incidents match.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(pd.DataFrame([{"ID":i.get("incident_id","—"),"Service":i.get("service","—"),"Failure":i.get("failure_type","—"),"Priority":i.get("priority","—"),"When":fmt_ts(i.get("timestamp")),"Users":i.get("affected_users") or "—","Status":"✅ Resolved" if i.get("resolved") else "🔴 Open"} for i in filtered]),use_container_width=True,hide_index=True)
        dc=st.session_state.setdefault("detail_cache",{})
        for inc in filtered:
            iid=inc.get("incident_id",""); res=bool(inc.get("resolved")); p=inc.get("priority","P3")
            hdr=f"{priority_badge(p)} {iid} — {inc.get('service','?')} — {inc.get('failure_type','?')}"
            if res: hdr+=" ✅"
            with st.expander(hdr):
                if inc.get("agent3_summary"): st.markdown(f'<div style="font-size:13px;color:{C_MUTED};margin-bottom:12px;font-style:italic;">{inc["agent3_summary"]}</div>', unsafe_allow_html=True)
                if iid not in dc: dc[iid]=api_get(f"/incidents/{iid}/detail")
                detail=dc.get(iid)
                if detail:
                    rou=detail.get("routing") or {}
                    if rou:
                        rc1a,rc2a,rc3a=st.columns(3)
                        rc1a.metric("Team",rou.get("owning_team","—")); rc2a.metric("Assignee",rou.get("assignee","—")); rc3a.metric("Confidence",fmt_pct(rou.get("routing_confidence")))
                        if rou.get("priority_reasoning"): st.markdown(f'<div style="font-size:12px;color:{C_MUTED};margin:8px 0;"><strong style="color:{C_TEXT};">Why:</strong> {rou["priority_reasoning"]}</div>', unsafe_allow_html=True)
                    to=detail.get("ticket_output") or {}
                    if to.get("jira_ticket_id"):
                        ju=to.get("jira_url"); lk=f"[{to['jira_ticket_id']}]({ju})" if ju else to["jira_ticket_id"]
                        st.markdown(f"**Jira:** {lk}")
                    for ticket in to.get("ticket_contents") or []:
                        st.markdown("---"); st.markdown(f'**{ticket.get("recipient_name","?")} — {output_type_label(ticket.get("output_type",""))}**')
                        if ticket.get("title"): st.markdown(f'*{ticket["title"]}*')
                        st.markdown(ticket.get("body",""))
                else: st.caption("Full detail unavailable.")
                if inc.get("probable_cause"): st.caption(f"Probable cause: {inc['probable_cause']}")
                if not res:
                    if st.button("Mark Resolved",key=f"res_{iid}"):
                        resp=api_post(f"/incidents/{iid}/resolve")
                        if resp is not None: st.session_state["detail_cache"].pop(iid,None); st.rerun()
                else:
                    mins=inc.get("resolution_time_minutes")
                    if mins is not None: st.caption(f"Resolved in {mins} minutes.")

# ── TAB 3 ────────────────────────────────────────────────────────────────────
with tab_intel:
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    hcol,bcol=st.columns([5,1])
    with hcol:
        st.markdown(f'<h2 style="margin:0 0 4px 0;font-size:24px;font-weight:800;letter-spacing:-.03em;color:{C_TEXT};">Architectural Intelligence</h2>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{C_MUTED};font-size:13px;margin:0 0 20px 0;">Agent 4 mines incident history for recurring patterns and files proactive architectural flags.</p>', unsafe_allow_html=True)
    with bcol:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        run_audit=st.button("🔍  Run Audit",type="primary")

    latest_audit=api_get("/audit/latest")
    if run_audit:
        with st.spinner("Agent 4 analyzing incident history and code graph..."):
            ar=api_post("/audit",timeout=LLM_TIMEOUT)
        if ar: latest_audit=ar; st.success("Audit complete.")

    if latest_audit:
        a1,a2,a3,a4=st.columns(4)
        fl=(latest_audit or {}).get("flags") or []; plmc=sum(1 for f in fl if f.get("plm_ticket_id"))
        for col,val,lbl,color in [(a1,fmt_ts(latest_audit.get("timestamp")),"Last audit",C_TEXT),(a2,str(latest_audit.get("incidents_analyzed",0)),"Incidents analyzed",C_TEXT),(a3,str(latest_audit.get("patterns_found",0)),"Patterns found",C_ACCENT),(a4,str(plmc),"PLM tickets filed",C_RED if plmc else C_TEXT)]:
            col.markdown(f'<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;padding:14px 16px;text-align:center;"><div style="font-size:18px;font-weight:800;color:{color};letter-spacing:-.02em;">{val}</div><div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:{C_MUTED};margin-top:3px;">{lbl}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="text-align:center;padding:30px;color:{C_MUTED};font-size:14px;background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;">No audit yet. Click Run Audit.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    flags=(latest_audit or {}).get("flags") or []
    if flags:
        st.markdown(f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};margin-bottom:14px;">Architectural Flags</div>', unsafe_allow_html=True)
        for flag in flags:
            pattern=str(flag.get("pattern_type","unknown")).replace("_"," ").upper(); svc=flag.get("affected_service","?"); conf=clamp01(flag.get("confidence",0)); plm_id=flag.get("plm_ticket_id")
            locs=flag.get("flagged_locations") or []; loc_text=", ".join(f'`{l.get("file_path","?")}:{l.get("line_number","?")}`' for l in locs); contributing=flag.get("contributing_incident_ids") or []
            plm_html=f'<span class="badge badge-p1">PLM: {plm_id}</span>' if plm_id else f'<span class="badge badge-warn">No PLM ticket</span>'
            st.markdown(f"""<div class="flag-card">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
                <span class="badge badge-p1">{pattern}</span>
                <span style="font-size:14px;font-weight:700;color:{C_TEXT2};">{svc}</span>
                {plm_html}
                <span style="margin-left:auto;font-size:12px;color:#aaa;">confidence {int(conf*100)}%</span>
              </div>
              {confidence_bar(conf)}
              <div style="font-size:13px;color:{C_TEXT2};margin-top:12px;line-height:1.7;">{flag.get('assessment','')}</div>
            </div>""", unsafe_allow_html=True)
            if loc_text: st.markdown(f"**Flagged locations:** {loc_text}")
            if contributing: st.caption("Contributing incidents: "+", ".join(contributing))

    st.markdown(f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{C_MUTED};margin:24px 0 8px 0;">Code Knowledge Graph</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;padding:60px;color:{C_MUTED};font-size:14px;background:{C_CARD};border:1px solid {C_BORDER};border-radius:8px;">Code graph — run an audit to populate.</div>', unsafe_allow_html=True)
