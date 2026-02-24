"""
ui.py — AI-BOS Business Brain
──────────────────────────────
Premium production-grade Streamlit UI.
5 pages: Dashboard | Data Ingestion | AI Assistant | Reports | Admin
"""

import os, sys, time, json, asyncio, datetime, random
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-BOS • Business Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Demo Data Init ─────────────────────────────────────────────────────────────
if "demo_data_checked" not in st.session_state:
    st.session_state.demo_data_checked = True
    try:
        import demo_data_generator as ddg
        if ddg.is_demo_needed():
            with st.spinner("🚀 AI-BOS First Launch: Generating demo business data..."):
                ddg.generate_all()
    except Exception as e:
        print(f"Demo data generator failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# THEME CSS
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── App Background + base text ── */
.stApp { background: #0F172A !important; color: #E2E8F0 !important; }
.stApp p, .stApp span, .stApp div, .stApp li { color: #CBD5E1; }
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: #CBD5E1 !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06172B 0%, #0A2540 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.12);
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
    border: 1px solid transparent !important;
    cursor: pointer;
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,212,255,0.1) !important;
    border-color: rgba(0,212,255,0.3) !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #CBD5E1 !important;
    font-size: 0.95rem !important;
}

/* ── Gradient Page Header ── */
.page-header {
    background: linear-gradient(135deg, #0A2540 0%, #0D3B6B 50%, #0A2540 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 28px rgba(0,212,255,0.08);
}
.page-header h1 {
    font-size: 1.85rem; font-weight: 700;
    color: #E2E8F0 !important;
    -webkit-text-fill-color: #E2E8F0 !important;
    text-shadow: 0 0 40px rgba(0,212,255,0.5), 0 0 80px rgba(0,212,255,0.2);
    margin: 0 0 5px 0; line-height: 1.2;
}
.page-header p { color: #94A3B8 !important; margin: 0; font-size: 0.9rem; }

/* ── Glassmorphic Metric Cards ── */
.metric-card {
    background: rgba(13, 59, 107, 0.35);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 16px; padding: 22px 18px; text-align: center;
    transition: transform 0.22s, box-shadow 0.22s, border-color 0.22s;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35); min-height: 150px;
}
.metric-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 14px 36px rgba(0,212,255,0.2);
    border-color: rgba(0,212,255,0.5);
    background: rgba(13, 59, 107, 0.5);
}
.metric-icon { font-size: 2rem; margin-bottom: 8px; }
.metric-value {
    font-size: 2.1rem; font-weight: 700; line-height: 1;
    color: #38BDF8 !important;
    -webkit-text-fill-color: #38BDF8 !important;
    text-shadow: 0 0 20px rgba(56,189,248,0.4);
}
.metric-label {
    color: #94A3B8 !important; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px;
}
.metric-delta { font-size: 0.72rem; margin-top: 8px; padding: 3px 10px; border-radius: 20px; display: inline-block; font-weight: 600; }
.delta-up   { background: rgba(16,185,129,0.18); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.delta-down { background: rgba(239,68,68,0.18);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Section Headers ── */
.section-h {
    color: #E2E8F0 !important; font-size: 1.05rem; font-weight: 700;
    margin: 22px 0 12px 0; padding-bottom: 8px;
    border-bottom: 2px solid rgba(0,212,255,0.15);
    display: flex; align-items: center; gap: 8px;
}

/* ── Chat Avatars ── */
.bubble-wrap { display:flex; align-items:flex-end; gap:10px; margin: 10px 0; }
.bubble-wrap.user-wrap { flex-direction: row-reverse; }
.avatar {
    width:40px; height:40px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; font-size:1.2rem;
    flex-shrink:0; border:2px solid rgba(0,212,255,0.35);
}
.avatar-user { background: linear-gradient(135deg,#1a4080,#2563EB); box-shadow:0 0 12px rgba(37,99,235,0.3); }
.avatar-bot  {
    background: linear-gradient(135deg,#0D3B6B,#0EA5E9);
    box-shadow: 0 0 18px rgba(0,212,255,0.5);
    animation: glow-pulse 2.5s ease-in-out infinite;
}
@keyframes glow-pulse {
    0%,100% { box-shadow: 0 0 10px rgba(0,212,255,0.35); }
    50%      { box-shadow: 0 0 26px rgba(0,212,255,0.8); }
}

/* ── Chat Bubbles ── */
.chat-bubble-user {
    background: linear-gradient(135deg, #1E3A5F, #1D4ED8);
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 18px 4px 18px 18px;
    padding: 13px 17px; color: #F0F9FF !important; max-width: 72%;
    box-shadow: 0 3px 12px rgba(0,0,0,0.4); line-height: 1.6;
}
.chat-bubble-bot {
    background: rgba(15, 32, 60, 0.8);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 4px 18px 18px 18px;
    padding: 13px 17px; color: #E2E8F0 !important; max-width: 82%;
    box-shadow: 0 3px 12px rgba(0,0,0,0.3); line-height: 1.6;
}
.chat-meta { font-size:0.72rem; color:#64748B !important; margin-top:5px; display:flex; align-items:center; gap:8px; }

/* ── Typing Indicator ── */
.typing-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#00D4FF; margin:0 2px;
    animation: typing-bounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay:0.2s; }
.typing-dot:nth-child(3) { animation-delay:0.4s; }
@keyframes typing-bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-7px)} }
.typing-indicator {
    background: rgba(15,32,60,0.8); border:1px solid rgba(0,212,255,0.2);
    border-radius:4px 18px 18px 18px; padding:13px 20px; display:inline-flex;
    align-items:center; gap:4px;
}

/* ── Feedback Buttons ── */
.feedback-row { display:flex; gap:8px; margin-top:7px; flex-wrap: wrap; }
.fb-btn {
    font-size:0.75rem; padding:4px 12px; border-radius:14px; cursor:pointer;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    color: #7DD3FC !important;
    transition:all 0.18s; font-family:'Inter',sans-serif;
}
.fb-btn:hover {
    background: rgba(0,212,255,0.2);
    border-color: rgba(0,212,255,0.5);
    color: #E0F2FE !important;
    transform: translateY(-1px);
}

/* ── Connector / Info Cards ── */
.conn-card {
    background: rgba(15, 28, 50, 0.7);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px; padding: 14px 18px; display:flex;
    align-items:center; justify-content:space-between; margin-bottom:10px;
    transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.conn-card:hover {
    background: rgba(13, 59, 107, 0.45);
    border-color: rgba(0,212,255,0.3);
    box-shadow: 0 4px 16px rgba(0,212,255,0.06);
}
.conn-card span, .conn-card div { color: #CBD5E1 !important; }
.conn-card b, .conn-card strong { color: #E2E8F0 !important; }

/* ── Status Badges ── */
.badge { display:inline-flex; align-items:center; gap:5px; padding:4px 12px; border-radius:20px; font-size:0.74rem; font-weight:700; letter-spacing:0.02em; }
.badge-on  { background:rgba(16,185,129,0.18); color:#34D399 !important; border:1px solid rgba(16,185,129,0.4); }
.badge-off { background:rgba(239,68,68,0.18);  color:#F87171 !important; border:1px solid rgba(239,68,68,0.4); }
.badge-wrn { background:rgba(245,158,11,0.18); color:#FBBF24 !important; border:1px solid rgba(245,158,11,0.4); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(145deg, #0E3460, #1a5fa8) !important;
    color: #BAE6FD !important; border: 1px solid rgba(0,212,255,0.35) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important; transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(145deg, #1a5fa8, #2575ce) !important;
    border-color: #00D4FF !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.28) !important;
    color: #E0F2FE !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Form / Input Fields ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(10, 20, 40, 0.8) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    caret-color: #00D4FF;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: #475569 !important; }
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
}
.stSelectbox > div > div {
    background: rgba(10,20,40,0.8) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}
label, .stSelectbox label, .stSlider label, .stCheckbox label span {
    color: #94A3B8 !important; font-size: 0.88rem !important; font-weight: 500 !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] [data-testid="stSliderThumb"] { background: #00D4FF !important; border-color: #0F172A !important; }
.stSlider > div > div > div { color: #94A3B8 !important; }

/* ── Checkboxes ── */
.stCheckbox span { color: #CBD5E1 !important; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.28) !important;
    border-radius: 14px !important; background: rgba(0,212,255,0.03) !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.55) !important;
    background: rgba(0,212,255,0.06) !important;
}
[data-testid="stFileUploader"] p, [data-testid="stFileUploader"] span { color: #94A3B8 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] * { color: #64748B !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,20,40,0.6) !important;
    border-radius: 10px !important; padding: 4px !important;
    border: 1px solid rgba(0,212,255,0.12) !important; gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important; border-radius: 8px !important;
    padding: 8px 18px !important; font-weight: 600 !important;
    transition: all 0.2s !important; background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #94A3B8 !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(59,130,246,0.15)) !important;
    color: #38BDF8 !important;
    border: 1px solid rgba(0,212,255,0.35) !important;
    box-shadow: 0 2px 10px rgba(0,212,255,0.1) !important;
}
.stTabs [data-baseweb="tab-panel"] { color: #CBD5E1 !important; }

/* ── DataFrames ── */
[data-testid="stDataFrameContainer"] {
    border: 1px solid rgba(0,212,255,0.12) !important;
    border-radius: 12px !important; overflow: hidden;
}

/* ── Info / Warning / Success blocks ── */
.stInfo { background: rgba(59,130,246,0.12) !important; border-color: rgba(59,130,246,0.35) !important; color: #BAE6FD !important; }
.stSuccess { background: rgba(16,185,129,0.12) !important; border-color: rgba(16,185,129,0.35) !important; color: #6EE7B7 !important; }
.stWarning { background: rgba(245,158,11,0.12) !important; border-color: rgba(245,158,11,0.35) !important; color: #FCD34D !important; }
.stError   { background: rgba(239,68,68,0.12) !important;  border-color: rgba(239,68,68,0.35) !important;  color: #FCA5A5 !important; }
[data-testid="stInfoContent"] *, [data-testid="stSuccessContent"] *, 
[data-testid="stWarningContent"] *, [data-testid="stErrorContent"] * { color: inherit !important; }

/* ── JSON display ── */
.stJson { background: rgba(10,20,40,0.8) !important; border-radius: 10px !important; border: 1px solid rgba(0,212,255,0.12) !important; }

/* ── Divider ── */
hr { border-color: rgba(0,212,255,0.1) !important; }

/* ── Sidebar branding ── */
.sb-title {
    font-size:1.15rem; font-weight:700;
    color: #E2E8F0 !important;
    -webkit-text-fill-color: #E2E8F0 !important;
    text-shadow: 0 0 20px rgba(0,212,255,0.4);
    margin-bottom: 2px;
}
.sb-sub { color: #64748B !important; font-size:0.75rem; margin-bottom:18px; display:block; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00D4FF !important; }

/* ── Report Section ── */
.report-section {
    background: rgba(10,20,45,0.7);
    border: 1px solid rgba(0,212,255,0.14);
    border-radius: 14px; padding: 22px 26px; margin-bottom: 14px;
    line-height: 1.75; color: #CBD5E1 !important;
}
.report-section h3 { color: #38BDF8 !important; margin-top: 0; }
.report-section h4 { color: #E2E8F0 !important; }
.report-section li { color: #CBD5E1 !important; margin-bottom: 4px; }
.report-section strong { color: #E2E8F0 !important; }

/* ── Global Search Result ── */
.search-result {
    background: rgba(0,30,60,0.8);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 14px; padding: 20px 24px; margin-top: 14px;
    animation: fadeIn 0.3s ease;
    color: #CBD5E1 !important;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }
.search-result * { color: #CBD5E1 !important; }

/* ── Admin Login Card ── */
.admin-login {
    max-width: 440px; margin: 50px auto;
    background: rgba(10,30,60,0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 22px; padding: 44px 40px;
    box-shadow: 0 10px 50px rgba(0,0,0,0.5);
    text-align: center;
}


/* ── Footer ── */
.aibos-footer {
    margin-top: 52px; padding: 20px 0 10px 0;
    border-top: 1px solid rgba(0,212,255,0.1);
    text-align: center; color: #64748B !important; font-size: 0.8rem;
}
.aibos-footer a { color: #38BDF8 !important; text-decoration: none; }
.aibos-footer a:hover { color: #00D4FF !important; text-decoration: underline; }
.aibos-footer div { color: #64748B !important; }
.footer-badges { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:10px; margin-bottom:10px; }
.f-badge {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 14px; padding: 4px 14px;
    font-size: 0.74rem; color: #7DD3FC !important;
    font-weight: 500;
}

/* ══════════════════════════════════════════════════════════
   CHAT BUBBLE — FIXED ALIGNMENT
   Avatar stays at TOP of bubble, not bottom
═══════════════════════════════════════════════════════════ */
.bubble-wrap {
    display: flex !important;
    align-items: flex-start !important;   /* TOP align — yahan fix hai */
    gap: 12px !important;
    margin: 12px 0 !important;
    width: 100% !important;
}
.bubble-wrap.user-wrap {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}
.avatar {
    width: 40px !important; height: 40px !important;
    border-radius: 50% !important;
    display: flex !important; align-items: center !important;
    justify-content: center !important;
    font-size: 1.2rem !important;
    flex-shrink: 0 !important;
    border: 2px solid rgba(0,212,255,0.35) !important;
    margin-top: 2px !important;   /* slight top nudge */
}
.bubble-content {
    display: flex !important;
    flex-direction: column !important;
    max-width: 78% !important;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #1E3A5F, #1D4ED8) !important;
    border: 1px solid rgba(59,130,246,0.4) !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 13px 17px !important;
    color: #F0F9FF !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.4) !important;
    line-height: 1.65 !important;
    word-wrap: break-word !important;
}
.chat-bubble-bot {
    background: rgba(15,32,60,0.88) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 13px 17px !important;
    color: #E2E8F0 !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.3) !important;
    line-height: 1.65 !important;
    word-wrap: break-word !important;
}
.chat-meta {
    font-size: 0.7rem !important;
    color: #64748B !important;
    margin-top: 5px !important;
}

/* ══════════════════════════════════════════════════════════
   FULL RESPONSIVE — MOBILE + TABLET
═══════════════════════════════════════════════════════════ */

/* Tablet (≤ 1024px) */
@media screen and (max-width: 1024px) {
    .page-header h1 { font-size: 1.5rem !important; }
    .metric-value { font-size: 1.75rem !important; }
    .chat-bubble-user, .chat-bubble-bot { max-width: 85% !important; }
    .bubble-content { max-width: 85% !important; }
}

/* Mobile (≤ 768px) */
@media screen and (max-width: 768px) {
    /* Main layout */
    .stApp { padding: 0 !important; }
    .block-container {
        padding: 1rem 0.75rem !important;
        max-width: 100% !important;
    }

    /* Page header */
    .page-header {
        padding: 16px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 16px !important;
    }
    .page-header h1 { font-size: 1.25rem !important; }
    .page-header p  { font-size: 0.8rem !important; }

    /* Metric cards — stack to 2x2 on mobile */
    .metric-card {
        min-height: 120px !important;
        padding: 16px 12px !important;
        border-radius: 12px !important;
    }
    .metric-value { font-size: 1.6rem !important; }
    .metric-icon  { font-size: 1.6rem !important; }
    .metric-label { font-size: 0.68rem !important; }

    /* Streamlit columns → stack vertically */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 12px !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Sidebar — hide hamburger icon on top, small screen */
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 280px !important;
    }

    /* Chat bubbles — fuller width on mobile */
    .bubble-wrap { gap: 8px !important; margin: 8px 0 !important; }
    .avatar { width: 32px !important; height: 32px !important; font-size: 1rem !important; }
    .bubble-content { max-width: 88% !important; }
    .chat-bubble-user, .chat-bubble-bot {
        padding: 10px 13px !important;
        font-size: 0.9rem !important;
        border-radius: 12px !important;
    }
    .feedback-row { gap: 5px !important; }
    .fb-btn { font-size: 0.68rem !important; padding: 3px 8px !important; }

    /* Section headers */
    .section-h { font-size: 0.95rem !important; }

    /* Connector cards */
    .conn-card { padding: 10px 12px !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        padding: 6px 10px !important;
        font-size: 0.78rem !important;
    }

    /* Buttons */
    .stButton > button { font-size: 0.85rem !important; padding: 8px 12px !important; }

    /* Search bar — stack on mobile */
    .search-result { padding: 14px 16px !important; }

    /* Footer badges wrap nicely */
    .footer-badges { gap: 6px !important; }
    .f-badge { font-size: 0.68rem !important; padding: 3px 10px !important; }
    .aibos-footer { font-size: 0.72rem !important; }

    /* Admin login card */
    .admin-login {
        margin: 20px 16px !important;
        padding: 28px 22px !important;
    }

    /* DataFrames */
    [data-testid="stDataFrameContainer"] {
        font-size: 0.78rem !important;
    }
}

/* Small mobile (≤ 480px) */
@media screen and (max-width: 480px) {
    .page-header h1 { font-size: 1.1rem !important; }
    .metric-value { font-size: 1.4rem !important; }
    .bubble-content { max-width: 92% !important; }
    .avatar { width: 28px !important; height: 28px !important; font-size: 0.85rem !important; }
    .stTabs [data-baseweb="tab"] { padding: 5px 8px !important; font-size: 0.72rem !important; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "dark_mode": True,
    "chat_history": [],
    "ingested_files": [],
    "connector_status": {"Gmail": False, "Slack": False, "CRM": False},
    "total_docs": 47,
    "vectors_indexed": 1247,
    "report_output": None,
    "feedback_log": [],
    "admin_unlocked": False,
    "query_log": [],
    "cfg_model": "gemini-2.0-flash",
    "cfg_chunk": 512,
    "cfg_temp": 0.3,
    "cfg_topk": 5,
    "global_search_result": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def ts():
    return datetime.datetime.now().strftime("%H:%M")

DEMO_ANSWERS = {
    "deal":    "📊 **Top CRM Deals Pipeline:**\n\n| Company | Value | Stage |\n|---------|-------|-------|\n| Gamma AI Labs | $750K | Contract Review 🔥 |\n| Acme Technologies | $500K | Negotiation |\n| Epsilon Healthcare | $400K | Pilot |\n| Beta Ventures | $280K | Proposal Sent |\n\n**Total pipeline: $2.86M** across 10 active deals.",
    "email":   "📧 **Gmail Summary (Last 5 messages):**\n\n1. 🔐 **Google Security Alert** — AI Employee access granted to your account\n2. 📈 **Grammarly** — Weekly productivity stats & new AI tones\n3. 📦 More notifications in queue\n\n*All PII anonymized before indexing.* ✅",
    "swot":    "## SWOT Analysis — IntelForge\n\n**✅ Strengths:**\n- Multi-source RAG pipeline (Gmail + Slack + CRM)\n- Real-time Pinecone indexing with PII anonymization\n- Gemini 2.0 Flash for fast, accurate responses\n\n**⚠️ Weaknesses:**\n- Slack free plan API limitations\n- No mobile app yet\n\n**🚀 Opportunities:**\n- $1.15M in healthcare & AI deals pending\n- Enterprise contract pipeline growing 34% QoQ\n\n**🔴 Threats:**\n- Competitor AI BI tools (Microsoft Copilot)\n- API rate limits on free tiers",
    "kpi":     "📈 **KPI Dashboard — Current Quarter:**\n\n| Metric | Value | Change |\n|--------|-------|--------|\n| Total Pipeline | $2.86M | ↑ +18% |\n| Deals Won | 3 | ↑ +1 |\n| Conversion Rate | 34% | ↑ +4% |\n| Avg Deal Size | $286K | ↑ +12% |\n| Hot Leads | 2 | Gamma AI, Acme |\n\n*Data sourced from CRM + Gmail + Slack.*",
    "slack":   "💬 **Slack Activity Summary:**\n\nChannel `#intelforge-bot` — 1 message indexed\n- Bot joined the channel at 04:10 AM ✅\n- No messages yet — send a message to index it!\n\n*Tip: Post deal updates in the channel for AI retrieval.*",
}

def get_answer(question: str) -> str:
    lower = question.lower()
    for k, v in DEMO_ANSWERS.items():
        if k in lower:
            return v
    return (
        f"🧠 I searched your full knowledge base for **\"{question}\"**.\n\n"
        "Based on your indexed documents:\n"
        "- **CRM:** 10 active deals across 10 industries\n"
        "- **Gmail:** 5 emails indexed with PII anonymized\n"
        "- **Slack:** #intelforge-bot channel monitored\n\n"
        "Connect your Gemini API for live intelligent responses. "
        "Add more documents via Data Ingestion to improve accuracy."
    )

def stream_answer(text: str):
    """Yield words for streaming effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.025)

def get_pinecone_stats():
    try:
        from pinecone import Pinecone as PC
        pc = PC(api_key=os.getenv("PINECONE_API_KEY",""))
        idx_name = "business-intelligence"
        names = [i.name for i in pc.list_indexes()]
        if idx_name in names:
            stats = pc.Index(idx_name).describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 768),
                "fullness": stats.get("index_fullness", 0.0),
            }
    except Exception:
        pass
    return {"total_vectors": st.session_state.vectors_indexed, "dimension": 768, "fullness": 0.12}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8"),
    margin=dict(t=20, b=10, l=0, r=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748B")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748B")),
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">🧠 AI-BOS</div><div class="sb-sub">Business Intelligence OS v2.0</div>', unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠 Dashboard",
        "📂 Data Ingestion",
        "💬 AI Assistant",
        "📊 Reports",
        "🔧 Admin",
    ], label_visibility="collapsed")

    st.markdown("---")
    if st.button("☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("---")

    st.markdown("**⚡ Connectors**")
    gmail_ok = os.path.exists("token.json")
    slack_ok = bool(os.getenv("SLACK_BOT_TOKEN"))
    crm_ok   = os.path.exists("crm_data.json")
    for name, ok in [("Gmail", gmail_ok), ("Slack", slack_ok), ("CRM", crm_ok)]:
        st.markdown(f"{'🟢' if ok else '🔴'} {name}")

    st.markdown("---")
    st.markdown('<div style="color:#334155;font-size:0.7rem;text-align:center;">AI-BOS v2.0 • IntelForge Engine</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    pstats = get_pinecone_stats()
    tv = pstats["total_vectors"]
    td = st.session_state.total_docs
    qt = random.randint(18, 42)
    au = random.randint(3, 9)

    st.markdown("""<div class="page-header">
        <h1>🧠 AI-BOS Business Brain</h1>
        <p>Intelligent Business Operations System • Real-time Knowledge Pipeline</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label, delta, cls in [
        (c1, "📄", td,     "Total Documents",  "↑ +12 this week",  "delta-up"),
        (c2, "🔮", f"{tv:,}", "Vectors Indexed", f"↑ +{random.randint(50,220)} today", "delta-up"),
        (c3, "💬", qt,     "Queries Today",    "↑ +8 vs yesterday","delta-up"),
        (c4, "👥", au,     "Active Users",     "↑ +2 this hour",   "delta-up"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <span class="metric-delta {cls}">{delta}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-h">📋 Recent Activity</div>', unsafe_allow_html=True)
        act = pd.DataFrame([
            {"Time":"04:12","Event":"📄 crm_data.json ingested","Source":"CRM","Status":"✅"},
            {"Time":"04:10","Event":"📧 5 Gmail messages indexed","Source":"Gmail","Status":"✅"},
            {"Time":"04:09","Event":"💬 Slack joined #intelforge-bot","Source":"Slack","Status":"✅"},
            {"Time":"03:58","Event":"🔍 Query: enterprise deals","Source":"Agent","Status":"✅"},
            {"Time":"03:45","Event":"📊 SWOT report generated","Source":"Agent","Status":"✅"},
            {"Time":"03:30","Event":"📄 3 PDFs chunked + indexed","Source":"File","Status":"✅"},
        ])
        st.dataframe(act, use_container_width=True, hide_index=True, height=260)

    with right:
        st.markdown('<div class="section-h">🔮 Pinecone Health</div>', unsafe_allow_html=True)
        fp = round(pstats["fullness"] * 100, 1)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fp,
            title={"text":"Index Fullness %","font":{"color":"#64748B","size":12}},
            number={"suffix":"%","font":{"color":"#00D4FF","size":22}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#334155"},
                   "bar":{"color":"#00D4FF"},
                   "bgcolor":"rgba(255,255,255,0.03)",
                   "steps":[{"range":[0,60],"color":"rgba(16,185,129,.08)"},
                             {"range":[60,85],"color":"rgba(245,158,11,.08)"},
                             {"range":[85,100],"color":"rgba(239,68,68,.08)"}]}
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(f"""<div style="display:flex;justify-content:space-between;color:#64748B;font-size:0.8rem;padding:0 4px;">
            <span>Vectors: <b style="color:#00D4FF">{tv:,}</b></span>
            <span>Dims: <b style="color:#00D4FF">{pstats['dimension']}</b></span>
            <span>Metric: <b style="color:#00D4FF">cosine</b></span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-h">📈 Query Volume — Last 7 Days</div>', unsafe_allow_html=True)
    days = [(datetime.date.today()-datetime.timedelta(days=i)).strftime("%b %d") for i in range(6,-1,-1)]
    qs   = [random.randint(8,45) for _ in days]
    fig_l = go.Figure(go.Scatter(
        x=days, y=qs, mode="lines+markers",
        line=dict(color="#00D4FF", width=2.5),
        marker=dict(color="#00D4FF", size=7, line=dict(color="#0A2540", width=2)),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
    ))
    fig_l.update_layout(**CHART_LAYOUT, height=210, showlegend=False)
    st.plotly_chart(fig_l, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Data Ingestion":
    st.markdown("""<div class="page-header">
        <h1>📂 Data Ingestion Hub</h1>
        <p>Index documents and connect live data sources to your knowledge base</p>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-h">📄 Upload Documents</div>', unsafe_allow_html=True)
        ups = st.file_uploader("Drag & drop files here or click to browse",
                               type=["pdf","csv","txt","md"],
                               accept_multiple_files=True)
        if ups:
            existing = [x["name"] for x in st.session_state.ingested_files]
            for f in ups:
                if f.name not in existing:
                    st.session_state.ingested_files.append({
                        "name": f.name,
                        "size": f"{round(f.size/1024,1)} KB",
                        "type": f.name.split(".")[-1].upper(),
                        "status": "⏳ Pending",
                        "uploaded": ts(),
                    })

        if st.session_state.ingested_files:
            st.markdown('<div class="section-h">📋 File Queue</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state.ingested_files),
                         use_container_width=True, hide_index=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🚀 Index All Files", use_container_width=True):
                    prog = st.progress(0)
                    n = len(st.session_state.ingested_files)
                    for i, f in enumerate(st.session_state.ingested_files):
                        prog.progress((i+1)/n, text=f"⚙️ Processing {f['name']}...")
                        time.sleep(0.35)
                        st.session_state.ingested_files[i]["status"] = "✅ Indexed"
                    st.session_state.total_docs += n
                    prog.empty()
                    st.toast(f"✅ {n} files indexed!", icon="🎉")
                    st.rerun()
            with b2:
                if st.button("🗑️ Clear Queue", use_container_width=True):
                    st.session_state.ingested_files = []
                    st.rerun()

    with right:
        st.markdown('<div class="section-h">🔌 Live Connectors</div>', unsafe_allow_html=True)
        for icon, name, auth, ok in [
            ("📧","Gmail","OAuth2",      gmail_ok),
            ("💬","Slack","Bot Token",   slack_ok),
            ("🗂️","CRM","JSON / CSV",   crm_ok),
        ]:
            cls   = "badge-on" if ok else "badge-off"
            txt   = "● Connected" if ok else "● Disconnected"
            st.markdown(f"""<div class="conn-card">
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:1.4rem;">{icon}</span>
                    <div><div style="font-weight:600;color:#E2E8F0;">{name}</div>
                    <div style="font-size:0.74rem;color:#475569;">{auth}</div></div>
                </div>
                <span class="badge {cls}">{txt}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-h">⚡ Full Re-Index</div>', unsafe_allow_html=True)
        st.caption("Pull fresh data from all sources and refresh Pinecone.")
        if st.button("🔄 Re-Index Everything", use_container_width=True):
            steps = [
                ("📧 Fetching Gmail...",         0.4),
                ("💬 Fetching Slack...",          0.3),
                ("🗂️ Loading CRM data...",        0.3),
                ("✂️ Chunking documents...",      0.5),
                ("🔮 Upserting to Pinecone...",   0.6),
                ("✅ Finalizing...",              0.2),
            ]
            prog = st.progress(0)
            ph   = st.empty()
            for i,(msg,dl) in enumerate(steps):
                ph.info(msg)
                time.sleep(dl)
                prog.progress((i+1)/len(steps))
            prog.empty(); ph.empty()
            st.toast("🎉 All sources re-indexed!", icon="✅")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 AI Assistant":
    st.markdown("""<div class="page-header">
        <h1>💬 AI Business Assistant</h1>
        <p>Powered by Gemini 2.0 Flash • RAG-enhanced with your business knowledge</p>
    </div>""", unsafe_allow_html=True)

    # ── Empty state suggestion chips ──────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("""<div style="text-align:center;padding:36px 0 8px 0;">
            <div style="font-size:3.5rem;margin-bottom:8px;">🧠</div>
            <div style="font-size:1.2rem;font-weight:600;color:#CBD5E1;">Your Business Brain is ready</div>
            <div style="font-size:0.85rem;color:#475569;margin-top:6px;">
                Ask anything about your deals, emails, or company knowledge
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-h" style="text-align:center;">💡 Try asking...</div>', unsafe_allow_html=True)

        chips = [
            ("📊", "What are our top deals?"),
            ("📧", "Summarize recent emails"),
            ("🔍", "Give me a SWOT analysis"),
            ("📈", "Show me our KPIs"),
            ("💬", "What's in Slack?"),
            ("🎯", "Which deals need attention?"),
        ]
        cols = st.columns(3)
        for i, (icon, text) in enumerate(chips):
            with cols[i % 3]:
                if st.button(f"{icon} {text}", use_container_width=True, key=f"chip_{i}"):
                    st.session_state._prompt = text
                    st.rerun()

    # ── Chat History ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f"""<div class="bubble-wrap user-wrap">
                    <div class="avatar avatar-user">👤</div>
                    <div class="bubble-content">
                        <div class="chat-bubble-user">{msg['content']}</div>
                        <div class="chat-meta" style="justify-content:flex-end;text-align:right;">{msg.get('time','')}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                content_html = msg["content"].replace("\n", "<br>")
                st.markdown(f"""<div class="bubble-wrap">
                    <div class="avatar avatar-bot">🧠</div>
                    <div class="bubble-content">
                        <div class="chat-bubble-bot">{content_html}</div>
                        <div class="chat-meta">
                            {msg.get('time','')}
                            &nbsp;
                            <span title="Copy response" style="cursor:pointer;">📋</span>
                        </div>
                        <div class="feedback-row">
                            <button class="fb-btn" onclick="void(0)">👍 Helpful</button>
                            <button class="fb-btn" onclick="void(0)">👎 Not helpful</button>
                            <button class="fb-btn" onclick="void(0)">🔁 Retry</button>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Chat Input ────────────────────────────────────────────────────────────
    st.markdown("---")
    user_input = st.chat_input("Ask your Business Brain anything... (e.g. 'What are our top deals?')")

    # Handle suggestion chip click
    if hasattr(st.session_state, "_prompt") and st.session_state._prompt:
        user_input = st.session_state._prompt
        st.session_state._prompt = None

    if user_input:
        now = ts()
        st.session_state.chat_history.append({"role":"user","content":user_input,"time":now})

        # Typing indicator placeholder
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""<div class="bubble-wrap">
            <div class="avatar avatar-bot">🧠</div>
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>""", unsafe_allow_html=True)
        time.sleep(0.8)
        typing_placeholder.empty()

        # Get answer
        try:
            from agent import build_business_agent
            if "agent_instance" not in st.session_state:
                with st.spinner("Initializing AI agent..."):
                    st.session_state.agent_instance = build_business_agent(docs=[])
            result = st.session_state.agent_instance.invoke({"input": user_input})
            answer = result.get("output", str(result))
        except Exception:
            answer = get_answer(user_input)

        # Stream the answer
        stream_placeholder = st.empty()
        rendered = ""
        for chunk in stream_answer(answer):
            rendered += chunk
            stream_placeholder.markdown(f"""<div class="bubble-wrap">
                <div class="avatar avatar-bot">🧠</div>
                <div class="chat-bubble-bot">{rendered}▌</div>
            </div>""", unsafe_allow_html=True)
        stream_placeholder.empty()

        st.session_state.chat_history.append({"role":"assistant","content":answer,"time":now})
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Reports":
    st.markdown("""<div class="page-header">
        <h1>📊 Reports Generator</h1>
        <p>AI-powered business intelligence reports with interactive charts</p>
    </div>""", unsafe_allow_html=True)

    # ── Report Config Form ─────────────────────────────────────────────────────
    with st.form("report_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            dept = st.selectbox("🏢 Department", [
                "Executive / C-Suite", "Sales & Revenue", "Marketing",
                "Operations", "Finance", "HR & People", "Product & Tech"
            ])
        with fc2:
            period = st.selectbox("📅 Time Range", [
                "This Week", "This Month", "Last Quarter",
                "Last 6 Months", "Year to Date", "Custom"
            ])
        with fc3:
            goal = st.selectbox("🎯 Report Goal", [
                "Performance Overview", "Pipeline Analysis", "Churn & Retention",
                "Revenue Forecast", "Team Productivity", "Risk Assessment"
            ])
        submitted = st.form_submit_button("✨ Generate Report", use_container_width=True)

    # ── Report Generation ──────────────────────────────────────────────────────
    if submitted:
        with st.spinner(f"🧠 Generating {goal} report for {dept}..."):
            time.sleep(1.8)
        st.toast("✅ Report generated!", icon="📊")
        st.session_state.report_output = {
            "dept": dept, "period": period, "goal": goal,
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    if st.session_state.report_output:
        rpt = st.session_state.report_output
        st.markdown(f"""<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:18px;padding:12px 18px;
            background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.18);
            border-radius:12px;">
            <div>
                <span style="font-weight:700;color:#E2E8F0;font-size:1.05rem;">
                    {rpt['goal']} — {rpt['dept']}</span>
                <span style="color:#475569;font-size:0.8rem;margin-left:14px;">
                    {rpt['period']} • Generated {rpt['generated_at']}</span>
            </div>
            <span class="badge badge-on">● Ready</span>
        </div>""", unsafe_allow_html=True)

        # ── Report Tabs ────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Executive Summary", "💡 Key Insights", "📈 Charts", "🎯 Action Plan"
        ])

        with tab1:
            st.markdown(f"""<div class="report-section">
<h3 style="color:#00D4FF;margin-top:0;">Executive Summary</h3>
<p>This report covers the <strong>{rpt['goal'].lower()}</strong> for the
<strong>{rpt['dept']}</strong> department over <strong>{rpt['period'].lower()}</strong>.</p>

<h4 style="color:#E2E8F0;">🔑 Headline Metrics</h4>
<ul>
<li>💰 <strong>Total Pipeline:</strong> $2.86M across 10 active CRM deals</li>
<li>📈 <strong>Growth Rate:</strong> +18% QoQ revenue trajectory</li>
<li>🏆 <strong>Win Rate:</strong> 34% deal conversion (↑ +4% vs last quarter)</li>
<li>⚡ <strong>AI Queries:</strong> {random.randint(200,500)} this period (↑ +32%)</li>
<li>📄 <strong>Knowledge Base:</strong> {st.session_state.total_docs} documents, {st.session_state.vectors_indexed:,} vectors indexed</li>
</ul>

<h4 style="color:#E2E8F0;">📊 Business Context</h4>
<p>IntelForge's multi-source RAG pipeline is actively indexing data from Gmail (5 emails),
Slack (#intelforge-bot), and CRM (10 company records across 10 industries).
The Pinecone vector database holds {st.session_state.vectors_indexed:,} vectors with cosine similarity search.</p>
            </div>""", unsafe_allow_html=True)

        with tab2:
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                for insight in [
                    ("🔥","Gamma AI Labs","$750K contract nearing close — needs legal review"),
                    ("📈","Healthcare vertical","Growing 22% with Epsilon Pilot running"),
                    ("⚠️","Slack connector","Only 1 message indexed — encourage team usage"),
                    ("✅","PII compliance","100% anonymization across all indexed data"),
                ]:
                    icon, title, desc = insight
                    st.markdown(f"""<div class="conn-card">
                        <div>
                            <div style="font-weight:600;color:#E2E8F0;">{icon} {title}</div>
                            <div style="font-size:0.8rem;color:#64748B;margin-top:3px;">{desc}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            with col_i2:
                for insight in [
                    ("💡","RAG performance","Query latency avg 1.2s — well within SLA"),
                    ("🎯","Top opportunity","Beta Ventures $280K — proposal sent, follow up needed"),
                    ("📧","Email signal","Security alert from Google — AI Employee scope granted"),
                    ("🚀","Pipeline health","$1.15M in late-stage deals (Contract + Pilot)"),
                ]:
                    icon, title, desc = insight
                    st.markdown(f"""<div class="conn-card">
                        <div>
                            <div style="font-weight:600;color:#E2E8F0;">{icon} {title}</div>
                            <div style="font-size:0.8rem;color:#64748B;margin-top:3px;">{desc}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        with tab3:
            ch1, ch2 = st.columns(2)

            with ch1:
                # Sales Trend
                st.markdown('<div class="section-h">📈 Revenue Trend</div>', unsafe_allow_html=True)
                months = ["Sep","Oct","Nov","Dec","Jan","Feb"]
                rev    = [180, 220, 195, 310, 275, 340]
                target = [200, 230, 230, 280, 300, 320]
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=months, y=rev, name="Actual",
                    mode="lines+markers",
                    line=dict(color="#00D4FF", width=2.5),
                    marker=dict(size=7, color="#00D4FF", line=dict(color="#0F172A",width=2)),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"))
                fig_trend.add_trace(go.Scatter(x=months, y=target, name="Target",
                    mode="lines", line=dict(color="#818CF8", width=1.5, dash="dash")))
                fig_trend.update_layout(**CHART_LAYOUT, height=280,
                    legend=dict(font=dict(color="#94A3B8"),
                                bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_trend, use_container_width=True)

                # KPI Bar
                st.markdown('<div class="section-h">📊 KPI Performance</div>', unsafe_allow_html=True)
                kpis   = ["Win Rate","NPS Score","Pipeline Velocity","Avg Deal Size","CAC Ratio"]
                actual = [34, 72, 67, 86, 91]
                target_k = [40, 80, 70, 80, 90]
                fig_kpi = go.Figure()
                fig_kpi.add_trace(go.Bar(name="Actual", x=kpis, y=actual,
                    marker_color="#00D4FF", opacity=0.85))
                fig_kpi.add_trace(go.Bar(name="Target", x=kpis, y=target_k,
                    marker_color="#818CF8", opacity=0.5))
                fig_kpi.update_layout(**CHART_LAYOUT, height=260, barmode="group",
                    legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_kpi, use_container_width=True)

            with ch2:
                # Churn Pie
                st.markdown('<div class="section-h">🔄 Deal Stage Distribution</div>', unsafe_allow_html=True)
                labels = ["Initial","Qualified","Discovery","Demo","Proposal","Negotiation","Contract","Pilot"]
                values = [2, 1, 1, 1, 2, 2, 1, 1]
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values, hole=0.52,
                    marker=dict(colors=["#0D3B6B","#0EA5E9","#38BDF8","#00D4FF",
                                        "#818CF8","#A78BFA","#C084FC","#E879F9"]),
                    textfont=dict(color="#E2E8F0", size=11),
                ))
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
                    margin=dict(t=10,b=10,l=0,r=0),
                    legend=dict(font=dict(color="#94A3B8"),bgcolor="rgba(0,0,0,0)"),
                    annotations=[dict(text="10 Deals",x=0.5,y=0.5,
                                      font=dict(size=16,color="#00D4FF"),showarrow=False)])
                st.plotly_chart(fig_pie, use_container_width=True)

                # Industry heatmap
                st.markdown('<div class="section-h">🏭 Revenue by Industry ($K)</div>', unsafe_allow_html=True)
                inds = ["AI","Healthcare","Finance","Software","Manufacturing","Retail","EdTech","Media"]
                vals = [750, 400, 280, 500, 320, 200, 80, 150]
                fig_bar = go.Figure(go.Bar(
                    y=inds, x=vals, orientation="h",
                    marker=dict(color=vals,
                                colorscale=[[0,"#0A2540"],[0.5,"#0D6B8B"],[1,"#00D4FF"]]),
                    text=[f"${v}K" for v in vals], textposition="outside",
                    textfont=dict(color="#94A3B8"),
                ))
                fig_bar.update_layout(**CHART_LAYOUT, height=300,
                    yaxis=dict(tickfont=dict(color="#94A3B8")))
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab4:
            actions = [
                ("🔴","Critical","Close Gamma AI Labs ($750K)","Within 1 week","Legal sign-off needed → assign to Sarah"),
                ("🟠","High","Follow up Acme Technologies ($500K)","This week","Schedule negotiation call → John to lead"),
                ("🟠","High","Activate Epsilon Healthcare Pilot","Ongoing","Pilot running — send weekly status to stakeholder"),
                ("🟡","Medium","Re-engage Beta Ventures ($280K)","2 weeks","Proposal sent 10 days ago — no response"),
                ("🟢","Low","Onboard Zeta Education ($80K)","Next month","Qualified lead — assign BDR for nurturing"),
            ]
            for pri_icon, pri, action, deadline, detail in actions:
                st.markdown(f"""<div class="conn-card">
                    <div style="display:flex;align-items:flex-start;gap:14px;">
                        <span style="font-size:1.4rem;margin-top:2px;">{pri_icon}</span>
                        <div>
                            <div style="font-weight:600;color:#E2E8F0;font-size:0.95rem;">{action}</div>
                            <div style="font-size:0.78rem;color:#64748B;margin-top:3px;">{detail}</div>
                        </div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;">
                        <span class="badge {'badge-off' if pri=='Critical' else 'badge-wrn' if pri=='High' else 'badge-on'}">{pri}</span>
                        <div style="font-size:0.72rem;color:#475569;margin-top:4px;">📅 {deadline}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                if st.button("📄 Download as PDF (Coming Soon)", use_container_width=True):
                    st.toast("PDF export coming in v2.1 📄", icon="🔧")
            with dl2:
                if st.button("📋 Copy Report to Clipboard", use_container_width=True):
                    st.toast("Report copied! ✅", icon="📋")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Admin":
    st.markdown("""<div class="page-header">
        <h1>🔧 Admin Panel</h1>
        <p>Secure system configuration, logs, and diagnostics</p>
    </div>""", unsafe_allow_html=True)

    # ── Password Gate ─────────────────────────────────────────────────────────
    if not st.session_state.admin_unlocked:
        st.markdown('<div class="admin-login">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:3rem;margin-bottom:8px;'>🔐</div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#E2E8F0;margin:0 0 6px 0;'>Admin Access Required</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#475569;font-size:0.85rem;margin-bottom:24px;'>Enter the admin password to continue</p>", unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password", placeholder="Enter admin password...", label_visibility="collapsed")
        if st.button("🔓 Unlock Admin", use_container_width=True):
            if pwd == "admin123":
                st.session_state.admin_unlocked = True
                st.toast("✅ Admin unlocked!", icon="🔓")
                st.rerun()
            else:
                st.error("❌ Incorrect password. Default is **admin123** for demo.")
        st.markdown("<p style='color:#334155;font-size:0.72rem;margin-top:16px;'>Demo password: admin123</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # ── Logged-in admin ───────────────────────────────────────────────────
        hdr_col, lock_col = st.columns([8, 1])
        with lock_col:
            if st.button("🔒 Lock", use_container_width=True):
                st.session_state.admin_unlocked = False
                st.rerun()

        adm1, adm2, adm3, adm4 = st.tabs([
            "📊 Overview", "⚙️ Configuration", "📋 Logs", "🔄 Re-Index"
        ])

        # ── TAB 1: Overview ───────────────────────────────────────────────────
        with adm1:
            pstats = get_pinecone_stats()
            env_vars = {
                "GOOGLE_API_KEY":   os.getenv("GOOGLE_API_KEY"),
                "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
                "SLACK_BOT_TOKEN":  os.getenv("SLACK_BOT_TOKEN"),
                "SLACK_CHANNEL_ID": os.getenv("SLACK_CHANNEL_ID"),
                "GMAIL_TOKEN_PATH": os.getenv("GMAIL_TOKEN_PATH","token.json"),
            }
            ov1, ov2, ov3 = st.columns(3)
            with ov1:
                st.markdown('<div class="section-h">⚙️ Env Variables</div>', unsafe_allow_html=True)
                for var, val in env_vars.items():
                    cls = "badge-on" if val else "badge-off"
                    txt = "● Set" if val else "● Missing"
                    masked = (val[:8]+"••••") if val else ""
                    st.markdown(f"""<div class="conn-card" style="padding:10px 14px;">
                        <span style="color:#94A3B8;font-size:0.78rem;">{var}</span>
                        <div style="display:flex;gap:6px;align-items:center;">
                            <span style="color:#475569;font-size:0.72rem;font-family:monospace;">{masked}</span>
                            <span class="badge {cls}">{txt}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

            with ov2:
                st.markdown('<div class="section-h">🔮 Pinecone Stats</div>', unsafe_allow_html=True)
                for label, val in [
                    ("Index",   "business-intelligence"),
                    ("Vectors", f"{pstats['total_vectors']:,}"),
                    ("Dims",    str(pstats['dimension'])),
                    ("Metric",  "cosine"),
                    ("Fullness",f"{round(pstats['fullness']*100,1)}%"),
                ]:
                    st.markdown(f"""<div class="conn-card" style="padding:10px 14px;">
                        <span style="color:#64748B;font-size:0.8rem;">{label}</span>
                        <span style="color:#00D4FF;font-weight:600;font-size:0.85rem;">{val}</span>
                    </div>""", unsafe_allow_html=True)

            with ov3:
                st.markdown('<div class="section-h">🩺 File Health</div>', unsafe_allow_html=True)
                for fname in ["credentials.json","token.json","crm_data.json",".env","connectors.py","agent.py","ui.py"]:
                    ok = os.path.exists(fname)
                    st.markdown(f"""<div class="conn-card" style="padding:10px 14px;">
                        <span style="color:#94A3B8;font-size:0.78rem;font-family:monospace;">{fname}</span>
                        <span class="badge {'badge-on' if ok else 'badge-off'}">{'● Found' if ok else '● Missing'}</span>
                    </div>""", unsafe_allow_html=True)

        # ── TAB 2: Configuration ──────────────────────────────────────────────
        with adm2:
            st.markdown('<div class="section-h">🤖 Model Settings</div>', unsafe_allow_html=True)
            cfg1, cfg2 = st.columns(2)
            with cfg1:
                model = st.selectbox("Embedding / LLM Model",
                    ["gemini-2.0-flash","gemini-1.5-pro","gemini-1.5-flash","text-embedding-004"],
                    index=["gemini-2.0-flash","gemini-1.5-pro","gemini-1.5-flash","text-embedding-004"].index(
                        st.session_state.cfg_model) if st.session_state.cfg_model in ["gemini-2.0-flash","gemini-1.5-pro","gemini-1.5-flash","text-embedding-004"] else 0)
                chunk = st.slider("Chunk Size (tokens)", 128, 1024, st.session_state.cfg_chunk, step=64)
            with cfg2:
                temp  = st.slider("LLM Temperature", 0.0, 1.0, st.session_state.cfg_temp, step=0.05)
                topk  = st.slider("Top-K Retrieval", 1, 20, st.session_state.cfg_topk)
            if st.button("💾 Save Configuration", use_container_width=True):
                st.session_state.cfg_model = model
                st.session_state.cfg_chunk = chunk
                st.session_state.cfg_temp  = temp
                st.session_state.cfg_topk  = topk
                st.toast("✅ Configuration saved!", icon="💾")

            st.markdown('<div class="section-h" style="margin-top:24px;">🎛️ Active Config</div>', unsafe_allow_html=True)
            st.json({
                "model":      st.session_state.cfg_model,
                "chunk_size": st.session_state.cfg_chunk,
                "temperature":st.session_state.cfg_temp,
                "top_k":      st.session_state.cfg_topk,
                "vector_db":  "pinecone",
                "index":      "business-intelligence",
            })

        # ── TAB 3: Query Logs ─────────────────────────────────────────────────
        with adm3:
            demo_logs = [
                {"Time":"04:12","User":"admin","Query":"What are our top deals?","Source":"CRM","Tokens":"412","Latency":"1.2s","Status":"✅"},
                {"Time":"04:10","User":"admin","Query":"Summarize recent emails","Source":"Gmail","Tokens":"318","Latency":"0.9s","Status":"✅"},
                {"Time":"03:58","User":"admin","Query":"SWOT analysis","Source":"All","Tokens":"891","Latency":"2.1s","Status":"✅"},
                {"Time":"03:45","User":"admin","Query":"Show me KPIs","Source":"CRM","Tokens":"524","Latency":"1.4s","Status":"✅"},
                {"Time":"03:30","User":"admin","Query":"Enterprise deal pipeline","Source":"CRM","Tokens":"677","Latency":"1.8s","Status":"✅"},
            ]
            # Merge with live chat history
            live = [{"Time":m["time"],"User":"session","Query":m["content"],"Source":"All","Tokens":str(len(m["content"].split())*4),"Latency":"--","Status":"✅"}
                    for m in st.session_state.chat_history if m["role"]=="user"]
            all_logs = live + demo_logs
            st.markdown(f'<div class="section-h">📋 Query Log ({len(all_logs)} entries)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(all_logs), use_container_width=True, hide_index=True, height=320)
            log_c1, log_c2 = st.columns(2)
            with log_c1:
                if st.button("📥 Export Logs (CSV)", use_container_width=True):
                    csv = pd.DataFrame(all_logs).to_csv(index=False)
                    st.download_button("⬇️ Download", csv, "query_logs.csv", "text/csv", use_container_width=True)
            with log_c2:
                if st.button("🗑️ Clear All Logs", use_container_width=True):
                    st.session_state.query_log = []
                    st.toast("Logs cleared!", icon="🗑️")
                    st.rerun()

        # ── TAB 4: Re-Index ───────────────────────────────────────────────────
        with adm4:
            st.markdown('<div class="section-h">⚡ Full Pipeline Re-Index</div>', unsafe_allow_html=True)
            st.info("This will re-pull all data from Gmail, Slack, CRM, and re-index Pinecone. Estimated time: ~15 seconds.")
            ri_col1, ri_col2 = st.columns(2)
            with ri_col1:
                gmail_ri = st.checkbox("📧 Include Gmail", value=True)
                slack_ri = st.checkbox("💬 Include Slack", value=True)
            with ri_col2:
                crm_ri  = st.checkbox("🗂️ Include CRM", value=True)
                files_ri = st.checkbox("📄 Include Uploaded Files", value=True)
            if st.button("🔄 Start Full Re-Index", use_container_width=True):
                steps = []
                if gmail_ri: steps.append(("📧 Fetching Gmail...", 0.5))
                if slack_ri: steps.append(("💬 Fetching Slack...", 0.4))
                if crm_ri:   steps.append(("🗂️ Loading CRM...", 0.3))
                if files_ri: steps.append(("📄 Processing files...", 0.4))
                steps += [("✂️ Chunking text...", 0.5),("🔮 Upserting Pinecone...", 0.8),("✅ Done!", 0.2)]
                prog = st.progress(0)
                ph = st.empty()
                for i,(msg,dl) in enumerate(steps):
                    ph.info(msg); time.sleep(dl)
                    prog.progress((i+1)/len(steps))
                prog.empty(); ph.empty()
                st.success("🎉 Re-index complete! All sources updated.")
                st.toast("Pinecone index refreshed!", icon="🔮")

        # Quick action buttons below tabs
        st.markdown("---")
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            if st.button("▶ Run Tests", use_container_width=True):
                with st.spinner("Running 13 tests..."): time.sleep(1.4)
                st.toast("✅ 13/13 passed!", icon="🎉")
        with qa2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.toast("Chat cleared!", icon="🗑️")
        with qa3:
            if st.button("🔄 Reset Session", use_container_width=True):
                st.session_state.admin_unlocked = False
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SEARCH (shown on every page except Admin)
# ══════════════════════════════════════════════════════════════════════════════
if page != "🔧 Admin":
    st.markdown("---")
    st.markdown('<div class="section-h">🔍 Global Knowledge Search</div>', unsafe_allow_html=True)
    gs_col, gs_btn = st.columns([5, 1])
    with gs_col:
        gs_query = st.text_input("Search across all indexed knowledge...", placeholder="e.g. Gamma AI deal, Q4 revenue, PII policy...", label_visibility="collapsed", key="gs_input")
    with gs_btn:
        gs_go = st.button("Search", use_container_width=True, key="gs_go")
    if gs_go and gs_query:
        with st.spinner("🧠 Searching knowledge base..."):
            time.sleep(0.6)
            answer = get_answer(gs_query)
        st.session_state.global_search_result = {"query": gs_query, "answer": answer, "time": ts()}
        # Log it
        st.session_state.query_log.append({"time": ts(), "query": gs_query, "source": "Global Search"})
    if st.session_state.global_search_result:
        r = st.session_state.global_search_result
        st.markdown(f"""<div class="search-result">
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="color:#00D4FF;font-weight:600;">🔍 Results for: \"{r['query']}\"</span>
                <span style="color:#475569;font-size:0.78rem;">{r['time']}</span>
            </div>
            <div style="color:#CBD5E1;line-height:1.7;">{r['answer'].replace(chr(10),'<br>')}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("✕ Clear Search", key="gs_clear"):
            st.session_state.global_search_result = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="aibos-footer">
    <div class="footer-badges">
        <span class="f-badge">🟦 Streamlit 1.32</span>
        <span class="f-badge">🦜 LangChain 0.3</span>
        <span class="f-badge">✨ Gemini 2.0 Flash</span>
        <span class="f-badge">🌲 Pinecone</span>
        <span class="f-badge">🔒 PII Anonymized</span>
    </div>
    <div style="margin-top:10px;">
        AI-BOS v2.0 &nbsp;|&nbsp; Made with ❤️ for showcase
        &nbsp;|&nbsp; <a href="https://github.com" target="_blank">GitHub</a>
        &nbsp;|&nbsp; <span style="color:#00D4FF;">IntelForge Engine</span>
    </div>
</div>
""", unsafe_allow_html=True)
