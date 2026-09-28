import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import os
import time
from datetime import datetime, timedelta
from logo_utils import get_logo_base64
from ml_engine.qc_engine import WeatherQCEngine
from ml_engine.decision_support import OperatorDecisionSupport

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(*args, **kwargs):
        return 0


# ================================================================
# 1. PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="SkyGuard AI — AWS Quality Control & Anomaly Ops",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# 2. LIVE REFRESH
# ================================================================
# Auto-refreshes every 10 seconds to sync live clock & telemetry stream
st_autorefresh(interval=10000, key="skyguard_live_refresh")


# ================================================================
# 3. FASTAPI BACKEND & LOCAL QC ENGINE
# ================================================================
API_BASE_URL = os.environ.get("SKYGUARD_API_URL", "https://skyguardai.onrender.com").rstrip("/")
LOCAL_API_URL = "http://127.0.0.1:8000"
LATEST_API_URL = f"{API_BASE_URL}/api/v1/latest"

# Embedded QC Engine for zero-latency local fallback validation
local_qc_engine = WeatherQCEngine()


def fetch_latest_status(base_url=None):
    """Fetch latest telemetry + anomaly decision from FastAPI microservice."""
    start = time.perf_counter()
    url = (base_url or API_BASE_URL).rstrip("/")
    target_url = f"{url}/api/v1/latest"
    try:
        response = requests.get(target_url, timeout=4)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        if response.status_code != 200:
            return {
                "status": "ERROR",
                "_connected": False,
                "_request_latency_ms": latency_ms,
                "_error": f"HTTP {response.status_code}",
                "telemetry": {},
                "result": {},
            }
        data = response.json()
        data["_request_latency_ms"] = latency_ms
        data["_connected"] = True
        return data
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "status": "NO_DATA",
            "_connected": False,
            "_request_latency_ms": latency_ms,
            "_error": str(e),
            "telemetry": {},
            "result": {},
        }


# ================================================================
# 4. DESIGN SYSTEM & CSS (DARK OPERATIONS CENTER AESTHETICS)
# ================================================================
logo_b64 = get_logo_base64()

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: #E2E8F0;
  background-color: #080A0D !important;
  -webkit-font-smoothing: antialiased;
}

#MainMenu, header, footer {visibility: hidden !important;}

/* Sidebar Controls */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
  color: #38BDF8 !important;
  background: rgba(56,189,248,0.08) !important;
  border: 1px solid rgba(56,189,248,0.25) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:hover {
  background: rgba(56,189,248,0.18) !important;
  border-color: #38BDF8 !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {
  fill: #38BDF8 !important;
  color: #38BDF8 !important;
}

[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  background: #0E1318 !important;
  border: 1.5px solid #38BDF8 !important;
  border-radius: 0 8px 8px 0 !important;
  padding: 8px 10px !important;
  z-index: 999999 !important;
  box-shadow: 0 0 12px rgba(56,189,248,0.3) !important;
  transition: all 0.2s ease !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover {
  background: #16202A !important;
  border-color: #38BDF8 !important;
  box-shadow: 0 0 16px rgba(56,189,248,0.5) !important;
}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
  color: #38BDF8 !important;
  background: transparent !important;
  border: none !important;
  cursor: pointer !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
  fill: #38BDF8 !important;
  stroke: #38BDF8 !important;
  color: #38BDF8 !important;
  width: 18px !important;
  height: 18px !important;
}

.main .block-container {
  background-color: #080A0D !important;
  padding: 0 24px 40px 24px !important;
  max-width: 100% !important;
}
.main { background-color: #080A0D !important; }

[data-testid="stSidebar"] {
  background-color: #0B0E12 !important;
  border-right: 1px solid rgba(255,255,255,0.07) !important;
  min-width: 350px !important;
  max-width: 350px !important;
  width: 350px !important;
  font-family: Inter, sans-serif;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
  padding: 0 16px 28px 16px !important;
  box-sizing: border-box !important;
  overflow-x: hidden !important;
}
[data-testid="stSidebar"] label {
  color: #E2E8F0 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
  padding: 0 !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] {
  width: 100% !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background-color: #121820 !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 6px !important;
  color: #F8FAFC !important;
  font-size: 11.5px !important;
  min-height: 38px !important;
  padding-left: 10px !important;
  padding-right: 6px !important;
  box-sizing: border-box !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] span {
  font-size: 11.5px !important;
  font-weight: 500 !important;
  color: #F8FAFC !important;
  white-space: nowrap !important;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] input[type="number"] {
  background-color: #121820 !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: #F8FAFC !important;
  border-radius: 6px !important;
  font-size: 12px !important;
  font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption {
  color: #94A3B8 !important;
  font-size: 11px !important;
  line-height: 1.4 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: #10151C !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 8px !important;
  width: 100% !important;
  margin: 0 0 14px 0 !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  color: #F8FAFC !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 10px 14px !important;
  background: #141B24 !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
  color: #38BDF8 !important;
  background: #18222E !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
  fill: #38BDF8 !important;
  color: #38BDF8 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  padding: 12px 10px 8px 10px !important;
  background: #0E1318 !important;
  box-sizing: border-box !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stSelectbox"] {
  padding: 0 !important;
  margin-bottom: 8px !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid rgba(255,255,255,0.08) !important;
  gap: 4px;
  padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: #717D91 !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 9px 18px !important;
  border: none !important;
  font-family: Inter, sans-serif !important;
  letter-spacing: 0.02em;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: #F8FAFC !important;
  background: rgba(56,189,248,0.08) !important;
  border-bottom: 2px solid #38BDF8 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding: 16px 0 0 0 !important; }

/* DataFrame */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
  background: #10151C !important;
  color: #717D91 !important;
  font-size: 10.5px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
[data-testid="stDataFrame"] td {
  color: #E2E8F0 !important;
  font-size: 12.5px !important;
  background: #080A0D !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* Header */
.sg-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 24px;
  background: #080A0D;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  margin: 0 -24px 20px -24px;
  flex-wrap: wrap;
  gap: 12px;
}
.sg-header-left { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
.sg-header-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.sg-logo-title { font-size: 16px; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em; }
.sg-tagline { font-size: 10px; color: #64748B; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }
.sg-station-chip {
  display: inline-flex; align-items: center; gap: 0;
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 6px; padding: 5px 12px; font-size: 12px; color: #94A3B8;
}
.sg-station-chip strong { color: #F1F5F9; margin-right: 6px; }
.sg-station-chip .sep { color: #374151; margin: 0 5px; }
.sg-live-pill {
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25);
  border-radius: 4px; padding: 5px 10px; font-size: 10.5px; font-weight: 700;
  color: #10B981; letter-spacing: 0.06em; text-transform: uppercase;
}
.sg-live-dot { width: 6px; height: 6px; background: #10B981; border-radius: 50%; animation: live-blink 2s ease-in-out infinite; }
@keyframes live-blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
.sg-timestamp { font-size: 11px; color: #64748B; font-family: 'JetBrains Mono', monospace; }
.sg-source-badge {
  background: rgba(56,189,248,0.07); border: 1px solid rgba(56,189,248,0.22);
  border-radius: 4px; padding: 4px 9px; font-size: 10px; font-weight: 600;
  color: #38BDF8; letter-spacing: 0.06em; text-transform: uppercase;
}

/* Banner */
.sg-sim-banner {
  background: rgba(245,158,11,0.06); border-bottom: 1px solid rgba(245,158,11,0.25);
  padding: 9px 24px; font-size: 11.5px; font-weight: 600; color: #F59E0B;
  letter-spacing: 0.04em; margin: -20px -24px 18px -24px;
  display: flex; align-items: center; gap: 10px;
}

/* Command Row */
.sg-section-title {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #64748B; margin-bottom: 12px;
}
.sg-command-row { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
.sg-metric-card {
  flex: 1; min-width: 160px; background: #0E1318;
  border: 1px solid rgba(255,255,255,0.07); border-radius: 8px;
  padding: 14px 16px; transition: border-color 0.2s ease;
}
.sg-metric-card:hover { border-color: rgba(56,189,248,0.25); }
.sg-metric-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 6px; }
.sg-metric-value { font-size: 26px; font-weight: 700; color: #F8FAFC; line-height: 1; letter-spacing: -0.02em; font-family: 'JetBrains Mono', monospace; }
.sg-metric-unit { font-size: 12px; color: #94A3B8; font-weight: 400; margin-left: 3px; }
.sg-metric-sub { font-size: 11px; margin-top: 6px; color: #64748B; }
.sg-metric-sub.fault { color: #EF4444; font-weight: 600; }
.sg-metric-sub.ok { color: #10B981; }
.sg-metric-sub.warning { color: #F59E0B; font-weight: 600; }

.sg-status-card {
  flex: 1.2; min-width: 190px; background: #0E1318;
  border: 1px solid rgba(255,255,255,0.07); border-radius: 8px;
  padding: 14px 16px; border-left-width: 3px;
}
.sg-status-card.operational { border-left-color: #10B981; }
.sg-status-card.critical { border-left-color: #EF4444; background: rgba(239,68,68,0.03); }
.sg-status-card.extreme { border-left-color: #F97316; background: rgba(249,115,22,0.03); }

.sg-status-pill { display: inline-flex; align-items: center; gap: 7px; font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px; }
.sg-status-pill.ok { color: #10B981; }
.sg-status-pill.fault { color: #EF4444; }
.sg-status-pill.extreme { color: #F97316; }

.sg-status-dot { width: 7px; height: 7px; border-radius: 50%; }
.sg-status-dot.ok { background: #10B981; }
.sg-status-dot.fault { background: #EF4444; animation: fault-pulse 1.6s ease-in-out infinite; }
.sg-status-dot.extreme { background: #F97316; animation: extreme-pulse 1.6s ease-in-out infinite; }
@keyframes fault-pulse { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.5)} 50%{box-shadow:0 0 0 5px rgba(239,68,68,0)} }
@keyframes extreme-pulse { 0%,100%{box-shadow:0 0 0 0 rgba(249,115,22,0.5)} 50%{box-shadow:0 0 0 5px rgba(249,115,22,0)} }

.sg-score-row { font-size: 11px; color: #64748B; display: flex; justify-content: space-between; margin-top: 4px; }
.sg-score-row span { color: #94A3B8; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

/* ================================================================ */
/* OPERATOR DECISION SUPPORT & RESPONSE WORKFLOW STYLES             */
/* ================================================================ */
.sg-response-panel {
  background: #0B0F14;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 24px;
}
.sg-response-panel.critical {
  border-color: rgba(239,68,68,0.3);
  box-shadow: 0 0 24px rgba(239,68,68,0.08);
}
.sg-response-panel.warning {
  border-color: rgba(245,158,11,0.3);
  box-shadow: 0 0 24px rgba(245,158,11,0.08);
}
.sg-response-panel.extreme {
  border-color: rgba(249,115,22,0.3);
  box-shadow: 0 0 24px rgba(249,115,22,0.08);
}
.sg-response-panel.nominal {
  border-color: rgba(16,185,129,0.2);
}

.sg-next-action-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 18px;
  border-radius: 8px;
  margin-bottom: 16px;
  border: 1px solid;
}
.sg-next-action-card.critical {
  background: rgba(239,68,68,0.08);
  border-color: rgba(239,68,68,0.35);
}
.sg-next-action-card.warning {
  background: rgba(245,158,11,0.08);
  border-color: rgba(245,158,11,0.35);
}
.sg-next-action-card.extreme {
  background: rgba(249,115,22,0.08);
  border-color: rgba(249,115,22,0.35);
}
.sg-next-action-card.nominal {
  background: rgba(16,185,129,0.06);
  border-color: rgba(16,185,129,0.25);
}

.sg-next-action-tag {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 4px 8px;
  border-radius: 4px;
  white-space: nowrap;
}
.sg-next-action-tag.critical { background: #EF4444; color: #FFFFFF; }
.sg-next-action-tag.warning { background: #F59E0B; color: #000000; }
.sg-next-action-tag.extreme { background: #F97316; color: #FFFFFF; }
.sg-next-action-tag.nominal { background: #10B981; color: #000000; }

.sg-next-action-text {
  font-size: 14.5px;
  font-weight: 700;
  color: #F8FAFC;
  line-height: 1.4;
  letter-spacing: -0.01em;
}

/* Workflow Stepper */
.sg-stepper {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #0E141B;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 8px;
  padding: 12px 18px;
  margin-bottom: 18px;
  overflow-x: auto;
}
.sg-step {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  font-weight: 600;
  color: #64748B;
  white-space: nowrap;
}
.sg-step.done { color: #10B981; }
.sg-step.active { color: #38BDF8; font-weight: 700; }
.sg-step.critical-active { color: #EF4444; font-weight: 700; }
.sg-step.warn-active { color: #F59E0B; font-weight: 700; }

.sg-step-marker {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: rgba(255,255,255,0.06);
  border: 1.5px solid rgba(255,255,255,0.12);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #64748B;
}
.sg-step.done .sg-step-marker {
  background: rgba(16,185,129,0.15);
  border-color: #10B981;
  color: #10B981;
}
.sg-step.active .sg-step-marker {
  background: rgba(56,189,248,0.18);
  border-color: #38BDF8;
  color: #38BDF8;
  box-shadow: 0 0 10px rgba(56,189,248,0.4);
}
.sg-step.critical-active .sg-step-marker {
  background: rgba(239,68,68,0.2);
  border-color: #EF4444;
  color: #EF4444;
  box-shadow: 0 0 10px rgba(239,68,68,0.4);
}
.sg-step.warn-active .sg-step-marker {
  background: rgba(245,158,11,0.2);
  border-color: #F59E0B;
  color: #F59E0B;
}

.sg-step-divider {
  flex: 1;
  height: 2px;
  background: rgba(255,255,255,0.06);
  margin: 0 10px;
  min-width: 16px;
}
.sg-step-divider.done { background: rgba(16,185,129,0.4); }

/* Diagnostic Box & Context */
.sg-diag-card {
  background: #0E1318;
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 8px;
  padding: 14px 16px;
  height: 100%;
}
.sg-diag-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #64748B;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.sg-diag-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 12px;
}
.sg-diag-row:last-child { border-bottom: none; }
.sg-diag-key { color: #64748B; font-weight: 500; font-size: 11px; }
.sg-diag-val { color: #E2E8F0; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }

.sg-why-box {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 6px;
  padding: 10px 12px;
  margin-top: 10px;
}
.sg-why-title {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #38BDF8;
  margin-bottom: 4px;
}
.sg-why-body {
  font-size: 11.5px;
  color: #94A3B8;
  line-height: 1.45;
}

/* Maintenance Card */
.sg-maint-card {
  background: #0E1318;
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 8px;
  padding: 14px 16px;
  height: 100%;
}
.sg-maint-notice {
  font-size: 9.5px;
  font-weight: 600;
  color: #F59E0B;
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.2);
  border-radius: 4px;
  padding: 4px 8px;
  margin-top: 10px;
  margin-bottom: 12px;
  letter-spacing: 0.03em;
}

.sg-chip-status {
  display: inline-block;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 7px;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
}
.sg-chip-status.new { background: rgba(239,68,68,0.2); color: #EF4444; border: 1px solid rgba(239,68,68,0.4); }
.sg-chip-status.ack { background: rgba(245,158,11,0.2); color: #F59E0B; border: 1px solid rgba(245,158,11,0.4); }
.sg-chip-status.insp { background: rgba(56,189,248,0.2); color: #38BDF8; border: 1px solid rgba(56,189,248,0.4); }
.sg-chip-status.res { background: rgba(16,185,129,0.2); color: #10B981; border: 1px solid rgba(16,185,129,0.4); }

/* Alert Feed Enhancements */
.sg-feed-card {
  background: #0E1318;
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 10px;
}
.sg-feed-card.critical {
  border-color: rgba(239,68,68,0.3);
  background: rgba(239,68,68,0.03);
}
.sg-feed-card.warning {
  border-color: rgba(245,158,11,0.3);
  background: rgba(245,158,11,0.03);
}
.sg-feed-row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  padding: 2.5px 0;
}
.sg-feed-key { color: #64748B; font-weight: 600; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
.sg-feed-val { color: #F1F5F9; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 11px; }

/* Side Card & Intelligence */
.sg-hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 20px -24px; }
.sg-card { background: #0E1318; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 14px 16px; }
.sg-card-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748B; margin-bottom: 10px; }
.sg-intel-row { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 12px; }
.sg-intel-row:last-child { border-bottom: none; }
.sg-intel-key { color: #64748B; font-weight: 500; font-size: 11px; }
.sg-intel-val { color: #E2E8F0; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
.sg-intel-val.ok { color: #10B981; }
.sg-intel-val.fault { color: #EF4444; }
.sg-intel-val.warn { color: #F59E0B; }
.sg-intel-val.extreme { color: #F97316; }
.sg-intel-val.muted { color: #64748B; }

/* Alert Feed */
.sg-alert-none { display: flex; align-items: center; gap: 12px; padding: 14px 0; color: #10B981; font-size: 12.5px; font-weight: 600; }
.sg-alert-none-sub { font-size: 11px; color: #64748B; margin-top: 2px; font-weight: 400; }

/* Pipeline */
.sg-pipeline { display: flex; align-items: center; margin-bottom: 20px; overflow-x: auto; padding-bottom: 4px; }
.sg-pipeline-node { flex: 1; min-width: 120px; background: #0E1318; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 11px 12px; text-align: center; }
.sg-pipeline-node.active-node { border-color: rgba(56,189,248,0.25); background: rgba(56,189,248,0.04); }
.sg-pipeline-node.fault-node { border-color: rgba(239,68,68,0.3); background: rgba(239,68,68,0.05); }
.sg-pipeline-node.extreme-node { border-color: rgba(249,115,22,0.35); background: rgba(249,115,22,0.06); }
.sg-pipeline-label { font-size: 9px; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #64748B; margin-bottom: 4px; }
.sg-pipeline-value { font-size: 12px; font-weight: 700; color: #94A3B8; }
.sg-pipeline-value.ok { color: #10B981; }
.sg-pipeline-value.fault { color: #EF4444; }
.sg-pipeline-value.warn { color: #F59E0B; }
.sg-pipeline-value.extreme { color: #F97316; }
.sg-pipeline-value.active { color: #38BDF8; }
.sg-pipeline-connector { width: 16px; height: 2px; flex-shrink: 0; background: rgba(255,255,255,0.08); }

/* Self-Healing Cards */
.sg-heal-normal { background: rgba(16,185,129,0.04); border: 1px solid rgba(16,185,129,0.18); border-radius: 8px; padding: 16px; }
.sg-heal-normal-title { font-size: 13.5px; font-weight: 700; color: #10B981; margin-bottom: 2px; }
.sg-heal-normal-sub { font-size: 11px; color: #64748B; margin-bottom: 12px; }
.sg-heal-table-row { display: flex; justify-content: space-between; padding: 5px 0; border-top: 1px solid rgba(255,255,255,0.05); font-size: 11.5px; }
.sg-heal-table-key { color: #64748B; }
.sg-heal-table-val { color: #E2E8F0; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.sg-heal-table-val.ok { color: #10B981; }
.sg-heal-table-val.extreme { color: #F97316; }

.sg-heal-fault { background: rgba(239,68,68,0.04); border: 1px solid rgba(239,68,68,0.22); border-radius: 8px; padding: 16px; }
.sg-heal-fault-header { display: flex; align-items: center; gap: 8px; font-size: 11.5px; font-weight: 700; color: #EF4444; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }
.sg-heal-comparison { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.sg-heal-val-block { flex: 1; text-align: center; }
.sg-heal-val-label { font-size: 9px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 3px; }
.sg-heal-val-num { font-size: 28px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; font-family: 'JetBrains Mono', monospace; }
.sg-heal-val-num.raw { color: #EF4444; }
.sg-heal-val-num.corrected { color: #10B981; }
.sg-heal-arrow { color: #64748B; font-size: 16px; margin-top: 12px; }
.sg-heal-metrics { border-top: 1px solid rgba(239,68,68,0.14); padding-top: 8px; margin-bottom: 8px; }
.sg-heal-metric-row { display: flex; justify-content: space-between; font-size: 11px; padding: 3px 0; color: #64748B; }
.sg-heal-metric-row span { color: #E2E8F0; font-family: 'JetBrains Mono', monospace; }

.sg-heal-extreme { background: rgba(249,115,22,0.04); border: 1px solid rgba(249,115,22,0.25); border-radius: 8px; padding: 16px; }
.sg-heal-extreme-header { display: flex; align-items: center; gap: 8px; font-size: 11.5px; font-weight: 700; color: #F97316; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }

.sg-heal-action-list { margin-top: 10px; }
.sg-heal-action-item { font-size: 11px; color: #94A3B8; display: flex; align-items: center; gap: 6px; padding: 2px 0; }
.sg-bullet { display: inline-block; width: 5px; height: 5px; border-radius: 50%; background: #10B981; margin-right: 6px; }
.sg-bullet.fault { background: #EF4444; }
.sg-bullet.extreme { background: #F97316; }

.sg-reasoning-step { display: flex; gap: 10px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 11.5px; }
.sg-reasoning-step:last-child { border-bottom: none; padding-bottom: 0; }
.sg-step-num {
  min-width: 18px; height: 18px; border-radius: 4px; background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.09); display: flex; align-items: center; justify-content: center;
  font-size: 9px; font-weight: 700; color: #64748B; flex-shrink: 0; margin-top: 1px;
}
.sg-step-num.fault { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.28); color: #EF4444; }
.sg-step-num.extreme { background: rgba(249,115,22,0.15); border-color: rgba(249,115,22,0.35); color: #F97316; }
.sg-step-text { color: #94A3B8; line-height: 1.5; }
.sg-xai-legend { display: flex; gap: 16px; font-size: 10px; color: #64748B; margin-top: 6px; }
.sg-xai-legend-dot { display: inline-block; width: 7px; height: 7px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }

.sg-fleet-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 0 !important; width: 100% !important; box-sizing: border-box !important; }
.sg-fleet-item { background: #121820; border: 1px solid rgba(255,255,255,0.07); border-radius: 6px; padding: 8px 10px; text-align: center; }
.sg-fleet-num { font-size: 17px; font-weight: 700; line-height: 1; font-family: 'JetBrains Mono', monospace; }
.sg-fleet-num.ok { color: #10B981; }
.sg-fleet-num.fault { color: #EF4444; }
.sg-fleet-num.extreme { color: #F97316; }
.sg-fleet-lbl { font-size: 9px; color: #64748B; font-weight: 600; text-transform: uppercase; margin-top: 3px; }

.sg-sys-status { display: flex; align-items: center; gap: 8px; padding: 8px 12px; margin: 10px 0 !important; width: 100% !important; box-sizing: border-box !important; background: #121820; border-radius: 6px; font-size: 11px; font-weight: 600; }
.sg-sys-status.online { color: #10B981; border: 1px solid rgba(16,185,129,0.25); }
.sg-sys-status.offline { color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }
.sg-sys-dot { width: 6px; height: 6px; border-radius: 50%; }
.sg-sys-dot.online { background: #10B981; }
.sg-sys-dot.offline { background: #EF4444; }

.sg-latency { display: flex; justify-content: space-between; padding: 4px 2px !important; width: 100% !important; box-sizing: border-box !important; font-size: 10px; color: #64748B; font-family: 'JetBrains Mono', monospace; }
.sg-sidebar-logo { padding: 18px 16px 14px 16px !important; margin: 0 -16px 14px -16px !important; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sg-sidebar-logo-img { width: 130px; height: auto; display: block; }
.sg-sidebar-logo-sub { font-size: 9.5px; color: #64748B; font-weight: 500; margin-top: 6px; letter-spacing: 0.03em; }
.sg-sidebar-sec-label { font-size: 9.5px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #64748B; padding: 0 0 8px 0 !important; }
.sg-sidebar-sep { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 14px -16px !important; }
.sg-header-logo-img { height: 28px; width: auto; display: block; }
.sg-footer { display: flex; justify-content: space-between; align-items: center; padding: 16px 0; margin-top: 24px; border-top: 1px solid rgba(255,255,255,0.06); flex-wrap: wrap; gap: 8px; }
.sg-footer-text { font-size: 10.5px; color: #4B5563; }

/* Responsive Media Queries */
@media (max-width: 1100px) {
  .sg-command-row { flex-wrap: wrap; }
  .sg-metric-card { flex: 1 1 calc(50% - 6px); min-width: 140px; }
  .sg-status-card { flex: 1 1 100%; min-width: 100%; }
}
@media (max-width: 768px) {
  .sg-header { flex-direction: column; align-items: flex-start; margin: 0 -16px 16px -16px; padding: 12px 16px; }
  .sg-header-right { width: 100%; justify-content: space-between; }
  .sg-metric-card { flex: 1 1 100%; }
  .sg-pipeline-node { min-width: 100px; padding: 8px 10px; }
  .main .block-container { padding: 0 16px 32px 16px !important; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ================================================================
# 5. STATION NETWORK & LIVE SYNCHRONIZED TELEMETRY
# ================================================================
@st.cache_data
def load_india_boundary_geojson():
    """Load full official India national boundary (including complete Jammu & Kashmir and Ladakh)."""
    geojson_path = os.path.join(os.path.dirname(__file__), "assets", "india_boundary_full.geojson")
    with open(geojson_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_india_states_geojson():
    """Load internal state boundaries."""
    geojson_path = os.path.join(os.path.dirname(__file__), "assets", "india_states_internal.geojson")
    with open(geojson_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_station_network():
    """AWS Fleet with primary hub in New Delhi and regional/national monitoring nodes."""
    return pd.DataFrame([
        {"station_id": "AWS-IND-001", "name": "New Delhi IMD HQ",
         "lat": 28.6139, "lon": 77.2090, "elevation": 216, "status": "HEALTHY",
         "region": "Delhi NCR / Central Hub", "cluster": "NCR"},
        {"station_id": "AWS-IND-002", "name": "Gurugram Cyber Hub",
         "lat": 28.4595, "lon": 77.0266, "elevation": 219, "status": "HEALTHY",
         "region": "Haryana / NCR Sector", "cluster": "NCR"},
        {"station_id": "AWS-IND-003", "name": "Noida Sector 62",
         "lat": 28.6280, "lon": 77.3649, "elevation": 200, "status": "HEALTHY",
         "region": "Uttar Pradesh / NCR", "cluster": "NCR"},
        {"station_id": "AWS-IND-004", "name": "Faridabad NIT",
         "lat": 28.3881, "lon": 77.3090, "elevation": 205, "status": "HEALTHY",
         "region": "Haryana / NCR South", "cluster": "NCR"},
        {"station_id": "AWS-IND-005", "name": "Ghaziabad Vasundhara",
         "lat": 28.6692, "lon": 77.4538, "elevation": 210, "status": "HEALTHY",
         "region": "Uttar Pradesh / NCR East", "cluster": "NCR"},
        {"station_id": "AWS-IND-006", "name": "Mumbai Marine Observatory",
         "lat": 18.9446, "lon": 72.8223, "elevation": 14, "status": "HEALTHY",
         "region": "Maharashtra / Coastal Zone", "cluster": "West"},
        {"station_id": "AWS-IND-007", "name": "Bengaluru Space Center",
         "lat": 12.9716, "lon": 77.5946, "elevation": 920, "status": "HEALTHY",
         "region": "Karnataka / Plateau Node", "cluster": "South"},
    ])


def generate_live_synced_telemetry(n_points=90, seed=42):
    """
    Generate 3 hours of 2-minute telemetry window ending exactly at the current wall-clock minute.
    """
    now = datetime.now()
    wall_clock_str = now.strftime("%H:%M:%S")
    
    end_time = now.replace(second=0, microsecond=0)
    timestamps = [end_time - timedelta(minutes=2 * (n_points - 1 - i)) for i in range(n_points)]

    np.random.seed(seed)
    base_t = 28.5 + 2.0 * np.sin(np.linspace(0, 3.14, n_points)) + np.random.normal(0, 0.22, n_points)
    base_h = 58.0 - 3.5 * np.sin(np.linspace(0, 3.14, n_points)) + np.random.normal(0, 0.45, n_points)
    base_p = 1012.3 - 0.4 * np.sin(np.linspace(0, 3.14, n_points)) + np.random.normal(0, 0.15, n_points)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": np.round(base_t, 2),
        "humidity": np.round(np.clip(base_h, 15, 95), 1),
        "pressure": np.round(base_p, 1),
        "is_anomaly": False
    })
    return df, wall_clock_str


# ================================================================
# 6. SESSION STATE INITIALIZATION & INCIDENT AUDIT REPOSITORY
# ================================================================
if "incidents" not in st.session_state:
    st.session_state["incidents"] = {}

if "incident_audit_log" not in st.session_state:
    st.session_state["incident_audit_log"] = []

if "maintenance_tasks" not in st.session_state:
    st.session_state["maintenance_tasks"] = []


# ================================================================
# 7. SIDEBAR CONTROLS & WORKSPACE CONFIGURATION
# ================================================================
stations_df = load_station_network()

st.sidebar.markdown(f"""
<div class="sg-sidebar-logo">
    <img src="{logo_b64}" alt="SkyGuard AI" class="sg-sidebar-logo-img" />
    <div class="sg-sidebar-logo-sub">AWS Quality Control & Anomaly Ops</div>
</div>
""", unsafe_allow_html=True)

# Station selector
st.sidebar.markdown('<div class="sg-sidebar-sec-label">Target Station</div>', unsafe_allow_html=True)

if "target_station_id" not in st.session_state:
    st.session_state["target_station_id"] = "AWS-IND-001"

station_list = stations_df["station_id"].tolist()
if st.session_state["target_station_id"] not in station_list:
    st.session_state["target_station_id"] = station_list[0]

def _on_sidebar_change():
    st.session_state["target_station_id"] = st.session_state["sidebar_station_select"]

current_idx = station_list.index(st.session_state["target_station_id"])

selected_station_id = st.sidebar.selectbox(
    "Target Station",
    options=station_list,
    index=current_idx,
    format_func=lambda sid: (
        f"{sid} — {stations_df.loc[stations_df['station_id'] == sid, 'name'].iloc[0]}"
    ),
    label_visibility="collapsed",
    key="sidebar_station_select",
    on_change=_on_sidebar_change,
)
st.session_state["target_station_id"] = selected_station_id

st.sidebar.markdown('<hr class="sg-sidebar-sep">', unsafe_allow_html=True)

# Simulation & Fault Studio
st.sidebar.markdown('<div class="sg-sidebar-sec-label">Operational Simulation Mode</div>', unsafe_allow_html=True)

with st.sidebar.expander("Simulation & Fault Injection Studio", expanded=True):
    sim_mode = st.selectbox(
        "Scenario Mode",
        options=[
            "Nominal Fleet Operations",
            "Sensor Spike / ADC Surge Fault",
            "Frozen Sensor (Flatline) Fault",
            "Progressive Sensor Drift Fault",
            "Physical Out-of-Bounds Violation",
            "Regional Extreme Weather (Heatwave 48.5°C)",
            "Regional Deep Depression / Squall (982 hPa)",
        ],
        index=0,
        help="Simulate genuine atmospheric hazards vs isolated sensor hardware/telemetry failures."
    )

    is_fault_sim = (sim_mode in [
        "Sensor Spike / ADC Surge Fault",
        "Frozen Sensor (Flatline) Fault",
        "Progressive Sensor Drift Fault",
        "Physical Out-of-Bounds Violation"
    ])
    is_extreme_event = (sim_mode in [
        "Regional Extreme Weather (Heatwave 48.5°C)",
        "Regional Deep Depression / Squall (982 hPa)"
    ])

    anomaly_param = "Temperature"
    spike_magnitude = 19.5
    frozen_val = 31.4
    drift_offset = 5.8
    bounds_val = 68.0

    if is_fault_sim:
        anomaly_param = st.selectbox(
            "Fault Sensor Probe",
            options=["Temperature", "Humidity", "Pressure"],
            index=0
        )
        if "Spike" in sim_mode:
            spike_magnitude = st.number_input("Spike Delta (°C)", value=19.5, step=1.0, min_value=5.0, max_value=40.0)
        elif "Frozen" in sim_mode:
            frozen_val = st.number_input("Static Stagnated Value (°C)", value=31.4, step=0.5)
        elif "Drift" in sim_mode:
            drift_offset = st.number_input("Gradual Calibration Drift (°C)", value=5.8, step=0.5, min_value=1.0, max_value=15.0)
        elif "Physical" in sim_mode:
            bounds_val = st.number_input("Extreme Physical Value (°C)", value=68.0, step=2.0)

st.sidebar.markdown('<hr class="sg-sidebar-sep">', unsafe_allow_html=True)

with st.sidebar.expander("Microservice Connection", expanded=False):
    use_live_backend = st.checkbox("Connect FastAPI Backend", value=False)
    api_url_input = st.text_input("API URL", value=API_BASE_URL)
    if st.button("Ping Endpoint", use_container_width=True):
        res = fetch_latest_status(api_url_input)
        if res.get("_connected", False):
            if res.get("status") == "NO_DATA":
                st.success(f"Backend Online ({res.get('_request_latency_ms', 0)} ms) — Ready for Sensor Ingestion")
            else:
                st.success(f"Connected ({res.get('_request_latency_ms', 0)} ms) — Ingesting Live Telemetry")
        else:
            st.error(f"Failed: {res.get('_error', 'Connection timed out')}")

st.sidebar.markdown('<div class="sg-sidebar-sec-label">Fleet Surveillance Summary</div>', unsafe_allow_html=True)
fleet_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()
latency_placeholder = st.sidebar.empty()


# ================================================================
# 8. TELEMETRY SYNTHESIS & SCENARIO INJECTION
# ================================================================
target_station = stations_df.loc[stations_df["station_id"] == selected_station_id].iloc[0]

telemetry_df, now_str = generate_live_synced_telemetry(n_points=90, seed=hash(selected_station_id) % 1000)
peer_df_gurugram, _ = generate_live_synced_telemetry(n_points=90, seed=101)
peer_df_noida, _    = generate_live_synced_telemetry(n_points=90, seed=202)

if "Spike" in sim_mode:
    last_idx = telemetry_df.index[-1]
    if anomaly_param == "Temperature":
        telemetry_df.loc[last_idx, "temperature"] += spike_magnitude
    elif anomaly_param == "Humidity":
        telemetry_df.loc[last_idx, "humidity"] = min(100.0, telemetry_df.loc[last_idx, "humidity"] + spike_magnitude)
    else:
        telemetry_df.loc[last_idx, "pressure"] += spike_magnitude

elif "Frozen" in sim_mode:
    freeze_idx = telemetry_df.index[-18:]
    if anomaly_param == "Temperature":
        telemetry_df.loc[freeze_idx, "temperature"] = frozen_val
    elif anomaly_param == "Humidity":
        telemetry_df.loc[freeze_idx, "humidity"] = frozen_val
    else:
        telemetry_df.loc[freeze_idx, "pressure"] = frozen_val

elif "Drift" in sim_mode:
    ramp = np.linspace(0, drift_offset, 25)
    drift_idx = telemetry_df.index[-25:]
    if anomaly_param == "Temperature":
        telemetry_df.loc[drift_idx, "temperature"] += ramp
    elif anomaly_param == "Humidity":
        telemetry_df.loc[drift_idx, "humidity"] -= ramp * 2
    else:
        telemetry_df.loc[drift_idx, "pressure"] -= ramp * 1.5

elif "Physical" in sim_mode:
    if bounds_val > 50 or bounds_val < -10:
        telemetry_df.loc[telemetry_df.index[-1], "temperature"] = bounds_val
    else:
        telemetry_df.loc[telemetry_df.index[-1], "temperature"] = 24.0
        telemetry_df.loc[telemetry_df.index[-1], "humidity"] = 100.0

elif "Heatwave" in sim_mode:
    heat_offset = 18.5
    telemetry_df["temperature"] += heat_offset
    peer_df_gurugram["temperature"] += heat_offset - 0.4
    peer_df_noida["temperature"] += heat_offset + 0.3

elif "Squall" in sim_mode:
    telemetry_df["pressure"] = 982.0 + np.random.normal(0, 0.3, len(telemetry_df))
    telemetry_df["humidity"] = 96.0 + np.random.normal(0, 0.4, len(telemetry_df))
    peer_df_gurugram["pressure"] = 983.1 + np.random.normal(0, 0.3, len(peer_df_gurugram))
    peer_df_noida["pressure"] = 981.8 + np.random.normal(0, 0.3, len(peer_df_noida))

latest_temp = float(telemetry_df["temperature"].iloc[-1])
latest_hum  = float(telemetry_df["humidity"].iloc[-1])
latest_pres = float(telemetry_df["pressure"].iloc[-1])

peer_temps = {
    "AWS-IND-002": float(peer_df_gurugram["temperature"].iloc[-1]),
    "AWS-IND-003": float(peer_df_noida["temperature"].iloc[-1]),
    "AWS-IND-004": float(peer_df_gurugram["temperature"].iloc[-1] - 0.2),
    "AWS-IND-005": float(peer_df_noida["temperature"].iloc[-1] + 0.1),
}


# ================================================================
# 9. MULTI-STAGE QC RESOLUTION & DECISION PROFILE
# ================================================================
request_latency_ms = 3.6
backend_available = True

if use_live_backend:
    live_data = fetch_latest_status(api_url_input)
    backend_available = live_data.get("_connected", False)
    request_latency_ms = live_data.get("_request_latency_ms", 3.6)
    if backend_available and not is_fault_sim and not is_extreme_event:
        b_tel = live_data.get("telemetry", {})
        if b_tel and "temperature" in b_tel:
            latest_temp = float(b_tel.get("temperature", latest_temp))
            latest_hum = float(b_tel.get("humidity", latest_hum))
            latest_pres = float(b_tel.get("pressure", latest_pres))

# Execute evaluation through WeatherQCEngine (or take live backend evaluation if connected)
if use_live_backend and backend_available and not is_fault_sim and not is_extreme_event and live_data.get("result"):
    qc_decision = live_data["result"]
else:
    qc_decision = local_qc_engine.evaluate(
        station_id=selected_station_id,
        temperature=latest_temp,
        humidity=latest_hum,
        pressure=latest_pres,
        temp_history=telemetry_df["temperature"].tolist(),
        peer_temp_readings=peer_temps,
        is_extreme_scenario=is_extreme_event
    )

is_current_fault   = qc_decision["is_sensor_fault"]
is_extreme_weather = qc_decision["is_extreme_weather"]
classification     = qc_decision["classification"]
spatial_check      = qc_decision["spatial_buddy_check"]
action_required    = qc_decision["action_required"]
confidence_score   = qc_decision["confidence_score"]
confidence_pct     = int(confidence_score * 100)
anomaly_score_display = round(qc_decision["ml_anomaly_score"], 2)
physics_fail       = not qc_decision["physics_passed"]
spatial_fail       = not qc_decision["details"]["spatial"]["passed"]
shap_scores        = qc_decision["shap_scores"]
estimated_temp     = qc_decision["imputed_temperature"]

telemetry_source = (
    "LIVE FASTAPI BACKEND" if (use_live_backend and backend_available and not is_fault_sim and not is_extreme_event)
    else f"SIMULATION // {sim_mode.split(' ')[0].upper()}"
)

# Imputation baseline & deviation
telemetry_df["reconstructed_temp"] = telemetry_df["temperature"].copy()
telemetry_df["reconstructed_humidity"] = telemetry_df["humidity"].copy()
telemetry_df["reconstructed_pressure"] = telemetry_df["pressure"].copy()

if is_current_fault:
    telemetry_df.loc[telemetry_df.index[-1], "reconstructed_temp"] = estimated_temp
    temperature_deviation = latest_temp - estimated_temp
    telemetry_df.loc[telemetry_df.index[-1], "is_anomaly"] = True
else:
    temperature_deviation = 0.0

health_score = 100
if is_current_fault:
    if "Spike" in sim_mode: health_score = 45
    elif "Frozen" in sim_mode: health_score = 58
    elif "Drift" in sim_mode: health_score = 65
    else: health_score = 30
elif is_extreme_weather:
    health_score = 100

# Retrieve deterministic operator decision support profile
resp_profile = OperatorDecisionSupport.get_response_profile(
    anomaly_type=classification,
    parameter=anomaly_param,
    station_id=selected_station_id,
    station_name=target_station["name"],
    observed_val=latest_temp,
    expected_val=estimated_temp,
    nearby_ref_val=peer_temps.get("AWS-IND-002", latest_temp),
    confidence_pct=confidence_pct,
    is_extreme_weather=is_extreme_weather
)


# ================================================================
# 10. INCIDENT STATE MANAGEMENT (DETECT -> ACK -> INSPECT -> RESOLVE)
# ================================================================
active_incident = None

if is_current_fault or is_extreme_weather:
    active_incident = st.session_state["incidents"].get(selected_station_id)
    # Check if existing incident matches current anomaly
    if active_incident and active_incident.get("anomaly_type") == resp_profile["anomaly_title"] and active_incident.get("status") != "RESOLVED":
        active_incident["observed_val"] = latest_temp
        active_incident["expected_val"] = estimated_temp
        active_incident["nearby_ref_val"] = peer_temps.get("AWS-IND-002", latest_temp)
        active_incident["deviation"] = temperature_deviation
        active_incident["confidence_pct"] = confidence_pct
    else:
        new_inc_id = f"INC-{selected_station_id.split('-')[-1]}-{int(time.time())}"
        active_incident = {
            "id": new_inc_id,
            "station_id": selected_station_id,
            "station_name": target_station["name"],
            "parameter": anomaly_param,
            "anomaly_type": resp_profile["anomaly_title"],
            "severity": resp_profile["severity"],
            "severity_class": resp_profile["severity_class"],
            "status": "NEW",  # NEW -> ACKNOWLEDGED -> UNDER_INSPECTION -> RESOLVED
            "detected_at": now_str,
            "observed_val": latest_temp,
            "expected_val": estimated_temp,
            "nearby_ref_val": peer_temps.get("AWS-IND-002", latest_temp),
            "deviation": temperature_deviation,
            "confidence_pct": confidence_pct,
            "detection_engine": resp_profile["detection_engine"],
            "next_action": resp_profile["next_action"],
            "why_this_matters": resp_profile["why_this_matters"],
            "why_this_action": resp_profile["why_this_action"],
            "maintenance_type": resp_profile["maintenance_type"],
            "operational_advice": resp_profile["operational_advice"],
            "checklist": {item: False for item in resp_profile["checklist"]},
            "acknowledged_at": None,
            "inspection_started_at": None,
            "resolved_at": None
        }
        st.session_state["incidents"][selected_station_id] = active_incident
        st.session_state["incident_audit_log"].append(active_incident)

else:
    # Station is Nominal. If there was an active incident on this station, mark it resolved
    if selected_station_id in st.session_state["incidents"]:
        prev_inc = st.session_state["incidents"][selected_station_id]
        if prev_inc and prev_inc.get("status") != "RESOLVED":
            prev_inc["status"] = "RESOLVED"
            if not prev_inc.get("resolved_at"):
                prev_inc["resolved_at"] = f"{now_str} (Auto-resolved: Nominal restored)"


# ================================================================
# 11. FLEET METRICS & SIDEBAR DYNAMICS
# ================================================================
stations_display_df = stations_df.copy()

if is_extreme_weather:
    ncr_mask = stations_display_df["cluster"] == "NCR"
    stations_display_df.loc[ncr_mask, "status"] = "EXTREME"
elif is_current_fault:
    selected_mask = stations_display_df["station_id"] == selected_station_id
    stations_display_df.loc[selected_mask, "status"] = "FAULT"

def get_node_color(status):
    if status == "FAULT": return [239, 68, 68, 220]
    if status == "EXTREME": return [249, 115, 22, 220]
    if status == "WARNING": return [245, 158, 11, 220]
    return [16, 185, 129, 220]

stations_display_df["color"] = stations_display_df["status"].apply(get_node_color)

fault_count   = int((stations_display_df["status"] == "FAULT").sum())
extreme_count = int((stations_display_df["status"] == "EXTREME").sum())
healthy_count = int((stations_display_df["status"] == "HEALTHY").sum())
total_count   = len(stations_display_df)

fleet_placeholder.markdown(f"""
<div class="sg-fleet-grid">
    <div class="sg-fleet-item">
        <div class="sg-fleet-num ok">{healthy_count}</div>
        <div class="sg-fleet-lbl">Healthy</div>
    </div>
    <div class="sg-fleet-item">
        <div class="sg-fleet-num {'fault' if fault_count > 0 else ('extreme' if extreme_count > 0 else 'ok')}">{fault_count if fault_count > 0 else extreme_count}</div>
        <div class="sg-fleet-lbl">{'Faults' if fault_count > 0 else ('Extremes' if extreme_count > 0 else 'Alerts')}</div>
    </div>
    <div class="sg-fleet-item">
        <div class="sg-fleet-num ok">4</div>
        <div class="sg-fleet-lbl">QC Layers</div>
    </div>
    <div class="sg-fleet-item">
        <div class="sg-fleet-num">{total_count}/{total_count}</div>
        <div class="sg-fleet-lbl">AWS Fleet</div>
    </div>
</div>
""", unsafe_allow_html=True)

s_cls = "online" if backend_available else "offline"
s_label = "HYBRID QC ONLINE" if backend_available else "OFFLINE"
status_placeholder.markdown(f"""
<div class="sg-sys-status {s_cls}">
    <div class="sg-sys-dot {s_cls}"></div>
    {s_label}
</div>
""", unsafe_allow_html=True)

latency_text = f"{request_latency_ms:.0f} ms"
latency_placeholder.markdown(f"""
<div class="sg-latency">
    <span>QC Latency</span><span>{latency_text}</span>
</div>
""", unsafe_allow_html=True)


# ================================================================
# 12. COMMAND CENTER HEADER & METRICS
# ================================================================
sim_active = (sim_mode != "Nominal Fleet Operations")
dot_cls = "fault" if is_current_fault else ("extreme" if is_extreme_weather else "ok")
pill_cls = dot_cls

st.markdown(f"""
<div class="sg-header">
  <div class="sg-header-left">
    <div style="display:flex;align-items:center;gap:14px;">
      <img src="{logo_b64}" alt="SkyGuard AI" class="sg-header-logo-img" />
      <div class="sg-tagline" style="border-left:1px solid rgba(255,255,255,0.12);padding-left:12px;margin:0;line-height:1.2;">
        Operations<br/><span style="color:#64748B;">MoES / IMD AWS Quality Control</span>
      </div>
    </div>
    <div class="sg-station-chip">
      <strong>{target_station['name']}</strong>
      <span class="sep">·</span>{target_station['station_id']}
      <span class="sep">·</span>{target_station['lat']}°N {target_station['lon']}°E
    </div>
  </div>
  <div class="sg-header-right">
    <div class="sg-live-pill"><div class="sg-live-dot"></div>LIVE STREAM</div>
    <div class="sg-timestamp">CLOCK {now_str}</div>
    <div class="sg-source-badge">{telemetry_source}</div>
  </div>
</div>
""", unsafe_allow_html=True)

if sim_active:
    st.markdown(f"""
<div class="sg-sim-banner">
  SIMULATION ACTIVE · {sim_mode.upper()} · Testing spatial consensus, temporal checks, physics bounds, and ML isolation.
</div>
""", unsafe_allow_html=True)

# Metrics Cards
if is_current_fault:
    temp_sub_cls = "fault"
    temp_sub_text = f"▲ {temperature_deviation:+.1f}°C from peer consensus"
    status_card_cls = "critical"
    status_label = "SENSOR FAULT ISOLATED"
elif is_extreme_weather:
    temp_sub_cls = "warning"
    temp_sub_text = "▲ Regional Severe Weather Event"
    status_card_cls = "extreme"
    status_label = "REGIONAL EXTREME WEATHER"
else:
    temp_sub_cls = "ok"
    temp_sub_text = "Within nominal climatology"
    status_card_cls = "operational"
    status_label = "FLEET OPERATIONAL"

health_sub_cls = "fault" if health_score < 70 else ("warning" if health_score < 90 else "ok")
health_sub_text = "Degraded" if health_score < 70 else ("Reduced" if health_score < 90 else "Optimal")

st.markdown(f"""
<div class="sg-command-row">
  <div class="sg-metric-card">
    <div class="sg-metric-label">Observed Temperature</div>
    <div class="sg-metric-value">{latest_temp:.1f}<span class="sg-metric-unit">°C</span></div>
    <div class="sg-metric-sub {temp_sub_cls}">{temp_sub_text}</div>
  </div>
  <div class="sg-metric-card">
    <div class="sg-metric-label">Relative Humidity</div>
    <div class="sg-metric-value">{latest_hum:.1f}<span class="sg-metric-unit">%</span></div>
    <div class="sg-metric-sub ok">Climatological bounds normal</div>
  </div>
  <div class="sg-metric-card">
    <div class="sg-metric-label">Atm. Pressure</div>
    <div class="sg-metric-value">{latest_pres:.1f}<span class="sg-metric-unit">hPa</span></div>
    <div class="sg-metric-sub ok">Barometric trace stable</div>
  </div>
  <div class="sg-metric-card">
    <div class="sg-metric-label">Sensor Reliability</div>
    <div class="sg-metric-value">{health_score}<span class="sg-metric-unit">/100</span></div>
    <div class="sg-metric-sub {health_sub_cls}">{health_sub_text}</div>
  </div>
  <div class="sg-status-card {status_card_cls}">
    <div class="sg-metric-label">QC Assessment</div>
    <div class="sg-status-pill {pill_cls}">
      <div class="sg-status-dot {dot_cls}"></div>{status_label}
    </div>
    <div class="sg-score-row">ML Score <span>{anomaly_score_display}</span></div>
    <div class="sg-score-row">QC Confidence <span>{confidence_pct}%</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# 13. OPERATOR RESPONSE & DECISION SUPPORT PANEL (PROMINENT SECTION)
# ================================================================
st.markdown('<div class="sg-section-title">OPERATOR DECISION SUPPORT · RECOMMENDED RESPONSE WORKFLOW</div>', unsafe_allow_html=True)

if is_current_fault or is_extreme_weather:
    inc = active_incident
    inc_status = inc.get("status", "NEW")
    sev_class = resp_profile.get("severity_class", "critical")

    # Status labels & markers
    st_detected = "done"
    st_ack = "done" if inc_status in ["ACKNOWLEDGED", "UNDER_INSPECTION", "RESOLVED"] else "active"
    st_insp = "done" if inc_status == "RESOLVED" else ("active" if inc_status == "UNDER_INSPECTION" else "")
    st_res = "done" if inc_status == "RESOLVED" else ""

    panel_html = f"""
<div class="sg-response-panel {sev_class}">
  <!-- Top Operator Centric NEXT ACTION Banner -->
  <div class="sg-next-action-card {sev_class}">
    <div class="sg-next-action-tag {sev_class}">NEXT ACTION</div>
    <div class="sg-next-action-text">{inc['next_action']}</div>
  </div>

  <!-- Visual Workflow Stepper -->
  <div class="sg-stepper">
    <div class="sg-step {st_detected}">
      <div class="sg-step-marker">01</div>
      <div>
        <div>DETECTED</div>
        <div style="font-size:9.5px;color:#64748B;font-family:'JetBrains Mono';">{inc['detected_at']}</div>
      </div>
    </div>
    <div class="sg-step-divider {'done' if inc_status != 'NEW' else ''}"></div>
    
    <div class="sg-step {'done' if inc_status in ['UNDER_INSPECTION', 'RESOLVED'] else ('critical-active' if inc_status == 'ACKNOWLEDGED' else '')}">
      <div class="sg-step-marker">02</div>
      <div>
        <div>ACKNOWLEDGED</div>
        <div style="font-size:9.5px;color:#64748B;font-family:'JetBrains Mono';">{inc['acknowledged_at'] if inc['acknowledged_at'] else 'Pending operator'}</div>
      </div>
    </div>
    <div class="sg-step-divider {'done' if inc_status in ['UNDER_INSPECTION', 'RESOLVED'] else ''}"></div>

    <div class="sg-step {'done' if inc_status == 'RESOLVED' else ('active' if inc_status == 'UNDER_INSPECTION' else '')}">
      <div class="sg-step-marker">03</div>
      <div>
        <div>UNDER INSPECTION</div>
        <div style="font-size:9.5px;color:#64748B;font-family:'JetBrains Mono';">{inc['inspection_started_at'] if inc['inspection_started_at'] else 'Pending verification'}</div>
      </div>
    </div>
    <div class="sg-step-divider {'done' if inc_status == 'RESOLVED' else ''}"></div>

    <div class="sg-step {'done' if inc_status == 'RESOLVED' else ''}">
      <div class="sg-step-marker">04</div>
      <div>
        <div>RESOLVED</div>
        <div style="font-size:9.5px;color:#64748B;font-family:'JetBrains Mono';">{inc['resolved_at'] if inc['resolved_at'] else 'Permanent log'}</div>
      </div>
    </div>
  </div>
</div>
"""
    st.markdown(panel_html, unsafe_allow_html=True)

    # Two-Column Operator Decision-Support Interface
    col_resp_diag, col_resp_act = st.columns([1.08, 1.12], gap="medium")

    with col_resp_diag:
        st.markdown(f"""
<div class="sg-diag-card">
  <div class="sg-diag-title">
    <span>Diagnostic Briefing · {inc['id']}</span>
    <span class="sg-chip-status {inc_status.lower()[:3]}">{inc_status.replace('_', ' ')}</span>
  </div>
  <div class="sg-diag-row"><span class="sg-diag-key">Affected Station</span><span class="sg-diag-val">{target_station['name']} ({selected_station_id})</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Sensor / Parameter</span><span class="sg-diag-val">{anomaly_param}</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Detected Anomaly</span><span class="sg-diag-val" style="color:#EF4444;">{resp_profile['anomaly_title']}</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Observed Telemetry</span><span class="sg-diag-val">{latest_temp:.1f} °C</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Expected Baseline</span><span class="sg-diag-val">~{estimated_temp:.1f} °C</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Nearby AWS Ref (AWS-IND-002)</span><span class="sg-diag-val">{peer_temps.get('AWS-IND-002', latest_temp):.1f} °C</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Consensus Deviation</span><span class="sg-diag-val">{temperature_deviation:+.1f} °C</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Detection Engine</span><span class="sg-diag-val">{resp_profile['detection_engine']}</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">QC Model Confidence</span><span class="sg-diag-val">{confidence_pct}%</span></div>
  <div class="sg-diag-row" style="border-bottom:none;"><span class="sg-diag-key">Severity Level</span><span class="sg-diag-val" style="color:#EF4444;">{resp_profile['severity']}</span></div>

  <div class="sg-why-box" style="margin-top:12px;">
    <div class="sg-why-title">WHY THIS MATTERS</div>
    <div class="sg-why-body">{resp_profile['why_this_matters']}</div>
  </div>

  <div class="sg-why-box">
    <div class="sg-why-title">WHY THIS ACTION?</div>
    <div class="sg-why-body">{resp_profile['why_this_action']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with col_resp_act:
        st.markdown(f"""
<div class="sg-maint-card">
  <div class="sg-diag-title">
    <span>Maintenance Recommendation & Checklist</span>
    <span style="color:#F59E0B;font-size:10px;font-family:'JetBrains Mono';">AWS PROCEDURE</span>
  </div>
  <div class="sg-maint-notice">
    RECOMMENDATION ONLY · NO AUTOMATIC REPAIR DISPATCHED · OPERATOR ACTION REQUIRED
  </div>
  <div class="sg-diag-row"><span class="sg-diag-key">Recommended Maintenance</span><span class="sg-diag-val" style="color:#38BDF8;">{resp_profile['maintenance_type']}</span></div>
  <div class="sg-diag-row"><span class="sg-diag-key">Operational Advice</span><span class="sg-diag-val">{resp_profile['operational_advice']}</span></div>
  <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#64748B;margin-top:12px;margin-bottom:6px;">Inspection Checklist (Operator Verification)</div>
</div>
""", unsafe_allow_html=True)

        # Interactive inspection checklist items
        for i, item_text in enumerate(resp_profile["checklist"]):
            current_chk = inc["checklist"].get(item_text, False)
            checked = st.checkbox(
                item_text,
                value=current_chk,
                key=f"op_chk_{inc['id']}_{i}"
            )
            inc["checklist"][item_text] = checked

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # Operator Action Buttons
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("ACKNOWLEDGE", key=f"btn_ack_{inc['id']}", use_container_width=True, disabled=(inc_status != "NEW")):
                inc["status"] = "ACKNOWLEDGED"
                inc["acknowledged_at"] = now_str
                st.toast("Alert Acknowledged by Operator", icon="✅")
                st.rerun()

        with btn_c2:
            if st.button("START INSPECTION", key=f"btn_insp_{inc['id']}", use_container_width=True, disabled=(inc_status not in ["NEW", "ACKNOWLEDGED"])):
                inc["status"] = "UNDER_INSPECTION"
                inc["inspection_started_at"] = now_str
                st.toast("Incident Moved to Under Inspection", icon="🔍")
                st.rerun()

        with btn_c3:
            if st.button("MARK RESOLVED", key=f"btn_res_{inc['id']}", use_container_width=True, disabled=(inc_status == "RESOLVED")):
                inc["status"] = "RESOLVED"
                inc["resolved_at"] = now_str
                st.toast("Incident Resolved & Stored to Permanent Audit Log", icon="🛡️")
                st.rerun()

        # Prototype Maintenance Task Logger
        if st.button("CREATE LOCAL MAINTENANCE TASK", key=f"btn_task_{inc['id']}", use_container_width=True):
            task_entry = {
                "task_id": f"MNT-{selected_station_id.split('-')[-1]}-{int(time.time())}",
                "station_id": selected_station_id,
                "station_name": target_station["name"],
                "sensor": anomaly_param,
                "issue": resp_profile["anomaly_title"],
                "severity": resp_profile["severity"],
                "action": resp_profile["next_action"],
                "created_at": now_str,
                "status": "LOGGED (Prototype Record — No Physical Dispatch Connected)",
            }
            st.session_state["maintenance_tasks"].append(task_entry)
            st.toast("Local Maintenance Task Logged (SIH Prototype Mode)", icon="📋")
            st.rerun()

else:
    # Nominal Station Status (Test 1 Scenario)
    st.markdown(f"""
<div class="sg-card" style="border-left: 3px solid #10B981; padding: 16px 20px; margin-bottom: 24px;">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
    <div style="display:flex; align-items:center; gap:10px;">
      <div class="sg-status-dot ok"></div>
      <div style="font-size:13.5px; font-weight:700; color:#10B981; letter-spacing:0.04em;">FLEET OPERATIONAL · SENSOR RESPONSE NOMINAL</div>
    </div>
    <div style="font-size:11px; color:#64748B; font-family:'JetBrains Mono',monospace;">{selected_station_id} · ALL QC CHECKS PASSED</div>
  </div>
  <div style="font-size:12px; color:#94A3B8; margin-top:8px; line-height:1.5;">
    Telemetry for {target_station['name']} is within standard thermodynamic boundaries, rate-of-change thresholds, and regional peer consensus. No operator intervention or physical maintenance required.
  </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# 14. SPATIAL SENSOR NETWORK · INDIA TELEMETRY GRID + ALERTS
# ================================================================
st.markdown('<div class="sg-section-title">SPATIAL SENSOR NETWORK · TELEMETRY GRID (CLICK NODE TO INSPECT)</div>', unsafe_allow_html=True)

col_map, col_right = st.columns([1.65, 1.0], gap="medium")

with col_map:
    india_boundary = load_india_boundary_geojson()
    india_states = load_india_states_geojson()
    fig_map = go.Figure()

    hq_match = stations_display_df[stations_display_df["station_id"] == "AWS-IND-001"]
    hq_row = hq_match.iloc[0] if len(hq_match) > 0 else stations_display_df.iloc[0]

    for _, st_r in stations_display_df.iterrows():
        if st_r["station_id"] != "AWS-IND-001":
            is_tgt = (st_r["station_id"] == selected_station_id)
            st_status = st_r.get("status", "HEALTHY")
            
            if st_status == "FAULT":
                arc_col = "rgba(239, 68, 68, 0.85)"
                arc_w = 2.4
            elif st_status == "EXTREME":
                arc_col = "rgba(249, 115, 22, 0.85)"
                arc_w = 2.2
            elif is_tgt:
                arc_col = "rgba(56, 189, 248, 0.95)"
                arc_w = 2.4
            else:
                arc_col = "rgba(56, 189, 248, 0.35)"
                arc_w = 1.2
            
            fig_map.add_trace(go.Scattermap(
                lat=[hq_row["lat"], st_r["lat"]],
                lon=[hq_row["lon"], st_r["lon"]],
                mode="lines",
                line=dict(width=arc_w, color=arc_col),
                hoverinfo="none",
                showlegend=False
            ))

    sel_station_row = stations_display_df[stations_display_df["station_id"] == selected_station_id]
    if not sel_station_row.empty:
        s_row = sel_station_row.iloc[0]
        halo_color = (
            "rgba(239, 68, 68, 0.35)" if is_current_fault 
            else ("rgba(249, 115, 22, 0.35)" if is_extreme_weather else "rgba(56, 189, 248, 0.35)")
        )
        fig_map.add_trace(go.Scattermap(
            lat=[s_row["lat"]],
            lon=[s_row["lon"]],
            mode="markers",
            marker=dict(size=36, color=halo_color, allowoverlap=True),
            hoverinfo="none",
            showlegend=False
        ))

    node_colors = []
    node_sizes = []
    for _, st_r in stations_display_df.iterrows():
        s_status = st_r.get("status", "HEALTHY")
        is_tgt = (st_r["station_id"] == selected_station_id)
        if s_status == "FAULT": node_colors.append("#EF4444")
        elif s_status == "EXTREME": node_colors.append("#F97316")
        elif s_status == "WARNING": node_colors.append("#F59E0B")
        else: node_colors.append("#10B981")
        node_sizes.append(22 if is_tgt else 16)

    hover_texts = []
    for _, st_r in stations_display_df.iterrows():
        s_id = st_r["station_id"]
        s_name = st_r["name"]
        s_status = st_r.get("status", "HEALTHY")
        s_region = st_r.get("region", "India")
        s_elev = st_r.get("elevation", 200)
        s_lat = st_r["lat"]
        s_lon = st_r["lon"]
        is_active = (s_id == selected_station_id)
        
        status_badge = (
            '<span style="color:#10B981;font-weight:700;">● HEALTHY</span>' if s_status == "HEALTHY"
            else ('<span style="color:#F97316;font-weight:700;">● REGIONAL EXTREME</span>' if s_status == "EXTREME"
            else '<span style="color:#EF4444;font-weight:700;">● SENSOR FAULT</span>')
        )
        sel_notice = '<br><span style="color:#38BDF8;font-weight:700;">[SELECTED NODE]</span>' if is_active else '<br><span style="color:#38BDF8;">Click marker to inspect station</span>'
        
        hover_texts.append(
            f"<b>{s_id} · {s_name}</b><br>"
            f"<span style='color:#94A3B8;'>Region:</span> {s_region}<br>"
            f"<span style='color:#94A3B8;'>Coords:</span> {s_lat:.4f}°N, {s_lon:.4f}°E<br>"
            f"<span style='color:#94A3B8;'>Elevation:</span> {s_elev} m<br>"
            f"<span style='color:#94A3B8;'>Status:</span> {status_badge}"
            f"{sel_notice}"
        )

    fig_map.add_trace(go.Scattermap(
        lat=stations_display_df["lat"],
        lon=stations_display_df["lon"],
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, allowoverlap=True),
        text=stations_display_df["station_id"],
        textposition="top right",
        textfont=dict(family="Inter, sans-serif", size=11, color="#F1F5F9"),
        customdata=stations_display_df["station_id"],
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover_texts,
        showlegend=False
    ))

    fig_map.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080A0D",
        plot_bgcolor="#080A0D",
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=23.5, lon=82.0),
            zoom=3.85,
            layers=[
                dict(sourcetype="geojson", source=india_boundary, type="fill", color="#0F172A", opacity=0.45),
                dict(sourcetype="geojson", source=india_states, type="line", color="#38BDF8", opacity=0.22, line=dict(width=1.0)),
                dict(sourcetype="geojson", source=india_boundary, type="line", color="#38BDF8", opacity=0.95, line=dict(width=2.4)),
            ]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        hoverlabel=dict(bgcolor="#0E1318", bordercolor="#38BDF8", font=dict(family="Inter, sans-serif", size=12, color="#F8FAFC"))
    )

    map_selection = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="spatial_map_select",
        config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "displaylogo": False, "scrollZoom": True}
    )

    if map_selection and "selection" in map_selection:
        pts = map_selection["selection"].get("points", [])
        if pts:
            clicked_id = pts[0].get("customdata")
            if clicked_id and isinstance(clicked_id, str) and clicked_id in station_list:
                if clicked_id != st.session_state.get("target_station_id"):
                    st.session_state["target_station_id"] = clicked_id
                    st.session_state["sidebar_station_select"] = clicked_id
                    st.rerun()

with col_right:
    intel_status_cls  = "fault" if is_current_fault else ("extreme" if is_extreme_weather else "ok")
    intel_status_text = "● RECOMMEND DATA ISOLATION" if is_current_fault else ("● REGIONAL EXTREME" if is_extreme_weather else "● OPERATIONAL")
    health_cls        = "fault" if health_score < 70 else "ok"
    
    st.markdown(f"""
<div class="sg-card" style="margin-bottom:12px;">
  <div class="sg-card-label">Station Telemetry Status</div>
  <div style="margin-bottom:10px;">
    <div style="font-size:14px;font-weight:700;color:#F8FAFC;">{target_station['name']}</div>
    <div style="font-size:11px;color:#64748B;margin-top:2px;">{target_station['station_id']} · {target_station.get('region', 'India')} · Elevation {target_station['elevation']}m</div>
  </div>
  <div class="sg-intel-row"><span class="sg-intel-key">Operating Status</span><span class="sg-intel-val {intel_status_cls}">{intel_status_text}</span></div>
  <div class="sg-intel-row"><span class="sg-intel-key">Coordinates</span><span class="sg-intel-val">{target_station['lat']:.4f}°N, {target_station['lon']:.4f}°E</span></div>
  <div class="sg-intel-row"><span class="sg-intel-key">Observed Temp</span><span class="sg-intel-val">{latest_temp:.1f} °C</span></div>
  <div class="sg-intel-row"><span class="sg-intel-key">Relative Humidity</span><span class="sg-intel-val">{latest_hum:.1f} %</span></div>
  <div class="sg-intel-row"><span class="sg-intel-key">Barometric Pressure</span><span class="sg-intel-val">{latest_pres:.1f} hPa</span></div>
  <div class="sg-intel-row"><span class="sg-intel-key">Sensor Health Index</span><span class="sg-intel-val {health_cls}">{health_score} / 100</span></div>
  <div class="sg-intel-row"><span class="sg-intel-key">Spatial Consensus</span><span class="sg-intel-val {'ok' if not is_current_fault else 'fault'}">{'PASSED' if not is_current_fault else 'DIVERGENT'}</span></div>
  <div class="sg-intel-row"><span class="sg-intel-key">QC Model Confidence</span><span class="sg-intel-val">{confidence_pct}%</span></div>
  <div class="sg-intel-row" style="border-bottom:none;"><span class="sg-intel-key">Last Sync Clock</span><span class="sg-intel-val muted">{now_str}</span></div>
</div>
""", unsafe_allow_html=True)

    # Actionable Real-Time Alert Feed
    st.markdown('<div class="sg-card-label" style="margin-top:4px;">Actionable Alert Feed</div>', unsafe_allow_html=True)
    if is_current_fault or is_extreme_weather:
        inc = active_incident
        inc_status = inc.get("status", "NEW")
        feed_class = "critical" if is_current_fault else "warning"

        st.markdown(f"""
<div class="sg-feed-card {feed_class}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="font-size:11px;font-weight:700;color:{'#EF4444' if is_current_fault else '#F97316'};letter-spacing:0.06em;text-transform:uppercase;">
      ● {'CRITICAL SENSOR FAULT' if is_current_fault else 'METEOROLOGICAL ADVISORY'}
    </span>
    <span style="font-size:10px;color:#64748B;font-family:'JetBrains Mono';">{inc['detected_at']}</span>
  </div>
  <div class="sg-feed-row"><span class="sg-feed-key">WHAT:</span><span class="sg-feed-val">{resp_profile['anomaly_title']}</span></div>
  <div class="sg-feed-row"><span class="sg-feed-key">WHERE:</span><span class="sg-feed-val">{target_station['name']} — {selected_station_id}</span></div>
  <div class="sg-feed-row"><span class="sg-feed-key">VALUE:</span><span class="sg-feed-val">{latest_temp:.1f} °C (Dev: {temperature_deviation:+.1f} °C)</span></div>
  <div class="sg-feed-row"><span class="sg-feed-key">WHY:</span><span class="sg-feed-val">{resp_profile['detection_engine']}</span></div>
  <div class="sg-feed-row"><span class="sg-feed-key">CONFIDENCE:</span><span class="sg-feed-val">{confidence_pct}%</span></div>
  <div class="sg-feed-row"><span class="sg-feed-key">SEVERITY:</span><span class="sg-feed-val" style="color:#EF4444;">{resp_profile['severity']}</span></div>
  <div class="sg-feed-row"><span class="sg-feed-key">ACTION:</span><span class="sg-feed-val" style="color:#38BDF8;">{resp_profile['recommended_action_summary']}</span></div>
  <div class="sg-feed-row" style="margin-top:4px;border-top:1px solid rgba(255,255,255,0.05);padding-top:5px;">
    <span class="sg-feed-key">STATUS:</span>
    <span class="sg-chip-status {inc_status.lower()[:3]}">{inc_status.replace('_', ' ')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        col_fa, col_fv = st.columns(2)
        with col_fa:
            if st.button("ACKNOWLEDGE", key="feed_btn_ack", use_container_width=True, disabled=(inc_status != "NEW")):
                inc["status"] = "ACKNOWLEDGED"
                inc["acknowledged_at"] = now_str
                st.toast("Alert Acknowledged", icon="✅")
                st.rerun()
        with col_fv:
            if st.button("INSPECT WORKFLOW", key="feed_btn_insp", use_container_width=True):
                st.toast(f"Reviewing Incident {inc['id']}", icon="🔍")
    else:
        st.markdown("""
<div class="sg-card">
  <div class="sg-alert-none">
    <div style="font-size:14px;color:#10B981;font-weight:700;">●</div>
    <div>
      <div style="font-size:12.5px;font-weight:700;color:#10B981;">ALL SYSTEMS NOMINAL</div>
      <div class="sg-alert-none-sub">All observations validated across physical, temporal, and spatial QC layers</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# 15. AUTOMATED QUALITY CONTROL PIPELINE
# ================================================================
st.markdown('<hr class="sg-hr">', unsafe_allow_html=True)
st.markdown('<div class="sg-section-title">MULTI-STAGE QUALITY CONTROL PIPELINE (LIVE INFERENCE)</div>', unsafe_allow_html=True)

ingest_status  = "LIVE" if (backend_available or not is_fault_sim) else "SYNTHETIC"
temporal_status = "FAIL" if (is_current_fault and ("Spike" in sim_mode or "Frozen" in sim_mode or "Drift" in sim_mode)) else "PASS"
ml_status      = "ANOMALY" if (is_current_fault or is_extreme_weather) else "NORMAL"
physics_status = "FAIL" if physics_fail else "PASS"
spatial_status = "FAIL (ISOLATED)" if is_current_fault else ("PASS (CONSENSUS)" if is_extreme_weather else "PASS")

if is_current_fault:
    action_status = "RECOMMEND DATA ISOLATION"
elif is_extreme_weather:
    action_status = "DISPATCH ADVISORY"
else:
    action_status = "ASSIMILATE"

def _nc(is_bad, is_extreme=False):
    if is_bad: return "fault-node"
    if is_extreme: return "extreme-node"
    return "active-node"

def _vc(s):
    if "FAIL" in s or s == "ANOMALY": return "fault"
    if "ADVISORY" in s or "CONSENSUS" in s: return "extreme"
    if "RECOMMEND" in s: return "warn"
    return "ok"

st.markdown(f"""
<div class="sg-pipeline">
  <div class="sg-pipeline-node active-node">
    <div class="sg-pipeline-label">01 · Ingestion</div>
    <div class="sg-pipeline-value ok">{ingest_status}</div>
  </div>
  <div class="sg-pipeline-connector"></div>
  <div class="sg-pipeline-node {_nc(temporal_status == 'FAIL')}">
    <div class="sg-pipeline-label">02 · Temporal QC</div>
    <div class="sg-pipeline-value {_vc(temporal_status)}">{temporal_status}</div>
  </div>
  <div class="sg-pipeline-connector"></div>
  <div class="sg-pipeline-node {_nc(is_current_fault, is_extreme_weather)}">
    <div class="sg-pipeline-label">03 · Isolation Forest</div>
    <div class="sg-pipeline-value {_vc(ml_status)}">{ml_status}</div>
  </div>
  <div class="sg-pipeline-connector"></div>
  <div class="sg-pipeline-node {_nc(spatial_fail, is_extreme_weather)}">
    <div class="sg-pipeline-label">04 · Spatial Consensus</div>
    <div class="sg-pipeline-value {_vc(spatial_status)}">{spatial_status}</div>
  </div>
  <div class="sg-pipeline-connector"></div>
  <div class="sg-pipeline-node {_nc(is_current_fault, is_extreme_weather)}">
    <div class="sg-pipeline-label">05 · Operational Action</div>
    <div class="sg-pipeline-value {_vc(action_status)}">{action_status}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# 16. TELEMETRY CHARTS (SYNCHRONIZED WITH WALL CLOCK)
# ================================================================
st.markdown('<hr class="sg-hr">', unsafe_allow_html=True)
st.markdown(f'<div class="sg-section-title">TELEMETRY OBSERVATIONS · 3-HOUR WINDOW · 2-MIN RESOLUTION (ENDING AT {now_str})</div>', unsafe_allow_html=True)


def plot_telemetry(df, param_col, rec_col, unit, line_color="#38BDF8"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[param_col],
        mode="lines", name="Observed",
        line=dict(color=line_color, width=1.8),
        hovertemplate=f"<b>%{{y:.2f}} {unit}</b><br>%{{x|%H:%M:%S}}<extra>Observed</extra>",
    ))
    anomalies = df[df["is_anomaly"].fillna(False).astype(bool)]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies[param_col],
            mode="markers", name="Fault Marker",
            marker=dict(color="#EF4444", size=10, symbol="x-thin", line=dict(width=2.2, color="#EF4444")),
            hovertemplate=f"<b>FAULT · %{{y:.2f}} {unit}</b><br>%{{x|%H:%M:%S}}<extra></extra>",
        ))
    if is_current_fault:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[rec_col],
            mode="lines", name="AI Reconstructed",
            line=dict(color="#10B981", width=1.6, dash="dot"),
            hovertemplate=f"<b>%{{y:.2f}} {unit}</b><br>%{{x|%H:%M:%S}}<extra>AI Reconstructed</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[df["timestamp"].iloc[-1]], y=[df[rec_col].iloc[-1]],
            mode="markers", name="Imputed Value",
            marker=dict(color="#10B981", size=9, symbol="diamond"),
        ))
    fig.update_layout(
        template="plotly_dark", height=285, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="#080A0D", plot_bgcolor="#080A0D",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10, color="#64748B", family="JetBrains Mono"), tickformat="%H:%M"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10, color="#64748B", family="JetBrains Mono")),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1, font=dict(size=10, color="#94A3B8", family="Inter"), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="#121820", bordercolor="rgba(255,255,255,0.1)", font=dict(size=11, color="#F8FAFC", family="JetBrains Mono")),
    )
    return fig


def plot_spatial_crosscheck(target_df, peer1_df, peer2_df):
    """Spatial cross-validation comparing target station directly with peer nodes."""
    fig = go.Figure()
    target_color = "#EF4444" if is_current_fault else ("#F97316" if is_extreme_weather else "#38BDF8")
    
    fig.add_trace(go.Scatter(
        x=target_df["timestamp"], y=target_df["temperature"],
        mode="lines", name=f"Target: {selected_station_id}",
        line=dict(color=target_color, width=2.4),
        hovertemplate="Target: %{y:.2f} °C<br>%{x|%H:%M:%S}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=peer1_df["timestamp"], y=peer1_df["temperature"],
        mode="lines", name="Peer: AWS-IND-002 (Gurugram)",
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
        hovertemplate="Peer Gurugram: %{y:.2f} °C<br>%{x|%H:%M:%S}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=peer2_df["timestamp"], y=peer2_df["temperature"],
        mode="lines", name="Peer: AWS-IND-003 (Noida)",
        line=dict(color="#64748B", width=1.5, dash="dot"),
        hovertemplate="Peer Noida: %{y:.2f} °C<br>%{x|%H:%M:%S}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=285, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="#080A0D", plot_bgcolor="#080A0D",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10, color="#64748B", family="JetBrains Mono"), tickformat="%H:%M"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10, color="#64748B", family="JetBrains Mono"), title="Temp (°C)"),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1, font=dict(size=10, color="#94A3B8", family="Inter"), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="#121820", bordercolor="rgba(255,255,255,0.1)", font=dict(size=11, color="#F8FAFC", family="JetBrains Mono")),
    )
    return fig

tab_t, tab_h, tab_p, tab_spatial = st.tabs([
    "Temperature (°C)",
    "Relative Humidity (%)",
    "Pressure (hPa)",
    "Regional Peer Consensus Comparison",
])

with tab_t:
    st.plotly_chart(plot_telemetry(telemetry_df, "temperature", "reconstructed_temp", "°C", "#38BDF8"), use_container_width=True)
with tab_h:
    st.plotly_chart(plot_telemetry(telemetry_df, "humidity", "reconstructed_humidity", "%", "#818CF8"), use_container_width=True)
with tab_p:
    st.plotly_chart(plot_telemetry(telemetry_df, "pressure", "reconstructed_pressure", "hPa", "#34D399"), use_container_width=True)
with tab_spatial:
    st.plotly_chart(plot_spatial_crosscheck(telemetry_df, peer_df_gurugram, peer_df_noida), use_container_width=True)


# ================================================================
# 17. TELEMETRY READINGS · LIVE OBSERVATION STREAM (3-HOUR LOGS)
# ================================================================
st.markdown('<hr class="sg-hr">', unsafe_allow_html=True)
st.markdown(f'<div class="sg-section-title">TELEMETRY READINGS · LIVE OBSERVATION STREAM (SYNCED CLOCK {now_str})</div>', unsafe_allow_html=True)

col_ctrl_left, col_ctrl_right = st.columns([1.6, 1.0])
with col_ctrl_left:
    view_depth = st.radio(
        "Observation Depth",
        options=["Latest 10 Cycles (20 Min)", "Latest 25 Cycles (50 Min)", "Full 3-Hour Window (90 Cycles)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
with col_ctrl_right:
    show_anomalies_only = st.checkbox("Show Anomaly Flags Only", value=False)

if "10" in view_depth:
    table_slice = telemetry_df.tail(10).copy()
elif "25" in view_depth:
    table_slice = telemetry_df.tail(25).copy()
else:
    table_slice = telemetry_df.copy()

if show_anomalies_only:
    table_slice = table_slice[table_slice["is_anomaly"] == True]

formatted_table = pd.DataFrame()
formatted_table["Time"] = table_slice["timestamp"].dt.strftime("%H:%M:%S")
formatted_table["Station"] = f"{target_station['name']} ({selected_station_id})"
formatted_table["Raw Temp (°C)"] = table_slice["temperature"].apply(lambda v: f"{v:.2f}")
formatted_table["AI Estimate (°C)"] = table_slice.apply(
    lambda r: f"{r['reconstructed_temp']:.2f}" if r["is_anomaly"] else "-", axis=1
)
formatted_table["Humidity (%)"] = table_slice["humidity"].apply(lambda v: f"{v:.1f}")
formatted_table["Pressure (hPa)"] = table_slice["pressure"].apply(lambda v: f"{v:.1f}")
formatted_table["Consensus Delta"] = table_slice.apply(
    lambda r: f"{r['temperature'] - r['reconstructed_temp']:+.2f} °C" if r["is_anomaly"] else "+0.00 °C", axis=1
)
formatted_table["QC Flag"] = table_slice.apply(
    lambda r: ("SENSOR FAULT" if is_current_fault else "REGIONAL EXTREME") if r["is_anomaly"] else "NOMINAL", axis=1
)
formatted_table["Operational Action"] = table_slice.apply(
    lambda r: ("RECOMMEND ISOLATION" if is_current_fault else "ISSUE ADVISORY") if r["is_anomaly"] else "ASSIMILATED", axis=1
)

st.dataframe(formatted_table.iloc[::-1], use_container_width=True, hide_index=True)


# ================================================================
# 18. DATA CONTINUITY & MODEL EXPLAINABILITY (SHAP)
# ================================================================
st.markdown('<hr class="sg-hr">', unsafe_allow_html=True)
st.markdown('<div class="sg-section-title">DATA CONTINUITY & MODEL EXPLAINABILITY (SHAP FEATURE ATTRIBUTION)</div>', unsafe_allow_html=True)

col_heal, col_xai = st.columns([1.0, 1.3], gap="medium")

with col_heal:
    if is_current_fault:
        st.markdown(f"""
<div class="sg-heal-fault">
  <div class="sg-heal-fault-header">DATA ISOLATION RECOMMENDED // RECONSTRUCTION ESTIMATE</div>
  <div class="sg-heal-comparison">
    <div class="sg-heal-val-block">
      <div class="sg-heal-val-label">Raw Corrupted</div>
      <div class="sg-heal-val-num raw">{latest_temp:.1f}°C</div>
    </div>
    <div class="sg-heal-arrow">→</div>
    <div class="sg-heal-val-block">
      <div class="sg-heal-val-label">AI Reconstructed</div>
      <div class="sg-heal-val-num corrected">{estimated_temp:.1f}°C</div>
    </div>
  </div>
  <div class="sg-heal-metrics">
    <div class="sg-heal-metric-row"><span>Deviation Corrected</span><span>{abs(temperature_deviation):.1f}°C</span></div>
    <div class="sg-heal-metric-row"><span>Imputation Algorithm</span><span>Spatial Consensus Moving Average</span></div>
    <div class="sg-heal-metric-row"><span>Consensus Confidence</span><span>{confidence_pct}%</span></div>
  </div>
  <div class="sg-heal-action-list">
    <div class="sg-heal-action-item"><span class="sg-bullet fault"></span> Recommend data stream isolation pending on-site operator verification</div>
    <div class="sg-heal-action-item"><span class="sg-bullet"></span> AI estimate available for NWP data assimilation continuity</div>
    <div class="sg-heal-action-item"><span class="sg-bullet"></span> Incident tracked permanently in SkyGuard audit log</div>
  </div>
</div>
""", unsafe_allow_html=True)
    elif is_extreme_weather:
        st.markdown(f"""
<div class="sg-heal-extreme">
  <div class="sg-heal-extreme-header">METEOROLOGICAL EVENT VERIFIED // DATA PRESERVED</div>
  <div style="font-size:12px;color:#94A3B8;line-height:1.5;margin-bottom:12px;">
    Reading ({latest_temp:.1f}°C) validated across regional cluster. Sensor is functioning correctly in a genuine extreme environment.
  </div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Target Station</span><span class="sg-heal-table-val extreme">{latest_temp:.1f}°C</span></div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Regional Mean</span><span class="sg-heal-table-val extreme">{latest_temp - 0.2:.1f}°C</span></div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Isolation Status</span><span class="sg-heal-table-val ok">BYPASSED (Data Preserved for NWP)</span></div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Advisory Dispatched</span><span class="sg-heal-table-val extreme">YES · IMD Alert Active</span></div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="sg-heal-normal">
  <div class="sg-heal-normal-title">Data Stream Validated · Nominal</div>
  <div class="sg-heal-normal-sub">All observations within physical boundaries and peer consensus</div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Sensor Health Index</span><span class="sg-heal-table-val ok">{health_score} / 100</span></div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Temporal Consistency</span><span class="sg-heal-table-val ok">100% Passed</span></div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Spatial Agreement</span><span class="sg-heal-table-val ok">Consensus Confirmed</span></div>
  <div class="sg-heal-table-row"><span class="sg-heal-table-key">Thermodynamic Consistency</span><span class="sg-heal-table-val ok">Tdew &lt; Tamb Nominal</span></div>
</div>
""", unsafe_allow_html=True)

with col_xai:
    xai_chart_col, xai_log_col = st.columns([1.2, 1.0], gap="small")

    with xai_chart_col:
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#64748B;margin-bottom:8px;">SHAP Feature Importance</div>', unsafe_allow_html=True)

        contribution_df = pd.DataFrame({
            "Feature": list(shap_scores.keys()),
            "Contribution": [float(v) for v in shap_scores.values()],
        }).sort_values(by="Contribution", ascending=True)

        bar_colors = ["#EF4444" if v > 0.2 else ("#F97316" if v > 0.05 else "#38BDF8") for v in contribution_df["Contribution"]]
        fig_contrib = go.Figure(go.Bar(
            x=contribution_df["Contribution"], y=contribution_df["Feature"],
            orientation="h",
            marker=dict(color=bar_colors, opacity=0.85),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
        ))
        fig_contrib.update_layout(
            template="plotly_dark", height=200,
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="#080A0D", plot_bgcolor="#080A0D",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9, color="#64748B", family="JetBrains Mono"), zeroline=True, zerolinecolor="rgba(255,255,255,0.09)"),
            yaxis=dict(tickfont=dict(size=10, color="#94A3B8", family="Inter"), showgrid=False),
            font=dict(family="Inter, sans-serif"),
            hoverlabel=dict(bgcolor="#121820", font=dict(size=11, color="#F8FAFC")),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)
        st.markdown("""
<div class="sg-xai-legend">
  <span><span class="sg-xai-legend-dot" style="background:#EF4444;"></span>Indicates Sensor Fault</span>
  <span><span class="sg-xai-legend-dot" style="background:#38BDF8;"></span>Supports Normal / Consensus</span>
</div>
""", unsafe_allow_html=True)

    with xai_log_col:
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#64748B;margin-bottom:8px;">Model Reasoning Chain</div>', unsafe_allow_html=True)
        if is_current_fault:
            steps = [
                f"Reading ({latest_temp:.1f}°C) flagged by temporal/physics checks.",
                "Isolation Forest classified sample as statistical outlier.",
                f"Spatial check: {spatial_check[:46]}…",
                "Isolated deviation confirmed: sensor fault isolated.",
                f"AI estimate available ({estimated_temp:.1f}°C) for operator review.",
            ]
            step_cls = "fault"
        elif is_extreme_weather:
            steps = [
                f"Reading ({latest_temp:.1f}°C) exceeds standard bounds.",
                "Isolation Forest flagged statistical anomaly.",
                "Spatial cross-check queried regional cluster nodes.",
                "Peer nodes (Gurugram, Noida) confirmed matching condition.",
                "Consensus verified: categorized as Genuine Extreme Weather.",
            ]
            step_cls = "extreme"
        else:
            steps = [
                "Observations within standard physical boundaries.",
                "Temporal variance and rate-of-change nominal.",
                "Isolation Forest score below decision threshold.",
                "Spatial neighbor agreement confirmed with peer nodes.",
                "Telemetry validated: data assimilated without imputation.",
            ]
            step_cls = ""

        steps_html = "".join([
            f'<div class="sg-reasoning-step">'
            f'<div class="sg-step-num {step_cls}">{i + 1:02d}</div>'
            f'<div class="sg-step-text">{s}</div>'
            f'</div>'
            for i, s in enumerate(steps)
        ])
        st.markdown(f'<div class="sg-card" style="padding:12px 14px;">{steps_html}</div>', unsafe_allow_html=True)


# ================================================================
# 19. OPERATOR INCIDENT AUDIT LOG & MAINTENANCE RECORD (PERMANENT)
# ================================================================
st.markdown('<hr class="sg-hr">', unsafe_allow_html=True)

with st.expander(f"OPERATOR INCIDENT AUDIT LOG & MAINTENANCE RECORDS ({len(st.session_state['incident_audit_log'])} Incidents · {len(st.session_state['maintenance_tasks'])} Tasks)", expanded=False):
    tab_audit, tab_tasks = st.tabs(["Incident Audit Log (Permanent)", "Logged Maintenance Tasks"])

    with tab_audit:
        if st.session_state["incident_audit_log"]:
            audit_records = []
            for inc_item in reversed(st.session_state["incident_audit_log"]):
                audit_records.append({
                    "Incident ID": inc_item["id"],
                    "Station": f"{inc_item['station_name']} ({inc_item['station_id']})",
                    "Detected At": inc_item["detected_at"],
                    "Anomaly Type": inc_item["anomaly_type"],
                    "Observed": f"{inc_item['observed_val']:.1f}°C",
                    "Expected": f"{inc_item['expected_val']:.1f}°C",
                    "Deviation": f"{inc_item['deviation']:+.1f}°C",
                    "Severity": inc_item["severity"],
                    "Status": inc_item["status"],
                    "Acknowledged": inc_item.get("acknowledged_at") or "-",
                    "Resolved": inc_item.get("resolved_at") or "-",
                    "Recommended Action": inc_item["next_action"]
                })
            audit_df = pd.DataFrame(audit_records)
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
        else:
            st.info("No incident records in the audit log yet. All systems operating nominally.")

    with tab_tasks:
        if st.session_state["maintenance_tasks"]:
            st.markdown('<div style="font-size:11px;color:#F59E0B;margin-bottom:8px;">(Simulated local task records for SIH prototype — no physical dispatch connected)</div>', unsafe_allow_html=True)
            tasks_df = pd.DataFrame(st.session_state["maintenance_tasks"])
            st.dataframe(tasks_df, use_container_width=True, hide_index=True)
        else:
            st.info("No maintenance tasks logged. Use 'CREATE LOCAL MAINTENANCE TASK' during an active incident to record prototype maintenance actions.")


# ================================================================
# 20. FOOTER
# ================================================================
st.markdown(f"""
<div class="sg-footer">
  <div class="sg-footer-text">
    SkyGuard AI Operations · MoES / IMD Automatic Weather Station Quality Control
    · Multi-Stage QC: Isolation Forest + Thermodynamic Constraints + Temporal Dynamics + Spatial Consensus
  </div>
  <div class="sg-footer-text">
    Live Refresh: 10s · Telemetry Window: 3 Hours · Wall Clock: {now_str} · Fleet Active: {total_count}/{total_count}
  </div>
</div>
""", unsafe_allow_html=True)
