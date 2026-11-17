import os

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tempfile import NamedTemporaryFile
from datetime import datetime
import base64, traceback, html
from io import BytesIO
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepTrace · Deepfake detection for video and photos",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  DESIGN SYSTEM  (detection lives in engine.py)
# ─────────────────────────────────────────────────────────────────────────────
INK, INK2, INK3 = "#0d0f0e", "#141816", "#1b201d"
IVORY, MUTE, LINE = "#f0efe7", "#9a9d95", "rgba(240,239,231,.10)"
LIME, RED, AMBER = "#c9f36a", "#ff5a4e", "#f2b84b"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');
:root{{--ink:{INK};--ink2:{INK2};--ink3:{INK3};--ivory:{IVORY};--mute:{MUTE};--line:{LINE};
  --lime:{LIME};--red:{RED};--amber:{AMBER};
  --serif:'Instrument Serif',Georgia,serif;--sans:'Inter',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace;}}
html,body,.stApp,[class*="css"]{{font-family:var(--sans);color:var(--ivory);}}
.stApp{{background:radial-gradient(1200px 500px at 85% -10%,rgba(201,243,106,.07),transparent 60%),var(--ink);}}
#MainMenu,footer,header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stDecoration"]{{display:none!important;}}
.block-container{{padding:1.4rem 2.2rem 3.5rem!important;max-width:1240px!important;}}
::selection{{background:var(--lime);color:var(--ink);}}
::-webkit-scrollbar{{width:6px;height:6px}}::-webkit-scrollbar-thumb{{background:#2a302c;border-radius:6px}}
p,li,label,span,div{{-webkit-font-smoothing:antialiased;}}
a{{color:var(--lime)!important;text-decoration:none!important;}}

/* top bar */
.dt-top{{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.2rem 0 1.1rem;border-bottom:1px solid var(--line);}}
.dt-logo{{display:flex;align-items:baseline;gap:.75rem;}}
.dt-logo b{{font-family:var(--serif);font-weight:400;font-size:1.75rem;letter-spacing:-.01em;line-height:1;}}
.dt-logo b i{{color:var(--lime);}}
.dt-tag,.dt-mono{{font-family:var(--mono);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);}}
.dt-right{{display:flex;align-items:center;gap:.6rem;}}
.dt-pill{{display:inline-flex;align-items:center;gap:.45rem;padding:.38rem .75rem;border:1px solid var(--line);border-radius:99px;font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ivory);}}
.dt-dot{{width:7px;height:7px;border-radius:50%;background:var(--lime);box-shadow:0 0 10px var(--lime);animation:dtp 2.4s infinite;}}
@keyframes dtp{{50%{{opacity:.35}}}}

/* hero */
.dt-hero{{display:grid;grid-template-columns:1.35fr 1fr;gap:2.5rem;align-items:end;padding:2.6rem 0 2rem;}}
.dt-kick{{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;color:var(--lime);margin-bottom:1rem;}}
.dt-h1{{font-family:var(--serif);font-weight:400;font-size:clamp(2.8rem,5.2vw,4.6rem);line-height:.95;letter-spacing:-.02em;margin:0;}}
.dt-h1 em{{color:var(--lime);}}
h1.dt-h1,.stMarkdown h1.dt-h1,.dt-rhead h2,.stMarkdown .dt-rhead h2{{font-family:var(--serif)!important;font-weight:400!important;padding:0!important;color:var(--ivory)!important;}}
.stMarkdown h1.dt-h1{{font-size:clamp(2.8rem,5.2vw,4.6rem)!important;line-height:.95!important;letter-spacing:-.02em!important;}}
.stMarkdown .dt-rhead h2{{font-size:2.4rem!important;line-height:1!important;}}
[data-testid='stHeaderActionElements'],.stMarkdown a.anchor-link,[data-testid='StyledLinkIconContainer'] a{{display:none!important;}}
.dt-sub,.stMarkdown p.dt-sub{{color:var(--mute)!important;font-size:1rem!important;line-height:1.6!important;max-width:44ch;margin:1.3rem 0 0!important;}}
.dt-small{{margin-top:.9rem;color:var(--mute);font-size:.84rem;line-height:1.55;}}
[data-testid='stVerticalBlockBorderWrapper'],div[data-testid='stVerticalBlock'][class*='border'],[data-testid='stVerticalBlock'] > div > [data-testid='stVerticalBlockBorderWrapper']{{background:var(--ink2)!important;border:1px solid var(--line)!important;border-radius:18px!important;}}
.dt-steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;}}
.dt-step{{padding:.9rem 1rem;border:1px solid var(--line);border-radius:14px;background:rgba(240,239,231,.02);}}
.dt-step .n{{font-family:var(--mono);font-size:.64rem;letter-spacing:.12em;color:var(--mute);}}
.dt-step .t{{font-weight:600;font-size:.92rem;margin-top:.35rem;}}
.dt-step.on{{border-color:rgba(201,243,106,.55);background:rgba(201,243,106,.06);}}
.dt-step.on .n{{color:var(--lime);}}
.dt-step.done .n::after{{content:"  ✓";color:var(--lime);}}
.dt-trust{{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;margin-top:1rem;font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mute);}}
.dt-trust span::before{{content:"— ";color:var(--lime);}}

/* tabs */
[data-baseweb="tab-list"]{{gap:0!important;border-bottom:1px solid var(--line)!important;background:transparent!important;}}
[data-baseweb="tab"]{{height:2.8rem!important;padding:0 1.3rem!important;background:transparent!important;color:var(--mute)!important;font-family:var(--mono)!important;font-size:.72rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;}}
[data-baseweb="tab"] p{{font-family:var(--mono)!important;font-size:.72rem!important;}}
[aria-selected="true"][data-baseweb="tab"]{{color:var(--ivory)!important;}}
[data-baseweb="tab-highlight"]{{background:var(--lime)!important;height:2px!important;}}
[data-baseweb="tab-border"]{{display:none!important;}}
[data-baseweb="tab-panel"]{{padding-top:1.6rem!important;}}

/* cards */
.dt-card{{background:var(--ink2);border:1px solid var(--line);border-radius:18px;padding:1.3rem 1.4rem;margin-bottom:1rem;}}
.dt-lbl{{font-family:var(--mono);font-size:.64rem;letter-spacing:.14em;text-transform:uppercase;color:var(--mute);margin-bottom:.9rem;display:flex;justify-content:space-between;gap:1rem;}}
.dt-lbl b{{color:var(--ivory);font-weight:500;}}
.dt-kv{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;}}
.dt-kv > div{{background:var(--ink2);padding:.8rem .9rem;}}
.dt-kv span{{display:block;font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);}}
.dt-kv b{{display:block;font-weight:500;font-size:.98rem;margin-top:.3rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.dt-plan{{display:flex;flex-direction:column;}}
.dt-plan div{{display:flex;justify-content:space-between;padding:.62rem 0;border-top:1px solid var(--line);font-size:.9rem;}}
.dt-plan div:first-child{{border-top:0;}}
.dt-plan span{{color:var(--mute);}}
.dt-list{{margin:0;padding:0;list-style:none;}}
.dt-list li{{display:flex;gap:.8rem;padding:.65rem 0;border-top:1px solid var(--line);font-size:.92rem;color:#d6d5cd;line-height:1.5;}}
.dt-list li:first-child{{border-top:0;}}
.dt-list li i{{font-style:normal;font-family:var(--mono);font-size:.7rem;color:var(--lime);padding-top:.15rem;min-width:1.4rem;}}
.dt-empty{{min-height:220px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:.4rem;}}
.dt-empty .big{{font-family:var(--serif);font-size:1.8rem;}}
.dt-empty .big em{{color:var(--lime);}}
.dt-empty p{{color:var(--mute);font-size:.9rem;margin:0;max-width:34ch;}}

/* uploader */
[data-testid="stFileUploader"] section,[data-testid="stFileUploaderDropzone"]{{background:rgba(240,239,231,.025)!important;border:1.5px dashed rgba(240,239,231,.22)!important;border-radius:16px!important;padding:1.6rem!important;transition:border-color .2s,background .2s;}}
[data-testid="stFileUploaderDropzone"]:hover{{border-color:var(--lime)!important;background:rgba(201,243,106,.04)!important;}}
[data-testid="stFileUploaderDropzone"] button,[data-testid="stBaseButton-secondary"]{{background:transparent!important;color:var(--ivory)!important;border:1px solid rgba(240,239,231,.3)!important;border-radius:99px!important;font-family:var(--mono)!important;font-size:.72rem!important;letter-spacing:.08em;}}
[data-testid="stFileUploaderDropzoneInstructions"] span,[data-testid="stFileUploaderDropzoneInstructions"] small{{color:var(--mute)!important;}}
[data-testid="stVideo"],video{{border-radius:14px!important;overflow:hidden;}}

/* buttons */
.stButton>button,[data-testid="stBaseButton-primary"]{{width:100%;height:3.2rem;background:var(--lime)!important;color:var(--ink)!important;border:0!important;border-radius:99px!important;font-family:var(--mono)!important;font-weight:500!important;font-size:.8rem!important;letter-spacing:.12em;text-transform:uppercase;box-shadow:0 10px 30px -12px rgba(201,243,106,.6)!important;transition:transform .2s,box-shadow .2s!important;}}
.stButton>button p{{font-family:var(--mono)!important;font-size:.8rem!important;}}
.stButton>button:hover{{transform:translateY(-1px);box-shadow:0 14px 34px -12px rgba(201,243,106,.8)!important;}}
[data-testid="stDownloadButton"]>button{{width:100%;background:transparent!important;color:var(--ivory)!important;border:1px solid rgba(240,239,231,.25)!important;border-radius:99px!important;height:2.9rem;font-family:var(--mono)!important;font-size:.72rem!important;letter-spacing:.1em;text-transform:uppercase;}}
[data-testid="stDownloadButton"]>button:hover{{border-color:var(--lime)!important;color:var(--lime)!important;}}

/* expander + sliders */
[data-testid="stExpander"]{{background:var(--ink2)!important;border:1px solid var(--line)!important;border-radius:16px!important;}}
[data-testid="stExpander"] summary{{font-family:var(--mono);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--ivory);}}
[data-testid="stExpander"] summary p{{font-family:var(--mono)!important;font-size:.72rem!important;}}
[data-testid="stSlider"] label p{{font-size:.86rem!important;color:var(--ivory)!important;}}
[data-baseweb="slider"] [role="slider"]{{background:var(--lime)!important;border:0!important;box-shadow:0 0 0 4px rgba(201,243,106,.18)!important;}}
[data-testid="stSliderThumbValue"]{{color:var(--lime)!important;font-family:var(--mono)!important;}}
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{{color:var(--mute)!important;font-family:var(--mono)!important;font-size:.7rem!important;}}
[data-testid="stSpinner"] p{{color:var(--ivory)!important;}}
[data-testid="stAlert"]{{border-radius:14px!important;}}

/* verdict */
.dt-verdict{{display:grid;grid-template-columns:auto 1fr;gap:1.8rem;align-items:center;padding:1.6rem 1.8rem;border-radius:22px;border:1px solid var(--line);margin-top:1.4rem;}}
.dt-verdict.fake{{background:linear-gradient(120deg,rgba(255,90,78,.12),rgba(255,90,78,.02) 60%);border-color:rgba(255,90,78,.35);}}
.dt-verdict.real{{background:linear-gradient(120deg,rgba(201,243,106,.11),rgba(201,243,106,.02) 60%);border-color:rgba(201,243,106,.32);}}
.dt-vlbl{{font-family:var(--mono);font-size:.66rem;letter-spacing:.14em;text-transform:uppercase;}}
.dt-vtitle{{font-family:var(--serif);font-size:clamp(2rem,3.4vw,2.9rem);line-height:1;margin:.35rem 0 .55rem;letter-spacing:-.01em;}}
.dt-vtext{{color:#cfcec6;font-size:.95rem;line-height:1.55;max-width:60ch;margin:0;}}
.dt-kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem;margin-top:1.1rem;}}
.dt-kpi{{background:var(--ink2);border:1px solid var(--line);border-radius:14px;padding:.85rem .95rem;}}
.dt-kpi span{{display:block;font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);}}
.dt-kpi b{{display:block;font-family:var(--serif);font-weight:400;font-size:1.7rem;margin-top:.25rem;line-height:1;}}
.dt-note{{display:flex;gap:.7rem;align-items:flex-start;padding:.9rem 1.1rem;border-radius:14px;background:rgba(242,184,75,.07);border:1px solid rgba(242,184,75,.25);color:#e9d9b4;font-size:.86rem;line-height:1.5;margin-top:1rem;}}
.dt-note b{{color:var(--amber);font-family:var(--mono);font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;white-space:nowrap;padding-top:.15rem;}}

/* report */
.dt-rhead{{display:flex;justify-content:space-between;align-items:flex-end;gap:1.5rem;flex-wrap:wrap;margin-bottom:1.2rem;}}
.dt-rhead h2{{font-family:var(--serif);font-weight:400;font-size:2.4rem;line-height:1;margin:0;letter-spacing:-.01em;}}
.dt-meta{{font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;color:var(--mute);margin-top:.6rem;}}
.dt-badge{{display:inline-flex;align-items:center;gap:.5rem;padding:.55rem 1rem;border-radius:99px;font-family:var(--mono);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;}}
.dt-badge.fake{{background:rgba(255,90,78,.14);color:var(--red);border:1px solid rgba(255,90,78,.4);}}
.dt-badge.real{{background:rgba(201,243,106,.12);color:var(--lime);border:1px solid rgba(201,243,106,.4);}}
.dt-thumbs{{display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem;}}
.dt-thumb{{border-radius:12px;overflow:hidden;border:1px solid var(--line);background:var(--ink3);}}
.dt-thumb img{{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;}}
.dt-thumb > div{{padding:.55rem .65rem .65rem;font-family:var(--mono);font-size:.64rem;letter-spacing:.06em;color:var(--mute);}}
.dt-thumb div b{{font-weight:500;}}
.dt-bar{{height:4px;border-radius:4px;background:rgba(240,239,231,.08);margin-top:.45rem;overflow:hidden;}}
.dt-bar i{{display:block;height:100%;border-radius:4px;}}
.dt-stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border-radius:12px;overflow:hidden;border:1px solid var(--line);}}
.dt-stats div{{background:var(--ink2);padding:.75rem .85rem;}}
.dt-stats span{{display:block;font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);}}
.dt-stats b{{display:block;font-family:var(--mono);font-weight:500;font-size:.95rem;margin-top:.3rem;}}
[data-testid="stDataFrame"]{{border-radius:12px;overflow:hidden;}}

/* method */
.dt-pipe{{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;}}
.dt-pipe > div{{background:var(--ink2);border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;}}
.dt-pipe i{{font-style:normal;font-family:var(--mono);font-size:.66rem;letter-spacing:.12em;color:var(--lime);}}
.dt-pipe b{{display:block;font-family:var(--serif);font-weight:400;font-size:1.45rem;margin:.4rem 0 .35rem;}}
.dt-pipe p{{color:var(--mute);font-size:.88rem;line-height:1.55;margin:0;}}
.dt-arch{{font-family:var(--mono);font-size:.78rem;line-height:2.15;color:#cfcec6;}}
.dt-arch em{{font-style:normal;color:var(--lime);}}
.dt-foot{{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-top:3rem;padding-top:1.2rem;border-top:1px solid var(--line);font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mute);}}

@media (max-width: 900px){{
  .block-container{{padding:1rem 1rem 3rem!important;}}
  .dt-hero{{grid-template-columns:1fr;gap:1.4rem;padding-top:1.8rem;}}
  .dt-kv{{grid-template-columns:repeat(2,1fr);}}
  .dt-kpis{{grid-template-columns:repeat(2,1fr);}}
  .dt-verdict{{grid-template-columns:1fr;}}
  .dt-thumbs{{grid-template-columns:repeat(2,1fr);}}
  .dt-pipe{{grid-template-columns:1fr;}}
  .dt-right .dt-pill:last-child{{display:none;}}
}}
</style>
""", unsafe_allow_html=True)


st.markdown(f"""
<style>
/* ── v5.1 additions ─────────────────────────────────────────────── */
.dt-verdict{{margin-top:0;grid-template-columns:auto 1fr auto;}}
.dt-facts{{display:flex;flex-direction:column;gap:.55rem;min-width:170px;padding-left:1.6rem;border-left:1px solid var(--line);}}
.dt-facts div{{display:flex;justify-content:space-between;gap:1.2rem;font-size:.85rem;}}
.dt-facts span{{font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);padding-top:.2rem;}}
.dt-verdict.warn{{background:linear-gradient(120deg,rgba(242,184,75,.12),rgba(242,184,75,.02) 60%);border-color:rgba(242,184,75,.35);}}
.dt-badge.warn{{background:rgba(242,184,75,.12);color:var(--amber);border:1px solid rgba(242,184,75,.4);}}
.dt-thumb img{{aspect-ratio:1/1;}}
.dt-bench{{width:100%;border-collapse:collapse;font-size:.84rem;}}
.dt-bench th{{font-family:var(--mono);font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);text-align:left;font-weight:400;padding:.5rem .6rem;border-bottom:1px solid var(--line);}}
.dt-bench td{{padding:.6rem;border-bottom:1px solid var(--line);}}
.dt-bench td.n{{font-family:var(--mono);}}
[data-testid="stProgress"] > div > div > div > div{{background:var(--lime)!important;}}
[data-testid="stProgress"] p{{font-family:var(--mono)!important;font-size:.72rem!important;letter-spacing:.08em;color:var(--mute)!important;}}
[data-testid="stTabs"] [role="tablist"]{{gap:0!important;border-bottom:1px solid var(--line)!important;background:transparent!important;}}
[data-testid="stTabs"] [role="tab"]{{height:2.8rem!important;padding:0 1.3rem!important;background:transparent!important;color:var(--mute)!important;font-family:var(--mono)!important;font-size:.72rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;}}
[data-testid="stTabs"] [role="tab"] p{{font-family:var(--mono)!important;font-size:.72rem!important;}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{color:var(--ivory)!important;}}
[data-testid="stImage"] img{{border-radius:14px;}}
[data-testid="stCheckbox"] label p,[data-testid="stToggle"] label p{{font-family:var(--mono)!important;font-size:.7rem!important;letter-spacing:.1em;text-transform:uppercase;color:var(--mute)!important;}}

/* big drop zone before a file is chosen */
.st-key-dt_drop [data-testid="stFileUploaderDropzone"]{{min-height:150px;flex-direction:column;justify-content:center;gap:1rem;}}

/* what-it-checks cards */
.dt-checks{{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;}}
.dt-check{{background:var(--ink2);border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.15rem;}}
.dt-check i{{font-style:normal;font-family:var(--mono);font-size:.64rem;letter-spacing:.12em;color:var(--lime);}}
.dt-check b{{display:block;font-family:var(--serif);font-weight:400;font-size:1.35rem;margin:.35rem 0 .3rem;}}
.dt-check p{{margin:0;color:var(--mute);font-size:.85rem;line-height:1.5;}}

/* case report header */
.dt-case{{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;margin:2.4rem 0 1.1rem;padding-top:1.6rem;border-top:1px solid var(--line);}}
.dt-case h2{{font-family:var(--serif)!important;font-weight:400!important;font-size:2.1rem!important;line-height:1.05!important;margin:0!important;padding:0!important;color:var(--ivory)!important;word-break:break-word;}}

/* signal cards with a verdict-band scale */
.dt-signals{{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-top:.8rem;}}
.dt-sig{{background:var(--ink2);border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem 1.05rem;}}
.dt-sig .h{{display:flex;justify-content:space-between;align-items:baseline;font-family:var(--mono);font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);}}
.dt-sig .v{{font-family:var(--serif);font-size:2.1rem;line-height:1;margin:.45rem 0 .7rem;}}
.dt-sig .c{{font-size:.82rem;color:var(--mute);line-height:1.45;margin-top:.65rem;}}
.dt-band{{position:relative;height:8px;border-radius:8px;display:flex;overflow:visible;}}
.dt-band i{{display:block;height:100%;}}
.dt-band i:first-child{{border-radius:8px 0 0 8px;}} .dt-band i:last-of-type{{border-radius:0 8px 8px 0;}}
.dt-band b{{position:absolute;top:-5px;width:4px;height:18px;margin-left:-2px;border-radius:3px;background:var(--ivory);box-shadow:0 0 0 3px var(--ink2);}}
.dt-band-l{{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.58rem;letter-spacing:.08em;color:var(--mute);margin-top:.45rem;}}

/* evidence */
.dt-ev-cap{{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mute);margin:.55rem 0 0;}}
.dt-heat{{display:grid;grid-template-columns:repeat(3,1fr);gap:.55rem;}}
.dt-heat .dt-thumb > div{{padding:.45rem .55rem .5rem;}}
.dt-legend{{display:flex;align-items:center;gap:.6rem;font-family:var(--mono);font-size:.6rem;letter-spacing:.08em;color:var(--mute);margin-top:.8rem;}}
.dt-legend span.g{{flex:0 0 110px;height:6px;border-radius:6px;background:linear-gradient(90deg,rgba(255,210,60,.15),#ffa03c,#ff5a3c);}}

/* provenance */
.dt-prov li b{{display:block;font-weight:600;font-size:.9rem;color:var(--ivory);}}
.dt-prov li span{{color:var(--mute);font-size:.84rem;}}
.dt-prov li i.ai{{color:var(--red);}} .dt-prov li i.edit{{color:var(--amber);}} .dt-prov li i.info{{color:var(--mute);}}

@media (max-width: 900px){{
  .dt-checks,.dt-signals{{grid-template-columns:1fr;}}
  .dt-heat{{grid-template-columns:repeat(3,1fr);}}
  .dt-case h2{{font-size:1.6rem!important;}}
  .dt-verdict{{padding:1.3rem;gap:1rem;justify-items:start;grid-template-columns:1fr;}}
}}
@media (max-width: 600px){{
  .dt-tag,.dt-steps,.dt-trust{{display:none!important;}}
  .dt-hero{{padding:1.4rem 0 1.2rem!important;}}
  .dt-pill{{white-space:nowrap;}}
  .dt-lbl span + span{{display:none;}}
  .dt-heat{{grid-template-columns:repeat(2,1fr);}}
  .dt-facts{{padding:1rem 0 0;border-left:0;border-top:1px solid var(--line);width:100%;}}
}}
</style>
""", unsafe_allow_html=True)

from engine import Engine, NoFaceError, T_REAL, T_FAKE, T_AIGEN
import provenance

VIDEO_EXT = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
IMAGE_EXT = ["jpg", "jpeg", "png", "webp", "bmp"]
VERSION = "v5.1"
SPACE_ID = "ruwalamihir/deepfake_detector"

# ─────────────────────────────────────────────────────────────────────────────
#  STATE + ENGINE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {"result": None, "err": None, "err_friendly": False, "max_frames": 16, "case": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner="Loading detection models (first visit only)…")
def load_engine():
    return Engine(threads=int(os.environ.get("DT_THREADS", "0")) or None)


def vid_meta(path: str):
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 25
    nf = c.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    return float(fps), float(nf) / float(fps), w, h, int(nf)


def b64_jpeg(rgb: np.ndarray, size: int | None = 220, max_w: int | None = None, q: int = 80) -> str:
    im = Image.fromarray(rgb)
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    elif max_w and im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def to_state(r, prov: dict, **extra) -> dict:
    """Flatten an engine Result into plain data, and fold in the provenance check."""
    verdict, notes = r.verdict, list(r.notes)
    prov_ai = prov["ai_declared"] and verdict != "fake"
    if prov_ai:
        verdict = "aigen"
        notes.insert(0, "The file's own metadata says it was made or edited with generative AI. "
                        "That label is rarely added by mistake, so it outweighs a low model score.")
    return {
        "kind": r.kind, "verdict": verdict, "fake_prob": r.fake_prob, "headline": r.headline,
        "swap_prob": r.swap_prob, "aigen_prob": r.aigen_prob, "scores": r.scores,
        "frame_idx": r.frame_idx, "n_sampled": r.n_sampled, "n_dropped": r.n_dropped,
        "n_faces": r.n_faces, "low_quality": r.low_quality, "brightness": r.brightness,
        "sharpness": r.sharpness, "notes": notes, "prov": prov, "prov_ai": prov_ai,
        "thumbs": [(lbl, sc, b64_jpeg(img)) for lbl, sc, img in r.thumbs],
        "heat": [(lbl, sc, b64_jpeg(img)) for lbl, sc, img in r.heat],
        "evidence": b64_jpeg(r.evidence, size=None, max_w=1100) if r.evidence is not None else None,
        "evidence_label": r.evidence_label,
        "people": [{"prob": f["prob"], "n": f["n"], "b64": b64_jpeg(f["thumb"], 160)} for f in r.faces],
        "stamp": datetime.now().strftime("%d %b %Y · %H:%M"),
        **extra,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PRESENTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
VERDICTS = {
    "real": dict(color=LIME, cls="real", title="No manipulation found", badge="✓ No manipulation found"),
    "fake": dict(color=RED, cls="fake", title="Likely manipulated", badge="⚠ Likely manipulated"),
    "aigen": dict(color=RED, cls="fake", title="Likely AI-generated", badge="⚠ Likely AI-generated"),
    "inconclusive": dict(color=AMBER, cls="warn", title="Inconclusive", badge="? Inconclusive"),
}


def H(s):
    """Render an HTML fragment (single line, so Markdown never turns it into a code block)."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def fmt_size(b):
    return f"{b/1e6:.1f} MB" if b >= 1e6 else f"{b/1e3:.0f} KB"


def score_color(s: float, lo=T_REAL, hi=T_FAKE) -> str:
    return RED if s >= hi else (LIME if s <= lo else AMBER)


def band(p: float, lo: float, hi: float) -> str:
    """A 0-1 scale split into clean / uncertain / flagged zones, with a marker at p."""
    return (f'<div class="dt-band"><i style="width:{lo*100:.0f}%;background:rgba(201,243,106,.35)"></i>'
            f'<i style="width:{(hi-lo)*100:.0f}%;background:rgba(242,184,75,.32)"></i>'
            f'<i style="width:{(1-hi)*100:.0f}%;background:rgba(255,90,78,.38)"></i>'
            f'<b style="left:{min(max(p, 0.005), 0.995)*100:.1f}%"></b></div>'
            f'<div class="dt-band-l"><span>0</span><span>{lo:.2f}</span><span>{hi:.2f}</span><span>1</span></div>')


def ring(p, color, size=148, label="MANIPULATION", center=None):
    r, c = 54, 2 * np.pi * 54
    center = center if center is not None else f"{p*100:.0f}%"
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 140 140" role="img" aria-label="{label} {center}">'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="rgba(240,239,231,.08)" stroke-width="10"/>'
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" '
            f'stroke-dasharray="{p*c:.1f} {c:.1f}" transform="rotate(-90 70 70)"/>'
            f'<text x="70" y="74" text-anchor="middle" fill="{IVORY}" font-family="Instrument Serif, Georgia, serif" font-size="36">{center}</text>'
            f'<text x="70" y="96" text-anchor="middle" fill="{MUTE}" font-family="IBM Plex Mono, monospace" font-size="8.5" letter-spacing="1.2">{label}</text>'
            f'</svg>')


def ring_for(res, size=148):
    color = VERDICTS[res["verdict"]]["color"]
    if res["prov_ai"] and res["aigen_prob"] < T_AIGEN:
        return ring(1.0, color, size, "DECLARED", "AI")
    label = "AI-GENERATED" if res["headline"] == "aigen" else "MANIPULATION"
    return ring(res["fake_prob"], color, size, label)


def verdict_text(res) -> str:
    v, sw, ag = res["verdict"], res["swap_prob"], res["aigen_prob"]
    n = len(res["scores"])
    unit = "frames" if res["kind"] == "video" else ("face" if n == 1 else "faces")
    if v == "aigen" and res["prov_ai"] and ag < T_AIGEN:
        return ("The file's own metadata declares it AI-generated or AI-edited. The detectors scored it "
                f"<b>{sw*100:.0f}%</b> for face manipulation and <b>{ag*100:.0f}%</b> for AI generation, "
                "so they did not spot it from the pixels alone.")
    if v == "real":
        return (f"Across {n} {unit} the face scored <b>{sw*100:.0f}%</b> for manipulation and the image "
                f"<b>{ag*100:.0f}%</b> for AI generation. Neither detector found a signal. "
                "That is not a guarantee: a careful, modern fake can still pass.")
    if v == "fake":
        return (f"Across {n} {unit} the face scored <b>{sw*100:.0f}%</b> for manipulation. This pattern is typical "
                "of face swaps and re-enactment. The heatmaps below show where the evidence sits.")
    if v == "aigen":
        return (f"The AI-generation detector scored this <b>{ag*100:.0f}%</b>. The image content looks synthesised "
                "rather than captured by a camera.")
    if res["low_quality"]:
        return ("The face is too dark, blurry or compressed for a fair call. To a detector those flaws look "
                "like forgery artefacts, so DeepTrace gives no verdict instead of guessing.")
    return (f"The manipulation score is <b>{sw*100:.0f}%</b>, between the clean band (up to {T_REAL*100:.0f}%) "
            f"and the flagged band (from {T_FAKE*100:.0f}%). DeepTrace can't call this one either way.")


def band_word(p, lo, hi):
    return "flagged" if p >= hi else ("clean" if p <= lo else "uncertain")


def signals_html(res) -> str:
    sw, ag, prov = res["swap_prob"], res["aigen_prob"], res["prov"]
    n = len(res["scores"])
    unit = "frames" if res["kind"] == "video" else ("face" if n == 1 else "faces")
    sw_cap = ("Too low quality to judge" if res["low_quality"]
              else f"{band_word(sw, T_REAL, T_FAKE).capitalize()} · mean of {len(res['scores'])} {unit}")
    sw_col = AMBER if res["low_quality"] else score_color(sw)
    ai_items = [f for f in prov["findings"] if f[0] == "ai"]
    if ai_items:
        p_word, p_col, p_cap = "AI declared", RED, ai_items[0][1]
    elif any(f[1] == "Content Credentials present" for f in prov["findings"]):
        p_word, p_col, p_cap = "Credentials", IVORY, "C2PA manifest found (not verified)"
    elif any(f[0] == "edit" for f in prov["findings"]):
        p_word, p_col, p_cap = "Edited", AMBER, next(f[1] for f in prov["findings"] if f[0] == "edit")
    elif prov["camera"]:
        p_word, p_col, p_cap = "Camera tag", IVORY, esc(prov["camera"])
    else:
        p_word, p_col, p_cap = "No markers", MUTE, "Normal: most apps strip metadata"
    return f"""
<div class="dt-signals">
  <div class="dt-sig"><div class="h"><span>Face manipulation</span><span>swap · re-enactment</span></div>
    <div class="v" style="color:{sw_col}">{sw*100:.1f}%</div>{band(sw, T_REAL, T_FAKE)}<div class="c">{sw_cap}</div></div>
  <div class="dt-sig"><div class="h"><span>AI generation</span><span>diffusion · GAN</span></div>
    <div class="v" style="color:{score_color(ag, 0.5, T_AIGEN)}">{ag*100:.1f}%</div>{band(ag, 0.5, T_AIGEN)}
    <div class="c">{band_word(ag, 0.5, T_AIGEN).capitalize()} · whole image and face</div></div>
  <div class="dt-sig"><div class="h"><span>Provenance</span><span>file metadata</span></div>
    <div class="v" style="color:{p_col}">{p_word}</div>
    <div class="c" style="margin-top:0">{p_cap}</div></div>
</div>"""


_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=MUTE, size=12),
    margin=dict(l=8, r=8, t=12, b=8),
    xaxis=dict(gridcolor="rgba(240,239,231,.06)", showgrid=True, zeroline=False, linecolor="rgba(240,239,231,.12)"),
    yaxis=dict(gridcolor="rgba(240,239,231,.06)", showgrid=True, zeroline=False),
    hoverlabel=dict(bgcolor=INK3, bordercolor="rgba(240,239,231,.2)", font=dict(family="IBM Plex Mono, monospace", color=IVORY, size=12)),
)


def chart_timeline(scores, frame_idx, fps):
    xs = [i / fps for i in frame_idx]
    fig = go.Figure()
    fig.add_hrect(y0=T_FAKE, y1=1.02, fillcolor="rgba(255,90,78,.07)", line_width=0)
    fig.add_hrect(y0=0, y1=T_REAL, fillcolor="rgba(201,243,106,.05)", line_width=0)
    fig.add_trace(go.Scatter(x=xs, y=scores, mode="lines", line=dict(color=IVORY, width=1.6),
                             hovertemplate="%{x:.1f}s · score %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=scores, mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(color=[score_color(s) for s in scores], size=9, line=dict(width=2, color=INK2))))
    for y, lbl, col in ((T_FAKE, f"flagged ≥ {T_FAKE:.2f}", RED), (T_REAL, f"clean ≤ {T_REAL:.2f}", LIME)):
        fig.add_hline(y=y, line=dict(color=col, width=1, dash="dot"), annotation_text=lbl, annotation_position="top left",
                      annotation_font=dict(color=col, family="IBM Plex Mono, monospace", size=10))
    fig.update_layout(**_BG, yaxis_range=[0, 1.02], showlegend=False, height=260,
                      xaxis_title=dict(text="seconds into the clip", font=dict(size=11)))
    return fig


def build_html_report(res) -> str:
    """A self-contained, printable evidence report (images embedded)."""
    V = VERDICTS[res["verdict"]]
    col = {"real": "#3f7d14", "fake": "#c0362c", "aigen": "#c0362c", "inconclusive": "#a8680a"}[res["verdict"]]
    is_video = res["kind"] == "video"
    xs = ([f"{i / res['fps']:.1f}s" for i in res["frame_idx"]] if is_video else [f"face {i + 1}" for i in res["frame_idx"]])
    rows = "".join(f"<tr><td>{x}</td><td>{s:.3f}</td><td>{band_word(s, T_REAL, T_FAKE)}</td></tr>"
                   for x, s in zip(xs, res["scores"]))
    heat = "".join(f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>{esc(l)} · {s*100:.0f}%</figcaption></figure>'
                   for l, s, b in res["heat"][:6])
    prov = "".join(f"<li><b>{esc(t)}</b> {esc(d)}</li>" for _, t, d in res["prov"]["findings"])
    notes = "".join(f"<li>{esc(n)}</li>" for n in res["notes"]) or "<li>None.</li>"
    ev = (f'<img class="ev" src="data:image/jpeg;base64,{res["evidence"]}"><p class="cap">{esc(res["evidence_label"])}</p>'
          if res["evidence"] else "")
    media = (f"{res['duration']:.1f}s · {res['fps']:.0f} fps · {res['resolution']}" if is_video
             else f"photo · {res['resolution']}")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepTrace report · {esc(res['filename'])}</title>
<style>
body{{font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;color:#1a1c1b;background:#fff;max-width:880px;margin:0 auto;padding:40px 24px 60px}}
h1{{font:400 34px/1.1 Georgia,serif;margin:6px 0 4px;word-break:break-word}} h2{{font:600 13px/1 ui-monospace,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;color:#666;margin:34px 0 12px}}
.k{{font:12px ui-monospace,Menlo,monospace;letter-spacing:.1em;text-transform:uppercase;color:#777}}
.v{{border-left:5px solid {col};background:#f6f6f3;padding:18px 22px;border-radius:8px;margin-top:22px}}
.v b.t{{display:block;font:400 30px/1.1 Georgia,serif;color:{col};margin-bottom:6px}}
.g{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .g div{{background:#f6f6f3;border-radius:8px;padding:12px 14px}}
.g span{{display:block;font:11px ui-monospace,Menlo,monospace;letter-spacing:.1em;text-transform:uppercase;color:#777}} .g b{{font:400 26px Georgia,serif}}
img.ev{{width:100%;border-radius:8px}} .cap{{font-size:12px;color:#777;margin:6px 0 0}}
.h{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} figure{{margin:0}} figure img{{width:100%;border-radius:6px}} figcaption{{font:12px ui-monospace,Menlo,monospace;color:#555;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font:13px ui-monospace,Menlo,monospace}} td,th{{text-align:left;padding:5px 8px;border-bottom:1px solid #e5e5e0}}
ul{{padding-left:18px}} .foot{{margin-top:40px;font-size:12px;color:#777;border-top:1px solid #e5e5e0;padding-top:14px}}
@media print{{body{{padding:0}}}} @media(max-width:600px){{.g,.h{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<div class="k">DeepTrace {VERSION} · evidence report · {esc(res['stamp'])}</div>
<h1>{esc(res['filename'])}</h1><div class="k">{esc(media)}</div>
<div class="v"><b class="t">{V['title']}</b>{verdict_text(res)}</div>
<h2>Signals</h2>
<div class="g"><div><span>Face manipulation</span><b>{res['swap_prob']*100:.1f}%</b></div>
<div><span>AI generation</span><b>{res['aigen_prob']*100:.1f}%</b></div>
<div><span>People analysed</span><b>{res['n_faces']}</b></div></div>
<p class="cap">Face manipulation: up to {T_REAL:.2f} clean, from {T_FAKE:.2f} flagged, between is inconclusive. AI generation flagged from {T_AIGEN:.2f}.</p>
<h2>Evidence</h2>{ev}
<h2>Where the detector found evidence</h2><div class="h">{heat}</div>
<p class="cap">Red marks the regions that pushed the face towards "manipulated". The colour scale is fixed, so a clean face shows little or no heat.</p>
<h2>Notes</h2><ul>{notes}</ul>
<h2>Provenance (file metadata)</h2><ul>{prov}</ul>
<h2>Every score</h2><table><tr><th>{'Time' if is_video else 'Face'}</th><th>Manipulation</th><th>Band</th></tr>{rows}</table>
<div class="foot">DeepTrace is a screening tool. Scores are estimates from two models (DFDC-winner EfficientNet-B7 for face manipulation,
Community Forensics ViT for AI generation), not proof. On the published benchmark it wrongly flagged about 2% of real videos and
let about 7% of fake videos pass as clean. Check the source and other evidence before acting on a result, and never use it alone to accuse someone.</div>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  TOP BAR + HERO
# ─────────────────────────────────────────────────────────────────────────────
H(f"""
<div class="dt-top">
  <div class="dt-logo"><b>Deep<i>Trace</i></b><span class="dt-tag">Face forensics</span></div>
  <div class="dt-right"><span class="dt-pill"><span class="dt-dot"></span>3 checks ready</span><span class="dt-pill">{VERSION}</span></div>
</div>
""")

uploaded_key = "dt_uploader"
hero_ph = st.empty()


def render_hero():
    up = st.session_state.get(uploaded_key)
    res = st.session_state.result
    has_res = res is not None and up is not None and st.session_state.case == f"{up.name}:{up.size}"
    s1 = "done" if up is not None else "on"
    s2 = "done" if has_res else ("on" if up is not None else "")
    s3 = "on" if has_res else ""
    hero_ph.markdown(" ".join(line.strip() for line in f"""
<div class="dt-hero">
  <div>
    <div class="dt-kick">Deepfake detection for video and photos</div>
    <h1 class="dt-h1">Is this face <em>real?</em></h1>
    <p class="dt-sub">Upload a clip or a photo. DeepTrace checks every face for swaps and re-enactment, checks the image for AI generation, reads the file's own metadata, and shows you the evidence. When it can't tell, it says so.</p>
    <div class="dt-trust"><span>Three independent checks</span><span>Visual evidence</span><span>Measured error rates</span></div>
  </div>
  <div class="dt-steps">
    <div class="dt-step {s1}"><div class="n">01</div><div class="t">Upload</div></div>
    <div class="dt-step {s2}"><div class="n">02</div><div class="t">Analyse</div></div>
    <div class="dt-step {s3}"><div class="n">03</div><div class="t">Evidence</div></div>
  </div>
</div>
""".splitlines()), unsafe_allow_html=True)


render_hero()
tab_a, tab_d = st.tabs(["Analyze", "Method & accuracy"])

# ═══════════════════════════════════════════════════════════════════
#  ANALYZE
# ═══════════════════════════════════════════════════════════════════
with tab_a:
    col_v, col_m = st.columns([1.25, 1], gap="large")

    with col_v:
        H('<div class="dt-lbl"><span>Evidence file</span><span>Video or photo · up to 200 MB</span></div>')
        with st.container(key="dt_drop"):
            uploaded = st.file_uploader("Upload a video or a photo", type=VIDEO_EXT + IMAGE_EXT,
                                        label_visibility="collapsed", key=uploaded_key)
        is_image = bool(uploaded) and uploaded.name.rsplit(".", 1)[-1].lower() in IMAGE_EXT
        if uploaded:
            if is_image:
                st.image(uploaded, width="stretch")
            else:
                st.video(uploaded)
        else:
            H("""
<div class="dt-checks" style="margin-top:1rem">
  <div class="dt-check"><i>01 · FACE</i><b>Manipulation</b><p>Face swaps and re-enactment, scored face by face and frame by frame.</p></div>
  <div class="dt-check"><i>02 · IMAGE</i><b>AI generation</b><p>Pictures made from scratch by diffusion or GAN models.</p></div>
  <div class="dt-check"><i>03 · FILE</i><b>Provenance</b><p>AI labels, Content Credentials and camera tags in the metadata.</p></div>
</div>
""")

    with col_m:
        if not uploaded:
            H("""
<div class="dt-card">
  <div class="dt-lbl"><span>For a reliable result</span></div>
  <ul class="dt-list">
    <li><i>01</i>Use footage where the face is clearly visible, reasonably large and well lit.</li>
    <li><i>02</i>Upload the original file. Every re-share, screenshot and screen recording removes evidence, including the metadata.</li>
    <li><i>03</i>Video beats a photo: 16 frames give a much steadier answer than one.</li>
  </ul>
</div>
<div class="dt-card">
  <div class="dt-lbl"><span>What you get</span></div>
  <ul class="dt-list">
    <li><i>→</i>One of three answers: no manipulation found, likely manipulated or AI-generated, or inconclusive</li>
    <li><i>→</i>The frame where the face looks most suspicious, and heatmaps of where the evidence sits</li>
    <li><i>→</i>A shareable evidence report, plus the raw scores</li>
  </ul>
</div>
<div class="dt-small" style="margin-top:0">Files are analysed on this server and are not kept after your session.</div>
""")
        else:
            ck = f"tmp_{uploaded.name}_{uploaded.size}"
            if ck not in st.session_state:
                with NamedTemporaryFile(delete=False, suffix="." + uploaded.name.rsplit(".", 1)[-1].lower()) as f:
                    f.write(uploaded.getvalue())
                    st.session_state[ck] = f.name
            tmp = st.session_state[ck]
            ext = esc(uploaded.name.rsplit(".", 1)[-1].upper())
            if is_image:
                im0 = cv2.imread(tmp)
                ih, iw = im0.shape[:2] if im0 is not None else (0, 0)
                kv = (f"<div><span>Type</span><b>Photo</b></div><div><span>Resolution</span><b>{iw}×{ih}</b></div>"
                      f"<div><span>Size</span><b>{fmt_size(uploaded.size)}</b></div>")
                plan = ("<div><span>Faces</span><b>up to 4, each scored</b></div>"
                        "<div><span>Checks</span><b>manipulation · AI · metadata</b></div>"
                        "<div><span>Typical time</span><b>5–20 s</b></div>")
            else:
                fps, dur, w, h, nf = vid_meta(tmp)
                kv = (f"<div><span>Duration</span><b>{dur:.1f}s</b></div><div><span>Frame rate</span><b>{fps:.0f} fps</b></div>"
                      f"<div><span>Resolution</span><b>{w}×{h}</b></div><div><span>Size</span><b>{fmt_size(uploaded.size)}</b></div>"
                      f"<div><span>Frames</span><b>{nf:,}</b></div><div><span>Format</span><b>{ext}</b></div>")
                plan = (f"<div><span>Frames to scan</span><b>{min(st.session_state.max_frames, max(1, nf))}, spread evenly</b></div>"
                        "<div><span>Checks</span><b>manipulation · AI · metadata</b></div>"
                        f"<div><span>Verdict bands</span><b>clean ≤ {T_REAL:.2f} · flagged ≥ {T_FAKE:.2f}</b></div>"
                        "<div><span>Typical time</span><b>under a minute</b></div>")
            H(f"""
<div class="dt-card">
  <div class="dt-lbl"><span>Case file</span><b>{esc(uploaded.name[:40])}</b></div>
  <div class="dt-kv">{kv}</div>
</div>
<div class="dt-card"><div class="dt-lbl"><span>Analysis plan</span></div><div class="dt-plan">{plan}</div></div>
""")
            run = st.button("Run analysis  →", type="primary", width="stretch")
            if not is_image:
                with st.expander("Advanced settings"):
                    st.session_state.max_frames = st.slider(
                        "Frames to analyse", 8, 32, int(st.session_state.max_frames), 4,
                        help="More frames gives a steadier verdict but takes longer. The benchmark used 16.")
                    st.caption("The verdict bands are fixed. They were set on a benchmark, and moving them "
                               "by hand would change what a verdict means.")

    # ── RUN ───────────────────────────────────────────────────────────
    if uploaded and run:
        st.session_state.result = st.session_state.err = None
        st.session_state.err_friendly = False
        bar = st.progress(0.0, text="Loading models…")
        try:
            eng = load_engine()
            prov = provenance.inspect(tmp)
            if is_image:
                bar.progress(0.3, text="Finding faces and scoring them…")
                im = cv2.imread(tmp)
                if im is None:
                    raise ValueError("Could not read this image.")
                r = eng.analyse_image(im)
                st.session_state.result = to_state(r, prov, filename=uploaded.name, fps=1.0, duration=0.0,
                                                   resolution=f"{im.shape[1]}x{im.shape[0]}")
            else:
                def cb(p):
                    bar.progress(min(1.0, float(p)), text=("Finding faces in sampled frames…" if p < 0.25 else
                                                           "Scoring faces for manipulation…" if p < 0.9 else
                                                           "Checking for AI generation…"))
                r = eng.analyse_video(tmp, max_frames=st.session_state.max_frames, progress=cb)
                fps2, dur2, w2, h2, _ = vid_meta(tmp)
                st.session_state.result = to_state(r, prov, filename=uploaded.name, fps=fps2, duration=dur2,
                                                   resolution=f"{w2}x{h2}")
            st.session_state.case = f"{uploaded.name}:{uploaded.size}"
        except NoFaceError as e:
            st.session_state.err, st.session_state.err_friendly = str(e), True
        except Exception:
            st.session_state.err = traceback.format_exc()
        bar.empty()

    if uploaded and st.session_state.err:
        if st.session_state.err_friendly:
            H(f'<div class="dt-note"><b>No face found</b><span>{esc(st.session_state.err)}</span></div>')
        else:
            st.error("The analysis stopped with an error. The details below help with debugging.")
            with st.expander("Error details"):
                st.code(st.session_state.err, language="python")

    # ── CASE REPORT (same page, right under the file) ─────────────────
    res = st.session_state.result
    if res and uploaded and st.session_state.case == f"{uploaded.name}:{uploaded.size}":
        V = VERDICTS[res["verdict"]]
        is_video = res["kind"] == "video"
        media = (f"{res['duration']:.1f}s · {res['fps']:.0f} fps · {res['resolution']}" if is_video
                 else f"photo · {res['resolution']}")
        H(f"""
<div class="dt-case">
  <div><div class="dt-kick" style="margin:0 0 .5rem">Case report</div><h2>{esc(res['filename'][:70])}</h2>
  <div class="dt-meta">{res['stamp']} · {media} · DeepTrace {VERSION}</div></div>
  <span class="dt-badge {V['cls']}">{V['badge']}</span>
</div>
<div class="dt-verdict {V['cls']}">
  <div>{ring_for(res)}</div>
  <div><div class="dt-vlbl" style="color:{V['color']}">Verdict</div>
  <div class="dt-vtitle">{V['title']}</div><p class="dt-vtext">{verdict_text(res)}</p></div>
  <div class="dt-facts">
    <div><span>{'Frames' if is_video else 'Faces'}</span><b>{len(res['scores'])}</b></div>
    <div><span>People</span><b>{res['n_faces']}</b></div>
    <div><span>Quality</span><b style="color:{AMBER if res['low_quality'] else LIME}">{'Too low' if res['low_quality'] else 'OK'}</b></div>
    <div><span>Metadata</span><b style="color:{RED if res['prov_ai'] else IVORY}">{'AI label' if res['prov_ai'] else 'No AI label'}</b></div>
  </div>
</div>
{signals_html(res)}
""")
        for note in res["notes"]:
            H(f'<div class="dt-note"><b>Note</b><span>{esc(note)}</span></div>')

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        e1, e2 = st.columns([1.5, 1], gap="medium")
        with e1:
            if res["evidence"]:
                H(f"""
<div class="dt-card"><div class="dt-lbl"><span>{'Key frame' if is_video else 'Faces found'}</span><span>box colour follows the verdict bands</span></div>
<img src="data:image/jpeg;base64,{res['evidence']}" alt="{esc(res['evidence_label'])}" style="width:100%;border-radius:12px;display:block">
<div class="dt-ev-cap">{esc(res['evidence_label'])}</div></div>
""")
        with e2:
            show_heat = st.toggle("Heatmap overlay", value=True, key="dt_heat")
            items = res["heat"] if show_heat else res["thumbs"]
            cards = "".join(
                f'<div class="dt-thumb"><img src="data:image/jpeg;base64,{b}" alt="{esc(l)}">'
                f'<div><b style="color:{score_color(s)}">{s*100:.0f}%</b> · {esc(l)}</div></div>'
                for l, s, b in items[:6])
            legend = ('<div class="dt-legend"><span class="g"></span><span>more evidence of manipulation · '
                      'fixed scale, so clean faces stay clear</span></div>' if show_heat else "")
            H(f"""
<div class="dt-card"><div class="dt-lbl"><span>{'Most suspicious frames' if is_video else 'Faces'}</span><span>highest first</span></div>
<div class="dt-heat">{cards}</div>{legend}</div>
""")

        if is_video:
            t1, t2 = st.columns([1.5, 1], gap="medium")
            with t1:
                with st.container(border=True):
                    H('<div class="dt-lbl"><span>Timeline</span><span>manipulation score of the lead face</span></div>')
                    st.plotly_chart(chart_timeline(res["scores"], res["frame_idx"], res["fps"]),
                                    width="stretch", config={"displayModeBar": False})
            with t2:
                pr = np.array(res["scores"])
                H(f"""
<div class="dt-card">
  <div class="dt-lbl"><span>Statistics</span><span>lead face</span></div>
  <div class="dt-stats">
    <div><span>Mean</span><b>{pr.mean():.3f}</b></div><div><span>Median</span><b>{np.median(pr):.3f}</b></div>
    <div><span>Spread (sd)</span><b>{pr.std():.3f}</b></div><div><span>Min</span><b>{pr.min():.3f}</b></div>
    <div><span>Max</span><b>{pr.max():.3f}</b></div><div><span>Flagged / clean</span><b>{int((pr >= T_FAKE).sum())} / {int((pr <= T_REAL).sum())}</b></div>
    <div><span>Frames sampled</span><b>{res['n_sampled']}</b></div><div><span>No face</span><b>{res['n_dropped']}</b></div>
    <div><span>Bright / sharp</span><b>{res['brightness']:.0f} / {res['sharpness']:.0f}</b></div>
  </div>
  <div class="dt-small">The verdict uses the mean across frames, the rule the benchmark was measured with.</div>
</div>
""")
            if len(res["people"]) > 1:
                cards = "".join(
                    f'<div class="dt-thumb"><img src="data:image/jpeg;base64,{p["b64"]}" alt="Person {i+1}">'
                    f'<div><b style="color:{score_color(p["prob"])}">{p["prob"]*100:.0f}%</b> · person {i+1} · {p["n"]} frames</div></div>'
                    for i, p in enumerate(res["people"]))
                H(f'<div class="dt-card"><div class="dt-lbl"><span>People in this clip</span><span>each scored separately</span></div><div class="dt-thumbs">{cards}</div></div>')

        icon = {"ai": "●", "edit": "●", "info": "○"}
        prov_items = "".join(f'<li><i class="{lv}">{icon[lv]}</i><span><b>{esc(t)}</b>{esc(d)}</span></li>'
                             for lv, t, d in res["prov"]["findings"])
        H(f"""
<div class="dt-card"><div class="dt-lbl"><span>Provenance</span><span>what the file says about itself</span></div>
<ul class="dt-list dt-prov">{prov_items}</ul>
<div class="dt-small">Missing metadata proves nothing, because most apps strip it. A label that says "AI-generated" is strong evidence.</div></div>
""")

        unit = "time_s" if is_video else "face"
        xs = [round(i / res["fps"], 2) for i in res["frame_idx"]] if is_video else [i + 1 for i in res["frame_idx"]]
        df = pd.DataFrame({unit: xs, "manipulation_score": [round(s, 4) for s in res["scores"]],
                           "band": [band_word(s, T_REAL, T_FAKE) for s in res["scores"]]})
        H('<div class="dt-lbl" style="margin-top:.6rem"><span>Export</span><span>share the evidence, not just the verdict</span></div>')
        d1, d2, _ = st.columns([1, 1, 1.2])
        stem = res["filename"].rsplit(".", 1)[0]
        with d1:
            st.download_button("↓ Evidence report (.html)", build_html_report(res), file_name=f"deeptrace_{stem}.html",
                               mime="text/html", width="stretch")
        with d2:
            st.download_button("↓ Raw scores (.csv)", df.to_csv(index=False), file_name=f"deeptrace_{stem}.csv",
                               mime="text/csv", width="stretch")

        from urllib.parse import quote
        fb_title = f"Wrong result? DeepTrace said: {V['title']}"
        fb_body = (f"**DeepTrace said:** {V['title']}\n"
                   f"**Face manipulation:** {res['swap_prob']*100:.1f}%  ·  **AI generation:** {res['aigen_prob']*100:.1f}%\n"
                   f"**Media:** {'video' if is_video else 'photo'}, {res['resolution']}  ·  {VERSION}\n\n"
                   "**What it really is** (real / face swap / AI-generated / edited):\n\n"
                   "**Where it came from** (your own camera, a site, an app):\n\n"
                   "_Please don't upload anything private. A public link is enough._")
        fb_url = (f"https://huggingface.co/spaces/{SPACE_ID}/discussions/new?"
                  f"title={quote(fb_title)}&description={quote(fb_body)}")
        H(f'<div class="dt-small" style="margin-top:1rem">Think DeepTrace got this wrong? '
          f'<a href="{fb_url}" target="_blank" rel="noopener">Report it in one click</a>. '
          'The scores are filled in for you; your file is never shared. Failure cases are how the next version improves.</div>')

        H('<div class="dt-note"><b>Read this</b><span>DeepTrace is a screening tool. Its scores are estimates, not proof. It misses some fakes and can be wrong about real footage. Check the source and other evidence before acting on a result, and never use it alone to accuse someone.</span></div>')

# ═══════════════════════════════════════════════════════════════════
#  METHOD & ACCURACY
# ═══════════════════════════════════════════════════════════════════
BENCH_HTML = """
<div class="dt-card" style="margin-top:1rem">
  <div class="dt-lbl"><span>Measured results</span><span>full pipeline, same settings as this app</span></div>
  <table class="dt-bench">
    <tr><th>Test set</th><th>Files</th><th>Correct</th><th>Inconclusive</th><th>Wrong</th></tr>
    <tr><td>Home webcam clips (real)</td><td class="n">14</td><td class="n" style="color:var(--lime)">9</td><td class="n" style="color:var(--amber)">5</td><td class="n" style="color:var(--red)">0</td></tr><tr><td>Google DFD (real)</td><td class="n">16</td><td class="n" style="color:var(--lime)">14</td><td class="n" style="color:var(--amber)">2</td><td class="n" style="color:var(--red)">0</td></tr><tr><td>Celeb-DF v2 (real) <span style="color:var(--lime)">●</span></td><td class="n">40</td><td class="n" style="color:var(--lime)">28</td><td class="n" style="color:var(--amber)">10</td><td class="n" style="color:var(--red)">2</td></tr><tr><td>WildDeepfake (real)</td><td class="n">13</td><td class="n" style="color:var(--lime)">7</td><td class="n" style="color:var(--amber)">6</td><td class="n" style="color:var(--red)">0</td></tr><tr><td>Phone selfie clips (real) <span style="color:var(--lime)">●</span></td><td class="n">5</td><td class="n" style="color:var(--lime)">5</td><td class="n" style="color:var(--amber)">0</td><td class="n" style="color:var(--red)">0</td></tr><tr><td>Google DFD (face swaps)</td><td class="n">16</td><td class="n" style="color:var(--lime)">11</td><td class="n" style="color:var(--amber)">3</td><td class="n" style="color:var(--red)">2</td></tr><tr><td>Celeb-DF v2 (face swaps) <span style="color:var(--lime)">●</span></td><td class="n">40</td><td class="n" style="color:var(--lime)">30</td><td class="n" style="color:var(--amber)">8</td><td class="n" style="color:var(--red)">2</td></tr><tr><td>WildDeepfake (internet fakes)</td><td class="n">14</td><td class="n" style="color:var(--lime)">3</td><td class="n" style="color:var(--amber)">11</td><td class="n" style="color:var(--red)">0</td></tr><tr><td>Commercial face swaps <span style="color:var(--lime)">●</span></td><td class="n">5</td><td class="n" style="color:var(--lime)">1</td><td class="n" style="color:var(--amber)">3</td><td class="n" style="color:var(--red)">1</td></tr><tr><td>Real photos (FFHQ, CelebA-HQ, selfies) <span style="color:var(--lime)">●</span></td><td class="n">85</td><td class="n" style="color:var(--lime)">79</td><td class="n" style="color:var(--amber)">3</td><td class="n" style="color:var(--red)">3</td></tr><tr><td>AI-generated faces (Flux, SDXL) <span style="color:var(--lime)">●</span></td><td class="n">120</td><td class="n" style="color:var(--lime)">87</td><td class="n" style="color:var(--amber)">14</td><td class="n" style="color:var(--red)">19</td></tr>
  </table>
  <div class="dt-small" style="margin-top:.8rem">Across 88 real videos, 2 were wrongly flagged (2%). Across 75 fake videos, 45 were flagged (60%), 25 were inconclusive and 5 passed as clean (7%). <span style="color:var(--lime)">●</span> marks held-out sets that played no part in choosing the models or thresholds. The samples are small, so read these as rough rates, not guarantees. Details are in EVALUATION.md in the Space repository.</div>
</div>
"""

with tab_d:
    H(f"""
<div class="dt-rhead"><div><div class="dt-kick" style="margin:0 0 .5rem">How it works</div><h2>From a file to a verdict</h2></div></div>
<div class="dt-pipe">
  <div><i>01 · SAMPLE</i><b>Frame sampling</b><p>16 frames are taken evenly across the whole clip, so the start, middle and end are all covered. A photo is analysed as it is.</p></div>
  <div><i>02 · DETECT</i><b>Face detection</b><p>OpenCV's YuNet finds every face. People are tracked across frames and each person is scored separately.</p></div>
  <div><i>03 · SCORE</i><b>Face manipulation</b><p>EfficientNet-B7 networks from the winning entry of the Deepfake Detection Challenge, trained on about 100,000 phone and webcam clips.</p></div>
  <div><i>04 · SCORE</i><b>AI generation</b><p>Whole frames and face crops go through the Community Forensics detector, trained on images from thousands of generators.</p></div>
  <div><i>05 · READ</i><b>Provenance</b><p>The file's metadata is searched for IPTC AI labels, generator settings, Content Credentials and camera tags.</p></div>
  <div><i>06 · DECIDE</i><b>Three-way verdict</b><p>Mean score up to {T_REAL:.2f}: clean. From {T_FAKE:.2f}: flagged. Between, or if the face is too dark or blurry: inconclusive.</p></div>
</div>
""")
    H(BENCH_HTML)
    m1, m2 = st.columns(2, gap="medium")
    with m1:
        H("""
<div class="dt-card" style="margin-top:1rem">
  <div class="dt-lbl"><span>Reading the heatmaps</span></div>
  <ul class="dt-list">
    <li><i>—</i><span>They are class activation maps: each region's exact contribution to the "manipulated" score of the first face-manipulation network.</span></li>
    <li><i>—</i><span>The colour scale is fixed across all files. A clean face stays clear; heat means the network found evidence there.</span></li>
    <li><i>—</i><span>They show where the network looked, not a mask of what was edited.</span></li>
  </ul>
</div>
<div class="dt-card">
  <div class="dt-lbl"><span>Models</span></div>
  <ul class="dt-list">
    <li><i>—</i><span><b>Face manipulation:</b> EfficientNet-B7, Selim Seferbekov's 1st-place DFDC solution (MIT).</span></li>
    <li><i>—</i><span><b>AI generation:</b> Community Forensics ViT-S/16-384, Park and Owens, CVPR 2025 (MIT).</span></li>
    <li><i>—</i><span><b>Face detection:</b> YuNet, OpenCV model zoo (MIT).</span></li>
  </ul>
</div>
""")
    with m2:
        H("""
<div class="dt-card" style="margin-top:1rem">
  <div class="dt-lbl"><span>Known limits</span></div>
  <ul class="dt-list">
    <li><i>—</i>It misses some fakes. A clean result is not a certificate of authenticity.</li>
    <li><i>—</i>New generators appear faster than detectors learn them. Expect the newest fakes to pass more often.</li>
    <li><i>—</i>Fully AI-generated video (Sora, Veo and similar) has not been benchmarked.</li>
    <li><i>—</i>Dark, blurry, tiny or heavily re-compressed faces get no verdict.</li>
    <li><i>—</i>Filters, beauty modes and heavy make-up can raise scores on real footage.</li>
    <li><i>—</i>Audio, lip-sync and voice cloning are not analysed.</li>
  </ul>
</div>
""")

render_hero()
H(f'<div class="dt-foot"><span>DeepTrace {VERSION} · built by <a href="https://mihirruwala.cv" target="_blank">Mihir Ruwala</a></span><span><a href="https://huggingface.co/spaces/{SPACE_ID}/discussions" target="_blank" rel="noopener">Feedback</a> · screening tool, not legal or forensic proof</span></div>')
