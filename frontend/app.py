"""Frontend Streamlit premium V6 — Bank Marketing MLOps.

Version V6 : polish final orienté soutenance et démonstration.
- Hero d’accueil plus compact et plus dashboard.
- Slider du seuil métier harmonisé avec la charte violette.
- Tableau Prévisions épuré par défaut pour privilégier les colonnes métier.
- Historique brut conservé dans un expander pour audit.
- Fallback plus propre pour le modèle servi lorsque l’API renvoie unknown.

Pages incluses :
- Accueil : vision produit, KPIs, pipeline et santé des services.
- Prédiction : formulaire client + appel API /predict.
- Suivi du modèle : versions MLflow, alias et promotion de modèles.
- Évaluation : porte qualité, métriques et artefacts.
- Prévisions : historique des prédictions + feedback métier.

Lancement recommandé :
    make frontend
avec les services utiles :
    make api
    make mlflow
    make db
"""

from __future__ import annotations

import html
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from streamlit_option_menu import option_menu

from bank_marketing.config import (
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)
from bank_marketing.evaluate import evaluate_model

# ==============================================================================
# Configuration
# ==============================================================================

st.set_page_config(
    page_title="Bank Marketing — MLOps",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", "").rstrip("/")
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "").rstrip("/")
AIRFLOW_UI_URL = os.environ.get("AIRFLOW_UI_URL", "").rstrip("/")
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/goamegah/mlops").rstrip("/")
AUTHOR = os.environ.get("AUTHOR", "Godwin Amegah")
AUTHOR_SUBTITLE = os.environ.get("AUTHOR_SUBTITLE", "ESGI · IABD — Fil rouge MLOps")
INITIALS = "".join(part[0] for part in AUTHOR.split()[:2]).upper() or "BM"

# Éviter que l'UI se fige si MLflow est indisponible.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

CATEGORIES: dict[str, list[str]] = {
    "job": [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["yes", "no"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["unknown", "failure", "other", "success"],
}

# ==============================================================================
# Design system : SVG + CSS + composants HTML
# ==============================================================================

PRIMARY = "#4F46E5"
PRIMARY_DARK = "#4338CA"
PURPLE = "#7C3AED"
BLUE = "#0EA5E9"
GREEN = "#16A34A"
ORANGE = "#EA580C"
RED = "#DC2626"
SLATE = "#25263A"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
CARD = "#FFFFFF"

GITHUB_SVG = (
    "<svg width='15' height='15' viewBox='0 0 16 16' fill='currentColor' "
    "style='vertical-align:-2px;'><path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 "
    "5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09"
    "-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87"
    " 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2"
    "-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04"
    " 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65"
    " 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0"
    " 0016 8c0-4.42-3.58-8-8-8z'/></svg>"
)


def _svg(body: str, size: int = 18, sw: float = 1.9) -> str:
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
        f"stroke='currentColor' stroke-width='{sw}' stroke-linecap='round' "
        f"stroke-linejoin='round' style='vertical-align:-3px;'>{body}</svg>"
    )


IC_HOME = "<path d='m3 11 9-8 9 8'/><path d='M5 10v10h14V10'/><path d='M9 20v-6h6v6'/>"
IC_TARGET = "<circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/>"
IC_USER = "<path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/><circle cx='12' cy='7' r='4'/>"
IC_BRANCH = "<line x1='6' y1='3' x2='6' y2='15'/><circle cx='18' cy='6' r='3'/><circle cx='6' cy='18' r='3'/><path d='M18 9a9 9 0 0 1-9 9'/>"
IC_SHIELD = "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/><path d='m9 12 2 2 4-4'/>"
IC_PACKAGE = "<path d='M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z'/><polyline points='3.27 6.96 12 12.01 20.73 6.96'/><line x1='12' y1='22.08' x2='12' y2='12'/>"
IC_FOLDER = "<path d='M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2z'/>"
IC_TREND = "<polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/>"
IC_CHECK = "<polyline points='20 6 9 17 4 12'/>"
IC_ALERT = "<path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3.05h16.94a2 2 0 0 0 1.71-3.05l-8.47-14.14a2 2 0 0 0-3.42 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/>"
IC_WALLET = "<circle cx='12' cy='11' r='8'/><path d='M7 11H4a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-7a2 2 0 0 0-2-2h-3'/>"
IC_PHONE = "<path d='M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z'/>"
IC_DATABASE = "<ellipse cx='12' cy='5' rx='9' ry='3'/><path d='M21 12c0 1.66-4 3-9 3s-9-1.34-9-3m0-7v6c0 1.66 4 3 9 3s9-1.34 9-3V5'/><line x1='3' y1='12' x2='3' y2='19'/><line x1='21' y1='12' x2='21' y2='19'/><ellipse cx='12' cy='19' rx='9' ry='3'/>"
IC_ROCKET = "<path d='M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z'/><path d='m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z'/><path d='M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0'/><path d='M15 9c0 .55 3.03.55 4 2 1.08 1.62 0 5 0 5'/>"
IC_TABLE = "<rect x='3' y='3' width='18' height='18' rx='2'/><path d='M3 9h18M3 15h18M9 3v18M15 3v18'/>"
IC_SPARK = "<path d='M13 2 3 14h8l-1 8 11-14h-8l1-6z'/>"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

          :root {{
            --primary: {PRIMARY};
            --primary-dark: {PRIMARY_DARK};
            --purple: {PURPLE};
            --blue: {BLUE};
            --green: {GREEN};
            --orange: {ORANGE};
            --red: {RED};
            --slate: {SLATE};
            --muted: {MUTED};
            --border: {BORDER};
            --card: {CARD};
            --shadow-sm: 0 8px 22px rgba(15, 23, 42, 0.055);
            --shadow-md: 0 16px 36px rgba(15, 23, 42, 0.10);
            --radius: 20px;
          }}

          html, body, [class*="css"], [data-testid="stMarkdownContainer"] {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}

          #MainMenu, footer, header {{ visibility: hidden; }}
          .stApp {{
            background:
              radial-gradient(circle at 18% 8%, rgba(79,70,229,0.10), transparent 26%),
              radial-gradient(circle at 88% 2%, rgba(14,165,233,0.10), transparent 30%),
              linear-gradient(180deg, #F8FAFF 0%, #F3F5FB 100%);
          }}
          .block-container {{
            max-width: 1260px;
            padding: .65rem 1.35rem 1.4rem 1.35rem;
          }}
          h1, h2, h3, h4 {{ color: var(--slate); letter-spacing: -0.025em; }}
          hr {{ border: none !important; border-top: 1px solid #EEF0F6 !important; margin: 1.35rem 0 !important; }}

          /* Sidebar premium */
          [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.96);
            border-right: 1px solid #EEF0F6;
            box-shadow: 12px 0 28px rgba(15, 23, 42, 0.035);
          }}
          [data-testid="stSidebar"] > div:first-child {{ padding-top: .95rem; }}
          .brand-card {{
            margin: .1rem .1rem .8rem .1rem;
            padding: .82rem .82rem;
            border-radius: 20px;
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFF 100%);
            border: 1px solid #EEF0F6;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
            text-align: center;
          }}
          .brand-icon {{
            width: 46px; height: 46px; margin: 0 auto .48rem auto;
            border-radius: 14px;
            display: grid; place-items: center;
            color: var(--primary);
            background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
            border: 1px solid #E0E7FF;
          }}
          .brand-title {{ font-weight: 850; font-size: 1rem; color: var(--slate); }}
          .brand-subtitle {{ font-size: .73rem; color: var(--muted); margin-top: .12rem; }}
          .sidebar-section-title {{
            font-size: .68rem;
            color: #9CA3AF;
            font-weight: 800;
            letter-spacing: .1em;
            text-transform: uppercase;
            margin: .85rem 0 .55rem .3rem;
          }}
          .sidebar-link {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: .55rem;
            width: 100%;
            padding: .55rem .68rem;
            border-radius: 13px;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            color: #374151 !important;
            text-decoration: none !important;
            font-weight: 700;
            font-size: .82rem;
            transition: all .18s ease;
            box-sizing: border-box;
            margin-bottom: .45rem;
          }}
          .sidebar-link:hover {{
            color: var(--primary) !important;
            border-color: #C7D2FE;
            box-shadow: 0 10px 24px rgba(79,70,229,.12);
          }}
          .mini-profile {{
            margin-top: .85rem;
            padding: .82rem;
            border-radius: 14px;
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFF 100%);
            border: 1px solid #EEF0F6;
          }}
          .mini-avatar {{
            width: 35px; height: 35px; border-radius: 14px;
            background: linear-gradient(135deg, var(--primary), var(--purple));
            color: white;
            display: grid; place-items: center;
            font-weight: 850;
          }}

          /* streamlit-option-menu */
          .nav-link {{
            border-radius: 14px !important;
            transition: all .18s ease !important;
            margin: 4px 0 !important;
          }}
          .nav-link:hover {{
            background: #EEF2FF !important;
            transform: translateX(2px);
          }}
          .nav-link.active {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--purple) 100%) !important;
            box-shadow: 0 14px 28px rgba(79,70,229,.24);
          }}

          /* Widgets */
          .stButton > button {{
            border-radius: 13px !important;
            font-weight: 750 !important;
            border: 1px solid transparent !important;
            background: linear-gradient(135deg, var(--primary), var(--purple)) !important;
            color: white !important;
            min-height: 2.35rem;
            box-shadow: 0 10px 24px rgba(79,70,229,.20);
            transition: all .18s ease !important;
          }}
          .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 16px 32px rgba(79,70,229,.30) !important;
          }}
          /* V5 : Streamlit applique parfois la couleur primaire du thème aux submit de formulaire.
             On force ici l'action principale en violet pour éviter le bouton rouge. */
          [data-testid="stFormSubmitButton"] button,
          [data-testid="stFormSubmitButton"] button[kind="primary"],
          [data-testid="stButton"] button[kind="primary"],
          button[kind="primary"] {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--purple) 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid transparent !important;
            border-radius: 13px !important;
            font-weight: 800 !important;
            box-shadow: 0 12px 26px rgba(79,70,229,.24) !important;
          }}
          [data-testid="stFormSubmitButton"] button:hover,
          [data-testid="stButton"] button[kind="primary"]:hover,
          button[kind="primary"]:hover {{
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--purple) 100%) !important;
            transform: translateY(-1px);
            box-shadow: 0 18px 36px rgba(79,70,229,.32) !important;
          }}
          [data-testid="stLinkButton"] a {{
            border-radius: 13px !important;
            font-weight: 750 !important;
            border: 1px solid #E5E7EB !important;
            background: #FFFFFF !important;
            color: #374151 !important;
            box-shadow: 0 8px 20px rgba(15,23,42,.045) !important;
            min-height: 2.35rem;
          }}
          [data-testid="stLinkButton"] a:hover {{
            color: var(--primary) !important;
            border-color: #C7D2FE !important;
            transform: translateY(-1px);
          }}
          .stDownloadButton > button {{ border-radius: 13px !important; }}
          [data-testid="stSelectbox"] label,
          [data-testid="stNumberInput"] label,
          [data-testid="stTextInput"] label {{
            font-weight: 750 !important;
            color: var(--slate) !important;
            font-size: .9rem !important;
          }}
          [data-testid="stForm"] {{
            border: 1px solid #E8EBF3;
            border-radius: 22px;
            background: rgba(255,255,255,.92);
            box-shadow: var(--shadow-sm);
            padding: 1rem 1.1rem 1.15rem 1.1rem;
          }}
          [data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #E8EBF3;
            box-shadow: var(--shadow-sm);
          }}

          /* Layout helpers */
          .premium-hero {{
            position: relative;
            overflow: hidden;
            min-height: 126px;
            border-radius: 22px;
            padding: 1.28rem 1.85rem;
            color: #fff;
            background:
              radial-gradient(circle at 78% 28%, rgba(255,255,255,.22), transparent 20%),
              radial-gradient(circle at 95% 12%, rgba(14,165,233,.28), transparent 20%),
              linear-gradient(135deg, #312E81 0%, #4F46E5 45%, #7C3AED 100%);
            box-shadow: 0 24px 58px rgba(79,70,229,.25);
            border: 1px solid rgba(255,255,255,.22);
            margin-bottom: .58rem;
          }}
          .premium-hero::before {{
            content: "";
            position: absolute;
            inset: auto -6% -42% 38%;
            height: 145px;
            background: rgba(255,255,255,.10);
            transform: rotate(-8deg);
            border-radius: 999px;
          }}
          .hero-content {{ position: relative; z-index: 2; max-width: 760px; }}
          .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            font-weight: 800;
            font-size: .68rem;
            letter-spacing: .08em;
            text-transform: uppercase;
            background: rgba(255,255,255,.16);
            border: 1px solid rgba(255,255,255,.18);
            border-radius: 999px;
            padding: .30rem .62rem;
            margin-bottom: .52rem;
          }}
          .hero-title {{ font-size: clamp(1.55rem, 2.55vw, 2.16rem); line-height: 1.05; font-weight: 900; letter-spacing: -.045em; }}
          .hero-subtitle {{ margin-top: .46rem; font-size: .82rem; line-height: 1.55; color: rgba(255,255,255,.88); }}
          .hero-visual {{ position: absolute; right: 1.35rem; top: 50%; transform: translateY(-50%); opacity: .98; z-index: 1; }}

          .page-header {{
            border-radius: 24px;
            padding: 1.05rem 1.2rem;
            background: linear-gradient(135deg, #FFFFFF 0%, #F7F8FF 55%, #EEF2FF 100%);
            border: 1px solid #E8EBF3;
            box-shadow: var(--shadow-sm);
            margin-bottom: .65rem;
          }}
          .page-header-inner {{ display: flex; gap: 1rem; align-items: flex-start; }}
          .page-icon {{
            min-width: 48px; width: 48px; height: 48px; border-radius: 14px;
            display: grid; place-items: center;
            color: var(--primary);
            background: #FFFFFF;
            border: 1px solid #E0E7FF;
            box-shadow: 0 10px 20px rgba(79,70,229,.12);
          }}
          .page-title {{ color: var(--slate); font-size: 1.25rem; line-height: 1.2; font-weight: 900; letter-spacing: -.025em; }}
          .page-desc {{ margin-top: .28rem; color: var(--muted); font-size: .88rem; line-height: 1.55; }}

          .grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .7rem; margin: .65rem 0 .82rem 0; }}
          .grid-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem; margin: .65rem 0 .82rem 0; }}
          .grid-2-1 {{ display: grid; grid-template-columns: 2fr 1fr; gap: .68rem; margin: .65rem 0 .82rem 0; align-items: stretch; }}
          .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; margin: .65rem 0 .82rem 0; }}

          .card {{
            background: rgba(255,255,255,.94);
            border: 1px solid #E8EBF3;
            border-radius: 22px;
            box-shadow: var(--shadow-sm);
            padding: .95rem 1.05rem;
          }}
          .card-title {{
            display: flex;
            align-items: center;
            gap: .55rem;
            color: var(--slate);
            font-weight: 900;
            font-size: .98rem;
            letter-spacing: -.015em;
            margin-bottom: .58rem;
          }}
          .card-text {{ color: #4B5563; font-size: .88rem; line-height: 1.62; }}
          .metric-card {{
            position: relative;
            overflow: hidden;
            min-height: 94px;
            background: rgba(255,255,255,.95);
            border: 1px solid #E8EBF3;
            border-radius: 22px;
            box-shadow: var(--shadow-sm);
            padding: .82rem .88rem;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
          }}
          .metric-card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-md); border-color: #CBD5E1; }}
          .metric-row {{ display: flex; align-items: center; gap: .68rem; }}
          .metric-icon {{
            min-width: 40px; width: 40px; height: 40px;
            border-radius: 14px;
            display: grid; place-items: center;
          }}
          .metric-label {{ color: #8A8FA3; font-size: .68rem; font-weight: 800; }}
          .metric-value {{ margin-top: .18rem; font-size: 1.42rem; font-weight: 950; line-height: 1.05; letter-spacing: -.035em; }}
          .pill {{ display: inline-flex; align-items: center; gap: .35rem; padding: .22rem .58rem; border-radius: 999px; font-weight: 850; font-size: .68rem; }}
          .section-title {{ display: flex; align-items: center; gap: .55rem; color: var(--slate); font-size: 1.02rem; font-weight: 950; margin: 1rem 0 .55rem 0; letter-spacing: -.025em; }}
          .muted {{ color: var(--muted); }}

          .pipeline-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: .65rem; margin-top: .55rem; }}
          .pipeline-step {{
            position: relative;
            background: #FFFFFF;
            border: 1px solid #E8EBF3;
            border-radius: 18px;
            padding: .72rem .62rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
            min-height: 108px;
            transition: all .18s ease;
          }}
          .pipeline-step:hover {{ transform: translateY(-4px); border-color: #C7D2FE; box-shadow: 0 18px 34px rgba(79,70,229,.13); }}
          .pipeline-step:not(:last-child)::after {{
            content: "";
            position: absolute;
            left: calc(100% + .08rem);
            top: 50px;
            width: .72rem;
            border-top: 2px dashed #C7D2FE;
            opacity: .95;
          }}
          .pipeline-num {{ width: 24px; height: 24px; border-radius: 999px; display: grid; place-items: center; background: #EEF2FF; color: var(--primary); font-weight: 900; margin: 0 auto .42rem auto; font-size: .68rem; }}
          .pipeline-ico {{ width: 35px; height: 35px; border-radius: 15px; display: grid; place-items: center; margin: 0 auto .48rem auto; }}
          .pipeline-title {{ font-size: .84rem; font-weight: 900; color: var(--slate); }}
          .pipeline-sub {{ margin-top: .15rem; font-size: .68rem; color: #8A8FA3; line-height: 1.35; }}

          .service-card {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 68px;
            border-radius: 20px;
            padding: .72rem .88rem;
            border: 1px solid;
            box-shadow: var(--shadow-sm);
          }}
          .status-left {{ display: flex; align-items: center; gap: .8rem; }}
          .status-dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; box-shadow: 0 0 0 5px rgba(22,163,74,.10); }}
          .service-title {{ font-weight: 900; color: var(--slate); }}
          .service-sub {{ margin-top: .12rem; font-weight: 800; font-size: .75rem; }}

          .result-panel {{
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%);
            border: 1px solid #E0E7FF;
            border-radius: 24px;
            padding: .95rem 1.05rem;
            box-shadow: var(--shadow-sm);
            margin-top: 1rem;
          }}
          .probability-bar {{ height: 12px; background: #EEF2FF; border-radius: 999px; overflow: hidden; margin-top: .65rem; }}
          .probability-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--primary), var(--purple)); }}


          /* V6 polish */
          [data-testid="stSlider"] label {{
            font-weight: 850 !important;
            color: var(--slate) !important;
          }}
          /* Harmonisation du slider Streamlit/BaseWeb : on force le violet plutôt que le rouge du thème par défaut. */
          [data-testid="stSlider"] [role="slider"] {{
            background: linear-gradient(135deg, var(--primary), var(--purple)) !important;
            border: 3px solid #FFFFFF !important;
            box-shadow: 0 6px 16px rgba(79,70,229,.28) !important;
          }}
          [data-testid="stSlider"] div[data-baseweb="slider"] > div {{
            background: #E5E7EB !important;
          }}
          [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
            background: linear-gradient(90deg, var(--primary), var(--purple)) !important;
          }}
          [data-testid="stCaptionContainer"] {{
            color: #6B7280 !important;
            font-weight: 650;
          }}
          .stDownloadButton > button {{
            background: #FFFFFF !important;
            color: var(--primary) !important;
            border: 1px solid #C7D2FE !important;
            box-shadow: 0 8px 18px rgba(79,70,229,.10) !important;
          }}
          .stDownloadButton > button:hover {{
            background: #EEF2FF !important;
            transform: translateY(-1px);
          }}

          @media (max-width: 1100px) {{
            .grid-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .grid-3 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .grid-2-1 {{ grid-template-columns: 1fr; }}
            .pipeline-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
            .pipeline-step::after {{ display: none !important; }}
            .hero-visual {{ opacity: .18; right: .5rem; }}
            .hero-content {{ max-width: 100%; }}
          }}
          @media (max-width: 720px) {{
            .block-container {{ padding-left: .85rem; padding-right: .85rem; }}
            .grid-4, .grid-3, .grid-2 {{ grid-template-columns: 1fr; }}
            .pipeline-grid {{ grid-template-columns: 1fr; }}
            .premium-hero {{ padding: 1.6rem 1.35rem; }}
            .page-header-inner {{ flex-direction: column; }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon_body: str, color: str, bg: str, pill: str | None = None) -> str:
    pill_html = f"<div style='margin-top:.48rem'>{pill}</div>" if pill else ""
    return (
        "<div class='metric-card'>"
        "<div class='metric-row'>"
        f"<div class='metric-icon' style='background:{bg}; color:{color};'>{_svg(icon_body, 22)}</div>"
        "<div>"
        f"<div class='metric-label'>{_escape(label)}</div>"
        f"<div class='metric-value' style='color:{color};'>{_escape(value)}</div>"
        "</div></div>"
        f"{pill_html}"
        "</div>"
    )


def pill(text: str, fg: str = PRIMARY, bg: str = "#EEF2FF", icon: str | None = None) -> str:
    icon_html = _svg(icon, 14, 2.3) if icon else ""
    return f"<span class='pill' style='color:{fg}; background:{bg};'>{icon_html}{_escape(text)}</span>"


def page_header(title: str, description: str, icon_body: str) -> None:
    st.markdown(
        f"""
        <div class='page-header'>
          <div class='page-header-inner'>
            <div class='page-icon'>{_svg(icon_body, 26)}</div>
            <div>
              <div class='page-title'>{_escape(title)}</div>
              <div class='page-desc'>{description}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, icon_body: str, color: str = PRIMARY) -> None:
    st.markdown(
        f"<div class='section-title'><span style='color:{color}'>{_svg(icon_body, 20)}</span>{_escape(title)}</div>",
        unsafe_allow_html=True,
    )


def info_card(title: str, text: str, icon_body: str, color: str = PRIMARY) -> str:
    return (
        "<div class='card'>"
        f"<div class='card-title'><span style='color:{color}'>{_svg(icon_body, 19)}</span>{_escape(title)}</div>"
        f"<div class='card-text'>{text}</div>"
        "</div>"
    )


def quality_pill(value: str | None, threshold: float, missing_text: str = "Non loggé") -> str:
    """Pastille de qualité. V5 : une métrique absente est affichée comme donnée non loggée."""
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return pill(missing_text, "#6B7280", "#F3F4F6")
    if x >= threshold:
        return pill("Bon", GREEN, "#DCFCE7", IC_CHECK)
    if x >= threshold - 0.15:
        return pill("À surveiller", "#B45309", "#FEF3C7", IC_ALERT)
    return pill("À améliorer", "#C2410C", "#FFEDD5", IC_ALERT)


def fmt_metric(value: str | int | float | None, missing: str = "—") -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return missing


# ==============================================================================
# Données externes : API / MLflow
# ==============================================================================

@st.cache_data(ttl=10, show_spinner=False)
def ping(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_data(ttl=15, show_spinner=False)
def registry_rows_cached() -> list[dict[str, Any]]:
    return load_registry_rows()


def _clean_value(value: Any) -> str:
    """Normalise une valeur de métrique/tag MLflow pour l'affichage."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "-"}:
        return ""
    return text


def first_tag(tags: dict[str, Any], *keys: str) -> str:
    """Retourne le premier tag non vide parmi plusieurs noms possibles."""
    for key in keys:
        text = _clean_value(tags.get(key))
        if text:
            return text
    return ""


def first_metric(tags: dict[str, Any], metrics: dict[str, Any], *keys: str) -> str:
    """Retourne une métrique depuis les tags OU les vraies métriques du run MLflow.

    V3 : c'est important, car selon les scripts d'entraînement les métriques
    sont parfois stockées dans `run.data.metrics` et pas dans les tags de la
    ModelVersion. C'est typiquement ce qui peut faire apparaître un F1-score
    comme “Non disponible” dans le dashboard alors qu'il existe dans MLflow.
    """
    for source in (tags, metrics):
        for key in keys:
            text = _clean_value(source.get(key))
            if text:
                return text
    return ""


def load_registry_rows() -> list[dict[str, Any]]:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        registered = client.get_registered_model(MODEL_NAME)
    except RestException:
        return []

    aliases_by_version: dict[int, list[str]] = {}
    for alias, version in dict(registered.aliases).items():
        aliases_by_version.setdefault(int(version), []).append(alias)

    rows: list[dict[str, Any]] = []
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        tags = dict(v.tags or {})
        metrics: dict[str, Any] = {}

        # V3 : récupération robuste des métriques directement depuis le run MLflow.
        # Certains entraînements loggent les métriques dans le run, mais ne les
        # copient pas en tags sur la ModelVersion. On évite donc les “—” inutiles.
        run_id = getattr(v, "run_id", None)
        if run_id:
            try:
                metrics = dict(client.get_run(run_id).data.metrics or {})
            except Exception:
                metrics = {}

        version = int(v.version)
        aliases = ", ".join(sorted(aliases_by_version.get(version, [])))
        rows.append(
            {
                "version": version,
                "alias": aliases,
                "model_family": first_tag(tags, "model_family", "family", "estimator", "algorithm"),
                "search_method": first_tag(tags, "search_method", "search", "tuning_method"),
                "f1": first_metric(
                    tags,
                    metrics,
                    "f1",
                    "f1_score",
                    "test_f1",
                    "test_f1_score",
                    "eval_f1",
                    "validation_f1",
                    "val_f1",
                    "best_f1",
                    "f1_macro",
                    "weighted_f1",
                ),
                "roc_auc": first_metric(
                    tags,
                    metrics,
                    "roc_auc",
                    "roc_auc_score",
                    "test_roc_auc",
                    "test_roc_auc_score",
                    "auc",
                    "test_auc",
                    "eval_roc_auc",
                    "validation_roc_auc",
                    "val_roc_auc",
                    "best_roc_auc",
                ),
                "cv_roc_auc": first_metric(tags, metrics, "cv_roc_auc", "mean_cv_roc_auc", "cv_auc", "mean_auc"),
                "run_id": run_id or "",
            }
        )
    rows.sort(key=lambda r: r["version"], reverse=True)
    return rows


def safe_registry_rows() -> list[dict[str, Any]]:
    if not ping(f"{MLFLOW_TRACKING_URI}/health"):
        return []
    try:
        return registry_rows_cached()
    except Exception:
        return []


def current_prod(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((r for r in rows if "prod" in str(r.get("alias", "")).lower()), None)


# ==============================================================================
# Pages
# ==============================================================================

def render_home() -> None:
    rows = safe_registry_rows()
    prod = current_prod(rows)

    hero_visual = (
        "<svg width='230' height='140' viewBox='0 0 260 160' fill='none'>"
        "<rect x='16' y='22' width='160' height='108' rx='18' fill='rgba(255,255,255,.18)' stroke='rgba(255,255,255,.45)'/>"
        "<rect x='35' y='84' width='16' height='30' rx='4' fill='rgba(255,255,255,.72)'/>"
        "<rect x='62' y='64' width='16' height='50' rx='4' fill='rgba(255,255,255,.96)'/>"
        "<rect x='89' y='48' width='16' height='66' rx='4' fill='rgba(255,255,255,.70)'/>"
        "<circle cx='137' cy='72' r='24' fill='none' stroke='rgba(255,255,255,.30)' stroke-width='9'/>"
        "<circle cx='137' cy='72' r='24' fill='none' stroke='rgba(255,255,255,.95)' stroke-width='9' stroke-dasharray='88 70' stroke-linecap='round' transform='rotate(-90 137 72)'/>"
        "<ellipse cx='220' cy='47' rx='26' ry='10' fill='rgba(255,255,255,.80)'/>"
        "<path d='M194 47v56a26 10 0 0 0 52 0V47' fill='rgba(255,255,255,.18)' stroke='rgba(255,255,255,.50)'/>"
        "<ellipse cx='220' cy='75' rx='26' ry='10' stroke='rgba(255,255,255,.50)'/>"
        "<ellipse cx='220' cy='103' rx='26' ry='10' stroke='rgba(255,255,255,.50)'/>"
        "</svg>"
    )

    st.markdown(
        f"""
        <div class='premium-hero'>
          <div class='hero-content'>
            <div class='hero-kicker'>{_svg(IC_ROCKET, 15, 2.2)} Plateforme MLOps</div>
            <div class='hero-title'>Bank Marketing — production, suivi et décision métier.</div>
            <div class='hero-subtitle'>
              Un dashboard propre pour piloter tout le cycle : données, entraînement, registry,
              API de prédiction, évaluation qualité et feedback terrain.
            </div>
          </div>
          <div class='hero-visual'>{hero_visual}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        metric_card(
            "Modèle en production",
            f"v{prod['version']}" if prod else "—",
            IC_PACKAGE,
            PRIMARY,
            "#EEF2FF",
            pill("Actif", PRIMARY, "#EEF2FF", IC_CHECK) if prod else pill("Non assigné", "#6B7280", "#F3F4F6"),
        ),
        metric_card("Versions registry", str(len(rows)), IC_FOLDER, BLUE, "#E0F2FE", pill("MLflow", "#0369A1", "#E0F2FE")),
        metric_card(
            "ROC AUC prod",
            fmt_metric(prod.get("roc_auc") if prod else None),
            IC_TREND,
            GREEN,
            "#ECFDF5",
            quality_pill(prod.get("roc_auc") if prod else None, 0.75),
        ),
        metric_card(
            "F1-score prod",
            fmt_metric(prod.get("f1") if prod else None, "Non loggé"),
            IC_TARGET,
            ORANGE,
            "#FFF7ED",
            quality_pill(prod.get("f1") if prod else None, 0.50, "Non loggé"),
        ),
    ]
    st.markdown(f"<div class='grid-4'>{''.join(cards)}</div>", unsafe_allow_html=True)

    problem = info_card(
        "Problématique métier",
        "Une banque mène des campagnes de <b>marketing téléphonique</b> pour vendre des <b>dépôts à terme</b>. "
        "L’objectif est de prédire, <i>avant l’appel</i>, si un client va souscrire afin de prioriser les prospects "
        "les plus prometteurs et réduire le coût des campagnes.<br><br>"
        "<b>Données :</b> UCI <i>Bank Marketing</i>, environ 45 000 contacts, 16 variables et classes déséquilibrées. "
        "<b>Tâche :</b> classification binaire sur la cible <code>y</code>.",
        IC_TARGET,
        PRIMARY,
    )
    author = (
        "<div class='card' style='height:100%; display:flex; flex-direction:column;'>"
        f"<div class='card-title'><span style='color:{PRIMARY}'>{_svg(IC_USER, 19)}</span>Auteur</div>"
        "<div style='display:flex; align-items:center; gap:.9rem;'>"
        f"<div style='width:56px;height:56px;border-radius:18px;background:linear-gradient(135deg,{PRIMARY},{PURPLE});color:white;display:grid;place-items:center;font-weight:950;font-size:1.25rem;'>{_escape(INITIALS)}</div>"
        "<div>"
        f"<div style='font-weight:950;color:{SLATE};font-size:1.05rem;'>{_escape(AUTHOR)}</div>"
        f"<div style='color:{MUTED};font-size:.84rem;margin-top:.15rem;'>{_escape(AUTHOR_SUBTITLE)}</div>"
        "</div></div>"
        "<div style='height:1px;background:#EEF0F6;margin:1rem 0;'></div>"
        f"<a class='sidebar-link' href='{_escape(GITHUB_URL)}' target='_blank'>{GITHUB_SVG} Voir le code source</a>"
        "</div>"
    )
    st.markdown(f"<div class='grid-2-1'>{problem}{author}</div>", unsafe_allow_html=True)

    section_title("Pipeline de bout en bout", IC_BRANCH)
    steps = [
        ("1", "Données", "UCI Bank Marketing", IC_DATABASE, "#EEF2FF", PRIMARY),
        ("2", "Entraînement", "sklearn · XGBoost · LightGBM", IC_SPARK, "#F3E8FF", PURPLE),
        ("3", "Registry", "MLflow Model Registry", IC_FOLDER, "#E0F2FE", BLUE),
        ("4", "API", "FastAPI /predict", IC_ROCKET, "#FFF7ED", ORANGE),
        ("5", "Dashboard", "Streamlit", IC_TABLE, "#ECFDF5", GREEN),
        ("6", "Orchestration", "Airflow", IC_BRANCH, "#F5F3FF", PRIMARY),
    ]
    pipeline_html = ""
    for num, title, sub, icon, bg, color in steps:
        pipeline_html += (
            "<div class='pipeline-step'>"
            f"<div class='pipeline-num'>{num}</div>"
            f"<div class='pipeline-ico' style='background:{bg};color:{color};'>{_svg(icon, 22)}</div>"
            f"<div class='pipeline-title'>{_escape(title)}</div>"
            f"<div class='pipeline-sub'>{_escape(sub)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='pipeline-grid'>{pipeline_html}</div>", unsafe_allow_html=True)

    section_title("État des services", IC_SHIELD)
    statuses = [
        ("API", "Endpoint de prédiction", ping(f"{API_URL}/health"), IC_ROCKET),
        ("MLflow", "Registry & tracking", ping(f"{MLFLOW_TRACKING_URI}/health"), IC_FOLDER),
        ("Base de données", "Journal des prédictions", ping(f"{API_URL}/predictions?limit=1"), IC_DATABASE),
    ]
    service_html = ""
    for name, subtitle, ok, icon in statuses:
        color = GREEN if ok else RED
        bg = "#ECFDF5" if ok else "#FEF2F2"
        border = "#BBF7D0" if ok else "#FECACA"
        state = "En ligne" if ok else "Hors ligne"
        service_html += (
            f"<div class='service-card' style='background:{bg};border-color:{border};'>"
            "<div class='status-left'>"
            f"<span class='status-dot' style='background:{color};'></span>"
            "<div>"
            f"<div class='service-title'>{_escape(name)}</div>"
            f"<div class='service-sub' style='color:{color};'>{_escape(state)} · {_escape(subtitle)}</div>"
            "</div></div>"
            f"<div style='color:{color};'>{_svg(icon, 22)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='grid-3'>{service_html}</div>", unsafe_allow_html=True)


def render_predict() -> None:
    page_header(
        "Prédiction client",
        "Renseigne le profil d’un client et interroge <code>POST /predict</code> en temps réel. Le résultat est enregistré en base pour suivi et feedback.",
        IC_TARGET,
    )

    with st.form("predict_form"):
        st.markdown(
            f"<div class='grid-3' style='margin-top:0'>"
            f"{info_card('Client & risque', 'Profil client et informations de risque de base.', IC_USER)}"
            f"{info_card('Finance & contact', 'Solde, prêts, canal utilisé et période de contact.', IC_WALLET, ORANGE)}"
            f"{info_card('Campagne', 'Intensité de campagne, historique et résultat précédent.', IC_PHONE, BLUE)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # V6 : répartition 5/5/5 des champs pour supprimer le grand vide visuel
        # observé dans la V4 lorsque la colonne "Campagne" était beaucoup plus longue.
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge", min_value=18, max_value=120, value=39, step=1)
            job = st.selectbox("Métier", CATEGORIES["job"])
            marital = st.selectbox("Statut marital", CATEGORIES["marital"])
            education = st.selectbox("Éducation", CATEGORIES["education"])
            default = st.selectbox("Défaut de paiement", CATEGORIES["default"])
        with c2:
            balance = st.number_input("Solde bancaire", value=448, step=1)
            housing = st.selectbox("Prêt immobilier", CATEGORIES["housing"])
            loan = st.selectbox("Prêt personnel", CATEGORIES["loan"])
            contact = st.selectbox("Canal de contact", CATEGORIES["contact"])
            month = st.selectbox("Mois", CATEGORIES["month"])
        with c3:
            day = st.number_input("Jour du mois", min_value=1, max_value=31, value=16, step=1)
            campaign = st.number_input("Nombre d’appels campagne", min_value=1, value=2, step=1)
            pdays = st.number_input("Jours depuis dernier contact (-1 = jamais)", min_value=-1, value=-1, step=1)
            previous = st.number_input("Contacts antérieurs", min_value=0, value=0, step=1)
            poutcome = st.selectbox("Résultat campagne précédente", CATEGORIES["poutcome"])

        threshold = st.slider(
            "Seuil métier de priorisation",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            help="Ce seuil ne change pas la prédiction du modèle : il sert à décider à partir de quelle probabilité un prospect devient prioritaire métier.",
        )

        submitted = st.form_submit_button("Lancer la prédiction", type="primary", use_container_width=True)

    if not submitted:
        st.markdown(
            f"<div class='card' style='margin-top:1rem;'><div class='card-title'><span style='color:{PRIMARY}'>{_svg(IC_ALERT)}</span>Lecture du score</div>"
            "<div class='card-text'>Plus la probabilité est élevée, plus le prospect mérite d’être priorisé dans la campagne. "
            "Le seuil métier sert à transformer le score en décision opérationnelle selon le budget d’appel, le coût commercial et l’objectif de conversion.</div></div>",
            unsafe_allow_html=True,
        )
        return

    payload = {
        "age": int(age),
        "balance": int(balance),
        "day": int(day),
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
    }

    try:
        response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        response.raise_for_status()
        prediction = response.json()
    except httpx.HTTPError as exc:
        st.error(f"Appel à l’API impossible : {exc}. Vérifie que l’API est démarrée.")
        return

    proba = float(prediction.get("probability", 0.0))
    pred_label = int(prediction.get("prediction", 0))
    label = "Souscrit" if pred_label == 1 else "Ne souscrit pas"
    color = GREEN if pred_label == 1 else ORANGE
    decision = "Prospect prioritaire" if proba >= threshold else "Prospect non prioritaire"

    try:
        served = httpx.get(f"{API_URL}/model-info", timeout=5.0).json().get("version", "?")
    except Exception:
        served = "?"
    if served in ("?", "unknown", None, ""):
        prod_row = current_prod(safe_registry_rows())
        served = prod_row.get("version", "?") if prod_row else "?"
    served_display = f"v{served}" if served not in ("?", "unknown", None, "") else "API non exposée"

    result_html = (
        "<div class='result-panel'>"
        f"<div class='card-title'><span style='color:{color}'>{_svg(IC_TARGET)}</span>Résultat de prédiction</div>"
        "<div class='grid-4' style='margin:.6rem 0 0 0'>"
        f"{metric_card('Décision modèle', label, IC_TARGET, color, '#ECFDF5' if pred_label == 1 else '#FFF7ED', pill(decision, color, '#ECFDF5' if pred_label == 1 else '#FFF7ED'))}"
        f"{metric_card('Probabilité', f'{proba:.1%}', IC_TREND, PRIMARY, '#EEF2FF')}"
        f"{metric_card('Seuil métier', f'{threshold:.0%}', IC_SHIELD, PURPLE, '#F3E8FF')}"
        f"{metric_card('Modèle servi', served_display, IC_PACKAGE, BLUE, '#E0F2FE')}"
        "</div>"
        f"<div class='probability-bar'><div class='probability-fill' style='width:{min(max(proba, 0), 1) * 100:.1f}%;'></div></div>"
        f"<div class='card-text' style='margin-top:.65rem;'>Prédiction enregistrée en base — id : <code>{_escape(prediction.get('id', ''))}</code></div>"
        "</div>"
    )
    st.markdown(result_html, unsafe_allow_html=True)
    st.toast("Prédiction effectuée", icon="✅")

    with st.expander("Voir le payload envoyé et la réponse API"):
        rq_col, rp_col = st.columns(2)
        with rq_col:
            st.markdown("**Payload envoyé**")
            st.json(payload)
        with rp_col:
            st.markdown("**Réponse API**")
            st.json(prediction)


def render_tracking() -> None:
    page_header(
        "Suivi du modèle",
        f"Consulte les versions du modèle <code>{_escape(MODEL_NAME)}</code>, leurs alias MLflow et leurs métriques principales.",
        IC_FOLDER,
    )

    try:
        rows = registry_rows_cached()
    except Exception as exc:
        st.error(f"MLflow est injoignable : {exc}. Démarre-le avec `make mlflow`.")
        return

    if not rows:
        st.info("Aucune version dans le registry. Lance un entraînement avec `make train-optuna` ou `make train-models`.")
        return

    prod = current_prod(rows)
    cards = [
        metric_card("Production", f"v{prod['version']}" if prod else "—", IC_PACKAGE, PRIMARY, "#EEF2FF", pill("Alias prod", PRIMARY, "#EEF2FF") if prod else None),
        metric_card("Versions", str(len(rows)), IC_FOLDER, BLUE, "#E0F2FE"),
        metric_card("ROC AUC prod", fmt_metric(prod.get("roc_auc") if prod else None), IC_TREND, GREEN, "#ECFDF5", quality_pill(prod.get("roc_auc") if prod else None, 0.75)),
        metric_card("F1-score prod", fmt_metric(prod.get("f1") if prod else None, "Non loggé"), IC_TARGET, ORANGE, "#FFF7ED", quality_pill(prod.get("f1") if prod else None, 0.50, "Non loggé")),
    ]
    st.markdown(f"<div class='grid-4'>{''.join(cards)}</div>", unsafe_allow_html=True)

    section_title("Historique des versions", IC_TABLE)
    df = pd.DataFrame(rows)
    if "f1" in df.columns:
        df["f1"] = df["f1"].apply(lambda x: x if _clean_value(x) else "Non loggé")
    for col in ["alias", "model_family", "search_method", "roc_auc", "cv_roc_auc", "run_id"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if _clean_value(x) else "-")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if prod and not _clean_value(prod.get("f1")):
        st.warning("La version en production n’a pas de F1-score loggé dans MLflow. Lance l’évaluation qualité pour le recalculer, puis loggue cette métrique lors du prochain entraînement.")

    section_title("Promouvoir une version", IC_BRANCH)
    st.markdown(
        f"<div class='grid-2' style='margin-top:0'>"
        f"{info_card('Bonne pratique', 'Promouvoir d’abord en <b>staging</b>, vérifier la qualité, puis assigner <b>prod</b> si le modèle est stable.', IC_SHIELD)}"
        f"{info_card('Effet immédiat', 'Changer un alias MLflow peut modifier le modèle servi selon la logique de chargement de ton API.', IC_ALERT, ORANGE)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    pc1, pc2, pc3 = st.columns([2, 2, 1])
    with pc1:
        sel_version = st.selectbox("Version", [r["version"] for r in rows], key="promote_version")
    with pc2:
        sel_alias = st.selectbox("Alias", ["prod", "staging", "dev"], key="promote_alias")
    with pc3:
        st.write("")
        st.write("")
        if st.button("Assigner", type="primary", use_container_width=True):
            MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).set_registered_model_alias(MODEL_NAME, sel_alias, str(sel_version))
            registry_rows_cached.clear()
            st.toast(f"Alias '{sel_alias}' assigné à la version v{sel_version}", icon="✅")
            st.rerun()

    existing_aliases = sorted({a.strip() for r in rows for a in str(r.get("alias", "")).split(",") if a.strip()})
    if existing_aliases:
        section_title("Retirer un alias", IC_ALERT, ORANGE)
        dc1, dc2 = st.columns([3, 1])
        with dc1:
            del_alias = st.selectbox("Alias à retirer", existing_aliases, key="delete_alias")
        with dc2:
            st.write("")
            st.write("")
            if st.button("Retirer", use_container_width=True):
                MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).delete_registered_model_alias(MODEL_NAME, del_alias)
                registry_rows_cached.clear()
                st.toast(f"Alias '{del_alias}' retiré", icon="🗑️")
                st.rerun()


def render_evaluation() -> None:
    page_header(
        "Évaluation qualité",
        f"Lance une évaluation d’une version MLflow et vérifie la porte qualité : ROC AUC ≥ <b>{EVAL_ROC_AUC_MIN}</b> et F1-score ≥ <b>{EVAL_F1_MIN}</b>.",
        IC_SHIELD,
    )

    try:
        rows = registry_rows_cached()
    except Exception as exc:
        st.error(f"MLflow est injoignable : {exc}. Démarre-le avec `make mlflow`.")
        return

    if not rows:
        st.info("Aucune version à évaluer. Lance un entraînement d’abord.")
        return

    st.markdown(
        f"<div class='grid-3' style='margin-top:0'>"
        f"{info_card('Seuil ROC AUC', f'Le modèle doit atteindre au moins <b>{EVAL_ROC_AUC_MIN}</b>.', IC_TREND, GREEN)}"
        f"{info_card('Seuil F1-score', f'Le modèle doit atteindre au moins <b>{EVAL_F1_MIN}</b>.', IC_TARGET, ORANGE)}"
        f"{info_card('Décision', 'La porte qualité accepte uniquement les versions qui passent les deux seuils.', IC_SHIELD, PRIMARY)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        eval_version = st.selectbox("Version à évaluer", [r["version"] for r in rows])
    with c2:
        st.write("")
        st.write("")
        eval_btn = st.button("Évaluer", type="primary", use_container_width=True)

    selected_row = next((r for r in rows if int(r.get("version", -1)) == int(eval_version)), None)

    if not eval_btn:
        # V5 : éviter une page vide avant le clic. On affiche le contexte de la version sélectionnée.
        section_title("Contexte avant évaluation", IC_SHIELD)
        current_f1 = fmt_metric(selected_row.get("f1") if selected_row else None, "À recalculer")
        current_roc = fmt_metric(selected_row.get("roc_auc") if selected_row else None, "À recalculer")
        context_cards = [
            metric_card("Version sélectionnée", f"v{eval_version}", IC_PACKAGE, PRIMARY, "#EEF2FF"),
            metric_card("ROC AUC registry", current_roc, IC_TREND, GREEN, "#ECFDF5"),
            metric_card("F1 registry", current_f1, IC_TARGET, ORANGE, "#FFF7ED", quality_pill(selected_row.get("f1") if selected_row else None, EVAL_F1_MIN, "À recalculer")),
            metric_card("Action attendue", "Courbes + décision", IC_TABLE, BLUE, "#E0F2FE"),
        ]
        st.markdown(f"<div class='grid-4'>{''.join(context_cards)}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='card' style='margin-top:.65rem;'>"
            f"<div class='card-title'><span style='color:{PRIMARY}'>{_svg(IC_CHECK)}</span>Ce que fait l’évaluation</div>"
            "<div class='card-text'>Elle recharge la version MLflow sélectionnée, calcule les métriques sur le jeu de test, "
            "applique la porte qualité et affiche les artefacts utiles comme la courbe ROC et la courbe précision-rappel.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Évaluation du modèle sur le jeu de test..."):
        try:
            result = evaluate_model(model_uri=f"models:/{MODEL_NAME}/{eval_version}", validate=False)
        except Exception as exc:
            st.error(f"Évaluation impossible : {exc}")
            return

    metrics = result.metrics
    f1 = float(metrics.get("f1_score", 0.0))
    roc = float(metrics.get("roc_auc", 0.0))
    precision = float(metrics.get("precision_score", 0.0))
    recall = float(metrics.get("recall_score", 0.0))
    roc_ok = roc >= EVAL_ROC_AUC_MIN
    f1_ok = f1 >= EVAL_F1_MIN
    accepted = roc_ok and f1_ok

    cards = [
        metric_card("ROC AUC", f"{roc:.4f}", IC_TREND, GREEN if roc_ok else RED, "#ECFDF5" if roc_ok else "#FEF2F2", pill("OK" if roc_ok else "KO", GREEN if roc_ok else RED, "#DCFCE7" if roc_ok else "#FEE2E2", IC_CHECK if roc_ok else IC_ALERT)),
        metric_card("F1-score", f"{f1:.4f}", IC_TARGET, GREEN if f1_ok else RED, "#ECFDF5" if f1_ok else "#FEF2F2", pill("OK" if f1_ok else "KO", GREEN if f1_ok else RED, "#DCFCE7" if f1_ok else "#FEE2E2", IC_CHECK if f1_ok else IC_ALERT)),
        metric_card("Précision", f"{precision:.4f}", IC_CHECK, BLUE, "#E0F2FE"),
        metric_card("Rappel", f"{recall:.4f}", IC_ALERT, PURPLE, "#F3E8FF"),
    ]
    st.markdown(f"<div class='grid-4'>{''.join(cards)}</div>", unsafe_allow_html=True)

    if accepted:
        st.success("Porte qualité ACCEPTÉE — la version respecte les seuils définis.")
    else:
        st.error("Porte qualité REJETÉE — une ou plusieurs métriques sont sous le seuil.")

    images = [(name, art.content) for name, art in result.artifacts.items() if type(art.content).__module__.startswith("PIL")]
    if images:
        section_title("Artefacts d’évaluation", IC_TABLE)

        def _is_primary_artifact(name: str) -> bool:
            lname = name.lower()
            return "roc" in lname or "precision_recall" in lname or "precision-recall" in lname

        primary_images = [(n, img) for n, img in images if _is_primary_artifact(n)]
        other_images = [(n, img) for n, img in images if not _is_primary_artifact(n)]

        if not primary_images:
            primary_images, other_images = images[:2], images[2:]

        st.markdown(
            f"<div class='card' style='margin-bottom:.65rem;'>"
            f"<div class='card-title'><span style='color:{PRIMARY}'>{_svg(IC_TREND)}</span>Courbes principales</div>"
            "<div class='card-text'>Les courbes principales permettent de juger rapidement la séparation des classes et la performance sur la classe positive.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        acols = st.columns(2)
        for i, (name, img) in enumerate(primary_images):
            acols[i % 2].image(img, caption=name, use_container_width=True)

        if other_images:
            with st.expander("Voir les artefacts complémentaires"):
                extra_cols = st.columns(2)
                for i, (name, img) in enumerate(other_images):
                    extra_cols[i % 2].image(img, caption=name, use_container_width=True)


def render_history() -> None:
    page_header(
        "Prévisions & feedback",
        "Consulte le journal des prédictions stockées en base, analyse les sorties récentes et renseigne la vérité terrain.",
        IC_TABLE,
    )

    try:
        resp = httpx.get(f"{API_URL}/predictions", params={"limit": 100}, timeout=10.0)
        resp.raise_for_status()
        journal = resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Journal indisponible : {exc}. Vérifie que l’API et la base de données sont démarrées.")
        return

    if not journal:
        st.info("Aucune prévision enregistrée. Va dans la page Prédiction pour créer une première prédiction.")
        return

    df = pd.DataFrame(journal)
    total = len(df)
    positive = int((df.get("prediction", pd.Series(dtype=int)) == 1).sum()) if "prediction" in df else 0
    avg_proba = float(pd.to_numeric(df.get("probability", pd.Series(dtype=float)), errors="coerce").mean()) if "probability" in df else 0.0
    feedback_count = int(df["actual"].notna().sum()) if "actual" in df else 0

    cards = [
        metric_card("Prévisions", str(total), IC_TABLE, PRIMARY, "#EEF2FF"),
        metric_card("Prédictions positives", str(positive), IC_TARGET, ORANGE, "#FFF7ED"),
        metric_card("Probabilité moyenne", f"{avg_proba:.1%}", IC_TREND, GREEN, "#ECFDF5"),
        metric_card("Feedbacks", str(feedback_count), IC_CHECK, BLUE, "#E0F2FE"),
    ]
    st.markdown(f"<div class='grid-4'>{''.join(cards)}</div>", unsafe_allow_html=True)

    section_title("Historique des prédictions", IC_TABLE)

    # V5 : tableau orienté métier plutôt que dataframe brut.
    work_df = df.copy()
    if "created_at" in work_df.columns:
        work_df["created_at_sort"] = pd.to_datetime(work_df["created_at"], errors="coerce")
        work_df = work_df.sort_values("created_at_sort", ascending=False, na_position="last")

    pred_num = pd.to_numeric(work_df.get("prediction", pd.Series(index=work_df.index, dtype=float)), errors="coerce")
    actual_num = pd.to_numeric(work_df.get("actual", pd.Series(index=work_df.index, dtype=float)), errors="coerce")
    proba_num = pd.to_numeric(work_df.get("probability", pd.Series(index=work_df.index, dtype=float)), errors="coerce")

    work_df["Prédiction"] = pred_num.map({1: "Souscrit", 0: "Non souscrit"}).fillna("—")
    work_df["Résultat réel"] = actual_num.map({1: "Souscrit", 0: "Non souscrit"}).fillna("—")
    work_df["Probabilité"] = proba_num.map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    if "id" in work_df.columns:
        work_df["ID court"] = work_df["id"].astype(str).str.slice(0, 10) + "…"

    fh1, fh2, fh3 = st.columns([1, 1, 1])
    with fh1:
        pred_filter = st.selectbox("Filtrer prédiction", ["Toutes", "Souscrit", "Non souscrit"], key="history_pred_filter")
    with fh2:
        feedback_filter = st.selectbox("Filtrer feedback", ["Tous", "Avec feedback", "Sans feedback"], key="history_feedback_filter")
    with fh3:
        row_limit = st.selectbox("Lignes affichées", [10, 20, 50, 100], index=1, key="history_limit")

    if pred_filter != "Toutes":
        work_df = work_df[work_df["Prédiction"] == pred_filter]
    if feedback_filter == "Avec feedback":
        work_df = work_df[actual_num.notna()]
    elif feedback_filter == "Sans feedback":
        work_df = work_df[actual_num.isna()]

    rename_map = {
        "created_at": "Date",
        "model_version": "Modèle",
        "age": "Âge",
        "job": "Métier",
        "contact": "Contact",
        "month": "Mois",
        "balance": "Solde",
        "loan": "Prêt perso",
        "housing": "Prêt immo",
        "pdays": "Jours depuis contact",
        "campaign": "Nb appels",
        "default": "Défaut paiement",
    }
    work_df = work_df.rename(columns=rename_map)
    # V6 : vue principale volontairement épurée. Les variables détaillées restent disponibles
    # dans le journal brut complet juste en dessous.
    preferred_cols = [
        "Date", "Prédiction", "Probabilité", "Résultat réel", "Modèle", "ID court",
        "Âge", "Métier", "Contact", "Mois",
    ]
    display_cols = [c for c in preferred_cols if c in work_df.columns]
    visible_df = work_df[display_cols].head(int(row_limit))
    st.caption(f"{len(work_df)} ligne(s) après filtres — {len(visible_df)} affichée(s).")
    st.dataframe(visible_df, use_container_width=True, hide_index=True, height=320)
    csv_export = work_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger la vue filtrée CSV",
        data=csv_export,
        file_name="predictions_filtrees.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Voir le journal brut complet"):
        st.dataframe(df, use_container_width=True, hide_index=True, height=340)

    section_title("Enregistrer un feedback", IC_CHECK, GREEN)
    st.markdown(
        f"<div class='grid-2' style='margin-top:0'>"
        f"{info_card('Vérité terrain', 'Le feedback permet de comparer la prédiction au résultat réel et d’alimenter les futures analyses de dérive.', IC_DATABASE, GREEN)}"
        f"{info_card('Usage MLOps', 'Ces retours peuvent servir à déclencher un réentraînement ou à ajuster le seuil métier.', IC_BRANCH, PRIMARY)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    options = {
        row["id"]: f"{str(row['id'])[:8]}…  pred={row.get('prediction')}  p={row.get('probability')}"
        for row in journal
        if "id" in row
    }
    if not options:
        st.warning("Les prédictions retournées ne contiennent pas d’identifiant exploitable pour le feedback.")
        return

    fc1, fc2, fc3 = st.columns([2.5, 1.5, 1])
    with fc1:
        fb_id = st.selectbox("Prédiction à annoter", list(options), format_func=options.get)
    with fc2:
        fb_actual = st.selectbox("Résultat réel", [1, 0], format_func=lambda x: "Souscrit" if x == 1 else "Non souscrit")
    with fc3:
        st.write("")
        st.write("")
        if st.button("Envoyer", type="primary", use_container_width=True):
            try:
                r = httpx.post(f"{API_URL}/feedback", json={"prediction_id": fb_id, "actual": fb_actual}, timeout=10.0)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(f"Feedback impossible : {exc}")
            else:
                st.toast("Feedback enregistré", icon="✅")
                st.rerun()


# ==============================================================================
# Sidebar + navigation
# ==============================================================================

PAGES = {
    "Accueil": render_home,
    "Prédiction": render_predict,
    "Suivi du modèle": render_tracking,
    "Évaluation": render_evaluation,
    "Prévisions": render_history,
}


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div class='brand-card'>
              <div class='brand-icon'>{_svg(IC_HOME, 27)}</div>
              <div class='brand-title'>Bank Marketing</div>
              <div class='brand-subtitle'>MLOps Control Center</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=list(PAGES.keys()),
            icons=["house", "bullseye", "diagram-3", "shield-check", "table"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": PRIMARY, "font-size": "1rem"},
                "nav-link": {
                    "font-size": "0.91rem",
                    "text-align": "left",
                    "padding": "0.62rem 0.82rem",
                    "color": "#374151",
                    "font-weight": "650",
                },
                "nav-link-selected": {
                    "background": f"linear-gradient(135deg, {PRIMARY} 0%, {PURPLE} 100%)",
                    "color": "white",
                    "font-weight": "800",
                },
            },
        )

        st.markdown("<div class='sidebar-section-title'>Ressources</div>", unsafe_allow_html=True)
        st.markdown(
            f"<a class='sidebar-link' href='{_escape(GITHUB_URL)}' target='_blank'>{GITHUB_SVG} Code source</a>",
            unsafe_allow_html=True,
        )
        if API_PUBLIC_URL:
            st.link_button("API Swagger", f"{API_PUBLIC_URL}/docs", use_container_width=True)
            st.link_button("API ReDoc", f"{API_PUBLIC_URL}/redoc", use_container_width=True)
        if MLFLOW_UI_URL:
            st.link_button("MLflow Registry", MLFLOW_UI_URL, use_container_width=True)
        if AIRFLOW_UI_URL:
            st.link_button("Airflow", AIRFLOW_UI_URL, use_container_width=True)

        st.markdown(
            f"""
            <div class='mini-profile'>
              <div style='display:flex; gap:.75rem; align-items:center;'>
                <div class='mini-avatar'>{_escape(INITIALS)}</div>
                <div>
                  <div style='font-weight:900;color:{SLATE};font-size:.9rem;'>{_escape(AUTHOR)}</div>
                  <div style='color:{MUTED};font-size:.72rem;line-height:1.35;'>{_escape(AUTHOR_SUBTITLE)}</div>
                </div>
              </div>
            </div>
            <div style='text-align:center;margin-top:1rem;color:#9CA3AF;font-size:.7rem;'>© 2026 Bank Marketing MLOps</div>
            """,
            unsafe_allow_html=True,
        )

    return selected


inject_css()
selected_page = render_sidebar()
PAGES[selected_page]()