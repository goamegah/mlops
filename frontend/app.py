"""Frontend Streamlit : dashboard MLOps Bank Marketing.

Seance 14 bis - TP Streamlit (interface modernisee : menu lateral + pages).
    - Accueil          : problematique, pipeline, etat des services.
    - Prediction       : formulaire -> endpoint /predict de l'API.
    - Suivi du modele  : Model Registry MLflow (versions, alias, prod).
    - Evaluation       : porte qualite + metriques + artefacts.
    - Previsions       : journal des predictions (base) + feedback.
    Lancement : `make frontend` (avec `make api`, `make mlflow`, `make db`).
"""

from __future__ import annotations

import os

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

st.set_page_config(
    page_title="Bank Marketing — MLOps",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Configuration : URLs publiques (navigateur) via .env, liens, branding ---
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "").rstrip("/")
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", "").rstrip("/")
AIRFLOW_UI_URL = os.environ.get("AIRFLOW_UI_URL", "").rstrip("/")
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/goamegah/mlops")
AUTHOR = "Godwin Amegah"
INITIALS = "".join(part[0] for part in AUTHOR.split()[:2]).upper()
ACCENT = "#4F46E5"

# Logo GitHub officiel (Octocat) en SVG inline ; `currentColor` -> suit la couleur du texte.
GITHUB_SVG = (
    "<svg width='15' height='15' viewBox='0 0 16 16' fill='currentColor'"
    " style='vertical-align:-2px;'><path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53"
    " 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09"
    "-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87"
    " 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2"
    "-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04"
    " 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65"
    " 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0"
    " 0016 8c0-4.42-3.58-8-8-8z'/></svg>"
)


def _svg(body: str, size: int = 18, sw: float = 1.9) -> str:
    """Icone au trait (style Lucide) ; `currentColor` -> suit la couleur du parent."""
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none'"
        f" stroke='currentColor' stroke-width='{sw}' stroke-linecap='round'"
        f" stroke-linejoin='round' style='vertical-align:-3px;'>{body}</svg>"
    )


# Corps de chemins (icones sobres, monochromes) reutilises dans tout le dashboard.
IC_ROCKET = (
    "<path d='M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0"
    " 0 0-2.91-.09z'/><path d='m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72"
    "-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z'/><path d='M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5"
    " 0'/><path d='M15 9c0 .55 3.03.55 4 2 1.08 1.62 0 5 0 5'/>"
)
IC_TARGET = (
    "<circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/>"
    "<circle cx='12' cy='12' r='2'/>"
)
IC_USER = "<path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/><circle cx='12' cy='7' r='4'/>"
IC_BRANCH = (
    "<line x1='6' y1='3' x2='6' y2='15'/><circle cx='18' cy='6' r='3'/>"
    "<circle cx='6' cy='18' r='3'/><path d='M18 9a9 9 0 0 1-9 9'/>"
)
IC_SHIELD = "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/><path d='m9 12 2 2 4-4'/>"
IC_PACKAGE = (
    "<path d='M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1"
    " 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z'/>"
    "<polyline points='3.27 6.96 12 12.01 20.73 6.96'/><line x1='12' y1='22.08' x2='12' y2='12'/>"
)
IC_FOLDER = (
    "<path d='M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0"
    " 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2z'/>"
)
IC_TREND = "<polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/>"
IC_WALLET = (
    "<circle cx='12' cy='11' r='8'/><path d='M7 11H4a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-7"
    " a2 2 0 0 0-2-2h-3'/>"
)
IC_PHONE = (
    "<path d='M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79"
    " 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0"
    " 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2"
    " 2 0 0 1 22 16.92z'/>"
)
IC_DATABASE = (
    "<ellipse cx='12' cy='5' rx='9' ry='3'/><path d='M21 12c0 1.66-4 3-9 3s-9-1.34-9-3m0-7v6c0"
    " 1.66 4 3 9 3s9-1.34 9-3V5'/><line x1='3' y1='12' x2='3' y2='19'/><line x1='21' y1='12' x2='21'"
    " y2='19'/><ellipse cx='12' cy='19' rx='9' ry='3'/>"
)
IC_CHECK = "<polyline points='20 6 9 17 4 12'/>"
IC_ALERT = (
    "<path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3.05h16.94a2 2 0 0 0 1.71-3.05l-8.47-14.14a2"
    " 2 0 0 0-3.42 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/>"
)

api_url = DEFAULT_API_URL

# MLflow : echouer vite si le serveur de tracking est injoignable, au lieu de
# la tempete de retries par defaut (7 essais + backoff) qui fige l'UI quand
# MLflow est down (ex. en local ou lors d'un hoquet reseau).
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")

# Valeurs possibles des variables categorielles (dataset Bank Marketing).
CATEGORIES = {
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

# --- CSS global : design system a classes (cartes, hover, hierarchie) ---
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      #MainMenu, footer {visibility: hidden;}
      .stApp {background: linear-gradient(180deg, #F7F8FF 0%, #F3F5FB 100%);}
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; padding-left: 1.5rem; padding-right: 1.5rem;}
      h1, h2, h3, h4 {letter-spacing: -0.02em; color: #25263A;}

      /* Sidebar */
      [data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #EEF0F6;}
      .nav-link.active {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        box-shadow: 0 4px 12px rgba(79,70,229,0.35);
      }
      [data-testid="stSidebar"] .stLinkButton>a, [data-testid="stSidebar"] .stButton>button {
        border-radius: 10px; font-size: 0.85rem; justify-content: flex-start;
        border: 1px solid #E5E7EB; transition: 0.18s;
      }
      [data-testid="stSidebar"] .stLinkButton>a:hover,
      [data-testid="stSidebar"] .stButton>button:hover {
        border-color: #4F46E5; color: #4F46E5; box-shadow: 0 6px 16px rgba(79,70,229,0.12);
      }
      .stButton>button {border-radius: 11px; font-weight: 600; transition: 0.18s; background-color: #4F46E5 !important; color: white !important; border: none !important;}
      .stButton>button:hover {background-color: #4338CA !important; box-shadow: 0 8px 20px rgba(79,70,229,0.25) !important;}

      /* Hero */
      .hero-card {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        border-radius: 24px; padding: 2.8rem 2.8rem; color: #fff; margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(79,70,229,0.12), 0 24px 56px rgba(79,70,229,0.20);
        min-height: 165px;
      }
      .hero-badge {
        min-width: 56px; width: 56px; height: 56px; border-radius: 15px;
        background: rgba(255,255,255,0.18); display: flex; align-items: center;
        justify-content: center; font-size: 1.7rem;
      }
      .hero-title {font-size: 1.95rem; font-weight: 800; line-height: 1.12;}
      .hero-subtitle {opacity: 0.92; margin-top: 0.45rem; font-size: 1rem; line-height: 1.55;}

      /* Rangees flex */
      .row {display: flex; gap: 16px; margin-bottom: 18px;}
      .row-stretch {display: flex; gap: 18px; align-items: stretch; margin-bottom: 8px;}

      /* KPI */
      .kpi-card {
        flex: 1; background: #fff; border: 1px solid #EDF0F7; border-radius: 18px;
        padding: 1.1rem 1.2rem; box-shadow: 0 10px 26px rgba(31,41,55,0.06);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
      }
      .kpi-card:hover {transform: translateY(-3px); box-shadow: 0 18px 36px rgba(31,41,55,0.11);}
      .kpi-head {display: flex; align-items: center; gap: 0.8rem;}
      .kpi-icon {
        min-width: 46px; width: 46px; height: 46px; border-radius: 13px;
        display: flex; align-items: center; justify-content: center; font-size: 1.35rem;
      }
      .kpi-label {font-size: 0.74rem; color: #8A8FA3; font-weight: 600;}
      .kpi-value {font-size: 1.6rem; font-weight: 800; line-height: 1.1;}
      .kpi-pill {font-size: 0.7rem; font-weight: 600; padding: 0.18rem 0.65rem; border-radius: 999px;}

      /* Sections */
      .section-card {
        background: #fff; border: 1px solid #E8EBF3; border-radius: 18px;
        padding: 1.4rem 1.6rem; box-shadow: 0 10px 26px rgba(31,41,55,0.05);
      }
      .section-title {
        font-size: 1.1rem; font-weight: 700; color: #2D3042; margin-bottom: 0.8rem;
        display: flex; align-items: center; gap: 0.5rem;
      }
      .section-text {color: #4B5063; font-size: 0.92rem; line-height: 1.65;}
      /* Titre de bloc hors carte (pipeline, services) */
      .block-title {
        font-size: 1.15rem; font-weight: 700; color: #25263A;
        margin: 1.6rem 0 0.9rem 0; display: flex; align-items: center; gap: 0.5rem;
      }

      /* Pipeline */
      .pipeline-card {
        flex: 1; background: #fff; border: 1px solid #E8EBF3; border-radius: 18px;
        padding: 1rem 0.5rem; text-align: center; box-shadow: 0 8px 22px rgba(31,41,55,0.05);
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
      }
      .pipeline-card:hover {
        transform: translateY(-4px); border-color: #C7D2FE;
        box-shadow: 0 18px 34px rgba(79,70,229,0.13);
      }
      .pipeline-index {
        width: 26px; height: 26px; margin: 0 auto; border-radius: 999px; background: #EEF2FF;
        color: #4F46E5; display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.72rem;
      }
      .pipeline-icon {font-size: 1.7rem; margin-top: 0.35rem;}
      .pipeline-title {font-weight: 600; margin-top: 0.2rem; font-size: 0.9rem; color: #303244;}
      .pipeline-sub {color: #A0A5B8; font-size: 0.7rem; margin-top: 0.1rem;}
      .pipeline-conn {width: 30px; align-self: center; border-top: 2px dashed #C7D2FE;}

      /* Services */
      .service-card {
        flex: 1; border-radius: 16px; padding: 0.9rem 1.1rem;
        display: flex; align-items: center; justify-content: space-between;
      }
      .service-name {font-weight: 700; color: #1F2937;}
      .service-status {font-size: 0.76rem; font-weight: 600; margin-top: 0.1rem;}
      .status-dot {height: 9px; width: 9px; border-radius: 999px; display: inline-block; margin-right: 8px;}

      /* Auteur */
      .author-avatar {
        min-width: 56px; width: 56px; height: 56px; border-radius: 50%;
        background: linear-gradient(135deg, #4F46E5, #7C3AED); color: #fff;
        display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.1rem;
      }
      .author-btn {
        margin-top: auto; display: flex; align-items: center; justify-content: center; gap: 0.5rem;
        background: #F7F8FB; border: 1px solid #EEF0F4; border-radius: 10px; padding: 0.6rem;
        color: #374151 !important; text-decoration: none !important;
        font-size: 0.85rem; font-weight: 600; transition: 0.18s;
      }
      .author-btn:hover {
        border-color: #4F46E5; color: #4F46E5 !important; box-shadow: 0 6px 16px rgba(79,70,229,0.12);
      }
      /* Lien GitHub de la sidebar (HTML custom pour porter le logo Octocat) */
      .sb-link {
        display: flex; align-items: center; gap: 0.5rem; width: 100%; box-sizing: border-box;
        background: #fff; border: 1px solid #E5E7EB; border-radius: 10px;
        padding: 0.45rem 0.75rem; margin-bottom: 0.5rem;
        color: #374151 !important; text-decoration: none !important;
        font-size: 0.85rem; font-weight: 600; transition: 0.18s;
      }
      .sb-link:hover {
        border-color: #4F46E5; color: #4F46E5 !important; box-shadow: 0 6px 16px rgba(79,70,229,0.12);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=10, show_spinner=False)
def _ping(url: str) -> bool:
    """True si l'URL repond 200 (etat d'un service ; resultat mis en cache 10 s)."""
    try:
        return httpx.get(url, timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def _load_registry_rows() -> list[dict]:
    """Lire les versions du Model Registry, leurs tags et leurs alias.

    Les alias sont une propriete du *registered model* (et NON des objets
    ModelVersion renvoyes par search_model_versions, ou `aliases` reste vide) :
    on les recupere via get_registered_model puis on les rattache a chaque version.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    try:
        registered = client.get_registered_model(MODEL_NAME)
    except RestException:
        # MLflow est joignable mais aucun modele n'est encore enregistre.
        return []
    aliases_by_version: dict[int, list[str]] = {}
    for alias, version in dict(registered.aliases).items():
        aliases_by_version.setdefault(int(version), []).append(alias)

    rows = []
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        tags = dict(v.tags or {})
        rows.append(
            {
                "version": int(v.version),
                "alias": ", ".join(aliases_by_version.get(int(v.version), [])),
                "model_family": tags.get("model_family", ""),
                "search_method": tags.get("search_method", ""),
                "f1": tags.get("f1", ""),
                "roc_auc": tags.get("roc_auc", tags.get("test_roc_auc", "")),
                "cv_roc_auc": tags.get("cv_roc_auc", ""),
            }
        )
    rows.sort(key=lambda r: r["version"], reverse=True)
    return rows


# ==============================================================================
# Pages
# ==============================================================================


def _fmt_metric(value: str | None) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _render_page_header(title: str, description: str, icon_body: str) -> None:
    """Affiche un header premium & cohérent pour chaque page du menu."""
    html = (
        f"<div style='background:linear-gradient(135deg, #F7F8FF 0%, #EEF2FF 100%); "
        f"border:1px solid #E5E7EB; border-radius:18px; padding:2rem 2.2rem; margin-bottom:2rem;'>"
        f"<div style='display:flex; align-items:flex-start; gap:1.2rem;'>"
        f"<div style='min-width:56px; width:56px; height:56px; border-radius:14px; "
        f"background:#fff; border:1.5px solid #E5E7EB; display:flex; align-items:center; "
        f"justify-content:center; color:{ACCENT};'>{_svg(icon_body, 26)}</div>"
        f"<div style='flex:1;'>"
        f"<div style='font-size:1.35rem; font-weight:700; color:#25263A; line-height:1.25;'>"
        f"{title}</div>"
        f"<div style='color:#6B7280; font-size:0.95rem; margin-top:0.5rem; line-height:1.5;'>"
        f"{description}</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    _accessible_links_card()
    st.write("")  # respiration


def _accessible_links_card() -> None:
    """Cartes d'accès aux services externes (MLflow, Airflow, Swagger)."""
    links = []
    if MLFLOW_UI_URL:
        links.append(("📊 MLflow Registry", MLFLOW_UI_URL, "Suivi des modèles et versions"))
    if AIRFLOW_UI_URL:
        links.append(("🌀 Airflow Orchestration", AIRFLOW_UI_URL, "Orchestration des pipelines"))
    if API_PUBLIC_URL:
        links.append(("📘 API Swagger", f"{API_PUBLIC_URL}/docs", "Documentation interactive de l'API"))

    if links:
        st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown("**🔗 Accès rapide aux services**")
        cols = st.columns(len(links))
        for col, (title, url, desc) in zip(cols, links):
            with col:
                st.markdown(
                    f"<div style='background:#F7F8FB; border:1px solid #EEF0F4; border-radius:12px;"
                    f" padding:0.9rem; box-shadow:0 2px 8px rgba(0,0,0,0.04);'>"
                    f"<a href='{url}' target='_blank' style='text-decoration:none;'>"
                    f"<div style='font-weight:600; color:#2D3042; margin-bottom:0.3rem;'>{title}</div>"
                    f"<div style='font-size:0.8rem; color:#6B7280;'>{desc}</div>"
                    f"</a></div>",
                    unsafe_allow_html=True,
                )


def _quality_pill(value: str | None, good_threshold: float) -> tuple[str, str, str]:
    """(texte, fond, couleur) d'une pastille de qualite selon un seuil."""
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ("—", "#F3F4F6", "#6B7280")
    if x >= good_threshold:
        return ("Bon", "#DCFCE7", "#16A34A")
    if x >= good_threshold - 0.15:
        return ("Moyen", "#FEF3C7", "#B45309")
    return ("À améliorer", "#FFEDD5", "#C2410C")


def _render_info_card(title: str, description: str, icon_body: str, color: str = ACCENT) -> str:
    """Carte info premium avec titre, description et icône."""
    return (
        f"<div style='background:#fff; border:1px solid #E5E7EB; border-radius:14px; "
        f"padding:1rem; box-shadow:0 4px 12px rgba(0,0,0,0.04);'>"
        f"<div style='display:flex; gap:0.8rem;'>"
        f"<div style='min-width:40px; width:40px; height:40px; border-radius:10px; "
        f"background:{color}15; display:flex; align-items:center; justify-content:center; color:{color};'>"
        f"{_svg(icon_body, 20)}</div>"
        f"<div style='flex:1;'>"
        f"<div style='font-weight:600; color:#111827; font-size:0.95rem;'>{title}</div>"
        f"<div style='font-size:0.85rem; color:#6B7280; margin-top:0.3rem;'>{description}</div>"
        f"</div></div></div>"
    )


def render_home() -> None:
    # --- Hero : bandeau degrade + illustration ---
    hero_svg = (
        "<svg width='245' height='145' viewBox='0 0 245 145' fill='none'>"
        # ecran / dashboard (plus visible)
        "<rect x='6' y='16' width='160' height='110' rx='14' fill='rgba(255,255,255,0.18)'"
        " stroke='rgba(255,255,255,0.50)' stroke-width='1.5'/>"
        "<rect x='24' y='76' width='16' height='34' rx='3' fill='rgba(255,255,255,0.75)'/>"
        "<rect x='48' y='58' width='16' height='52' rx='3' fill='rgba(255,255,255,0.95)'/>"
        "<rect x='72' y='44' width='16' height='66' rx='3' fill='rgba(255,255,255,0.75)'/>"
        # donut (plus visible)
        "<circle cx='128' cy='68' r='22' fill='none' stroke='rgba(255,255,255,0.40)'"
        " stroke-width='8.5'/>"
        "<circle cx='128' cy='68' r='22' fill='none' stroke='rgba(255,255,255,1.0)'"
        " stroke-width='8.5' stroke-dasharray='80 68' stroke-linecap='round'"
        " transform='rotate(-90 128 68)'/>"
        # base de donnees a anneaux (plus visible)
        "<ellipse cx='210' cy='48' rx='26' ry='10' fill='rgba(255,255,255,0.75)'/>"
        "<path d='M184 48 v52 a26 10 0 0 0 52 0 v-52' fill='rgba(255,255,255,0.22)'"
        " stroke='rgba(255,255,255,0.55)' stroke-width='1.5'/>"
        "<ellipse cx='210' cy='73' rx='26' ry='10' fill='none' stroke='rgba(255,255,255,0.55)'"
        " stroke-width='1.5'/>"
        "<ellipse cx='210' cy='98' rx='26' ry='10' fill='none' stroke='rgba(255,255,255,0.55)'"
        " stroke-width='1.5'/>"
        "</svg>"
    )
    st.markdown(
        f"<div class='hero-card'>"
        f"<div style='display:flex; align-items:center; gap:1.1rem; max-width:72%;'>"
        f"<div class='hero-badge'>{_svg(IC_ROCKET, 26)}</div>"
        f"<div><div class='hero-title'>Plateforme MLOps — Bank Marketing</div>"
        f"<div class='hero-subtitle'>De la donnée brute à la prédiction servie en production"
        f" — un pipeline complet, suivi, évalué et orchestré.</div></div></div>"
        f"<div style='position:absolute; right:26px; top:50%; transform:translateY(-50%);"
        f" opacity:0.9;'>{hero_svg}</div></div>",
        unsafe_allow_html=True,
    )

    # --- KPI (depuis le registry, best-effort ; on ne tente pas si MLflow est down) ---
    rows: list[dict] = []
    if _ping(f"{MLFLOW_TRACKING_URI}/health"):
        try:
            rows = _load_registry_rows()
        except Exception:  # noqa: BLE001 - MLflow injoignable / registry vide
            rows = []
    prod = next((r for r in rows if "prod" in r["alias"]), None)
    prod_pill = ("Actif", "#EEF2FF", "#4F46E5") if prod else ("Inactif", "#F3F4F6", "#6B7280")
    kpis = [
        (IC_PACKAGE, "Modèle en production", f"v{prod['version']}" if prod else "—", "#4F46E5",
         "#EEF2FF", prod_pill),
        (IC_FOLDER, "Versions au registry", str(len(rows)), "#0EA5E9", "#E0F2FE",
         ("Registry", "#E0F2FE", "#0369A1")),
        (IC_TREND, "ROC AUC (prod)", _fmt_metric(prod["roc_auc"] if prod else None), "#16A34A",
         "#ECFDF5", _quality_pill(prod["roc_auc"] if prod else None, 0.75)),
        (IC_TARGET, "F1-score (prod)", _fmt_metric(prod["f1"] if prod else None), "#EA580C",
         "#FFF7ED", _quality_pill(prod["f1"] if prod else None, 0.50)),
    ]
    cards = ""
    for icon, label, value, vcolor, ibg, (ptext, pbg, pfg) in kpis:
        cards += (
            "<div class='kpi-card'><div class='kpi-head'>"
            f"<div class='kpi-icon' style='background:{ibg}; color:{vcolor};'>{_svg(icon, 22)}</div>"
            f"<div><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value' style='color:{vcolor};'>{value}</div></div></div>"
            f"<div style='margin-top:0.75rem;'>"
            f"<span class='kpi-pill' style='background:{pbg}; color:{pfg};'>{ptext}</span>"
            "</div></div>"
        )
    st.markdown(f"<div class='row'>{cards}</div>", unsafe_allow_html=True)

    # --- Problematique + Auteur (cartes a hauteur egale) ---
    probleme = (
        "<div class='section-card' style='flex:2;'>"
        f"<div class='section-title'><span style='color:{ACCENT}; display:flex;'>"
        f"{_svg(IC_TARGET)}</span> Problématique</div>"
        "<div class='section-text'>"
        "Une banque mène des campagnes de <b>marketing téléphonique</b> pour placer des "
        "<b>dépôts à terme</b>. L'enjeu : prédire, <i>avant</i> l'appel, si un client va "
        "<b>souscrire</b> — afin de cibler les prospects les plus prometteurs et réduire le "
        "coût des campagnes.</div>"
        "<hr style='border:none; border-top:1px solid #F0F1F4; margin:0.9rem 0;'>"
        "<div class='section-text'>"
        "<b>Données</b> : UCI <i>Bank Marketing</i> — ~45 000 contacts, 16 variables, classes "
        "<b>déséquilibrées</b> (≈ 12 % de souscriptions).<br>"
        "<b>Tâche</b> : classification binaire sur la cible "
        "<code style='background:#F3F4F6; padding:0.05rem 0.3rem; border-radius:5px;'>y</code> "
        "(souscrit / ne souscrit pas).</div></div>"
    )
    auteur = (
        "<div class='section-card' style='flex:1; display:flex; flex-direction:column;'>"
        f"<div class='section-title'><span style='color:{ACCENT}; display:flex;'>"
        f"{_svg(IC_USER)}</span> Auteur</div>"
        "<div style='display:flex; align-items:center; gap:0.9rem;'>"
        f"<div class='author-avatar'>{INITIALS}</div>"
        f"<div><div style='font-weight:700; font-size:1.05rem; color:#111827;'>{AUTHOR}</div>"
        "<div style='color:#6B7280; font-size:0.8rem;'>ESGI · IABD</div>"
        "<div style='color:#9CA3AF; font-size:0.8rem;'>Fil rouge MLOps</div></div></div>"
        "<div style='margin-top:auto;'>"
        "<hr style='border:none; border-top:1px solid #F0F1F4; margin:1rem 0 0.9rem 0;'>"
        f"<a class='author-btn' style='margin-top:0;' href='{GITHUB_URL}' target='_blank'>"
        f"{GITHUB_SVG} Voir le code sur GitHub</a></div></div>"
    )
    st.markdown(f"<div class='row-stretch'>{probleme}{auteur}</div>", unsafe_allow_html=True)

    # --- Pipeline (cartes reliees par des pointilles) ---
    st.markdown(
        f"<div class='block-title'><span style='color:{ACCENT}; display:flex;'>"
        f"{_svg(IC_BRANCH, 20)}</span> Le pipeline de bout en bout</div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("📥", "Données", "UCI Bank Marketing"),
        ("🧠", "Entraînement", "sklearn · XGBoost · LightGBM"),
        ("📊", "Registry", "MLflow"),
        ("⚡", "API", "FastAPI"),
        ("🖥️", "Dashboard", "Streamlit"),
        ("🌀", "Orchestration", "Airflow"),
    ]
    pipe = ""
    for step, (icon, title, sub) in enumerate(steps, start=1):
        if step > 1:
            pipe += "<div class='pipeline-conn'></div>"
        pipe += (
            "<div class='pipeline-card'>"
            f"<div class='pipeline-index'>{step}</div>"
            f"<div class='pipeline-icon'>{icon}</div>"
            f"<div class='pipeline-title'>{title}</div>"
            f"<div class='pipeline-sub'>{sub}</div></div>"
        )
    st.markdown(
        f"<div style='display:flex; align-items:stretch; margin-bottom:8px;'>{pipe}</div>",
        unsafe_allow_html=True,
    )

    # --- Etat des services ---
    st.markdown(
        f"<div class='block-title'><span style='color:{ACCENT}; display:flex;'>"
        f"{_svg(IC_SHIELD, 20)}</span> État des services</div>",
        unsafe_allow_html=True,
    )
    statuses = [
        ("API", _ping(f"{api_url}/health")),
        ("MLflow", _ping(f"{MLFLOW_TRACKING_URI}/health")),
        ("Base de données", _ping(f"{api_url}/predictions?limit=1")),
    ]
    scards = ""
    for label, ok in statuses:
        bg = "#ECFDF5" if ok else "#FEF2F2"
        bd = "#BBF7D0" if ok else "#FECACA"
        fg = "#16A34A" if ok else "#DC2626"
        state = "En ligne" if ok else "Hors ligne"
        pulse = (
            f"<svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='{fg}'"
            " stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<path d='M3 12h4l2 5 4-10 2 5h4'/></svg>"
        )
        scards += (
            f"<div class='service-card' style='background:{bg}; border:1px solid {bd};'>"
            f"<div><span class='status-dot' style='background:{fg};'></span>"
            f"<span class='service-name'>{label}</span>"
            f"<div class='service-status' style='color:{fg};'>{state}</div></div>{pulse}</div>"
        )
    st.markdown(
        f"<div class='row' style='margin-bottom:0;'>{scards}</div>",
        unsafe_allow_html=True,
    )


def render_predict() -> None:
    _render_page_header(
        title="Prédiction — Tester l'endpoint",
        description="Renseigne le profil d'un client et teste l'endpoint `POST /predict` en temps réel.",
        icon_body=IC_TARGET,
    )

    with st.form("predict_form"):
        # Section 1 : Infos personnelles
        st.markdown(
            f"<div style='margin-bottom:1.2rem;'>"
            f"{_render_info_card('Infos personnelles', 'Données démographiques du client', IC_USER)}"
            f"</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("age", min_value=18, max_value=120, value=39, step=1)
            job = st.selectbox("job", CATEGORIES["job"])
        with col2:
            marital = st.selectbox("marital", CATEGORIES["marital"])
            education = st.selectbox("education", CATEGORIES["education"])

        st.divider()

        # Section 2 : Profil financier
        st.markdown(
            f"<div style='margin-bottom:1.2rem;'>"
            f"{_render_info_card('Profil financier', 'Situation économique du client', IC_WALLET)}"
            f"</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            balance = st.number_input("balance (euros, peut être négatif)", value=448, step=1)
            default = st.selectbox("default", CATEGORIES["default"])
        with col2:
            housing = st.selectbox("housing", CATEGORIES["housing"])
            loan = st.selectbox("loan", CATEGORIES["loan"])

        st.divider()

        # Section 3 : Historique & campagne
        st.markdown(
            f"<div style='margin-bottom:1.2rem;'>"
            f"{_render_info_card('Historique & campagne', 'Contacts antérieurs et contexte de l\'appel', IC_PHONE)}"
            f"</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            previous = st.number_input("previous (contacts antérieurs)", min_value=0, value=0, step=1)
            pdays = st.number_input("pdays (-1 = jamais contacté)", min_value=-1, value=-1, step=1)
            day = st.number_input("day (jour du mois)", min_value=1, max_value=31, value=16, step=1)
            campaign = st.number_input("campaign (nombre d'appels)", min_value=1, value=2, step=1)
        with col2:
            month = st.selectbox("month", CATEGORIES["month"])
            contact = st.selectbox("contact", CATEGORIES["contact"])
            poutcome = st.selectbox("poutcome", CATEGORIES["poutcome"])

        st.write("")  # respiration
        submitted = st.form_submit_button("Prédire", type="primary", width="stretch")

    if not submitted:
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
        response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
        response.raise_for_status()
        prediction = response.json()
    except httpx.HTTPError as exc:
        st.error(f"Appel à l'API impossible : {exc} (l'API est-elle démarrée ?)")
        return

    proba = float(prediction["probability"])
    label = "Souscrit (1)" if prediction["prediction"] == 1 else "Ne souscrit pas (0)"
    try:
        served = httpx.get(f"{api_url}/model-info", timeout=5.0).json().get("version", "?")
    except Exception:  # noqa: BLE001 - /model-info indisponible ou reponse inattendue
        served = "?"

    c1, c2, c3 = st.columns(3)
    c1.metric("Prédiction", label)
    c2.metric("Probabilité de souscription", f"{proba:.1%}")
    c3.metric("Modèle servi", f"v{served}" if served not in ("?", "unknown") else served)
    st.progress(proba)
    st.caption(f"Prédiction enregistrée en base — id : `{prediction.get('id', '')}`")
    st.toast("Prédiction effectuée", icon="✅")

    with st.expander("Détail de la requête (payload envoyé / réponse brute)"):
        rq_col, rp_col = st.columns(2)
        with rq_col:
            st.markdown("**Payload envoyé** (`POST /predict`)")
            st.json(payload)
        with rp_col:
            st.markdown("**Réponse brute de l'API**")
            st.json(prediction)


def render_tracking() -> None:
    _render_page_header(
        title="Suivi du modèle — Model Registry",
        description=f"Versions, alias et métriques du modèle `{MODEL_NAME}` dans MLflow.",
        icon_body=IC_FOLDER,
    )

    try:
        rows = _load_registry_rows()
    except Exception as exc:  # noqa: BLE001 - MLflow injoignable / non demarre
        st.error(f"MLflow injoignable ({exc}). Démarre-le avec `make mlflow`.")
        return

    if not rows:
        st.info(
            "Aucune version dans le registry. Lance un entraînement "
            "(`make train-optuna` ou `make train-models`)."
        )
        return

    prod = next((r for r in rows if "prod" in r["alias"]), None)

    # Metrics stylisées
    cards = ""
    metrics = [
        (IC_PACKAGE, "Modèle en production", f"v{prod['version']}" if prod else "—", "#4F46E5", "#EEF2FF"),
        (IC_TREND, "ROC AUC (prod)", _fmt_metric(prod["roc_auc"] if prod else None), "#0EA5E9", "#E0F2FE"),
        (IC_TARGET, "F1-score (prod)", _fmt_metric(prod["f1"] if prod else None), "#EA580C", "#FFF7ED"),
        (IC_FOLDER, "Versions au registry", str(len(rows)), "#8B5CF6", "#F3E8FF"),
    ]
    for icon, label, value, vcolor, ibg in metrics:
        cards += (
            "<div class='kpi-card'><div class='kpi-head'>"
            f"<div class='kpi-icon' style='background:{ibg}; color:{vcolor};'>{_svg(icon, 20)}</div>"
            f"<div><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value' style='color:{vcolor};'>{value}</div></div></div>"
            "</div>"
        )
    st.markdown(f"<div class='row'>{cards}</div>", unsafe_allow_html=True)
    st.write("")

    st.dataframe(pd.DataFrame(rows).replace("", "-"), width="stretch", hide_index=True)

    st.divider()
    st.markdown("**Promouvoir une version (alias)**")
    pc1, pc2 = st.columns(2)
    sel_version = pc1.selectbox("Version", [r["version"] for r in rows])
    sel_alias = pc2.selectbox("Alias", ["prod", "staging", "dev"])
    if st.button("Assigner l'alias", type="primary"):
        MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).set_registered_model_alias(
            MODEL_NAME, sel_alias, str(sel_version)
        )
        st.toast(f"Alias '{sel_alias}' -> v{sel_version}", icon="✅")
        st.rerun()

    existing_aliases = sorted({a.strip() for r in rows for a in r["alias"].split(",") if a.strip()})
    if existing_aliases:
        st.divider()
        st.markdown("**Retirer un alias**")
        del_alias = st.selectbox("Alias à retirer", existing_aliases)
        if st.button("Retirer l'alias"):
            MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).delete_registered_model_alias(
                MODEL_NAME, del_alias
            )
            st.toast(f"Alias '{del_alias}' retiré", icon="🗑️")
            st.rerun()


def render_evaluation() -> None:
    _render_page_header(
        title="Évaluation — Porte qualité",
        description=f"Évalue le modèle avec seuils : roc_auc ≥ {EVAL_ROC_AUC_MIN} | f1 ≥ {EVAL_F1_MIN}.",
        icon_body=IC_SHIELD,
    )

    try:
        eval_rows = _load_registry_rows()
    except Exception as exc:  # noqa: BLE001 - MLflow injoignable / non demarre
        st.error(f"MLflow injoignable ({exc}). Démarre-le avec `make mlflow`.")
        return

    if not eval_rows:
        st.info("Aucune version à évaluer. Lance un entraînement d'abord.")
        return

    # Cartes d'info : seuils qualité
    st.markdown(
        f"<div style='margin-bottom:1.5rem;'>"
        f"{_render_info_card('Seuils qualité', f'ROC AUC ≥ {EVAL_ROC_AUC_MIN} | F1-score ≥ {EVAL_F1_MIN}', IC_ALERT, '#EA580C')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        eval_version = st.selectbox("Version à évaluer", [r["version"] for r in eval_rows])
    with c2:
        st.write("")
        st.write("")
        eval_btn = st.button("Lancer l'évaluation", type="primary", use_container_width=True)

    if not eval_btn:
        return

    with st.spinner("Évaluation du modèle sur le jeu de test..."):
        try:
            result = evaluate_model(
                model_uri=f"models:/{MODEL_NAME}/{eval_version}", validate=False
            )
        except Exception as exc:  # noqa: BLE001 - evaluation impossible
            st.error(f"Évaluation impossible : {exc}")
            return

    st.write("")
    metrics = result.metrics
    f1 = float(metrics["f1_score"])
    roc = float(metrics["roc_auc"])
    roc_ok = roc >= EVAL_ROC_AUC_MIN
    f1_ok = f1 >= EVAL_F1_MIN
    prec = float(metrics.get('precision_score', float('nan')))
    rec = float(metrics.get('recall_score', float('nan')))

    # Metrics stylisées
    cards = ""
    metric_list = [
        (IC_TREND, "ROC AUC", f"{roc:.4f}", "#0EA5E9", "#E0F2FE", roc_ok),
        (IC_TARGET, "F1-score", f"{f1:.4f}", "#EA580C", "#FFF7ED", f1_ok),
        (IC_CHECK, "Précision", f"{prec:.4f}", "#10B981", "#ECFDF5", None),
        (IC_ALERT, "Rappel", f"{rec:.4f}", "#8B5CF6", "#F3E8FF", None),
    ]
    for icon, label, value, vcolor, ibg, ok in metric_list:
        badge = (
            f"<span class='kpi-pill' style='background:#DCFCE7; color:#16A34A;'>✓ OK</span>"
            if ok is True
            else f"<span class='kpi-pill' style='background:#FFEDD5; color:#C2410C;'>✗ KO</span>"
            if ok is False
            else ""
        )
        cards += (
            "<div class='kpi-card'><div class='kpi-head'>"
            f"<div class='kpi-icon' style='background:{ibg}; color:{vcolor};'>{_svg(icon, 20)}</div>"
            f"<div style='flex:1;'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value' style='color:{vcolor};'>{value}</div></div></div>"
            f"<div style='margin-top:0.6rem;'>{badge}</div></div>"
        )
    st.markdown(f"<div class='row'>{cards}</div>", unsafe_allow_html=True)

    st.write("")
    if roc_ok and f1_ok:
        st.success("✓ **Porte qualité : ACCEPTÉ** — Les deux seuils sont atteints.")
    else:
        st.error("✗ **Porte qualité : REJETÉ** — Un ou plusieurs seuils ne sont pas atteints.")

    images = [
        (name, art.content)
        for name, art in result.artifacts.items()
        if type(art.content).__module__.startswith("PIL")
    ]
    if images:
        st.divider()
        st.markdown("**Artefacts d'évaluation**")
        acols = st.columns(2)
        for i, (name, img) in enumerate(images):
            acols[i % 2].image(img, caption=name, width="stretch")


def render_history() -> None:
    _render_page_header(
        title="Journal des prévisions — Historique & feedback",
        description="Historique des prédictions en base et saisie du feedback (vérité terrain).",
        icon_body=IC_PACKAGE,
    )

    try:
        resp = httpx.get(f"{api_url}/predictions", params={"limit": 100}, timeout=10.0)
        resp.raise_for_status()
        journal = resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Journal indisponible : {exc} (API + MySQL démarrés ?)")
        return

    if not journal:
        st.info("Aucune prévision enregistrée. Va dans la page Prédiction pour en créer.")
        return

    # Cartes d'info
    st.markdown(
        f"<div style='margin-bottom:1.5rem;'>"
        f"{_render_info_card('Journal des prédictions', f'{len(journal)} prédictions enregistrées en base de données', IC_DATABASE)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.dataframe(pd.DataFrame(journal), width="stretch", hide_index=True)

    st.write("")
    st.markdown(
        f"<div style='margin-bottom:1.5rem;'>"
        f"{_render_info_card('Enregistrer un feedback', 'Marquez la vérité terrain pour améliorer le monitoring', IC_CHECK, '#10B981')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    options = {
        row["id"]: f"{row['id'][:8]}...  pred={row['prediction']}  p={row['probability']}"
        for row in journal
    }
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    fb_id = fc1.selectbox("Prédiction à annoter", list(options), format_func=options.get)
    fb_actual = fc2.selectbox(
        "Résultat réel", [1, 0], format_func=lambda x: "Souscrit (1)" if x == 1 else "Non (0)"
    )
    if fc3.button("Envoyer le feedback", width="stretch"):
        try:
            r = httpx.post(
                f"{api_url}/feedback",
                json={"prediction_id": fb_id, "actual": fb_actual},
                timeout=10.0,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            st.error(f"Feedback impossible : {exc}")
        else:
            st.toast("Feedback enregistré", icon="✅")
            st.rerun()


# ==============================================================================
# Barre laterale : branding, menu de navigation, ressources, auteur
# ==============================================================================
PAGES = {
    "Accueil": render_home,
    "Prédiction": render_predict,
    "Suivi du modèle": render_tracking,
    "Évaluation": render_evaluation,
    "Prévisions": render_history,
}

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding:0.4rem 0 0.8rem 0;">
          <div style="font-size:2.1rem;">🏦</div>
          <div style="font-weight:700; font-size:1.15rem; color:{ACCENT};">Bank Marketing</div>
          <div style="color:#6B7280; font-size:0.78rem;">Plateforme MLOps</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=list(PAGES.keys()),
        icons=["house", "bullseye", "diagram-3", "clipboard-check", "table"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#FFFFFF"},
            "icon": {"color": ACCENT, "font-size": "1rem"},
            "nav-link": {
                "font-size": "0.92rem",
                "text-align": "left",
                "margin": "3px 0",
                "padding": "0.55rem 0.8rem",
                "border-radius": "10px",
                "--hover-color": "#EEF2FF",
            },
            "nav-link-selected": {
                "background-color": ACCENT,
                "color": "white",
                "font-weight": "600",
            },
        },
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; letter-spacing:0.09em; color:#9CA3AF;"
        " font-weight:700; margin:0.2rem 0 0.5rem 0;'>🔗 RESSOURCES</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<a class='sb-link' href='{GITHUB_URL}' target='_blank'>"
        f"{GITHUB_SVG} Code source — GitHub</a>",
        unsafe_allow_html=True,
    )
    if API_PUBLIC_URL:
        st.link_button("📘 API — Swagger", f"{API_PUBLIC_URL}/docs", width="stretch")
        st.link_button("📕 API — ReDoc", f"{API_PUBLIC_URL}/redoc", width="stretch")
    if MLFLOW_UI_URL:
        st.link_button("📊 MLflow — Registry", MLFLOW_UI_URL, width="stretch")
    if AIRFLOW_UI_URL:
        st.link_button("🌀 Airflow — Orchestration", AIRFLOW_UI_URL, width="stretch")

    st.markdown(
        "<div style='text-align:center; margin-top:0.8rem; color:#9CA3AF; font-size:0.72rem;'>"
        "© 2026 Bank Marketing MLOps</div>",
        unsafe_allow_html=True,
    )

PAGES[selected]()
