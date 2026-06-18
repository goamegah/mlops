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
ACCENT = "#4F46E5"

api_url = DEFAULT_API_URL

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

# --- CSS global : rendu moderne (epure, coins arrondis, accent indigo) ---
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
      [data-testid="stSidebar"] {background-color: #FAFAFC; border-right: 1px solid #EEF0F4;}
      [data-testid="stSidebar"] .stButton>button,
      [data-testid="stSidebar"] .stLinkButton>a {
        border-radius: 10px; font-size: 0.85rem; justify-content: flex-start;
      }
      h1, h2, h3, h4 {letter-spacing: -0.01em;}
      .stButton>button, .stLinkButton>a {border-radius: 10px;}
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


def render_home() -> None:
    st.markdown("## Plateforme MLOps — Bank Marketing")
    st.caption(
        "De la donnée brute à la prédiction servie en production — "
        "un pipeline complet, suivi et orchestré."
    )
    st.write("")

    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### 🎯 Problématique")
            st.markdown(
                "Une banque mène des campagnes de **marketing téléphonique** pour placer des "
                "**dépôts à terme**. L'enjeu : prédire, *avant* l'appel, si un client va "
                "**souscrire** — afin de cibler les prospects les plus prometteurs et réduire "
                "le coût des campagnes."
            )
            st.markdown(
                "**Données** : UCI *Bank Marketing* — ~45 000 contacts, 16 variables, classes "
                "**déséquilibrées** (≈ 12 % de souscriptions). **Tâche** : classification "
                "binaire sur la cible `y` (souscrit / ne souscrit pas)."
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### 👤 Auteur")
            st.markdown(f"### {AUTHOR}")
            st.caption("ESGI · IABD — Fil rouge MLOps")
            st.write("")
            st.link_button("🐙 Voir le code sur GitHub", GITHUB_URL, width="stretch")

    st.write("")
    st.markdown("#### 🧭 Le pipeline de bout en bout")
    steps = [
        ("📥", "Données", "UCI Bank Marketing"),
        ("🧠", "Entraînement", "sklearn · XGBoost · LightGBM"),
        ("📊", "Registry", "MLflow"),
        ("⚡", "API", "FastAPI"),
        ("🖥️", "Dashboard", "Streamlit"),
        ("🌀", "Orchestration", "Airflow"),
    ]
    for col, (icon, title, sub) in zip(st.columns(len(steps)), steps):
        with col, st.container(border=True):
            st.markdown(
                f"<div style='text-align:center; padding:0.3rem 0;'>"
                f"<div style='font-size:1.7rem;'>{icon}</div>"
                f"<div style='font-weight:600; margin-top:0.25rem;'>{title}</div>"
                f"<div style='color:#9CA3AF; font-size:0.72rem; margin-top:0.15rem;'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### ⚙️ État des services")
    statuses = [
        ("API", _ping(f"{api_url}/health")),
        ("MLflow", _ping(f"{MLFLOW_TRACKING_URI}/health")),
        ("Base de données", _ping(f"{api_url}/predictions?limit=1")),
    ]
    for col, (label, ok) in zip(st.columns(len(statuses)), statuses):
        bg = "#ECFDF5" if ok else "#FEF2F2"
        fg = "#059669" if ok else "#DC2626"
        state = "en ligne" if ok else "hors ligne"
        col.markdown(
            f"<div style='background:{bg}; border-radius:12px; padding:0.8rem 1rem;'>"
            f"<span style='color:{fg};'>●</span> <b>{label}</b>"
            f"<div style='color:#6B7280; font-size:0.78rem; margin-top:0.1rem;'>{state}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_predict() -> None:
    st.markdown("## 🎯 Prédiction")
    st.caption("Renseigne le profil d'un client et interroge l'endpoint `POST /predict`.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("age", min_value=18, max_value=120, value=39, step=1)
            balance = st.number_input("balance (euros, peut etre negatif)", value=448, step=1)
            day = st.number_input("day (jour du mois)", min_value=1, max_value=31, value=16, step=1)
            campaign = st.number_input("campaign", min_value=1, value=2, step=1)
            pdays = st.number_input("pdays (-1 = jamais contacte)", min_value=-1, value=-1, step=1)
            previous = st.number_input("previous", min_value=0, value=0, step=1)
            job = st.selectbox("job", CATEGORIES["job"])
            marital = st.selectbox("marital", CATEGORIES["marital"])
        with col2:
            education = st.selectbox("education", CATEGORIES["education"])
            default = st.selectbox("default", CATEGORIES["default"])
            housing = st.selectbox("housing", CATEGORIES["housing"])
            loan = st.selectbox("loan", CATEGORIES["loan"])
            contact = st.selectbox("contact", CATEGORIES["contact"])
            month = st.selectbox("month", CATEGORIES["month"])
            poutcome = st.selectbox("poutcome", CATEGORIES["poutcome"])

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
    st.markdown("## 📊 Suivi du modèle")
    st.caption(f"Model Registry MLflow — modèle `{MODEL_NAME}`.")
    if MLFLOW_UI_URL:
        st.link_button("Ouvrir le Model Registry dans MLflow ↗", f"{MLFLOW_UI_URL}/#/models")

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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle en production", f"v{prod['version']}" if prod else "aucun")
    c2.metric("ROC AUC (prod)", (prod["roc_auc"] or "-") if prod else "-")
    c3.metric("F1-score (prod)", (prod["f1"] or "-") if prod else "-")
    c4.metric("Versions", len(rows))

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
    st.markdown("## ✅ Évaluation & porte qualité")
    st.caption(f"Seuils : roc_auc >= {EVAL_ROC_AUC_MIN}  |  f1_score >= {EVAL_F1_MIN}")

    try:
        eval_rows = _load_registry_rows()
    except Exception as exc:  # noqa: BLE001 - MLflow injoignable / non demarre
        st.error(f"MLflow injoignable ({exc}). Démarre-le avec `make mlflow`.")
        return

    if not eval_rows:
        st.info("Aucune version à évaluer. Lance un entraînement d'abord.")
        return

    eval_version = st.selectbox("Version à évaluer", [r["version"] for r in eval_rows])
    if not st.button("Lancer l'évaluation", type="primary"):
        return

    with st.spinner("Évaluation du modèle sur le jeu de test..."):
        try:
            result = evaluate_model(
                model_uri=f"models:/{MODEL_NAME}/{eval_version}", validate=False
            )
        except Exception as exc:  # noqa: BLE001 - evaluation impossible
            st.error(f"Évaluation impossible : {exc}")
            return

    metrics = result.metrics
    f1 = float(metrics["f1_score"])
    roc = float(metrics["roc_auc"])
    roc_ok = roc >= EVAL_ROC_AUC_MIN
    f1_ok = f1 >= EVAL_F1_MIN

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC", f"{roc:.4f}")
    m2.metric("F1-score", f"{f1:.4f}")
    m3.metric("Précision", f"{float(metrics.get('precision_score', float('nan'))):.4f}")
    m4.metric("Rappel", f"{float(metrics.get('recall_score', float('nan'))):.4f}")

    st.write(f"- roc_auc {roc:.4f} {'>=' if roc_ok else '<'} {EVAL_ROC_AUC_MIN} -> {'OK' if roc_ok else 'KO'}")
    st.write(f"- f1_score {f1:.4f} {'>=' if f1_ok else '<'} {EVAL_F1_MIN} -> {'OK' if f1_ok else 'KO'}")
    if roc_ok and f1_ok:
        st.success("Porte qualité : ACCEPTÉ (les deux seuils sont atteints).")
    else:
        st.error("Porte qualité : REJETÉ (un seuil n'est pas atteint).")

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
    st.markdown("## 🗂️ Journal des prévisions")
    st.caption("Historique des prédictions enregistrées en base, et saisie du feedback réel.")

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

    st.dataframe(pd.DataFrame(journal), width="stretch", hide_index=True)

    st.divider()
    st.markdown("**Enregistrer un feedback (vérité terrain)**")
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
            "container": {"padding": "0", "background-color": "transparent"},
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
    st.markdown("##### 🔗 Ressources")
    st.link_button("🐙 Code source — GitHub", GITHUB_URL, width="stretch")
    if API_PUBLIC_URL:
        st.link_button("📘 API — Swagger", f"{API_PUBLIC_URL}/docs", width="stretch")
        st.link_button("📕 API — ReDoc", f"{API_PUBLIC_URL}/redoc", width="stretch")
    if MLFLOW_UI_URL:
        st.link_button("📊 MLflow — Registry", MLFLOW_UI_URL, width="stretch")
    if AIRFLOW_UI_URL:
        st.link_button("🌀 Airflow — Orchestration", AIRFLOW_UI_URL, width="stretch")

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:1.4rem; color:#9CA3AF; font-size:0.78rem;">
          Réalisé par <b style="color:#374151;">{AUTHOR}</b><br>ESGI · IABD — Fil rouge MLOps
        </div>
        """,
        unsafe_allow_html=True,
    )

PAGES[selected]()
