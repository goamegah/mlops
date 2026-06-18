"""Frontend Streamlit : dashboard MLOps Bank Marketing.

Seance 14 bis - TP Streamlit
    - Prediction       : formulaire -> endpoint /predict de l'API (TP S12).
    - Suivi du modele  : Model Registry MLflow (versions, alias, modele en prod).
    - Evaluation       : porte qualite + metriques + artefacts (mlflow.evaluate).
    - Table previsions : journal des predictions (base) + saisie de feedback.
    Lancement : `make frontend` (avec `make api`, `make mlflow`, `make db`).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from bank_marketing.config import (
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)
from bank_marketing.evaluate import evaluate_model

st.set_page_config(page_title="Bank Marketing", page_icon="🏦", layout="wide")

DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# URLs publiques (navigateur) pour les liens cliquables - distinctes des URLs
# internes Docker (http://api:8000, http://mlflow:5000) qui ne servent qu'aux
# appels serveur. Definies via .env sur la VM ; vides en local -> liens masques.
MLFLOW_UI_URL = os.environ.get("MLFLOW_UI_URL", "").rstrip("/")
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", "").rstrip("/")
AIRFLOW_UI_URL = os.environ.get("AIRFLOW_UI_URL", "").rstrip("/")

# Lien vers le code source (valeur par defaut : le depot du projet) et auteur.
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/goamegah/mlops")
AUTHOR = "Godwin Amegah"

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


# --- Barre laterale : branding, liens externes, etat de la stack ---
with st.sidebar:
    st.title("🏦 Bank Marketing")
    st.caption("Demonstrateur MLOps - souscription a un depot a terme")
    api_url = st.text_input("URL de l'API", value=DEFAULT_API_URL)

    st.divider()
    st.subheader("🔗 Liens")
    st.link_button("🐙 Code source (GitHub)", GITHUB_URL, width="stretch")
    if API_PUBLIC_URL:
        st.link_button("📘 API — Swagger", f"{API_PUBLIC_URL}/docs", width="stretch")
        st.link_button("📕 API — ReDoc", f"{API_PUBLIC_URL}/redoc", width="stretch")
    if MLFLOW_UI_URL:
        st.link_button("📊 MLflow — Registry", MLFLOW_UI_URL, width="stretch")
    if AIRFLOW_UI_URL:
        st.link_button("🌀 Airflow — Orchestration", AIRFLOW_UI_URL, width="stretch")

    st.divider()
    st.subheader("⚙️ Etat de la stack")
    for label, ok in [
        ("API", _ping(f"{api_url}/health")),
        ("MLflow", _ping(f"{MLFLOW_TRACKING_URI}/health")),
        ("Base de donnees", _ping(f"{api_url}/predictions?limit=1")),
    ]:
        st.write(f"{'🟢' if ok else '🔴'} {label}")

    st.divider()
    st.caption(f"Réalisé par **{AUTHOR}**")
    st.caption("ESGI · IABD — Fil rouge MLOps")

st.title("🏦 Bank Marketing — Souscription à un dépôt à terme")

intro_col, author_col = st.columns([3, 1], gap="medium")
with intro_col:
    with st.container(border=True):
        st.markdown("#### 🎯 Problématique")
        st.markdown(
            "Une banque mène des campagnes de **marketing téléphonique** pour placer des "
            "**dépôts à terme**. L'enjeu : prédire, *avant* de contacter un client, s'il va "
            "**souscrire** — afin de cibler les prospects les plus prometteurs et réduire le "
            "coût des campagnes."
        )
        st.markdown(
            "**Données** : UCI *Bank Marketing* (~45 000 contacts, 16 variables, classes "
            "déséquilibrées ≈ 12 % de souscriptions). **Tâche** : classification binaire "
            "— cible `y` (souscrit / ne souscrit pas)."
        )
with author_col:
    with st.container(border=True):
        st.markdown("#### 👤 Auteur")
        st.markdown(f"**{AUTHOR}**")
        st.caption("ESGI · IABD")
        st.caption("Fil rouge MLOps")
        st.link_button("🐙 GitHub ↗", GITHUB_URL, width="stretch")

st.write("")  # respiration verticale

predict_tab, tracking_tab, eval_tab, history_tab = st.tabs(
    ["🎯 Prediction", "📊 Suivi du modele", "✅ Evaluation", "🗂️ Table previsions"]
)

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

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

        submitted = st.form_submit_button("Predire")

    if submitted:
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
            st.error(f"Appel a l'API impossible : {exc} (l'API est-elle demarree ?)")
        else:
            proba = float(prediction["probability"])
            label = "Souscrit (1)" if prediction["prediction"] == 1 else "Ne souscrit pas (0)"
            try:
                served = httpx.get(f"{api_url}/model-info", timeout=5.0).json().get("version", "?")
            except Exception:  # noqa: BLE001 - /model-info indisponible ou reponse inattendue
                served = "?"

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", label)
            c2.metric("Probabilite de souscription", f"{proba:.1%}")
            c3.metric("Modele servi", f"v{served}" if served not in ("?", "unknown") else served)
            st.progress(proba)
            st.caption(f"Prediction enregistree en base — id : `{prediction.get('id', '')}`")
            st.toast("Prediction effectuee", icon="✅")

            with st.expander("Detail de la requete (payload envoye / reponse brute)"):
                rq_col, rp_col = st.columns(2)
                with rq_col:
                    st.markdown("**Payload envoye** (`POST /predict`)")
                    st.json(payload)
                with rp_col:
                    st.markdown("**Reponse brute de l'API**")
                    st.json(prediction)

with tracking_tab:
    st.subheader("Model Registry MLflow")
    st.caption(
        f"Tracking URI (interne au reseau Docker) : `{MLFLOW_TRACKING_URI}`  |  "
        f"Modele : `{MODEL_NAME}`"
    )
    if MLFLOW_UI_URL:
        st.link_button("Ouvrir le Model Registry dans MLflow ↗", f"{MLFLOW_UI_URL}/#/models")

    try:
        rows = _load_registry_rows()
    except Exception as exc:  # noqa: BLE001 - MLflow injoignable / non demarre
        st.error(f"MLflow injoignable ({exc}). Demarre-le avec `make mlflow`.")
        rows = []

    if not rows:
        st.info(
            "Aucune version dans le registry. Lance un entrainement "
            "(`make train-optuna` ou `make train-models`)."
        )
    else:
        prod = next((r for r in rows if "prod" in r["alias"]), None)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modele en production", f"v{prod['version']}" if prod else "aucun")
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

        existing_aliases = sorted(
            {a.strip() for r in rows for a in r["alias"].split(",") if a.strip()}
        )
        if existing_aliases:
            st.divider()
            st.markdown("**Retirer un alias**")
            del_alias = st.selectbox("Alias a retirer", existing_aliases)
            if st.button("Retirer l'alias"):
                MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).delete_registered_model_alias(
                    MODEL_NAME, del_alias
                )
                st.toast(f"Alias '{del_alias}' retire", icon="🗑️")
                st.rerun()

with eval_tab:
    st.subheader("Evaluation & porte qualite")
    st.caption(f"Seuils : roc_auc >= {EVAL_ROC_AUC_MIN}  |  f1_score >= {EVAL_F1_MIN}")

    try:
        eval_rows = _load_registry_rows()
    except Exception as exc:  # noqa: BLE001 - MLflow injoignable / non demarre
        st.error(f"MLflow injoignable ({exc}). Demarre-le avec `make mlflow`.")
        eval_rows = []

    if not eval_rows:
        st.info("Aucune version a evaluer. Lance un entrainement d'abord.")
    else:
        eval_version = st.selectbox("Version a evaluer", [r["version"] for r in eval_rows])
        if st.button("Lancer l'evaluation"):
            with st.spinner("Evaluation du modele sur le jeu de test..."):
                try:
                    result = evaluate_model(
                        model_uri=f"models:/{MODEL_NAME}/{eval_version}", validate=False
                    )
                except Exception as exc:  # noqa: BLE001 - evaluation impossible
                    st.error(f"Evaluation impossible : {exc}")
                else:
                    metrics = result.metrics
                    f1 = float(metrics["f1_score"])
                    roc = float(metrics["roc_auc"])
                    roc_ok = roc >= EVAL_ROC_AUC_MIN
                    f1_ok = f1 >= EVAL_F1_MIN

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ROC AUC", f"{roc:.4f}")
                    m2.metric("F1-score", f"{f1:.4f}")
                    m3.metric(
                        "Precision", f"{float(metrics.get('precision_score', float('nan'))):.4f}"
                    )
                    m4.metric("Recall", f"{float(metrics.get('recall_score', float('nan'))):.4f}")

                    st.write(
                        f"- roc_auc {roc:.4f} {'>=' if roc_ok else '<'} "
                        f"{EVAL_ROC_AUC_MIN} -> {'OK' if roc_ok else 'KO'}"
                    )
                    st.write(
                        f"- f1_score {f1:.4f} {'>=' if f1_ok else '<'} "
                        f"{EVAL_F1_MIN} -> {'OK' if f1_ok else 'KO'}"
                    )
                    if roc_ok and f1_ok:
                        st.success("Porte qualite : ACCEPTE (les deux seuils sont atteints).")
                    else:
                        st.error("Porte qualite : REJETE (un seuil n'est pas atteint).")

                    images = [
                        (name, art.content)
                        for name, art in result.artifacts.items()
                        if type(art.content).__module__.startswith("PIL")
                    ]
                    if images:
                        st.divider()
                        st.markdown("**Artefacts d'evaluation**")
                        acols = st.columns(2)
                        for i, (name, img) in enumerate(images):
                            acols[i % 2].image(img, caption=name, width="stretch")

with history_tab:
    st.subheader("Journal des previsions (base de donnees)")

    try:
        resp = httpx.get(f"{api_url}/predictions", params={"limit": 100}, timeout=10.0)
        resp.raise_for_status()
        journal = resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Journal indisponible : {exc} (API + MySQL demarres ?)")
        journal = []

    if not journal:
        st.info("Aucune prevision enregistree. Va dans l'onglet Prediction pour en creer.")
    else:
        st.dataframe(pd.DataFrame(journal), width="stretch", hide_index=True)

        st.divider()
        st.markdown("**Enregistrer un feedback (verite terrain)**")
        options = {
            row["id"]: f"{row['id'][:8]}...  pred={row['prediction']}  p={row['probability']}"
            for row in journal
        }
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        fb_id = fc1.selectbox("Prediction a annoter", list(options), format_func=options.get)
        fb_actual = fc2.selectbox(
            "Resultat reel", [1, 0], format_func=lambda x: "Souscrit (1)" if x == 1 else "Non (0)"
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
                st.toast("Feedback enregistre", icon="✅")
                st.rerun()
