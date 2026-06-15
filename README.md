# Bank Marketing — Pipeline MLOps de classification

Projet fil rouge du cours d'**orchestration Machine Learning**. On construit, séance
après séance, un pipeline MLOps complet (de la donnée au déploiement) autour d'un
problème unique de **classification binaire**.

## Problématique

> **Un client va-t-il souscrire un dépôt à terme à la suite d'une campagne de
> télémarketing ?**

Une banque portugaise mène des campagnes d'appels téléphoniques pour proposer un
**dépôt à terme** (placement bloqué rémunéré). Chaque ligne du jeu de données décrit
un client contacté et l'issue de l'appel.

- **classe `1`** : le client **souscrit** le dépôt à terme (`y = "yes"`)
- **classe `0`** : le client **ne souscrit pas** (`y = "no"`)

**Pourquoi prédire cette cible est utile :** en estimant *avant* d'appeler la
probabilité qu'un client souscrive, la banque peut **prioriser les appels** sur les
profils les plus prometteurs, et ainsi **réduire le coût** de la campagne (temps des
téléconseillers) tout en maintenant le nombre de souscriptions.

## Données

- **Source :** [Bank Marketing — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  (téléchargeable publiquement, sans authentification).
- **Variante utilisée :** `bank-full.csv` — **45 211 lignes**, 16 variables + la cible.
- **Déséquilibre :** environ **11,7 %** de `yes` → on suit le **ROC AUC** plutôt que
  la simple accuracy.

La préparation ([`prepare_data.py`](src/bank_marketing/prepare_data.py)) télécharge
l'archive, mappe la cible `y` (`yes → 1`, `no → 0`) et **retire la colonne
`duration`** (durée du dernier appel) : elle n'est connue qu'**après** l'appel, donc
l'utiliser serait une **fuite de données** pour un modèle censé prédire *avant*
d'appeler.

**Variables retenues :**

- **Numériques :** `age`, `balance`, `day`, `campaign`, `pdays`, `previous`
- **Catégorielles :** `job`, `marital`, `education`, `default`, `housing`, `loan`,
  `contact`, `month`, `poutcome`

Tout est centralisé dans [`config.py`](src/bank_marketing/config.py), seul fichier à
adapter pour brancher le jeu de données ; `data.py` et `features.py` (split stratifié
+ `ColumnTransformer` : `StandardScaler` / `OneHotEncoder`) lisent cette configuration
automatiquement.

## Mise en route

```bash
make install                          # installe les dependances (uv, Python 3.13)
python -m bank_marketing.prepare_data # genere data/dataset.csv
python -m bank_marketing.train        # entraine la baseline -> f1=... roc_auc=...
```

> Les commandes s'exécutent depuis la racine (le `Makefile` fixe `PYTHONPATH=src`).
> Équivalents : `make data`, `make train`.

**Baseline actuelle** (régression logistique) : `roc_auc ≈ 0.77` sur le jeu de test.

## Structure

```
src/bank_marketing/
  config.py        configuration du dataset (cible, variables)   <- branchement
  prepare_data.py  telechargement + nettoyage -> data/dataset.csv
  data.py          chargement + split stratifie
  features.py      pre-processing (ColumnTransformer)
  train.py         entrainement de la baseline (LogisticRegression)
data/dataset.csv   jeu de donnees prepare
Makefile           cibles d'installation et du pipeline
pyproject.toml     dependances (uv)
```

## Démarche

Le projet **évolue séance par séance** : à chaque étape on enrichit le pipeline
(suivi d'expériences, optimisation d'hyperparamètres, conteneurisation, API,
orchestration, CI/CD…). Le code est complété progressivement, au rythme du cours.
