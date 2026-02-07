# PLANS.md — Churn MLOps

## Objectif global

Construire un projet MLOps complet autour d’un cas de **prédiction du churn client**, depuis les données jusqu’à une API monitorée, avec une approche réaliste de mise en production.

Ce projet sert de vitrine technique pour démontrer la capacité à :
- concevoir un pipeline ML robuste,
- exposer un modèle via une API,
- instrumenter l’observabilité,
- structurer un projet maintenable et évolutif.

---

## Principes directeurs

- Simplicité > sophistication
- Lisibilité > micro-optimisation
- Décisions explicites et documentées
- Pas de tuning excessif
- Orientation production (pas POC jetable)

---

## Organisation du projet

Le projet est découpé en **sprints successifs**, chacun ayant un objectif clair et un périmètre maîtrisé.

Chaque sprint doit :
- produire un état stable,
- être testable manuellement,
- être documenté a minima.

---

## Sprint 0 — Fondation & cadre projet

### Objectif
Poser une base saine : repo, conventions, outils, observabilité de base.

### Tâches

#### Infrastructure & Repo
- [x] Initialiser le repository GitHub (public)
- [x] Créer la structure de dossiers (api/, ml/, data/, scripts/, docs/)
- [x] Ajouter un README minimal (objectif du projet)
- [x] Définir une convention de versioning (vX.Y.Z)

#### Outils & Qualité
- [x] Mettre en place des hooks Git partagés
- [x] Vérifier que les hooks fonctionnent réellement
- [x] Définir les conventions de nommage et de structure

#### Observabilité (base)
- [x] Créer un projet Sentry (environnements dev / demo)
- [x] Ajouter la dépendance Sentry SDK
- [x] Centraliser l’initialisation Sentry (via variables d’environnement)
- [x] Tester volontairement la remontée d’une erreur
- [x] Documenter la partie observabilité dans le README

---

## Sprint 1 — Machine Learning (offline)

### Objectif
Construire un **modèle baseline simple, robuste et explicable**.

### Tâches

#### Données
- [x] Sélectionner un dataset de churn (ex : Telco Customer Churn)
- [x] Inspecter rapidement les données (types, valeurs manquantes)
- [x] Documenter les hypothèses principales

#### Modélisation
- [x] Définir clairement la cible (churn / no churn)
- [x] Choisir une métrique principale (ex : ROC-AUC, recall churn)
- [x] Implémenter un preprocessing clair et reproductible
    - [x] encodage
    - [x] scaling
- [x] Mettre en place un split train / validation / test reproductible

#### Modèle
- [x] Entraîner un modèle baseline (logistic regression, tree, etc.)
- [x] Pas de tuning excessif
- [x] Calculer les métriques offline
- [x] Choisir un seuil de décision argumenté
- [x] Sauvegarder les artefacts (modèle + preprocessing)

#### Documentation
- [x] Documenter les choix ML dans le README
- [x] Expliquer les compromis (simplicité vs performance)

---

## Sprint 2 — API de prédiction

### Objectif
Exposer le modèle via une **API FastAPI claire et robuste**.

### Tâches

#### API
- [ ] Initialiser FastAPI
- [ ] Définir les schémas d’entrée et de sortie (Pydantic)
- [ ] Implémenter l’endpoint `/predict`
- [ ] Implémenter la logique proba + seuil
- [ ] Ajouter `/health` et `/model`
- [ ] Gérer proprement les erreurs d’entrée (400 lisibles)

#### Observabilité API
- [ ] Logging structuré (JSON)
- [ ] Ajouter un `request_id` pour la corrélation logs / erreurs
- [ ] Vérifier la remontée correcte des erreurs côté Sentry
- [ ] Ajouter des tags Sentry (endpoint, model_version)
- [ ] Définir une stratégie de nommage des transactions

#### Tests & Docs
- [ ] Test manuel de l’API (Postman)
- [ ] Documenter le contrat API dans le README

---

## Sprint 3 — Monitoring & qualité ML

### Objectif
Assurer la **traçabilité, la reproductibilité et la non-régression**.

### Tâches

#### MLflow
- [ ] Déployer MLflow en local (docker-compose)
- [ ] Logger paramètres, métriques et artefacts
- [ ] Lier version modèle ↔ version code

#### Tests
- [ ] Tests unitaires sur le preprocessing
- [ ] Test “predict smoke”
- [ ] Test de non-régression simple sur un échantillon fixe

#### Qualité de code
- [ ] Configurer lint et format (ruff, black ou équivalent)
- [ ] Ajouter des tests automatiques

---

## Sprint 4 — CI & stabilisation

### Objectif
Rendre le projet **automatisé et présentable**.

### Tâches

- [ ] Mettre en place une GitHub Actions CI
    - [ ] tests
    - [ ] lint
    - [ ] build
- [ ] Ajouter un badge CI dans le README
- [ ] Nettoyer la documentation
- [ ] Vérifier la cohérence globale du projet

---

## Évolutions possibles (hors scope initial)

- [ ] Déploiement cloud (AWS App Runner / équivalent)
- [ ] Monitoring data drift
- [ ] Feature store
- [ ] Auth API
- [ ] Canary release / shadow mode

---

## Règles pour les agents LLM (résumé)

- Ne pas sur-concevoir
- Toujours expliquer les choix
- Proposer avant d’implémenter
- Respecter PEP8 et les conventions du projet
- Priorité à la lisibilité et à la robustesse

---

Fin du plan.
