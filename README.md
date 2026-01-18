# Churn MLOps

[![Licence](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python version](https://img.shields.io/badge/python-%3E%3D3.12-slim)](https://www.python.org/downloads/)
[![Dernier commit](https://img.shields.io/github/last-commit/arnaudstdr/churn-mlops/main)](https://github.com/arnaudstdr/churn-mlops/commits/main)
[![Stars](https://img.shields.io/github/stars/arnaudstdr/generate_mail?style=social)](https://github.com/arnaudstdr/generate_mail/stargazers)

## Objectif du projet

**Churn MLOps** est un projet vitrine dont l’objectif est de démontrer ma capacité à :

- construire un modèle de machine learning **utile et robuste**,
- l’**industrialiser** via une API,
- le **déployer**, le **monitorer** et le **faire évoluer** dans une logique proche de la production.

Le cas d’usage choisi est la **prédiction du churn client** (attrition), un problème courant et concret rencontré dans de nombreux contextes métiers (SaaS, télécoms, abonnements).

Ce projet ne cherche pas à maximiser la performance du modèle à tout prix, mais à montrer une **approche réaliste, lisible et maintenable**, orientée produit et exploitation.

## Philosophie

- Priorité à la **simplicité**, à la **clarté** et à la **robustesse**
- Pas de sur-ingénierie
- Décisions techniques **argumentées**
- Orientation **production / MLOps**, pas POC jetable

## État actuel

Le projet est en cours de construction et évolue par itérations (sprints).
Cette première version pose uniquement le **cadre et l’intention** du projet.

Les détails techniques (architecture, choix ML, API, monitoring, CI/CD, etc.) seront documentés progressivement.

## Git Hooks

Ce projet utilise des hooks Git partagés.

Après le clone :

```bash
./scripts/install-hooks.sh
```