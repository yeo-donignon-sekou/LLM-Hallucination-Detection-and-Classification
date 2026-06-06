# LLM-Hallucination-Detection-and-Classification
 Analyse et détection des hallucinations des LLMs à l’aide du Machine Learning, EDA, Feature Engineering et modèles de classification (Random Forest, Extra Trees, Gradient Boosting).


# 🧠 Détection des Hallucinations des LLMs & Classification par Machine Learning

## 📌 Présentation du Projet

Les grands modèles de langage (LLMs) comme GPT-4o, Claude, Gemini ou Llama révolutionnent aujourd’hui l’intelligence artificielle. Cependant, l’un de leurs principaux défis reste la génération d’**hallucinations**, c’est-à-dire des réponses qui semblent crédibles mais qui contiennent des informations fausses, inventées ou contradictoires.

Ce projet a pour objectif d’analyser, comprendre et prédire les hallucinations produites par les LLMs grâce aux techniques de Data Science et de Machine Learning.

Le notebook combine :

* Analyse exploratoire des données (EDA)
* Analyse statistique
* Prétraitement des données
* Feature Engineering
* Classification supervisée
* Évaluation des modèles
* Interprétation des facteurs de risque liés aux hallucinations

L’objectif final est de construire des modèles capables de prédire si une réponse générée par un LLM est halluciné ou non.

---

# 🎯 Objectifs du Projet

Les principaux objectifs de ce projet sont :

* Étudier les hallucinations générées par plusieurs LLMs
* Comparer les taux d’hallucination entre modèles
* Identifier les types de prompts les plus risqués
* Étudier les domaines les plus sensibles
* Évaluer l’efficacité des stratégies de mitigation
* Construire des variables explicatives pertinentes
* Entraîner plusieurs modèles de Machine Learning
* Détecter automatiquement les réponses halluciné

---

# 📂 Description du Dataset

Le dataset contient **200 réponses annotées** générées par plusieurs modèles de langage.

Chaque observation contient des informations telles que :

| Variable                | Description                         |
| ----------------------- | ----------------------------------- |
| `model_name`            | Modèle de langage utilisé           |
| `prompt_type`           | Type de prompt                      |
| `domain`                | Domaine du sujet                    |
| `task_type`             | Type de tâche NLP                   |
| `language`              | Langue utilisée                     |
| `hallucination_label`   | Variable cible                      |
| `hallucination_type`    | Type d’hallucination                |
| `severity`              | Niveau de sévérité                  |
| `domain_risk`           | Niveau de risque du domaine         |
| `annotation_confidence` | Niveau de confiance de l’annotateur |
| `mitigation_strategy`   | Stratégie de mitigation utilisée    |

---

# 🎯 Variable Cible

La variable cible du projet est :

```python id="y0z83l"
hallucination_label
```

Avec :

* `1` → Réponse halluciné
* `0` → Réponse non halluciné

Le problème est donc formulé comme une **classification binaire**.

---

# 🔍 Analyse Exploratoire des Données (EDA)

L’analyse exploratoire permet d’étudier les comportements des hallucinations selon plusieurs dimensions.

Les analyses réalisées incluent :

* Analyse des valeurs manquantes
* Distribution des hallucinations
* Analyse des niveaux de sévérité
* Comparaison des modèles LLM
* Analyse des types de prompts
* Analyse des domaines à risque
* Analyse des langues
* Étude des stratégies de mitigation
* Analyse du niveau de confiance des annotateurs
* Étude des corrélations

---

# 📊 Principaux Résultats

## ✅ Taux Global d’Hallucination

* **34,5 %** des réponses générées contiennent des hallucinations
* Environ **45 %** des hallucinations sont classées en sévérité élevée

Ces résultats montrent l’importance des systèmes de détection des hallucinations, notamment dans les domaines sensibles.

---

## 🤖 Comparaison des Modèles

Parmi les modèles testés :

* **Llama-3.1-70B** présente le taux d’hallucination le plus élevé
* **GPT-4o** et **Claude-3.5-Sonnet** affichent les meilleurs résultats

Cela met en évidence des différences importantes de fiabilité entre les architectures LLM.

---

## ⚠️ Impact des Types de Prompts

L’analyse montre que :

* Les **prompts adversariaux** génèrent le plus d’hallucinations
* Les prompts **Multi-Hop** sont également très risqués

Les tâches nécessitant un raisonnement complexe augmentent le risque d’erreurs.

---

## 🏥 Domaines à Haut Risque

Les domaines les plus sensibles sont :

* Médecine
* Droit

Ces domaines nécessitent un très haut niveau de fiabilité factuelle.

---

## 🛡️ Stratégies de Mitigation

Les stratégies de mitigation réduisent significativement les hallucinations.

Les méthodes les plus efficaces observées sont :

1. RAG (Retrieval-Augmented Generation)
2. Self-Consistency
3. Structured Prompting

Réduction observée :

* Environ **13 points de pourcentage** de réduction du taux d’hallucination

---

# 🛠️ Feature Engineering

Plusieurs variables dérivées ont été créées afin d’améliorer les performances des modèles :

| Variable créée          | Rôle                      |
| ----------------------- | ------------------------- |
| `log_prompt_len`        | Complexité du prompt      |
| `log_response_len`      | Complexité de la réponse  |
| `response_prompt_ratio` | Niveau de verbosité       |
| `is_high_stakes`        | Domaine sensible          |
| `is_adversarial`        | Prompt adversarial        |
| `is_multihop`           | Raisonnement multi-étapes |
| `mit_applied_int`       | Mitigation appliquée      |

Ces variables permettent de mieux capturer les signaux associés aux hallucinations.

---

# 🤖 Modèles de Machine Learning

Plusieurs algorithmes supervisés ont été entraînés et comparés :

* Régression Logistique
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting

---

# 📈 Évaluation des Modèles

Les modèles sont évalués à l’aide des métriques suivantes :

* Accuracy
* F1-Score
* ROC-AUC
* Validation croisée

Les modèles basés sur les arbres obtiennent les meilleures performances.

Meilleurs modèles :

✅ Random Forest
✅ Extra Trees

Ces modèles capturent efficacement les relations non linéaires entre les variables.

---

# 🔥 Importance des Variables

Les variables les plus importantes dans la prédiction des hallucinations sont :

* `annotation_confidence`
* `model_name_enc`
* `domain_risk_enc`
* `prompt_type_enc`
* `mit_applied_int`

Ces résultats montrent que les hallucinations dépendent fortement :

* du modèle utilisé,
* du type de prompt,
* du niveau de risque du domaine,
* de l’utilisation de stratégies de mitigation.

---

# 📉 Limites du Projet

Ce projet présente certaines limites :

* Dataset relativement petit (200 observations)
* Risque de biais d’échantillonnage
* Variance élevée en validation croisée
* Représentation sémantique limitée des textes

Les résultats doivent donc être interprétés avec prudence.

---

# 🚀 Améliorations Possibles

Plusieurs améliorations peuvent être envisagées :

* Augmenter la taille du dataset
* Utiliser des embeddings NLP avancés
* Fine-tuning de modèles BERT / RoBERTa
* Détection temps réel des hallucinations
* Intégration d’architectures Deep Learning
* Systèmes hybrides IA + validation humaine

---

# 🧰 Technologies Utilisées

## Data Science & Programmation

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Machine Learning

* Random Forest
* Extra Trees
* Gradient Boosting
* Logistic Regression

## Visualisation

* Heatmaps
* Courbes ROC
* Matrices de corrélation
* Feature Importance

---

# 📌 Conclusion

Ce projet démontre comment la Data Science et le Machine Learning peuvent être utilisés pour analyser et prédire les hallucinations générées par les grands modèles de langage.

Au-delà des performances des modèles, ce notebook met en évidence un enjeu majeur de l’intelligence artificielle moderne :

> Construire des systèmes d’IA fiables, explicables et robustes.

La détection des hallucinations devient essentielle à mesure que les LLMs sont intégrés dans des domaines critiques comme la santé, la finance, le droit ou l’éducation.

Ce projet constitue ainsi une base solide pour développer des systèmes d’IA plus sûrs et plus fiables.

---

# 👨‍💻 Auteur

**YEO Donignon Sékou**
Master 2 — Ingénierie Mathématique & Data Science
Université Côte d’Azur

📌 Domaines d’intérêt :

* Data Science
* Machine Learning
* Intelligence Artificielle
* NLP
* Finance Quantitative
* Fiabilité des systèmes d’IA
