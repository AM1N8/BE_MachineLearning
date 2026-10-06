# Identification des utilisateurs de Copilote: Projet ML

Projet de bureau d’étude (BE) du module **MOD 7.2 — Introduction à la science des données**, associé à la compétition Kaggle **« Qui utilise mon appli ? v2026/2027 — Groupe 2 »**.

## Problématique

Peut-on reconnaître un utilisateur à partir de sa manière d’utiliser un logiciel ?

Infologic édite le logiciel Copilote pour le secteur agroalimentaire. Ses testeurs internes utilisent différents profils pour tester les fonctionnalités. Identifier le testeur à partir de ses traces d’utilisation pourrait faciliter le signalement des problèmes. Cette approche pourrait également contribuer à repérer des comportements inhabituels sur un compte.

La tâche du projet consiste à **prédire l’identité de l’utilisateur ayant effectué une session**, à partir du navigateur, des actions réalisées et de leur contexte. Il s’agit d’une **classification supervisée multiclasse** : chaque utilisateur constitue une classe possible.

La détection d’intrusions est une motivation du projet, mais les données fournies ne comportent pas d’étiquettes « intrusion » : la cible à prédire reste l’identité de l’utilisateur.

## Données

Les données sont présentes dans le dossier `data/`.

| Fichier | Contenu | Rôle |
| --- | --- | --- |
| `train.csv` | 3 279 sessions étiquetées, réalisées par 247 utilisateurs | Apprentissage et validation |
| `test.csv` | 324 sessions sans identité | Sessions pour lesquelles produire les prédictions |
| `sample_submission.csv` | Exemple de soumission de 324 lignes | Format attendu par la compétition |

### Format des traces

Les fichiers d’apprentissage et de test sont des fichiers texte **sans en-tête**, dont les champs sont séparés par des virgules. **Chaque ligne représente une session**, et le nombre de champs varie selon sa longueur.

Dans `train.csv`, les premiers champs sont l’identifiant de l’utilisateur (`util`, généralement un trigramme) et le navigateur. Dans `test.csv`, le premier champ est directement le navigateur ; l’identifiant est absent.

Exemple illustratif d’une session d’apprentissage :

```text
sph,Firefox,Création d'un écran(infologic.core.gui.controllers.BlankController)<ACCUEIL_INST>,t5,Saisie dans un champ1,t10
```

Les traces peuvent contenir :

- **Des actions** : création d’un écran, double-clic, saisie, exécution d’un bouton, etc.
- **Un écran ou un contexte entre parenthèses** : `(infologic.core.gui.controllers.BlankController)`.
- **Une configuration d’écran entre chevrons** : `<ACCUEIL_INST>`.
- **Une chaîne de fiche entre dollars** : `$GP$`, correspondant à son regroupement logique dans le logiciel.
- **Un suffixe `1`** : il indique que l’utilisateur est en modification dans la fiche.
- **Des marqueurs temporels `tXX`** : ils délimitent des fenêtres temporelles exprimées en secondes, par exemple `t5` ou `t10`.

Les navigateurs observés sont Firefox, Google Chrome, Microsoft Edge et Opera. Le nombre de sessions par utilisateur varie de **4 à 75** : les classes sont donc inégalement représentées.

### Particularités des fichiers présents

- `train.csv.GZ` est identique à `train.csv` et n’est pas réellement compressé malgré son extension.
- `test.csv.GZ` est compressé au format gzip. Une fois décompressé, il contient les mêmes lignes que `test.csv`, avec des fins de ligne différentes.
- Une session de `train.csv` contient uniquement des marqueurs temporels après l’identité et le navigateur.
- Lorsqu’on transforme les traces en tableau, les sessions courtes peuvent être complétées par des valeurs manquantes. Ces valeurs ne correspondent pas nécessairement à des informations perdues : elles proviennent de la longueur variable des sessions.

## Démarche attendue

Le challenge demande des prédictions fiables. Le BE demande également une démarche expliquée et reproductible :

1. Explorer les données : utilisateurs, navigateurs, longueur des sessions et cas particuliers.
2. Construire des caractéristiques pertinentes : fréquence des actions, écrans utilisés, configurations, chaînes de fiches, informations temporelles, etc.
3. Réserver une partie des données étiquetées à la validation afin d’évaluer les modèles sur des sessions qu’ils n’ont pas utilisées pour apprendre.
4. Comparer des méthodes de classification et justifier les choix de modèle et de paramètres.
5. Analyser les résultats, les informations utiles et les limites de la solution.
6. Appliquer la solution retenue aux sessions de test et préparer la soumission.

Le fichier `test.csv` ne contient pas les réponses : il ne permet donc pas de calculer directement un score local. Les décisions de sélection du modèle doivent s’appuyer sur les données de validation issues de `train.csv`.

## Évaluation et soumission

La description de la compétition indique un **F1 score moyen**, qui combine précision et rappel :

```text
F1 = 2 × précision × rappel / (précision + rappel)
```

Le type exact de moyenne entre les classes n’est pas précisé dans le texte fourni et reste à confirmer. Les indications du BE recommandent également de mesurer l’accuracy, c’est-à-dire la proportion de sessions correctement classées.

La soumission doit contenir les colonnes **`RowId`** et **`prediction`**, avec une prédiction par session de test, en conservant l’ordre des sessions. Les identifiants vont de **1 à 324** et les prédictions doivent être les identifiants utilisateur d’origine.

Exemple de format, avec des prédictions fictives :

```csv
RowId,prediction
1,sph
2,muz
3,nuh
```

## Livrables du BE

D’après le notebook de consignes, la date limite est le **9 novembre 2026 au soir**. Les livrables attendus sont :

- **Le code Python réexécutable**, structuré et commenté, accompagné des dépendances nécessaires. Les consignes évoquent un environnement Python 3.14.
- **Un rapport PDF de 5 à 8 pages**, présentant le problème, les caractéristiques construites, les méthodes, les résultats, leur interprétation et les limites.
- **Les prédictions sur le jeu de test**, à déposer sur Moodle et sur la compétition Kaggle correspondant au groupe.

Le travail est prévu par groupes de trois élèves. Chaque membre doit être capable d’expliquer l’ensemble de la démarche et de répondre aux questions de l’enseignant.

## Organisation du dépôt

```text
BE_MachineLearning/
├── data/                         # Données d’apprentissage, de test et exemple de soumission
├── Notebooks/
│   ├── BE_CS1_instructions.ipynb  # Consignes et livrables du BE
│   └── BE_CS1_indications.ipynb   # Guide pédagogique et étapes à compléter
├── SAIF_Notebooks/
│   └── datamanip.ipynb           # Fichier actuellement vide
├── src/
│   └── be_ml/
│       └── __init__.py           # Point d’entrée initial du projet
├── pyproject.toml               # Configuration Python et dépendances déclarées
├── uv.lock                      # Versions verrouillées des dépendances
├── .python-version              # Version Python indiquée : 3.11
└── README.md
```

## État actuel du projet

Le dépôt contient les données, les supports pédagogiques et une structure Python initiale. Les dépendances directes déclarées sont **Pandas** et **Matplotlib**, avec une version Python minimale de **3.11**.

Les notebooks pédagogiques contiennent de nombreuses étapes à compléter. `SAIF_Notebooks/datamanip.ipynb` est vide et le point d’entrée Python affiche seulement un message initial. **Aucune chaîne de classification complète, aucun modèle entraîné et aucun résultat de validation ne sont présents à ce stade.**

La prochaine étape consiste à explorer les données et à définir les premières caractéristiques permettant de distinguer les utilisateurs.
