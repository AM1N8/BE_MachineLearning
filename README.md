# Identification des utilisateurs de Copilote: Projet ML DS

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
- Les marqueurs temporels sont insérés parmi les actions. Les valeurs `tNN` observées progressent par pas de 5 et sont exprimées en secondes selon l’énoncé. Il ne faut pas multiplier ces valeurs par 5.
- Les lignes ont des nombres de champs différents parce que les sessions n’ont pas toutes la même longueur. Cette variation est normale : il ne faut ni supprimer les sessions courtes, ni interpréter un remplissage artificiel comme une action.

### Cadrage et données préparées

La cible est `util`, l’identifiant de l’utilisateur. Le problème est une classification supervisée multiclasse, et non une classification d’intrusion : aucune étiquette d’intrusion n’est fournie. L’évaluation locale doit être faite sur une validation issue du train; le test fourni est sans étiquette.

Les notebooks expérimentaux `700_CNN.ipynb` et `701_RandomForest.ipynb` utilisent les tableaux préparés présents dans `data/processed/1clean/`. Le pipeline final `FINAL_PROJECT.ipynb`, lui, repart directement des fichiers bruts `train.csv` et `test.csv` :

- `cleaned_train.csv` : 3 279 sessions, la cible `util` et 765 caractéristiques numériques ;
- `cleaned_test.csv` : 324 sessions et les mêmes 765 caractéristiques, sans `util`.

Les caractéristiques comprennent des compteurs d’actions, d’écrans, de configurations et de chaînes, les indicateurs de navigateur et des résumés de session. Dans ces tableaux, `max_time` et `duration_seconds` sont identiques; le notebook Random Forest supprime ce doublon. Le test n’ayant pas d’étiquette, il sert à produire les prédictions, jamais à calculer localement précision, rappel ou F1.

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

`Notebooks/FINAL_PROJECT.ipynb` est le pipeline de bout en bout recommandé. Il repart de `data/train.csv` et `data/test.csv`, vérifie et explore les traces brutes, compare plusieurs classifieurs, puis construit trois vues TF-IDF séquentielles (événements complets, actions, contextes) et une représentation de compteurs pour le Random Forest. Les paramètres du mélange sont sélectionnés sur une validation interne et évalués sur un holdout stratifié séparé.

Sur l’évaluation externe de 656 sessions, le mélange SVM enrichi + Random Forest a obtenu une **accuracy de 0,9741**, un **F1 macro de 0,9726**, une précision macro de **0,9775**, un rappel macro de **0,9730** et un F1 pondéré de **0,9716**. Il a dépassé le SVM enrichi seul (accuracy **0,9680**, F1 macro **0,9654**) et le Random Forest seul (accuracy **0,8872**, F1 macro **0,8584**). Ridge, SGD, Complement Naive Bayes et k-NN ont aussi été comparés sur la validation interne; ils n’ont pas fait mieux que le SVM.

Ce résultat est une estimation issue d’un seul split interne, et non une garantie de généralisation ou un score Kaggle. L’objectif de **98–99 % n’a pas été atteint**. Les performances d’entraînement proches de 100 % montrent un écart de généralisation, particulièrement pour le Random Forest. Le jeu `test.csv` ne contenant pas les labels utilisateurs, aucune précision/rappel/F1 réel ne peut y être mesuré.

Après évaluation, le notebook réentraîne les deux modèles sur toutes les sessions étiquetées et produit la soumission `data/processed/final_submission.csv` au format `RowId,prediction`. Les essais précédents (`700_CNN.ipynb` et `701_RandomForest.ipynb`) sont conservés comme comparaisons historiques; ils ne remplacent pas le pipeline final basé sur les traces brutes.
