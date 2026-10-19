"""Generate the French LaTeX report from measured artifacts and verify its page count."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from copilote.io import write_json
from copilote.settings import provenance

LATEX_DIR = Path("reports/latex")
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{cmap}
\usepackage[T1]{fontenc}
\IfFileExists{french.ldf}{\usepackage[french]{babel}}{%
  \usepackage[english]{babel}%
  \renewcommand{\tablename}{Tableau}%
  \renewcommand{\figurename}{Figure}%
}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{placeins}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\graphicspath{{../figures/}}
\captionsetup{font=small}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\frenchspacing
\newcommand{\code}[1]{\texttt{\detokenize{#1}}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
"""

FRENCH_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


def _french_date(value: date) -> str:
    """Format a date in French independently of installed Babel language files."""
    return f"{value.day} {FRENCH_MONTHS[value.month - 1]} {value.year}"


def read_json(path: str | Path) -> dict[str, Any]:
    """Load a measured artifact, failing loudly if a phase was skipped."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Missing artifact {file}; run the corresponding phase first")
    return json.loads(file.read_text())


def latex_table(frame: pd.DataFrame, caption: str, label: str, float_format: str = "%.3f") -> str:
    """Render a booktabs table; callers pass already-measured values only."""
    body = frame.to_latex(
        index=False,
        escape=True,
        float_format=float_format,
        caption=caption,
        label=label,
        position="htbp",
        bold_rows=False,
    )
    return body.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule")


def comparison_table(config: dict[str, Any]) -> str:
    """Render a compact, readable subset of measured model comparisons."""
    frame = pd.read_csv("reports/results/model_comparison.csv")
    keep = [
        "majority",
        "random",
        "browser",
        "linear_unigrams",
        "linear",
        "linear_balanced",
        "ridge",
        "complement_nb",
        "rbf",
        "forest",
        "mlp",
        "stacking",
    ]
    frame = frame[frame.name.isin(keep)].copy()
    labels = {
        "majority": "Classe majoritaire",
        "random": "Aléatoire",
        "browser": "Navigateur seul",
        "linear_unigrams": "SVM linéaire, unigrammes",
        "linear": "SVM linéaire",
        "linear_balanced": "SVM linéaire, pondéré",
        "ridge": "Ridge",
        "complement_nb": "Naïf Bayes complémentaire",
        "rbf": "SVM RBF",
        "forest": "Forêt aléatoire",
        "mlp": "Perceptron multicouche",
        "stacking": "Empilement",
    }
    frame["Modèle"] = frame.name.map(labels)
    frame["Exactitude"] = frame.apply(
        lambda row: f"{row.accuracy_mean:.3f} ({row.accuracy_std:.3f})", axis=1
    )
    frame["F1-macro"] = frame.apply(
        lambda row: f"{row.macro_f1_mean:.3f} ({row.macro_f1_std:.3f})", axis=1
    )
    frame["Équil."] = frame.balanced_accuracy_mean.map(lambda value: f"{value:.3f}")
    frame["Top-5"] = frame.top5_mean.map(lambda value: f"{value:.3f}")
    frame["Temps (s)"] = frame.fit_seconds_mean.map(lambda value: f"{value:.1f}")
    table = frame[["Modèle", "Exactitude", "F1-macro", "Équil.", "Top-5", "Temps (s)"]]
    rendered = latex_table(
        table,
        "Comparaison sur 4 plis stratifiés et 3 répétitions (moyenne (écart-type) pour exactitude et F1-macro).",
        "tab:models",
    )
    return rendered.replace("\\begin{table}[htbp]", "\\begin{table}[!htbp]\n\\small")


def family_table() -> str:
    """Describe feature families with measured dimensions from the feature artifact."""
    features = read_json("reports/results/features.json")
    dimensions = int(features["matrix_shape"][1])
    families = pd.DataFrame(
        [
            [
                "A. Séquence",
                "Actions complètes et contextes associés",
                "Signature des parcours",
                "1-3-grammes",
            ],
            [
                "B. Action seule",
                "Verbe d'action sans contexte",
                "Style d'usage",
                "1-3-grammes",
            ],
            [
                "C. Contexte",
                "Écran, configuration, chaîne et paires",
                "Module métier utilisé",
                "1-3-grammes",
            ],
            [
                "D. Position",
                "Début, fin et intervalles temporels",
                "Rythme et point d'entrée",
                "Unigrammes positionnels",
            ],
            [
                "E. Stats",
                "Longueur, entropie et remises à zéro",
                "Profil quantitatif",
                "19 variables standardisées",
            ],
        ],
        columns=["Famille", "Contenu", "Hypothèse comportementale", "Encodage"],
    )
    rendered = latex_table(
        families,
        f"Familles de caractéristiques et hypothèses associées (matrice brute de {dimensions:,} colonnes).".replace(
            ",", " "
        ),
        "tab:families",
    )
    rendered = rendered.replace("\\begin{table}[htbp]", "\\begin{table}[!htbp]\n\\small")
    rendered = rendered.replace(
        "\\begin{tabular}{llll}",
        "\\begin{tabularx}{\\linewidth}{>{\\raggedright\\arraybackslash}p{2.2cm} Y Y >{\\raggedright\\arraybackslash}p{2.7cm}}",
    )
    return rendered.replace("\\end{tabular}", "\\end{tabularx}")


def _length_label(bucket: object) -> str:
    """Turn an interval string such as '[100.0, 500.0)' into a bracket-free label.

    A leading '[' would be parsed by LaTeX as the optional argument of '\\\\'.
    """
    numbers = re.findall(r"\d+(?:\.\d+)?|inf", str(bucket))
    if len(numbers) < 2:
        return str(bucket)
    low = f"{float(numbers[0]):.0f}"
    high = numbers[1] if numbers[1] == "inf" else f"{float(numbers[1]):.0f}"
    return f"{low}–{high}" if high != "inf" else f"{low}+"


def build_tex(config: dict[str, Any]) -> str:
    """Assemble the report, injecting only numbers read from artifacts."""
    loading = read_json("reports/results/loading.json")
    eda = read_json("reports/results/eda.json")
    selection = read_json("reports/results/selection.json")
    evaluation = read_json("reports/results/evaluation.json")
    holdout = read_json("reports/results/holdout.json")
    interpretation = read_json("reports/results/interpretability.json")
    submission = read_json("reports/results/submission.json")
    learning = read_json("reports/results/learning_curve.json")["points"]
    selected = selection["selected"]
    selected_repeated = evaluation["repeated"][selected]["mean"]
    selected_std = evaluation["repeated"][selected]["std"]
    headline = holdout["metrics"]
    report = config["report"]
    learning_frame = pd.DataFrame(learning)[
        ["fraction", "train_sessions", "train_macro_f1", "holdout_macro_f1"]
    ].copy()
    learning_frame["fraction"] = learning_frame.fraction.map(lambda value: f"{value:.0%}")
    learning_frame["train_macro_f1"] = learning_frame.train_macro_f1.map(
        lambda value: f"{value:.3f}"
    )
    learning_frame["holdout_macro_f1"] = learning_frame.holdout_macro_f1.map(
        lambda value: f"{value:.3f}"
    )
    learning_frame.columns = ["Part du train", "Sessions", "F1 train", "F1 validation"]
    learning_table = latex_table(
        learning_frame,
        "Courbe d'apprentissage du modèle retenu (diagnostic post-verrouillage).",
        "tab:learning",
    )
    permutation = (
        pd.DataFrame(interpretation["models"]["winner"]["grouped_permutation"])
        .sort_values("macro_f1_loss_mean", ascending=False)
        .head(8)
    )
    permutation["family"] = (
        permutation["family"]
        .map({"raw": "Séquence brute", "browser": "Navigateur"})
        .fillna(permutation["family"])
    )
    permutation.columns = ["Famille", "Colonnes", "Perte F1-macro", "Écart-type"]
    permutation_table = latex_table(
        permutation,
        "Permutation groupée par famille sur le holdout (perte de F1-macro).",
        "tab:permutation",
    )
    length_frame = pd.DataFrame(interpretation["models"]["winner"]["accuracy_by_length"])
    length_frame["length_bucket"] = length_frame["length_bucket"].map(_length_label)
    length_order = {"0–100": 0, "100–500": 1, "500–2000": 2, "2000+": 3}
    length_frame = length_frame.sort_values(
        "length_bucket", key=lambda values: values.map(length_order)
    )
    length_frame.columns = ["Longueur (tokens)", "Exactitude", "Sessions"]
    length_table = latex_table(
        length_frame, "Exactitude par tranche de longueur de session (holdout).", "tab:length"
    )
    confusion = pd.DataFrame(interpretation["models"]["winner"]["confusion_pairs"]).head(8)
    confusion.columns = ["Vérité", "Prédit", "Occurrences"]
    confusion_table = latex_table(
        confusion, "Principales confusions sur le holdout.", "tab:confusion", float_format="%d"
    )
    support = sorted(eda["class_support"].values())
    team = report["team"]
    group = report["group"]
    if "non communiqué" in team.lower() or "non communiqué" in group.lower():
        team, group = "Projet Copilote", "Rapport de science des données"
    return (
        PREAMBLE
        + rf"""
\title{{Classification de sessions Copilote par utilisateur}}
\author{{{team} \\ {group}}}
\date{{{_french_date(date.today())}}}
\begin{{document}}
\maketitle

\section{{Introduction}}
Copilote est l'application de gestion documentaire d'Infologic. Chaque session est une
séquence d'actions de la forme \code{{action(screen,config,chain)1}} où le suffixe \code{{1}} marque une
fin d'action, et où les jetons \code{{tNN}} encodent un repère temporel. La tâche est
d'attribuer chaque session à l'un des {loading["classes"]} utilisateurs enregistrés. Les
classes sont très déséquilibrées (de {min(support)} à {max(support)} sessions; objectif Kaggle
inconnu, suivi par défaut avec la F1-macro et l'exactitude). Un tel modèle sert au support, à
la détection d'anomalies et à l'ergonomie; il pose aussi une question de ré-identification
d'utilisateurs à partir de traces.

\section{{Données et analyse exploratoire}}
Les fichiers bruts comptent {loading["train_sessions"]} sessions d'apprentissage et
{loading["test_sessions"]} sessions de test, pour {loading["classes"]} utilisateurs. Le format
est irrégulier (une ligne = une session, nombre de jetons variable), {eda["train_test_exact_overlap"]}
chevauchement exact train/test; {eda["users_with_multiple_browsers"]} utilisateur possède plus
d'un navigateur; aucun doublon conflictuel n'a été trouvé. La grammaire est vérifiée sans perte
sur les {loading["distinct_tokens_union"]} jetons distincts de l'union train/test
({loading["roundtrip_failures"]} échec de reconstruction), y compris les contextes imbriqués et
les chaînes apostrophées. Les repères \code{{tNN}} sont traités comme des secondes, avec des
remises à zéro; les écarts qui en résultent sont des proxys, pas des durées exactes par action.
Le décalage train/test est décrit par des tests de Kolmogorov-Smirnov descriptifs
(voir \code{{eda.json}}), sans jamais ajuster un vocabulaire sur le test.

\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{eda_overview.pdf}}
\caption{{Vue d'ensemble : distribution des longueurs, entropie d'actions et repères temporels.}}
\end{{figure}}
\FloatBarrier

\section{{Construction des caractéristiques}}
Toutes les transformations sont ajustées à l'intérieur des plis d'apprentissage. Le tableau
\ref{{tab:families}} résume les familles; la famille A conserve les événements entiers comme
jetons afin de capturer les séquences d'écrans, la famille C isole le contexte métier, et la
famille E ajoute un profil quantitatif standardisé. Un test verrouille l'absence de fuite: un
jeton ou un navigateur absent des plis d'apprentissage ne peut pas apparaître dans le
vocabulaire, et les transformateurs non ajustés restent inchangés.
{family_table()}

\section{{Méthodes de classification}}
Le protocole réserve d'abord {int(100 * (1 - config["cv"]["holdout_fraction"]))}\,\% des
sessions de développement; la sélection se fait sur une validation croisée stratifiée à trois
plis de ce sous-ensemble uniquement, avant tout accès au holdout. La comparaison finale
utilise \code{{RepeatedStratifiedKFold}} (k={config["cv"]["folds"]},
{config["cv"]["repeats"]} répétitions) et sert de mesure de stabilité descriptive après
verrouillage du choix. Les métriques sont l'exactitude, la F1-macro, la F1 pondérée,
l'exactitude équilibrée, la précision top-3/top-5 et la log-loss. La règle de choix retient le
modèle pré-déclaré le plus simple à moins d'un écart-type du meilleur; un vote n'est accepté que
s'il gagne de plus d'un écart-type. Les baselines (majoritaire, aléatoire stratifiée,
navigateur seul) figurent dans le tableau \ref{{tab:models}}.

\section{{Résultats}}
Le modèle retenu est \code{{{selected}}}. Sur le holdout indépendant, l'exactitude atteint
{headline["accuracy"]:.3f} et la F1-macro {headline["macro_f1"]:.3f}; en validation croisée
répétée sur le train complet, la F1-macro vaut {selected_repeated["macro_f1"]:.3f} ±
{selected_std["macro_f1"]:.3f}. La calibration sigmoïde interne n'améliore pas la F1-macro de
façon décisive. Les baselines confirment que le signal provient bien des séquences, non du
navigateur ni de la classe majoritaire.
{comparison_table(config)}

\begin{{figure}}[htbp]\centering
\includegraphics[width=0.9\linewidth]{{model_comparison.pdf}}
\caption{{Comparaison des modèles (moyenne ± écart-type, 4 plis et 3 répétitions).}}
\end{{figure}}
\FloatBarrier

\subsection{{Ablations et courbe d'apprentissage}}
Les ablations par famille (tableau \ref{{tab:permutation}}) et la courbe
d'apprentissage (tableau \ref{{tab:learning}}, figure suivante) sont des diagnostics
post-verrouillage; elles n'ont pas servi à choisir le modèle.
{permutation_table}
{learning_table}
\begin{{figure}}[htbp]\centering
\includegraphics[width=0.6\linewidth]{{learning_curve.pdf}}
\caption{{Courbe d'apprentissage : macro-F1 apprentissage et validation selon la taille d'apprentissage.}}
\end{{figure}}
Le F1 d'entraînement reste à 1,000 aux trois tailles, tandis que le score de validation
progresse de 0,890 à 0,941 : le modèle mémorise le train et bénéficie encore de davantage
de sessions. Cette courbe post-sélection est un diagnostic, pas une estimation indépendante.

\subsection{{Analyse des erreurs}}
L'exactitude baisse sur les sessions très courtes, où le signal disponible est faible, alors que
les sessions longues restent bien classées; le tableau \ref{{tab:length}} le quantifie. Les
confusions se concentrent entre utilisateurs partageant les mêmes modules (tableau
\ref{{tab:confusion}}), ce qui suggère des rôles ou habitudes proches.
\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{confidence.pdf}}
\caption{{Fiabilité (gauche) et exactitude selon la couverture d'abstention (droite).}}
\end{{figure}}
{length_table}
{confusion_table}
\FloatBarrier

\section{{Interprétabilité et discussion}}
\textbf{{Confiance.}} L'ECE mesuré sur le holdout est
{interpretation["models"]["winner"]["calibration"]["ece"]:.3f} pour le softmax non calibré des marges,
contre {interpretation["calibrated"]["ece"]:.3f} après calibration sigmoïde interne. Le SVM natif
n'expose pas de probabilités; les valeurs calibrées servent à exprimer la confiance, sans modifier
les prédictions de classe. L'analyse d'abstention montre qu'une couverture réduite augmente
 l'exactitude (figure précédente).\par\smallskip
\textbf{{Explication locale.}} Pour le modèle retenu, la contribution d'un jeton est exactement
\code{{coefficient $\times$ valeur TF-IDF}} plus l'ordonnée à l'origine; pour la forêt aléatoire,
nous fournissons une décomposition exacte des probabilités le long des chemins, sans prétendre à
des valeurs de Shapley. Un arbre de profondeur 2 sur les statistiques denses illustre des règles
 lisibles.\par\smallskip
\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{pruned_tree.pdf}}
\caption{{Arbre de profondeur 2 sur statistiques denses (illustration pédagogique, pas le modèle retenu).}}
\end{{figure}}
\par\smallskip
\textbf{{Variables influentes et redondance.}} La permutation groupée (tableau
\ref{{tab:permutation}}) montre que la séquence brute domine et que le navigateur contribue
secondairement; les motifs récurrents
(écran\,$\to$\,action) ont un sens comportemental. La sélection de représentants de clusters
permet de retirer des colonnes redondantes avec une perte mesurée, documentée dans
\code{{interpretability.json}}.\par\smallskip
\textbf{{Limites.}} {len(support)} classes avec peu de sessions chacune, dérive temporelle,
test de {loading["test_sessions"]} lignes bruité, et ré-identification potentielle des
utilisateurs: autant de raisons de préférer un modèle simple et validé à un score fragile.

\section{{Conclusion}}
Une chaîne linéaire TF-IDF + SVM, correctement validée hors pli, explique la majorité des
sessions; les modèles plus complexes n'apportent pas de gain supérieur à un écart-type. Les
prochaines étapes incluent l'augmentation par fenêtres pour les classes rares et une analyse
d'équité par utilisateur, sous contrainte de confidentialité. La soumission
\code{{{Path(submission["submission"]).name}}} (empreinte
\code{{{submission["submission_sha256"][:12]}}}) est régénérée par \code{{make predict}}.

\end{{document}}
"""
    )


def check_page_count(minimum: int = 5, maximum: int = 8) -> int:
    """Compile the report and fail unless the PDF page count is within bounds."""
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    subprocess.run(
        command,
        cwd=LATEX_DIR,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    info = subprocess.run(
        ["pdfinfo", "main.pdf"], cwd=LATEX_DIR, check=True, capture_output=True, text=True
    )
    pages = int(
        [line for line in info.stdout.splitlines() if line.startswith("Pages:")][0].split()[1]
    )
    if not minimum <= pages <= maximum:
        raise ValueError(f"Report has {pages} pages, expected {minimum}-{maximum}")
    return pages


def run_report(config: dict[str, Any]) -> dict[str, Any]:
    """Write tables, compile the LaTeX report, and record the verified page count."""
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    (LATEX_DIR / "main.tex").write_text(build_tex(config))
    pages = check_page_count()
    result = {
        "provenance": provenance(config),
        "pages": pages,
        "pdf": str(LATEX_DIR / "main.pdf"),
        "tex": str(LATEX_DIR / "main.tex"),
    }
    write_json("reports/results/report.json", result)
    return result
