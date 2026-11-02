import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root / "datamanip"
initial = json.loads((data / "resultats_006_notebook/resume.json").read_text(encoding="utf-8"))
latest = json.loads((data / "resultats_008/recette.json").read_text(encoding="utf-8"))
with (data / "resultats_008/stabilite.csv").open(encoding="utf-8", newline="") as f:
    stability = list(csv.DictReader(f))


def validations(name):
    return [
        {
            "graine": int(row["graine"]),
            "nombre_plis": 4,
            "f1_macro_oof": float(row["f1_macro_oof"]),
            "moyenne_f1_macro_plis": float(row["moyenne_plis"]),
            "accuracy_oof": float(row["accuracy"]),
            "nombre_erreurs": int(row["erreurs"]),
        }
        for row in stability if row["modele"] == name
    ]


def artifact(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    assert columns == ["RowId", "prediction"] and len(rows) == 324
    return {
        "chemin_relatif_projet": path.relative_to(root).as_posix(),
        "colonnes": columns,
        "nombre_predictions": len(rows),
        "sha256_fichier_local_actuel": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def svm(c, representation, weight):
    return {
        "estimateur": "LinearSVC",
        "representation": representation,
        "poids": weight,
        "parametres": {
            "C": c, "dual": True, "class_weight": None,
            "max_iter": 30000, "random_state": 42,
        },
        "conversion_scores": "softmax(3 * decision_function), puis filtrage par navigateur",
    }


trees = {
    "estimateur": "ExtraTreesClassifier",
    "representation": "Statistiques de session et comptages des actions et contextes",
    "specialisation": "Un modèle par navigateur, avec repli global si navigateur inconnu",
    "poids": 0.1,
    "parametres": {
        "n_estimators": 200, "max_features": 0.5,
        "class_weight": "balanced", "n_jobs": 2, "random_state": 42,
    },
}
registry = {
    "version_schema": 1,
    "description": "Registre des deux ensembles développés dans ce travail et soumis par l'utilisateur ; les autres candidats non soumis sont exclus.",
    "competition": {
        "nom": "Qui utilise mon appli ? v2026/2027 - Groupe 2",
        "url": "https://www.kaggle.com/competitions/qui-utilise-mon-appli-v-2026-2027-groupe-2",
        "metrique_annoncee": "F1 score moyen",
        "type_score_enregistre": "public",
    },
    "donnees": {
        "sessions_apprentissage": 3279, "utilisateurs": 247,
        "sessions_test": 324,
        "sha256_train": latest["sha256_train_fichier"],
    },
    "pretraitement_svm_commun": {
        "familles": ["contextes", "actions", "sequences", "parcours"],
        "poids_familles": [1.0, 1.0, 1.0, 0.5],
        "frequence_documentaire_minimale": 2,
        "tfidf_sublinear_tf": True,
        "normalisation": "L2 par bloc TF-IDF puis L2 après concaténation pondérée",
        "filtrage_navigateur": "Exclut les utilisateurs non observés avec ce navigateur dans l'apprentissage, puis renormalise les scores",
    },
    "modeles_soumis": [
        {
            "id": "soumission_001",
            "nom": initial["modele_retenu"],
            "notebook": "datamanip/notebook_v01.ipynb",
            "source_recette": "datamanip/resultats_006_notebook/resume.json",
            "fichier_csv": artifact(data / "submission.csv"),
            "composants": [svm(20.0, "log", 0.9), trees],
            "score_kaggle": {
                "valeur": 0.971, "type": "public",
                "source": "Vérifié dans la page Kaggle au cours de la conversation",
            },
            "validation_locale": {
                "holdout_developpement_f1_macro": initial["f1_macro_validation"],
                "holdout_fraction": 0.2, "holdout_graine": 42,
                "validation_croisee_comparative_008": validations("reference_90_10"),
            },
        },
        {
            "id": "soumission_002",
            "nom": latest["modele"],
            "notebook": "datamanip/008_optimisation_cv.ipynb",
            "source_recette": "datamanip/resultats_008/recette.json",
            "fichier_csv": artifact(data / "resultats_008/submission_008.csv"),
            "composants": [
                svm(50.0, "log", 0.675),
                svm(20.0, "contexte_rythme", 0.225), trees,
            ],
            "variables_supplementaires": {
                "signatures_contextuelles": "Associations écran, configuration, chaîne et action",
                "poids_signatures": 0.5,
                "rythme": "Actions par fenêtre, intervalles, répétitions et actions par seconde",
                "transformation_rythme": "log1p puis normalisation L2",
                "poids_rythme": 0.25,
            },
            "score_kaggle": {
                "valeur": 0.971, "type": "public",
                "source": "Confirmé par l'utilisateur dans son dernier message",
            },
            "validation_locale": {
                "validation_croisee": validations(latest["modele"]),
            },
        },
    ],
    "notes": [
        "Ce JSON décrit les modèles et leurs résultats ; il ne contient pas leurs poids appris.",
        "Les empreintes CSV identifient les fichiers locaux actuels ; leur identité avec les fichiers téléversés n'a pas été vérifiée par téléchargement Kaggle.",
        "Les scores Kaggle sont enregistrés avec la précision affichée ou rapportée (0.971).",
        "Les deux graines de validation réutilisent les mêmes données et ne constituent pas deux tests indépendants.",
    ],
}
path = data / "modeles_soumis_kaggle.json"
path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
assert len(json.loads(path.read_text(encoding="utf-8"))["modeles_soumis"]) == 2
print(path)
