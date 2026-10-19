"""Aggregate completed phase-two experiments and verify their provenance."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from copilote.io import file_hash, read_sessions, write_json


def table(frame, columns):
    """Render result fields without an optional Markdown dependency."""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |"
        )
    return "\n".join(lines)


def main():
    active = json.loads(Path("reports/results/phase2/active.json").read_text())
    directory = Path(active["directory"])
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert file_hash(f"src/copilote/{name}.py") == digest
    assert file_hash("data/train.csv") == manifest["train"]
    assert file_hash("uv.lock") == manifest["lock"]
    assert file_hash("submissions/best_verified.csv") == manifest["baseline_submission"]
    summaries, lengths, persistent_changes = {}, [], []
    persistent = pd.read_csv(directory / "persistent_sessions.csv")
    for stage, filename in (
        ("research", "comparison.csv"),
        ("confirm", "confirmation.csv"),
        ("seed44", "seed44_comparison.csv"),
    ):
        summary = pd.read_csv(directory / filename)
        summaries[stage] = summary
        splits = json.loads((directory / f"{stage}_splits.json").read_text())
        frames = {}
        for name in summary.experiment:
            parts = []
            for fold, split in enumerate(splits):
                frame = pd.read_csv(directory / f"{stage}_{name}_{fold}.csv")
                assert frame.row_id.tolist() == split["validation"]
                assert not set(split["train"]) & set(split["validation"])
                parts.append(frame)
                if name == "reranker":
                    audit = json.loads(
                        (directory / f"{stage}_reranker_{fold}_inner.json").read_text()
                    )
                    heldout = []
                    for inner in audit["inner_splits"]:
                        assert not set(inner["train"]) & set(inner["heldout"])
                        assert set(inner["train"]) | set(inner["heldout"]) == set(split["train"])
                        heldout.extend(inner["heldout"])
                    assert sorted(heldout) == sorted(split["train"])
            frames[name] = pd.concat(parts, ignore_index=True)
        base = frames["baseline"]
        for name, frame in frames.items():
            assert frame.truth.equals(base.truth)
            correct, previous = frame.prediction == frame.truth, base.prediction == base.truth
            bins = pd.cut(
                frame.length,
                [-1, 20, 100, 500, np.inf],
                labels=["0–20", "21–100", "101–500", ">500"],
            )
            for label in bins.cat.categories:
                mask = bins == label
                lengths.append(
                    dict(
                        stage=stage,
                        experiment=name,
                        length_bin=label,
                        count=int(mask.sum()),
                        errors=int((~correct & mask).sum()),
                        corrected=int((correct & ~previous & mask).sum()),
                        introduced=int((~correct & previous & mask).sum()),
                    )
                )
            mask = frame.row_id.isin(persistent.row_id)
            persistent_changes.append(
                dict(
                    stage=stage,
                    experiment=name,
                    observations=int(mask.sum()),
                    errors=int((~correct & mask).sum()),
                    corrected=int((correct & ~previous & mask).sum()),
                    introduced=int((~correct & previous & mask).sum()),
                )
            )
    length_frame = pd.DataFrame(lengths)
    length_frame.to_csv(directory / "length_paired.csv", index=False)
    persistent_frame = pd.DataFrame(persistent_changes)
    persistent_frame.to_csv(directory / "persistent_paired.csv", index=False)
    eligible = set.intersection(*(set(s.loc[s.passes, "experiment"]) for s in summaries.values()))
    assert not eligible, "A candidate passed: implement separate reviewed challenger export."
    submitted = pd.read_csv("submissions/best_verified.csv")
    sample = pd.read_csv("data/sample_submission.csv")
    assert list(submitted.columns) == list(sample.columns)
    assert submitted.iloc[:, 0].equals(sample.iloc[:, 0])
    assert not submitted.isna().any().any()
    with threadpool_limits(limits=2):
        prediction = joblib.load("artifacts/best_verified.joblib").predict(
            read_sessions("data/test.csv", labelled=False)
        )
    assert np.array_equal(prediction, submitted.iloc[:, 1].to_numpy())
    assert not Path("submissions/phase2_challenger.csv").exists()
    decision = dict(
        decision="KEEP BASELINE",
        signature=active["signature"],
        baseline_sha256=manifest["baseline_submission"],
        challenger_generated=False,
        model_reload_exact=True,
        rows=len(submitted),
        outer_evaluations=sum(len(s) * 8 for s in summaries.values()),
        inner_split_audits=24,
        grouped_validation="not run: all inferred groups are singletons",
        metric_verified=False,
    )
    write_json(directory / "decision.json", decision)
    report = Path("reports/phase2_research.md")
    protocol = report.read_text().split("\n## Résultats exécutés")[0]
    sections = [
        protocol,
        "\n## Résultats exécutés\n",
        "**Décision : KEEP BASELINE.** Aucun challenger produit. 136 évaluations externes ; trois protocoles 4×2 sur les mêmes données, pas trois jeux de données indépendants.\n",
    ]
    columns = [
        "experiment",
        "macro_f1",
        "accuracy",
        "weighted_f1",
        "delta_f1",
        "lower95",
        "corrected",
        "introduced",
        "passes",
    ]
    for stage, summary in summaries.items():
        sections.extend([f"### {stage}\n", table(summary, columns), ""])
    sections.append(
        "Les signatures contextuelles gagnent sur les trois protocoles et les six répétitions : 40 corrections contre 16 dégradations sur 19 674 prédictions répétées. Cependant aucune borne corrigée n'est positive : hypothèse prometteuse, non promue. Les transitions et positions échouent dès l'ablation historique. Le contexte et les extrémités ne confirment pas leur macro-F1 avec la graine 44. Aucun assemblage n'a été ajouté après examen des scores externes.\n\nLe reranker corrige 14 erreurs mais en crée 16 : moins destructeur que l'ancien classifieur local, mais sans gain reproductible. Ses seuils sont choisis exclusivement dans le train externe. Les probabilités internes ne constituent pas une garantie de calibration externe.\n\nLes validations réutilisent des données déjà explorées : le biais adaptatif résiduel demeure. Les bornes corrigées sont un garde-fou conservateur, pas une preuve statistique indépendante. Le score Kaggle futur est inconnu.\n\nSources numériques : `"
        + str(directory)
        + "/*.json|csv`. `decision.json` vérifie les empreintes, les 24 audits internes et la reproduction exacte des 324 prédictions conservées. Reproduction complète : `make phase2`. Les tests groupés n'ont pas été exécutés, aucun groupe non trivial n'étant établi."
    )
    report.write_text("\n".join(sections))
    audit = json.loads((directory / "data_audit.json").read_text())
    error_report = [
        "# Phase 2 — dépendances et erreurs\n",
        "## Structure et métrique\n",
        f"{audit['rows']} sessions, {audit['classes']} utilisateurs. Les identifiants sont des positions de ligne, sans date absolue ni identifiant parent. « F1 score moyen » reste ambigu ; macro-F1, F1 pondéré et accuracy sont conservés. Aucun utilisateur multi-navigateur observé : hypothèse empirique du masque, non garantie de production.\n",
        "Aucun doublon exact ni paire proche selon Jaccard des trigrammes ≥0,9 et ratio de longueur ≥0,9. Tous les groupes inférés sont des singletons. Cela ne prouve pas l'indépendance : d'autres proximités et une provenance cachée restent possibles. Regrouper par utilisateur supprimerait les classes à prédire du train, ce qui évaluerait une autre tâche.\n",
        f"Même utilisateur entre lignes voisines : {audit['adjacent_same_label_fraction']:.6f}, contre {audit['random_expected_adjacent_fraction']:.6f} attendu aléatoirement. Corrélation de rang index/première horloge : {audit['row_first_clock_spearman']:.6f}. Pas d'ordre chronologique inter-session établi. Les écrans sont explicites sur {audit['explicit_screen_events']} événements et absents sur {audit['missing_screen_events']} : la propagation reste une hypothèse locale à la session.\n",
        "## Les 78 sessions toujours mal classées\n",
        f"Longueur médiane : {persistent.length.median():.0f} événements ; moyenne : {persistent.length.mean():.6f}. Les trois plus courtes ont 0, 1 et 2 événements. Les horloges ne sont pas comptées comme événements. Les voisins et recouvrements par utilisateur sont exportés dans `persistent_sessions.csv`, sans correction manuelle des labels ni utilisation de ces statistiques globales comme features.\n",
        table(
            persistent_frame,
            ["stage", "experiment", "observations", "errors", "corrected", "introduced"],
        ),
        "\nCes sessions ont été repérées sur les plis historiques : leur analyse sur d'autres graines reste descriptive, non indépendante.\n",
        "## Corrections par longueur (zéro événement inclus)\n",
        table(
            length_frame[length_frame.experiment.isin(["baseline", "signatures", "reranker"])],
            ["stage", "experiment", "length_bin", "count", "errors", "corrected", "introduced"],
        ),
        "\nLes dénombrements sont des observations répétées, pas des sessions uniques. Toutes les ablations sont disponibles dans `length_paired.csv`. Le reranker conserve le contexte global, les alternatives hors top-2 et les couvertures OOV, mais ne satisfait pas les critères. Aucun changement de soumission.\n",
        f"Source : `{directory}`.",
    ]
    Path("reports/phase2_error_analysis.md").write_text("\n".join(error_report))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
