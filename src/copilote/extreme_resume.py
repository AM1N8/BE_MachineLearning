"""Resume an interrupted sequential Optuna trial without discarding its budget."""

import json

import joblib
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split

from copilote.extreme import ART, ROOT, Campaign


def main():
    active = json.loads((ROOT / "active_run.json").read_text())
    campaign = Campaign(active["config"])
    storage = f"sqlite:///{ART / ('optuna_' + campaign.fingerprint[:12] + '.db')}"
    for summary in optuna.study.get_all_study_summaries(storage=storage):
        stage = summary.study_name
        path = ART / f"sampler_{campaign.fingerprint[:12]}_{stage}.joblib"
        sampler = joblib.load(path)
        study = optuna.load_study(study_name=stage, storage=storage, sampler=sampler)
        for frozen in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            # Replay draws from the last completed trial's persisted sampler state.
            # TPE is univariate here; current RUNNING observations are not training data.
            order = ["view", "C", "min_df", "class_weight", "bigram", "trigram", "weight"]
            for name in order:
                if name in frozen.params:
                    value = sampler.sample_independent(
                        study, frozen, name, frozen.distributions[name]
                    )
                    if value != frozen.params[name]:
                        raise ValueError(
                            f"Sampler replay differs for {stage}/{frozen.number}/{name}"
                        )
            if stage == "exploration":
                indices, _ = train_test_split(
                    np.arange(len(campaign.rows)),
                    test_size=0.2,
                    stratify=campaign.y,
                    random_state=42,
                )
            elif stage == "final_inner":
                indices = np.arange(len(campaign.rows))
            else:
                fold = int(stage.split("_")[0].removeprefix("outer"))
                indices = np.array(
                    json.loads((campaign.directory / "outer_splits.json").read_text())["folds"][
                        fold
                    ]["train"]
                )
            splits = list(
                StratifiedKFold(3, shuffle=True, random_state=42).split(
                    indices, campaign.y[indices]
                )
            )
            trial = optuna.trial.Trial(study, frozen._trial_id)
            try:
                result = campaign.evaluate(
                    f"trial{frozen.number:03d}",
                    frozen.user_attrs["spec"],
                    indices,
                    splits,
                    stage,
                    trial,
                )
                trial.set_user_attr("result", result)
                study.tell(trial, result["mean"]["macro_f1"])
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            joblib.dump(sampler, path)
            study.trials_dataframe().to_csv(
                campaign.directory / f"{stage}_optuna_trials.csv", index=False
            )


if __name__ == "__main__":
    main()
