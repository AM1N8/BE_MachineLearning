
from pathlib import Path
from collections import Counter
import contextlib
import io
import json
import re
import warnings

import numpy as np
import pandas as pd
import sklearn
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datamanip' / 'resultats_006'


def scores(y, pred):
    return dict(f1_macro=f1_score(y, pred, average='macro', zero_division=0),
                f1_pondere=f1_score(y, pred, average='weighted', zero_division=0),
                accuracy=accuracy_score(y, pred))


def add_counts(target, prefix, values):
    for value, count in Counter(values).items():
        target[prefix + json.dumps(value, ensure_ascii=False)] = count


def prepare():
    # Reuse exactly the baseline feature code, replacing presentation only.
    nb = json.loads((ROOT / 'datamanip/002_features.ipynb').read_text(encoding='utf-8'))
    env = {'__name__': '__main__', 'display': lambda *a, **k: None}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']).replace('from IPython.display import display', '')
                exec(compile(source, '002_features.ipynb', 'exec'), env)
    df, base = env['df_train'], env['X'].tolist()
    full, seq, enriched = [], [], []
    for original, row in zip(base, df.to_dict('records')):
        actions, verbs = row['actions'], row['types_actions']
        counts = {k: v for k, v in original.items()
                  if '_nombre::' in k or k == 'navigateur'}
        add_counts(counts, 'complete::', actions)
        full.append(counts)
        sequences = dict(counts)
        for size in (2, 3):
            add_counts(sequences, f'complete{size}::', zip(*(actions[j:] for j in range(size))))
            add_counts(sequences, f'verbe{size}::', zip(*(verbs[j:] for j in range(size))))
        screens = row['ecrans_contextes']
        transitions = list(zip(screens, screens[1:]))
        add_counts(sequences, 'transition::', transitions)
        seq.append(sequences)
        tree = dict(original)
        add_counts(tree, 'complete::', actions)
        for size in (2, 3):
            add_counts(tree, f'verbe{size}::', zip(*(verbs[j:] for j in range(size))))
        add_counts(tree, 'transition::', transitions)
        windows, count = [], 0
        for trace in row['traces']:
            if re.fullmatch(r't\d+', trace):
                windows.append(count)
                count = 0
            else:
                count += 1
        windows.append(count)
        # Counts between markers; no assumption that markers are time differences.
        tree.update({f'rythme::{name}': float(value) for name, value in {
            'moyenne': np.mean(windows), 'ecart_type': np.std(windows),
            'median': np.median(windows), 'maximum': max(windows),
            'part_fenetres_vides': np.mean(np.asarray(windows) == 0),
        }.items()})
        enriched.append(tree)
    y = df['util'].to_numpy()
    train, valid = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    assert set(train).isdisjoint(valid)
    assert set(y[train]) == set(y[valid])
    data = dict(df=df, y=y, classes=np.unique(y), train=train, valid=valid,
                views=dict(base=base, full=full, seq=seq, enriched=enriched))
    OUT.mkdir(exist_ok=True)
    print(f'{len(y)} sessions, {len(data["classes"])} classes; train={len(train)}, validation={len(valid)}')
    return data


def predict(data, config, train, valid):
    kind, view, c = config
    rows, y, classes = data['views'][view], data['y'], data['classes']
    def fit_one(tr, va, specialist=False):
        def features(ids):
            return [{k: v for k, v in rows[i].items() if not (specialist and k == 'navigateur')} for i in ids]
        vec = DictVectorizer(dtype=np.float32)
        a = vec.fit_transform(features(tr))
        b = vec.transform(features(va))
        if kind == 'svm':
            # Vocabulary filtering and IDF fitted exclusively on training fold.
            keep = np.asarray((a != 0).sum(axis=0)).ravel() >= 2
            tfidf = TfidfTransformer(sublinear_tf=True)
            a = tfidf.fit_transform(a[:, keep]); b = tfidf.transform(b[:, keep])
            model = LinearSVC(C=c, dual=True, max_iter=20000, random_state=42)
        else:
            model = ExtraTreesClassifier(n_estimators=200, max_features=.5,
                    class_weight='balanced', n_jobs=2, random_state=42)
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(a, y[tr])
        if kind == 'svm':
            # A score transformation for blending, not calibrated probabilities.
            values = softmax(3 * model.decision_function(b), axis=1)
        else:
            values = model.predict_proba(b)
        result = np.zeros((len(va), len(classes)), dtype=np.float32)
        result[:, np.searchsorted(classes, model.classes_)] = values
        return result
    if kind != 'specialists':
        return fit_one(train, valid)
    browsers = np.asarray([r['navigateur'] for r in rows])
    result = np.zeros((len(valid), len(classes)), dtype=np.float32)
    for browser in np.unique(browsers[valid]):
        tr = train[browsers[train] == browser]
        positions = np.flatnonzero(browsers[valid] == browser)
        va = valid[positions]
        if len(tr) == 0:
            result[positions] = fit_one(train, va)
        elif len(np.unique(y[tr])) == 1:
            result[positions, np.searchsorted(classes, y[tr[0]])] = 1
        else:
            result[positions] = fit_one(tr, va, specialist=True)
    return result


def evaluate_cv(data, configs, seed=42):
    train, y, classes = data['train'], data['y'], data['classes']
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, y[train]))
    predictions, results = {}, []
    for name, config in configs.items():
        out = np.zeros((len(train), len(classes)), dtype=np.float32)
        per_fold = []
        for fold, (a, b) in enumerate(splits, 1):
            out[b] = predict(data, config, train[a], train[b])
            metric = scores(y[train[b]], classes[out[b].argmax(axis=1)])
            per_fold.append(metric['f1_macro'])
            print(f'{name} | graine {seed} | pli {fold}/3 | F1 {metric["f1_macro"]:.4f}', flush=True)
        predictions[name] = out
        results.append(dict(modele=name, f1_macro_cv=float(np.mean(per_fold)),
                            ecart_type=float(np.std(per_fold)),
                            **scores(y[train], classes[out.argmax(axis=1)])))
        pd.DataFrame(results).to_csv(OUT / f'cv_{seed}.csv', index=False)
    return predictions, pd.DataFrame(results), splits


def search(data):
    configs = {'Extra Trees reference': ('specialists', 'base', None),
               'Extra Trees enrichi': ('specialists', 'enriched', None)}
    for view in ('full', 'seq'):
        for c in (1., 5., 20.):
            configs[f'SVM {view} C={c:g}'] = ('svm', view, c)
    oof, table, splits = evaluate_cv(data, configs)
    best_tree = table[table.modele.str.startswith('Extra')].sort_values('f1_macro_cv').iloc[-1].modele
    best_svm = table[table.modele.str.startswith('SVM')].sort_values('f1_macro_cv').iloc[-1].modele
    blends = []
    for weight in (.25, .5, .75):
        name = f'Ensemble poids SVM={weight}'
        out = (1-weight)*oof[best_tree] + weight*oof[best_svm]
        pred = data['classes'][out.argmax(axis=1)]
        fold_scores = [scores(data['y'][data['train'][b]], pred[b])['f1_macro'] for _, b in splits]
        blends.append(dict(modele=name, f1_macro_cv=float(np.mean(fold_scores)),
                           ecart_type=float(np.std(fold_scores)), **scores(data['y'][data['train']], pred)))
        configs[name] = ('blend', best_tree, best_svm, weight)
        oof[name] = out
    table = pd.concat([table, pd.DataFrame(blends)], ignore_index=True).sort_values('f1_macro_cv', ascending=False)
    selected = table.iloc[0].modele
    table.to_csv(OUT / 'selection_cv.csv', index=False)
    np.savez_compressed(OUT / 'predictions_oof.npz', **oof)
    print(table.to_string(index=False)); print('Choix CV :', selected)
    return dict(configs=configs, oof=oof, table=table, selected=selected)


def validate(data, experiment):
    configs, selected = experiment['configs'], experiment['selected']
    def fixed_predict(name, train, valid):
        config = configs[name]
        if config[0] == 'blend':
            _, tree, svm, w = config
            return (1-w)*predict(data, configs[tree], train, valid) + w*predict(data, configs[svm], train, valid)
        return predict(data, config, train, valid)
    # A second seed checks sensitivity; it is not an independent test.
    stable = []
    for name in dict.fromkeys(['Extra Trees reference', selected]):
        for fold, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=137).split(data['train'], data['y'][data['train']]), 1):
            tr, va = data['train'][a], data['train'][b]
            p = fixed_predict(name, tr, va)
            stable.append(dict(modele=name, pli=fold, **scores(data['y'][va], data['classes'][p.argmax(axis=1)])))
            print(f'Stabilite {name}, pli {fold}: {stable[-1]["f1_macro"]:.4f}', flush=True)
    pd.DataFrame(stable).to_csv(OUT / 'stabilite_137.csv', index=False)
    results, preds = [], {}
    for name in dict.fromkeys(['Extra Trees reference', selected]):
        p = fixed_predict(name, data['train'], data['valid'])
        pred = data['classes'][p.argmax(axis=1)]
        preds[name] = pred
        results.append(dict(modele=name, **scores(data['y'][data['valid']], pred)))
    result = pd.DataFrame(results)
    result.to_csv(OUT / 'validation.csv', index=False)
    df = data['df'].iloc[data['valid']]
    diagnostic = pd.DataFrame(dict(session_id=df.session_id.to_numpy(), util_reel=df.util.to_numpy(),
        navigateur=df.navigateur.to_numpy(), nb_actions=df.nb_actions.to_numpy(),
        sessions_apprentissage=[Counter(data['y'][data['train']])[u] for u in df.util],
        reference=preds['Extra Trees reference'], prediction=preds[selected]))
    diagnostic['erreur'] = diagnostic.util_reel != diagnostic.prediction
    diagnostic['corrige_reference'] = (diagnostic.reference != diagnostic.util_reel) & ~diagnostic.erreur
    diagnostic['regression_reference'] = (diagnostic.reference == diagnostic.util_reel) & diagnostic.erreur
    diagnostic.to_csv(OUT / 'diagnostic_validation.csv', index=False)
    report = classification_report(df.util, preds[selected], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT / 'rapport_utilisateurs.csv')
    assert len(diagnostic) == len(data['valid'])
    assert set(preds[selected]).issubset(data['classes'])
    assert np.isfinite(result[['f1_macro', 'f1_pondere', 'accuracy']]).all().all()
    summary = dict(selected=selected, config=configs[selected], sklearn=sklearn.__version__,
        validation=results, stabilite=stable, objectif_f1_macro=.97,
        objectif_atteint=bool(results[-1]['f1_macro'] >= .97),
        limites='Selection sur CV; graine 137 = sensibilite, pas test independant. Validation 20% deja consultee dans 003-005. Moyenne Kaggle a confirmer.')
    (OUT / 'resume.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result.to_string(index=False))
    print('Corrections :', diagnostic.corrige_reference.sum(), 'Regressions :', diagnostic.regression_reference.sum())
    print('Tous les controles sont valides. Objectif 97% atteint :', summary['objectif_atteint'])
    return result, diagnostic


if __name__ == '__main__':
    dataset = prepare()
    experiment = search(dataset)
    validate(dataset, experiment)
