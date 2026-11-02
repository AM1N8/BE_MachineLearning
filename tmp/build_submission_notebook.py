import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
old = json.loads((root / 'datamanip/006_sequences.ipynb').read_text(encoding='utf-8'))
definitions = {}
for cell in old['cells']:
    if cell['cell_type'] == 'code':
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = ast.unparse(node)
cells = []

def section(title, source):
    cells.append(dict(cell_type='markdown', metadata={}, source=[f'## {title}\n']))
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                      source=source.strip().splitlines(keepends=True)))

section('Imports', '''
from pathlib import Path
from collections import Counter
import json
import re
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
''')
section('Configuration', '''
RANDOM_STATE = 42
C_SVM = 20.0
POIDS_SVM = 0.90
FAMILLES = ['contextes', 'actions', 'sequences', 'parcours']
POIDS_FAMILLES = [1.0, 1.0, 1.0, 0.5]
DATA_DIR = None
if DATA_DIR is None:
    candidats = [p / 'data' for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / 'data/train.csv').is_file()]
    if not candidats and Path('/kaggle/input').exists():
        candidats = [p.parent for p in Path('/kaggle/input').rglob('train.csv')
                     if (p.parent / 'test.csv').is_file()]
    assert len(candidats) == 1, 'Renseigner DATA_DIR avec le dossier des données.'
    DATA_DIR = candidats[0]
DATA_DIR = Path(DATA_DIR)
OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else DATA_DIR.parent / 'datamanip'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print({'C': C_SVM, 'poids_svm': POIDS_SVM, 'poids_extra_trees': round(1 - POIDS_SVM, 2)})
''')
section('Données', '''
def charger_sessions(path, etiquete):
    lignes = []
    with path.open(encoding='utf-8-sig') as fichier:
        for ligne in fichier:
            champs = ligne.rstrip('\\r\\n').split(',')
            assert len(champs) >= (2 if etiquete else 1)
            lignes.append({'util': champs[0] if etiquete else None,
                           'navigateur': champs[1] if etiquete else champs[0],
                           'traces': champs[2:] if etiquete else champs[1:]})
    return pd.DataFrame(lignes)

train_df = charger_sessions(DATA_DIR / 'train.csv', True)
test_df = charger_sessions(DATA_DIR / 'test.csv', False)
soumission = pd.read_csv(DATA_DIR / 'sample_submission.csv')
assert list(soumission.columns) == ['RowId', 'prediction']
assert len(soumission) == len(test_df)
assert soumission.RowId.is_unique
assert np.array_equal(soumission.RowId.to_numpy(), np.arange(1, len(test_df) + 1))
df = pd.concat([train_df, test_df], ignore_index=True)
y = train_df.util.to_numpy()
classes = np.unique(y)
navigateurs = df.navigateur.to_numpy()
indices_train = np.arange(len(train_df))
indices_test = np.arange(len(train_df), len(df))
print({'train': len(train_df), 'test': len(test_df), 'utilisateurs': len(classes)})
''')
section('Préparation', '\n\n'.join(definitions[n] for n in [
    'est_temps', 'type_action', 'contextes', 'ajouter_comptages', 'ngrammes']) + '''

df['actions'] = df.traces.map(lambda traces: [t for t in traces if not est_temps(t)])
''')
section('Variables des arbres', definitions['caracteristiques_reference'] + '''

X_reference = [caracteristiques_reference(r) for r in df.to_dict('records')]
''')
section('Variables des séquences', definitions['representations'] + '''

REPRESENTATIONS = [representations(a, n) for a, n in zip(df.actions, navigateurs)]
''')
section('TF-IDF par blocs', '''
def encoder_blocs(train, evaluation):
    blocs_train, blocs_eval = [], []
    for famille, poids_bloc in zip(FAMILLES, POIDS_FAMILLES):
        vecteur = DictVectorizer(dtype=np.float32)
        a = vecteur.fit_transform([REPRESENTATIONS[i][famille] for i in train])
        b = vecteur.transform([REPRESENTATIONS[i][famille] for i in evaluation])
        garder = np.asarray((a != 0).sum(axis=0)).ravel() >= 2
        a, b = a[:, garder], b[:, garder]
        tfidf = TfidfTransformer(sublinear_tf=True)
        blocs_train.append(poids_bloc * tfidf.fit_transform(a))
        blocs_eval.append(poids_bloc * tfidf.transform(b))
    return (normalize(sparse.hstack(blocs_train, format='csr')),
            normalize(sparse.hstack(blocs_eval, format='csr')))
''')
section('Extra Trees', definitions['predire_arbres'])
section('SVM', definitions['filtrer_navigateur'] + '''

def predire_svm(train, evaluation):
    a, b = encoder_blocs(train, evaluation)
    modele = LinearSVC(C=C_SVM, dual=True, max_iter=30000, random_state=RANDOM_STATE)
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        modele.fit(a, y[train], sample_weight=np.ones(len(train)))
        marges = modele.decision_function(b)
    resultat = np.zeros((len(evaluation), len(classes)), dtype=np.float32)
    resultat[:, np.searchsorted(classes, modele.classes_)] = softmax(3 * marges, axis=1)
    return filtrer_navigateur(resultat, train, evaluation)
''')
section('Ensemble', '''
def predire_ensemble(train, evaluation):
    svm = predire_svm(train, evaluation)
    arbres = predire_arbres(train, evaluation)
    scores = POIDS_SVM * svm + (1 - POIDS_SVM) * arbres
    assert np.isfinite(scores).all()
    assert np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    return classes[scores.argmax(axis=1)]
''')
section('Validation de développement', '''
apprentissage, validation = train_test_split(
    indices_train, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
prediction_validation = predire_ensemble(apprentissage, validation)
resultats = pd.DataFrame([{
    'F1 macro': f1_score(y[validation], prediction_validation, average='macro', zero_division=0),
    'F1 pondéré': f1_score(y[validation], prediction_validation, average='weighted', zero_division=0),
    'Accuracy': accuracy_score(y[validation], prediction_validation),
    'Erreurs': int(np.sum(y[validation] != prediction_validation))
}])
print(resultats.to_string(index=False))
''')
section('Entraînement complet', '''
predictions_test = predire_ensemble(indices_train, indices_test)
assert len(predictions_test) == len(test_df)
assert np.isin(predictions_test, classes).all()
print({'sessions_apprentissage': len(indices_train), 'predictions': len(predictions_test)})
''')
section('Soumission Kaggle', '''
soumission['prediction'] = predictions_test
assert not soumission.isna().any().any()
fichier_soumission = OUTPUT_DIR / 'submission.csv'
soumission.to_csv(fichier_soumission, index=False)
verification = pd.read_csv(fichier_soumission)
pd.testing.assert_frame_equal(verification, soumission)
print(soumission.head().to_string(index=False))
print(fichier_soumission.resolve())
''')
notebook = dict(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}}, nbformat=4, nbformat_minor=5)
for i, cell in enumerate(cells):
    cell['id'] = f'kaggle-{i:02d}'
path = root / 'datamanip/007_soumission_kaggle.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
runner = (root / 'tmp/execute_notebook_006.py').read_text(encoding='utf-8')
(root / 'tmp/execute_submission_notebook.py').write_text(
    runner.replace('datamanip/006_sequences.ipynb', 'datamanip/007_soumission_kaggle.ipynb'),
    encoding='utf-8')
print(path)
