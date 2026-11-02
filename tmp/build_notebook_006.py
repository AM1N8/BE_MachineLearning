from pathlib import Path
import json
import textwrap
import shutil

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(text).strip().splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                      source=(textwrap.dedent(text).strip()+'\n').splitlines(True)))


md('''
# 006 — Séquences, sessions courtes et amélioration du F1

**Objectif :** améliorer le F1 macro en conservant une démarche vérifiable et un notebook autonome.
Tout le code de préparation, apprentissage, comparaison et diagnostic est présent dans les cellules.
Il ne faut importer ni `sequence_experiment.py`, ni exécuter un autre notebook.

La première expérience donnait **91,87 % de F1 macro sur 656 sessions de validation**,
avec 38 erreurs, dont 36 sur des sessions de 500 actions ou moins. L’objectif reste 97 %,
sans présumer qu’il est atteignable avec les informations disponibles.
''')
md('''
## 1. Protocole et hypothèses

| Problème observé | Expérience proposée |
|---|---|
| Les sessions courtes offrent moins d’occurrences | TF-IDF binaire : présence des motifs |
| Les séquences nombreuses peuvent dominer les contextes | TF-IDF et normalisation séparés par famille |
| Des habitudes peuvent être non linéaires | SVM à noyau RBF sur la représentation normalisée |
| Peu d’exemples courts pour certains utilisateurs | Deux extraits de 200 actions issus uniquement des sessions du pli d’apprentissage |
| Les classes sont déséquilibrées | Variante SVM à classes équilibrées |
| Le gain peut dépendre du découpage | Contrôle de sensibilité sur trois plis avec une seconde graine |

La séparation 80/20 et les trois plis internes utilisent la graine 42 comme auparavant.
Le choix se fait sur le **F1 macro moyen des trois plis d’apprentissage** ; les 20 % réservés
ne sont consultés qu’après fixation du choix. Cette validation a déjà été analysée dans les
expériences précédentes : elle reste une validation de développement, pas un test final indépendant.
Le contrôle avec une seconde graine n’est pas une validation imbriquée. Le test Kaggle n’est pas lu.
La moyenne F1 officielle de Kaggle reste à confirmer.
''')
md('## 2. Bibliothèques et chemins')
code('''
from pathlib import Path
from collections import Counter
import hashlib
import json
import platform
import re
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

try:
    from IPython.display import display
except ImportError:
    def display(objet):
        print(objet.to_string() if isinstance(objet, pd.DataFrame) else objet)

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'data/train.csv').is_file()), None)
assert ROOT is not None, 'Ouvrir le notebook depuis le projet.'
OUT = ROOT / 'datamanip/resultats_006_notebook'
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
DEBUT = perf_counter()
print('Python :', platform.python_version(), '| scikit-learn :', sklearn.__version__)
print('Résultats :', OUT)
''')
md('## 3. Charger les données et réserver la validation')
code('''
sessions = []
with (ROOT / 'data/train.csv').open(encoding='utf-8-sig') as fichier:
    for numero, ligne in enumerate(fichier, start=1):
        champs = ligne.rstrip('\\r\\n').split(',')
        assert len(champs) >= 3
        sessions.append(dict(session_id=numero, util=champs[0],
                             navigateur=champs[1], traces=champs[2:]))
df = pd.DataFrame(sessions)
y = df.util.to_numpy()
classes = np.unique(y)
navigateurs = df.navigateur.to_numpy()
indices_train, indices_validation = train_test_split(
    np.arange(len(df)), test_size=.20, stratify=y, random_state=RANDOM_STATE)
assert set(indices_train).isdisjoint(indices_validation)
assert set(y[indices_train]) == set(y[indices_validation]) == set(classes)
assert min(Counter(y[indices_train]).values()) >= 3
display(pd.DataFrame({'ensemble': ['apprentissage', 'validation'],
    'sessions': [len(indices_train), len(indices_validation)],
    'utilisateurs': [len(set(y[indices_train])), len(set(y[indices_validation]))]}))
''')
md('''
## 4. Fonctions de lecture des traces

Une action complète conserve son contexte. Le type d’action retire les suffixes.
Les marqueurs temporels sont retirés pour construire les séquences d’actions.
Les paires/triplets peuvent donc traverser un marqueur. Une transition d’écran désigne
deux mentions successives, sans prétendre reconstituer l’écran actif.
''')
code('''
def est_temps(trace):
    return re.fullmatch(r't\\d+', trace) is not None

def type_action(action):
    nom = re.split(r'[(<$]', action, maxsplit=1)[0].strip()
    return nom[:-1].rstrip() if nom.endswith('1') else nom

def contextes(actions, motif):
    return [valeur for action in actions for valeur in re.findall(motif, action)]

def ajouter_comptages(cible, prefixe, valeurs):
    for valeur, nombre in Counter(valeurs).items():
        cible[prefixe + json.dumps(valeur, ensure_ascii=False)] = nombre

def ngrammes(valeurs, taille):
    return zip(*(valeurs[j:] for j in range(taille)))

df['actions'] = df.traces.map(lambda traces: [t for t in traces if not est_temps(t)])
df['nb_actions'] = df.actions.map(len)
display(df[['session_id', 'util', 'navigateur', 'nb_actions']].head())
''')
md('''
## 5. Reproduire les caractéristiques de référence

Ces dictionnaires reproduisent les caractéristiques de `002`. Les spécialistes Extra Trees
gardent 200 arbres, `max_features=0.5` et `class_weight="balanced"`. Les caractéristiques
supplémentaires qui avaient dégradé les arbres ne sont pas ajoutées à cette référence.
''')
code('''
def caracteristiques_reference(ligne):
    actions = ligne['actions']
    verbes = [type_action(a) for a in actions]
    temps = [int(t[1:]) for t in ligne['traces'] if est_temps(t)]
    ecrans = contextes(actions, r'\\((.*?)\\)')
    configs = contextes(actions, r'<(.*?)>')
    chaines = contextes(actions, r'\\$(.*?)\\$')
    n = len(actions)
    modifications = sum(a.endswith('1') for a in actions)
    resultat = dict(nb_actions=n, nb_marqueurs_temporels=len(temps),
        dernier_repere_s=max(temps, default=0), sans_actions=int(n == 0),
        sans_marqueur_temporel=int(not temps), nb_actions_modification=modifications,
        part_actions_modification=modifications / n if n else 0,
        nb_types_actions_distincts=len(set(verbes)),
        nb_ecrans_contextes_distincts=len(set(ecrans)),
        nb_configurations_distinctes=len(set(configs)),
        nb_chaines_distinctes=len(set(chaines)), navigateur=ligne['navigateur'])
    for prefixe, valeurs in [('action_nombre', verbes), ('ecran_nombre', ecrans),
            ('configuration_nombre', configs), ('chaine_nombre', chaines)]:
        for valeur, nombre in Counter(valeurs).items():
            resultat[f'{prefixe}::{valeur}'] = nombre
    for valeur, nombre in Counter(verbes).items():
        resultat[f'action_proportion::{valeur}'] = nombre / n
    return resultat

X_reference = [caracteristiques_reference(r) for r in df.to_dict('records')]
assert all('util' not in r and 'session_id' not in r for r in X_reference)
print('Caractéristiques de référence construites :', len(X_reference))
''')
md('''
## 6. Représentations séquentielles et familles séparées

Le SVM historique utilise un TF-IDF commun à tous les comptages. Les nouvelles variantes
normalisent séparément quatre familles : contextes, actions complètes, séquences complètes,
et séquences de types/transitions. Les poids sont fixés avant la recherche : 1, 1, 1, 0.5.
Cela permet aux contextes d’exister même si une session contient beaucoup de séquences.
''')
code('''
def representations(actions, navigateur):
    verbes = [type_action(a) for a in actions]
    ecrans = contextes(actions, r'\\((.*?)\\)')
    configurations = contextes(actions, r'<(.*?)>')
    chaines = contextes(actions, r'\\$(.*?)\\$')
    contexte = {'navigateur': navigateur}
    for prefixe, valeurs in [('action_nombre', verbes), ('ecran_nombre', ecrans),
            ('configuration_nombre', configurations), ('chaine_nombre', chaines)]:
        for valeur, nombre in Counter(valeurs).items():
            contexte[f'{prefixe}::{valeur}'] = nombre
    complet, sequences, parcours = {}, {}, {}
    ajouter_comptages(complet, 'complete::', actions)
    for taille in (2, 3):
        ajouter_comptages(sequences, f'complete{taille}::', ngrammes(actions, taille))
        ajouter_comptages(parcours, f'verbe{taille}::', ngrammes(verbes, taille))
    ajouter_comptages(parcours, 'transition::', ngrammes(ecrans, 2))
    return {'historique': {**contexte, **complet, **sequences, **parcours},
            'contextes': contexte, 'actions': complet, 'sequences': sequences, 'parcours': parcours}

REPRESENTATIONS = [representations(a, n) for a, n in zip(df.actions, navigateurs)]
FAMILLES = ['contextes', 'actions', 'sequences', 'parcours']
POIDS_FAMILLES = [1., 1., 1., .5]
print('Exemples de motifs :')
display(pd.Series(REPRESENTATIONS[0]['sequences']).head(5))
''')
md('''
## 7. Extraits courts : uniquement pour l’apprentissage

Pour les sessions de plus de 400 actions, on prépare le début et le milieu, chacun de
200 actions. La fonction d’apprentissage n’accède qu’aux extraits des indices d’entraînement
du pli. Le vocabulaire et l’IDF sont appris sur les sessions originales de ce pli.
Chaque original reçoit un poids de 1 et chacun de ses deux extraits un poids de 0.25.
Les extraits conservent le label de leur session source ; ils ne deviennent jamais des exemples
de validation. C’est une hypothèse à tester : un extrait peut être moins distinctif.
''')
code('''
# Calcul déterministe par session, sans apprentissage ni utilisation des labels.
EXTRAITS = {}
for i, actions in enumerate(df.actions):
    if len(actions) > 400:
        debut_milieu = (len(actions) - 200) // 2
        EXTRAITS[i] = [representations(actions[:200], navigateurs[i]),
            representations(actions[debut_milieu:debut_milieu + 200], navigateurs[i])]
print('Sessions pouvant fournir deux extraits :', len(EXTRAITS))
''')
md('''
## 8. Encodage appris dans chaque pli

Le filtre conserve les motifs vus dans au moins deux sessions d’apprentissage.
La variante binaire remplace les comptes par des présences avant TF-IDF.
Les dictionnaires des sessions de validation ne participent ni au vocabulaire ni à l’IDF.
''')
code('''
def encoder_blocs(train, evaluation, mode, augmenter=False):
    familles = ['historique'] if mode == 'historique' else FAMILLES
    poids = [1.] if mode == 'historique' else POIDS_FAMILLES
    binariser = mode == 'binaire'
    parents, copies = [], []
    if augmenter:
        for i in train:
            for extrait in EXTRAITS.get(i, []):
                parents.append(i)
                copies.append(extrait)
    blocs_train, blocs_eval, blocs_extra = [], [], []
    for famille, poids_bloc in zip(familles, poids):
        vecteur = DictVectorizer(dtype=np.float32)
        a = vecteur.fit_transform([REPRESENTATIONS[i][famille] for i in train])
        b = vecteur.transform([REPRESENTATIONS[i][famille] for i in evaluation])
        garder = np.asarray((a != 0).sum(axis=0)).ravel() >= 2
        a, b = a[:, garder], b[:, garder]
        if binariser:
            a.data[:] = 1; b.data[:] = 1
        tfidf = TfidfTransformer(sublinear_tf=not binariser)
        blocs_train.append(poids_bloc * tfidf.fit_transform(a))
        blocs_eval.append(poids_bloc * tfidf.transform(b))
        if copies:
            c = vecteur.transform([r[famille] for r in copies])[:, garder]
            if binariser:
                c.data[:] = 1
            blocs_extra.append(poids_bloc * tfidf.transform(c))
    a = normalize(sparse.hstack(blocs_train, format='csr'))
    b = normalize(sparse.hstack(blocs_eval, format='csr'))
    poids_exemples = np.ones(len(train))
    etiquettes = y[train]
    if copies:
        c = normalize(sparse.hstack(blocs_extra, format='csr'))
        a = sparse.vstack([a, c], format='csr')
        etiquettes = np.concatenate([etiquettes, y[parents]])
        poids_exemples = np.concatenate([poids_exemples, np.full(len(copies), .25)])
        assert set(parents).issubset(set(train))
        assert set(parents).isdisjoint(evaluation)
    return a, b, etiquettes, poids_exemples
''')
md('## 9. Modèle de référence : spécialistes Extra Trees')
code('''
def predire_arbres(train, evaluation):
    resultat = np.zeros((len(evaluation), len(classes)), dtype=np.float32)
    for navigateur in np.unique(navigateurs[evaluation]):
        tr = train[navigateurs[train] == navigateur]
        positions = np.flatnonzero(navigateurs[evaluation] == navigateur)
        ev = evaluation[positions]
        # Repli global si le navigateur n’existe pas dans l’apprentissage.
        specialise = len(tr) > 0
        if not specialise:
            tr = train
        if len(np.unique(y[tr])) == 1:
            resultat[positions, np.searchsorted(classes, y[tr[0]])] = 1
            continue
        def lignes(indices):
            return [{k: v for k, v in X_reference[i].items()
                     if not (specialise and k == 'navigateur')} for i in indices]
        vecteur = DictVectorizer(dtype=np.float32)
        a = vecteur.fit_transform(lignes(tr)); b = vecteur.transform(lignes(ev))
        modele = ExtraTreesClassifier(n_estimators=200, max_features=.5,
            class_weight='balanced', n_jobs=2, random_state=RANDOM_STATE)
        with threadpool_limits(limits=2):
            modele.fit(a, y[tr])
            proba = modele.predict_proba(b)
        resultat[np.ix_(positions, np.searchsorted(classes, modele.classes_))] = proba
    return resultat
''')
md('''
## 10. SVM linéaires et SVM non linéaires

Les noyaux RBF sont calculés à partir des vecteurs normalisés :
`exp(-gamma * distance_carrée)`. Le noyau pré-calculé réduit le coût des problèmes
multiclasses. Les marges sont transformées en scores par `softmax(3 × marge)` pour
les combinaisons : **ce ne sont pas des probabilités calibrées**.

Pour les nouvelles variantes, les classes incompatibles avec le navigateur sont masquées
à partir des seules associations observées dans l’apprentissage. Un navigateur inconnu
ne déclenche aucun masque. Le SVM historique conserve son comportement initial.
''')
code('''
def filtrer_navigateur(scores, train, evaluation):
    resultat = scores.copy()
    for navigateur in np.unique(navigateurs[evaluation]):
        autorises = np.unique(y[train[navigateurs[train] == navigateur]])
        if len(autorises):
            positions = np.flatnonzero(navigateurs[evaluation] == navigateur)
            interdits = np.flatnonzero(~np.isin(classes, autorises))
            resultat[np.ix_(positions, interdits)] = 0
    return resultat / np.maximum(resultat.sum(axis=1, keepdims=True), 1e-30)

def predire_svm(config, train, evaluation):
    a, b, labels, poids = encoder_blocs(train, evaluation, config['mode'],
                                        config.get('augmentation', False))
    if config.get('noyau') == 'rbf':
        gamma = config['gamma']
        # Formule générale, y compris pour un éventuel vecteur nul.
        normes_a = np.asarray(a.multiply(a).sum(axis=1)).ravel()
        normes_b = np.asarray(b.multiply(b).sum(axis=1)).ravel()
        distance_train = np.maximum(normes_a[:, None] + normes_a[None, :] - 2*(a @ a.T).toarray(), 0)
        distance_eval = np.maximum(normes_b[:, None] + normes_a[None, :] - 2*(b @ a.T).toarray(), 0)
        a, b = np.exp(-gamma*distance_train), np.exp(-gamma*distance_eval)
        modele = SVC(C=config['C'], kernel='precomputed', decision_function_shape='ovr')
    else:
        modele = LinearSVC(C=config['C'], class_weight=config.get('class_weight'),
            dual=True, max_iter=30000, random_state=RANDOM_STATE)
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        modele.fit(a, labels, sample_weight=poids)
        marges = modele.decision_function(b)
    resultat = np.zeros((len(evaluation), len(classes)), dtype=np.float32)
    resultat[:, np.searchsorted(classes, modele.classes_)] = softmax(3*marges, axis=1)
    if config['mode'] != 'historique':
        resultat = filtrer_navigateur(resultat, train, evaluation)
    return resultat
''')
md('## 11. Configurations fixées avant consultation de la validation')
code('''
CONFIGURATIONS = {
    'Arbres reference': {'type': 'arbres'},
    'SVM historique': {'type': 'svm', 'mode': 'historique', 'C': 20.},
    'SVM blocs log C5': {'type': 'svm', 'mode': 'log', 'C': 5.},
    'SVM blocs log C20': {'type': 'svm', 'mode': 'log', 'C': 20.},
    'SVM binaire C5': {'type': 'svm', 'mode': 'binaire', 'C': 5.},
    'SVM binaire C20': {'type': 'svm', 'mode': 'binaire', 'C': 20.},
    'SVM binaire equilibre': {'type': 'svm', 'mode': 'binaire', 'C': 20., 'class_weight': 'balanced'},
    'SVM extraits courts': {'type': 'svm', 'mode': 'binaire', 'C': 20., 'augmentation': True},
    'SVM RBF gamma1': {'type': 'svm', 'mode': 'binaire', 'C': 20., 'noyau': 'rbf', 'gamma': 1.},
    'SVM RBF gamma5': {'type': 'svm', 'mode': 'binaire', 'C': 20., 'noyau': 'rbf', 'gamma': 5.},
}
display(pd.DataFrame(CONFIGURATIONS).T.fillna('—'))

def predire_config(nom, train, evaluation):
    config = CONFIGURATIONS[nom]
    if config['type'] == 'arbres':
        return predire_arbres(train, evaluation)
    return predire_svm(config, train, evaluation)

def mesurer(y_reel, prediction):
    return {'f1_macro': f1_score(y_reel, prediction, average='macro', zero_division=0),
            'f1_pondere': f1_score(y_reel, prediction, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_reel, prediction)}
''')
md('''
## 12. Validation croisée des modèles individuels

Chaque ligne reçoit une prédiction d’un modèle qui n’a pas appris sur elle. Les prédictions
hors pli sont sauvegardées après chaque modèle. Le score de sélection est la moyenne
des F1 macro des plis, distincte du F1 calculé en regroupant toutes les prédictions hors pli.
''')
code('''
plis = list(StratifiedKFold(3, shuffle=True, random_state=42).split(indices_train, y[indices_train]))
predictions_oof, lignes_cv, details_plis = {}, [], []
for nom in CONFIGURATIONS:
    debut_modele = perf_counter()
    oof = np.zeros((len(indices_train), len(classes)), dtype=np.float32)
    scores_plis = []
    for numero, (a, b) in enumerate(plis, 1):
        tr, ev = indices_train[a], indices_train[b]
        assert set(tr).isdisjoint(ev) and set(y[tr]) == set(classes)
        oof[b] = predire_config(nom, tr, ev)
        score = mesurer(y[ev], classes[oof[b].argmax(axis=1)])
        scores_plis.append(score['f1_macro'])
        details_plis.append({'modele': nom, 'pli': numero, **score})
        print(f'{nom} | pli {numero}/3 | F1 macro = {score["f1_macro"]:.4f}', flush=True)
    predictions_oof[nom] = oof
    lignes_cv.append({'modele': nom, 'f1_macro_cv': np.mean(scores_plis),
        'ecart_type_cv': np.std(scores_plis), 'duree_s': perf_counter()-debut_modele})
    pd.DataFrame(lignes_cv).to_csv(OUT/'modeles_cv.csv', index=False)
    pd.DataFrame(details_plis).to_csv(OUT/'details_plis.csv', index=False)
    np.savez_compressed(OUT/'predictions_oof.npz', **predictions_oof)
classement_modeles = pd.DataFrame(lignes_cv).sort_values('f1_macro_cv', ascending=False)
display(classement_modeles.round(4))
''')
md('''
## 13. Combinaisons et choix par CV uniquement

La combinaison historique (75 % SVM historique, 25 % arbres) est conservée comme référence.
Chaque nouveau SVM est combiné aux arbres avec des poids fixés à 0.50, 0.75 et 0.90.
Une combinaison égale des deux meilleurs modèles individuels hors arbres est aussi testée.
Cette recherche sur les mêmes plis peut optimiser le score de CV : le contrôle de sensibilité
et la validation réservée permettent de repérer un gain fragile, mais ne suppriment pas tout biais.
''')
code('''
RECETTES = {nom: {nom: 1.} for nom in CONFIGURATIONS}
RECETTES['Ensemble historique'] = {'Arbres reference': .25, 'SVM historique': .75}
for nom in CONFIGURATIONS:
    if nom not in ['Arbres reference', 'SVM historique']:
        for poids in (.5, .75, .9):
            RECETTES[f'{nom} + arbres ({poids:.2f})'] = {nom: poids, 'Arbres reference': 1-poids}
meilleurs_svm = classement_modeles.query('modele != "Arbres reference"').modele.iloc[:2].tolist()
RECETTES['Deux meilleurs SVM'] = {nom: .5 for nom in meilleurs_svm}

def combiner(recette, predictions):
    return sum(poids*predictions[nom] for nom, poids in recette.items())

lignes_ensembles = []
for nom, recette in RECETTES.items():
    prediction = classes[combiner(recette, predictions_oof).argmax(axis=1)]
    f1_plis = [mesurer(y[indices_train[b]], prediction[b])['f1_macro'] for _, b in plis]
    lignes_ensembles.append({'modele': nom, 'f1_macro_cv': np.mean(f1_plis),
        'ecart_type_cv': np.std(f1_plis), 'f1_macro_oof': mesurer(y[indices_train], prediction)['f1_macro']})
classement = pd.DataFrame(lignes_ensembles).sort_values('f1_macro_cv', ascending=False).reset_index(drop=True)
NOM_RETENU = classement.iloc[0].modele
RECETTE_RETENUE = RECETTES[NOM_RETENU].copy()
classement.to_csv(OUT/'classement_cv.csv', index=False)
(OUT/'choix_avant_validation.json').write_text(json.dumps(
    {'nom': NOM_RETENU, 'recette': RECETTE_RETENUE}, indent=2), encoding='utf-8')
display(classement.head(12).round(4))
print('Choix fixé avant validation :', NOM_RETENU, RECETTE_RETENUE)
''')
md('''
## 14. Sensibilité à un second découpage

Les paramètres et poids restent fixés. Seuls les composants nécessaires à la référence
historique et au choix retenu sont réentraînés. La graine 137 utilise les mêmes sessions
d’apprentissage : cette mesure n’est pas un test indépendant.
''')
code('''
RECETTES_FINALES = {'Ensemble historique': RECETTES['Ensemble historique'], NOM_RETENU: RECETTE_RETENUE}
composants_finaux = sorted({nom for recette in RECETTES_FINALES.values() for nom in recette})
stabilite = []
plis_bis = StratifiedKFold(3, shuffle=True, random_state=137)
for numero, (a, b) in enumerate(plis_bis.split(indices_train, y[indices_train]), 1):
    tr, ev = indices_train[a], indices_train[b]
    cache = {nom: predire_config(nom, tr, ev) for nom in composants_finaux}
    for nom, recette in RECETTES_FINALES.items():
        prediction = classes[combiner(recette, cache).argmax(axis=1)]
        score = mesurer(y[ev], prediction)
        stabilite.append({'modele': nom, 'pli': numero, **score})
        print(f'Stabilité | {nom} | pli {numero} | {score["f1_macro"]:.4f}', flush=True)
stabilite = pd.DataFrame(stabilite)
stabilite.to_csv(OUT/'stabilite.csv', index=False)
display(stabilite.groupby('modele').f1_macro.agg(['mean', 'std']).round(4))
''')
md('## 15. Évaluation sur les 20 % réservés')
code('''
# Le modèle retenu n’est pas modifié après cette cellule.
cache_validation = {nom: predire_config(nom, indices_train, indices_validation)
                    for nom in composants_finaux}
predictions_validation, resultats_validation = {}, []
for nom, recette in RECETTES_FINALES.items():
    prediction = classes[combiner(recette, cache_validation).argmax(axis=1)]
    predictions_validation[nom] = prediction
    resultats_validation.append({'modele': nom, **mesurer(y[indices_validation], prediction)})
resultats_validation = pd.DataFrame(resultats_validation)
resultats_validation.to_csv(OUT/'validation.csv', index=False)
display(resultats_validation.round(4))
''')
md('## 16. Corrections, régressions et erreurs restantes')
code('''
diagnostic = df.iloc[indices_validation][['session_id', 'util', 'navigateur', 'nb_actions']].copy()
diagnostic['reference'] = predictions_validation['Ensemble historique']
diagnostic['prediction'] = predictions_validation[NOM_RETENU]
diagnostic['erreur_reference'] = diagnostic.reference != diagnostic.util
diagnostic['erreur'] = diagnostic.prediction != diagnostic.util
diagnostic['correction'] = diagnostic.erreur_reference & ~diagnostic.erreur
diagnostic['regression'] = ~diagnostic.erreur_reference & diagnostic.erreur
effectifs_train = Counter(y[indices_train])
diagnostic['sessions_apprentissage'] = diagnostic.util.map(effectifs_train)
diagnostic.to_csv(OUT/'diagnostic_validation.csv', index=False)
display(pd.DataFrame({'mesure': ['erreurs référence', 'erreurs retenu', 'corrections', 'régressions'],
    'nombre': [diagnostic.erreur_reference.sum(), diagnostic.erreur.sum(),
               diagnostic.correction.sum(), diagnostic.regression.sum()]}))
display(diagnostic.loc[diagnostic.erreur].sort_values(['sessions_apprentissage', 'nb_actions']).head(20))
''')
md('''
## 17. Résultats selon la longueur et par utilisateur

Les taux d’erreur par longueur sont plus faciles à comparer que les nombres bruts.
Les F1 par utilisateur restent instables lorsque la validation ne contient qu’une ou deux sessions.
''')
code('''
diagnostic['longueur'] = pd.cut(diagnostic.nb_actions, [-1, 20, 100, 500, np.inf])
par_longueur = diagnostic.groupby('longueur', observed=True).agg(
    sessions=('erreur', 'size'), erreurs_reference=('erreur_reference', 'sum'),
    erreurs=('erreur', 'sum'), corrections=('correction', 'sum'), regressions=('regression', 'sum'))
par_longueur['taux_erreur_reference'] = par_longueur.erreurs_reference / par_longueur.sessions
par_longueur['taux_erreur'] = par_longueur.erreurs / par_longueur.sessions
par_longueur.to_csv(OUT/'erreurs_par_longueur.csv')
display(par_longueur.round(4))

rapport = pd.DataFrame(classification_report(y[indices_validation],
    predictions_validation[NOM_RETENU], output_dict=True, zero_division=0)).T.loc[classes]
rapport['sessions_apprentissage'] = [effectifs_train[u] for u in rapport.index]
rapport.to_csv(OUT/'rapport_utilisateurs.csv')
display(rapport.sort_values('f1-score').head(15).round(3))
''')
md('## 18. Contrôles, provenance et conclusion calculée')
code('''
assert set(indices_train).isdisjoint(indices_validation)
assert len(diagnostic) == len(indices_validation)
assert set(predictions_validation[NOM_RETENU]).issubset(set(classes))
assert np.isfinite(resultats_validation[['f1_macro', 'f1_pondere', 'accuracy']]).all().all()
assert NOM_RETENU == classement.iloc[0].modele
for valeurs in predictions_oof.values():
    assert valeurs.shape == (len(indices_train), len(classes))
    assert np.isfinite(valeurs).all()
    assert np.allclose(valeurs.sum(axis=1), 1, atol=1e-5)

score_reference = resultats_validation.set_index('modele').loc['Ensemble historique', 'f1_macro']
score_retenu = resultats_validation.set_index('modele').loc[NOM_RETENU, 'f1_macro']
resume = {'modele_retenu': NOM_RETENU, 'recette': RECETTE_RETENUE,
    'configurations': CONFIGURATIONS, 'f1_macro_validation': float(score_retenu),
    'gain_points': float(100*(score_retenu-score_reference)), 'objectif_97_atteint': bool(score_retenu >= .97),
    'python': platform.python_version(), 'sklearn': sklearn.__version__,
    'sha256_train': hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest(),
    'duree_s': perf_counter()-DEBUT,
    'limite': 'Validation de développement déjà examinée; seconde graine non indépendante; moyenne Kaggle à confirmer.'}
(OUT/'resume.json').write_text(json.dumps(resume, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'F1 macro validation : {100*score_retenu:.2f} %')
print(f'Gain par rapport au premier ensemble : {100*(score_retenu-score_reference):+.2f} points')
print('Objectif de 97 % atteint :', score_retenu >= .97)
print('Tous les contrôles sont validés.')
''')
md('''
## 19. Lire les résultats avec prudence

La normalisation, la binarisation, les extraits et les noyaux sont des hypothèses testées :
le classement indique ce qui aide effectivement sur ces plis. Le nouveau choix conserve ses
poids après observation de la validation, même si celle-ci déçoit. Un gain de CV seul ne prouve
pas une amélioration sur de futures sessions. Une validation imbriquée et la confirmation de
la métrique Kaggle restent nécessaires avant d’annoncer une performance finale.

Les tables ci-dessus sont les sorties de l’exécution des cellules. Les résultats détaillés sont
également conservés dans `datamanip/resultats_006_notebook`. Aucun fichier de soumission n’est créé.
''')

for i, cell in enumerate(cells):
    cell['id'] = f'cellule-006-{i:02d}'
notebook = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
path = ROOT / 'datamanip/006_sequences.ipynb'
backup = ROOT / 'datamanip/resultats_006/006_premiere_version.ipynb'
if path.exists() and not backup.exists():
    shutil.copy2(path, backup)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Notebook autonome créé :', len(cells), 'cellules')
