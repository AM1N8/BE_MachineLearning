# Analyse forensique des erreurs

Source : prédictions hors-fold existantes, mêmes 4 folds × 3 répétitions.
Ces erreurs ont déjà été observées : elles servent aux hypothèses, pas de confirmation indépendante.

| Modèle | Erreurs / prédictions | Sessions toujours erronées | Sessions erronées au moins une fois |
| --- | ---: | ---: | ---: |
| champion | 310 / 9837 | 66 | 149 |
| sqrt_tfidf | 318 / 9837 | 73 | 141 |
| original | 333 / 9837 | 72 | 159 |

## Confusions principales du champion

| Vrai utilisateur | Prédit | Erreurs (répétitions incluses) |
| --- | --- | ---: |
| flj | skm | 15 |
| skm | slq | 9 |
| ald | fuy | 7 |
| azn | wht | 6 |
| esu | lgd | 6 |
| lfq | czi | 6 |
| azx | guh | 6 |
| zok | lra | 5 |
| yrb | zqs | 5 |
| mmm | byo | 5 |
| hll | guh | 4 |
| slq | skm | 4 |
| vyu | oec | 4 |
| udj | qkd | 4 |
| cuw | zyk | 3 |

## Traces exhaustives

`reports/results/extreme/all_errors_*.csv` : chaque erreur, longueur, entropie, répétitions et navigateur.
`per_class_*.csv`, `confusion_*.csv`, `confusion_pairs_*.csv`, `persistence_*.csv` : tous les utilisateurs et sessions.
0 groupes identiques après suppression des marqueurs temporels ; 0 groupes comportent plusieurs étiquettes.
Aucun extrait d'événements bruts n'est inclus dans ces rapports.

## Nouvelle reproduction : huit folds externes

Source : `reports/results/improvement/4e7e8c2ad9a6`.

{'predictions': 6558, 'errors': 201, 'always_wrong': 78, 'high_margin_errors': 2, 'true_in_top2_errors': 109}

### Groupes et facteurs associés


browser
```
                sum  count      mean
browser                             
Firefox          72   2932  0.024557
Google Chrome   111   2678  0.041449
Microsoft Edge   18    902  0.019956
Opera             0     46  0.000000
```


length_bin
```
                sum  count      mean
length_bin                          
(0.0, 20.0]       6      8  0.750000
(100.0, 500.0]  113   3126  0.036148
(20.0, 100.0]    65    876  0.074201
(500.0, inf]     17   2548  0.006672
```


margin_bin
```
             sum  count      mean
margin_bin                       
(-inf, 0.1]   97    187  0.518717
(0.1, 0.3]    66    325  0.203077
(0.3, 1.0]    36   2116  0.017013
(1.0, inf]     2   3930  0.000509
```


Moyennes conditionnelles (association, pas causalité) :
```
           length       oov      rare  unseen_transition   entropy  repetition       margin
wrong                                                                                      
False  706.849929  0.002608  0.006458           0.033392  4.309398    0.842883  7237.232930
True   234.084577  0.019366  0.023096           0.045569  3.453432    0.779787     0.179401
```


Principales confusions :
```
truth  prediction
flj    skm           11
ald    fuy            6
slq    skm            5
azn    wht            4
azx    guh            4
ngs    slq            4
zok    lra            4
lfq    czi            4
vyu    oec            4
xby    xgv            4
lgd    esu            3
skm    slq            3
hll    guh            3
iyf    azo            2
itt    akx            2
```


Taxonomie : faible marge et vrai utilisateur en seconde position → frontière locale ; événements/transitions inédits → couverture de représentation ; erreurs répétées → ambiguïté persistante à examiner. Ces catégories ne prouvent ni une mauvaise étiquette ni un décalage de distribution. Aucun label n'est corrigé. Les marges ne sont pas calibrées.
