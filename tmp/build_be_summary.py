from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

root = Path(__file__).resolve().parents[1]
out = root / "rapport" / "demarche_machine_learning.docx"
out.parent.mkdir(parents=True, exist_ok=True)
doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(.65)
sec.bottom_margin = Inches(.65)
sec.left_margin = Inches(.8)
sec.right_margin = Inches(.8)
normal = doc.styles["Normal"]
normal.font.name = "Aptos"
normal.font.size = Pt(10)
normal.font.color.rgb = RGBColor(40, 48, 58)
normal.paragraph_format.space_after = Pt(5)
for style_name, size, color in [
    ("Title", 23, "17365D"), ("Heading 1", 16, "17365D"),
    ("Heading 2", 12, "2878A0")
]:
    style = doc.styles[style_name]
    style.font.name = "Aptos Display"
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor.from_string(color)

title = doc.add_paragraph(style="Title")
title.add_run("Identifier les utilisateurs à partir des traces")
p = doc.add_paragraph()
p.add_run("Présentation de la démarche de machine learning").italic = True
doc.add_paragraph(
    "Notre travail consiste à reconnaître l’utilisateur d’une session à partir des actions "
    "effectuées dans le logiciel. Nous avons exploré les traces, construit des variables, "
    "comparé plusieurs modèles puis évalué des combinaisons. La soumission Kaggle a obtenu "
    "un score public de 0,971 (97,1 %)."
)

def heading(text):
    doc.add_heading(text, level=1)

def bullet(text):
    doc.add_paragraph(text, style="List Bullet")

def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Light Shading Accent 1"
    for cell, val in zip(t.rows[0].cells, headers):
        cell.text = val
    for row in rows:
        cells = t.add_row().cells
        for cell, val in zip(cells, row):
            cell.text = val
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    return t

heading("1. Données")
doc.add_paragraph(
    "Une ligne correspond à une session d’utilisation. Le jeu d’apprentissage contient "
    "3 279 sessions et 247 utilisateurs. Le jeu de test comprend 324 sessions dont "
    "l’utilisateur est masqué."
)
bullet("Une session contient le navigateur et une suite ordonnée de traces.")
bullet("Les traces décrivent les actions, les écrans, les configurations, les chaînes et des repères temporels tXX.")
bullet("Le suffixe 1 signale certaines actions réalisées en modification.")
bullet("Un utilisateur est la classe que le modèle doit prédire.")
doc.add_paragraph(
    "La métrique est le F1 macro : le F1 est calculé pour chaque utilisateur puis moyenné, "
    "avec le même poids pour chaque classe. Cela évite que les utilisateurs les plus fréquents "
    "dominent seuls le score."
)

heading("2. Modèles et représentations")
table(
    ["Approche", "Rôle dans notre démarche"],
    [
        ("Caractéristiques de session", "Compter les actions, contextes, modifications et repères temporels."),
        ("SVM linéaire (LinearSVC)", "Séparer les utilisateurs à partir des motifs de traces pondérés par TF-IDF."),
        ("Extra Trees", "Combiner de nombreux arbres entraînés sur les statistiques de session, spécialisés par navigateur."),
        ("Ensemble", "Mélanger les scores du SVM et d’Extra Trees, qui apportent des signaux différents."),
    ],
)
doc.add_paragraph(
    "Les notebooks 001 à 005 explorent les données, fabriquent les premières variables, "
    "comparent des classificateurs et étudient l’effet du navigateur. Le notebook 006 ajoute "
    "l’ordre des actions : séquences de deux ou trois actions et transitions d’écran. "
    "Le notebook 008 teste aussi les associations entre action, écran, configuration et chaîne, "
    "ainsi que des mesures de rythme et de répétition."
)
doc.add_heading("Comment les modèles prennent une décision", level=2)
doc.add_paragraph(
    "Exemple simplifié : une session contient souvent « Création d’un écran », "
    "« Saisie dans un champ » et l’accès au contexte MAINT. Le modèle transforme ces "
    "indices en nombres et estime quel utilisateur correspond le mieux à ce profil."
)
doc.add_heading("SVM linéaire", level=2)
doc.add_paragraph(
    "Le SVM représente chaque session par un vecteur de caractéristiques. TF-IDF donne plus "
    "d’importance aux motifs utiles pour distinguer les sessions et moins aux éléments présents "
    "partout. Le modèle apprend ensuite une frontière qui sépare les utilisateurs. Pour une "
    "nouvelle session, il choisit la classe située du bon côté de la frontière avec la meilleure "
    "marge."
)
table(
    ["Vue simplifiée des sessions", "Décision du SVM"],
    [
        ("Session A : écrans fréquents X, Y", "Profil plus proche de l’utilisateur ALI"),
        ("Session B : écran Z, séquence Saisie → Validation", "Profil plus proche de l’utilisateur SAM"),
        ("Nouvelle session : écran Z, Saisie → Validation", "La frontière place la session côté SAM"),
    ],
)
doc.add_paragraph(
    "Schéma : motifs des sessions → vecteurs TF-IDF → frontière entre utilisateurs → utilisateur prédit."
)
doc.add_heading("Extra Trees", level=2)
doc.add_paragraph(
    "Extra Trees construit beaucoup d’arbres de décision. Chaque arbre pose une suite de questions "
    "sur les caractéristiques, par exemple : « le navigateur est-il Firefox ? », « le contexte "
    "MAINT apparaît-il souvent ? » ou « la session dépasse-t-elle 100 actions ? ». Les feuilles "
    "des arbres votent pour un utilisateur ; les votes sont réunis en une estimation de probabilité. "
    "Notre recette entraîne ces arbres séparément selon le navigateur."
)
table(
    ["Arbre", "Exemple de chemin"],
    [
        ("Arbre 1", "Firefox ? oui → beaucoup de MAINT ? oui → vote ALI"),
        ("Arbre 2", "Session longue ? oui → séquence fréquente ? oui → vote SAM"),
        ("Arbre 3", "Firefox ? oui → configuration observée ? oui → vote ALI"),
        ("Forêt", "Les votes des arbres sont agrégés pour choisir l’utilisateur."),
    ],
)
doc.add_heading("Mélange SVM et Extra Trees", level=2)
doc.add_paragraph(
    "Les modèles produisent des scores pour chaque utilisateur. Nous calculons une moyenne "
    "pondérée de leurs scores : le SVM contribue à hauteur de 90 % et Extra Trees de 10 % dans "
    "la première recette soumise. Pour la recette optimisée du notebook 008, les poids sont "
    "67,5 %, 22,5 % et 10 % pour les deux SVM et Extra Trees."
)
table(
    ["Exemple de scores pour une session", "Utilisateur ALI", "Utilisateur SAM"],
    [
        ("SVM", "0,70", "0,30"),
        ("Extra Trees", "0,20", "0,80"),
        ("Mélange 90 % / 10 %", "0,65", "0,35"),
        ("Prédiction", "ALI", ""),
    ],
)
doc.add_paragraph(
    "Les nombres de ces exemples sont illustratifs : ils servent à montrer le calcul, et ne "
    "correspondent pas aux scores d’une session particulière du jeu de données."
)

heading("3. Entraînement et validation")
doc.add_paragraph(
    "Nous comparons les approches sur des sessions dont l’étiquette est connue. Pendant la "
    "validation croisée à quatre plis du notebook 008, chaque pli sert à tour de rôle à "
    "l’évaluation, tandis que les autres servent à l’apprentissage. Les transformations TF-IDF "
    "sont ajustées uniquement sur la partie d’apprentissage de chaque pli."
)
bullet("Réglages testés pour LinearSVC : C = 3, 5, 10, 20 et 50.")
bullet("Représentations comparées : séquences TF-IDF et séquences enrichies par le contexte et le rythme.")
bullet("Mélanges testés : plusieurs poids pour le SVM et Extra Trees, puis combinaison de deux SVM et des arbres.")
table(
    ["Étape", "Résultat local"],
    [
        ("Comparaisons initiales (004)", "Extra Trees : F1 macro CV 0,8775 ; Random Forest : 0,8448 ; SVM TF-IDF : 0,8519."),
        ("Séquences (006)", "Meilleur ensemble : F1 macro de validation 0,9529."),
        ("Optimisation (008), graine 42", "F1 macro hors pli : 0,9689 ; moyenne des plis : 0,9664."),
        ("Optimisation (008), graine 137", "F1 macro hors pli : 0,9717 ; moyenne des plis : 0,9678."),
    ],
)
doc.add_paragraph(
    "La recette retenue par le notebook 008 donne 67,5 % au SVM sur séquences avec C=50, "
    "22,5 % au SVM enrichi contexte-rythme avec C=20, et 10 % à Extra Trees. Les deux graines "
    "réutilisent le même jeu d’apprentissage avec des répartitions différentes : elles donnent "
    "une indication de stabilité, pas deux tests indépendants."
)

heading("4. Test et soumission")
doc.add_paragraph(
    "Après le choix d’une recette, le modèle est réentraîné sur les 3 279 sessions étiquetées "
    "et prédit l’utilisateur des 324 sessions de test. Les vraies étiquettes du test restent "
    "cachées ; Kaggle compare les prédictions envoyées avec ses réponses."
)
doc.add_paragraph(
    "La soumission Kaggle a obtenu 0,971 de F1 public. Ce score est distinct du F1 de validation "
    "locale : il est calculé par Kaggle sur une partie cachée du test. La recette optimisée du "
    "notebook 008 a un meilleur résultat local, mais son score Kaggle ne peut être confirmé "
    "qu’après soumission de son CSV."
)
heading("À retenir pour la présentation")
doc.add_paragraph(
    "Nous sommes partis de comptages simples, puis nous avons exploité l’ordre des actions, "
    "le navigateur et les contextes. La validation croisée a guidé le choix d’un mélange de "
    "SVM et d’Extra Trees. Notre soumission Kaggle a atteint 97,1 % de F1 public."
)

doc.core_properties.title = "Démarche de classification des utilisateurs à partir des traces"
doc.core_properties.subject = "Données, modèles, entraînement et test"
doc.save(out)
print(out)
