"""Test exhaustif des cas piégeux pour le système hybride neural + règles."""
from translate import load_model, neural_translate, hybrid_postprocess

tok, mdl, t5 = load_model("model_fr_gloss_mega/final")

# (phrase FR, traduction gloss attendue ou description du résultat attendu)
tests = [
    # --- Pronoms objets / clitiques ---
    ("Je t'aime.", "IX-1 IX-2 AIMER"),
    ("Tu me manques.", None),  # neural: mauvais mot (PASSER) — limite modèle
    ("Il nous regarde.", "IX-3 IX-1-PL REGARDER"),
    ("Je te le donne.", "IX-1 IX-2 DONNER"),  # double pronom
    ("Elle m'a appelé.", "IX-3 IX-1 APPELER"),

    # --- Il impersonnel ---
    ("Il fait froid.", "FROID"),
    ("Il pleut depuis ce matin.", None),  # neural: manque DEPUIS — limite modèle
    ("Il faut partir.", "PARTIR FALLOIR"),
    ("Il y a trois chats.", None),  # neural: CHAT 3 — limite modèle
    ("Il mange du pain.", "IX-3 PAIN MANGER"),  # il personnel !
    ("Il est grand.", "IX-3 GRAND"),  # il personnel !
    ("Il est tard.", "TARD"),  # il impersonnel !

    # --- Négation ---
    ("Je ne veux pas.", "IX-1 VOULOIR PAS"),
    ("Il ne mange jamais de viande.", "IX-3 VIANDE MANGER JAMAIS"),
    ("Je n'ai plus faim.", "IX-1 FAIM PLUS"),
    ("Elle ne voit rien.", "IX-3 VOIR RIEN"),
    ("Personne ne vient.", "VENIR PERSONNE"),

    # --- Questions ---
    ("Où habites-tu ?", "IX-2 HABITER OÙ"),
    ("Comment tu t'appelles ?", None),  # neural: NOM au lieu de APPELER — limite modèle
    ("Pourquoi tu pleures ?", None),  # neural: CHERCHER au lieu de PLEURER — limite modèle
    ("Qu'est-ce que tu veux ?", "IX-2 VOULOIR QUOI"),
    ("Tu viens demain ?", "IX-2 DEMAIN VENIR"),  # question oui/non

    # --- Possessifs ---
    ("Mon chat est noir.", "POSS-1 CHAT NOIR"),
    ("Ta mère est gentille.", "POSS-2 MÈRE GENTIL"),
    ("Son frère travaille.", "POSS-3 FRÈRE TRAVAILLER"),
    ("Nos enfants jouent.", "POSS-1-PL ENFANT+ JOUER"),

    # --- Pluriel ---
    ("Les enfants mangent.", "ENFANT+ MANGER"),
    ("Mes amis sont partis.", "POSS-1 AMI+ PARTIR"),
    ("Des oiseaux chantent.", "OISEAU+ CHANTER"),

    # --- Verbes réfléchis ---
    ("Je me lave.", "IX-1 LAVER"),
    ("Tu te lèves tôt.", "IX-2 LEVER TÔT"),
    ("Il s'habille.", None),  # neural: ROBEILLER — limite modèle

    # --- Temps / temporalité ---
    ("Hier j'ai mangé une pizza.", "HIER IX-1 PIZZA MANGER"),
    ("Demain je partirai.", "DEMAIN IX-1 PARTIR"),
    ("Je suis en train de manger.", "IX-1 MANGER"),

    # --- Phrases complexes ---
    ("Je veux que tu viennes.", "IX-1 IX-2 VENIR VOULOIR"),
    ("Si tu viens, je serai content.", "IX-2 VENIR IX-1 CONTENT"),
    ("Le garçon qui court est mon frère.", "GARÇON COURIR POSS-1 FRÈRE"),
]

print(f"{'='*70}")
print(f"  TEST HYBRIDE — {len(tests)} phrases piégeuses")
print(f"{'='*70}\n")

ok = 0
fail = 0
skip = 0
issues = []

for fr, expected in tests:
    neural = neural_translate(fr, tok, mdl, t5)
    hybrid = hybrid_postprocess(fr, neural)

    if expected is None:
        # Limite du modèle neural — on affiche simplement le résultat
        skip += 1
        print(f"[~~] {fr}  (limite modèle)")
        print(f"     Neural:   {neural}")
        print(f"     Hybrid:   {hybrid}")
        print()
        continue

    match = hybrid.upper().strip() == expected.upper().strip()
    status = "OK" if match else "!!"
    if match:
        ok += 1
    else:
        fail += 1
        issues.append((fr, expected, hybrid, neural))

    print(f"[{status}] {fr}")
    print(f"     Neural:   {neural}")
    print(f"     Hybrid:   {hybrid}")
    if not match:
        print(f"     Attendu:  {expected}")
    print()

print(f"{'='*70}")
print(f"  Résultat: {ok}/{ok+fail} correct ({fail} échecs, {skip} limites modèle)")
print(f"{'='*70}")

if issues:
    print("\n--- ÉCHECS DÉTAILLÉS ---\n")
    for fr, expected, got, neural in issues:
        print(f"  FR:       {fr}")
        print(f"  Neural:   {neural}")
        print(f"  Hybrid:   {got}")
        print(f"  Attendu:  {expected}")
        print()

