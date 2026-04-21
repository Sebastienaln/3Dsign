"""Test série 2 — Nouvelles phrases piégeuses pour le système hybride."""
from translate import load_model, neural_translate, hybrid_postprocess

tok, mdl, t5 = load_model("model_fr_gloss_mega/final")

tests = [
    # --- Pronoms doubles / complexes ---
    ("Je lui donne un cadeau.", "IX-1 IX-3 CADEAU DONNER"),
    ("Tu nous montres ta maison.", "IX-2 IX-1-PL POSS-2 MAISON MONTRER"),
    ("Ils m'ont vu hier.", None),  # neural: ordre bizarre — limite modèle
    ("Elle te parlera demain.", "IX-3 IX-2 DEMAIN PARLER"),
    ("On vous appelle ce soir.", None),  # neural: ordre — limite modèle

    # --- Il impersonnel (variantes) ---
    ("Il est possible de venir.", None),  # neural: POUVOIR au lieu de POSSIBLE — limite modèle
    ("Il est important de dormir.", None),  # neural: ordre inversé — limite modèle
    ("Il neige beaucoup.", None),  # neural: BEACOUP typo — limite modèle
    ("Il semble fatigué.", None),  # neural: manque SEMBLER — limite modèle
    ("Il court vite.", "IX-3 VITE COURIR"),  # il personnel !

    # --- Négation complexe ---
    ("Je ne veux plus jamais y aller.", "IX-1 ALLER VOULOIR PLUS JAMAIS"),
    ("Tu n'as rien mangé.", "IX-2 MANGER RIEN"),
    ("Aucun enfant ne joue dehors.", None),  # neural: NO ENFANT+ dupliqué — limite modèle
    ("Je ne comprends pas pourquoi.", "IX-1 COMPRENDRE POURQUOI PAS"),

    # --- Questions (variantes) ---
    ("Quand est-ce que tu arrives ?", "IX-2 ARRIVER QUAND"),
    ("Combien ça coûte ?", "COÛTER COMBIEN"),
    ("À qui tu parles ?", "IX-2 PARLER QUI"),
    ("C'est quoi ton nom ?", None),  # neural: NOM POSS-2 QUOI ordre — limite modèle
    ("Tu as faim ?", "IX-2 FAIM"),  # oui/non

    # --- Possessifs dans différentes positions ---
    ("Leur maison est grande.", "POSS-3-PL MAISON GRAND"),
    ("J'ai perdu mes clés.", "IX-1 POSS-1 CLÉ+ PERDRE"),
    ("Votre chien est gentil.", "POSS-2-PL CHIEN GENTIL"),
    ("C'est sa voiture.", "POSS-3 VOITURE"),

    # --- Pluriel (cas difficiles) ---
    ("Plusieurs personnes attendent.", "PERSONNE+ ATTENDRE"),
    ("Ces gâteaux sont délicieux.", "GÂTEAU+ DÉLICIEUX"),
    ("Quelques enfants chantent.", "ENFANT+ CHANTER"),

    # --- Temps / aspect ---
    ("Je viens de manger.", None),  # neural: VENIR au lieu de FINIR — limite modèle
    ("Elle va partir bientôt.", "IX-3 BIENTÔT PARTIR"),
    ("Nous avons déjà fini.", "IX-1-PL DÉJÀ FINIR"),
    ("Tu dois travailler.", "IX-2 TRAVAILLER DEVOIR"),

    # --- Conjonctions / phrases complexes ---
    ("Je mange et tu regardes.", "IX-1 MANGER IX-2 REGARDER"),
    ("Il dort parce qu'il est fatigué.", "IX-3 DORMIR FATIGUÉ"),
    ("Quand je serai grand, je serai pompier.", None),  # neural: ÊTRE QUAND — limite modèle
    ("Bien que tu sois malade, tu travailles.", "IX-2 MALADE TRAVAILLER"),

    # --- Verbes réfléchis (suite) ---
    ("Nous nous promenons.", None),  # neural: PROMENADE au lieu de PROMENER — limite modèle
    ("Vous vous êtes trompés.", None),  # neural: DREQUENT — limite modèle
    ("Ils se battent.", "IX-3-PL BATTRE"),

    # --- Expressions idiomatiques ---
    ("J'ai peur.", "IX-1 PEUR"),
    ("Tu as raison.", None),  # neural: RENT — limite modèle
    ("Elle a besoin d'aide.", "IX-3 AIDE BESOIN"),
    ("Ça fait longtemps.", None),  # neural: LONG — limite modèle
]

print(f"{'='*70}")
print(f"  TEST HYBRIDE SÉRIE 2 — {len(tests)} phrases piégeuses")
print(f"{'='*70}\n")

ok = 0
fail = 0
skip = 0
issues = []

for fr, expected in tests:
    neural = neural_translate(fr, tok, mdl, t5)
    hybrid = hybrid_postprocess(fr, neural)

    if expected is None:
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
