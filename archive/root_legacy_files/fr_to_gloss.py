#!/usr/bin/env python3
"""
Script autonome pour traduire du français vers le gloss LSF.

Usage :
    python fr_to_gloss.py "Je mange une pomme."
    python fr_to_gloss.py "Les enfants jouent dans le jardin." --debug
    python fr_to_gloss.py --interactive
"""

import argparse
import sys
import os

# Ajouter le répertoire racine au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spoken_to_signed.text_to_gloss.rules_fr import text_to_gloss_fr


def main():
    parser = argparse.ArgumentParser(
        description="Traducteur français → gloss LSF (Langue des Signes Française)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python fr_to_gloss.py "Je mange une pomme."
  python fr_to_gloss.py "Où est la gare ?" --debug
  python fr_to_gloss.py --interactive

Règles appliquées :
  - Suppression des articles, prépositions, auxiliaires
  - Pronoms → signes d'indexation (IX-1, IX-2, IX-3…)
  - Possessifs → POSS-1, POSS-2, POSS-3…
  - Lemmatisation des verbes et noms
  - Marquage du pluriel (NOM+)
  - Réordonnancement : Temps → Lieu → Sujet → Objet → Verbe → Négation
  - Marqueur interrogatif en fin de phrase
  - Mise en majuscules (convention gloss)
        """,
    )
    parser.add_argument("text", nargs="?", type=str, help="Phrase en français à traduire")
    parser.add_argument("--debug", action="store_true", help="Afficher les détails de l'analyse spaCy")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif (saisie continue)")
    parser.add_argument("--details", action="store_true", help="Afficher le détail mot par mot")
    args = parser.parse_args()

    if args.interactive:
        print("=== Traducteur Français → Gloss LSF ===")
        print("Tapez une phrase en français (ou 'q' pour quitter)")
        print()
        while True:
            try:
                text = input("FR > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in {"q", "quit", "exit", "quitter"}:
                break
            if not text:
                continue
            result = text_to_gloss_fr(text, debug=args.debug)
            print(f"GLOSS > {result['gloss_string']}")
            if args.details:
                print("Détail :")
                for token, gloss in zip(result["tokens"], result["glosses"]):
                    print(f"  {token:20s} → {gloss}")
            print()
    elif args.text:
        result = text_to_gloss_fr(args.text, debug=args.debug)
        print(result["gloss_string"])
        if args.details:
            print("\nDétail :")
            for token, gloss in zip(result["tokens"], result["glosses"]):
                print(f"  {token:20s} → {gloss}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
