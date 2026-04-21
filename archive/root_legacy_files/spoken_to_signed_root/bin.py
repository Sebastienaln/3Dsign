import argparse
import sys

from spoken_to_signed.text_to_gloss.rules_fr import text_to_gloss_fr


def text_to_gloss():
    parser = argparse.ArgumentParser(
        description="Traducteur français → gloss LSF (Langue des Signes Française)",
    )
    parser.add_argument("--text", type=str, required=True, help="Phrase en français à traduire")
    parser.add_argument("--debug", action="store_true", help="Afficher les détails de l'analyse spaCy")
    parser.add_argument("--details", action="store_true", help="Afficher le détail mot par mot")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif (saisie continue)")
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
                for token, gloss in zip(result["tokens"], result["glosses"]):
                    print(f"  {token:20s} → {gloss}")
            print()
        return

    if not args.text:
        parser.print_help()
        sys.exit(1)

    result = text_to_gloss_fr(args.text, debug=args.debug)
    print(result["gloss_string"])
    if args.details:
        print("\nDétail :")
        for token, gloss in zip(result["tokens"], result["glosses"]):
            print(f"  {token:20s} → {gloss}")


if __name__ == "__main__":
    text_to_gloss()
