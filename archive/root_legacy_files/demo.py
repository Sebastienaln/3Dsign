"""
Demo rapide : Traduction français → gloss LSF (mode hybride).

Usage :
    python demo.py
    python demo.py --model model_fr_gloss_20k/final
"""

import argparse
import os
import sys

# S'assurer que le dossier du script est le répertoire de travail
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from translate import load_model, neural_translate, hybrid_postprocess

DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "model_fr_gloss_20k", "final")


def main():
    parser = argparse.ArgumentParser(description="Demo FR → Gloss LSF")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Chemin vers le modèle (défaut: model_fr_gloss_20k/final)")
    args = parser.parse_args()

    tokenizer, model, is_t5 = load_model(args.model)

    print("=" * 50)
    print("  Traducteur Français → Gloss LSF")
    print("  Tapez une phrase, ou 'q' pour quitter.")
    print("=" * 50)
    print()

    while True:
        try:
            text = input("FR > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text or text.lower() in ("q", "quit", "exit"):
            break

        neural = neural_translate(text, tokenizer, model, is_t5)
        gloss = hybrid_postprocess(text, neural)
        print(f"=> {gloss}\n")


if __name__ == "__main__":
    main()
