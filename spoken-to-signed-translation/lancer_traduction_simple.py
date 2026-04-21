#!/usr/bin/env python3
"""Lanceur simple pour traduire rapidement FR -> gloss.

Usage:
  python lancer_traduction_simple.py
  python lancer_traduction_simple.py "Le gouvernement doit agir vite."
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lance rapidement la traduction avec le dernier modele.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Phrase a traduire. Si absent, ouvre le mode interactif.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    translate_script = project_dir / "translate.py"
    model_dir = project_dir / "model_fr_gloss_mega_politique" / "final"

    if not translate_script.exists():
        print(f"Erreur: script introuvable: {translate_script}")
        return 1
    if not model_dir.exists():
        print(f"Erreur: modele introuvable: {model_dir}")
        return 1

    cmd = [
        sys.executable,
        str(translate_script),
        "--model",
        str(model_dir),
        "--mode",
        "hybrid",
    ]

    if args.text:
        cmd.extend(["--text", args.text])
    else:
        cmd.append("--interactive")

    return subprocess.call(cmd, cwd=str(project_dir))


if __name__ == "__main__":
    raise SystemExit(main())
