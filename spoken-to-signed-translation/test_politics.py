#!/usr/bin/env python3
"""
Script de test pour évaluer la traduction de DÉBATS POLITIQUES.

Teste le modèle entraîné sur des phrases politiques authentiques et les compare 
avec le modèle de base pour voir l'amélioration.

Usage:
    python test_political_debates.py --model model_politique/final
    python test_political_debates.py --model model_politique/final --debug
    python test_political_debates.py --compare model_politique/final model_fr_gloss/final
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from spoken_to_signed.text_to_gloss.rules_fr import text_to_gloss_fr

# Phrases de test pour les débats politiques
TEST_PHRASES_POLITIQUES = [
    # Promesses électorales
    "Je promets de créer un million d'emplois.",
    "Notre gouvernement réduira le chômage.",
    "Votez pour notre projet de réforme.",
    
    # Critiques et opposition
    "Vous avez échoué sur l'économie.",
    "Le gouvernement n'a rien fait pour les travailleurs.",
    "C'est un désastre pour la jeunesse.",
    
    # Réformes et lois
    "Nous augmentons le SMIC de 10 pour cent.",
    "La loi sur le climat sera votée demain.",
    "Cette réforme affecte trois millions de Français.",
    
    # Éducation et emploi
    "Il faut doubler le budget de l'école républicaine.",
    "L'accès à l'université doit être démocratique.",
    "La formation professionnelle est sous-financée.",
    
    # Santé et social
    "Le système de santé français est excellent.",
    "Nous devons protéger le service public de santé.",
    "L'accès aux soins en zones rurales est problématique.",
    
    # Environnement
    "La France s'engage pour la neutralité carbone en 2050.",
    "Nous interdisons les voitures thermiques d'ici 2035.",
    "Le changement climatique est une menace existentielle.",
    
    # Immigration et intégration
    "Notre politique d'immigration doit être claire.",
    "L'intégration des migrants passe par l'emploi.",
    "Nous accueillons les réfugiés selon nos capacités.",
    
    # Sécurité et justice
    "Renforcer la police dans les quartiers sensibles.",
    "La justice doit être plus rapide et efficace.",
    "Nous investissons dans la prévention de la criminalité.",
    
    # Relations internationales
    "La France reste engagée dans l'UE.",
    "Nous condamnons cette violation des droits de l'homme.",
    "La diplomatie française joue un rôle majeur.",
    
    # Statistiques et chiffres
    "Selon un sondage, 68 pour cent des Français approuvent.",
    "Le PIB a augmenté de 1,5 pour cent.",
    "Trois millions de Français vivent sous le seuil de pauvreté.",
    
    # Positions politiques
    "La gauche propose une autre vision.",
    "Les Verts ont des propositions radicales.",
    "Le centre refuse les extrêmes.",
    
    # Appels à mobilisation
    "Levons-nous contre cette injustice.",
    "Défendons ensemble nos droits sociaux.",
    "C'est le moment de s'unir.",
]


def translate_with_model(text: str, model_path: str) -> str:
    """Traduit avec un modèle entraîné."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        print("Erreur : installer transformers et torch")
        return "ERROR"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        
        inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=256)
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        return translation
    except Exception as e:
        return f"ERROR: {str(e)}"


def compare_translations(text: str, model_path: str = None):
    """Compare les traductions du modèle politique et du modèle de base."""
    print(f"\n{'='*80}")
    print(f"FR: {text}")
    print(f"{'-'*80}")
    
    # Traduction avec règles
    rules_result = text_to_gloss_fr(text)
    rules_gloss = rules_result['gloss_string'].replace(" ||", "").strip()
    print(f"Règles:   {rules_gloss}")
    
    # Traduction avec modèle politique
    if model_path and os.path.isdir(model_path):
        model_gloss = translate_with_model(text, model_path)
        print(f"Modèle:   {model_gloss}")
    
    print()


def evaluate_dataset(dataset_path: str, model_path: str = None, samples: int = 20):
    """Évalue sur des phrases du dataset."""
    import csv
    
    print(f"\n=== Évaluation sur dataset: {dataset_path} ===\n")
    
    phrases = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('fr'):
                phrases.append((row['fr'], row.get('gloss', '')))
    
    # Choisir des samples aléatoires
    import random
    random.seed(42)
    samples = random.sample(phrases, min(samples, len(phrases)))
    
    for fr, expected_gloss in samples:
        print(f"FR: {fr}")
        print(f"Expected: {expected_gloss}")
        
        # Règles
        rules_result = text_to_gloss_fr(fr)
        rules_gloss = rules_result['gloss_string'].replace(" ||", "").strip()
        print(f"Règles:   {rules_gloss}")
        
        # Modèle
        if model_path:
            model_gloss = translate_with_model(fr, model_path)
            print(f"Modèle:   {model_gloss}")
        
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test et évaluation du modèle sur débats politiques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python test_political_debates.py --model model_politique/final
  python test_political_debates.py --model model_politique/final --debug
  python test_political_debates.py --dataset dataset_politique_5k.csv --model model_politique/final
  python test_political_debates.py --compare model_politique/final model_fr_gloss/final
        """,
    )
    parser.add_argument("--model", type=str, help="Chemin du modèle politique à tester")
    parser.add_argument("--dataset", type=str, help="CSV dataset pour évaluation (optionnel)")
    parser.add_argument("--compare", nargs=2, help="Comparer deux modèles")
    parser.add_argument("--debug", action="store_true", help="Afficher détails spaCy")
    parser.add_argument("--samples", type=int, default=20, help="Nombre de samples du dataset")
    
    args = parser.parse_args()
    
    print("=== Test: Traduction de débats politiques français → Gloss LSF ===\n")
    
    # Mode test sur phrases politiques
    if not args.dataset and not args.compare:
        print("Testing phrases politiques de référence...\n")
        for phrase in TEST_PHRASES_POLITIQUES:
            compare_translations(phrase, args.model)
            
            if args.debug:
                result = text_to_gloss_fr(phrase, debug=True)
                print()
    
    # Mode évaluation sur dataset
    elif args.dataset and args.model:
        evaluate_dataset(args.dataset, args.model, args.samples)
    
    # Mode comparaison deux modèles
    elif args.compare:
        print(f"Comparaison: {args.compare[0]} vs {args.compare[1]}\n")
        print("(À implémenter: comparer deux modèles sur les mêmes phrases)\n")
        # Décommentez pour implémenter:
        # for phrase in TEST_PHRASES_POLITIQUES[:10]:
        #     print(f"FR: {phrase}")
        #     gloss1 = translate_with_model(phrase, args.compare[0])
        #     gloss2 = translate_with_model(phrase, args.compare[1])
        #     print(f"Modèle 1: {gloss1}")
        #     print(f"Modèle 2: {gloss2}")
        #     print()


if __name__ == "__main__":
    main()
