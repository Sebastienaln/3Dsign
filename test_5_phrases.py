"""
Script de test : 5 phrases avec les 3 modes (neural, rules, hybrid)
"""

from translate import load_model, neural_translate, rules_translate, hybrid_postprocess

# 5 phrases de test
test_phrases = [
    "Je bois un café.",
    "Où habites-tu ?",
    "Elle aime manger des pommes.",
    "Nous allons à l'école demain.",
    "Tu ne dois pas faire ça.",
]

print("=" * 80)
print("CHARGEMENT DES MODÈLES")
print("=" * 80)

print("\n📦 Chargement modèle FR-Gloss 20k...")
tok20, mdl20, is_t5_20 = load_model("model_fr_gloss_20k/final")
print("✅ Chargé !")

print("\n" + "=" * 80)
print("TEST SUR 5 PHRASES")
print("=" * 80)

for i, phrase in enumerate(test_phrases, 1):
    print(f"\n{'─' * 80}")
    print(f"Phrase {i}: {phrase}")
    print(f"{'─' * 80}")
    
    # Mode Neural
    print("\n🧠 MODE NEURAL:")
    neural_out = neural_translate(phrase, tok20, mdl20, is_t5_20)
    print(f"   {neural_out}")
    
    # Mode Rules
    print("\n📐 MODE RÈGLES:")
    try:
        rules_out = rules_translate(phrase)
        print(f"   {rules_out}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Mode Hybrid (Neural + Règles)
    print("\n🔀 MODE HYBRID (Neural + Règles):")
    hybrid_out = hybrid_postprocess(phrase, neural_out)
    print(f"   {hybrid_out}")

print("\n" + "=" * 80)
print("✅ TEST TERMINÉ")
print("=" * 80)
