# Guide: Débats Politiques Français → Modèle LSF

## Vue d'ensemble

Vous avez maintenant un **pipeline complet** pour générer un dataset spécialisé aux débats politiques français et entraîner votre modèle dessus:

1. **`generate_dataset_debates.py`** : Génère des paires fr → gloss LSF **spécifiques aux débats politiques**
2. **`merge_datasets.py`** : Fusionne plusieurs datasets (politique + général)
3. **`train_model.py`** : Entraîne le modèle sur n'importe quel dataset

---

## Étape 1: Générer le dataset de débats politiques

### Prérequis
- Clé API OpenAI (ou une API LLM compatible)
- Package `openai` installé: `pip install openai`

### Génération simple (5000 phrases)

```bash
python generate_dataset_debates.py \
  --api-key "sk-..." \
  --count 5000 \
  --output dataset_politique_5k.csv
```

### Génération complète (10000 phrases, recommandé)

```bash
python generate_dataset_debates.py \
  --api-key "sk-..." \
  --count 10000 \
  --output dataset_politique_10k.csv \
  --model gpt-4o
```

### Options
- `--api-key` : Clé OpenAI (ou variable d'env `OPENAI_API_KEY`)
- `--count` : Nombre de paires à générer (défaut: 5000)
- `--output` : Nom du fichier CSV (défaut: `dataset_politique.csv`)
- `--model` : Modèle LLM (défaut: `gpt-4o-mini`, alternatives: `gpt-4o`, `gpt-3.5-turbo`)
- `--base-url` : API compatible OpenAI (Mistral, Ollama, etc.)

### Temps d'exécution
- **5000 phrases** ≈ 15-30 min
- **10000 phrases** ≈ 45-90 min
- Peut être **repris** si interrompu (reprend du dernier checkpoint)

### Catégories générées
30+ catégories politiques:
- Discours de campagne
- Critiques et débats
- Réformes et lois
- Économie, emploi, budget
- Éducation, santé, social
- Immigration, sécurité, justice
- Environnement, climat
- Relations internationales
- Corruption, transparence
- Et 20+ autres...

---

## Étape 2: Utiliser le dataset

### Option A: Entraîner directement sur les débats politiques

```bash
python train_model.py \
  --dataset dataset_politique_10k.csv \
  --epochs 15 \
  --batch-size 32 \
  --output-dir model_politique
```

Cela créera un modèle spécialisé dans les débats politiques.

### Option B: Fusionner avec le dataset général (recommandé)

Pour garder les capacités générales + ajouter la spécialité politique:

```bash
python merge_datasets.py dataset_merged.csv \
  dataset_politique_10k.csv \
  dataset_fr_gloss.csv
```

Résultat: `dataset_merged.csv` avec ~30k paires (dédupliquées, mélangées)

Puis entraîner:

```bash
python train_model.py \
  --dataset dataset_merged.csv \
  --epochs 20 \
  --batch-size 32 \
  --output-dir model_fr_gloss_merged
```

### Option C: Continuer l'entraînement d'un modèle existant

```bash
python train_model.py \
  --dataset dataset_politique_10k.csv \
  --base-model ./model_fr_gloss/final \
  --epochs 10 \
  --batch-size 32 \
  --output-dir model_politique_finetuned
```

Cela reprend les poids du modèle préexistant et les affine sur les débats (transfer learning).

---

## Étape 3: Tester le modèle entraîné

### Tester avec des phrases politiques

```bash
python translate.py \
  --model ./model_politique/final \
  --text "Le gouvernement doit réformer le système éducatif." \
  --debug
```

### Mode interactif

```bash
python translate.py --model ./model_politique/final --interactive
```

### Comparer avec le modèle de base

```bash
python compare_models.py
```

(À adapter si vous voulez comparer spécifiquement politique vs général)

---

## Caractéristiques du dataset généré

### Vocabulaire politique
- Gouvernement, élections, réformes, lois, budget, économie
- Débats, promesses, critiques, défenses politiques
- Chiffres, statistiques, sondages électoraux
- Positions idéologiques (gauche/droite/centre)

### Styles de langage
- Discours formel de campagne
- Débat parlementaire académique
- Critique passionnée de l'opposition
- Appel à mobilisation citoyenne
- Analyse politico-médiatique

### Diversité
- Promesses électorales
- Réformes économiques et sociales
- Enjeux environnementaux
- Immigration et intégration
- Sécurité et justice
- Relations internationales

---

## Architecture du modèle

```
Modèle de base: Helsinki-NLP/opus-mt-fr-en (MarianMT)
     ↓
Fine-tuning sur dataset_politique_10k.csv
     ↓
Modèle optimisé pour débats politiques français
```

Le modèle conserve les capacités de traduction générale tout en s'optimisant pour le vocabulaire et les constructions spécifiques aux débats.

---

## Commandes complètes (workflow complet)

### Scénario 1: Dataset politique pur (spécialisé)

```bash
# Étape 1: Générer les données
python generate_dataset_debates.py \
  --api-key $OPENAI_API_KEY \
  --count 10000 \
  --output dataset_politique_10k.csv

# Étape 2: Entraîner le modèle
python train_model.py \
  --dataset dataset_politique_10k.csv \
  --epochs 15 \
  --batch-size 32 \
  --output-dir model_politique_10k

# Étape 3: Tester
python translate.py \
  --model model_politique_10k/final \
  --interactive
```

### Scénario 2: Fusion avec données générales (hybride)

```bash
# Étape 1: Générer les données politiques
python generate_dataset_debates.py \
  --api-key $OPENAI_API_KEY \
  --count 10000 \
  --output dataset_politique_10k.csv

# Étape 2: Fusionner avec le dataset existant
python merge_datasets.py dataset_hybrid_30k.csv \  dataset_politique_10k.csv \
  dataset_fr_gloss.csv

# Étape 3: Entraîner le modèle hybride
python train_model.py \
  --dataset dataset_hybrid_30k.csv \
  --epochs 20 \
  --batch-size 32 \
  --output-dir model_hybrid_30k

# Étape 4: Tester
python translate.py \
  --model model_hybrid_30k/final \
  --interactive
```

### Scénario 3: Transfer learning (recommandé)

```bash
# Étape 1: Générer politiques
python generate_dataset_debates.py \
  --api-key $OPENAI_API_KEY \
  --count 5000 \
  --output dataset_politique_5k.csv

# Étape 2: Fine-tuner le modèle existant
python train_model.py \
  --dataset dataset_politique_5k.csv \
  --base-model ./model_fr_gloss_mega/final \
  --epochs 10 \
  --batch-size 32 \
  --output-dir model_politique_finetuned

# Étape 3: Tester
python translate.py \
  --model model_politique_finetuned/final \
  --text "Les réformes du gouvernement créent des emplois."
```

---

## Configuration recommandée

Pour un **meilleur résultat** (balancé généraliste + politique):

```bash
# 1. Générer 7500-10000 phrases de débats
python generate_dataset_debates.py --count 7500 --output dataset_politique_7k.csv

# 2. Fusionner avec données générales
python merge_datasets.py dataset_final.csv dataset_politique_7k.csv dataset_fr_gloss.csv

# 3. Entraîner avec les bonnes hyperparamètres
python train_model.py \
  --dataset dataset_final.csv \
  --epochs 15 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --output-dir model_final
```

---

## Troubleshooting

### "API rate limit exceeded"
→ Attendez 30-60 sec, le script reprendra automatiquement

### "ERREUR : impossible de parser la réponse JSON"
→ Problème temporaire avec l'API; le script réessaye automatiquement

### Le dataset s'arrête à N phrases
→ Peut être normal (rendement faible ou bonne couverture des catégories)
→ Relancez: le script reprendra et complètera

### Manque de mémoire GPU pendant entraînement
→ Réduisez `--batch-size` (16, 8)
→ Réduisez `--epochs`

---

## Prochaines étapes

1. ✅ **Générer dataset politique** → `generate_dataset_debates.py`
2. ✅ **Fusionner datasets** → `merge_datasets.py`
3. ✅ **Entraîner modèle** → `train_model.py`
4. ✅ **Tester traductions** → `translate.py` ou `fr_to_gloss.py`
5. 📊 **Évaluer qualité** → Comparer avec modèle de base via `compare_models.py`
