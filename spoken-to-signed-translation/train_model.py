"""
Script d'entraînement d'un modèle seq2seq pour la traduction français → gloss LSF.

Utilise Helsinki-NLP/opus-mt-fr-en comme base (architecture MarianMT) et le
fine-tune sur le dataset fr → gloss.

Usage :
    python train_model.py --dataset dataset_fr_gloss.csv
    python train_model.py --dataset dataset_fr_gloss.csv --epochs 20 --batch-size 16
    python train_model.py --dataset dataset_fr_gloss.csv --base-model t5-small --model-type t5
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def _get_resume_checkpoint(output_dir: str, resume_from: str | None) -> str | None:
    """Retourne le checkpoint de reprise explicite ou le dernier disponible."""
    if resume_from:
        return resume_from

    if not os.path.isdir(output_dir):
        return None

    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda name: int(name.split("-")[-1]))
    return os.path.join(output_dir, checkpoints[-1])

def load_dataset(csv_path: str) -> list[dict]:
    """Charge le dataset depuis un CSV."""
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = row.get("fr", "").strip()
            gloss = row.get("gloss", "").strip()
            if fr and gloss:
                pairs.append({"fr": fr, "gloss": gloss})
    return pairs


def split_dataset(pairs: list[dict], val_ratio: float = 0.1):
    """Sépare en train/validation."""
    import random
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_ratio))
    return pairs[:split_idx], pairs[split_idx:]


def train_marian(
    pairs_train: list[dict],
    pairs_val: list[dict],
    base_model: str,
    output_dir: str,
    resume_from_checkpoint: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Entraîne un modèle MarianMT (ou compatible)."""
    from transformers import (
        MarianTokenizer,
        MarianMTModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    import evaluate
    import numpy as np

    print(f"Chargement du tokenizer et modèle de base : {base_model}")
    tokenizer = MarianTokenizer.from_pretrained(base_model)
    model = MarianMTModel.from_pretrained(base_model)

    # Préparer les données au format HuggingFace Dataset
    def make_hf_dataset(pairs):
        return Dataset.from_dict({
            "fr": [p["fr"] for p in pairs],
            "gloss": [p["gloss"] for p in pairs],
        })

    train_dataset = make_hf_dataset(pairs_train)
    val_dataset = make_hf_dataset(pairs_val)

    def preprocess(examples):
        inputs = tokenizer(
            examples["fr"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            text_target=examples["gloss"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("Tokenisation du dataset...")
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["fr", "gloss"])
    val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=["fr", "gloss"])

    # Métrique BLEU
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Remplacer -100 par le token de padding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=10,
        save_total_limit=2,
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Début de l'entraînement ===")
    print(f"  Train : {len(pairs_train)} paires")
    print(f"  Val   : {len(pairs_val)} paires")
    print(f"  Epochs: {epochs}")
    print(f"  Batch : {batch_size}")
    print(f"  LR    : {learning_rate}")
    print()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Sauvegarder le meilleur modèle
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModèle sauvegardé dans {final_dir}")

    return final_dir


def train_t5(
    pairs_train: list[dict],
    pairs_val: list[dict],
    base_model: str,
    output_dir: str,
    resume_from_checkpoint: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Entraîne un modèle T5 / mT5."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    import evaluate
    import numpy as np

    print(f"Chargement du tokenizer et modèle de base : {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    PREFIX = "traduire français vers gloss LSF: "

    def make_hf_dataset(pairs):
        return Dataset.from_dict({
            "fr": [PREFIX + p["fr"] for p in pairs],
            "gloss": [p["gloss"] for p in pairs],
        })

    train_dataset = make_hf_dataset(pairs_train)
    val_dataset = make_hf_dataset(pairs_val)

    def preprocess(examples):
        inputs = tokenizer(
            examples["fr"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            text_target=examples["gloss"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("Tokenisation du dataset...")
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["fr", "gloss"])
    val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=["fr", "gloss"])

    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=10,
        save_total_limit=2,
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Début de l'entraînement ===")
    print(f"  Train : {len(pairs_train)} paires")
    print(f"  Val   : {len(pairs_val)} paires")
    print(f"  Epochs: {epochs}")
    print(f"  Batch : {batch_size}")
    print(f"  LR    : {learning_rate}")
    print()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModèle sauvegardé dans {final_dir}")

    return final_dir


def main():
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle de traduction français → gloss LSF",
    )
    parser.add_argument("--dataset", type=str, default="data/active/dataset_mega.csv",
                        help="Chemin vers le CSV du dataset (defaut: data/active/dataset_mega.csv)")
    parser.add_argument("--model-type", type=str, choices=["marian", "t5"], default="marian",
                        help="Type de modèle : marian ou t5 (défaut: marian)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Modèle pré-entraîné de base (défaut: auto selon model-type)")
    parser.add_argument("--output", type=str, default="model_fr_gloss",
                        help="Dossier de sortie du modèle (défaut: model_fr_gloss)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Chemin checkpoint pour reprendre (défaut: dernier checkpoint dans --output)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Nombre d'epochs (défaut: 30)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Taille de batch (défaut: 16)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (défaut: 5e-5)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Ratio de validation (défaut: 0.1)")
    args = parser.parse_args()

    # Modèle par défaut selon le type
    if args.base_model is None:
        if args.model_type == "marian":
            args.base_model = "Helsinki-NLP/opus-mt-fr-en"
        else:
            args.base_model = "google/mt5-small"

    # Charger le dataset
    print(f"Chargement du dataset : {args.dataset}")
    pairs = load_dataset(args.dataset)
    print(f"  {len(pairs)} paires chargées")

    if len(pairs) < 10:
        print("Erreur : pas assez de données pour entraîner un modèle.")
        sys.exit(1)

    pairs_train, pairs_val = split_dataset(pairs, args.val_ratio)
    print(f"  Train: {len(pairs_train)} | Val: {len(pairs_val)}")

    resume_checkpoint = _get_resume_checkpoint(args.output, args.resume_from)
    if resume_checkpoint:
        print(f"Reprise depuis checkpoint : {resume_checkpoint}")

    # Entraîner
    if args.model_type == "marian":
        final_dir = train_marian(
            pairs_train, pairs_val,
            args.base_model, args.output,
            resume_checkpoint,
            args.epochs, args.batch_size, args.lr,
        )
    else:
        final_dir = train_t5(
            pairs_train, pairs_val,
            args.base_model, args.output,
            resume_checkpoint,
            args.epochs, args.batch_size, args.lr,
        )

    print(f"\n=== Entraînement terminé ===")
    print(f"Modèle sauvegardé dans : {final_dir}")
    print(f"\nPour l'utiliser :")
    print(f"  python translate.py --model {final_dir} --text \"Je mange une pomme.\"")


if __name__ == "__main__":
    main()
