import os
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
import evaluate

from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# =========================
# 0. CONFIG
# =========================
MODEL_NAME = "google/mt5-small"
CSV_PATH = "clean_rule.csv"
OUTPUT_DIR = "./mt5_fr2gloss_with_rules_v2"

TEXT_COL = "text"
RULE_COL = "rules_gloss"
TARGET_COL = "gloss"

MAX_SOURCE_LENGTH = 160
MAX_TARGET_LENGTH = 32

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 19
NUM_BEAMS = 4

USE_FP16 = False

EARLY_STOPPING_PATIENCE = 3
LABEL_SMOOTHING = 0.1

TASK_PREFIX = "translate French and rule gloss to gold gloss: "


# =========================
# 1. BASIC CHECKS
# =========================
def check_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Can't find file: {path}")

    df = pd.read_csv(path)

    required_cols = {TEXT_COL, RULE_COL, TARGET_COL}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, current columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TEXT_COL, RULE_COL, TARGET_COL]).copy()

    for col in [TEXT_COL, RULE_COL, TARGET_COL]:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df[TEXT_COL] != "") &
        (df[RULE_COL] != "") &
        (df[TARGET_COL] != "")
    ].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid samples after preprocessing")

    print(f"Valid samples after cleaning: {len(df)}")
    return df


# =========================
# 2. SPLIT DATA
# =========================
def build_splits(df: pd.DataFrame):
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_df = df.iloc[:n_train]
    valid_df = df.iloc[n_train:n_train + n_valid]
    test_df = df.iloc[n_train + n_valid:]

    print(f"Total samples: {n}")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test : {len(test_df)}")

    train_path = "train_tmp.csv"
    valid_path = "valid_tmp.csv"
    test_path = "test_tmp.csv"

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, valid_path, test_path


# =========================
# 3. LOAD DATASET
# =========================
def load_local_dataset(train_path, valid_path, test_path):
    dataset = load_dataset(
        "csv",
        data_files={
            "train": train_path,
            "validation": valid_path,
            "test": test_path,
        },
    )
    return dataset


# =========================
# 4. MODEL / TOKENIZER
# =========================
def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = MAX_TARGET_LENGTH
    model.generation_config.num_beams = NUM_BEAMS

    return tokenizer, model


# =========================
# 5. BUILD INPUT
# =========================
def build_input_text(text, rule_gloss):
    return f"{TASK_PREFIX}French: {text} || RuleGloss: {rule_gloss}"


# =========================
# 6. PREPROCESS
# =========================
def preprocess_function(examples, tokenizer):
    inputs = [
        build_input_text(text, rule)
        for text, rule in zip(examples[TEXT_COL], examples[RULE_COL])
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples[TARGET_COL],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# =========================
# 7. METRICS
# =========================
bleu_metric = evaluate.load("sacrebleu")


def normalize_for_match(s: str) -> str:
    return " ".join(str(s).strip().upper().split())


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.array(preds)
    labels = np.array(labels)

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    labels = np.where(labels < 0, tokenizer.pad_token_id, labels)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels_text = [label.strip() for label in decoded_labels]
    decoded_labels_bleu = [[label] for label in decoded_labels_text]

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels_bleu
    )

    pred_lens = [len(x.split()) for x in decoded_preds]

    exact_matches = [
        int(normalize_for_match(p) == normalize_for_match(g))
        for p, g in zip(decoded_preds, decoded_labels_text)
    ]
    exact_match = float(np.mean(exact_matches)) if exact_matches else 0.0

    return {
        "bleu": round(bleu["score"], 4),
        "exact_match": round(exact_match, 4),
        "gen_len": round(float(np.mean(pred_lens)), 4),
    }


# =========================
# 8. SAVE TEST PREDICTIONS
# =========================
def predict_samples_to_csv(trainer, tokenizer, raw_test_dataset, save_path):
    rows = []

    for i in range(len(raw_test_dataset)):
        text = raw_test_dataset[i][TEXT_COL]
        rule = raw_test_dataset[i][RULE_COL]
        gold = raw_test_dataset[i][TARGET_COL]

        model_input = build_input_text(text, rule)

        inputs = tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        ).to(trainer.model.device)

        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            max_length=None,
            num_beams=NUM_BEAMS,
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        rows.append({
            "text": text,
            "rules_gloss": rule,
            "gold_gloss": gold,
            "pred_gloss": pred,
            "exact_match": int(normalize_for_match(pred) == normalize_for_match(gold)),
        })

    df_pred = pd.DataFrame(rows)
    df_pred.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Saved test predictions to: {save_path}")


# =========================
# 9. SHOW RANDOM EXAMPLES
# =========================
def show_random_examples(trainer, tokenizer, raw_test_dataset, n=5):
    sample_size = min(n, len(raw_test_dataset))
    indices = random.sample(range(len(raw_test_dataset)), sample_size)

    for idx in indices:
        text = raw_test_dataset[idx][TEXT_COL]
        rule = raw_test_dataset[idx][RULE_COL]
        gold = raw_test_dataset[idx][TARGET_COL]

        model_input = build_input_text(text, rule)

        inputs = tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        ).to(trainer.model.device)

        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            max_length=None,
            num_beams=NUM_BEAMS,
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print("=" * 100)
        print("INPUT :", model_input)
        print("GOLD  :", gold)
        print("PRED  :", pred)
        print("MATCH :", normalize_for_match(pred) == normalize_for_match(gold))


# =========================
# 10. MAIN
# =========================
def main():
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("Step 1: check CSV")
    df = check_csv(CSV_PATH)

    print("Step 2: split train / valid / test")
    train_path, valid_path, test_path = build_splits(df)

    print("Step 3: load dataset")
    raw_datasets = load_local_dataset(train_path, valid_path, test_path)

    print("Step 4: build tokenizer and model")
    tokenizer, model = build_model_and_tokenizer()

    print("Step 5: tokenize")
    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=None,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        num_train_epochs=NUM_EPOCHS,

        predict_with_generate=True,
        generation_num_beams=NUM_BEAMS,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,

        label_smoothing_factor=LABEL_SMOOTHING,
        fp16=USE_FP16,

        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )

    print("Step 6: start training")
    trainer.train()

    print("Step 7: evaluate on test set")
    test_metrics = trainer.evaluate(
        tokenized_datasets["test"],
        metric_key_prefix="test"
    )
    print(test_metrics)

    print("Step 8: save model")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")

    print("Step 9: save test predictions")
    predict_samples_to_csv(
        trainer=trainer,
        tokenizer=tokenizer,
        raw_test_dataset=raw_datasets["test"],
        save_path=os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    )

    print("Step 10: show random examples")
    show_random_examples(
        trainer=trainer,
        tokenizer=tokenizer,
        raw_test_dataset=raw_datasets["test"],
        n=5,
    )


if __name__ == "__main__":
    main()