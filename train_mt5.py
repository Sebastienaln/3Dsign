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
CSV_PATH = "cleaned.csv"
OUTPUT_DIR = "./mt5_fr2gloss"

TEXT_COL = "text"
TARGET_COL = "gloss"

MAX_SOURCE_LENGTH = 96
MAX_TARGET_LENGTH = 32
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 15
NUM_BEAMS = 4

USE_FP16 = False

# small dataset prefix 小数据集建议加前缀，帮助模型明确任务
TASK_PREFIX = "translate French to gloss: "


# =========================
# 1. BASIC CHECKS
# =========================
def check_csv(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"can't find: {path}")

    df = pd.read_csv(path)
    required_cols = {TEXT_COL, TARGET_COL}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV should contain colone {required_cols}current column is: {list(df.columns)}"
        )


    # delete the valid data 去掉空值
    df = df.dropna(subset=[TEXT_COL, TARGET_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df = df[(df[TEXT_COL] != "") & (df[TARGET_COL] != "")].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("don't have samples after the preprocessing of the data")

    return df


# =========================
# 2. SPLIT DATA
# =========================
def build_splits(df: pd.DataFrame):
    # 打乱
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_df = df.iloc[:n_train]
    valid_df = df.iloc[n_train:n_train + n_valid]
    test_df = df.iloc[n_train + n_valid:]

    print(f"总样本数 total sample: {n}")
    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test : {len(test_df)}")

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
# 4. TOKENIZER / MODEL
# =========================
def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model


# =========================
# 5. PREPROCESS
# =========================
def preprocess_function(examples, tokenizer):
    inputs = [TASK_PREFIX + x for x in examples[TEXT_COL]]

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
# 6. METRICS
# =========================
bleu_metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    # 有些情况下 preds 是 tuple，取第一个
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.array(preds)
    labels = np.array(labels)

    # 如果 preds 不是 token ids，而是 logits（3维），取 argmax
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # 防止出现负数 id，统一替换成 pad_token_id
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    labels = np.where(labels < 0, tokenizer.pad_token_id, labels)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    pred_lens = [len(x.split()) for x in decoded_preds]

    return {
        "bleu": round(bleu["score"], 4),
        "gen_len": round(float(np.mean(pred_lens)), 4),
    }

def save_prediction_examples(raw_dataset, model, tokenizer, save_path, sample_size=20):
    test_raw = raw_dataset["test"]
    sample_size = min(sample_size, len(test_raw))
    indices = random.sample(range(len(test_raw)), sample_size)

    rows = []

    model.eval()

    for idx in indices:
        src = test_raw[idx][TEXT_COL]
        tgt = test_raw[idx][TARGET_COL]

        inputs = tokenizer(
            TASK_PREFIX + src,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=NUM_BEAMS,
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        rows.append({
            "index": idx,
            "source_text": src,
            "gold_gloss": tgt,
            "predicted_gloss": pred,
        })

    df_pred = pd.DataFrame(rows)
    df_pred.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Saved prediction examples to: {save_path}")

# =========================
# 7. MAIN
# =========================
def main():
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("Step 1: check cleaned.csv")
    df = check_csv(CSV_PATH)

    print("Step 2: split train/valid/test")
    train_path, valid_path, test_path = build_splits(df)

    print("Step 3: read the local dataset")
    raw_datasets = load_local_dataset(train_path, valid_path, test_path)

    print("Step 4: getting tokenizer and model")
    tokenizer, model = build_model_and_tokenizer()

    print("Step 5: tokenization")
    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=NUM_BEAMS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
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

    print("Step 6: Start training")
    trainer.train()

    print("Step 7: evaluation en test set")
    test_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
    print(test_metrics)

    print("Step 8: save model")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"training over, the result is in: {OUTPUT_DIR}")

    # 输出几条测试集预测，方便你肉眼检查
    print("\nStep 9: show randomly 5 examples")
    test_raw = raw_datasets["test"]
    sample_size = min(5, len(test_raw))
    indices = random.sample(range(len(test_raw)), sample_size)

    for idx in indices:
        src = test_raw[idx][TEXT_COL]
        tgt = test_raw[idx][TARGET_COL]

        inputs = tokenizer(
            TASK_PREFIX + src,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=NUM_BEAMS,
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("=" * 80)
        print("FR   :", src)
        print("GOLD :", tgt)
        print("PRED :", pred)
    print("\nStep 10: save prediction examples")
    save_prediction_examples(
        raw_dataset=raw_datasets,
        model=model,
        tokenizer=tokenizer,
        save_path=os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        sample_size=20,
    )

if __name__ == "__main__":
    main()