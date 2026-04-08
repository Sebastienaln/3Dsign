import re
import pandas as pd
from rules_standalone import text_to_gloss_fr


def load_data(text_path, gloss_path):
    with open(text_path, "r", encoding="utf-8") as f:
        texts = [l.strip() for l in f.readlines()]
    with open(gloss_path, "r", encoding="utf-8") as f:
        glosses = [l.strip() for l in f.readlines()]

    assert len(texts) == len(glosses), "数据未对齐"
    print(f"Loaded {len(texts)} samples")
    return texts, glosses


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_gloss(text):
    text = text.upper()
    text = re.sub(r"[.,:;!?()\[\]\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid(text, gloss):
    t_len = len(text.split())
    g_len = len(gloss.split())

    if t_len == 0 or g_len == 0:
        return False
    if t_len < 3 or g_len < 2:
        return False
    if t_len > 100 or g_len > 100:
        return False

    ratio = g_len / t_len
    if ratio < 0.2 or ratio > 3:
        return False

    return True


def process_pipeline(texts, glosses):
    rows = []

    for t, g in zip(texts, glosses):
        t_clean = clean_text(t)
        g_clean = clean_gloss(g)

        if not is_valid(t_clean, g_clean):
            continue

        try:
            rules_out = text_to_gloss_fr(t_clean)
            rules_gloss = clean_gloss(rules_out["gloss_string"])
        except Exception as e:
            print(f"[WARN] failed on: {t_clean} -> {e}")
            rules_gloss = ""

        rows.append({
            "text": t_clean,
            "gloss": g_clean,
            "rules_gloss": rules_gloss,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    texts, glosses = load_data("phrases.txt", "glosses.txt")
    df = process_pipeline(texts, glosses)
    df.to_csv("clean_rule.csv", index=False, encoding="utf-8")
    print("Saved clean_rule.csv")