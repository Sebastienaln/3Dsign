import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
def load_data(text_path, gloss_path):
    with open(text_path, "r", encoding="utf-8") as f:
        texts = [l.strip() for l in f.readlines()]
    with open(gloss_path, "r", encoding="utf-8") as f:
        glosses = [l.strip() for l in f.readlines()]

    assert len(texts) == len(glosses), " 数据未对齐"
    print(f" Loaded {len(texts)} samples")
    return texts, glosses


# =========================
# 2. CLEAN FUNCTIONS
# =========================
def clean_gloss(text):
    text = text.upper() 

    # remove punctuation
    text = re.sub(r"[.,:;!?()\[\]-]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # entity normalization
    entity_map = {
        "EMMANUEL MACRON": "MACRON",
        "M MACRON": "MACRON",
        "MARINE LE PEN": "LE_PEN",
        "LE PEN": "LE_PEN"
    }
    for k, v in entity_map.items():
        text = text.replace(k, v)

    # remove filler
    fillers = ["VOIR", "VOICI", "ALORS", "BON"]
    tokens = [t for t in text.split() if t not in fillers]

    return " ".join(tokens)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# 3. FILTERING RULES
# =========================
def is_valid(text, gloss):
    t_len = len(text.split())
    g_len = len(gloss.split())

    # empty
    if t_len == 0 or g_len == 0:
        return False

    # too short / too long
    if t_len < 3 or g_len < 2:
        return False
    if t_len > 100 or g_len > 100:
        return False

    # length ratio
    ratio = g_len / t_len
    if ratio < 0.2 or ratio > 3:
        return False

    return True


# =========================
# 4. PIPELINE
# =========================
def process_pipeline(texts, glosses):
    cleaned_data = []

    for t, g in zip(texts, glosses):
        t_clean = clean_text(t)
        g_clean = clean_gloss(g)

        if is_valid(t_clean, g_clean):
            cleaned_data.append((t_clean, g_clean))

    print(f" After cleaning: {len(cleaned_data)} samples")
    return cleaned_data


# =========================
# 5. STATISTICS
# =========================
def compute_stats(data):
    text_lens = [len(t.split()) for t, _ in data]
    gloss_lens = [len(g.split()) for _, g in data]

    print("\n DATA STATISTICS")
    print(f"Text avg length: {np.mean(text_lens):.2f}")
    print(f"Gloss avg length: {np.mean(gloss_lens):.2f}")
    print(f"Max text length: {max(text_lens)}")
    print(f"Max gloss length: {max(gloss_lens)}")

    return text_lens, gloss_lens


# =========================
# 6. VOCAB ANALYSIS
# =========================
def vocab_stats(data):
    gloss_tokens = []
    for _, g in data:
        gloss_tokens.extend(g.split())

    counter = Counter(gloss_tokens)

    print("\n VOCAB STATS")
    print(f"Vocab size: {len(counter)}")

    print("\nTop 20 tokens:")
    for k, v in counter.most_common(20):
        print(k, v)

    return counter


# =========================
# 7. VISUALIZATION
# =========================
def plot_distributions(text_lens, gloss_lens):
    plt.figure()
    plt.hist(text_lens, bins=50)
    plt.title("Text Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("text_length_distribution.png")
    plt.close()

    plt.figure()
    plt.hist(gloss_lens, bins=50)
    plt.title("Gloss Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("gloss_length_distribution.png")
    plt.close()


def plot_ratio(data):
    ratios = [len(g.split()) / len(t.split()) for t, g in data]

    plt.figure()
    plt.hist(ratios, bins=50)
    plt.title("Length Ratio (gloss/text)")
    plt.xlabel("Ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("length_ratio_distribution.png")
    plt.close()


def plot_top_vocab(counter, top_k=20):
    top = counter.most_common(top_k)
    words = [x[0] for x in top]
    freqs = [x[1] for x in top]

    plt.figure(figsize=(12, 5))
    plt.bar(words, freqs)
    plt.xticks(rotation=45)
    plt.title("Top Gloss Tokens")
    plt.tight_layout()
    plt.savefig("top_gloss_tokens.png")
    plt.close()


# =========================
# 8. SAVE CLEAN DATA
# =========================
def save_data(data, path="cleaned.csv"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("text,gloss\n")
        for t, g in data:
            f.write(f'"{t}","{g}"\n')

    print(f" Saved to {path}")


# =========================
# 9. MAIN
# =========================
if __name__ == "__main__":
    texts, glosses = load_data("phrases.txt", "glosses.txt")

    data = process_pipeline(texts, glosses)

    text_lens, gloss_lens = compute_stats(data)
    counter = vocab_stats(data)

    save_data(data, "cleaned.csv")   

    plot_distributions(text_lens, gloss_lens)
    plot_ratio(data)
    plot_top_vocab(counter)

    print("All done.")