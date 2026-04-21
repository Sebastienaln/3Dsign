"""
Script de traduction français → gloss LSF avec système hybride.

Modes :
  --mode neural   : modèle neural seul
  --mode rules    : règles seules
  --mode hybrid   : neural + corrections par règles (défaut)

Usage :
    python translate.py --text "Je mange une pomme."
    python translate.py --text "Où habites-tu ?" --mode hybrid --compare
    python translate.py --interactive --mode hybrid
"""

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constantes pour le post-traitement hybride
# ---------------------------------------------------------------------------

PRONOUN_IX_MAP = {
    "je": "IX-1", "j'": "IX-1", "me": "IX-1", "moi": "IX-1", "m'": "IX-1",
    "tu": "IX-2", "te": "IX-2", "toi": "IX-2", "t'": "IX-2",
    "il": "IX-3", "elle": "IX-3", "lui": "IX-3",
    "se": "IX-3", "s'": "IX-3",
    "nous": "IX-1-PL", "on": "IX-1-PL",
    "vous": "IX-2-PL",
    "ils": "IX-3-PL", "elles": "IX-3-PL",
}

NEGATION_GLOSSES = {"PAS", "NON", "PLUS", "JAMAIS", "RIEN", "PERSONNE", "AUCUN"}

QUESTION_GLOSSES = {"OÙ", "QUAND", "COMMENT", "COMBIEN", "POURQUOI", "QUI", "QUOI"}

ARTICLES_FR = {
    "le", "la", "les", "l'", "un", "une", "des", "du", "de", "d'", "au", "aux",
}

DROP_PREPS = {"à", "de", "du", "des", "en", "par", "pour", "sur", "sous", "avec", "dans", "chez", "entre"}

AUXILIARY_LEMMAS = {"être", "avoir", "aller"}


# ---------------------------------------------------------------------------
# Chargement modèle neural
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Charge le modèle et le tokenizer."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print(f"Chargement du modèle depuis {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    is_t5 = "t5" in model.config.model_type.lower() if hasattr(model.config, "model_type") else False
    return tokenizer, model, is_t5


def neural_translate(text: str, tokenizer, model, is_t5: bool = False, num_beams: int = 4) -> str:
    """Traduit une phrase française en gloss LSF (modèle seul)."""
    import torch

    if is_t5:
        text = "traduire français vers gloss LSF: " + text

    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=128, num_beams=num_beams, early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Traduction par règles
# ---------------------------------------------------------------------------

def rules_translate(text: str) -> str:
    """Traduit via le système à règles et retourne la chaîne gloss."""
    sys.path.insert(0, str(Path(__file__).parent))
    from spoken_to_signed.text_to_gloss.rules_fr import text_to_gloss_fr
    result = text_to_gloss_fr(text)
    return result["gloss_string"].replace(" ||", "").replace(" |", "").strip()


# ---------------------------------------------------------------------------
# Post-traitement hybride : corrections par règles sur la sortie neurale
# (sans spaCy — utilise uniquement des listes de mots et regex)
# ---------------------------------------------------------------------------

# Mots français qui impliquent un pronom sujet
_FR_PRONOUN_SUBJECTS = {
    "je": "IX-1", "j'": "IX-1", "j\u2019": "IX-1", "j": "IX-1",
    "tu": "IX-2",
    "il": "IX-3", "elle": "IX-3",
    "on": "IX-1-PL", "nous": "IX-1-PL",
    "vous": "IX-2-PL",
    "ils": "IX-3-PL", "elles": "IX-3-PL",
}

# Pronoms objets français → gloss IX correspondant
# Inclut les formes élidées nues (t, m, l, s) après split sur l'apostrophe
_FR_PRONOUN_OBJECTS = {
    "me": "IX-1", "m'": "IX-1", "m\u2019": "IX-1", "m": "IX-1", "moi": "IX-1",
    "te": "IX-2", "t'": "IX-2", "t\u2019": "IX-2", "t": "IX-2", "toi": "IX-2",
    "le": "IX-3", "la": "IX-3", "l'": "IX-3", "l\u2019": "IX-3", "l": "IX-3", "lui": "IX-3",
    "se": "IX-3", "s'": "IX-3", "s\u2019": "IX-3", "s": "IX-3", "soi": "IX-3",
    "nous": "IX-1-PL",
    "vous": "IX-2-PL",
    "les": "IX-3-PL", "leur": "IX-3-PL", "eux": "IX-3-PL",
}

# Possessifs français → gloss POSS correspondant
_FR_POSSESSIVES = {
    "mon": "POSS-1", "ma": "POSS-1", "mes": "POSS-1",
    "ton": "POSS-2", "ta": "POSS-2", "tes": "POSS-2",
    "son": "POSS-3", "sa": "POSS-3", "ses": "POSS-3",
    "notre": "POSS-1-PL", "nos": "POSS-1-PL",
    "votre": "POSS-2-PL", "vos": "POSS-2-PL",
    "leur": "POSS-3-PL", "leurs": "POSS-3-PL",
}

# Verbes/expressions impersonnels : après "il", pas de IX-3
_IMPERSONAL_PATTERNS = {
    "il fait", "il pleut", "il neige", "il faut", "il y a",
    "il suffit", "il reste", "il manque", "il s'agit",
    "il semble", "il paraît", "il parait", "il convient",
    "il existe", "il vaut", "il importe",
    "il est tard", "il est tôt", "il est temps",
    "il est possible", "il est impossible", "il est nécessaire",
    "il est important", "il est interdit", "il est probable",
    "il est difficile", "il est facile", "il est clair",
    "il est évident", "il est vrai",
}

# Pluriels français courants (forme plurielle → lemme singulier)
_PLURAL_SUFFIXES = [
    ("eaux", "eau"), ("aux", "al"), ("eux", "eu"),
]

_FR_QUESTION_WORDS = {"qui", "que", "quoi", "où", "quand", "comment", "pourquoi", "combien", "quel", "quelle", "quels", "quelles"}

# Pronoms sujets inversés dans les questions (habites-tu, mange-t-il, etc.)
_INVERTED_PRONOUN_RE = re.compile(
    r"-(?:t-)?(je|tu|il|elle|on|nous|vous|ils|elles)\b", re.IGNORECASE
)

# Expressions françaises à supprimer du gloss (résidus grammaticaux)
_FR_NOISE_TOKENS = {"TRAIN", "EST-CE", "QUE", "CE", "SI", "NE", "N", "QU", "C", "D",
                    "PARCE", "BIEN-QUOI", "BIEN", "NO", "BAIS"}

_TEMPORAL_GLOSSES = {"HIER", "DEMAIN", "AUJOURD'HUI", "AUJOURDHUI", "MAINTENANT", "BIENTÔT", "BIENTOT", "TÔT", "TOT", "TARD"}

# Expressions périphrastiques à nettoyer
_PERIPHRASTIC_RE = re.compile(
    r"\b(en train de|est-ce que|est-ce qu|qu'est-ce que|qu'est-ce qu)\b", re.IGNORECASE
)

_EXCLAMATIVE_QUESTCEQUE_IL_FAIT_RE = re.compile(
    r"qu['’]?est[- ]?ce qu['’]?il fait\b", re.IGNORECASE
)

# Mots à ignorer pour la copie de termes inconnus depuis la source
_SOURCE_COPY_STOPWORDS = {
    "je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "moi", "toi", "lui", "leur", "leurs",
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l",
    "et", "ou", "mais", "donc", "or", "ni", "car", "que", "qui", "quoi",
    "où", "quand", "comment", "pourquoi", "combien",
    "à", "au", "aux", "en", "dans", "sur", "sous", "avec", "pour", "par", "chez",
    "est", "sont", "étais", "était", "être", "ai", "as", "a", "avons", "avez", "ont", "avoir",
    "vais", "va", "allons", "allez", "vont", "aller",
    "bonjour", "bonsoir", "salut",
}


def _is_impersonal_il(source_text: str) -> bool:
    """Vérifie si 'il' dans la phrase est impersonnel."""
    lower = source_text.lower()
    if re.search(r"\bil fait (beau|chaud|froid|gris|mauvais|bon)\b", lower):
        return True
    return any(lower.startswith(p) or (", " + p) in lower or ("; " + p) in lower
               for p in _IMPERSONAL_PATTERNS)


def _is_exclamative_questceque_il_fait(source_text: str) -> bool:
    """Détecte les tournures exclamatives du type 'qu'est-ce qu'il fait beau !'."""
    lower = source_text.lower().strip()
    return lower.endswith("!") and bool(_EXCLAMATIVE_QUESTCEQUE_IL_FAIT_RE.search(lower))


def _extract_copy_candidates(source_text: str) -> list[str]:
    """
    Extrait des termes source à recopier en gloss si absents de la sortie:
    - acronymes (SNCF, GPT)
    - tokens avec chiffres (A320, B2)
    - noms propres (Majuscule interne de phrase)
    - termes entre guillemets
    """
    candidates = []

    # 1) Termes explicitement cités entre guillemets
    for q in re.findall(r"[\"«](.*?)[\"»]", source_text):
        q = q.strip()
        if q:
            candidates.append(q.upper())

    # 2) Heuristiques sur les tokens
    raw_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'-]+", source_text)
    for i, tok in enumerate(raw_tokens):
        stripped = tok.strip("'’-")
        if not stripped:
            continue
        low = stripped.lower()
        if low in _SOURCE_COPY_STOPWORDS:
            continue

        is_acronym = stripped.isupper() and len(stripped) >= 2
        has_digit = any(ch.isdigit() for ch in stripped)
        # Nom propre: majuscule hors premier mot de phrase
        is_proper = stripped[0].isupper() and i > 0

        if is_acronym or has_digit or is_proper:
            candidates.append(stripped.upper())

    # Dédupliquer en conservant l'ordre
    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _detect_inverted_pronouns(source_text: str) -> list[str]:
    """Détecte les pronoms inversés dans les questions (habites-tu → IX-2)."""
    pronouns = []
    for m in _INVERTED_PRONOUN_RE.finditer(source_text.lower()):
        pron = m.group(1)
        if pron in _FR_PRONOUN_SUBJECTS:
            pronouns.append(_FR_PRONOUN_SUBJECTS[pron])
    return pronouns


def _detect_pronouns_in_source(source_text: str) -> tuple[list[str], list[str]]:
    """
    Détecte les pronoms sujets ET objets dans le texte source français.
    Retourne (sujets, objets) dans l'ordre d'apparition.
    Gère le 'il' impersonnel (pas de IX-3).
    """
    words = re.split(r"[\s''\u2019]+", source_text.lower().strip(" .!?"))
    impersonal = _is_impersonal_il(source_text)

    # Collecter sujets dans l'ordre d'apparition
    subjects = []
    seen_subj = set()
    for i, w in enumerate(words):
        if w in _FR_PRONOUN_SUBJECTS:
            ix = _FR_PRONOUN_SUBJECTS[w]
            if w == "il" and impersonal:
                continue
            if ix not in seen_subj:
                subjects.append(ix)
                seen_subj.add(ix)

    # Ajouter les pronoms inversés (habites-tu, mange-t-il, etc.)
    for ix in _detect_inverted_pronouns(source_text):
        if ix not in seen_subj:
            subjects.append(ix)
            seen_subj.add(ix)

    # Collecter objets
    objects = []
    seen_obj = set()
    for i, w in enumerate(words):
        if w in _FR_PRONOUN_OBJECTS:
            ix = _FR_PRONOUN_OBJECTS[w]
            # "le/la/les" sont souvent des articles, pas des pronoms objets.
            if w in {"le", "la", "les", "l"}:
                if i > 0 and words[i - 1] in _FR_PRONOUN_SUBJECTS:
                    if ix not in seen_obj:
                        objects.append(ix)
                        seen_obj.add(ix)
                continue
            # "nous/vous" : c'est un objet seulement si un autre sujet existe déjà
            if w in {"nous", "vous"}:
                if subjects and ix not in seen_subj:
                    if ix not in seen_obj:
                        objects.append(ix)
                        seen_obj.add(ix)
                continue
            # "leur" est souvent un possessif, pas un pronom objet
            # On l'ignore ici — les possessifs sont gérés séparément
            if w == "leur":
                continue
            # "se/s'" sont des réfléchis — pas d'IX séparé en LSF
            if w in {"se", "s", "s'", "s\u2019"}:
                continue
            # Pronoms objets clairs : t/t', m/m', te, me, lui, etc.
            if ix not in seen_obj:
                objects.append(ix)
                seen_obj.add(ix)

    return subjects, objects


def _detect_possessives_in_source(source_text: str) -> list[tuple[str, str]]:
    """
    Détecte les possessifs dans le texte source.
    Retourne une liste de (POSS-X, NOM_SUIVANT_UPPER) avec le nom singularisé.
    """
    words = re.split(r"[\s''\u2019]+", source_text.lower().strip(" .!?"))
    result = []
    for i, w in enumerate(words):
        if w in _FR_POSSESSIVES:
            poss = _FR_POSSESSIVES[w]
            if i + 1 < len(words):
                noun = words[i + 1]
                # Singulariser le nom pour matcher le gloss
                lemma = noun
                for pl_end, sg_end in _PLURAL_SUFFIXES:
                    if noun.endswith(pl_end):
                        lemma = noun[:-len(pl_end)] + sg_end
                        break
                else:
                    if noun.endswith("s") and not noun.endswith("ss"):
                        lemma = noun[:-1]
                result.append((poss, lemma.upper()))
            else:
                result.append((poss, None))
    return result


def _detect_plurals_in_source(source_text: str) -> set[str]:
    """Détecte les noms potentiellement pluriels dans le texte source (heuristique)."""
    words = re.split(r"[\s''\u2019]+", source_text.lower().strip(" .!?"))
    plurals = set()
    plural_determiners = {"les", "des", "mes", "tes", "ses", "nos", "vos", "leurs", "ces", "plusieurs", "quelques"}
    for i, w in enumerate(words):
        if w in plural_determiners and i + 1 < len(words):
            noun = words[i + 1]
            lemma = noun
            for pl_end, sg_end in _PLURAL_SUFFIXES:
                if noun.endswith(pl_end):
                    lemma = noun[:-len(pl_end)] + sg_end
                    break
            else:
                if noun.endswith("s") and not noun.endswith("ss"):
                    lemma = noun[:-1]
            plurals.add(lemma.upper())
    return plurals


def hybrid_postprocess(source_text: str, neural_gloss: str) -> str:
    """
    Applique des corrections basées sur les règles linguistiques LSF
    à la sortie brute du modèle neural (sans spaCy).

    Corrections appliquées :
      0. Nettoyage : majuscules, ponctuation, bruit grammatical
      1. Ajout des pronoms IX (sujets et objets) manquants
      2. Ajout des possessifs POSS manquants
      3. Suppression d'articles/prépositions résiduels
      4. Placement de la négation en fin de phrase
      5. Placement du mot interrogatif en fin de phrase
      6. Ajout du marqueur pluriel (+) si absent
      7. Dé-duplication
    """
    tokens = list(neural_gloss.split())

    # --- 0. Normaliser en majuscules + supprimer la ponctuation résiduelle ---
    tokens = [t.upper() for t in tokens]
    tokens = [t.strip("?!.,;:") for t in tokens]
    tokens = [t for t in tokens if t]  # supprimer les tokens vides

    # --- 0b. Supprimer les résidus grammaticaux français ---
    tokens = [t for t in tokens if t not in _FR_NOISE_TOKENS]

    # --- 1. Pronoms : vérifier que les IX sont présents ---
    source_subjects, source_objects = _detect_pronouns_in_source(source_text)
    existing_ix = {t for t in tokens if t.startswith("IX-") or t.startswith("POSS-")}
    expected_ix = set(source_subjects + source_objects)

    # Ajouter les pronoms sujets manquants dans l'ordre
    insert_offset = 0
    for ix in source_subjects:
        if ix not in existing_ix:
            tokens.insert(insert_offset, ix)
            existing_ix.add(ix)
        # Avancer après le dernier sujet IX existant ou inséré
        for i, t in enumerate(tokens):
            if t == ix:
                insert_offset = i + 1
                break

    # Ajouter les pronoms objets manquants (après le dernier sujet IX)
    obj_insert_pos = insert_offset
    for ix in source_objects:
        if ix not in existing_ix:
            tokens.insert(obj_insert_pos, ix)
            existing_ix.add(ix)
            obj_insert_pos += 1

    # Supprimer les IX parasites si la source n'a pas de pronom correspondant
    if expected_ix:
        cleaned_ix = []
        kept_expected = set()
        for t in tokens:
            if t.startswith("IX-"):
                if t in expected_ix and t not in kept_expected:
                    cleaned_ix.append(t)
                    kept_expected.add(t)
                continue
            cleaned_ix.append(t)
        tokens = cleaned_ix
    else:
        tokens = [t for t in tokens if not t.startswith("IX-")]

    # --- 2. Possessifs : vérifier que les POSS sont présents ---
    possessives = _detect_possessives_in_source(source_text)
    for poss, noun in possessives:
        if poss not in existing_ix:
            # Chercher le nom associé dans les tokens pour insérer POSS juste avant
            inserted = False
            for i, t in enumerate(tokens):
                base = t.rstrip("+")
                if noun and (base == noun or t == noun):
                    tokens.insert(i, poss)
                    inserted = True
                    break
            if not inserted:
                tokens.append(poss)
            existing_ix.add(poss)

    # Normalisation lexicale ciblée
    tokens = ["FALLOIR" if t == "FAUT" else t for t in tokens]

    # --- 3. Supprimer articles/prépositions qui auraient fuité ---
    cleaned = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower in ARTICLES_FR or t_lower in DROP_PREPS:
            continue
        if t_lower in {"être", "avoir", "est", "sont", "a", "ont", "été"}:
            continue
        cleaned.append(t)
    tokens = cleaned

    # --- 3b. Corrections lexicales ciblées à partir de la source ---
    source_lower = source_text.lower()

    # "aller à l'école" : si ECOLE manque, l'injecter avant ALLER
    if ("école" in source_lower or "ecole" in source_lower) and all(t not in {"ÉCOLE", "ECOLE"} for t in tokens):
        if "ALLER" in tokens:
            tokens.insert(tokens.index("ALLER"), "ÉCOLE")
        else:
            tokens.append("ÉCOLE")

    # Si la phrase source ne commence pas par un marqueur temporel,
    # éviter les sorties du type "DEMAIN IX-3 IX-2 PARLER".
    source_words = re.split(r"[\s''\u2019]+", source_text.lower().strip(" .!?"))
    source_starts_temporal = bool(source_words) and source_words[0] in {
        "hier", "demain", "aujourd'hui", "aujourdhui", "maintenant", "bientôt", "bientot", "tôt", "tot", "tard"
    }
    if tokens and tokens[0] in _TEMPORAL_GLOSSES and not source_starts_temporal:
        ix_tokens = [t for t in tokens if t.startswith("IX-") or t.startswith("POSS-")]
        other_tokens = [t for t in tokens if not (t.startswith("IX-") or t.startswith("POSS-"))]
        if ix_tokens:
            tokens = ix_tokens + other_tokens

    # --- 4. Question : mot interrogatif → fin de phrase ---
    #     (avant négation pour que la négation aille après le mot interrogatif)
    source_lower = source_text.lower()
    is_exclamative = _is_exclamative_questceque_il_fait(source_text)
    is_question = (source_text.strip().endswith("?") or any(
        w in source_lower for w in _FR_QUESTION_WORDS
    )) and not is_exclamative
    if is_question:
        q_tokens = []
        non_q_tokens = []
        for t in tokens:
            if t in QUESTION_GLOSSES:
                q_tokens.append(t)
            else:
                non_q_tokens.append(t)
        tokens = non_q_tokens + q_tokens
    elif is_exclamative:
        tokens = [t for t in tokens if t != "QUOI"]

    # --- 5. Négation → fin de phrase (après les mots interrogatifs) ---
    neg_tokens = []
    rest_tokens = []
    for t in tokens:
        if t in NEGATION_GLOSSES:
            neg_tokens.append(t)
        else:
            rest_tokens.append(t)
    neg_tokens = ["PAS" if t == "NON" else t for t in neg_tokens]
    # "plus jamais" : réinjecter PLUS s'il a disparu alors que la source l'impose.
    source_lower = source_text.lower()
    if "plus jamais" in source_lower and "JAMAIS" in neg_tokens and "PLUS" not in neg_tokens:
        neg_tokens.insert(0, "PLUS")
    tokens = rest_tokens + neg_tokens

    # --- 6. Pluriel : vérifier les marqueurs + ---
    plural_lemmas = _detect_plurals_in_source(source_text)
    if plural_lemmas:
        new_tokens = []
        for t in tokens:
            base = t.rstrip("+")
            if base in plural_lemmas and not t.endswith("+"):
                new_tokens.append(base + "+")
            else:
                new_tokens.append(t)
        tokens = new_tokens

    # --- 7. Dédupliquer les tokens consécutifs identiques ---
    deduped = []
    for t in tokens:
        if not deduped or t != deduped[-1]:
            deduped.append(t)
    tokens = deduped

    # --- 8. Copie de termes inconnus (fallback) ---
    # Ajoute des termes source importants (noms propres/acronymes/etc.)
    # si absents de la sortie gloss.
    copy_candidates = _extract_copy_candidates(source_text)
    for c in copy_candidates:
        if c not in tokens:
            tokens.append(c)

    # Ajustement ciblé: "se lever tôt" -> LEVER TÔT (pas TÔT LEVER)
    for i in range(len(tokens) - 1):
        if tokens[i] in {"TÔT", "TOT"} and tokens[i + 1] == "LEVER":
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
            break

    # Subordonnée "vouloir que ..." : placer les IX en tête de clause.
    if " que " in f" {source_text.lower()} " and "VOULOIR" in tokens:
        ix_tokens = [t for t in tokens if t.startswith("IX-") or t.startswith("POSS-")]
        other_tokens = [t for t in tokens if not (t.startswith("IX-") or t.startswith("POSS-"))]
        if ix_tokens and tokens and not (tokens[0].startswith("IX-") or tokens[0].startswith("POSS-")):
            tokens = ix_tokens + other_tokens

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Fonction de traduction principale
# ---------------------------------------------------------------------------

def translate(text: str, mode: str, tokenizer=None, model=None,
              is_t5: bool = False, num_beams: int = 4) -> str:
    """
    Traduit selon le mode choisi.
      - 'neural'  : modèle seul
      - 'rules'   : règles seules
      - 'hybrid'  : neural + post-traitement par règles
    """
    if mode == "rules":
        return rules_translate(text)

    neural_gloss = neural_translate(text, tokenizer, model, is_t5, num_beams)

    if mode == "hybrid":
        return hybrid_postprocess(text, neural_gloss)

    return neural_gloss  # mode == "neural"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Traduction français → gloss LSF (hybride neural + règles)",
    )
    parser.add_argument("--model", type=str, default="model_fr_gloss_mega_politique/final",
                        help="Chemin vers le modèle entraîné")
    parser.add_argument("--text", type=str, default=None,
                        help="Phrase en français à traduire")
    parser.add_argument("--mode", choices=["neural", "rules", "hybrid"],
                        default="hybrid",
                        help="Mode de traduction (défaut: hybrid)")
    parser.add_argument("--interactive", action="store_true",
                        help="Mode interactif")
    parser.add_argument("--compare", action="store_true",
                        help="Afficher les 3 modes côte à côte")
    parser.add_argument("--beams", type=int, default=4,
                        help="Nombre de beams (défaut: 4)")
    args = parser.parse_args()

    tokenizer, model, is_t5 = None, None, False
    if args.mode in ("neural", "hybrid"):
        tokenizer, model, is_t5 = load_model(args.model)
        print("Modèle chargé.\n")

    def do_translate(text: str):
        result = translate(text, args.mode, tokenizer, model, is_t5, args.beams)
        print(result)

        if args.compare:
            print()
            if args.mode != "neural":
                neural = neural_translate(text, tokenizer, model, is_t5, args.beams)
                print(f"  Neural : {neural}")
            if args.mode != "rules":
                rules = rules_translate(text)
                print(f"  Règles : {rules}")
            if args.mode == "hybrid":
                # Déjà affiché en résultat principal
                pass
            elif args.mode == "neural":
                hybrid = hybrid_postprocess(text, result)
                print(f"  Hybride: {hybrid}")
                rules = rules_translate(text)
                print(f"  Règles : {rules}")
            elif args.mode == "rules":
                neural = neural_translate(text, tokenizer, model, is_t5, args.beams)
                print(f"  Neural : {neural}")
                hybrid = hybrid_postprocess(text, neural)
                print(f"  Hybride: {hybrid}")

    if args.interactive:
        mode_label = {"neural": "Neural", "rules": "Règles", "hybrid": "Hybride"}
        print(f"=== Traducteur FR → Gloss LSF [{mode_label[args.mode]}] ===")
        print("Tapez une phrase en français (ou 'q' pour quitter)\n")
        while True:
            try:
                text = input("FR > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in {"q", "quit", "exit", "quitter"}:
                break
            if not text:
                continue
            do_translate(text)
            print()

    elif args.text:
        do_translate(args.text)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
