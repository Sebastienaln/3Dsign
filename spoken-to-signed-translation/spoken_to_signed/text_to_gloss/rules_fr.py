"""
Traducteur français → gloss LSF (Langue des Signes Française) basé sur des règles.

Règles principales appliquées :
  1. Analyse morphosyntaxique via spaCy (modèle français)
  2. Suppression des articles (le, la, les, un, une, des…)
  3. Suppression des prépositions non-significatives
  4. Remplacement des pronoms par des signes d'indexation (IX)
  5. Remplacement des possessifs par POSS-1/2/3
  6. Lemmatisation des verbes (infinitif) et noms
  7. Marquage du pluriel (NOM+)
  8. Déplacement du temps en début de phrase
  9. Déplacement du lieu après le temps
  10. Réordonnancement SOV (Sujet-Objet-Verbe)
  11. Déplacement de la négation en fin de clause
  12. Marqueur interrogatif en fin de phrase
  13. Mise en majuscules (convention gloss)
"""

import sys
from typing import Optional

from .common import load_spacy_model
from .types import Gloss

SPACY_MODELS_FR = ("fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm")

# --- Mappings pour les pronoms personnels → signes d'indexation ---
PRONOUN_IX_MAP = {
    "je": "IX-1",
    "j'": "IX-1",
    "me": "IX-1",
    "moi": "IX-1",
    "m'": "IX-1",
    "tu": "IX-2",
    "te": "IX-2",
    "toi": "IX-2",
    "t'": "IX-2",
    "il": "IX-3",
    "elle": "IX-3",
    "lui": "IX-3",
    "le": "IX-3",
    "la": "IX-3",
    "l'": "IX-3",
    "se": "IX-3",
    "s'": "IX-3",
    "soi": "IX-3",
    "nous": "IX-1-PL",
    "vous": "IX-2-PL",
    "ils": "IX-3-PL",
    "elles": "IX-3-PL",
    "les": "IX-3-PL",
    "leur": "IX-3-PL",
    "eux": "IX-3-PL",
    "on": "IX-1-PL",
    "ce": "IX-3",
    "c'": "IX-3",
    "ça": "IX-3",
    "cela": "IX-3",
    "ceci": "IX-3",
}

# --- Mappings pour les déterminants possessifs → POSS ---
POSSESSIVE_MAP = {
    "mon": "POSS-1",
    "ma": "POSS-1",
    "mes": "POSS-1",
    "ton": "POSS-2",
    "ta": "POSS-2",
    "tes": "POSS-2",
    "son": "POSS-3",
    "sa": "POSS-3",
    "ses": "POSS-3",
    "notre": "POSS-1-PL",
    "nos": "POSS-1-PL",
    "votre": "POSS-2-PL",
    "vos": "POSS-2-PL",
    "leur": "POSS-3-PL",
    "leurs": "POSS-3-PL",
}

# --- Mots de temps courants ---
TIME_WORDS = {
    "hier", "aujourd'hui", "demain", "maintenant", "avant", "après",
    "bientôt", "déjà", "toujours", "jamais", "souvent", "parfois",
    "longtemps", "récemment", "autrefois", "actuellement", "ensuite",
    "puis", "tard", "tôt", "lundi", "mardi", "mercredi", "jeudi",
    "vendredi", "samedi", "dimanche", "janvier", "février", "mars",
    "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
    "novembre", "décembre", "matin", "soir", "nuit", "midi",
    "semaine", "mois", "année", "an",
}

# --- Articles à supprimer ---
ARTICLES = {"le", "la", "les", "l'", "un", "une", "des", "du", "de", "d'", "au", "aux"}

# --- Prépositions à supprimer (non-significatives en LSF) ---
DROP_PREPOSITIONS = {"à", "de", "du", "des", "en", "par", "pour", "sur", "sous", "avec", "dans", "chez", "entre"}

# --- Auxiliaires à supprimer ---
AUXILIARY_LEMMAS = {"être", "avoir", "aller"}

# --- Mots de négation ---
NEGATION_WORDS = {"ne", "n'", "pas", "plus", "jamais", "rien", "aucun", "aucune", "personne", "guère", "point"}

# --- Mots interrogatifs ---
QUESTION_WORDS = {"qui", "que", "quoi", "où", "quand", "comment", "pourquoi", "combien", "quel", "quelle", "quels", "quelles", "lequel", "laquelle", "lesquels", "lesquelles", "est-ce"}

# Suffixes d'infinitif français
_INFINITIVE_ENDINGS = ("er", "ir", "re", "oir")


def _fix_verb_lemma(lemma: str, text: str) -> str:
    """Corrige le lemme d'un verbe si spaCy ne retourne pas l'infinitif."""
    if lemma.endswith(_INFINITIVE_ENDINGS):
        return lemma
    # Essayer de reconstruire l'infinitif à partir de la forme fléchie
    lower = text.lower()
    # Verbes en -er (1er groupe, ~90% des verbes français)
    for suffix in ("e", "es", "ent", "ons", "ez", "ais", "ait", "aient", "ai", "as", "a", "é", "ée", "és", "ées", "ant"):
        if lower.endswith(suffix):
            stem = lower[: len(lower) - len(suffix)]
            if len(stem) >= 2:
                return stem + "er"
    return lemma


def _debug_token(token):
    """Affiche les informations détaillées d'un token (debug)."""
    print(
        f"  {token.text:15s} lemma={token.lemma_:15s} pos={token.pos_:6s} "
        f"tag={token.tag_:6s} dep={token.dep_:10s} head={str(token.head):15s} "
        f"morph={token.morph}",
        file=sys.stderr,
    )


def _is_article(token) -> bool:
    """Vérifie si le token est un article (à supprimer)."""
    if token.pos_ == "DET" and "Poss=Yes" not in str(token.morph):
        return True
    if token.text.lower() in ARTICLES and token.pos_ in {"DET", "ADP"}:
        return True
    return False


def _is_auxiliary(token) -> bool:
    """Vérifie si le token est un verbe auxiliaire."""
    return token.pos_ == "AUX" and token.lemma_ in AUXILIARY_LEMMAS


def _is_drop_preposition(token) -> bool:
    """Vérifie si la préposition doit être supprimée."""
    return token.pos_ == "ADP" and token.text.lower() in DROP_PREPOSITIONS


def _is_negation(token) -> bool:
    """Vérifie si le token est un mot de négation."""
    return token.text.lower() in NEGATION_WORDS and token.dep_ in {"advmod", "det", "obj", "nsubj", "obl"}


def _is_question(doc) -> bool:
    """Vérifie si la phrase est interrogative."""
    text = doc.text.strip()
    if text.endswith("?"):
        return True
    for token in doc:
        if token.text.lower() in QUESTION_WORDS:
            return True
    return False


def _get_pronoun_gloss(token) -> Optional[str]:
    """Retourne le gloss d'indexation pour un pronom, ou None."""
    lower = token.text.lower()
    return PRONOUN_IX_MAP.get(lower)


def _get_possessive_gloss(token) -> Optional[str]:
    """Retourne le gloss de possessif, ou None."""
    lower = token.text.lower()
    return POSSESSIVE_MAP.get(lower)


def _glossify_token(token) -> Optional[tuple[str, str]]:
    """
    Convertit un token spaCy en paire (gloss, mot_original).
    Retourne None si le token doit être supprimé.
    """
    text = token.text
    lower = text.lower()

    # Supprimer la ponctuation
    if token.pos_ == "PUNCT":
        return None

    # Supprimer les articles
    if _is_article(token):
        return None

    # Supprimer les auxiliaires (être, avoir, aller comme auxiliaire)
    if _is_auxiliary(token):
        return None

    # Supprimer les prépositions non-significatives
    if _is_drop_preposition(token):
        return None

    # Supprimer "ne"/"n'" (la négation est gérée séparément avec PAS)
    if lower in {"ne", "n'"}:
        return None

    # Pronoms → IX
    if token.pos_ == "PRON":
        # Pronoms relatifs : supprimer
        if token.dep_ in {"nsubj", "obj", "iobj", "obl"} or token.dep_.startswith("nsubj"):
            ix = _get_pronoun_gloss(token)
            if ix:
                return (ix, text)
        # Pronoms relatifs (qui, que, dont, où) → supprimer
        if token.dep_ in {"mark", "ref"} or lower in {"qui", "que", "qu'", "dont", "où"}:
            return None
        # Pronom par défaut
        ix = _get_pronoun_gloss(token)
        if ix:
            return (ix, text)
        return None

    # Déterminants possessifs → POSS
    if token.pos_ == "DET" and "Poss=Yes" in str(token.morph):
        poss = _get_possessive_gloss(token)
        if poss:
            return (poss, text)

    # Négation : "pas", "plus", "jamais", etc. → gloss spécifique
    if lower == "pas":
        return ("PAS", text)
    if lower == "plus" and token.dep_ == "advmod" and token.head.pos_ in {"VERB", "AUX"}:
        return ("PLUS", text)
    if lower == "jamais":
        return ("JAMAIS", text)
    if lower == "rien":
        return ("RIEN", text)
    if lower == "personne" and token.dep_ in {"obj", "nsubj"}:
        return ("PERSONNE", text)
    if lower in {"aucun", "aucune"} and token.dep_ == "det":
        return ("AUCUN", text)

    # Noms → lemme + marquage pluriel
    if token.pos_ == "NOUN" or token.pos_ == "PROPN":
        gloss = token.lemma_.upper()
        if "Number=Plur" in str(token.morph):
            gloss += "+"
        return (gloss, text)

    # Verbes → lemme (infinitif)
    if token.pos_ == "VERB":
        lemma = _fix_verb_lemma(token.lemma_, token.text)
        return (lemma.upper(), text)

    # Adjectifs → lemme
    if token.pos_ == "ADJ":
        return (token.lemma_.upper(), text)

    # Nombres
    if token.pos_ == "NUM":
        return (token.text.upper(), text)

    # Adverbes (sauf négation déjà traitée)
    if token.pos_ == "ADV":
        return (token.lemma_.upper(), text)

    # Conjonctions de coordination : garder "mais", supprimer "et"
    if token.pos_ in {"CCONJ", "SCONJ"}:
        if lower == "et":
            return None
        if lower == "mais":
            return ("MAIS", text)
        if lower == "ou":
            return ("OU", text)
        if lower in {"si", "quand", "parce", "comme", "car"}:
            return (token.text.upper(), text)
        return None

    # Interjections
    if token.pos_ == "INTJ":
        return (token.text.upper(), text)

    # Par défaut : lemme en majuscules
    return (token.lemma_.upper(), text)


def _reorder_clause(gloss_pairs: list[tuple[str, str]], doc) -> list[tuple[str, str]]:
    """
    Réordonne les éléments du gloss selon la grammaire LSF :
      Temps → Lieu → Sujet → Objet → Verbe → Négation → (Interrogatif)
    """
    time_items = []
    location_items = []
    subject_items = []
    object_items = []
    verb_items = []
    negation_items = []
    adjective_items = []
    other_items = []

    # On utilise la position du token original pour faire correspondre gloss et token
    token_map = {}
    for token in doc:
        token_map[token.text] = token

    for gloss, original in gloss_pairs:
        token = token_map.get(original)

        # Temps
        if original.lower() in TIME_WORDS:
            time_items.append((gloss, original))
        # Lieu (entité nommée LOC ou GPE)
        elif token and token.ent_type_ in {"LOC", "GPE"}:
            location_items.append((gloss, original))
        # Négation
        elif gloss in {"PAS", "PLUS", "JAMAIS", "RIEN", "PERSONNE", "AUCUN"}:
            negation_items.append((gloss, original))
        # Sujet
        elif token and token.dep_ in {"nsubj", "nsubj:pass"} or (
            token and token.dep_ == "expl" and gloss.startswith("IX")
        ):
            subject_items.append((gloss, original))
        # Objet
        elif token and token.dep_ in {"obj", "iobj", "obl", "obl:arg", "obl:agent", "obl:mod"}:
            object_items.append((gloss, original))
        # Verbe
        elif token and token.pos_ == "VERB":
            verb_items.append((gloss, original))
        # Adjectif : suit le nom qu'il qualifie (on les garde après les noms)
        elif token and token.pos_ == "ADJ":
            adjective_items.append((gloss, original))
        else:
            other_items.append((gloss, original))

    # Ordre LSF : TEMPS + LIEU + SUJET + autres + OBJET + ADJECTIF + VERBE + NÉGATION
    result = time_items + location_items + subject_items + other_items + object_items + adjective_items + verb_items + negation_items

    return result


def _process_sentence(doc, debug: bool = False) -> list[tuple[str, str]]:
    """Traite une phrase complète et retourne la liste de paires (gloss, mot)."""
    if debug:
        print("--- Analyse spaCy ---", file=sys.stderr)
        for token in doc:
            _debug_token(token)
        print("---", file=sys.stderr)

    # Étape 1 : Glossifier chaque token
    gloss_pairs = []
    for token in doc:
        result = _glossify_token(token)
        if result is not None:
            gloss_pairs.append(result)

    # Étape 2 : Réordonner selon la grammaire LSF
    gloss_pairs = _reorder_clause(gloss_pairs, doc)

    # Étape 3 : Marquer les questions
    if _is_question(doc):
        # En LSF, le marqueur interrogatif est à la fin
        # Si un mot interrogatif est présent, le déplacer en fin
        question_items = []
        remaining = []
        for gloss, original in gloss_pairs:
            if original.lower() in QUESTION_WORDS:
                question_items.append((gloss, original))
            else:
                remaining.append((gloss, original))
        gloss_pairs = remaining + question_items
        # Ajouter un marqueur de question si pas de mot interrogatif
        if not question_items:
            gloss_pairs.append(("?", "?"))

    return gloss_pairs


def text_to_gloss_fr(text: str, debug: bool = False) -> dict:
    """
    Point d'entrée principal : convertit un texte français en gloss LSF.

    Args:
        text: Texte français en entrée.
        debug: Si True, affiche les détails de l'analyse spaCy sur stderr.

    Returns:
        dict avec :
            - "glosses": liste des gloss
            - "tokens": liste des mots originaux
            - "gloss_string": chaîne gloss lisible
    """
    if not text or text.strip() == "":
        return {"glosses": [], "tokens": [], "gloss_string": ""}

    spacy_model = load_spacy_model(SPACY_MODELS_FR)

    # Découper en phrases
    doc = spacy_model(text)
    sentences = list(doc.sents)

    all_glosses = []
    all_tokens = []
    clause_strings = []

    for sent in sentences:
        sent_doc = spacy_model(sent.text)
        pairs = _process_sentence(sent_doc, debug=debug)

        glosses = [g for g, _ in pairs]
        tokens = [t for _, t in pairs]

        all_glosses.extend(glosses)
        all_tokens.extend(tokens)

        if glosses:
            clause_strings.append(" ".join(glosses))

    gloss_string = " | ".join(clause_strings)
    if gloss_string:
        gloss_string += " ||"

    return {
        "glosses": all_glosses,
        "tokens": all_tokens,
        "gloss_string": gloss_string,
    }


def text_to_gloss(text: str, language: str = "fr", debug: bool = False, **unused_kwargs) -> list[Gloss]:
    """
    Interface compatible avec le pipeline spoken-to-signed.

    Args:
        text: Texte français à traduire.
        language: Langue source (doit être "fr").
        debug: Mode debug.

    Returns:
        list[Gloss] — liste de phrases, chaque phrase étant une liste de (mot, gloss).
    """
    if language != "fr":
        raise NotImplementedError(f"rules_fr ne supporte que le français, pas '{language}'.")

    result = text_to_gloss_fr(text, debug=debug)
    glosses = result["glosses"]
    tokens = result["tokens"]

    return [list(zip(tokens, glosses))]
