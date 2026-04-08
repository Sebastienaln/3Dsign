"""
Script de génération de dataset pour DÉBATS POLITIQUES FRANÇAIS → gloss LSF via LLM.

Génère un dataset spécialisé dans le langage des débats politiques, les discours,
critiques, réformes, gouvernement, etc. - idéal pour entraîner un modèle de traduction
d'événements politiques.

Usage :
    python generate_dataset_debates.py --api-key "..." --count 5000
    python generate_dataset_debates.py --api-key "..." --count 10000 --output dataset_politics_10k.csv
    python generate_dataset_debates.py --api-key "..." --count 5000 --model gpt-4o
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# CATÉGORIES SPÉCIALES DÉBATS POLITIQUES FRANÇAIS
# ---------------------------------------------------------------------------

CATEGORIES_DEBATES = [
    # --- Discours politiques et rhétorique ---
    {
        "name": "discours_campagne",
        "description": "Discours de campagne électorale, promesses politiques, appels aux électeurs",
        "examples": [
            ("Mes chères concitoyennes, c'est le moment du changement.", "POSS-1 CITOYEN+ MOMENT CHANGEMENT"),
            ("Je promets de réformer le système éducatif.", "IX-1 SYSTÈME ÉDUCATIF RÉFORMER PROMETTRE"),
            ("Votez pour notre projet, votez pour la France.", "POSS-1 PROJET FRANCE VOTER POUR"),
        ],
    },
    {
        "name": "critique_opposition",
        "description": "Discours critiques de l'opposition, remise en question des politiques",
        "examples": [
            ("Vous avez échoué sur l'économie.", "IX-2-PL ÉCONOMIE ÉCHOUER"),
            ("C'est un désastre pour les travailleurs.", "TRAVAILLEUR+ DÉSASTRE"),
            ("Votre gouvernement n'a rien fait.", "POSS-2-PL GOUVERNEMENT RIEN FAIRE PAS"),
        ],
    },
    {
        "name": "defense_gouvernement",
        "description": "Défense des politiques gouvernementales, justification de mesures",
        "examples": [
            ("Nos réformes ont créé 50 000 emplois.", "POSS-1-PL RÉFORME+ 50000 EMPLOI+ CRÉER"),
            ("Le chômage a chuté de trois points.", "CHÔMAGE 3 POINT DIMINUER"),
            ("Nous avons réduit la dette publique.", "IX-1-PL DETTE PUBLIQUE RÉDUIRE"),
        ],
    },
    {
        "name": "debat_chambres",
        "description": "Débats à l'Assemblée nationale ou au Sénat, questions parlementaires",
        "examples": [
            ("Ministère, quelle est votre position sur le climat ?", "MINISTRE CLIMAT POSITION QUI"),
            ("Vous n'avez pas répondu à ma question.", "POSS-2-PL QUESTION RÉPONDRE PAS"),
            ("Le Gouvernement doit expliquer ses choix.", "GOUVERNEMENT CHOIX EXPLIQUER DEVOIR"),
        ],
    },
    {
        "name": "reformes_lois",
        "description": "Discussions sur les réformes, nouvelles lois, budgets, mesures législatives",
        "examples": [
            ("La loi sur la retraite sera votée demain.", "DEMAIN RETRAITE LOI VOTER"),
            ("Nous devons augmenter le SMIC rapidement.", "SMIC AUGMENTER RAPIDEMENT DEVOIR"),
            ("Cette réforme affectera trois millions de Français.", "RÉFORME TROIS MILLION FRANÇAIS AFFECTER"),
        ],
    },
    {
        "name": "elections_suffrage",
        "description": "Sujets liés aux élections, candidats, campagnes, participation électorale",
        "examples": [
            ("Les élections présidentielles sont dans trois mois.", "TROIS MOIS ÉLECTION PRÉSIDENTIEL"),
            ("Le taux de participation a dépassé 75 %.", "75 % PARTICIPATION TAUX DÉPASSER"),
            ("Tous les candidats se sont exprimés sur ce sujet.", "CANDIDAT+ TOUS SUJET EXPRIMER"),
        ],
    },
    {
        "name": "ideologie_valeurs",
        "description": "Débats sur les idéologies, valeurs, principes (gauche/droite, socialisme, libéralisme, etc.)",
        "examples": [
            ("La gauche défend l'égalité et la solidarité.", "GAUCHE ÉGALITÉ SOLIDARITÉ DÉFENDRE"),
            ("La droite privilégie la liberté d'entreprendre.", "DROITE LIBERTÉ ENTREPRENDRE PRIVILÉGIER"),
            ("Nous croyons aux valeurs républicaines.", "POSS-1-PL VALEUR+ RÉPUBLICAIN CROIRE"),
        ],
    },
    {
        "name": "economie_emploi",
        "description": "Politiques économiques, emploi, chômage, croissance, investissements",
        "examples": [
            ("Le chômage chez les jeunes est trop élevé.", "JEUNE+ CHÔMAGE HAUT TROP"),
            ("Nous investissons un milliard dans l'industrie verte.", "INDUSTRIE VERT MILLIARD INVESTIR IX-1-PL"),
            ("La France crépe de croissance.", "FRANCE CROISSANCE MANQUER"),
        ],
    },
    {
        "name": "education_jeunesse",
        "description": "Débats sur l'école, l'éducation, l'accès à la formation, les universités",
        "examples": [
            ("Il faut doubler le budget de l'école républicaine.", "ÉCOLE RÉPUBLICAIN BUDGET DOUBLER DEVOIR"),
            ("L'accès à l'université pour tous reste un objectif.", "UNIVERSITÉ ACCÈS OBJECTIF RESTER"),
            ("La formation professionnelle est sous-financée.", "FORMATION PROFESSIONNEL FINANCER PAS ASSEZ"),
        ],
    },
    {
        "name": "sante_social",
        "description": "Politiques de santé, assurance maladie, système social, prestations familiales",
        "examples": [
            ("Le système de santé français est l'un des meilleurs.", "SYSTÈME SANTÉ FRANÇAIS MEILLEUR PARMI"),
            ("Il faut améliorer l'accès aux soins en zones rurales.", "ZONE RURAL SOIN ACCÈS AMÉLIORER DEVOIR"),
            ("Je m'engage à protéger le service public de santé.", "SANTÉ SERVICE PUBLIC PROTÉGER ENGAGEMENT IX-1"),
        ],
    },
    {
        "name": "environnement_climat",
        "description": "Politique climatique, écologie, transition énergétique, environnement",
        "examples": [
            ("La France s'engage pour la neutralité carbone en 2050.", "FRANCE 2050 CARBONE NEUTRALITÉ ENGAGEMENT"),
            ("Nous interdisons les voitures thermiques d'ici dix ans.", "10 AN VOITURE THERMIQUE INTERDIRE IX-1-PL"),
            ("Le changement climatique est une menace existentielle.", "CLIMAT CHANGEMENT MENACE EXISTENTIEL"),
        ],
    },
    {
        "name": "immigration_integration",
        "description": "Débats sur l'immigration, l'intégration, l'asile, les migrants",
        "examples": [
            ("Notre politique d'immigration doit être claire et humaine.", "IMMIGRATION POLITIQUE CLAIR HUMAIN DEVOIR"),
            ("L'intégration des migrants passe par la langue et l'emploi.", "MIGRANT+ INTÉGRATION LANGUE EMPLOI PASSER"),
            ("Nous accueillons les réfugiés selon nos capacités.", "RÉFUGIÉ+ ACCUEILLIR CAPACITÉ SELON POSS-1-PL"),
        ],
    },
    {
        "name": "securite_justice",
        "description": "Sécurité, police, justice, criminalité, droits des citoyens",
        "examples": [
            ("Renforcer la présence policière dans les quartiers sensibles.", "QUARTIER SENSIBLE POLICE PRÉSENCE RENFORCER"),
            ("La justice doit être plus rapide et plus efficace.", "JUSTICE RAPIDE EFFICACE PLUS DEVOIR"),
            ("Nous investissons dans la prévention de la criminalité.", "CRIMINALITÉ PRÉVENTION INVESTIR IX-1-PL"),
        ],
    },
    {
        "name": "relations_internationales",
        "description": "Politique étrangère, relations avec l'UE, alliances internationales, diplomatie",
        "examples": [
            ("La France reste engagée dans l'UE et l'OTAN.", "FRANCE UE OTAN ENGAGÉ RESTER"),
            ("Nous condamnons cette violation des droits de l'homme.", "HOMME DROIT VIOLATION IX-3 CONDAMNER IX-1-PL"),
            ("La diplomatie française joue un rôle majeur en Europe.", "EUROPE RÔLE MAJEUR JOUER DIPLOMATIE FRANÇAIS"),
        ],
    },
    {
        "name": "territoire_decentralisation",
        "description": "Débats sur la décentralisation, les régions, les collectivités, l'autonomie territoriale",
        "examples": [
            ("Les régions doivent avoir plus d'autonomie budgétaire.", "RÉGION+ BUDGET AUTONOMIE PLUS DEVOIR"),
            ("La décentralisation crée des inégalités entre territoires.", "INÉGALITÉ+ TERRITOIRE+ DÉCENTRALISATION CRÉER"),
            ("Paris concentre trop de pouvoir politique.", "POUVOIR POLITIQUE PARIS CONCENTRER TROP"),
        ],
    },
    {
        "name": "corruption_transparence",
        "description": "Lutte contre la corruption, transparence politique, conflit d'intérêts, affaires politiques",
        "examples": [
            ("Il faut renforcer la transparence des financements politiques.", "FINANCING POLITIQUE TRANSPARENCE RENFORCER DEVOIR"),
            ("Cette affaire révèle les dérives du pouvoir.", "POUVOIR DÉRIVE AFFAIRE IX-3 RÉVÉLER"),
            ("Zéro tolérance pour la corruption.", "CORRUPTION TOLÉRANCE ZÉRO"),
        ],
    },
    {
        "name": "budget_finances_publiques",
        "description": "Budget national, fiscalité, impôts, dépenses publiques, austérité",
        "examples": [
            ("Nous baissons les impôts pour les classes moyennes.", "CLASSE MOYEN IMPÔT BAISSER IX-1-PL"),
            ("Le budget doit priorité à l'investissement public.", "INVESTISSEMENT PUBLIC BUDGET PRIORITÉ DEVOIR"),
            ("La dette publique atteint 120 % du PIB.", "PIB % 120 ATTEINDRE DETTE PUBLIC"),
        ],
    },
    {
        "name": "logement_urbanisme",
        "description": "Politique du logement, urbanisme, habitat, crise du logement",
        "examples": [
            ("Il faut construire 500 000 logements neufs.", "500000 LOGEMENT NOUVEAU CONSTRUIRE DEVOIR"),
            ("Le prix des loyers explose dans les grandes villes.", "VILLE GRAND LOYER PRIX EXPLOSER"),
            ("L'accès au logement est un droit fondamental.", "LOGEMENT ACCÈS DROIT FONDAMENTAL"),
        ],
    },
    {
        "name": "culture_patrimoine",
        "description": "Culture, patrimoine, arts, cinéma, soutien aux artistes, identité culturelle",
        "examples": [
            ("La culture française rayonne dans le monde.", "MONDE CULTURE FRANÇAIS RAYONNER"),
            ("Nous augmentons le budget du cinéma français.", "CINÉMA FRANÇAIS BUDGET AUGMENTER IX-1-PL"),
            ("Le patrimoine doit être préservé pour les générations futures.", "GÉNÉRATION FUTUR PATRIMOINE PRÉSERVER DEVOIR"),
        ],
    },
    {
        "name": "technologie_innovation",
        "description": "Politique de l'innovation, numérique, startups, 5G, infrastructures technologiques",
        "examples": [
            ("La France investit dans l'IA et les technologies vertes.", "TECHNOLOGIE VERT IA INVESTIR FRANCE"),
            ("Le haut débit doit être accessible à tous les Français.", "FRANÇAIS TOUS DÉBIT HAUT ACCÈS DEVOIR"),
            ("Les startups sont le moteur de notre économie.", "STARTUP ÉCONOMIE POSS-1-PL MOTEUR"),
        ],
    },
    # --- Langage spécifique à la politique ---
    {
        "name": "affirmations_engagement",
        "description": "Engagements politiques, promesses formelles, serments de campagne",
        "examples": [
            ("Je m'engage solennellement à tenir mes promesses.", "PROMESSE POSS-1 TENIR ENGAGEMENT SOLENNEL IX-1"),
            ("C'est ma priorité absolue pour les cinq années à venir.", "PRIORITÉ ABSOLUE POSS-1 ANNÉE CINQ À-VENIR"),
            ("Je serai le président/la présidente de tous les Français.", "FRANÇAIS TOUS PRÉSIDENT IX-1 ÊTRE"),
        ],
    },
    {
        "name": "critiques_virales",
        "description": "Critiques acérées, attaques politiques, sarcasme politique, déception",
        "examples": [
            ("Vous mentez à la France.", "IX-2-PL FRANCE MENTIR"),
            ("Il Y a un gouffre entre vos promesses et la réalité.", "PROMESSE POSS-2-PL RÉALITÉ GOUFFRE"),
            ("C'est de la poudre aux yeux !", "YEUX POUDRE"),
        ],
    },
    {
        "name": "appels_mobilisation",
        "description": "Appels à mobilisation, participation citoyenne, marches, manifestations",
        "examples": [
            ("Levons-nous contre cette injustice !", "INJUSTICE IX-1-PL LEVER"),
            ("Tous dehors pour défendre nos droits !", "DROIT+ POSS-1-PL DÉFENDRE DEHORS TOUS"),
            ("C'est le moment de s'unir.", "MOMENT S'UNIR"),
        ],
    },
    {
        "name": "recontexte_historique",
        "description": "Références historiques, rappel du passé, leçons de l'histoire",
        "examples": [
            ("Comme l'a dit de Gaulle, la France doit rester indépendante.", "INDÉPENDANT FRANCE RESTER GAULLE DIT"),
            ("Les années 1930 nous ont montré les dangers.", "DANGER MONTRER ANNÉE 1930"),
            ("L'histoire ne doit pas se répéter.", "RÉPÉTER NE HISTOIRE"),
        ],
    },
    {
        "name": "statistiques_chiffres",
        "description": "Présentation de chiffres, statistiques, données économiques, sondages politiques",
        "examples": [
            ("Selon un sondage récent, 68 % des Français approuvent.", "68 % FRANÇAIS APPROUVER SONDAGE RÉCENT SELON"),
            ("Le PIB a augmenté de 1,5 % cette année.", "ANNÉE CETTE PIB % 1.5 AUGMENTER"),
            ("Trois millions de Français vivent sous le seuil de pauvreté.", "PAUVRETÉ SEUIL MILLION TROIS FRANÇAIS VIVRE"),
        ],
    },
    {
        "name": "positions_ideologiques_comparaison",
        "description": "Comparaison des positions entre partis, distinction entre gauche/droite, centristes",
        "examples": [
            ("La gauche socialiste propose une autre vision.", "VISION AUTRE PROPOSER SOCIALISTE GAUCHE"),
            ("Les Verts ont des propositions radicales.", "PROPOSITION RADICAL VERT+ AVOIR"),
            ("Le centre refuse les extrêmes.", "EXTRÊME REFUSER CENTRE"),
        ],
    },
    {
        "name": "acceptation_desaccord",
        "description": "Expression du désaccord politique, points de divergence, débats",
        "examples": [
            ("Nous voyons les choses différemment.", "CHOSE DIFFÉRENT VOIR IX-1-PL"),
            ("Je respecte votre avis, mais je persiste.", "AVIS POSS-2-PL RESPECTER PERSISTER"),
            ("Il y a un profond désaccord entre nous.", "NOUS DÉSACCORD PROFOND"),
        ],
    },
    {
        "name": "resultats_elections_sondages",
        "description": "Annonce de résultats électoraux, interprétation des sondages, analyses post-vote",
        "examples": [
            ("Notre coalition a remporté 350 sièges.", "SIÈGE 350 REMPORTER COALITION POSS-1-PL"),
            ("L'abstention franchit la barre des 50 %.", "50 % BARRE FRANCHIR ABSTENTION"),
            ("Les résultats montrent une forte poussée de la droite.", "DROITE POUSSÉE FORT MONTRER RÉSULTAT"),
        ],
    },
]

# --- Thèmes et styles supplémentaires pour les débats ---
VOCABULARY_THEMES_DEBATES = [
    "politique générale", "élections et campagnes", "économie et finances",
    "emploi et travail", "santé et sécurité sociale", "éducation",
    "environnement et écologie", "immigration et intégration", "sécurité",
    "justice et droits", "relations internationales", "Europe et UE",
    "culture et patrimoine", "technologie et innovation", "décentralisation",
    "corruption et transparence", "budget et fiscalité", "logement"
]

REGISTER_STYLES_DEBATES = [
    "discours formel de campagne",
    "débat parlementaire académique",
    "critique passionnée de l'opposition",
    "défense des politiques gouvernementales",
    "appel à mobilisation citoyenne",
    "analyse politico-médiatique",
    "langage de politicien rhétorique",
    "ton de sondeur/analyste politique",
]

SYSTEM_PROMPT = """Tu es un expert en politique française et en Langue des Signes Française (LSF).
Tu génères des paires de traduction du français politique vers la notation gloss LSF.

Le contexte: il s'agit de débats politiques français, discours, réformes, politiques gouvernementales,
critiques de l'opposition, campagnes électorales, etc.

RÈGLES STRICTES à respecter :
1. SUPPRIMER les articles : le, la, les, l', un, une, des, du, de, d', au, aux
2. SUPPRIMER les prépositions non-significatives : à, de, dans, sur, sous, avec, pour, par, en, chez, entre
3. SUPPRIMER les auxiliaires : être, avoir, aller (quand ils sont auxiliaires)
4. PRONOMS personnels → indexation :
   - je/me/moi → IX-1
   - tu/te/toi → IX-2
   - il/elle/lui → IX-3
   - nous/on → IX-1-PL
   - vous → IX-2-PL
   - ils/elles → IX-3-PL
5. POSSESSIFS → POSS :
   - mon/ma/mes → POSS-1
   - ton/ta/tes → POSS-2
   - son/sa/ses → POSS-3
   - notre/nos → POSS-1-PL
   - votre/vos → POSS-2-PL
   - leur/leurs → POSS-3-PL
6. ORDRE des mots : TEMPS → LIEU → SUJET → OBJET → ADJECTIF → VERBE → NÉGATION
7. PLURIEL : marquer les noms pluriels avec + (ex: ENFANT+)
8. VERBES : utiliser l'infinitif en MAJUSCULES
9. TOUT en MAJUSCULES
10. QUESTIONS : mot interrogatif (OÙ, QUAND, COMMENT, POURQUOI, COMBIEN, QUI, QUOI) à la FIN
11. Questions oui/non : ajouter ? à la FIN
12. NÉGATION (PAS, PLUS, JAMAIS, RIEN, PERSONNE, AUCUN) : à la FIN de la clause

CONSTRUCTIONS SPÉCIALES :
13. VERBES PRONOMINAUX (se laver, s'habiller, s'engager) : Supprimer le pronom réfléchi, garder l'infinitif.
14. IL IMPERSONNEL (il faut, il semble, il est possible) : Supprimer "il" impersonnel.
15. "IL Y A" existentiel → NOM NOMBRE AVOIR.
16. PASSÉ RÉCENT "venir de + infinitif" → VERBE FINIR (aspect accompli).
17. "DEPUIS" temporel → conserver DEPUIS en gloss.
18. Expressions idiomatiques AVOIR : avoir raison → RAISON AVOIR, avoir tort → TORT AVOIR
19. "Ça fait longtemps" → LONGTEMPS (pas "LONG")
20. MANQUER : "Tu me manques" → IX-2 IX-1 MANQUER

VOCABULAIRE POLITIQUE EN MAJUSCULES (orthographe stricte) :
GOUVERNEMENT, PRÉSIDENT, MINISTRE, DÉPUTÉ, SÉNATEUR, ASSEMBLÉE, CHAMBRE, SÉNAT,
ÉLECTION, VOTE, CAMPAGNE, RÉFORME, LOI, DÉCRET, POLITIQ, PROMESSE, ENGAGEMENT,
GAUCHE, DROITE, CENTRE, SOCIALISTE, LIBÉRAL, CONSERVATEUR, ÉCOLOGISTE, INDÉPENDANT,
ÉCONOMIE, CHÔMAGE, IMPÔT, BUDGET, DETTE, CROISSANCE, EMPLOI, SMIC, SALAIRE,
ÉCOLE, UNIVERSITÉ, ÉDUCATION, SANTÉ, HÔPITAL, SOCIAL, RETRAITE, PENSION,
ENVIRONNEMENT, CLIMAT, POLLUTION, ÉNERGIE, TRANSITION, DURABILITÉ,
IMMIGRATION, RÉFUGIÉ, INTÉGRATION, ASILE, CITOYEN, FRANÇAIS,
JUSTICE, POLICE, SÉCURITÉ, CRIMINALITÉ, PRISON, DROIT, CONSTITUTION,
DIPLOMATIE, FRANCE, EUROPE, UE, INTERNATIONAL, ALLIE, ENNEMI, CONFLIT

FORMAT DE SORTIE : Un objet JSON, rien d'autre.
{"pairs": [{"fr": "phrase en français politique", "gloss": "GLOSS EN LSF"}, ...]}

IMPORTANT :
- Génère des phrases AUTHENTIQUES et NATURELLES du débat politique français
- Utilise un vocabulaire RICHE des politiques et débats
- Varie les types de phrases : promesses, critiques, défenses, appels, statistiques
- Ne répète PAS les mêmes phrases, sois TRÈS créatif
- Les gloss doivent être des traductions fidèles selon les règles LSF
"""

# ---------------------------------------------------------------------------
# Fonctions d'aide (identiques à generate_dataset.py)
# ---------------------------------------------------------------------------

def build_user_prompt(category: dict, count: int, existing_phrases: set) -> str:
    """Construit le prompt utilisateur avec des variateurs aléatoires."""
    examples_text = "\n".join(
        f'  fr: "{ex[0]}" → gloss: "{ex[1]}"' for ex in category["examples"]
    )

    theme = random.choice(VOCABULARY_THEMES_DEBATES)
    style = random.choice(REGISTER_STYLES_DEBATES)

    avoid_text = ""
    if existing_phrases:
        sample = random.sample(list(existing_phrases), min(15, len(existing_phrases)))
        avoid_text = f"\n\nNe génère PAS ces phrases (déjà dans le dataset) :\n" + "\n".join(f'- "{p}"' for p in sample)

    return f"""Génère exactement {count} paires de traduction français → gloss LSF basées sur des débats politiques.

Catégorie : {category['description']}
Thème de vocabulaire à privilégier : {theme}
Style de langage : {style}

Exemples de référence :
{examples_text}
{avoid_text}

Sois TRÈS créatif et authentique. Ces phrases doivent venir d'un vrai débat politique français.
Réponds UNIQUEMENT avec un JSON valide : {{"pairs": [{{"fr": "...", "gloss": "..."}}, ...]}}"""


def call_llm(api_key: str, model: str, system_prompt: str, user_prompt: str, base_url: str = None) -> str:
    """Appelle l'API OpenAI (ou compatible) et retourne la réponse."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Erreur : installe openai avec : pip install openai")
        sys.exit(1)

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        max_tokens=8192,
    )

    return response.choices[0].message.content


def parse_llm_response(response_text: str) -> list[dict]:
    """Parse la réponse JSON du LLM."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "pairs" in data:
            return data["pairs"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
                if isinstance(data, dict) and "pairs" in data:
                    return data["pairs"]
            except json.JSONDecodeError:
                pass

    print(f"  ERREUR : impossible de parser la réponse JSON")
    return []


def _save_csv(pairs: list[dict], output_file: str):
    """Sauvegarde incrémentale du dataset."""
    fieldnames = ["fr", "gloss", "category"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            row = {k: pair.get(k, "") for k in fieldnames}
            writer.writerow(row)


def _load_existing_csv(path: str) -> tuple[list[dict], set]:
    """Charge un CSV existant pour reprendre la génération."""
    pairs = []
    phrases = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("fr") and row.get("gloss"):
                    pairs.append(dict(row))
                    phrases.add(row["fr"])
    return pairs, phrases


def generate_dataset_debates(
    api_key: str,
    model: str,
    total_count: int,
    output_file: str,
    base_url: str = None,
):
    """Génère le dataset de débats politiques."""
    all_pairs, existing_phrases = _load_existing_csv(output_file)
    if all_pairs:
        print(f"Reprise : {len(all_pairs)} paires existantes chargées depuis {output_file}")

    remaining = total_count - len(all_pairs)
    if remaining <= 0:
        print(f"Dataset déjà complet ({len(all_pairs)} paires). Rien à faire.")
        return

    per_category = max(5, remaining // len(CATEGORIES_DEBATES))
    remainder = remaining - per_category * len(CATEGORIES_DEBATES)

    print(f"=== Génération de {total_count} paires DÉBATS POLITIQUES → gloss LSF ===")
    print(f"Modèle : {model}")
    print(f"Catégories politiques : {len(CATEGORIES_DEBATES)}")
    print(f"~{per_category} phrases par catégorie (+ {remainder} bonus)")
    print()

    for i, category in enumerate(CATEGORIES_DEBATES):
        count = per_category + (1 if i < remainder else 0)
        batch_size = 50
        generated_for_cat = 0
        low_yield_count = 0

        print(f"[{i+1}/{len(CATEGORIES_DEBATES)}] {category['name']} ({count} phrases)...")

        while generated_for_cat < count:
            batch_count = min(batch_size, count - generated_for_cat)
            prompt = build_user_prompt(category, batch_count, existing_phrases)

            try:
                response = call_llm(api_key, model, SYSTEM_PROMPT, prompt, base_url)
                pairs = parse_llm_response(response)

                added = 0
                for pair in pairs:
                    if "fr" in pair and "gloss" in pair and pair["fr"] not in existing_phrases:
                        pair["category"] = category["name"]
                        all_pairs.append(pair)
                        existing_phrases.add(pair["fr"])
                        added += 1
                        generated_for_cat += 1

                print(f"  lot +{added} (total catégorie: {generated_for_cat}/{count}, total: {len(all_pairs)})", flush=True)

                _save_csv(all_pairs, output_file)

                if added < 10:
                    low_yield_count += 1
                    if low_yield_count >= 3:
                        print(f"  → Rendement faible, passage à la catégorie suivante")
                        break
                else:
                    low_yield_count = 0

            except Exception as e:
                print(f"  ERREUR : {e}")
                low_yield_count += 1
                if low_yield_count >= 3:
                    print(f"  → Trop d'erreurs, passage à la catégorie suivante")
                    break

            time.sleep(1)

    print(f"\nTotal : {len(all_pairs)} paires générées")
    _save_csv(all_pairs, output_file)
    print(f"Terminé ! {len(all_pairs)} paires sauvegardées dans {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Génère un dataset DÉBATS POLITIQUES français → gloss LSF",
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="Clé API OpenAI (ou variable d'env OPENAI_API_KEY)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="URL de base pour une API compatible OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Modèle à utiliser (défaut: gpt-4o-mini)")
    parser.add_argument("--count", type=int, default=5000,
                        help="Nombre de paires à générer (défaut: 5000)")
    parser.add_argument("--output", type=str, default="dataset_politique.csv",
                        help="Fichier CSV de sortie (défaut: dataset_politique.csv)")
    args = parser.parse_args()

    if not args.api_key:
        print("Erreur : fournis une clé API avec --api-key ou la variable d'env OPENAI_API_KEY")
        sys.exit(1)

    generate_dataset_debates(
        api_key=args.api_key,
        model=args.model,
        total_count=args.count,
        output_file=args.output,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
