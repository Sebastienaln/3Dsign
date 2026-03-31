"""
Script de génération de dataset français → gloss LSF via un LLM (API OpenAI-compatible).

Usage :
    python generate_dataset.py --api-key "..." --count 20000
    python generate_dataset.py --api-key "..." --count 20000 --output dataset_20k.csv

Le script :
  1. Envoie des prompts variés au LLM pour générer des paires fr/gloss
  2. Utilise de nombreuses catégories thématiques et grammaticales pour maximiser la diversité
  3. Sauvegarde incrémentalement dans un CSV (reprend si interrompu)
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
# Catégories thématiques et grammaticales (diversité maximale)
# ---------------------------------------------------------------------------

CATEGORIES = [
    # --- Structures grammaticales de base ---
    {
        "name": "svo_simple",
        "description": "Phrases simples sujet-verbe-objet, vocabulaire varié (animaux, objets, aliments, personnes)",
        "examples": [
            ("Je mange une pomme.", "IX-1 POMME MANGER"),
            ("Tu regardes la télévision.", "IX-2 TÉLÉVISION REGARDER"),
            ("Elle lit un livre.", "IX-3 LIVRE LIRE"),
        ],
    },
    {
        "name": "negation_pas",
        "description": "Phrases négatives avec ne...pas",
        "examples": [
            ("Je ne veux pas partir.", "IX-1 PARTIR VOULOIR PAS"),
            ("Tu ne comprends pas la question.", "IX-2 QUESTION COMPRENDRE PAS"),
            ("Elle ne dort pas encore.", "IX-3 ENCORE DORMIR PAS"),
        ],
    },
    {
        "name": "negation_autres",
        "description": "Phrases négatives avec ne...plus, ne...jamais, ne...rien, ne...personne, ne...aucun",
        "examples": [
            ("Il ne mange jamais de viande.", "IX-3 VIANDE MANGER JAMAIS"),
            ("Nous ne travaillons plus ici.", "IX-1-PL ICI TRAVAILLER PLUS"),
            ("Tu ne vois rien dans le noir.", "IX-2 NOIR VOIR RIEN"),
        ],
    },
    {
        "name": "questions_ouvertes",
        "description": "Questions ouvertes avec où, quand, comment, pourquoi, combien",
        "examples": [
            ("Où est la gare ?", "GARE OÙ"),
            ("Comment tu t'appelles ?", "IX-2 APPELER COMMENT"),
            ("Pourquoi tu pleures ?", "IX-2 PLEURER POURQUOI"),
        ],
    },
    {
        "name": "questions_fermees",
        "description": "Questions fermées (oui/non), avec est-ce que, inversion sujet-verbe",
        "examples": [
            ("Est-ce que tu viens demain ?", "DEMAIN IX-2 VENIR ?"),
            ("Tu as faim ?", "IX-2 FAIM ?"),
            ("Vous parlez français ?", "IX-2-PL FRANÇAIS PARLER ?"),
        ],
    },
    {
        "name": "questions_qui_que",
        "description": "Questions avec qui, que, quoi, quel",
        "examples": [
            ("Qui a pris mon stylo ?", "POSS-1 STYLO PRENDRE QUI"),
            ("Que fais-tu ce soir ?", "SOIR IX-2 FAIRE QUOI"),
            ("Quel est ton sport préféré ?", "POSS-2 SPORT PRÉFÉRÉ QUOI"),
        ],
    },
    {
        "name": "temps_passe",
        "description": "Phrases au passé avec marqueurs temporels (hier, la semaine dernière, il y a...)",
        "examples": [
            ("Hier, j'ai mangé au restaurant.", "HIER IX-1 RESTAURANT MANGER"),
            ("La semaine dernière, il a plu.", "SEMAINE DERNIÈRE PLUIE"),
            ("Il y a deux ans, nous avons déménagé.", "DEUX AN AVANT IX-1-PL DÉMÉNAGER"),
        ],
    },
    {
        "name": "temps_futur",
        "description": "Phrases au futur avec marqueurs temporels (demain, bientôt, la semaine prochaine...)",
        "examples": [
            ("Demain nous partirons en vacances.", "DEMAIN IX-1-PL VACANCES PARTIR"),
            ("Bientôt tu auras ton diplôme.", "BIENTÔT IX-2 DIPLÔME OBTENIR"),
            ("Le mois prochain, elle déménage.", "MOIS PROCHAIN IX-3 DÉMÉNAGER"),
        ],
    },
    {
        "name": "temps_present_habitude",
        "description": "Phrases au présent d'habitude (toujours, souvent, parfois, tous les jours...)",
        "examples": [
            ("Maintenant je travaille.", "MAINTENANT IX-1 TRAVAILLER"),
            ("Il court tous les matins.", "MATIN TOUS IX-3 COURIR"),
            ("Parfois elle chante sous la douche.", "PARFOIS IX-3 DOUCHE CHANTER"),
        ],
    },
    {
        "name": "lieux_villes",
        "description": "Phrases avec des villes, pays, régions",
        "examples": [
            ("Mon frère habite à Paris.", "PARIS POSS-1 FRÈRE HABITER"),
            ("Nous allons en Espagne cet été.", "ÉTÉ IX-1-PL ESPAGNE ALLER"),
            ("Elle vient du Japon.", "IX-3 JAPON VENIR"),
        ],
    },
    {
        "name": "lieux_batiments",
        "description": "Phrases avec des bâtiments et endroits (école, hôpital, magasin, gare, aéroport...)",
        "examples": [
            ("Les enfants jouent dans le jardin.", "ENFANT+ JARDIN JOUER"),
            ("Elle travaille à l'hôpital.", "HÔPITAL IX-3 TRAVAILLER"),
            ("Je vais à la bibliothèque.", "IX-1 BIBLIOTHÈQUE ALLER"),
        ],
    },
    {
        "name": "possessifs",
        "description": "Phrases avec des possessifs (mon, ton, son, notre, votre, leur)",
        "examples": [
            ("Mon chat dort sur le canapé.", "POSS-1 CHAT CANAPÉ DORMIR"),
            ("Ton père est gentil.", "POSS-2 PÈRE GENTIL"),
            ("Leur maison est grande.", "POSS-3-PL MAISON GRAND"),
        ],
    },
    {
        "name": "pluriels",
        "description": "Phrases avec des noms au pluriel (enfants, amis, livres, voitures...)",
        "examples": [
            ("Les enfants jouent dehors.", "ENFANT+ DEHORS JOUER"),
            ("Mes amis arrivent demain.", "DEMAIN POSS-1 AMI+ ARRIVER"),
            ("Les chiens courent dans le parc.", "CHIEN+ PARC COURIR"),
        ],
    },
    {
        "name": "adjectifs_descriptifs",
        "description": "Phrases avec des adjectifs de description physique ou morale",
        "examples": [
            ("La grande maison est belle.", "MAISON GRAND BEAU"),
            ("Le petit chat noir dort.", "CHAT PETIT NOIR DORMIR"),
            ("Mon voisin est très gentil.", "POSS-1 VOISIN TRÈS GENTIL"),
        ],
    },
    {
        "name": "adjectifs_etat_emotion",
        "description": "Phrases décrivant des états ou émotions avec adjectifs",
        "examples": [
            ("Je suis fatigué.", "IX-1 FATIGUÉ"),
            ("Elle est contente de te voir.", "IX-3 IX-2 VOIR CONTENT"),
            ("Les élèves sont stressés.", "ÉLÈVE+ STRESSÉ"),
        ],
    },
    {
        "name": "phrases_complexes_et",
        "description": "Phrases avec coordination (et, aussi, également)",
        "examples": [
            ("Je mange une pomme et tu bois du café.", "IX-1 POMME MANGER IX-2 CAFÉ BOIRE"),
            ("Il travaille et elle étudie.", "IX-3 TRAVAILLER IX-3 ÉTUDIER"),
            ("Tu chantes et danses bien.", "IX-2 CHANTER DANSER BIEN"),
        ],
    },
    {
        "name": "phrases_complexes_mais",
        "description": "Phrases avec opposition (mais, cependant, pourtant)",
        "examples": [
            ("Il pleut mais je sors quand même.", "PLUIE MAIS IX-1 SORTIR QUAND-MÊME"),
            ("Je suis fatigué mais je continue.", "IX-1 FATIGUÉ MAIS IX-1 CONTINUER"),
            ("Elle est petite mais très forte.", "IX-3 PETIT MAIS TRÈS FORT"),
        ],
    },
    {
        "name": "phrases_complexes_si_condition",
        "description": "Phrases conditionnelles avec si, quand, lorsque",
        "examples": [
            ("Si tu veux, on peut aller au cinéma.", "IX-2 VOULOIR SI IX-1-PL CINÉMA ALLER POUVOIR"),
            ("Quand il fait beau, je me promène.", "BEAU TEMPS QUAND IX-1 PROMENER"),
            ("Si je gagne, je t'invite au restaurant.", "IX-1 GAGNER SI IX-2 RESTAURANT INVITER"),
        ],
    },
    # --- Thèmes de la vie quotidienne ---
    {
        "name": "alimentation",
        "description": "Phrases sur la nourriture, les repas, la cuisine (fruits, légumes, plats, boissons, recettes)",
        "examples": [
            ("Je bois un café chaque matin.", "MATIN CHAQUE IX-1 CAFÉ BOIRE"),
            ("Elle prépare une soupe de légumes.", "IX-3 LÉGUME+ SOUPE PRÉPARER"),
            ("Tu veux du pain avec du fromage ?", "IX-2 PAIN FROMAGE VOULOIR ?"),
        ],
    },
    {
        "name": "transport",
        "description": "Phrases sur les transports (bus, train, voiture, vélo, avion, métro, tram)",
        "examples": [
            ("Je prends le bus tous les matins.", "MATIN TOUS IX-1 BUS PRENDRE"),
            ("Le train arrive à 8 heures.", "8 HEURE TRAIN ARRIVER"),
            ("Elle va au travail en vélo.", "IX-3 TRAVAIL VÉLO ALLER"),
        ],
    },
    {
        "name": "sante",
        "description": "Phrases sur la santé (douleur, médecin, maladie, médicaments, hôpital)",
        "examples": [
            ("Elle a mal à la tête.", "IX-3 TÊTE DOULEUR"),
            ("Je dois aller chez le médecin.", "IX-1 MÉDECIN ALLER DEVOIR"),
            ("Il prend des médicaments tous les jours.", "JOUR TOUS IX-3 MÉDICAMENT+ PRENDRE"),
        ],
    },
    {
        "name": "education",
        "description": "Phrases sur l'école, les études, les cours, les examens, les devoirs",
        "examples": [
            ("Les élèves passent un examen demain.", "DEMAIN ÉLÈVE+ EXAMEN PASSER"),
            ("Je fais mes devoirs le soir.", "SOIR IX-1 POSS-1 DEVOIR+ FAIRE"),
            ("Le professeur explique la leçon.", "PROFESSEUR LEÇON EXPLIQUER"),
        ],
    },
    {
        "name": "travail",
        "description": "Phrases sur le travail, les métiers, le bureau, les collègues, les réunions",
        "examples": [
            ("Il a une réunion à 14 heures.", "14 HEURE IX-3 RÉUNION AVOIR"),
            ("Ma collègue est en vacances.", "POSS-1 COLLÈGUE VACANCES"),
            ("Tu cherches un nouveau travail ?", "IX-2 NOUVEAU TRAVAIL CHERCHER ?"),
        ],
    },
    {
        "name": "famille",
        "description": "Phrases sur la famille (parents, enfants, frères, sœurs, grands-parents, oncles, tantes)",
        "examples": [
            ("Ma grand-mère a 80 ans.", "POSS-1 GRAND-MÈRE 80 AN"),
            ("Son frère est plus grand que lui.", "POSS-3 FRÈRE IX-3 PLUS GRAND"),
            ("Nous rendons visite à nos cousins.", "IX-1-PL POSS-1-PL COUSIN+ VISITER"),
        ],
    },
    {
        "name": "loisirs_sport",
        "description": "Phrases sur les loisirs et le sport (football, natation, musique, cinéma, jeux vidéo, lecture)",
        "examples": [
            ("Il joue au football le samedi.", "SAMEDI IX-3 FOOTBALL JOUER"),
            ("Tu aimes la musique classique ?", "IX-2 MUSIQUE CLASSIQUE AIMER ?"),
            ("Nous allons au cinéma ce soir.", "SOIR IX-1-PL CINÉMA ALLER"),
        ],
    },
    {
        "name": "meteo_nature",
        "description": "Phrases sur la météo et la nature (pluie, soleil, neige, vent, froid, chaud, saisons)",
        "examples": [
            ("Aujourd'hui il fait très chaud.", "AUJOURD'HUI TRÈS CHAUD"),
            ("Il neige depuis ce matin.", "MATIN NEIGE"),
            ("Au printemps, les fleurs poussent.", "PRINTEMPS FLEUR+ POUSSER"),
        ],
    },
    {
        "name": "courses_achats",
        "description": "Phrases sur les achats, le shopping, les prix, les magasins, le marché",
        "examples": [
            ("Nous faisons les courses au supermarché.", "IX-1-PL SUPERMARCHÉ COURSE+ FAIRE"),
            ("Ce pantalon coûte 30 euros.", "PANTALON 30 EURO COÛTER"),
            ("Tu as acheté quoi au marché ?", "IX-2 MARCHÉ ACHETER QUOI"),
        ],
    },
    {
        "name": "maison_logement",
        "description": "Phrases sur la maison, l'appartement, les pièces, les meubles, le ménage",
        "examples": [
            ("Je nettoie la cuisine.", "IX-1 CUISINE NETTOYER"),
            ("Notre appartement a trois chambres.", "POSS-1-PL APPARTEMENT CHAMBRE TROIS"),
            ("Le jardin est derrière la maison.", "JARDIN MAISON DERRIÈRE"),
        ],
    },
    {
        "name": "animaux",
        "description": "Phrases sur les animaux domestiques et sauvages",
        "examples": [
            ("Mon chien court dans le parc.", "POSS-1 CHIEN PARC COURIR"),
            ("Les oiseaux chantent le matin.", "MATIN OISEAU+ CHANTER"),
            ("Elle a peur des araignées.", "IX-3 ARAIGNÉE+ PEUR"),
        ],
    },
    {
        "name": "technologie",
        "description": "Phrases sur la technologie, les ordinateurs, les téléphones, Internet",
        "examples": [
            ("Je regarde une vidéo sur mon téléphone.", "IX-1 POSS-1 TÉLÉPHONE VIDÉO REGARDER"),
            ("L'ordinateur est en panne.", "ORDINATEUR PANNE"),
            ("Tu as reçu mon message ?", "IX-2 POSS-1 MESSAGE RECEVOIR ?"),
        ],
    },
    {
        "name": "voyages",
        "description": "Phrases sur les voyages, les vacances, l'hôtel, le tourisme",
        "examples": [
            ("Nous partons en vacances lundi.", "LUNDI IX-1-PL VACANCES PARTIR"),
            ("L'hôtel est près de la plage.", "HÔTEL PLAGE PRÈS"),
            ("Tu as déjà visité le Maroc ?", "IX-2 DÉJÀ MAROC VISITER ?"),
        ],
    },
    {
        "name": "sentiments_emotions",
        "description": "Phrases exprimant des sentiments et émotions (joie, tristesse, colère, surprise, peur, amour)",
        "examples": [
            ("Je suis très content aujourd'hui.", "AUJOURD'HUI IX-1 TRÈS CONTENT"),
            ("Il est triste parce que son ami est parti.", "POSS-3 AMI PARTIR IX-3 TRISTE"),
            ("Tu me manques beaucoup.", "IX-2 IX-1 MANQUER BEAUCOUP"),
        ],
    },
    {
        "name": "verbes_modaux",
        "description": "Phrases avec verbes modaux (pouvoir, devoir, vouloir, savoir, falloir)",
        "examples": [
            ("Je dois partir maintenant.", "MAINTENANT IX-1 PARTIR DEVOIR"),
            ("Tu peux m'aider ?", "IX-2 IX-1 AIDER POUVOIR ?"),
            ("Il faut faire attention.", "ATTENTION FAIRE FALLOIR"),
        ],
    },
    {
        "name": "comparaisons",
        "description": "Phrases comparatives (plus que, moins que, aussi que, le plus, le moins)",
        "examples": [
            ("Marie est plus grande que Pierre.", "MARIE PIERRE PLUS GRAND"),
            ("Ce film est moins intéressant que le livre.", "FILM LIVRE MOINS INTÉRESSANT"),
            ("C'est le meilleur restaurant de la ville.", "RESTAURANT VILLE MEILLEUR"),
        ],
    },
    {
        "name": "ordres_imperatif",
        "description": "Phrases à l'impératif, ordres, demandes, invitations",
        "examples": [
            ("Ferme la porte, s'il te plaît.", "PORTE FERMER S'IL-TE-PLAÎT"),
            ("Viens ici tout de suite !", "ICI VENIR TOUT-DE-SUITE"),
            ("Regarde ce beau paysage !", "PAYSAGE BEAU REGARDER"),
        ],
    },
    {
        "name": "descriptions_personnes",
        "description": "Descriptions physiques de personnes (taille, couleur cheveux, âge, vêtements)",
        "examples": [
            ("Mon père a les cheveux gris.", "POSS-1 PÈRE CHEVEUX GRIS"),
            ("La fille porte une robe rouge.", "FILLE ROBE ROUGE PORTER"),
            ("Il mesure un mètre quatre-vingts.", "IX-3 UN MÈTRE QUATRE-VINGTS MESURER"),
        ],
    },
    {
        "name": "heure_horaires",
        "description": "Phrases avec des heures, des horaires, des durées",
        "examples": [
            ("Il est trois heures de l'après-midi.", "APRÈS-MIDI 3 HEURE"),
            ("Le cours dure deux heures.", "COURS 2 HEURE DURER"),
            ("Je me lève à 7 heures.", "7 HEURE IX-1 LEVER"),
        ],
    },
    {
        "name": "nombres_quantites",
        "description": "Phrases avec des nombres, des quantités, des mesures",
        "examples": [
            ("Il y a vingt élèves dans la classe.", "CLASSE ÉLÈVE 20"),
            ("Je voudrais deux kilos de pommes.", "IX-1 POMME 2 KILO VOULOIR"),
            ("Elle a cinq chats.", "IX-3 CHAT 5"),
        ],
    },
    {
        "name": "communication_interaction",
        "description": "Phrases de communication quotidienne (salutations, remerciements, excuses, présentations)",
        "examples": [
            ("Bonjour, comment ça va ?", "BONJOUR ALLER COMMENT"),
            ("Merci beaucoup pour ton aide.", "POSS-2 AIDE MERCI BEAUCOUP"),
            ("Excuse-moi, je suis en retard.", "EXCUSE IX-1 RETARD"),
        ],
    },
    {
        "name": "pronoms_on_nous",
        "description": "Phrases avec on/nous et activités collectives",
        "examples": [
            ("On va au parc ensemble.", "IX-1-PL ENSEMBLE PARC ALLER"),
            ("Nous préparons une fête pour samedi.", "SAMEDI IX-1-PL FÊTE PRÉPARER"),
            ("On devrait se dépêcher.", "IX-1-PL DÉPÊCHER DEVOIR"),
        ],
    },
    {
        "name": "pronoms_vous_ils",
        "description": "Phrases avec vous/ils/elles et actions de groupe",
        "examples": [
            ("Vous avez fini vos devoirs ?", "IX-2-PL POSS-2-PL DEVOIR+ FINIR ?"),
            ("Ils partent en Italie demain.", "DEMAIN IX-3-PL ITALIE PARTIR"),
            ("Elles dansent très bien.", "IX-3-PL TRÈS BIEN DANSER"),
        ],
    },
    {
        "name": "relative_qui_que",
        "description": "Phrases avec des relatives (qui, que, dont, où)",
        "examples": [
            ("Le garçon qui court est mon frère.", "GARÇON COURIR POSS-1 FRÈRE"),
            ("Le film que j'ai vu était super.", "FILM IX-1 VOIR SUPER"),
            ("La ville où je suis né est petite.", "VILLE IX-1 NAÎTRE PETIT"),
        ],
    },
    {
        "name": "cause_consequence",
        "description": "Phrases de cause/conséquence (parce que, donc, car, alors, du coup)",
        "examples": [
            ("Je reste à la maison parce qu'il pleut.", "PLUIE DONC IX-1 MAISON RESTER"),
            ("Elle est malade, donc elle ne vient pas.", "IX-3 MALADE DONC VENIR PAS"),
            ("Il a bien travaillé, alors il a réussi.", "IX-3 BIEN TRAVAILLER DONC RÉUSSIR"),
        ],
    },
    # --- Catégories ciblées pour corriger les faiblesses du modèle neural ---
    {
        "name": "verbes_reflechis",
        "description": "Phrases avec des verbes pronominaux / réfléchis (se promener, s'habiller, se tromper, se laver, se réveiller, s'asseoir, se battre, se dépêcher, se souvenir, se coucher). Le pronom réfléchi est supprimé en gloss, seul le verbe à l'infinitif reste.",
        "examples": [
            ("Il s'habille vite.", "IX-3 VITE HABILLER"),
            ("Nous nous promenons dans le parc.", "IX-1-PL PARC PROMENER"),
            ("Vous vous êtes trompés.", "IX-2-PL TROMPER"),
        ],
    },
    {
        "name": "il_impersonnel_adjectif",
        "description": "Constructions impersonnelles 'il est + adjectif + de + infinitif' (il est possible, important, nécessaire, facile, difficile, interdit, dangereux, utile, inutile, agréable). 'Il' impersonnel est supprimé en gloss.",
        "examples": [
            ("Il est possible de venir demain.", "DEMAIN VENIR POSSIBLE"),
            ("Il est important de dormir huit heures.", "8 HEURE DORMIR IMPORTANT"),
            ("Il est interdit de fumer ici.", "ICI FUMER INTERDIT"),
        ],
    },
    {
        "name": "il_impersonnel_verbe",
        "description": "Constructions impersonnelles avec verbes : il faut, il semble, il paraît, il suffit, il arrive que, il vaut mieux. 'Il' impersonnel est supprimé.",
        "examples": [
            ("Il semble fatigué.", "FATIGUÉ SEMBLER"),
            ("Il paraît que le magasin ferme.", "MAGASIN FERMER PARAÎTRE"),
            ("Il suffit de demander.", "DEMANDER SUFFIRE"),
        ],
    },
    {
        "name": "expressions_idiomatiques_avoir",
        "description": "Expressions idiomatiques avec avoir (avoir raison, avoir tort, avoir peur, avoir faim, avoir soif, avoir besoin, avoir envie, avoir honte, avoir mal, avoir sommeil, avoir chaud, avoir froid). Le verbe avoir est conservé ou implicite selon l'expression.",
        "examples": [
            ("Tu as raison.", "IX-2 RAISON AVOIR"),
            ("Elle a tort de mentir.", "IX-3 MENTIR TORT AVOIR"),
            ("J'ai besoin d'aide.", "IX-1 AIDE BESOIN AVOIR"),
        ],
    },
    {
        "name": "expressions_idiomatiques_faire",
        "description": "Expressions idiomatiques avec faire/ça fait (ça fait longtemps, faire attention, faire semblant, faire la queue, faire du bruit). Aussi : manquer (tu me manques = IX-2 IX-1 MANQUER).",
        "examples": [
            ("Ça fait longtemps.", "LONGTEMPS"),
            ("Tu me manques beaucoup.", "IX-2 IX-1 MANQUER BEAUCOUP"),
            ("Il fait attention en traversant.", "IX-3 TRAVERSER ATTENTION FAIRE"),
        ],
    },
    {
        "name": "passe_recent_venir_de",
        "description": "Passé récent avec 'venir de + infinitif', signifiant une action qui vient de se terminer. En gloss : FINIR (marqueur d'aspect récent) + verbe.",
        "examples": [
            ("Je viens de manger.", "IX-1 MANGER FINIR"),
            ("Elle vient de partir.", "IX-3 PARTIR FINIR"),
            ("Nous venons d'arriver.", "IX-1-PL ARRIVER FINIR"),
        ],
    },
    {
        "name": "futur_proche_aller",
        "description": "Futur proche avec 'aller + infinitif'. En gloss : BIENTÔT + verbe (ou marqueur temporel futur).",
        "examples": [
            ("Je vais manger.", "IX-1 BIENTÔT MANGER"),
            ("Il va pleuvoir.", "BIENTÔT PLUIE"),
            ("Nous allons déménager.", "IX-1-PL BIENTÔT DÉMÉNAGER"),
        ],
    },
    {
        "name": "pronoms_objets_complexes",
        "description": "Phrases avec pronoms objets (me/m', te/t', le/la/l', lui, nous, vous, les, leur) combinés avec divers verbes et temps. L'objet pronominalisé est traduit par IX-N.",
        "examples": [
            ("Ils m'ont vu hier.", "HIER IX-3-PL IX-1 VOIR"),
            ("On vous appelle ce soir.", "SOIR IX-1-PL IX-2-PL APPELER"),
            ("Elle nous a invités à dîner.", "IX-3 IX-1-PL DÎNER INVITER"),
        ],
    },
    {
        "name": "il_y_a_existence",
        "description": "Constructions existentielles 'il y a + nom (+ nombre)'. En gloss : NOM NOMBRE AVOIR ou LIEU NOM AVOIR.",
        "examples": [
            ("Il y a trois chats dans le jardin.", "JARDIN CHAT TROIS AVOIR"),
            ("Il y a un problème.", "PROBLÈME AVOIR"),
            ("Il y a beaucoup de monde.", "MONDE BEAUCOUP AVOIR"),
        ],
    },
    {
        "name": "depuis_duree",
        "description": "Phrases avec 'depuis' indiquant une durée ou un point de départ temporel. DEPUIS est conservé en gloss.",
        "examples": [
            ("Il pleut depuis ce matin.", "MATIN DEPUIS PLUIE"),
            ("J'habite ici depuis cinq ans.", "ICI 5 AN DEPUIS IX-1 HABITER"),
            ("Elle attend depuis une heure.", "1 HEURE DEPUIS IX-3 ATTENDRE"),
        ],
    },
    {
        "name": "questions_informelles",
        "description": "Questions informelles avec 'c'est quoi', 'c'est qui', 'c'est où', 'c'est comment', 'il est où'. Le mot interrogatif va à la fin en gloss.",
        "examples": [
            ("C'est quoi ton nom ?", "POSS-2 NOM QUOI"),
            ("C'est qui ce monsieur ?", "MONSIEUR QUI"),
            ("Il est où le chat ?", "CHAT OÙ"),
        ],
    },
    {
        "name": "quand_futur_condition",
        "description": "Phrases conditionnelles ou temporelles avec 'quand + futur' (quand je serai grand, quand tu viendras, quand il fera beau). Marqueur temporel QUAND en début de gloss.",
        "examples": [
            ("Quand je serai grand, je serai pompier.", "QUAND IX-1 GRAND IX-1 POMPIER"),
            ("Quand tu viendras, on ira au cinéma.", "QUAND IX-2 VENIR IX-1-PL CINÉMA ALLER"),
            ("Quand il fera beau, nous irons à la plage.", "QUAND BEAU TEMPS IX-1-PL PLAGE ALLER"),
        ],
    },
    {
        "name": "negation_aucun_personne",
        "description": "Négation avec 'aucun(e)', 'personne ne', 'rien ne'. AUCUN/PERSONNE/RIEN en fin de gloss comme marqueur de négation.",
        "examples": [
            ("Aucun enfant ne joue dehors.", "ENFANT+ DEHORS JOUER AUCUN"),
            ("Personne ne m'a aidé.", "IX-1 AIDER PERSONNE"),
            ("Rien ne fonctionne.", "FONCTIONNER RIEN"),
        ],
    },
    {
        "name": "verbes_souvent_confondus",
        "description": "Phrases ciblant des verbes que le modèle confond : PLEURER (pas CHERCHER), APPELER (pas NOM), MANQUER, PROMENER (pas PROMENADE), HABILLER (pas ROBEILLER), TROMPER (pas DREQUENT). Insister sur la forme infinitive correcte.",
        "examples": [
            ("Pourquoi tu pleures ?", "IX-2 PLEURER POURQUOI"),
            ("Comment tu t'appelles ?", "IX-2 APPELER COMMENT"),
            ("Elle se promène au bord de la rivière.", "IX-3 RIVIÈRE PROMENER"),
        ],
    },
    {
        "name": "mots_longs_frequents",
        "description": "Phrases utilisant des mots fréquents souvent mal reproduits : BEAUCOUP, LONGTEMPS, TOUJOURS, MAINTENANT, VRAIMENT, ENSEMBLE, AUJOURD'HUI, QUELQUEFOIS. Insister sur l'orthographe complète.",
        "examples": [
            ("Il neige beaucoup.", "NEIGE BEAUCOUP"),
            ("Ça fait longtemps que je ne t'ai pas vu.", "LONGTEMPS IX-1 IX-2 VOIR PAS"),
            ("Nous sommes toujours ensemble.", "IX-1-PL TOUJOURS ENSEMBLE"),
        ],
    },
]

SYSTEM_PROMPT = """Tu es un expert linguiste spécialisé en Langue des Signes Française (LSF).
Tu génères des paires de traduction du français écrit vers la notation gloss LSF.

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
13. VERBES PRONOMINAUX (se laver, se promener, s'habiller, se tromper) :
    Le pronom réfléchi (se/s') est supprimé, garder UNIQUEMENT l'infinitif.
    Ex: "Il s'habille" → IX-3 HABILLER (PAS "ROBEILLER", pas "PROMENADE" pour PROMENER)
14. IL IMPERSONNEL (il pleut, il faut, il semble, il est possible/important) :
    Supprimer "il" impersonnel. Ex: "Il est possible de venir" → VENIR POSSIBLE
15. "IL Y A" existentiel → NOM NOMBRE AVOIR.
    Ex: "Il y a trois chats" → CHAT TROIS AVOIR
16. PASSÉ RÉCENT "venir de + infinitif" → VERBE FINIR (aspect accompli).
    Ex: "Je viens de manger" → IX-1 MANGER FINIR
17. "DEPUIS" temporel → conserver DEPUIS en gloss.
    Ex: "Il pleut depuis ce matin" → MATIN DEPUIS PLUIE
18. Expressions idiomatiques AVOIR : avoir raison → RAISON AVOIR, avoir tort → TORT AVOIR
19. "Ça fait longtemps" → LONGTEMPS (pas "LONG")
20. MANQUER : "Tu me manques" → IX-2 IX-1 MANQUER (pas PASSER)

ORTHOGRAPHE STRICTE DES GLOSS :
- Écrire les mots en ENTIER : BEAUCOUP (pas BEACOUP), LONGTEMPS (pas LONG),
  TOUJOURS, MAINTENANT, VRAIMENT, ENSEMBLE, AUJOURD'HUI
- PROMENER (pas PROMENADE), HABILLER (pas ROBEILLER), TROMPER (pas DREQUENT),
  PLEURER (pas CHERCHER), APPELER (pas NOM quand c'est le verbe s'appeler)

FORMAT DE SORTIE : Un objet JSON, rien d'autre.
{"pairs": [{"fr": "phrase en français", "gloss": "GLOSS EN LSF"}, ...]}

IMPORTANT :
- Génère des phrases VARIÉES et NATURELLES
- Utilise un vocabulaire RICHE et DIVERSIFIÉ
- Varie les longueurs : phrases courtes (3-5 mots) et longues (10-20 mots)
- Ne répète PAS les mêmes phrases, sois créatif
- Les gloss doivent être des traductions fidèles, pas des copies mot-à-mot
"""

# ---------------------------------------------------------------------------
# Variateurs de prompts pour maximiser la diversité
# ---------------------------------------------------------------------------

VOCABULARY_THEMES = [
    "nourriture et cuisine", "animaux", "sport et fitness", "musique et art",
    "technologie et informatique", "nature et environnement", "mode et vêtements",
    "médecine et santé", "voyage et tourisme", "école et éducation",
    "travail et bureau", "famille et relations", "transport et véhicules",
    "maison et ménage", "argent et finances", "météo et saisons",
    "fêtes et célébrations", "cinéma et séries", "jardinage et plantes",
    "bricolage et réparation", "lecture et littérature", "jeux et divertissement",
    "politique et société", "histoire et culture", "religion et spiritualité",
]

REGISTER_STYLES = [
    "langage courant informel entre amis",
    "langage soutenu/formel",
    "langage enfantin (phrases simples)",
    "langage professionnel",
    "conversations familières du quotidien",
    "récit/narration d'événements",
    "instructions et consignes",
    "expression d'opinions et de sentiments",
]


def build_user_prompt(category: dict, count: int, existing_phrases: set) -> str:
    """Construit le prompt utilisateur avec des variateurs aléatoires."""
    examples_text = "\n".join(
        f'  fr: "{ex[0]}" → gloss: "{ex[1]}"' for ex in category["examples"]
    )

    # Ajouter des variateurs aléatoires pour la diversité
    theme = random.choice(VOCABULARY_THEMES)
    style = random.choice(REGISTER_STYLES)

    avoid_text = ""
    if existing_phrases:
        # Échantillon aléatoire pour varier les exclusions montrées
        sample = random.sample(list(existing_phrases), min(15, len(existing_phrases)))
        avoid_text = f"\n\nNe génère PAS ces phrases (déjà dans le dataset) :\n" + "\n".join(f'- "{p}"' for p in sample)

    return f"""Génère exactement {count} paires de traduction français → gloss LSF.

Catégorie : {category['description']}
Thème de vocabulaire à privilégier : {theme}
Style de langage : {style}

Exemples de référence :
{examples_text}
{avoid_text}

Sois TRÈS créatif et varié dans les phrases, utilise des mots et situations différentes à chaque fois.
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
    # Nettoyer la réponse (enlever les ```json ... ```)
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
        # Essayer de trouver le JSON dans la réponse
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


def validate_with_rules(pairs: list[dict]) -> list[dict]:
    """Compare les gloss du LLM avec le traducteur à règles (info seulement)."""
    try:
        from spoken_to_signed.text_to_gloss.rules_fr import text_to_gloss_fr
    except ImportError:
        print("  Impossible de charger le traducteur à règles, validation ignorée")
        return pairs

    validated = []
    for pair in pairs:
        fr = pair["fr"]
        gloss_llm = pair["gloss"]

        result = text_to_gloss_fr(fr)
        gloss_rules = result["gloss_string"].replace(" ||", "").strip()

        match = gloss_llm.strip() == gloss_rules.strip()
        pair["gloss_rules"] = gloss_rules
        pair["match"] = match
        validated.append(pair)

    return validated


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


def generate_dataset(
    api_key: str,
    model: str,
    total_count: int,
    output_file: str,
    base_url: str = None,
):
    """Génère le dataset complet."""
    # Charger un dataset existant si on reprend
    all_pairs, existing_phrases = _load_existing_csv(output_file)
    if all_pairs:
        print(f"Reprise : {len(all_pairs)} paires existantes chargées depuis {output_file}")

    # Répartir les phrases restantes entre les catégories
    remaining = total_count - len(all_pairs)
    if remaining <= 0:
        print(f"Dataset déjà complet ({len(all_pairs)} paires). Rien à faire.")
        return

    per_category = max(5, remaining // len(CATEGORIES))
    remainder = remaining - per_category * len(CATEGORIES)

    print(f"=== Génération de {total_count} paires fr → gloss LSF ===")
    print(f"Modèle : {model}")
    print(f"Catégories : {len(CATEGORIES)}")
    print(f"~{per_category} phrases par catégorie")
    print()

    for i, category in enumerate(CATEGORIES):
        count = per_category + (1 if i < remainder else 0)
        batch_size = 50  # Le LLM ne peut pas générer trop de paires d'un coup
        generated_for_cat = 0
        low_yield_count = 0  # Compteur de lots à faible rendement

        print(f"[{i+1}/{len(CATEGORIES)}] {category['name']} ({count} phrases)...")

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

                # Sauvegarde incrémentale
                _save_csv(all_pairs, output_file)

                # Si le rendement est trop faible, passer à la suite
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

            # Pause entre les appels pour éviter le rate limiting
            time.sleep(1)

    print(f"\nTotal : {len(all_pairs)} paires générées")

    # Sauvegarde finale
    _save_csv(all_pairs, output_file)
    print(f"Terminé ! {len(all_pairs)} paires sauvegardées dans {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Génère un dataset français → gloss LSF via un LLM",
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="Clé API OpenAI (ou variable d'env OPENAI_API_KEY)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="URL de base pour une API compatible OpenAI (ex: Mistral, Ollama)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Modèle à utiliser (défaut: gpt-4o-mini)")
    parser.add_argument("--count", type=int, default=20000,
                        help="Nombre de paires à générer (défaut: 20000)")
    parser.add_argument("--output", type=str, default="dataset_fr_gloss.csv",
                        help="Fichier CSV de sortie (défaut: dataset_fr_gloss.csv)")
    args = parser.parse_args()

    if not args.api_key:
        print("Erreur : fournis une clé API avec --api-key ou la variable d'env OPENAI_API_KEY")
        sys.exit(1)

    generate_dataset(
        api_key=args.api_key,
        model=args.model,
        total_count=args.count,
        output_file=args.output,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
