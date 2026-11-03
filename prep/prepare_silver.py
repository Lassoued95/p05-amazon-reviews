#!/usr/bin/env python3
"""
P05 - Phase 3 : preparation bronze -> silver (extracteur streaming, reproductible).

Entree : Appliances.jsonl(.gz)
Sorties (dossier --sortie, qui ne doit pas exister : aucun ecrasement silencieux) :
  tokens.tsv    review_key <TAB> parent_asin <TAB> tokens separes par des espaces   (-> MR1)
  ratings.tsv   review_key <TAB> parent_asin <TAB> month <TAB> rating               (-> MR3)
  rejets.json   compteurs de rejets par motif
  manifest.json parametres, compteurs, N (produits avec au moins un texte valide), hashes

Echantillonnage deterministe par produit :
  garde un produit si blake2b(parent_asin) mod 100 < --pct   (100 = tout garder)

Usage :
  python prep/prepare_silver.py data/Appliances.jsonl --pct 1 --sortie data/silver/dev_p1
  python prep/prepare_silver.py data/toy/reviews_toy.jsonl --pct 100 --sortie data/silver/toy

Bibliotheque standard uniquement (Python 3.8+).
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

# Liste anglaise standard. Les negations (no, nor, not, never) n'y figurent volontairement pas.
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself of off on once only or other our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours
yourself yourselves
""".split())

HTML_TAG = re.compile(r"<[^>]+>")
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def sha256_fichier(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def bucket_produit(parent_asin):
    h = hashlib.blake2b(parent_asin.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 100


def tokeniser(texte, cfg):
    t = texte.lower().replace("\u2019", "'")
    if cfg["strip_html_tags"]:
        t = HTML_TAG.sub(" ", t)
    if cfg["normalize_negations"]:
        t = (t.replace("cannot", "can not").replace("can't", "can not")
              .replace("won't", "will not").replace("shan't", "shall not"))
        t = re.sub(r"n't\b", " not", t)
    sortie = []
    for tok in TOKEN.findall(t):
        if len(tok) < cfg["min_token_len"]:
            continue
        if cfg["drop_numeric_tokens"] and tok.isdigit():
            continue
        if tok in STOPWORDS:
            continue
        sortie.append(tok)
    return sortie


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fichier")
    p.add_argument("--pct", type=int, default=100, help="pourcentage de produits retenus (1..100)")
    p.add_argument("--sortie", required=True, help="dossier de sortie (ne doit pas exister)")
    p.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_nlp_v1.json"))
    p.add_argument("--max-lines", type=int, default=None, help="test rapide uniquement")
    args = p.parse_args()

    if not os.path.exists(args.fichier):
        sys.exit(f"Fichier introuvable : {args.fichier}")
    if os.path.exists(args.sortie):
        sys.exit(f"Le dossier {args.sortie} existe deja : choisir un nouveau dossier (pas d'ecrasement).")
    if not 1 <= args.pct <= 100:
        sys.exit("--pct doit etre entre 1 et 100")
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg["timestamp_unit"] != "ms":
        sys.exit("Cette version ne gere que timestamp_unit = ms (verifie par l'audit).")

    os.makedirs(args.sortie)
    f_tok = open(os.path.join(args.sortie, "tokens.tsv"), "w", encoding="utf-8", newline="\n")
    f_rat = open(os.path.join(args.sortie, "ratings.tsv"), "w", encoding="utf-8", newline="\n")

    c = Counter()                  # compteurs generaux
    rejets = Counter()             # rejets par motif (un avis peut cumuler plusieurs motifs)
    vus = set()
    produits_texte = set()         # produits avec au moins une ligne dans tokens.tsv  -> N
    produits_notes = set()
    mois_min = mois_max = None

    ouvrir = gzip.open if args.fichier.endswith(".gz") else open
    with ouvrir(args.fichier, "rt", encoding="utf-8", errors="replace") as f:
        for ligne in f:
            if args.max_lines and c["lignes_lues"] >= args.max_lines:
                break
            if not ligne.strip():
                continue
            c["lignes_lues"] += 1
            try:
                avis = json.loads(ligne)
                if not isinstance(avis, dict):
                    raise ValueError
            except ValueError:
                rejets["json_invalide"] += 1
                c["rejetes"] += 1
                continue

            pa = avis.get("parent_asin")
            if pa and bucket_produit(str(pa)) >= args.pct:
                c["hors_echantillon"] += 1
                continue

            motifs = []
            if not pa:
                motifs.append("parent_asin_absent")
            note = avis.get("rating")
            if isinstance(note, bool) or not isinstance(note, (int, float)) or not (1 <= note <= 5):
                motifs.append("note_hors_1_5")
            ts = avis.get("timestamp")
            mois = None
            if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0:
                if ts <= 1e11:
                    motifs.append("timestamp_unite_inattendue")
                else:
                    try:
                        mois = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m")
                    except (OverflowError, OSError, ValueError):
                        motifs.append("date_non_decodable")
            else:
                motifs.append("date_non_decodable")
            texte = f"{avis.get('title') or ''} {avis.get('text') or ''}".strip()
            if not texte:
                motifs.append("texte_vide")
            if motifs:
                c["rejetes"] += 1
                for m in motifs:
                    rejets[m] += 1
                continue

            # identifiant technique stable et anonymise (le user_id n'est jamais ecrit)
            brut = f"{avis.get('user_id')}|{pa}|{ts}|{texte}".encode("utf-8")
            cle = hashlib.blake2b(brut, digest_size=16)
            if cle.digest() in vus:
                c["doublons"] += 1
                continue
            vus.add(cle.digest())
            review_key = cle.hexdigest()

            c["valides"] += 1
            f_rat.write(f"{review_key}\t{pa}\t{mois}\t{float(note):.1f}\n")
            produits_notes.add(pa)
            mois_min = mois if mois_min is None or mois < mois_min else mois_min
            mois_max = mois if mois_max is None or mois > mois_max else mois_max

            tokens = tokeniser(texte, cfg)
            if not tokens:
                c["tokens_vides"] += 1
                continue
            f_tok.write(f"{review_key}\t{pa}\t{' '.join(tokens)}\n")
            c["avis_avec_tokens"] += 1
            c["tokens_total"] += len(tokens)
            produits_texte.add(pa)

    f_tok.close()
    f_rat.close()

    rejets_json = {"lignes_lues": c["lignes_lues"], "rejetes": c["rejetes"], "par_motif": dict(rejets)}
    with open(os.path.join(args.sortie, "rejets.json"), "w", encoding="utf-8") as f:
        json.dump(rejets_json, f, ensure_ascii=False, indent=2)

    manifest = {
        "script": "prep/prepare_silver.py",
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entree": {"fichier": os.path.basename(args.fichier),
                   "taille_octets": os.path.getsize(args.fichier),
                   "sha256": sha256_fichier(args.fichier),
                   "max_lines": args.max_lines},
        "echantillonnage": {"methode": "blake2b(parent_asin) mod 100 < pct", "pct": args.pct},
        "config_nlp": cfg,
        "config_sha256": sha256_fichier(args.config),
        "compteurs": dict(c),
        "N_produits_avec_texte_valide": len(produits_texte),
        "produits_avec_notes": len(produits_notes),
        "mois_min": mois_min,
        "mois_max": mois_max,
        "sorties_sha256": {n: sha256_fichier(os.path.join(args.sortie, n)) for n in ("tokens.tsv", "ratings.tsv")},
    }
    with open(os.path.join(args.sortie, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=== PREPARATION SILVER ===")
    print(f"Lignes lues            : {c['lignes_lues']:,}")
    print(f"Hors echantillon       : {c['hors_echantillon']:,}   (pct = {args.pct})")
    print(f"Rejetes                : {c['rejetes']:,}  {dict(rejets)}")
    print(f"Doublons               : {c['doublons']:,}")
    print(f"Avis valides           : {c['valides']:,}")
    print(f"Avis avec tokens       : {c['avis_avec_tokens']:,}  (tokens vides : {c['tokens_vides']:,})")
    print(f"Tokens ecrits          : {c['tokens_total']:,}")
    print(f"N (produits avec texte): {len(produits_texte):,}")
    print(f"Periode (mois)         : {mois_min} -> {mois_max}")
    print(f"Sorties dans           : {args.sortie}")


if __name__ == "__main__":
    main()