#!/usr/bin/env python3
"""
P05 - Etape 1 : exploration du fichier Amazon Reviews'23 (Appliances).

Lit le .jsonl.gz en streaming (sans le decompresser sur le disque) et produit :
  - un resume lisible dans le terminal ;
  - profil_dataset.json : chiffres pour la note de cadrage et le manifeste ;
  - echantillon_temoin.json : 5 avis tires au hasard (graine fixe), user_id masque.

Usage :
  python explorer_dataset.py Appliances.jsonl.gz
  python explorer_dataset.py Appliances.jsonl.gz --max-lines 50000   # test rapide

Uniquement la bibliotheque standard Python 3.8+.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone

SEUIL_AVIS_PRODUIT = 20  # seuil pedagogique du cahier des charges (configurable)


def sha256_fichier(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def unite_timestamp(ts):
    """13 chiffres environ -> millisecondes, 10 -> secondes."""
    return "ms" if ts > 1e11 else "s"


def mediane_depuis_histo(histo):
    total = sum(histo.values())
    if total == 0:
        return None
    cumul, milieu = 0, total / 2
    for valeur in sorted(histo):
        cumul += histo[valeur]
        if cumul >= milieu:
            return valeur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fichier", help="chemin vers Appliances.jsonl.gz")
    p.add_argument("--max-lines", type=int, default=None, help="limiter pour un test rapide")
    p.add_argument("--sortie", default=".", help="dossier des fichiers produits")
    args = p.parse_args()

    if not os.path.exists(args.fichier):
        sys.exit(f"Fichier introuvable : {args.fichier}")

    total = valides = 0
    rejets = Counter()            # un avis peut cumuler plusieurs motifs
    rejetes = 0
    schema = Counter()            # champs rencontres
    types = {}                    # type observe par champ
    notes = Counter()
    annees = Counter()
    ts_min = ts_max = None
    unites = Counter()
    par_produit = Counter()       # parent_asin -> nb avis valides
    asins = set()
    longueurs = Counter()         # nb de mots (title + text) -> nb avis
    verified = Counter()
    helpful_pos = 0
    empreintes = set()
    doublons = 0
    rng = random.Random(42)
    echantillon = []

    with gzip.open(args.fichier, "rt", encoding="utf-8", errors="replace") as f:
        for ligne in f:
            if args.max_lines and total >= args.max_lines:
                break
            total += 1
            if total % 200_000 == 0:
                print(f"  ... {total:,} lignes lues", file=sys.stderr)

            try:
                avis = json.loads(ligne)
                if not isinstance(avis, dict):
                    raise ValueError
            except ValueError:
                rejets["json_invalide"] += 1
                rejetes += 1
                continue

            for cle, val in avis.items():
                schema[cle] += 1
                types.setdefault(cle, type(val).__name__)

            motifs = []
            pa = avis.get("parent_asin")
            if not pa:
                motifs.append("parent_asin_absent")

            note = avis.get("rating")
            if not isinstance(note, (int, float)) or not (1 <= note <= 5):
                motifs.append("note_hors_1_5")

            ts = avis.get("timestamp")
            date = None
            if isinstance(ts, (int, float)) and ts > 0:
                u = unite_timestamp(ts)
                try:
                    date = datetime.fromtimestamp(ts / 1000 if u == "ms" else ts, tz=timezone.utc)
                    unites[u] += 1
                except (OverflowError, OSError, ValueError):
                    date = None
            if date is None:
                motifs.append("date_non_decodable")

            texte = f"{avis.get('title') or ''} {avis.get('text') or ''}".strip()
            if not texte:
                motifs.append("texte_vide")

            if motifs:
                rejetes += 1
                for m in motifs:
                    rejets[m] += 1
                continue

            # doublon : meme utilisateur, produit, date et texte
            cle_dedup = f"{avis.get('user_id')}|{pa}|{ts}|{texte}".encode("utf-8")
            emp = int.from_bytes(hashlib.blake2b(cle_dedup, digest_size=8).digest(), "big")
            if emp in empreintes:
                doublons += 1
                continue
            empreintes.add(emp)

            valides += 1
            notes[int(round(note))] += 1
            annees[date.year] += 1
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
            par_produit[pa] += 1
            if avis.get("asin"):
                asins.add(avis["asin"])
            longueurs[len(texte.split())] += 1
            verified[bool(avis.get("verified_purchase"))] += 1
            if (avis.get("helpful_vote") or 0) > 0:
                helpful_pos += 1

            # echantillon temoin : reservoir sampling, graine fixe -> reproductible
            propre = {k: v for k, v in avis.items() if k not in ("images", "user_id")}
            propre["user_id"] = "<masque>"
            if len(echantillon) < 5:
                echantillon.append(propre)
            else:
                j = rng.randint(0, valides - 1)
                if j < 5:
                    echantillon[j] = propre

    def date_iso(t):
        if t is None:
            return None
        return datetime.fromtimestamp(t / 1000 if unite_timestamp(t) == "ms" else t,
                                      tz=timezone.utc).strftime("%Y-%m-%d")

    nb_produits = len(par_produit)
    produits_seuil = sum(1 for n in par_produit.values() if n >= SEUIL_AVIS_PRODUIT)
    avis_seuil = sum(n for n in par_produit.values() if n >= SEUIL_AVIS_PRODUIT)

    profil = {
        "fichier": os.path.basename(args.fichier),
        "taille_octets": os.path.getsize(args.fichier),
        "sha256": sha256_fichier(args.fichier),
        "date_analyse_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lignes_lues": total,
        "limite_max_lines": args.max_lines,
        "avis_rejetes": rejetes,
        "rejets_par_motif": dict(rejets),
        "doublons_supprimes": doublons,
        "avis_valides_uniques": valides,
        "taux_conservation": round(valides / total, 4) if total else None,
        "schema_effectif": {k: {"presence": n, "type": types[k]} for k, n in schema.most_common()},
        "unite_timestamp": dict(unites),
        "periode": {"min": date_iso(ts_min), "max": date_iso(ts_max)},
        "avis_par_annee": dict(sorted(annees.items())),
        "distribution_notes": dict(sorted(notes.items())),
        "produits_parent_asin": nb_produits,
        "variantes_asin": len(asins),
        f"produits_avec_{SEUIL_AVIS_PRODUIT}_avis_ou_plus": produits_seuil,
        f"avis_couverts_par_ces_produits": avis_seuil,
        "top5_produits_les_plus_commentes": par_produit.most_common(5),
        "mots_par_avis": {
            "mediane": mediane_depuis_histo(longueurs),
            "moyenne": round(sum(k * v for k, v in longueurs.items()) / valides, 1) if valides else None,
        },
        "part_verified_purchase": round(verified[True] / valides, 4) if valides else None,
        "part_helpful_vote_positif": round(helpful_pos / valides, 4) if valides else None,
    }

    os.makedirs(args.sortie, exist_ok=True)
    with open(os.path.join(args.sortie, "profil_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(profil, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.sortie, "echantillon_temoin.json"), "w", encoding="utf-8") as f:
        json.dump(echantillon, f, ensure_ascii=False, indent=2)

    print("\n=== PROFIL DU DATASET ===")
    print(f"Lignes lues            : {total:,}")
    print(f"Avis rejetes           : {rejetes:,}  {dict(rejets)}")
    print(f"Doublons supprimes     : {doublons:,}")
    print(f"Avis valides uniques   : {valides:,}  (conservation {profil['taux_conservation']})")
    print(f"Champs rencontres      : {', '.join(schema)}")
    print(f"Unite du timestamp     : {dict(unites)}")
    print(f"Periode couverte       : {profil['periode']['min']} -> {profil['periode']['max']}")
    print(f"Distribution des notes : {profil['distribution_notes']}")
    print(f"Produits (parent_asin) : {nb_produits:,}   variantes (asin) : {len(asins):,}")
    print(f"Produits >= {SEUIL_AVIS_PRODUIT} avis    : {produits_seuil:,} (couvrent {avis_seuil:,} avis)")
    print(f"Top 5 produits         : {par_produit.most_common(5)}")
    print(f"Mots par avis          : {profil['mots_par_avis']}")
    print("\nFichiers ecrits : profil_dataset.json, echantillon_temoin.json")


if __name__ == "__main__":
    main()
