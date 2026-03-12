# -*- coding: utf-8 -*-
"""
BAG VBO Verwerking - Early filtering + 2 workers
-------------------------------------------------
Geheugenefficient door:
  1. Early filtering per worker: alleen bronhouder-matches worden teruggestuurd
  2. Slechts 2 workers: beperkte geheugenduplicatie, wel snelheidswinst
  3. del + gc.collect() per bestand in elke worker
  4. Bestanden altijd correct gesloten via 'with open()'

Bugfix namespace-stripper:
  - [\w-]+ i.p.v. \w+ zodat 'Objecten-ref:PandRef' correct wordt gestript.
    PandRef en NummeraanduidingRef zijn nu gevuld.

Bugfix BeginGeldigheid-filter:
  - Voorheen werd elk voorkomen afzonderlijk getoetst op beginGeldigheid.
    Daardoor kwamen VBO's door waarvan een LATER voorkomen toevallig in 2025
    viel, terwijl het VBO al eerder (bijv. 2021) voor het eerst was gevormd.
  - Fix: per identificatie wordt het EERSTE voorkomen bepaald (laagste
    voorkomenidentificatie). Alleen als dat eerste voorkomen een beginGeldigheid
    heeft die begint met BEGINJAAR, telt het VBO mee.
"""

# ==================================================
# Config
# ==================================================

INPUT_DIR            = r"C:\GISdatabase\BAG\data\9999VBO08012026"
TUSSENRESULTAAT      = r"UTRECHT_VBO_Woonfunctie_Gevormd_v3.shp"
OUTPUT_SHP           = r"UTRECHT_VBO_VG_2025_v3.shp"
OUTPUT_SHP_VBO_Uniek = r"UTRECHT_VBO_Wonen_Uniek_v3.shp"
OUTPUT_CSV           = r"UTRECHT_VBO_VG_id_2025_v3.csv"
BRONHOUDERS          = ['0344']
STATUS               = 'Verblijfsobject gevormd'
STATUS_TOO           = 'Verblijfsobject ten onrechte opgevoerd'
GEBRUIKSFUNCTIE      = ['woonfunctie']
BEGINJAAR            = '2025'
CRS                  = "EPSG:28992"

NUM_WORKERS          = 5

# ==================================================
# Imports
# ==================================================

import os
import re
import gc
import glob
import warnings
from multiprocessing import Pool

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ==================================================
# Namespace strippen
# ==================================================

def strip_namespaces(xml_text):
    """
    Verwijder alle XML-namespace prefixen en declaraties.
    [\w-]+ zodat prefixen met koppelteken (Objecten-ref, sl-bag-extract)
    ook worden gestript.
    """
    xml_text = re.sub(r'\s+xmlns(?:[\w-]+)?(?::[\w-]+)?="[^"]*"', '', xml_text)
    xml_text = re.sub(r'<([\w-]+):', '<',  xml_text)
    xml_text = re.sub(r'</([\w-]+):', '</', xml_text)
    return xml_text


# ==================================================
# Helperfunctie tekstextractie
# ==================================================

def get_text(tag, block, default=None):
    m = re.search(r'<' + tag + r'[^>]*>(.*?)</' + tag + r'>', block, re.DOTALL)
    return m.group(1).strip() if m else default


# ==================================================
# Worker-functie: parse + early filter
# ==================================================

def verwerk_bestand(args):
    filepath, bronhouders_tuple = args
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()

        clean = strip_namespaces(raw)
        del raw
        gc.collect()

        records = _parse_en_filter(clean, bronhouders_tuple)
        del clean
        gc.collect()

        return records

    except Exception as e:
        return [{'_fout': str(e), '_bestand': filepath}]


def _parse_en_filter(xml_text, bronhouders_tuple):
    """
    Parseer Verblijfsobject-blokken en bewaar alleen bronhouder-matches
    (early filter). Alle voorkomens per VBO worden bewaard — de
    BeginGeldigheid-logica verderop vereist het complete voorkomshistorie.
    """
    records = []
    vbo_pattern = re.compile(r'<Verblijfsobject>(.*?)</Verblijfsobject>', re.DOTALL)

    for match in vbo_pattern.finditer(xml_text):
        block = match.group(1)

        id_match = re.search(
            r'<identificatie[^>]*domein="NL\.IMBAG\.Verblijfsobject"[^>]*>(.*?)</identificatie>',
            block, re.DOTALL
        )
        if not id_match:
            continue
        identificatie = id_match.group(1).strip()

        # *** EARLY FILTER ***
        if not identificatie.startswith(bronhouders_tuple):
            continue

        pos_match = re.search(r'<pos>\s*([\d.]+)\s+([\d.]+)', block)
        geometry = (
            Point(float(pos_match.group(1)), float(pos_match.group(2)))
            if pos_match else None
        )

        records.append({
            'identificatie':          identificatie,
            'gebruiksdoel':           get_text('gebruiksdoel',           block),
            'oppervlakte':            get_text('oppervlakte',            block),
            'status':                 get_text('status',                 block),
            'geconstateerd':          get_text('geconstateerd',          block),
            'documentdatum':          get_text('documentdatum',          block),
            'documentnummer':         get_text('documentnummer',         block),
            'voorkomenidentificatie': get_text('voorkomenidentificatie', block),
            'beginGeldigheid':        get_text('beginGeldigheid',        block),
            'eindGeldigheid':         get_text('eindGeldigheid',         block),
            'tijdstipRegistratie':    get_text('tijdstipRegistratie',    block),
            'PandRef':                get_text('PandRef',                block),
            'NummeraanduidingRef':    get_text('NummeraanduidingRef',    block),
            'geometry':               geometry,
        })

    return records


# ==================================================
# Debugstap op het eerste bestand
# ==================================================

def debug_eerste_bestand(bestanden):
    print("\n" + "=" * 60)
    print("DEBUG: Analyse van het eerste bestand")
    print("=" * 60)

    eerste = bestanden[0]
    print(f"Bestand : {eerste}")
    print(f"Grootte : {os.path.getsize(eerste) / 1024 / 1024:.1f} MB\n")

    with open(eerste, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    print("--- Eerste 500 tekens (ruw) ---")
    print(raw[:500])
    print()

    clean = strip_namespaces(raw)
    print("--- Eerste 500 tekens (namespace-gestript) ---")
    print(clean[:500])
    print()

    pandref_ruw   = len(re.findall(r'Objecten-ref:PandRef',            raw))
    pandref_clean = len(re.findall(r'<PandRef',                        clean))
    numref_ruw    = len(re.findall(r'Objecten-ref:NummeraanduidingRef', raw))
    numref_clean  = len(re.findall(r'<NummeraanduidingRef',            clean))
    print(f"<Objecten-ref:PandRef> in ruw XML        : {pandref_ruw}")
    print(f"<PandRef> na namespace-strip              : {pandref_clean}  "
          f"{'OK' if pandref_clean > 0 else 'PROBLEEM!'}")
    print(f"<Objecten-ref:NummeraanduidingRef> in ruw : {numref_ruw}")
    print(f"<NummeraanduidingRef> na namespace-strip  : {numref_clean}  "
          f"{'OK' if numref_clean > 0 else 'PROBLEEM!'}")
    print()

    n_vbo = len(re.findall(r'<Verblijfsobject>', clean))
    print(f"Aantal <Verblijfsobject>-blokken : {n_vbo}")

    records = _parse_en_filter(clean, tuple(BRONHOUDERS))
    print(f"Records na bronhouder-filter     : {len(records)}")

    if records:
        print("\nEerste gefilterd record:")
        for k, v in records[0].items():
            if k != 'geometry':
                print(f"  {k:30s}: {v}")
        print(f"  {'geometry':30s}: {records[0]['geometry']}")

    print("=" * 60 + "\n")
    del raw, clean, records
    gc.collect()


# ==================================================
# Hoofdverwerking
# ==================================================

def main():
    bestanden = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xml")))
    print(f"Aantal XML-bestanden gevonden: {len(bestanden)}")

    if not bestanden:
        print(f"[FOUT] Geen XML-bestanden gevonden in: {INPUT_DIR}")
        return

    debug_eerste_bestand(bestanden)

    bronhouders_tuple = tuple(BRONHOUDERS)
    taken = [(fp, bronhouders_tuple) for fp in bestanden]

    print(f"Starten met {NUM_WORKERS} workers (early filtering actief)...\n")

    alle_records = []
    fouten = []

    with Pool(processes=NUM_WORKERS) as pool:
        for records in tqdm(
            pool.imap(verwerk_bestand, taken, chunksize=2),
            total=len(bestanden),
            desc="Verwerken",
            unit="bestand"
        ):
            for r in records:
                if '_fout' in r:
                    fouten.append(r)
                else:
                    alle_records.append(r)

    if fouten:
        print(f"\n[WAARSCHUWING] {len(fouten)} bestanden gaven een fout:")
        for f in fouten[:5]:
            print(f"  {f['_bestand']}: {f['_fout']}")

    print(f"\nTotaal bronhouder-matches: {len(alle_records):,}")

    if not alle_records:
        print("[FOUT] Geen records gevonden. Controleer bronhoudercode en XML-formaat.")
        return

    # ==================================================
    # GeoDataFrame aanmaken
    # ==================================================
    gdf_all = gpd.GeoDataFrame(alle_records, geometry='geometry', crs=CRS)
    del alle_records
    gc.collect()

    gdf_all['oppervlakte']            = pd.to_numeric(gdf_all['oppervlakte'],            errors='coerce')
    gdf_all['voorkomenidentificatie'] = pd.to_numeric(gdf_all['voorkomenidentificatie'], errors='coerce')

    # ==================================================
    # Filter 1: Bronhouder (al gedaan via early filter)
    # ==================================================
    gpd_Bronhouder = gdf_all
    print(f"\n{'='*50}")
    print(f"gpd_Bronhouder ({BRONHOUDERS}): {len(gpd_Bronhouder):,} objecten")

    # ==================================================
    # Filter 2: Gebruiksfunctie
    # ==================================================
    gpd_Gebruiksfunctie = gpd_Bronhouder[
        gpd_Bronhouder['gebruiksdoel'].isin(GEBRUIKSFUNCTIE)
    ].copy()
    print(f"gpd_Gebruiksfunctie {GEBRUIKSFUNCTIE}: {len(gpd_Gebruiksfunctie):,} objecten")

    # ==================================================
    # Filter 3a/3b: StatusVG en StatusTOO
    # ==================================================
    gpd_StatusVG  = gpd_Gebruiksfunctie[gpd_Gebruiksfunctie['status'] == STATUS    ].copy()
    gpd_StatusTOO = gpd_Gebruiksfunctie[gpd_Gebruiksfunctie['status'] == STATUS_TOO].copy()

    # ==================================================
    # Filter 4: gpd_Status — VG minus TOO-identificaties
    # ==================================================
    id_too     = set(gpd_StatusTOO['identificatie'].dropna())
    gpd_Status = gpd_StatusVG[~gpd_StatusVG['identificatie'].isin(id_too)].copy()

    n_dupl = (gpd_Status['identificatie'].value_counts() > 1).sum()

    print(f"\nBronhouder + gebruiksfunctie + statusVG + statusTOO + gpd_Status:")
    print(f"  StatusVG  : {len(gpd_StatusVG):,}")
    print(f"  StatusTOO : {len(gpd_StatusTOO):,}")
    print(f"  gpd_Status: {len(gpd_Status):,}  (VG minus TOO-identificaties)")
    print(f"  ID's die >1x voorkomen in gpd_Status: {n_dupl:,}")

    # ==================================================
    # Tussenresultaat shapefile
    # ==================================================
    print(f"\nTussenresultaat opslaan: {TUSSENRESULTAAT} ...")
    gpd_Status.to_file(TUSSENRESULTAAT, driver='ESRI Shapefile', encoding='utf-8')
    print("  Opgeslagen")

    # ==================================================
    # Filter 5: BeginGeldigheid — op basis van het EERSTE voorkomen
    # ----------------------------------------------------------
    # Probleem met de vorige aanpak:
    #   gpd_Status bevat alle voorkomens van "Verblijfsobject gevormd".
    #   Een VBO dat al in 2021 werd gevormd kan een latere administratieve
    #   mutatie hebben met voorkomenI 5 en beginGeldigheid 2025-xx-xx.
    #   De oude filter pakte dat voorkomen mee, wat ongewenst is.
    #
    # Correcte aanpak:
    #   Per identificatie zoeken we het LAAGSTE voorkomenidentificatie-nummer
    #   (= het moment waarop het VBO voor het EERST de status "gevormd" kreeg).
    #   Alleen als dat eerste voorkomen beginGeldigheid heeft in BEGINJAAR,
    #   nemen we het VBO op. Latere voorkomens van hetzelfde VBO worden
    #   bewust NIET meegeteld, omdat het VBO toen al bestond.
    # ==================================================

    # Stap A: per identificatie het rijnummer van het laagste voorkomenidentificatie
    idx_eerste = (
        gpd_Status
        .sort_values('voorkomenidentificatie')
        .groupby('identificatie', sort=False)
        .first()
        .reset_index()
    )

    # Stap B: welke identificaties hebben hun EERSTE voorkomen in BEGINJAAR?
    mask_eerste_in_jaar = idx_eerste['beginGeldigheid'].str.startswith(BEGINJAAR, na=False)
    id_eerste_in_jaar   = set(idx_eerste.loc[mask_eerste_in_jaar, 'identificatie'])

    # Stap C: uit gpd_Status alleen die identificaties bewaren
    #         (alle bijbehorende voorkomens blijven intact voor de sortering)
    gpd_BeginGeldigheid = gpd_Status[
        gpd_Status['identificatie'].isin(id_eerste_in_jaar)
    ].copy()

    # Sorteren: identificatie laag→hoog, dan voorkomenidentificatie laag→hoog
    gpd_BeginGeldigheid = gpd_BeginGeldigheid.sort_values(
        by=['identificatie', 'voorkomenidentificatie'],
        ascending=[True, True]
    ).reset_index(drop=True)

    print(f"\ngpd_BeginGeldigheid (eerste voorkomen VG in '{BEGINJAAR}'): "
          f"{len(gpd_BeginGeldigheid):,} voorkomens "
          f"van {len(id_eerste_in_jaar):,} unieke VBO's")

    # Ter controle: hoeveel VBO's vielen af omdat hun eerste VG voor BEGINJAAR lag?
    id_alle_vg   = set(gpd_Status['identificatie'])
    id_voor_jaar = id_alle_vg - id_eerste_in_jaar
    print(f"  VBO's afgevallen (eerste VG was vóór {BEGINJAAR}): "
          f"{len(id_voor_jaar):,}")

    # ==================================================
    # OUTPUT_SHP: alle voorkomens van de gekwalificeerde VBO's
    # ==================================================
    print(f"\nOutput shapefile opslaan: {OUTPUT_SHP} ...")
    gpd_BeginGeldigheid.to_file(OUTPUT_SHP, driver='ESRI Shapefile', encoding='utf-8')
    print("  Opgeslagen")

    gpd_BeginGeldigheid.drop(columns=['geometry']).to_csv(
        OUTPUT_CSV, index=False, encoding='utf-8-sig', sep=';'
    )
    print(f"  CSV opgeslagen: {OUTPUT_CSV}")

    # ==================================================
    # gpd_VBO_Uniek: één rij per identificatie (eerste voorkomen)
    # Na de sortering hierboven staat het laagste voorkomenidentificatie bovenaan.
    # ==================================================
    gpd_VBO_Uniek = gpd_BeginGeldigheid.drop_duplicates(
        subset='identificatie', keep='first'
    ).copy()

    n_verwijderd = len(gpd_BeginGeldigheid) - len(gpd_VBO_Uniek)
    print(f"\ngpd_VBO_Uniek: {len(gpd_VBO_Uniek):,} unieke VBO's "
          f"({n_verwijderd} latere voorkomens verwijderd)")

    print(f"Output shapefile uniek opslaan: {OUTPUT_SHP_VBO_Uniek} ...")
    gpd_VBO_Uniek.to_file(OUTPUT_SHP_VBO_Uniek, driver='ESRI Shapefile', encoding='utf-8')
    print("  Opgeslagen")

    # ==================================================
    # Eindoverzicht
    # ==================================================
    print(f"\n{'='*60}")
    print("EINDOVERZICHT")
    print(f"{'='*60}")
    print(f"  gpd_Bronhouder          : {len(gpd_Bronhouder):,}")
    print(f"  gpd_Gebruiksfunctie     : {len(gpd_Gebruiksfunctie):,}")
    print(f"  gpd_StatusVG            : {len(gpd_StatusVG):,}")
    print(f"  gpd_StatusTOO           : {len(gpd_StatusTOO):,}")
    print(f"  gpd_Status              : {len(gpd_Status):,}  (ID's >1x: {n_dupl})")
    print(f"  gpd_BeginGeldigheid     : {len(gpd_BeginGeldigheid):,}  "
          f"({len(id_eerste_in_jaar):,} unieke VBO's waarvan eerste VG in {BEGINJAAR})")
    print(f"  gpd_VBO_Uniek           : {len(gpd_VBO_Uniek):,}")
    print(f"{'='*60}\n")


# ==================================================
# Entry point  --  multiprocessing vereist deze guard op Windows
# ==================================================

if __name__ == '__main__':
    main()
