# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:26:41 2026

@author: Thomas


"""

"""
==================================================
Config
==================================================
"""
INPUT_DIR       = r"C:\GISdatabase\BAG\data\9999PND08012026"
OUTPUT_SHP      = r"UTRECHT_Panden_BG_2025_V3.shp"
OUTPUT_CSV      = r"UTRECHT_it2_panden_BG_id_2025_V3.csv"
BRONHOUDERS     = ['0344']
STATUS          = 'Bouw gestart'
BEGINGELDIGHEID = 2025
CRS             = "EPSG:28992"
MAX_WORKERS     = 4

# ==============================================================================
# Imports
# ==============================================================================
import os, re, csv, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# Namespace-stripping  (maakt script robuust voor alle BAG-varianten)
# ==============================================================================
RE_STRIP_NS      = re.compile(r'<(/?)[\w]+:([\w]+)')
RE_STRIP_DECL    = re.compile(r'\s+xmlns(?::\w+)?="[^"]*"')
RE_STRIP_ATTR_NS = re.compile(r'\s[\w]+:([\w]+)=')

try:
    from lxml import etree
    USE_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    USE_LXML = False


def strip_namespaces(xml_bytes: bytes) -> bytes:
    text = xml_bytes.decode("utf-8", errors="replace")
    text = RE_STRIP_NS.sub(r'<\1\2', text)
    text = RE_STRIP_DECL.sub('', text)
    text = RE_STRIP_ATTR_NS.sub(r' \1=', text)
    return text.encode("utf-8")


def txt(elem, tag):
    child = elem.find(f".//{tag}")
    return child.text.strip() if child is not None and child.text else None


def parse_poslist(poslist_text: str, dim: int = 3):
    nums = list(map(float, poslist_text.split()))
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), dim)]


def build_polygon(polygon_elem):
    ext = polygon_elem.find(".//exterior//posList")
    if ext is None or not ext.text:
        return None
    dim = int(polygon_elem.get("srsDimension", 3))
    exterior = parse_poslist(ext.text, dim)
    holes = [
        parse_poslist(pl.text, dim)
        for interior in polygon_elem.findall(".//interior")
        for pl in [interior.find(".//posList")]
        if pl is not None and pl.text
    ]
    return Polygon(exterior, holes) if len(exterior) >= 3 else None


def extract_geometry(pand_elem):
    geom_container = pand_elem.find("geometrie")
    if geom_container is None:
        return None
    polygons = [p for pe in geom_container.findall(".//Polygon")
                for p in [build_polygon(pe)] if p is not None]
    if not polygons:
        return None
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


# ==============================================================================
# Per-bestand verwerking (subproces)
# ==============================================================================
def process_file(xml_path: str, bronhouders: list):
    """
    Leest één XML-bestand en geeft ALLE Pand-voorkomens terug die voldoen
    aan de bronhouder-filter. Status/geldigheid-filtering gebeurt daarna
    in het hoofdproces, zodat we alle voorkomens per pand kunnen vergelijken.
    """
    bronhouders_set = set(bronhouders)
    records = []

    try:
        raw   = Path(xml_path).read_bytes()
        clean = strip_namespaces(raw)
        root  = etree.fromstring(clean) if USE_LXML else \
                etree.fromstring(clean.decode("utf-8"))
    except Exception:
        return records

    for pand in root.iter("Pand"):
        id_elem = pand.find("identificatie")
        if id_elem is None:
            continue
        identificatie = (id_elem.text or "").strip()
        if not any(identificatie.startswith(b) for b in bronhouders_set):
            continue

        status       = txt(pand, "status")
        bouwjaar     = txt(pand, "oorspronkelijkBouwjaar")
        geconst      = txt(pand, "geconstateerd")
        docdatum     = txt(pand, "documentdatum")
        docnummer    = txt(pand, "documentnummer")
        domein       = id_elem.get("domein", "")

        begin_geldigheid = None
        tijdstip_reg     = None
        tijdstip_reg_lv  = None
        voorkomenid      = None

        voorkomen = pand.find(".//Voorkomen")
        if voorkomen is not None:
            voorkomenid      = txt(voorkomen, "voorkomenidentificatie")
            begin_geldigheid = txt(voorkomen, "beginGeldigheid")
            tijdstip_reg     = txt(voorkomen, "tijdstipRegistratie")
            tijdstip_reg_lv  = txt(voorkomen, "tijdstipRegistratieLV")

        geom = extract_geometry(pand)

        records.append({
            "identificatie":   identificatie,
            "domein":          domein,
            "status":          status,
            "bouwjaar":        bouwjaar,
            "geconstateerd":   geconst,
            "documentdatum":   docdatum,
            "documentnummer":  docnummer,
            "beginGeldigheid": begin_geldigheid,
            "tijdstipReg":     tijdstip_reg,
            "tijdstipRegLV":   tijdstip_reg_lv,
            "voorkomenId":     voorkomenid,
            "bronbestand":     os.path.basename(xml_path),
            "geometry":        geom,
        })

    return records


# ==============================================================================
# Debug
# ==============================================================================
def debug_first_file(xml_files):
    print("\n[DEBUG] Eerste <Pand>-element na namespace-stripping:")
    try:
        raw   = Path(str(xml_files[0])).read_bytes()
        clean = strip_namespaces(raw)
        root  = etree.fromstring(clean) if USE_LXML else \
                etree.fromstring(clean.decode("utf-8"))
        pand  = next(root.iter("Pand"), None)
        if pand is not None:
            snippet = etree.tostring(pand, pretty_print=True).decode() if USE_LXML \
                      else __import__("xml.etree.ElementTree", fromlist=[""]).tostring(pand).decode()
            print(snippet[:2000])
        else:
            print("  [!] Geen <Pand>-elementen gevonden na stripping!")
    except Exception as exc:
        print(f"  [!] Debug-fout: {exc}")
    print()


# ==============================================================================
# Main
# ==============================================================================
def main():
    t0 = time.time()

    xml_files = sorted(Path(INPUT_DIR).glob("*.xml"))
    if not xml_files:
        print(f"[FOUT] Geen XML-bestanden gevonden in: {INPUT_DIR}")
        return

    print(f"\n{'='*65}")
    print(f"  BAG Panden filter  –  v3")
    print(f"{'='*65}")
    print(f"  Inputmap             : {INPUT_DIR}")
    print(f"  Aantal bestanden     : {len(xml_files)}")
    print(f"  Bronhouders          : {BRONHOUDERS}")
    print(f"  Status               : {STATUS!r}")
    print(f"  beginGeldigheid jaar : {BEGINGELDIGHEID}")
    print(f"  Workers              : {MAX_WORKERS}")
    print(f"{'='*65}\n")

    debug_first_file(xml_files)

    # ── Stap 0 – Parallelle inlees ────────────────────────────────────────────
    all_records = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_file, str(fp), BRONHOUDERS): fp
            for fp in xml_files
        }
        with tqdm(total=len(xml_files), desc="Verwerken",
                  unit="bestand", colour="cyan", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                fp = futures[fut]
                try:
                    records = fut.result()
                    all_records.extend(records)
                except Exception as exc:
                    tqdm.write(f"  [WAARSCHUWING] {fp.name}: {exc}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"bronhouder-hits": len(all_records)})

    if not all_records:
        print("\n[FOUT] Geen panden gevonden die voldoen aan BRONHOUDERS.")
        return

    # ── Stap 1 – GDF_Bronhouder ───────────────────────────────────────────────
    # Alle voorkomens van panden waarvan identificatie begint met een bronhoudercode.
    GDF_Bronhouder = gpd.GeoDataFrame(
        pd.DataFrame(all_records), geometry="geometry", crs=CRS
    )
    n_bronhouder = len(GDF_Bronhouder)

    # ── Stap 2 – GDF_Status ───────────────────────────────────────────────────
    # Alle voorkomens met status == 'Bouw gestart' (kan meerdere voorkomens
    # per pand bevatten, bijv. voorkomenId 2 in 2024 én voorkomenId 3 in 2025).
    GDF_Status = GDF_Bronhouder[
        GDF_Bronhouder["status"] == STATUS
    ].copy().reset_index(drop=True)
    n_status = len(GDF_Status)

    # ── Stap 3 – GDF_BeginGeldigheid  (KERNFIX) ───────────────────────────────
    #
    # We willen alleen panden waarbij de EERSTE keer dat de status
    # 'Bouw gestart' verschijnt, in 2025 valt.
    #
    # Aanpak:
    #   a) Bepaal per identificatie de vroegste beginGeldigheid in GDF_Status.
    #   b) Houd alleen identificaties waarvan die vroegste datum met '2025' begint.
    #   c) Selecteer uit GDF_Status het voorkomen met die vroegste datum
    #      (= het 'Bouw gestart'-voorkomen van 2025 zelf).
    #
    jaar_str = str(BEGINGELDIGHEID)

    # a) vroegste beginGeldigheid per pand (sortering werkt op ISO-datumstrings)
    GDF_Status["beginGeldigheid"] = GDF_Status["beginGeldigheid"].fillna("")
    eerste_bouwgestart = (
        GDF_Status.groupby("identificatie")["beginGeldigheid"]
        .min()
        .reset_index()
        .rename(columns={"beginGeldigheid": "eersteBeginGeldigheid"})
    )

    # b) filter: eerste datum moet in 2025 vallen
    eerste_in_2025 = eerste_bouwgestart[
        eerste_bouwgestart["eersteBeginGeldigheid"].str.startswith(jaar_str)
    ]["identificatie"]

    # c) selecteer alleen het bijbehorende 2025-voorkomen (laagste voorkomenId
    #    binnen 2025 voor het geval er meerdere 2025-voorkomens zijn)
    GDF_BeginGeldigheid = (
        GDF_Status[
            GDF_Status["identificatie"].isin(eerste_in_2025) &
            GDF_Status["beginGeldigheid"].str.startswith(jaar_str)
        ]
        .copy()
    )

    # Bewaar het voorkomen met de laagste voorkomenId per pand
    # (bij meerdere 2025-voorkomens willen we de vroegste)
    GDF_BeginGeldigheid["voorkomenId"] = pd.to_numeric(
        GDF_BeginGeldigheid["voorkomenId"], errors="coerce"
    )
    GDF_BeginGeldigheid = (
        GDF_BeginGeldigheid
        .sort_values("voorkomenId")
        .drop_duplicates(subset="identificatie", keep="first")
        .reset_index(drop=True)
    )
    n_geldigheid = len(GDF_BeginGeldigheid)

    # ── Statistieken ──────────────────────────────────────────────────────────
    id_counts  = GDF_BeginGeldigheid["identificatie"].value_counts()
    duplicaten = id_counts[id_counts > 1]

    print(f"\n{'='*65}")
    print(f"  RESULTATEN")
    print(f"{'='*65}")
    print(f"  [1] GDF_Bronhouder      – alle voorkomens bronhouder      : {n_bronhouder:>8,}")
    print(f"  [2] GDF_Status          – + status '{STATUS}'    : {n_status:>8,}")
    print(f"  [3] GDF_BeginGeldigheid – + EERSTE 'Bouw gestart' in {jaar_str}: {n_geldigheid:>8,}")
    print(f"\n  Unieke identificaties in eindresultaat                  : {len(id_counts):>8,}")
    print(f"  Panden met duplicaat-identificatienummer                 : {len(duplicaten):>8,}")
    if not duplicaten.empty:
        print(f"\n  Top-10 duplicaten:")
        for ident, cnt in duplicaten.head(10).items():
            print(f"    {ident}  →  {cnt}×")
    print(f"{'='*65}\n")

    if n_geldigheid == 0:
        print("[INFO] Eindresultaat is leeg – geen output geschreven.")
        print(f"\n  Status-waarden in GDF_Bronhouder (top 10):")
        print(GDF_Bronhouder["status"].value_counts().head(10).to_string())
        print(f"\n  beginGeldigheid-voorbeelden in GDF_Status:")
        if n_status > 0:
            print(GDF_Status["beginGeldigheid"].dropna().head(20).to_string())
        return

    # ── Shapefile ─────────────────────────────────────────────────────────────
    gdf_out  = GDF_BeginGeldigheid[GDF_BeginGeldigheid.geometry.notna()].copy()
    n_geom   = len(gdf_out)
    n_nogeom = n_geldigheid - n_geom

    if n_geom > 0:
        gdf_out.to_file(OUTPUT_SHP, driver="ESRI Shapefile", encoding="utf-8")
        print(f"  ✔ Shapefile opgeslagen  : {OUTPUT_SHP}  ({n_geom:,} rijen)")
    else:
        print("  ⚠ Geen panden met geometrie — shapefile niet aangemaakt.")

    if n_nogeom:
        print(f"  ⚠ {n_nogeom} panden zonder geometrie niet in shapefile.")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_cols = [c for c in GDF_BeginGeldigheid.columns if c != "geometry"]
    GDF_BeginGeldigheid[csv_cols].to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )
    print(f"  ✔ CSV opgeslagen        : {OUTPUT_CSV}  ({n_geldigheid:,} rijen)")

    elapsed = time.time() - t0
    print(f"\n  ⏱ Verwerkingstijd: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")


if __name__ == "__main__":
    main()
