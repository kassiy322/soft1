import os
import json
import sys
import time
import webbrowser
from html import escape

import fiona
import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
import pydeck as pdk
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point

try:
    import requests
except ImportError:
    requests = None

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ==========================================
# КОНСТАНТЫ
# ==========================================
METRIC_CRS = "EPSG:32637"
BBOX_EXTRA_BUFFER_M = 1000
SIMPLIFY_TOLERANCE = 1.0
DISPLAY_SIMPLIFY_TOLERANCE_M = 20.0
NODE_SNAP_M = 5.0
MIN_START_COMPONENT_NODES = 100
START_SEARCH_RADIUS_M = 3000
START_SEARCH_CANDIDATES = 1000
MAX_DISPLAY_ROADS = 50000
MAX_HEATMAP_POINTS = 50000
DATA_EXTENSIONS = {".shp", ".gpkg"}
PLACE_BUFFER_M = 2500
PLACE_QUERY_RADIUS_KM = 20
PLACE_QUERY_SLEEP_S = 1.2
PLACE_MAX_ROWS_IN_HTML = 200
DEFAULT_PLACE_LIMIT = 200
MAX_WIKIDATA_REQUESTS_PER_RUN = 100
POPULATION_CACHE_FILE = "population_cache.json"
POPULATION_HTML_START = "<!-- POPULATION_PANEL_START -->"
POPULATION_HTML_END = "<!-- POPULATION_PANEL_END -->"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
WIKIDATA_HEADERS = {"User-Agent": "ReachabilityPopulationMap/1.0"}
SETTLEMENT_TYPES = {
    "city",
    "town",
    "village",
    "hamlet",
    "isolated_dwelling",
    "locality",
}
PLACE_NAME_COLUMNS = [
    "name",
    "name:ru",
    "official_name",
    "short_name",
    "int_name",
    "name:en",
]
PLACE_ALT_NAME_COLUMNS = [
    "name:ru",
    "official_name",
    "short_name",
    "int_name",
    "name:en",
    "alt_name",
]
PLACE_TYPE_COLUMNS = ["fclass", "place", "type", "class"]
PLACE_POPULATION_COLUMNS = ["population", "pop", "population:latest"]

CAR_FCLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
}

EXCLUDE_FCLASSES = {
    "footway",
    "path",
    "cycleway",
    "steps",
    "track",
    "pedestrian",
    "service",
}

DEFAULT_SPEEDS_KMH = {
    "motorway": 100,
    "motorway_link": 60,
    "trunk": 90,
    "trunk_link": 55,
    "primary": 80,
    "primary_link": 50,
    "secondary": 70,
    "secondary_link": 45,
    "tertiary": 60,
    "tertiary_link": 40,
    "unclassified": 50,
    "residential": 40,
    "living_street": 20,
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_place_name(value):
    text = normalize_text(value)
    if not text:
        return ""

    replacements = [
        "городской округ",
        "муниципальный округ",
        "посёлок городского типа",
        "поселок городского типа",
        "рабочий поселок",
        "рабочий посёлок",
        "сельское поселение",
        "город ",
        "г. ",
        "село ",
        "деревня ",
        "д. ",
        "поселок ",
        "посёлок ",
        "пгт ",
    ]
    for item in replacements:
        text = text.replace(item, " ")

    allowed = []
    for ch in text.replace("ё", "е"):
        if ch.isalnum() or ch.isspace():
            allowed.append(ch)
        else:
            allowed.append(" ")
    return " ".join("".join(allowed).split())


def format_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        return f"{int(round(float(value))):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def parse_population_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        population = int(value)
        return population if population > 0 else None
    if isinstance(value, float):
        if np.isnan(value):
            return None
        population = int(round(value))
        return population if population > 0 else None

    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return None
    try:
        population = int(digits)
        return population if population > 0 else None
    except ValueError:
        return None


def parse_speed_kmh(value, fclass):
    if pd.notna(value):
        text = str(value).lower().strip()
        digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in text)
        parts = [part for part in digits.split() if part]
        if parts:
            try:
                speed = float(parts[0])
                if 5 <= speed <= 150:
                    return speed
            except ValueError:
                pass
    return DEFAULT_SPEEDS_KMH.get(normalize_text(fclass), 40)


def oneway_mode(value):
    text = normalize_text(value)
    if text in {"yes", "true", "1", "t", "forward"}:
        return "forward"
    if text in {"-1", "reverse", "backward"}:
        return "reverse"
    if text in {"no", "false", "0", "f", "b", "both"}:
        return "both"
    return "both"


def snap_point(x, y):
    return (
        round(x / NODE_SNAP_M) * NODE_SNAP_M,
        round(y / NODE_SNAP_M) * NODE_SNAP_M,
    )


def iter_line_parts(geometry):
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def make_wgs84_bbox(lat, lon, radius_m):
    radius_deg_lat = radius_m / 111320
    lon_scale = max(np.cos(np.radians(lat)), 0.1)
    radius_deg_lon = radius_m / (111320 * lon_scale)
    return (
        lon - radius_deg_lon,
        lat - radius_deg_lat,
        lon + radius_deg_lon,
        lat + radius_deg_lat,
    )


def find_data_sources(data_path):
    if os.path.isfile(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        return [data_path] if ext in DATA_EXTENSIONS else []

    sources = []
    for root, _, files in os.walk(data_path):
        for file_name in files:
            path = os.path.join(root, file_name)
            lower_name = file_name.lower()
            ext = os.path.splitext(lower_name)[1]
            if ext == ".gpkg":
                sources.append(path)
            elif ext == ".shp" and "roads" in lower_name:
                sources.append(path)
    return sorted(sources)


def find_place_sources(data_path):
    if os.path.isfile(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".gpkg":
            return [data_path]
        if ext == ".shp":
            parent_dir = os.path.dirname(data_path)
            siblings = []
            for file_name in os.listdir(parent_dir):
                lower_name = file_name.lower()
                if lower_name.endswith(".shp") and "place" in lower_name:
                    siblings.append(os.path.join(parent_dir, file_name))
            return sorted(siblings)
        return []

    sources = []
    for root, _, files in os.walk(data_path):
        for file_name in files:
            path = os.path.join(root, file_name)
            lower_name = file_name.lower()
            ext = os.path.splitext(lower_name)[1]
            if ext == ".gpkg":
                sources.append(path)
            elif ext == ".shp" and "place" in lower_name:
                sources.append(path)
    return sorted(sources)


def gpkg_road_layers(path):
    try:
        layers = list(fiona.listlayers(path))
    except Exception:
        return []

    road_layers = [layer for layer in layers if "road" in layer.lower()]
    if road_layers:
        return road_layers
    return layers


def gpkg_place_layers(path):
    try:
        layers = list(fiona.listlayers(path))
    except Exception:
        return []

    place_layers = [
        layer
        for layer in layers
        if "place" in layer.lower() or "settlement" in layer.lower()
    ]
    return place_layers


def read_vector_layer(path, bbox, layer_name=None):
    try:
        kwargs = {"bbox": bbox, "engine": "pyogrio"}
        if layer_name is not None:
            kwargs["layer"] = layer_name
        return gpd.read_file(path, **kwargs)
    except Exception:
        kwargs = {"bbox": bbox}
        if layer_name is not None:
            kwargs["layer"] = layer_name
        return gpd.read_file(path, **kwargs)


def read_source_roads(path, bbox):
    ext = os.path.splitext(path)[1].lower()
    layer_names = [None]
    if ext == ".gpkg":
        layer_names = gpkg_road_layers(path)

    frames = []
    for layer_name in layer_names:
        try:
            frame = read_vector_layer(path, bbox, layer_name=layer_name)
        except Exception:
            continue

        if frame.empty or "fclass" not in frame.columns:
            continue
        frames.append(frame)

    if not frames:
        return gpd.GeoDataFrame()
    return pd.concat(frames, ignore_index=True)


def read_source_places(path, bbox):
    ext = os.path.splitext(path)[1].lower()
    layer_names = [None]
    if ext == ".gpkg":
        layer_names = gpkg_place_layers(path)

    frames = []
    for layer_name in layer_names:
        try:
            frame = read_vector_layer(path, bbox, layer_name=layer_name)
        except Exception:
            continue

        if frame.empty:
            continue

        columns_lower = {col.lower() for col in frame.columns}
        has_name = any(col in frame.columns for col in PLACE_NAME_COLUMNS)
        has_place_type = any(col in frame.columns for col in PLACE_TYPE_COLUMNS)
        if not has_name and "name" not in columns_lower:
            continue
        if not has_place_type and "place" not in columns_lower and "fclass" not in columns_lower:
            continue
        frames.append(frame)

    if not frames:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)


def prepare_places_frame(frame):
    if frame.empty or "geometry" not in frame.columns:
        return gpd.GeoDataFrame()

    name_col = next((col for col in PLACE_NAME_COLUMNS if col in frame.columns), None)
    if name_col is None:
        return gpd.GeoDataFrame()

    alt_name_col = next((col for col in PLACE_ALT_NAME_COLUMNS if col in frame.columns and col != name_col), None)
    type_col = next((col for col in PLACE_TYPE_COLUMNS if col in frame.columns), None)
    pop_col = next((col for col in PLACE_POPULATION_COLUMNS if col in frame.columns), None)

    places = frame.copy()
    places = places[places.geometry.notna()].copy()
    if places.empty:
        return gpd.GeoDataFrame()

    if type_col is not None:
        places["_place_type"] = places[type_col].map(normalize_text)
        places = places[places["_place_type"].isin(SETTLEMENT_TYPES)].copy()
    else:
        places["_place_type"] = ""

    if places.empty:
        return gpd.GeoDataFrame()

    places["_name"] = places[name_col].fillna("").astype(str).str.strip()
    places = places[places["_name"] != ""].copy()
    if places.empty:
        return gpd.GeoDataFrame()

    if alt_name_col is not None:
        places["_alt_name"] = places[alt_name_col].fillna("").astype(str).str.strip()
    else:
        places["_alt_name"] = ""

    if pop_col is not None:
        places["_source_population"] = places[pop_col].map(parse_population_value)
    else:
        places["_source_population"] = None

    places["_normalized_name"] = places["_name"].map(normalize_place_name)
    places = places[places["_normalized_name"] != ""].copy()
    if places.empty:
        return gpd.GeoDataFrame()

    geom_types = places.geometry.geom_type.fillna("")
    point_mask = geom_types.isin(["Point", "MultiPoint"])
    rep_points = places.geometry.representative_point()
    if point_mask.any():
        rep_points.loc[point_mask] = places.geometry.loc[point_mask]
    places = gpd.GeoDataFrame(
        {
            "name": places["_name"],
            "alt_name": places["_alt_name"],
            "place_type": places["_place_type"],
            "source_population": places["_source_population"],
            "normalized_name": places["_normalized_name"],
            "osm_id": places["osm_id"] if "osm_id" in places.columns else pd.Series([None] * len(places), index=places.index),
        },
        geometry=rep_points,
        crs=places.crs,
    )
    return places.reset_index(drop=True)


def build_place_name_variants(row):
    variants = {
        normalize_place_name(row.get("name")),
        normalize_place_name(row.get("alt_name")),
    }
    return {item for item in variants if item}


def population_cache_path(base_path):
    output_dir = base_path if os.path.isdir(base_path) else os.path.dirname(base_path)
    return os.path.join(output_dir, POPULATION_CACHE_FILE)


def load_population_cache(base_path):
    path = population_cache_path(base_path)
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_population_cache(base_path, cache):
    path = population_cache_path(base_path)
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(cache, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


def place_cache_key(row):
    return "|".join(
        [
            normalize_place_name(row.get("name")),
            f"{float(row.get('lat')):.3f}",
            f"{float(row.get('lon')):.3f}",
        ]
    )


def fetch_population_from_wikidata(session, row):
    name_variants = build_place_name_variants(row)
    query = f"""
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?place ?placeLabel ?pop ?distance WHERE {{
      SERVICE wikibase:around {{
        ?place wdt:P625 ?location .
        bd:serviceParam wikibase:center "Point({row['lon']} {row['lat']})"^^geo:wktLiteral ;
                        wikibase:radius "{PLACE_QUERY_RADIUS_KM}" ;
                        wikibase:distance ?distance .
      }}
      ?place wdt:P17 <http://www.wikidata.org/entity/Q159> ;
             wdt:P1082 ?pop ;
             wdt:P31/wdt:P279* <http://www.wikidata.org/entity/Q486972> .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ru,en". }}
    }}
    ORDER BY ?distance
    LIMIT 10
    """

    response = session.get(
        WIKIDATA_URL,
        params={"format": "json", "query": query},
        headers=WIKIDATA_HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("results", {}).get("bindings", [])

    candidates = []
    for item in results:
        label = item.get("placeLabel", {}).get("value", "")
        population = parse_population_value(item.get("pop", {}).get("value"))
        distance_km = float(item.get("distance", {}).get("value", 999999))
        label_normalized = normalize_place_name(label)
        name_match = label_normalized in name_variants if label_normalized else False
        if population is None:
            continue
        candidates.append(
            {
                "population": population,
                "distance_km": distance_km,
                "name_match": name_match,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (0 if item["name_match"] else 1, item["distance_km"]))
    best = candidates[0]
    if best["name_match"] or best["distance_km"] <= 3:
        return best["population"]
    return None


def enrich_places_with_population(places_gdf, log_signal, cache_base_path=None, max_wikidata_requests=None):
    if max_wikidata_requests is None:
        max_wikidata_requests = MAX_WIKIDATA_REQUESTS_PER_RUN
    if places_gdf.empty:
        return places_gdf

    places = places_gdf.copy()
    places["population"] = places["source_population"]
    places["population_source"] = np.where(places["population"].notna(), "source", "")

    cache = load_population_cache(cache_base_path) if cache_base_path else {}
    cache_hits = 0
    if cache:
        for index in places[places["population"].isna()].index:
            key = place_cache_key(places.loc[index])
            cached_population = cache.get(key)
            if cached_population is not None:
                population = parse_population_value(cached_population)
                if population is not None:
                    places.at[index, "population"] = population
                    places.at[index, "population_source"] = "cache"
                    cache_hits += 1
        if cache_hits:
            log_signal.emit(f"Население из локального кэша: найдено для {cache_hits} населенных пунктов.")

    unresolved_mask = places["population"].isna()
    unresolved_total = int(unresolved_mask.sum())
    if unresolved_total == 0:
        return places

    if requests is None:
        log_signal.emit("Модуль requests не найден: население будет показано только из локальных атрибутов, если они есть.")
        return places

    session = requests.Session()
    resolved = 0
    requests_made = 0

    for index in places[unresolved_mask].index:
        if requests_made >= max_wikidata_requests:
            log_signal.emit(
                f"Лимит Wikidata за запуск достигнут: {max_wikidata_requests} запросов. Остальные значения останутся пустыми."
            )
            break

        row = places.loc[index]
        log_signal.emit(
            f"Wikidata: запрос {requests_made + 1} из {min(unresolved_total, max_wikidata_requests)} — {row['name']}"
        )
        try:
            population = fetch_population_from_wikidata(session, row)
        except Exception:
            population = None
        requests_made += 1

        if population is not None:
            places.at[index, "population"] = population
            places.at[index, "population_source"] = "wikidata"
            if cache_base_path:
                cache[place_cache_key(row)] = int(population)
            resolved += 1

        time.sleep(PLACE_QUERY_SLEEP_S)

    if cache_base_path and cache:
        save_population_cache(cache_base_path, cache)
    log_signal.emit(f"Население через Wikidata: найдено для {resolved} из {requests_made} запросов.")
    return places


def deduplicate_places(places_gdf):
    if places_gdf.empty:
        return places_gdf

    places = places_gdf.copy()
    places["_dedupe_key"] = (
        places["normalized_name"]
        + "|"
        + places.geometry.x.round(1).astype(str)
        + "|"
        + places.geometry.y.round(1).astype(str)
    )
    places["_has_population"] = places["source_population"].notna()
    places = places.sort_values(
        by=["_has_population", "normalized_name"],
        ascending=[False, True],
    )
    places = places.drop_duplicates(subset="_dedupe_key").copy()
    return places.drop(columns=["_dedupe_key", "_has_population"]).reset_index(drop=True)


def build_population_html(places_gdf, cost_mode):
    found_population_mask = places_gdf["population"].notna() if "population" in places_gdf.columns else pd.Series(dtype=bool)
    population_found_count = int(found_population_mask.sum()) if not places_gdf.empty else 0
    total_population = int(places_gdf.loc[found_population_mask, "population"].sum()) if population_found_count else 0
    metric_header = "Время, мин" if cost_mode == "time" else "По дороге, км"

    summary_html = (
        '<div id="population-summary-panel">'
        '<div class="population-title">Населенные пункты в зоне доступности</div>'
        f'<div class="population-summary-item"><span>Найдено пунктов</span><strong>{len(places_gdf)}</strong></div>'
        f'<div class="population-summary-item"><span>С найденным населением</span><strong>{population_found_count}</strong></div>'
        f'<div class="population-summary-item total"><span>Общее население</span><strong>{format_int(total_population)}</strong></div>'
        "</div>"
    )

    if places_gdf.empty:
        rows_html = '<tr><td colspan="6">Населенные пункты в достижимой зоне не найдены.</td></tr>'
    else:
        display_df = places_gdf.copy()
        display_df = display_df.sort_values(
            by=["population", "reach_sort_value", "name"],
            ascending=[False, True, True],
            na_position="last",
        ).head(PLACE_MAX_ROWS_IN_HTML)

        rows = []
        for _, row in display_df.iterrows():
            rows.append(
                "<tr>"
                f"<td>{escape(str(row['name']))}</td>"
                f"<td>{escape(str(row['place_type'] or '—'))}</td>"
                f"<td>{row['lat']:.5f}</td>"
                f"<td>{row['lon']:.5f}</td>"
                f"<td>{format_int(row.get('population'))}</td>"
                f"<td>{row['reach_display']}</td>"
                "</tr>"
            )
        rows_html = "".join(rows)

    table_html = f"""
    <div id="population-table-panel">
      <div class="population-title">Таблица населенных пунктов</div>
      <div class="population-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Населенный пункт</th>
              <th>Тип</th>
              <th>Широта</th>
              <th>Долгота</th>
              <th>Население</th>
              <th>{metric_header}</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
    """

    styles = """
    <style>
      #population-summary-panel, #population-table-panel {
        position: fixed;
        left: 16px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
        font-family: Arial, sans-serif;
        color: #1d1d1d;
      }
      #population-summary-panel {
        top: 16px;
        width: 360px;
        padding: 14px 16px;
      }
      #population-table-panel {
        bottom: 16px;
        width: min(1100px, calc(100vw - 32px));
        max-height: 42vh;
        padding: 14px 16px;
      }
      .population-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 10px;
      }
      .population-summary-item {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
      }
      .population-summary-item.total {
        border-bottom: 0;
        padding-top: 10px;
      }
      .population-summary-item.total strong {
        font-size: 22px;
      }
      .population-table-wrap {
        overflow: auto;
        max-height: calc(42vh - 48px);
      }
      #population-table-panel table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      #population-table-panel th,
      #population-table-panel td {
        padding: 8px 10px;
        text-align: left;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        background: rgba(255, 255, 255, 0.82);
      }
      #population-table-panel thead th {
        position: sticky;
        top: 0;
        background: #f3f6f8;
      }
    </style>
    """
    return styles + summary_html + table_html


def inject_population_html(output_file, html_block):
    try:
        with open(output_file, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception:
        return

    start_index = content.find(POPULATION_HTML_START)
    end_index = content.find(POPULATION_HTML_END)
    if start_index != -1 and end_index != -1 and end_index > start_index:
        content = content[:start_index] + content[end_index + len(POPULATION_HTML_END):]

    html_block = POPULATION_HTML_START + "\n" + html_block + "\n" + POPULATION_HTML_END
    if "</body>" in content:
        content = content.replace("</body>", html_block + "\n</body>", 1)
    else:
        content += html_block

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(content)


# ==========================================
# ПОТОК ДЛЯ РАСЧЕТОВ
# ==========================================
class MapWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str, float, object)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        shp_path,
        lat,
        lon,
        limit_value,
        cost_mode,
        car_only,
        use_oneway,
        show_roads,
        show_heatmap,
    ):
        super().__init__()
        self.shp_path = shp_path
        self.lat = lat
        self.lon = lon
        self.limit_value = limit_value
        self.cost_mode = cost_mode
        self.car_only = car_only
        self.use_oneway = use_oneway
        self.show_roads = show_roads
        self.show_heatmap = show_heatmap

    def run(self):
        t_start = time.time()
        try:
            if self.cost_mode == "time":
                limit_cost = self.limit_value * 60
                clip_radius_m = max(self.limit_value / 60 * 110000, 1000)
                limit_label = f"{self.limit_value:g} мин"
                cost_label = "примерному времени в пути"
            else:
                limit_cost = self.limit_value * 1000
                clip_radius_m = limit_cost
                limit_label = f"{self.limit_value:g} км"
                cost_label = "расстоянию по дорогам"

            self.log_signal.emit("Старт обработки.")
            self.log_signal.emit(f"Источник данных: {self.shp_path}")
            self.log_signal.emit(f"Режим расчета: по {cost_label}, лимит: {limit_label}")
            self.log_signal.emit(
                "Пояснение: программа строит граф дорог из данных Geofabrik. "
                "В базовом режиме каждое ребро оценивается длиной, а не прямым расстоянием по карте."
            )
            self.progress_signal.emit(5)

            self.log_signal.emit("1. Поиск и загрузка дорожных данных...")
            data_sources = find_data_sources(self.shp_path)
            if not data_sources:
                raise ValueError(
                    "Не найдены файлы дорог. Укажите файл .shp/.gpkg или папку "
                    "с распакованными архивами Geofabrik."
                )

            read_bbox = make_wgs84_bbox(self.lat, self.lon, clip_radius_m + BBOX_EXTRA_BUFFER_M)
            frames = []
            for source_path in data_sources:
                frame = read_source_roads(source_path, read_bbox)
                if not frame.empty:
                    frames.append(frame)
                    self.log_signal.emit(
                        f"Найдены дороги в файле: {os.path.basename(source_path)} ({len(frame)} строк)"
                    )

            if not frames:
                raise ValueError(
                    "В выбранных данных нет дорог рядом с заданными координатами. "
                    "Если точка не в этом федеральном округе, укажите папку, где лежат все "
                    "распакованные .gpkg архивы Geofabrik по России."
                )

            gdf = gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True),
                geometry="geometry",
                crs=frames[0].crs,
            )
            self.progress_signal.emit(15)

            required_columns = {"geometry", "fclass"}
            missing_columns = required_columns - set(gdf.columns)
            if missing_columns:
                raise ValueError(
                    "В дорожных данных не найдены обязательные поля: "
                    + ", ".join(sorted(missing_columns))
                    + ". Нужен слой roads Geofabrik."
                )

            optional_columns = [
                col for col in ["oneway", "maxspeed", "layer", "bridge", "tunnel"] if col in gdf.columns
            ]
            gdf = gdf[["geometry", "fclass", *optional_columns]].copy()
            self.log_signal.emit(
                "Найдены дополнительные поля: "
                + (", ".join(optional_columns) if optional_columns else "нет")
            )
            self.progress_signal.emit(20)

            self.log_signal.emit("2. Подготовка проекций...")
            center_metric = (
                gpd.GeoSeries([Point(self.lon, self.lat)], crs="EPSG:4326")
                .to_crs(METRIC_CRS)
                .iloc[0]
            )
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.to_crs(METRIC_CRS)
            self.progress_signal.emit(28)

            self.log_signal.emit("3. Фильтрация автомобильных дорог...")
            before_filter = len(gdf)
            fclass_normalized = gdf["fclass"].map(normalize_text)
            if self.car_only:
                gdf = gdf[fclass_normalized.isin(CAR_FCLASSES)].copy()
                self.log_signal.emit(
                    "Включен строгий режим: оставлены только основные автомобильные классы OSM."
                )
            else:
                gdf = gdf[~fclass_normalized.isin(EXCLUDE_FCLASSES)].copy()
                self.log_signal.emit(
                    "Включен мягкий режим: исключены явные пешеходные, велосипедные и служебные классы."
                )
            self.log_signal.emit(f"Дорог до фильтра: {before_filter}, после фильтра: {len(gdf)}")

            self.log_signal.emit("4. Обрезка по области расчета и упрощение геометрии...")
            bbox_buffer = center_metric.buffer(clip_radius_m + BBOX_EXTRA_BUFFER_M)
            gdf = gpd.clip(gdf, bbox_buffer)
            gdf["geometry"] = gdf["geometry"].simplify(SIMPLIFY_TOLERANCE)
            gdf = gdf[~gdf["geometry"].is_empty].copy()
            gdf = gdf.reset_index(drop=True)
            self.progress_signal.emit(38)

            if gdf.empty:
                raise ValueError(
                    "После фильтрации и обрезки не осталось дорог. "
                    "Проверьте координаты, радиус или отключите строгий автомобильный фильтр."
                )

            self.log_signal.emit(f"5. Сборка дорожного графа ({len(gdf)} линий)...")
            if self.use_oneway and "oneway" not in gdf.columns:
                self.log_signal.emit(
                    "Опция односторонних дорог включена, но поля oneway нет. "
                    "Граф будет рассчитан как двусторонний."
                )
                effective_oneway = False
            else:
                effective_oneway = self.use_oneway

            if effective_oneway:
                self.log_signal.emit(
                    "Учитываются односторонние дороги: T/yes — по направлению линии, "
                    "F/no/B — в обе стороны, -1/reverse — против направления линии."
                )
            else:
                self.log_signal.emit(
                    "Односторонние дороги не учитываются: все подходящие линии считаются проезжими в обе стороны."
                )

            coords_to_id = {}
            id_to_coords = {}
            edges = []
            weights = []
            lengths_m = []
            line_ids = []
            edge_fclasses = []
            current_id = 0

            def get_or_add(point_tuple):
                nonlocal current_id
                if point_tuple not in coords_to_id:
                    coords_to_id[point_tuple] = current_id
                    id_to_coords[current_id] = point_tuple
                    current_id += 1
                return coords_to_id[point_tuple]

            def add_edge(from_id, to_id, weight, length_m, line_id, fclass):
                if from_id == to_id or weight <= 0:
                    return
                edges.append((from_id, to_id))
                weights.append(weight)
                lengths_m.append(length_m)
                line_ids.append(line_id)
                edge_fclasses.append(fclass)

            for line_id, row in gdf.iterrows():
                fclass = row["fclass"]
                speed_kmh = parse_speed_kmh(row.get("maxspeed"), fclass)
                speed_mps = speed_kmh * 1000 / 3600
                direction = oneway_mode(row.get("oneway")) if effective_oneway else "both"

                for line in iter_line_parts(row.geometry):
                    coords = np.array(line.coords)
                    for i in range(len(coords) - 1):
                        p1 = snap_point(coords[i][0], coords[i][1])
                        p2 = snap_point(coords[i + 1][0], coords[i + 1][1])
                        id1 = get_or_add(p1)
                        id2 = get_or_add(p2)
                        length_m = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                        weight = length_m / speed_mps if self.cost_mode == "time" else length_m

                        if direction == "forward":
                            add_edge(id1, id2, weight, length_m, line_id, fclass)
                        elif direction == "reverse":
                            add_edge(id2, id1, weight, length_m, line_id, fclass)
                        else:
                            add_edge(id1, id2, weight, length_m, line_id, fclass)
                            if effective_oneway:
                                add_edge(id2, id1, weight, length_m, line_id, fclass)

            if current_id == 0 or not edges:
                raise ValueError("Граф пустой: нет пригодных ребер после обработки геометрий.")

            graph = ig.Graph(n=current_id, edges=edges, directed=effective_oneway)
            graph.es["weight"] = weights
            graph.es["length_m"] = lengths_m
            graph.es["line_id"] = line_ids
            graph.es["fclass"] = edge_fclasses
            self.log_signal.emit(f"Узлов графа: {current_id}, ребер графа: {len(edges)}")
            self.progress_signal.emit(60)

            self.log_signal.emit("6. Поиск ближайшего дорожного узла к стартовой точке...")
            node_array = np.array([id_to_coords[i] for i in range(current_id)])
            tree = cKDTree(node_array)

            components = graph.as_undirected().components()
            component_sizes = components.sizes()
            component_by_node = components.membership
            largest_component_size = max(component_sizes) if component_sizes else 0
            min_component_size = min(MIN_START_COMPONENT_NODES, largest_component_size)

            nearest_distance, nearest_idx = tree.query([center_metric.x, center_metric.y])
            nearest_idx = int(nearest_idx)
            start_vid = nearest_idx
            distance_to_network = float(nearest_distance)
            nearest_component_size = component_sizes[component_by_node[nearest_idx]]

            if nearest_component_size < min_component_size:
                search_count = min(START_SEARCH_CANDIDATES, current_id)
                candidate_distances, candidate_indices = tree.query(
                    [center_metric.x, center_metric.y],
                    k=search_count,
                )
                candidate_distances = np.atleast_1d(candidate_distances)
                candidate_indices = np.atleast_1d(candidate_indices)

                for candidate_distance, candidate_idx in zip(candidate_distances, candidate_indices):
                    candidate_idx = int(candidate_idx)
                    if candidate_distance > START_SEARCH_RADIUS_M:
                        break
                    candidate_component_size = component_sizes[component_by_node[candidate_idx]]
                    if candidate_component_size >= min_component_size:
                        start_vid = candidate_idx
                        distance_to_network = float(candidate_distance)
                        self.log_signal.emit(
                            "Ближайший дорожный кусок изолирован "
                            f"({nearest_component_size} узлов). "
                            "Старт перенесен на ближайшую связанную автосеть."
                        )
                        break

            start_component_size = component_sizes[component_by_node[start_vid]]
            self.log_signal.emit(f"Расстояние от заданной точки до выбранной дороги: {distance_to_network:.0f} м")
            self.log_signal.emit(f"Размер связной дорожной сети от старта: {start_component_size} узлов")
            self.progress_signal.emit(65)

            self.log_signal.emit("7. Расчет достижимости алгоритмом Дейкстры...")
            distances = np.array(graph.distances(source=start_vid, weights="weight", mode="out")[0])
            finite_distances = np.isfinite(distances)
            reachable_ids = np.where(finite_distances & (distances <= limit_cost))[0]
            self.progress_signal.emit(80)

            if len(reachable_ids) == 0:
                raise ValueError(
                    "Нет достижимых узлов в заданном лимите. "
                    "Проверьте координаты — возможно, точка далеко от дорожной сети."
                )

            self.log_signal.emit("8. Подготовка данных для карты...")
            xs = node_array[reachable_ids, 0]
            ys = node_array[reachable_ids, 1]
            nodes_gdf = gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in zip(xs, ys)],
                crs=METRIC_CRS,
            ).to_crs("EPSG:4326")

            chart_data = pd.DataFrame(
                {
                    "lon": nodes_gdf.geometry.x,
                    "lat": nodes_gdf.geometry.y,
                }
            )
            heatmap_total = len(chart_data)
            if heatmap_total > MAX_HEATMAP_POINTS:
                chart_data = chart_data.sample(n=MAX_HEATMAP_POINTS, random_state=1)
                self.log_signal.emit(
                    f"Для размера HTML тепловая карта уменьшена: "
                    f"{heatmap_total} точек -> {len(chart_data)} точек."
                )

            reachable_line_ids = set()
            if self.show_roads:
                for edge in graph.es:
                    source, target = edge.tuple
                    if not np.isfinite(distances[source]):
                        continue
                    if distances[source] + edge["weight"] <= limit_cost:
                        reachable_line_ids.add(edge["line_id"])

            roads_path_data = None
            reachable_roads_total = len(reachable_line_ids)
            if reachable_line_ids:
                roads_gdf = gdf.loc[sorted(reachable_line_ids), ["geometry", "fclass"]].copy()
                roads_gdf["geometry"] = roads_gdf["geometry"].simplify(DISPLAY_SIMPLIFY_TOLERANCE_M)
                roads_gdf = roads_gdf[~roads_gdf["geometry"].is_empty].copy()
                if len(roads_gdf) > MAX_DISPLAY_ROADS:
                    priority = {
                        "motorway",
                        "motorway_link",
                        "trunk",
                        "trunk_link",
                        "primary",
                        "primary_link",
                        "secondary",
                        "secondary_link",
                    }
                    priority_mask = roads_gdf["fclass"].map(normalize_text).isin(priority)
                    priority_roads = roads_gdf[priority_mask]
                    other_roads = roads_gdf[~priority_mask]
                    remaining = max(MAX_DISPLAY_ROADS - len(priority_roads), 0)
                    if len(priority_roads) >= MAX_DISPLAY_ROADS:
                        roads_gdf = priority_roads.sample(n=MAX_DISPLAY_ROADS, random_state=2)
                    elif len(other_roads) > remaining:
                        roads_gdf = pd.concat(
                            [
                                priority_roads,
                                other_roads.sample(n=remaining, random_state=2),
                            ]
                        )
                    self.log_signal.emit(
                        f"Для размера HTML дорожный слой уменьшен: "
                        f"{reachable_roads_total} линий -> {len(roads_gdf)} линий."
                    )

                roads_gdf = roads_gdf.to_crs("EPSG:4326")
                path_rows = []
                for _, row in roads_gdf.iterrows():
                    for line in iter_line_parts(row.geometry):
                        coords = [[float(x), float(y)] for x, y in line.coords]
                        if len(coords) >= 2:
                            path_rows.append({"path": coords, "fclass": row["fclass"]})
                roads_path_data = pd.DataFrame(path_rows)

            self.log_signal.emit(f"Достижимых узлов в расчете: {len(reachable_ids)}")
            self.log_signal.emit(f"Достижимых исходных линий дорог: {reachable_roads_total}")
            self.progress_signal.emit(88)

            self.progress_signal.emit(92)

            self.log_signal.emit("9. Генерация WebGL карты...")
            layers = []
            if roads_path_data is not None and not roads_path_data.empty:
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        roads_path_data,
                        get_path="path",
                        get_color=[0, 115, 180, 210],
                        width_min_pixels=2,
                        get_width=3,
                        pickable=True,
                    )
                )

            if self.show_heatmap:
                layers.append(
                    pdk.Layer(
                        "HeatmapLayer",
                        chart_data,
                        get_position=["lon", "lat"],
                        auto_highlight=True,
                        radius_pixels=20,
                        intensity=1.0,
                        threshold=0.1,
                        color_range=[
                            [0, 0, 255, 100],
                            [0, 180, 0, 150],
                            [255, 210, 0, 200],
                            [255, 0, 0, 250],
                        ],
                    )
                )

            center_data = pd.DataFrame(
                {
                    "lon": [self.lon],
                    "lat": [self.lat],
                    "label": ["Старт"],
                }
            )
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    center_data,
                    get_position=["lon", "lat"],
                    get_fill_color=[220, 0, 0, 255],
                    get_line_color=[255, 255, 255, 255],
                    stroked=True,
                    line_width_min_pixels=3,
                    get_radius=300,
                    radius_min_pixels=8,
                )
            )
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    center_data,
                    get_position=["lon", "lat"],
                    get_text="label",
                    get_size=20,
                    get_color=[0, 0, 0, 255],
                    get_background_color=[255, 255, 255, 210],
                    background=True,
                    get_alignment_baseline="'bottom'",
                    get_pixel_offset=[0, -15],
                )
            )

            deck = pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=self.lat,
                    longitude=self.lon,
                    zoom=8,
                    pitch=0,
                    bearing=0,
                ),
                map_style=pdk.map_styles.CARTO_LIGHT,
                tooltip={"text": "Класс дороги: {fclass}"},
            )

            output_dir = self.shp_path if os.path.isdir(self.shp_path) else os.path.dirname(self.shp_path)
            output_file = os.path.join(output_dir, "routes_map.html")
            deck.to_html(output_file)

            self.progress_signal.emit(100)
            elapsed = round(time.time() - t_start, 1)
            reachability_context = {
                "shp_path": self.shp_path,
                "output_file": output_file,
                "read_bbox": read_bbox,
                "cost_mode": self.cost_mode,
                "reachable_node_coords": node_array[reachable_ids].copy(),
                "reachable_costs": distances[reachable_ids].copy(),
            }
            self.finished_signal.emit(output_file, elapsed, reachability_context)

        except Exception as exc:
            self.error_signal.emit(str(exc))


class PopulationWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str, float)
    error_signal = pyqtSignal(str)

    def __init__(self, reachability_context, max_places):
        super().__init__()
        self.context = reachability_context
        self.max_places = max_places

    def run(self):
        t_start = time.time()
        try:
            shp_path = self.context["shp_path"]
            read_bbox = self.context["read_bbox"]
            cost_mode = self.context["cost_mode"]
            output_file = self.context["output_file"]
            reachable_node_coords = self.context["reachable_node_coords"]
            reachable_costs = self.context["reachable_costs"]

            self.log_signal.emit("Старт поиска населенных пунктов по готовой зоне доступности.")
            if len(reachable_node_coords) == 0:
                raise ValueError("Нет сохраненных достижимых узлов. Сначала сгенерируйте карту.")

            place_sources = find_place_sources(shp_path)
            if not place_sources:
                raise ValueError("Слои населенных пунктов не найдены рядом с исходными данными.")

            self.progress_signal.emit(5)
            frames = []
            for source_path in place_sources:
                frame = read_source_places(source_path, read_bbox)
                if not frame.empty:
                    frames.append(frame)
                    self.log_signal.emit(
                        f"Найдены населенные пункты в файле: {os.path.basename(source_path)} ({len(frame)} строк)"
                    )

            if not frames:
                raise ValueError("Подходящие слои places/settlements не найдены в выбранных источниках.")

            places_raw = gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True),
                geometry="geometry",
                crs=frames[0].crs,
            )
            places_prepared = prepare_places_frame(places_raw)
            if places_prepared.empty:
                raise ValueError("В слоях населенных пунктов не нашлось подходящих объектов city/town/village/hamlet.")

            if places_prepared.crs is None:
                places_prepared = places_prepared.set_crs("EPSG:4326")
            elif places_prepared.crs.to_epsg() != 4326:
                places_prepared = places_prepared.to_crs("EPSG:4326")
            places_metric = places_prepared.to_crs(METRIC_CRS)
            self.progress_signal.emit(25)

            tree = cKDTree(reachable_node_coords)
            place_coords = np.column_stack(
                [places_metric.geometry.x.to_numpy(), places_metric.geometry.y.to_numpy()]
            )
            nearest_distances, nearest_indices = tree.query(place_coords, k=1)
            nearest_distances = np.atleast_1d(nearest_distances)
            nearest_indices = np.atleast_1d(nearest_indices).astype(int)

            places_metric["distance_to_network_m"] = nearest_distances
            places_metric["reach_cost"] = reachable_costs[nearest_indices]
            places_metric = places_metric[
                np.isfinite(places_metric["reach_cost"])
                & (places_metric["distance_to_network_m"] <= PLACE_BUFFER_M)
            ].copy()
            places_metric = deduplicate_places(places_metric)
            found_total = len(places_metric)
            self.log_signal.emit(f"Населенных пунктов в достижимой зоне найдено: {found_total}")

            if places_metric.empty:
                places_for_html = pd.DataFrame(
                    columns=["name", "place_type", "lat", "lon", "population", "reach_display", "reach_sort_value"]
                )
            else:
                places_metric = places_metric.sort_values(
                    by=["reach_cost", "distance_to_network_m", "name"],
                    ascending=[True, True, True],
                ).head(self.max_places)
                self.log_signal.emit(
                    f"Будет обработано населенных пунктов: {len(places_metric)} из {found_total}. Лимит: {self.max_places}."
                )

                if cost_mode == "time":
                    places_metric["reach_display"] = places_metric["reach_cost"].map(lambda value: f"{value / 60:.1f}")
                else:
                    places_metric["reach_display"] = places_metric["reach_cost"].map(lambda value: f"{value / 1000:.1f}")

                places_wgs84 = places_metric.to_crs("EPSG:4326").copy()
                places_wgs84["lat"] = places_wgs84.geometry.y
                places_wgs84["lon"] = places_wgs84.geometry.x
                self.progress_signal.emit(45)

                places_wgs84 = enrich_places_with_population(
                    places_wgs84,
                    self.log_signal,
                    cache_base_path=shp_path,
                    max_wikidata_requests=self.max_places,
                )
                places_for_html = pd.DataFrame(
                    {
                        "name": places_wgs84["name"],
                        "place_type": places_wgs84["place_type"],
                        "lat": places_wgs84["lat"],
                        "lon": places_wgs84["lon"],
                        "population": places_wgs84["population"],
                        "reach_display": places_wgs84["reach_display"],
                        "reach_sort_value": places_wgs84["reach_cost"],
                    }
                )

            inject_population_html(output_file, build_population_html(places_for_html, cost_mode))
            self.progress_signal.emit(100)
            elapsed = round(time.time() - t_start, 1)
            self.finished_signal.emit(output_file, elapsed)

        except Exception as exc:
            self.error_signal.emit(str(exc))


# ==========================================
# ИНТЕРФЕЙС
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор карты автомобильной доступности")
        self.resize(860, 720)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Путь к .shp/.gpkg файлу или папке с архивами Geofabrik...")
        self.btn_browse = QPushButton("Файл")
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_browse_folder = QPushButton("Папка РФ")
        self.btn_browse_folder.clicked.connect(self.browse_folder)
        file_layout.addWidget(QLabel("Данные дорог:"))
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.btn_browse)
        file_layout.addWidget(self.btn_browse_folder)
        layout.addLayout(file_layout)

        coord_group = QGroupBox("Исходная точка и лимит")
        coord_layout = QGridLayout(coord_group)
        self.lat_input = QLineEdit()
        self.lon_input = QLineEdit()
        self.limit_input = QLineEdit("100")
        self.lat_input.setPlaceholderText("например 55.7558")
        self.lon_input.setPlaceholderText("например 37.6173")
        self.limit_input.setPlaceholderText("например 100")

        self.cost_mode_combo = QComboBox()
        self.cost_mode_combo.addItem("По расстоянию дорог, км", "distance")
        self.cost_mode_combo.addItem("По примерному времени, минут", "time")

        coord_layout.addWidget(QLabel("Широта:"), 0, 0)
        coord_layout.addWidget(self.lat_input, 0, 1)
        coord_layout.addWidget(QLabel("Долгота:"), 0, 2)
        coord_layout.addWidget(self.lon_input, 0, 3)
        coord_layout.addWidget(QLabel("Лимит:"), 1, 0)
        coord_layout.addWidget(self.limit_input, 1, 1)
        coord_layout.addWidget(QLabel("Режим:"), 1, 2)
        coord_layout.addWidget(self.cost_mode_combo, 1, 3)
        layout.addWidget(coord_group)

        options_group = QGroupBox("Опции точности")
        options_layout = QGridLayout(options_group)
        self.car_only_checkbox = QCheckBox("Оставлять только автомобильные классы дорог")
        self.car_only_checkbox.setChecked(True)
        self.oneway_checkbox = QCheckBox("Учитывать односторонние дороги, если есть поле oneway")
        self.oneway_checkbox.setChecked(True)
        self.roads_checkbox = QCheckBox("Показывать достижимые дороги линиями")
        self.roads_checkbox.setChecked(True)
        self.heatmap_checkbox = QCheckBox("Показывать тепловую карту узлов")
        self.heatmap_checkbox.setChecked(True)

        options_layout.addWidget(self.car_only_checkbox, 0, 0)
        options_layout.addWidget(self.oneway_checkbox, 0, 1)
        options_layout.addWidget(self.roads_checkbox, 1, 0)
        options_layout.addWidget(self.heatmap_checkbox, 1, 1)
        layout.addWidget(options_group)

        population_group = QGroupBox("Население")
        population_layout = QGridLayout(population_group)
        self.population_limit_input = QLineEdit(str(DEFAULT_PLACE_LIMIT))
        self.population_limit_input.setPlaceholderText("например 200")
        population_layout.addWidget(QLabel("Обработать пунктов, максимум:"), 0, 0)
        population_layout.addWidget(self.population_limit_input, 0, 1)
        layout.addWidget(population_group)

        explanation = QTextEdit()
        explanation.setReadOnly(True)
        explanation.setMaximumHeight(150)
        explanation.setPlainText(
            "Что именно считает программа:\n"
            "Программа берет линии дорог из SHP или GPKG Geofabrik, превращает их в граф и считает, "
            "куда можно доехать от заданных координат по дорожной сети.\n\n"
            "Для расчета по всей России можно указать папку, где лежат распакованные .gpkg архивы "
            "федеральных округов. Программа сама возьмет дороги из файлов рядом с заданной точкой.\n\n"
            "Почему появились опции точности:\n"
            "Если не учитывать поле oneway, все подходящие линии считаются двусторонними. "
            "Для грубой оценки это работает, но в городе и на развязках результат может быть слишком оптимистичным. "
            "Строгий автомобильный фильтр убирает пешеходные, велосипедные, служебные и грунтовые классы, "
            "чтобы карта лучше отвечала вопросу: куда реально можно доехать на авто.\n\n"
            "Режим по расстоянию считает километры по дороге. "
            "Режим по времени использует maxspeed из SHP, а если его нет — примерные скорости по классу дороги.\n\n"
            "Важно про размер карты:\n"
            "Расчет выполняется по полному графу, но HTML-карта для браузера облегчается: "
            "при большом радиусе программа ограничивает число отображаемых линий и точек, "
            "чтобы файл открывался нормально."
        )
        layout.addWidget(explanation)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Сгенерировать карту")
        self.btn_start.setStyleSheet("font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_processing)

        self.btn_open_folder = QPushButton("Открыть папку")
        self.btn_open_folder.setStyleSheet("padding: 10px;")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)

        self.btn_population = QPushButton("Найти население в зоне")
        self.btn_population.setStyleSheet("padding: 10px;")
        self.btn_population.clicked.connect(self.start_population_processing)
        self.btn_population.setEnabled(False)

        self.btn_contact = QPushButton("Связаться")
        self.btn_contact.setStyleSheet(
            "background-color: red; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_contact.clicked.connect(self.open_contact)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_population)
        btn_layout.addWidget(self.btn_open_folder)
        btn_layout.addWidget(self.btn_contact)
        layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self.log_output)

        self.worker = None
        self.population_worker = None
        self.output_path = None
        self.reachability_context = None

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл дорог Geofabrik",
            "",
            "Файлы Geofabrik (*.shp *.gpkg)",
        )
        if file_name:
            self.file_input.setText(file_name)

    def browse_folder(self):
        folder_name = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с распакованными архивами Geofabrik",
            "",
        )
        if folder_name:
            self.file_input.setText(folder_name)

    def append_log(self, text):
        self.log_output.append(text)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def process_finished(self, output_path, elapsed, reachability_context):
        self.output_path = output_path
        self.reachability_context = reachability_context
        self.append_log(f"\nКарта сохранена: {output_path}")
        self.append_log(f"Время выполнения: {elapsed} сек.")
        self.btn_start.setEnabled(True)
        self.btn_open_folder.setEnabled(True)
        self.btn_population.setEnabled(True)
        webbrowser.open(f"file://{output_path}")

    def process_error(self, error_msg):
        self.append_log(f"\nОШИБКА: {error_msg}")
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{error_msg}")
        self.btn_start.setEnabled(True)
        self.btn_population.setEnabled(self.reachability_context is not None)
        self.progress_bar.setValue(0)

    def population_finished(self, output_path, elapsed):
        self.output_path = output_path
        self.append_log(f"\nТаблица населения добавлена в карту: {output_path}")
        self.append_log(f"Время обработки населения: {elapsed} сек.")
        self.btn_start.setEnabled(True)
        self.btn_population.setEnabled(True)
        self.btn_open_folder.setEnabled(True)
        webbrowser.open(f"file://{output_path}")

    def population_error(self, error_msg):
        self.append_log(f"\nОШИБКА НАСЕЛЕНИЯ: {error_msg}")
        QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обработке населения:\n{error_msg}")
        self.btn_start.setEnabled(True)
        self.btn_population.setEnabled(self.reachability_context is not None)
        self.btn_open_folder.setEnabled(self.output_path is not None)
        self.progress_bar.setValue(0)

    def start_population_processing(self):
        if self.reachability_context is None:
            QMessageBox.warning(self, "Внимание", "Сначала сгенерируйте карту.")
            return

        try:
            max_places = int(self.population_limit_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Внимание", "Лимит населенных пунктов должен быть целым числом")
            return

        if max_places <= 0:
            QMessageBox.warning(self, "Внимание", "Лимит населенных пунктов должен быть больше нуля")
            return

        self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_population.setEnabled(False)
        self.btn_open_folder.setEnabled(False)

        self.population_worker = PopulationWorker(
            reachability_context=self.reachability_context,
            max_places=max_places,
        )
        self.population_worker.log_signal.connect(self.append_log)
        self.population_worker.progress_signal.connect(self.update_progress)
        self.population_worker.finished_signal.connect(self.population_finished)
        self.population_worker.error_signal.connect(self.population_error)
        self.population_worker.start()

    def start_processing(self):
        shp_path = self.file_input.text().strip()
        lat_text = self.lat_input.text().strip().replace(",", ".")
        lon_text = self.lon_input.text().strip().replace(",", ".")
        limit_text = self.limit_input.text().strip().replace(",", ".")

        if not shp_path or not os.path.exists(shp_path):
            QMessageBox.warning(self, "Внимание", "Укажите путь к файлу .shp/.gpkg или папке Geofabrik")
            return

        if os.path.isfile(shp_path) and os.path.splitext(shp_path)[1].lower() not in DATA_EXTENSIONS:
            QMessageBox.warning(self, "Внимание", "Поддерживаются только файлы .shp и .gpkg")
            return

        try:
            lat = float(lat_text)
            lon = float(lon_text)
            limit_value = float(limit_text)
        except ValueError:
            QMessageBox.warning(self, "Внимание", "Координаты и лимит должны быть числами")
            return

        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            QMessageBox.warning(self, "Внимание", "Проверьте широту и долготу")
            return
        if limit_value <= 0:
            QMessageBox.warning(self, "Внимание", "Лимит должен быть больше нуля")
            return

        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_open_folder.setEnabled(False)
        self.btn_population.setEnabled(False)
        self.output_path = None
        self.reachability_context = None

        self.worker = MapWorker(
            shp_path=shp_path,
            lat=lat,
            lon=lon,
            limit_value=limit_value,
            cost_mode=self.cost_mode_combo.currentData(),
            car_only=self.car_only_checkbox.isChecked(),
            use_oneway=self.oneway_checkbox.isChecked(),
            show_roads=self.roads_checkbox.isChecked(),
            show_heatmap=self.heatmap_checkbox.isChecked(),
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.process_finished)
        self.worker.error_signal.connect(self.process_error)
        self.worker.start()

    def open_output_folder(self):
        if self.output_path:
            folder = os.path.dirname(self.output_path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def open_contact(self):
        QDesktopServices.openUrl(QUrl("https://kwork.ru/user/reload_marketing"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
