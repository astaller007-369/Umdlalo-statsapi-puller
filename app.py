"""
TheStatsAPI Season Stats Exporter
=================================

A single-file Streamlit app that pulls the requested stat metrics for every
past and upcoming (unplayed) match, for a chosen set of leagues and a chosen
number of seasons (1-5, counting back from the current season), into one CSV.

Data source: TheStatsAPI (https://www.thestatsapi.com), base URL
https://api.thestatsapi.com/api. Endpoints/field names below come from the
live docs at https://www.thestatsapi.com/docs/llms.txt.

RUN LOCALLY
    pip install -r requirements.txt
    streamlit run app.py

DEPLOY (GitHub + Streamlit Community Cloud)
    1. Push this file + requirements.txt to a GitHub repo.
    2. On https://share.streamlit.io, point it at the repo, entry point app.py.
    3. Don't put your API key in secrets/the repo - the sidebar asks for it
       every session by design, so it's never committed.

CSV COLUMNS
    competition   -> "<Country> <League name>", e.g. "England Premier League"
    season        -> e.g. "24/25"
    matchday, date (YYYY-MM-DD, ascending), status, home_team, away_team,
    goals_home, goals_away, then one <metric>_home/<metric>_away pair per
    requested stat. Blank for unplayed matches (and, occasionally, for
    minor competitions TheStatsAPI itself has no coverage for on a field).

GAP-FILLING / PERSISTENCE
    Every fetched row is merged against a local SQLite file
    (./data/match_stats_cache.db) keyed by (competition, date, home_team,
    away_team). Blank cells are backfilled from a previous run's values for
    that same match, and the more-complete row is saved back to the cache.
    This file persists across app restarts as long as the disk persists
    (on Streamlit Community Cloud: for the life of the running instance,
    wiped on redeploy - mount a persistent volume if you need more).

RATE LIMITS
    TheStatsAPI plans cap requests/minute (Starter 30, Growth 60, Scale
    higher). Set "Max requests / minute" in the sidebar's Advanced section
    to match your plan; the app enforces it client-side and auto-retries
    on 429s. A full multi-league, multi-season run can be thousands of
    calls (one match-list call per league/season, one /stats call per
    finished match) - budget your plan's monthly quota accordingly.

LEAGUE MATCHING
    The ~69 target leagues are stored below as (country, name) hints and
    fuzzy-matched against your account's live /football/competitions list
    each session (IDs are never hard-coded, since coverage varies by
    plan/account). Anything that can't be confidently matched is skipped
    and reported in a warning rather than silently mismatched - if that
    happens a lot, tweak the hint tuple for that league.
"""

from __future__ import annotations

import difflib
import io
import re
import sqlite3
import threading
import time
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ============================================================================
# 1. TARGET LEAGUES
# ============================================================================

TARGET_LEAGUES_RAW = [
    ("Austria", "2. Liga"),
    ("Belgium", "Pro League"),
    ("Colombia", "Primera A Apertura"),
    ("Chile", "Primera Division"),
    ("Czechia", "First League"),
    ("Ecuador", "LigaPro Serie A"),
    ("England", "League Two"),
    ("Germany", "3. Liga"),
    ("Finland", "Veikkausliiga"),
    ("Finland", "Ykkosliiga"),
    ("Ireland", "Premier Division"),
    ("India", "Indian Super League"),
    ("Indonesia", "Liga 1"),
    ("Norway", "1st Division"),
    ("Portugal", "Liga Portugal Betclic"),
    ("Poland", "Ekstraklasa"),
    ("Romania", "Super Liga"),
    ("South Africa", "Premier Division"),
    ("Serbia", "Mozzart Bet Superliga"),
    ("Turkey", "Trendyol 1. Lig"),
    ("Belgium", "Challenger Pro League"),
    ("Switzerland", "Challenge League"),
    ("Italy", "Serie B"),
    ("Italy", "Serie A"),
    ("Norway", "Eliteserien"),
    ("Brazil", "Brasileirao Serie B"),
    ("Brazil", "Brasileirao Serie A"),
    ("Germany", "Bundesliga"),
    ("Austria", "Bundesliga"),
    ("Iceland", "Besta deild karla"),
    ("Germany", "2. Bundesliga"),
    ("Hungary", "NB I"),
    ("Spain", "LaLiga 2"),
    ("Spain", "LaLiga"),
    ("Denmark", "Superliga"),
    ("Portugal", "Liga Portugal 2"),
    ("Bulgaria", "Parva Liga"),
    ("Slovakia", "Nike Liga"),
    ("Slovenia", "PrvaLiga"),
    ("Tunisia", "Ligue Professionnelle 1"),
    ("Turkey", "Trendyol Super Lig"),
    ("Switzerland", "Super League"),
    ("Denmark", "1. Division"),
    ("Sweden", "Allsvenskan"),
    ("Australia", "A-League Men"),
    ("Canada", "Canadian Premier League"),
    ("England", "Championship"),
    ("Croatia", "HNL"),
    ("Netherlands", "Eerste Divisie"),
    ("Netherlands", "Eredivisie"),
    ("England", "League One"),
    ("South Korea", "K League 1"),
    ("Japan", "J1 League"),
    ("USA", "MLS"),
    ("France", "Ligue 1"),
    ("France", "Ligue 2"),
    ("Poland", "Puchar Polski"),
    ("England", "Premier League"),
    ("Scotland", "Championship"),
    ("Russia", "Premier League"),
    ("Scotland", "Premiership"),
    ("Sweden", "Superettan"),
    ("USA", "USL Championship"),
    ("Ukraine", "Premier League"),
    ("Australia", "NPL Victoria"),
    ("Cyprus", "First Division"),
    ("Scotland", "League One"),
    ("Slovakia", "Slovensky Pohar"),
    ("Greece", "Stoiximan Super League"),
]

# De-duplicate while preserving order (a few leagues were listed twice).
_seen = set()
TARGET_LEAGUES = []
for _country, _name in TARGET_LEAGUES_RAW:
    _key = (_country or "").strip().lower(), _name.strip().lower()
    if _key not in _seen:
        _seen.add(_key)
        TARGET_LEAGUES.append((_country, _name))


def display_name(country, name):
    return f"{country} - {name}" if country else name


# ============================================================================
# 2. STAT METRICS
# ============================================================================
# Maps requested stat names to their location in the response of
# GET /football/matches/{match_id}/stats. That endpoint groups fields under
# categories: overview, shots, attack, passes, duels, defending, goalkeeping,
# np_expected_goals. Each field value looks like:
#   {"all": {"home": X, "away": Y}, "first_half": {...}, "second_half": {...}}
# We always take the "all" (full match) home/away values.
# "goals" isn't part of the stats endpoint - it comes from the match record's
# score, handled separately below.

METRIC_FIELDS = [
    ("goalkeeper_saves", "overview"),
    ("big_chances", "overview"),
    ("shots_on_target", "overview"),
    ("touches_in_penalty_area", "attack"),
    ("corner_kicks", "overview"),
    ("fouled_in_final_third", "attack"),
    ("accurate_crosses", "passes"),
    ("aerial_duels_percentage", "duels"),
    ("accurate_long_balls", "passes"),
    ("final_third_entries", "passes"),
    ("dribbles_percentage", "duels"),
    ("tackles_won_percentage", "defending"),
    ("ground_duels_percentage", "duels"),
]


def extract_metric(stats_payload, field, category):
    if not stats_payload:
        return None, None
    try:
        data = stats_payload.get("data", {})
        cat = data.get(category) or {}
        item = cat.get(field)
        if not item:
            return None, None
        all_vals = item.get("all") or {}
        return all_vals.get("home"), all_vals.get("away")
    except (AttributeError, TypeError):
        return None, None


# ============================================================================
# 3. THESTATSAPI CLIENT (auth, pagination, rate limiting, retries)
# ============================================================================

BASE_URL = "https://api.thestatsapi.com/api"


class ApiError(Exception):
    def __init__(self, status_code, code, message):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"{status_code} {code}: {message}")


class RateLimiter:
    """Thread-safe sliding-window limiter: at most `max_per_minute` calls in any 60s window."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max(1, max_per_minute)
        self._lock = threading.Lock()
        self._timestamps: deque = deque()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = 60 - (now - self._timestamps[0]) + 0.02
            time.sleep(max(sleep_for, 0.02))


class StatsApiClient:
    def __init__(self, api_key: str, requests_per_minute: int = 55, max_retries: int = 4, timeout: int = 20):
        self.api_key = api_key
        self.limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        attempt = 0
        while True:
            self.limiter.acquire()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ApiError(0, "network_error", str(exc))
                time.sleep(min(2 ** attempt, 20))
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise ApiError(429, "rate_limited", "Rate limit exceeded after retries")
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
                time.sleep(wait)
                continue

            if 500 <= resp.status_code < 600:
                attempt += 1
                if attempt > self.max_retries:
                    raise ApiError(resp.status_code, "server_error", resp.text[:200])
                time.sleep(min(2 ** attempt, 20))
                continue

            try:
                body = resp.json()
                err = body.get("error", {})
                code = err.get("code", "unknown_error")
                message = err.get("message", resp.text[:200])
            except ValueError:
                code = "unknown_error"
                message = resp.text[:200]
            raise ApiError(resp.status_code, code, message)

    def check_health(self) -> bool:
        try:
            resp = self.session.get(f"{BASE_URL}/health", timeout=self.timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_all_competitions(self) -> list[dict]:
        results = []
        page = 1
        while True:
            payload = self._get("/football/competitions", {"page": page, "per_page": 100})
            results.extend(payload.get("data", []))
            meta = payload.get("meta", {})
            if page >= meta.get("total_pages", 1):
                break
            page += 1
        return results

    def list_seasons(self, competition_id: str) -> list[dict]:
        payload = self._get(f"/football/competitions/{competition_id}/seasons")
        return payload.get("data", [])

    def list_all_matches(self, competition_id: str, season_id: str) -> list[dict]:
        results = []
        page = 1
        while True:
            payload = self._get(
                "/football/matches",
                {"competition_id": competition_id, "season_id": season_id, "page": page, "per_page": 100},
            )
            results.extend(payload.get("data", []))
            meta = payload.get("meta", {})
            if page >= meta.get("total_pages", 1):
                break
            page += 1
        return results

    def get_match_stats(self, match_id: str) -> dict:
        return self._get(f"/football/matches/{match_id}/stats")


# ============================================================================
# 4. PERSISTENT SQLITE CACHE (gap-filling)
# ============================================================================

DB_PATH = Path(__file__).resolve().parent / "data" / "match_stats_cache.db"
KEY_COLUMNS = ["competition", "date", "home_team", "away_team"]


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_table(conn, columns):
    cols_sql = ", ".join(f'"{c}" TEXT' for c in columns if c not in KEY_COLUMNS)
    key_sql = ", ".join(f'"{c}" TEXT' for c in KEY_COLUMNS)
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS match_stats (
                {key_sql}, {cols_sql},
                PRIMARY KEY ({", ".join(f'"{c}"' for c in KEY_COLUMNS)})
            )"""
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(match_stats)")}
    for c in columns:
        if c not in existing:
            conn.execute(f'ALTER TABLE match_stats ADD COLUMN "{c}" TEXT')
    conn.commit()


def merge_with_cache(df: pd.DataFrame) -> pd.DataFrame:
    """Fill null cells in df from the cache, then upsert the merged rows back into it."""
    if df.empty:
        return df

    value_columns = [c for c in df.columns if c not in KEY_COLUMNS]
    conn = _connect()
    try:
        _ensure_table(conn, list(df.columns))

        keys = df[KEY_COLUMNS].drop_duplicates()
        cached_rows = []
        for _, row in keys.iterrows():
            cur = conn.execute(
                "SELECT * FROM match_stats WHERE competition = ? AND date = ? AND home_team = ? AND away_team = ?",
                tuple(row[k] for k in KEY_COLUMNS),
            )
            r = cur.fetchone()
            if r is not None:
                colnames = [d[0] for d in cur.description]
                cached_rows.append(dict(zip(colnames, r)))

        if cached_rows:
            cache_df = pd.DataFrame(cached_rows).set_index(KEY_COLUMNS)
        else:
            cache_df = pd.DataFrame(columns=KEY_COLUMNS + value_columns).set_index(KEY_COLUMNS)

        df = df.set_index(KEY_COLUMNS)
        for col in value_columns:
            if col not in df.columns:
                continue
            if col in cache_df.columns:
                cached_series = cache_df[col].reindex(df.index)
                df[col] = df[col].where(df[col].notna() & (df[col] != ""), cached_series)
        df = df.reset_index()

        records = df.to_dict("records")
        placeholders = ", ".join("?" for _ in df.columns)
        col_list = ", ".join(f'"{c}"' for c in df.columns)
        conn.executemany(
            f"INSERT OR REPLACE INTO match_stats ({col_list}) VALUES ({placeholders})",
            [tuple(None if pd.isna(v) else v for v in rec.values()) for rec in records],
        )
        conn.commit()
    finally:
        conn.close()

    return df


def cache_row_count() -> int:
    if not DB_PATH.exists():
        return 0
    conn = _connect()
    try:
        cur = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='match_stats'")
        if cur.fetchone()[0] == 0:
            return 0
        return conn.execute("SELECT COUNT(*) FROM match_stats").fetchone()[0]
    finally:
        conn.close()


# ============================================================================
# 5. LEAGUE MATCHING + ROW-BUILDING HELPERS
# ============================================================================


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_candidate(country_hint, name_hint, comp):
    name_ratio = difflib.SequenceMatcher(None, normalize(name_hint), normalize(comp.get("name", ""))).ratio()
    if country_hint:
        country_ratio = difflib.SequenceMatcher(
            None, normalize(country_hint), normalize(comp.get("country", ""))
        ).ratio()
        if country_ratio < 0.5:
            return 0.55 * name_ratio
        return 0.7 * name_ratio + 0.3 * country_ratio
    return name_ratio


def resolve_leagues(competitions: list[dict]) -> dict:
    resolved = {}
    for country_hint, name_hint in TARGET_LEAGUES:
        best, best_score = None, 0.0
        for comp in competitions:
            s = score_candidate(country_hint, name_hint, comp)
            if s > best_score:
                best, best_score = comp, s
        disp = display_name(country_hint, name_hint)
        resolved[disp] = (best if best_score >= 0.6 else None, best_score)
    return resolved


def pick_seasons(seasons: list[dict], n: int) -> list[dict]:
    if not seasons:
        return []
    current_idx = next((i for i, s in enumerate(seasons) if s.get("is_current")), 0)
    return seasons[current_idx : current_idx + n]


def match_to_row(match: dict, competition_display: str, season_label: str) -> dict:
    score = match.get("score") or {}
    final = score.get("final_score") or {}
    goals_home = final.get("home", score.get("home"))
    goals_away = final.get("away", score.get("away"))

    utc_date = match.get("utc_date")
    date_str = ""
    if utc_date:
        try:
            date_str = datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            date_str = utc_date[:10]

    row = {
        "competition": competition_display,
        "season": season_label,
        "matchday": match.get("matchday"),
        "date": date_str,
        "status": match.get("status"),
        "home_team": (match.get("home_team") or {}).get("name"),
        "away_team": (match.get("away_team") or {}).get("name"),
        "goals_home": goals_home,
        "goals_away": goals_away,
        "_match_id": match.get("id"),
    }
    for field, _category in METRIC_FIELDS:
        row[f"{field}_home"] = None
        row[f"{field}_away"] = None
    return row


def apply_stats_to_row(row: dict, stats_payload: dict | None):
    for field, category in METRIC_FIELDS:
        h, a = extract_metric(stats_payload, field, category)
        row[f"{field}_home"] = h
        row[f"{field}_away"] = a


FINAL_COLUMNS = ["competition", "season", "matchday", "date", "status", "home_team", "away_team", "goals_home", "goals_away"]
for _field, _ in METRIC_FIELDS:
    FINAL_COLUMNS.append(f"{_field}_home")
    FINAL_COLUMNS.append(f"{_field}_away")


# ============================================================================
# 6. STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="TheStatsAPI Season Exporter", layout="wide")

st.sidebar.title("Settings")

api_key = st.sidebar.text_input("TheStatsAPI key", type="password", help="Sent as a Bearer token on every request.")

st.sidebar.markdown("---")
st.sidebar.subheader("Leagues")

all_display_names = [display_name(c, n) for c, n in TARGET_LEAGUES]

if "league_selection" not in st.session_state:
    st.session_state.league_selection = list(all_display_names)

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Select all", use_container_width=True):
    st.session_state.league_selection = list(all_display_names)
if col_b.button("Clear all", use_container_width=True):
    st.session_state.league_selection = []

selected_leagues = st.sidebar.multiselect(
    "Choose leagues to request",
    options=all_display_names,
    key="league_selection",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Seasons")
season_count = st.sidebar.selectbox(
    "Seasons to include (counting back from the current season)",
    options=[1, 2, 3, 4, 5],
    index=0,
    format_func=lambda n: f"{n} season" + ("s" if n > 1 else ""),
)

with st.sidebar.expander("Advanced"):
    requests_per_minute = st.number_input(
        "Max requests / minute", min_value=5, max_value=100, value=55, step=5,
        help="Match this to your TheStatsAPI plan's rate limit.",
    )
    max_workers = st.number_input("Parallel stats requests", min_value=1, max_value=10, value=5, step=1)

run_clicked = st.sidebar.button("Fetch data", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Local stat cache holds {cache_row_count()} match rows.")

st.title("TheStatsAPI — Season Stats Exporter")
st.write(
    "Pulls the selected stat metrics for every past and upcoming match, for every chosen league and "
    "season, into a single CSV. Unplayed fixtures are included with blank stat columns."
)

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if run_clicked:
    if not api_key:
        st.error("Enter your TheStatsAPI key in the sidebar first.")
        st.stop()
    if not selected_leagues:
        st.error("Select at least one league in the sidebar.")
        st.stop()

    client = StatsApiClient(api_key, requests_per_minute=int(requests_per_minute))

    status_box = st.empty()
    progress = st.progress(0.0)
    errors: list[str] = []

    if "competitions" not in st.session_state:
        status_box.info("Loading competitions list from TheStatsAPI...")
        try:
            st.session_state.competitions = client.list_all_competitions()
        except ApiError as e:
            st.error(f"Could not load competitions ({e.status_code} {e.code}): {e.message}")
            st.stop()
    competitions = st.session_state.competitions

    resolved = resolve_leagues(competitions)
    unresolved = [name for name in selected_leagues if resolved.get(name, (None, 0))[0] is None]
    to_fetch = [(name, resolved[name][0]) for name in selected_leagues if resolved.get(name, (None, 0))[0] is not None]

    if unresolved:
        st.warning(
            "Could not confidently match these leagues to a competition in your TheStatsAPI account "
            "(outside your plan's coverage, or the name differs from the API's naming) — skipped: "
            + ", ".join(unresolved)
        )

    if not to_fetch:
        st.error("None of the selected leagues could be matched to a competition. Nothing to fetch.")
        st.stop()

    all_rows: list[dict] = []
    total_leagues = len(to_fetch)

    for league_idx, (league_display, comp) in enumerate(to_fetch):
        comp_id = comp["id"]
        competition_column = f"{comp.get('country', '')} {comp.get('name', '')}".strip()
        status_box.info(f"[{league_idx + 1}/{total_leagues}] {league_display}: loading seasons...")

        try:
            seasons = client.list_seasons(comp_id)
        except ApiError as e:
            errors.append(f"{league_display}: seasons lookup failed ({e.code}: {e.message})")
            progress.progress((league_idx + 1) / total_leagues)
            continue

        chosen_seasons = pick_seasons(seasons, season_count)
        pending_stats: list[tuple[dict, str]] = []

        for season in chosen_seasons:
            season_label = season.get("year") or season.get("name") or season.get("id")
            status_box.info(f"[{league_idx + 1}/{total_leagues}] {league_display}: fetching matches for {season_label}...")
            try:
                matches = client.list_all_matches(comp_id, season["id"])
            except ApiError as e:
                errors.append(f"{league_display} {season_label}: match list failed ({e.code}: {e.message})")
                continue

            for match in matches:
                row = match_to_row(match, competition_column, str(season_label))
                all_rows.append(row)
                if match.get("status") == "finished":
                    pending_stats.append((row, match["id"]))

        if pending_stats:
            status_box.info(
                f"[{league_idx + 1}/{total_leagues}] {league_display}: fetching stats for {len(pending_stats)} finished matches..."
            )
            with ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
                future_to_row = {pool.submit(client.get_match_stats, match_id): row for row, match_id in pending_stats}
                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        payload = future.result()
                        apply_stats_to_row(row, payload)
                    except ApiError as e:
                        errors.append(f"{league_display}: stats for match {row.get('_match_id')} failed ({e.code})")

        progress.progress((league_idx + 1) / total_leagues)

    status_box.info("Merging with local cache to fill any stat gaps...")
    df = pd.DataFrame(all_rows)
    if df.empty:
        st.warning("No matches were returned for the selected leagues/seasons.")
        st.stop()

    df = df.drop(columns=["_match_id"], errors="ignore")
    df = merge_with_cache(df)

    df["_sort_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["_sort_date", "competition", "matchday"], na_position="last").drop(columns=["_sort_date"])
    df = df[[c for c in FINAL_COLUMNS if c in df.columns]]

    st.session_state.result_df = df
    status_box.empty()
    progress.empty()

    if errors:
        with st.expander(f"{len(errors)} warning(s) during fetch"):
            for e in errors:
                st.write("- " + e)

    st.success(f"Done. {len(df)} match rows across {total_leagues} league(s).")

if st.session_state.result_df is not None:
    df = st.session_state.result_df
    st.subheader("Preview")
    st.dataframe(df.head(300), use_container_width=True)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download full CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"thestatsapi_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary",
    )
