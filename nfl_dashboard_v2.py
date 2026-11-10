import os, json, math, random
import numpy as np
import pandas as pd
import streamlit as st
try:
    import streamlit.components.v1 as components
except Exception:
    components = None
import nflreadpy as nfl
from sklearn.linear_model import Ridge

st.set_page_config(page_title="NFL Simulator", page_icon="🏈", layout="wide")

CURRENT_SEASON = 2026
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
STATS = ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"]
DISCRETE = {"passing_tds", "interceptions", "rushing_tds", "receiving_tds"}
NICE = {"passing_yards": "Pass yds", "passing_tds": "Pass TD", "interceptions": "INT", "rushing_yards": "Rush yds", "rushing_tds": "Rush TD", "receptions": "Rec", "receiving_yards": "Rec yds", "receiving_tds": "Rec TD"}
SHOW = {"QB": ["passing_yards", "passing_tds", "interceptions", "rushing_yards"], "RB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards"], "FB": ["rushing_yards", "rushing_tds", "receptions", "receiving_yards"], "WR": ["receptions", "receiving_yards", "receiving_tds"], "TE": ["receptions", "receiving_yards", "receiving_tds"]}
SLOTS = {"RB": 2, "WR": 3, "TE": 1}
COLORS = {"ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D", "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00", "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731", "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837", "LV": "#A5ACAF", "LAC": "#0080C6", "LA": "#003594", "MIA": "#008E97", "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265", "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#69BE28", "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#5A1414"}

# ---- league-average fallbacks (used when play-by-play style data is unavailable or switched off) ----
CODES = ["TD", "FG", "PUNT", "INT", "FUM", "DOWNS"]
DEFAULT_LEAGUE = {"rates": {"TD": 0.22, "FG": 0.14, "PUNT": 0.49, "TO": 0.10, "DOWNS": 0.05}, "fg_pct": 0.85, "int_share": 0.60, "pass_td_share": 0.62, "ypp": 5.6, "mpp": 0.43, "plays_pd": 5.8, "tpd": 2.7}
PS_INT, PS_FUM = 0.09, 0.035      # chance a turnover is returned for a touchdown
TD_PTS = 6.94                     # average points per touchdown incl. extra point / 2-pt tries
QB_K = 100          # how strongly low-snap QBs are pulled toward replacement level
DROPBACKS = 36.0    # typical dropbacks per team per game
QB_WEIGHT = 1.0     # 1.0 = full effect of the QB difference; lower it to dampen
DRIVE_GAP = 0.15    # minutes between drives (kickoffs, timeouts, etc.)
STYLE_PRIOR = 60.0  # drives of league-average behavior blended into each team's style numbers
rng = np.random.default_rng()
rnd = random.Random()


def to_pandas(df):
    try:
        return df.to_pandas()
    except Exception:
        return pd.DataFrame(df.to_dicts())


def mmss(m):
    s = int(round(max(m, 0) * 60))
    return f"{s // 60}:{s % 60:02d}"


def yl(s):
    s = float(s)
    if s >= 99.5:
        return "goal line"
    if s <= 50:
        return f"own {max(int(round(s)), 1)}"
    return f"opp {max(int(round(100 - s)), 1)}"


# =====================================================================
# DATA LOADING
# =====================================================================
@st.cache_data(ttl=6 * 3600, show_spinner="Loading 10 seasons of NFL data (first time takes a minute)...")
def load_data():
    seasons = list(range(CURRENT_SEASON - 9, CURRENT_SEASON + 1))
    try:
        sched = to_pandas(nfl.load_schedules(seasons))
    except Exception:
        seasons = seasons[:-1]
        sched = to_pandas(nfl.load_schedules(seasons))
    sched["home_team"] = sched["home_team"].replace(FIX)
    sched["away_team"] = sched["away_team"].replace(FIX)
    pl = to_pandas(nfl.load_player_stats(seasons))
    ren = {}
    for new, options in [("tm", ["team", "recent_team"]), ("pname", ["player_display_name", "player_name"]), ("interceptions", ["passing_interceptions", "interceptions"])]:
        for o in options:
            if o in pl.columns:
                ren[o] = new
                break
    pl = pl.rename(columns=ren)
    if "season_type" in pl.columns:
        pl = pl[pl["season_type"] == "REG"].copy()
    pl["tm"] = pl["tm"].replace(FIX)
    for c in STATS + ["carries", "targets", "attempts"]:
        if c not in pl.columns:
            pl[c] = 0
        pl[c] = pl[c].fillna(0)
    sack_col = "sacks_suffered" if "sacks_suffered" in pl.columns else ("sacks" if "sacks" in pl.columns else None)
    pl["dropbacks"] = pl["attempts"] + (pl[sack_col].fillna(0) if sack_col else 0)
    pl["pass_epa"] = pl["passing_epa"].fillna(0) if "passing_epa" in pl.columns else 0.0
    keep = ["player_id", "pname", "position", "tm", "season", "week", "carries", "targets", "attempts", "dropbacks", "pass_epa"] + STATS
    pl = pl[[c for c in dict.fromkeys(keep) if c in pl.columns]].copy()
    return sched, pl, seasons


@st.cache_data(ttl=1200, show_spinner="Checking the latest injury reports...")
def load_injuries(season):
    try:
        df = to_pandas(nfl.load_injuries([season]))
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    for c in ["team", "week", "gsis_id", "position", "full_name", "report_primary_injury", "report_status", "practice_status"]:
        if c not in df.columns:
            df[c] = None
    df["team"] = df["team"].replace(FIX)
    return df[["team", "week", "gsis_id", "position", "full_name", "report_primary_injury", "report_status", "practice_status"]].copy()


AVAILABLE_OK = {"questionable", "probable"}


def unavailable(status):
    return bool(status) and str(status).lower() not in AVAILABLE_OK


def inj_lookup(inj_df, team, week):
    if inj_df is None or inj_df.empty or week is None:
        return {}
    x = inj_df[(inj_df["team"] == team) & (inj_df["week"] == week) & inj_df["report_status"].notna()]
    out = {}
    for r in x.itertuples(index=False):
        out[r.gsis_id] = {"status": str(r.report_status).title(), "injury": r.report_primary_injury if isinstance(r.report_primary_injury, str) else "", "name": r.full_name, "pos": r.position, "practice": r.practice_status if isinstance(r.practice_status, str) else ""}
    return out


def game_week(sched, season, home, away):
    x = sched[(sched["season"] == season) & sched["home_score"].isna() & (((sched["home_team"] == home) & (sched["away_team"] == away)) | ((sched["home_team"] == away) & (sched["away_team"] == home)))]
    return int(x["week"].min()) if len(x) else None


# =====================================================================
# TEAM STYLE (from play-by-play drives)
# =====================================================================
CATMAP = {"Touchdown": "TD", "Field goal": "FG", "Missed field goal": "FG", "Punt": "PUNT", "Turnover": "TO", "Opp touchdown": "TO", "Turnover on downs": "DOWNS"}
RATE_KEYS = ["TD", "FG", "PUNT", "TO", "DOWNS"]


def _mins(s):
    parts = s.astype(str).str.split(":", expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=s.index)
    return pd.to_numeric(parts[0], errors="coerce") + pd.to_numeric(parts[1], errors="coerce") / 60.0


def _summ(g, L=None, prior=STYLE_PRIOR):
    w = g["w"]
    n = float(w.sum())
    if n <= 0:
        return None
    def blend(val, lval, k=prior):
        return (n * val + k * lval) / (n + k) if lval is not None else val
    out = {"n": n, "rates": {}}
    for c in RATE_KEYS:
        r = float(w[g["cat"] == c].sum()) / n
        out["rates"][c] = blend(r, L["rates"][c]) if L else r
    fga = float(w[g["cat"] == "FG"].sum())
    made = float(w[(g["cat"] == "FG") & (g["result"] == "Field goal")].sum())
    out["fg_pct"] = made / fga if fga > 0 else 0.85
    plays = float((w * g["plays"]).sum())
    out["plays_pd"] = blend(plays / n, L["plays_pd"] if L else None)
    out["ypp"] = blend(float((w * g["yds"]).sum()) / max(plays, 1.0), L["ypp"] if L else None)
    tpd = float((w * g["tpo"]).sum()) / n
    out["tpd"] = blend(tpd, L["tpd"] if L else None)
    out["mpp"] = max(0.25, (out["tpd"] - DRIVE_GAP) / max(out["plays_pd"], 1.0))
    ints, fums = float((w * g["ints"]).sum()), float((w * g["fums"]).sum())
    ls = L["int_share"] if L else 0.6
    out["int_share"] = (ints + 12 * ls) / (ints + fums + 12)
    ptd, rtd = float((w * g["ptd"]).sum()), float((w * g["rtd"]).sum())
    lp = L["pass_td_share"] if L else 0.62
    out["pass_td_share"] = (ptd + 10 * lp) / (ptd + rtd + 10)
    return out


@st.cache_data(persist="disk", show_spinner="Learning each team's style from play-by-play (the first time takes a few minutes)...")
def load_style(latest, version):
    want = ["game_id", "season", "season_type", "posteam", "defteam", "fixed_drive", "fixed_drive_result", "drive_time_of_possession", "play_type", "yards_gained", "pass_touchdown", "rush_touchdown", "interception", "fumble_lost", "play_deleted"]
    frames = []
    for s in (latest - 1, latest):
        try:
            raw = nfl.load_pbp([s])
            frames.append(raw.select([c for c in want if c in raw.columns]).to_pandas())
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    need = {"game_id", "posteam", "defteam", "fixed_drive", "fixed_drive_result", "drive_time_of_possession"}
    if not need.issubset(df.columns):
        return None
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["posteam"].notna() & df["defteam"].notna() & df["fixed_drive"].notna()].copy()
    for c in ["play_deleted", "pass_touchdown", "rush_touchdown", "interception", "fumble_lost", "yards_gained"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["posteam"] = df["posteam"].replace(FIX)
    df["defteam"] = df["defteam"].replace(FIX)
    df["is_play"] = df["play_type"].isin(["pass", "run"]) & (df["play_deleted"] == 0)
    df["yds"] = np.where(df["is_play"], df["yards_gained"], 0.0)
    df["tpo"] = _mins(df["drive_time_of_possession"])
    g = df.groupby(["season", "game_id", "posteam", "defteam", "fixed_drive"], sort=False).agg(result=("fixed_drive_result", "first"), tpo=("tpo", "first"), plays=("is_play", "sum"), yds=("yds", "sum"), ints=("interception", "sum"), fums=("fumble_lost", "sum"), ptd=("pass_touchdown", "sum"), rtd=("rush_touchdown", "sum")).reset_index()
    g["cat"] = g["result"].map(CATMAP)
    g = g[g["cat"].notna() & (g["plays"] >= 1) & g["tpo"].notna()].copy()
    if len(g) < 500:
        return None
    g["w"] = np.where(g["season"] == g["season"].max(), 1.0, 0.5)
    league = _summ(g, None)
    if league is None:
        return None
    teams = {}
    for t in sorted(set(g["posteam"]) | set(g["defteam"])):
        o = _summ(g[g["posteam"] == t], league)
        d = _summ(g[g["defteam"] == t], league)
        if o is not None and d is not None:
            teams[t] = {"off": o, "def": d}
    return {"league": league, "teams": teams, "latest": int(latest)}


def exp_drives(S, a, b):
    if not S:
        return 11.0
    L = S["league"]
    ta, tb = S["teams"].get(a), S["teams"].get(b)
    pa = ta["off"]["tpd"] if ta else L["tpd"]
    pb = tb["off"]["tpd"] if tb else L["tpd"]
    return float(np.clip(60.0 / (pa + pb), 8.5, 13.5))


def make_model(S, off, dfn, mu, n_exp):
    """Per-drive outcome probabilities for one team's offense, scaled so the expected score equals mu."""
    L = S["league"] if S else DEFAULT_LEAGUE
    O = S["teams"].get(off) if S else None
    D = S["teams"].get(dfn) if S else None
    r = {}
    for c in RATE_KEYS:
        lr = L["rates"][c]
        o = O["off"]["rates"][c] if O else lr
        d = D["def"]["rates"][c] if D else lr
        r[c] = max(1e-4, o * d / lr)
    tot = sum(r.values())
    r = {k: v / tot for k, v in r.items()}
    fgp = float(np.clip(L["fg_pct"], 0.75, 0.93))
    int_share = (O["off"]["int_share"] + L["int_share"]) / 2 if O else L["int_share"]
    extra = n_exp * L["rates"]["TO"] * (L["int_share"] * PS_INT + (1 - L["int_share"]) * PS_FUM) * TD_PTS
    ppd = r["TD"] * TD_PTS + r["FG"] * fgp * 3
    c = max(mu - extra, 3.0) / (n_exp * ppd)
    pt = min(r["TD"] * c, 0.60)
    pf = min(r["FG"] * c, 0.36)
    oth = r["PUNT"] + r["TO"] + r["DOWNS"]
    sc = (1 - pt - pf) / oth
    ypp = (O["off"]["ypp"] if O else L["ypp"]) * (D["def"]["ypp"] if D else L["ypp"]) / L["ypp"]
    return {"TD": pt, "FG": pf, "PUNT": r["PUNT"] * sc, "INT": r["TO"] * sc * int_share, "FUM": r["TO"] * sc * (1 - int_share), "DOWNS": r["DOWNS"] * sc,
            "fg_pct": fgp, "ypp": float(np.clip(ypp, 4.0, 8.0)), "mpp": O["off"]["mpp"] if O else L["mpp"],
            "pscale": float(np.clip((O["off"]["plays_pd"] if O else L["plays_pd"]) / (pt * 8.3 + pf * 7.7 + (1 - pt - pf) * 4.9), 0.7, 1.4)),
            "pass_td": O["off"]["pass_td_share"] if O else L["pass_td_share"]}


# =====================================================================
# SCORE ENGINE (fast, vectorized): scores are built from touchdowns, field goals, extra points
# =====================================================================
def sample_points(m, m_opp, n, n_opp, form, r):
    pt = np.clip(m["TD"] * form, 0.0, 0.68)
    pf = np.clip(m["FG"] * form, 0.0, 0.40)
    td = r.binomial(n, pt)
    cond = np.clip(pf / np.maximum(1.0 - pt, 1e-6), 0.0, 1.0)
    fga = r.binomial(n - td, cond)
    fgm = r.binomial(fga, m["fg_pct"])
    two = r.binomial(td, 0.05)
    two_ok = r.binomial(two, 0.48)
    xp = r.binomial(td - two, 0.94)
    pts = 6 * td + xp + 2 * two_ok + 3 * fgm
    lam = n_opp * (m_opp["INT"] * PS_INT + m_opp["FUM"] * PS_FUM)
    pts = pts + 7 * r.poisson(lam)
    return pts


def calibrate_noise(target_sd, mu):
    """How much game-to-game 'hot/cold' variation is needed so simulated score spread matches real NFL results."""
    m = make_model(None, "A", "B", mu, 11.0)
    def sd_for(sig):
        r = np.random.default_rng(123)
        n = np.clip(np.round(r.normal(11, 1.3, 30000)), 6, 17).astype(int)
        form = np.exp(r.normal(-sig * sig / 2, sig, 30000)) if sig > 0 else np.ones(30000)
        return float(sample_points(m, m, n, n, form, r).std())
    if sd_for(0.0) >= target_sd:
        return 0.0
    lo, hi = 0.0, 0.6
    for _ in range(14):
        mid = (lo + hi) / 2
        if sd_for(mid) < target_sd:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate_drive_noise(target_sd, mu):
    """Same idea as calibrate_noise, for the drive-by-drive engine."""
    mm = make_model(None, "A", "B", mu, 11.0)
    prep = {"home": "A", "away": "B", "M": {"A": dict(mm), "B": dict(mm)}, "ros": {"A": make_roster([]), "B": make_roster([])}, "noise": 0.0}
    def sd_for(sig, n=1500):
        prep["noise"] = sig
        rnd.seed(3)
        return float(np.std([run_game(prep, True)["score"]["A"] for _ in range(n)]))
    try:
        if sd_for(0.0) >= target_sd:
            out = 0.0
        else:
            lo, hi = 0.0, 0.6
            for _ in range(9):
                mid = (lo + hi) / 2
                if sd_for(mid) < target_sd:
                    lo = mid
                else:
                    hi = mid
            out = (lo + hi) / 2
    finally:
        rnd.seed()
    return out


@st.cache_data(show_spinner="Fitting team ratings...")
def fit_ratings(_sched, decay, alpha, version):
    played = _sched.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(played)
    hi = played["home_team"].map(tidx).values
    ai = played["away_team"].map(tidx).values
    if "location" in played.columns:
        hfa = np.where(played["location"].astype(str).str.lower().eq("neutral"), 0, 1)
    else:
        hfa = np.ones(m)
    r = np.arange(m)
    X = np.zeros((2 * m, 2 * n + 1))
    X[r, hi] = 1
    X[r, n + ai] = 1
    X[r, -1] = hfa
    X[m + r, ai] = 1
    X[m + r, n + hi] = 1
    X[m + r, -1] = -hfa
    y = np.concatenate([played["home_score"].values, played["away_score"].values]).astype(float)
    w = np.tile(decay ** (played["season"].max() - played["season"].values), 2)
    model = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    sd = float(np.sqrt(np.average((y - model.predict(X)) ** 2, weights=w)))
    league_avg = float(model.intercept_)
    return {"league_avg": league_avg, "off": {t: float(model.coef_[tidx[t]]) for t in teams}, "dfn": {t: float(model.coef_[n + tidx[t]]) for t in teams}, "home_adv": float(model.coef_[-1]), "sd": sd, "teams": teams, "noise": calibrate_noise(sd, league_avg), "noise_drive": calibrate_drive_noise(sd, league_avg)}


def expected_points(R, home, away, neutral, adj=None):
    adj = adj or {}
    ha = 0 if neutral else R["home_adv"]
    mu_h = R["league_avg"] + R["off"][home] + R["dfn"][away] + ha + adj.get(home, 0.0)
    mu_a = R["league_avg"] + R["off"][away] + R["dfn"][home] - ha + adj.get(away, 0.0)
    return max(mu_h, 3.0), max(mu_a, 3.0)


def simulate_game(R, S, home, away, sims, neutral, adj=None):
    mu_h, mu_a = expected_points(R, home, away, neutral, adj)
    n_exp = exp_drives(S, home, away)
    Mh, Ma = make_model(S, home, away, mu_h, n_exp), make_model(S, away, home, mu_a, n_exp)
    sig = R["noise"]
    base = np.clip(np.round(rng.normal(n_exp, 1.3, sims)), 6, 17).astype(int)
    nh = base
    na = np.clip(base + rng.integers(-1, 2, sims), 6, 17)
    fh = np.exp(rng.normal(-sig * sig / 2, sig, sims))
    fa = np.exp(rng.normal(-sig * sig / 2, sig, sims))
    hs = sample_points(Mh, Ma, nh, na, fh, rng).astype(float)
    as_ = sample_points(Ma, Mh, na, nh, fa, rng).astype(float)
    tie = hs == as_
    coin = rng.random(sims) < 0.5
    hs = np.where(tie & coin, hs + 3, hs)
    as_ = np.where(tie & ~coin, as_ + 3, as_)
    pairs, counts = np.unique(np.stack([hs, as_], 1), axis=0, return_counts=True)
    top = pairs[counts.argmax()]
    return {"hs": hs, "as": as_, "common": (int(top[0]), int(top[1])), "mu": (mu_h, mu_a)}


# =====================================================================
# PLAYER POOLS / RATINGS
# =====================================================================
@st.cache_data(show_spinner="Building player pool...")
def build_pool(_pl, version):
    q = _pl[_pl["position"].isin(["RB", "FB", "WR", "TE"])].copy()
    q["pos2"] = q["position"].replace({"FB": "RB"})
    latest = int(q["season"].max())
    q = q[q["season"] >= latest - 1].sort_values(["season", "week"])
    q["key"] = q["season"].astype(int) * 100 + q["week"].astype(int)
    last_team = q.groupby("player_id")["tm"].last()
    team_keys = {t: sorted(g["key"].unique()) for t, g in q.groupby("tm")}
    pool = {}
    for pid, g in q.groupby("player_id"):
        if len(g) < 3:
            continue
        team = last_team.get(pid)
        if not isinstance(team, str):
            continue
        keys = team_keys.get(team, [])
        last_key = int(g["key"].iloc[-1])
        missed = sum(1 for k in keys if k > last_key)
        if missed > 10:
            continue
        recent = g[g["key"].isin(keys[-6:])]
        tail = g.tail(6)
        pos = g["pos2"].iloc[-1]
        score = float(recent["carries"].sum()) if pos == "RB" else float(recent["targets"].sum())
        pool.setdefault(team, {}).setdefault(pos, []).append({"pid": pid, "name": g["pname"].iloc[-1], "pos": pos, "mean": tail[STATS].mean(), "std": tail[STATS].std().fillna(0), "score": score, "missed": missed, "games": len(tail)})
    for team, by_pos in pool.items():
        for pos, lst in by_pos.items():
            lst.sort(key=lambda c: (-c["score"], -c["games"]))
    return pool


@st.cache_data(show_spinner="Rating quarterbacks...")
def build_qbs(_pl, version):
    q = _pl[(_pl["position"] == "QB") & (_pl["dropbacks"] > 0)]
    window = sorted(q["season"].unique())[-2:]
    q = q[q["season"].isin(window)]
    g = q.groupby("player_id").agg(name=("pname", "last"), db=("dropbacks", "sum"), epa=("pass_epa", "sum"))
    low = g[(g["db"] >= 20) & (g["db"] <= 150)]
    prior = float(np.clip(low["epa"].sum() / low["db"].sum(), -0.25, 0.0)) if len(low) else -0.10
    g["rate"] = (g["epa"] + prior * QB_K) / (g["db"] + QB_K)
    g["team"] = q.sort_values(["season", "week"]).groupby("player_id")["tm"].last()
    g = g.dropna(subset=["team"])
    qbs = {}
    for team in sorted(g["team"].unique()):
        mine = g[g["team"] == team]
        t = q[q["tm"] == team]
        games = t[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(6)
        recent = t.merge(games, on=["season", "week"]).groupby("player_id")["dropbacks"].sum()
        mine = mine.assign(recent=recent.reindex(mine.index).fillna(0)).sort_values(["recent", "db"], ascending=False)
        qbs[team] = [{"pid": pid, "name": r["name"], "team": team, "rate": float(r["rate"]), "db": int(r["db"])} for pid, r in mine.iterrows()]
    return qbs, prior


def qb_row(q, pl_df):
    x = pl_df[pl_df["player_id"] == q["pid"]].sort_values(["season", "week"]).tail(6)
    if len(x) == 0:
        return None
    return (q["name"], "QB", x[STATS].mean(), x[STATS].std().fillna(0))


def skill_rows(chosen):
    rows = []
    for pos, slots in SLOTS.items():
        sel = chosen.get(pos, [])
        if sel:
            f = min(1.0, slots / len(sel))
            rows += [(c["name"], pos, c["mean"] * f, c["std"] * f) for c in sel]
    return rows


# =====================================================================
# PLAYER STAT SIM
# =====================================================================
def sim_players(R, profiles, home, away, game):
    cols = ["Player", "Pos"] + [NICE[s] for s in STATS]
    tables, avg_pts = {}, {}
    for team, arr in ((home, game["hs"]), (away, game["as"])):
        pts = np.asarray(arr)[:10000]
        sims = len(pts)
        avg_pts[team] = float(pts.mean())
        f = np.clip(pts / (R["league_avg"] + R["off"][team]), 0.3, 2.0)
        rows = []
        for name, pos, m, s in profiles.get(team, []):
            row = {"Player": name, "Pos": pos}
            for stat in SHOW.get(pos, STATS):
                mean = m[stat] * f
                if stat in DISCRETE:
                    draw = rng.poisson(np.clip(mean, 0, None))
                else:
                    draw = np.clip(np.round(rng.normal(mean, s[stat])), 0, None)
                lo, hi = np.percentile(draw, [10, 90])
                row[NICE[stat]] = f"{draw.mean():.1f} ({lo:.0f}-{hi:.0f})"
            rows.append(row)
        tables[team] = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    return {"tables": tables, "avg_pts": avg_pts}


# =====================================================================
# DRIVE-BY-DRIVE ENGINE (team style + game flow)
# =====================================================================
def make_roster(rows):
    qb = next((n for n, p, m, s in rows if p == "QB"), "the QB")
    rush = [(n, float(m["rushing_tds"])) for n, p, m, s in rows if p in ("QB", "RB", "FB")] or [("the RB", 1.0)]
    rec = [(n, float(m["receiving_tds"])) for n, p, m, s in rows if p in ("WR", "TE", "RB")] or [("a receiver", 1.0)]
    ptd = sum(float(m["passing_tds"]) for n, p, m, s in rows if p == "QB")
    pf = ptd / (ptd + sum(x for n, x in rush)) if ptd > 0 else 0.6
    return qb, rush, rec, pf


def pick_one(lst):
    return rnd.choices([n for n, x in lst], weights=[max(x, 0.0) + 0.03 for n, x in lst])[0]


def poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rnd.random()
        if p <= L:
            return k
        k += 1


def pick_code(p):
    u, acc = rnd.random(), 0.0
    for k in CODES:
        acc += p[k]
        if u < acc:
            return k
    return "PUNT"


def flow_ctx(margin, half, clock_left, on):
    """Game flow: how the score and clock change what a team does."""
    ctx = {"u": 0.0, "ctrl": 0.0, "note": ""}
    if not on or half == 3:
        return ctx
    if half == 2:
        ramp = max(0.0, min(1.0, (12.0 - clock_left) / 12.0))
        if margin < 0:
            ctx["u"] = min(1.5, -margin / 8.0) * ramp
        elif margin > 0:
            ctx["ctrl"] = min(1.5, margin / 10.0) * ramp
    elif clock_left <= 2.0:
        ctx["u"] = 0.6
        ctx["note"] = "2-minute drill"
    if not ctx["note"]:
        if ctx["u"] > 0.35:
            ctx["note"] = f"trailing by {-margin}: hurry-up, throwing more"
        elif ctx["ctrl"] > 0.35:
            ctx["note"] = f"leading by {margin}: running the clock"
    return ctx


def apply_flow(p, ctx):
    u, c = min(ctx["u"], 1.2), min(ctx["ctrl"], 1.2)
    q = dict(p)
    if u > 0:
        shift = q["PUNT"] * 0.35 * u          # trailing teams go for it on 4th down instead of punting
        q["PUNT"] -= shift
        q["DOWNS"] += shift * 0.55
        q["TD"] += shift * 0.25
        q["FG"] += shift * 0.20
        q["INT"] *= 1 + 0.35 * u              # forcing throws
        q["FUM"] *= 1 + 0.15 * u
    if c > 0:
        q["TD"] *= 1 - 0.10 * c               # leaders grind it out and settle for field goals / punts
        q["INT"] *= 1 - 0.35 * c
        q["FUM"] *= 1 - 0.10 * c
    s = sum(q.values())
    return {k: v / s for k, v in q.items()}


def make_drive(code, start, m, tmult, ymult):
    ypp = max(3.5, m["ypp"] * ymult)
    if code == "TD":
        end = 100.0
        yds = 100.0 - start
        plays = max(2 if yds >= 40 else 1, int(round(yds / (ypp * 1.45) + rnd.gauss(0, 1.2))))
    elif code == "FG":
        end = min(98.0, max(rnd.gauss(77, 8), start + 3.0, 55.0))
        yds = end - start
        plays = max(2, int(round(yds / (ypp * 1.25) + rnd.gauss(0, 1.2) + 1.0)))
    elif code == "PUNT":
        plays = rnd.choices([3, 4, 5, 6, 7, 8], [.42, .17, .14, .12, .09, .06])[0]
        yds = max(-8.0, min(40.0, rnd.gauss(plays * 2.6, 5)))
        end = min(max(start + yds, 1.0), 78.0)
        yds = end - start
    else:
        plays = 2 + poisson(3)
        yds = max(-6.0, min(60.0, rnd.gauss(plays * 4.8, 9)))
        end = min(max(start + yds, 1.0), 99.0)
        yds = end - start
    plays = max(1, int(round(plays * m["pscale"])))
    mins = float(min(max(plays * m["mpp"] * tmult * rnd.uniform(0.85, 1.15), 0.12), 9.0))
    return plays, yds, end, mins


def prepare(R, S, profiles, home, away, neutral, adj, flow=True):
    mu_h, mu_a = expected_points(R, home, away, neutral, adj)
    n_exp = exp_drives(S, home, away)
    prep = {"home": home, "away": away, "mu": {home: mu_h, away: mu_a}, "n_exp": n_exp, "noise": R.get("noise_drive", R["noise"]),
            "M": {home: make_model(S, home, away, mu_h, n_exp), away: make_model(S, away, home, mu_a, n_exp)},
            "ros": {home: make_roster(profiles.get(home, [])), away: make_roster(profiles.get(away, []))}}
    # nudge scoring so the drive-by-drive games average out to the same expected score as the ratings say
    for _ in range(2):
        rnd.seed(7)
        games = [run_game(prep, flow) for _ in range(1500)]
        for t in (home, away):
            got = float(np.mean([g["score"][t] for g in games]))
            if got > 1.0:
                prep["M"][t]["cal"] = float(np.clip(prep["M"][t].get("cal", 1.0) * (prep["mu"][t] / got) ** 0.9, 0.7, 1.4))
    rnd.seed()
    return prep


def run_game(prep, flow=True):
    home, away, M, ros = prep["home"], prep["away"], prep["M"], prep["ros"]
    sig = prep["noise"]
    form = {t: math.exp(rnd.gauss(-sig * sig / 2, sig)) for t in (home, away)}
    other = {home: away, away: home}
    score = {home: 0, away: 0}
    drives = []

    def kickoff_start():
        return float(min(max(rnd.gauss(30, 6), 18), 45))

    def one_drive(team, start, clock_left, half):
        opp, m = other[team], M[team]
        margin = score[team] - score[opp]
        if half == 3:
            when = "OT " + mmss(clock_left)
        else:
            q = 2 * (half - 1) + (1 if clock_left > 15 else 2)
            when = f"Q{q} " + mmss(clock_left - 15 if clock_left > 15 else clock_left)
        if flow and half == 2 and margin > 0 and clock_left <= 2.0:
            plays = max(1, min(4, int(clock_left / 0.67) + 1))
            drives.append({"team": team, "when": when, "plays": plays, "yds": -plays, "mins": clock_left, "code": "KNEEL", "text": "Victory formation - kneels out the clock", "pts": 0, "def_pts": 0, "hs": score[home], "as": score[away], "start": start, "end": max(start - plays, 0.0), "note": f"leading by {margin}"})
            return clock_left, opp, kickoff_start()
        ctx = flow_ctx(margin, half, clock_left, flow)
        cal = m.get("cal", 1.0)
        pt = min(m["TD"] * cal * form[team], 0.65)
        pf = min(m["FG"] * cal * form[team], 0.38)
        oth = m["PUNT"] + m["INT"] + m["FUM"] + m["DOWNS"]
        sc = (1 - pt - pf) / oth
        p = {"TD": pt, "FG": pf, "PUNT": m["PUNT"] * sc, "INT": m["INT"] * sc, "FUM": m["FUM"] * sc, "DOWNS": m["DOWNS"] * sc}
        p = apply_flow(p, ctx)
        uu, cc = min(ctx["u"], 1.2), min(ctx["ctrl"], 1.2)
        tmult = max(0.55, 1 - 0.30 * uu + 0.25 * cc)
        ymult = 1 + 0.08 * uu - 0.06 * cc
        code = pick_code(p)
        plays, yds, end, mins = make_drive(code, start, m, tmult, ymult)
        pts, def_pts, text = 0, 0, ""
        if mins > clock_left:
            if rnd.random() < 0.5:
                ratio = clock_left / mins
                plays = max(1, int(plays * ratio))
                yds *= ratio
                end = start + yds
                code, text = "END", "Time expires"
            mins = clock_left
        nxt_team, nxt_start = opp, kickoff_start()
        if code == "TD":
            qb, rush, rec, pfrac = ros[team]
            pfrac = min(max(0.5 * pfrac + 0.5 * m["pass_td"] + 0.20 * uu - 0.15 * cc, 0.15), 0.88)
            reach = max(1, int(100 - start))
            if rnd.random() < pfrac:
                ln = min(reach, int(min(80, 1 + rnd.expovariate(1 / 13.0))))
                text = f"TOUCHDOWN - {qb} {ln}-yd pass to {pick_one(rec)}"
            else:
                ln = min(reach, int(min(50, 1 + rnd.expovariate(1 / 4.0))))
                text = f"TOUCHDOWN - {pick_one(rush)} {ln}-yd run"
            pts = 6
            if rnd.random() < 0.05:
                if rnd.random() < 0.48:
                    pts, text = 8, text + " (2-pt good)"
                else:
                    text += " (2-pt failed)"
            elif rnd.random() < 0.94:
                pts = 7
            else:
                text += " (PAT missed)"
        elif code == "FG":
            dist = int(min(max(round(117 - end), 18), 66))
            if rnd.random() < min(max(m["fg_pct"] + 0.0095 * (40 - dist), 0.2), 0.99):
                pts, text = 3, f"Field goal GOOD ({dist} yd)"
            else:
                code, text = "FGX", f"Field goal MISSED ({dist} yd)"
                nxt_team, nxt_start = opp, float(min(max(100 - (end - 7), 20), 80))
        elif code == "PUNT":
            text = "Punt"
            landing = end + rnd.gauss(42, 6.5)
            nxt_team, nxt_start = opp, (20.0 if landing >= 100 else float(min(max(100 - landing, 5), 60)))
        elif code in ("INT", "FUM"):
            text = f"INTERCEPTION - thrown by {ros[team][0]}" if code == "INT" else "FUMBLE lost"
            nxt_start = 20.0 if end >= 98 else float(min(max(100 - end + abs(rnd.gauss(5, 6)), 3), 97))
            if rnd.random() < (PS_INT if code == "INT" else PS_FUM):
                def_pts = 7
                text += " - RETURNED FOR A TOUCHDOWN"
                nxt_team, nxt_start = team, kickoff_start()
            else:
                nxt_team = opp
        elif code == "DOWNS":
            text = "Turnover on downs"
            nxt_team, nxt_start = opp, float(min(max(100 - end, 1), 99))
        if code == "END":
            nxt_team, nxt_start = opp, 30.0
        score[team] += pts
        score[opp] += def_pts
        drives.append({"team": team, "when": when, "plays": plays, "yds": int(round(yds)), "mins": mins, "code": code, "text": text, "pts": pts, "def_pts": def_pts, "hs": score[home], "as": score[away], "start": start, "end": end, "note": ctx["note"]})
        return mins, nxt_team, nxt_start

    first = home if rnd.random() < 0.5 else away
    for half in (1, 2):
        clock = 30.0
        team = first if half == 1 else other[first]
        start = 30.0
        while clock > 0.01:
            mins, team, start = one_drive(team, start, clock, half)
            clock -= mins + DRIVE_GAP
    ot = False
    if score[home] == score[away]:
        ot = True
        clock = 10.0
        team = home if rnd.random() < 0.5 else away
        start = 30.0
        poss = {home: 0, away: 0}
        while clock > 0.01:
            mins, nteam, nstart = one_drive(team, start, clock, 3)
            clock -= mins + DRIVE_GAP
            poss[team] += 1
            last = drives[-1]
            if last["def_pts"] > 0:
                break
            if sum(poss.values()) == 1 and last["pts"] >= 6:
                break
            if min(poss.values()) >= 1 and score[home] != score[away]:
                break
            team, start = nteam, nstart
    return {"home": home, "away": away, "score": dict(score), "drives": drives, "ot": ot}


def tstats(g, team):
    d = [x for x in g["drives"] if x["team"] == team]
    return {"Drives": len(d), "Plays": sum(x["plays"] for x in d), "Yards": sum(x["yds"] for x in d), "TOP": sum(x["mins"] for x in d), "TD": sum(x["code"] == "TD" for x in d), "FG": sum(x["code"] == "FG" for x in d), "Punts": sum(x["code"] == "PUNT" for x in d), "Turnovers": sum(x["code"] in ("INT", "FUM") for x in d)}


def drive_df(g):
    h, a = g["home"], g["away"]
    return pd.DataFrame([{"#": i, "Team": d["team"], "Clock": d["when"], "Start": yl(d["start"]), "Plays": d["plays"], "Yards": d["yds"], "Time": mmss(d["mins"]), "Result": d["text"], "Situation": d["note"], "Score": f"{h} {d['hs']} - {a} {d['as']}"} for i, d in enumerate(g["drives"], 1)])


def summary_df(g):
    rows = []
    for t in (g["home"], g["away"]):
        s = tstats(g, t)
        s["TOP"] = mmss(s["TOP"])
        rows.append({"Team": t, "Score": g["score"][t], **s})
    return pd.DataFrame(rows)


def typical_game(prep, sims, flow=True):
    home, away = prep["home"], prep["away"]
    games = [run_game(prep, flow) for _ in range(sims)]
    hs = np.array([g["score"][home] for g in games])
    as_ = np.array([g["score"][away] for g in games])
    hw = (hs > as_).mean()
    rows = []
    for t, wins in ((home, hw), (away, (as_ > hs).mean())):
        ss = [tstats(g, t) for g in games]
        row = {"Team": t, "Win %": round(wins * 100, 1), "Avg score": round(float(np.mean([g["score"][t] for g in games])), 1)}
        for key in ss[0]:
            avg = float(np.mean([x[key] for x in ss]))
            row[key] = mmss(avg) if key == "TOP" else round(avg, 1)
        rows.append(row)
    pool_g = [g for g in games if g["score"][home] != g["score"][away] and (g["score"][home] > g["score"][away]) == (hw >= 0.5)] or games
    best = min(pool_g, key=lambda g: abs(g["score"][home] - hs.mean()) + abs(g["score"][away] - as_.mean()))
    return best, pd.DataFrame(rows)


# =====================================================================
# DRIVE ANIMATION (runs in the browser)
# =====================================================================
ANIM_HTML = r"""
<style>
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,'Segoe UI',Arial,sans-serif;color:#e8eaed;background:transparent}
#wrap{background:#14181f;border-radius:10px;padding:12px}
#sb{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
.team{display:flex;align-items:center;gap:10px;font-weight:700;font-size:20px}
.chip{width:12px;height:36px;border-radius:3px}
.score{font-size:36px;font-variant-numeric:tabular-nums;min-width:48px;text-align:center}
#mid{text-align:center;font-size:14px;color:#aab;line-height:1.5}
canvas{width:100%;display:block;border-radius:6px;background:#2b6b34}
#ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button,select{background:#2a3140;color:#fff;border:1px solid #3b4558;border-radius:6px;padding:6px 12px;font-size:14px;cursor:pointer}
button:hover{background:#364055}
input[type=range]{flex:1;min-width:160px}
#info{font-size:14px;min-height:44px;margin-bottom:8px;line-height:1.45}
#cols{display:flex;gap:12px;flex-wrap:wrap}
#statsbox{flex:1 1 380px}
#listbox{flex:1 1 440px;max-height:240px;overflow-y:auto;border:1px solid #2c3442;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:4px 6px;text-align:right;border-bottom:1px solid #262d3a;white-space:nowrap}
th{position:sticky;top:0;background:#14181f;z-index:1}
th:first-child,td:first-child{text-align:left}
td.res{text-align:left;white-space:normal}
tr.cur td{background:#2a3550}
tr.row{cursor:pointer}
tr.row:hover td{background:#242c3b}
tr.dim td{opacity:.4}
.hint{font-size:12px;color:#8a93a3;margin-top:6px}
</style>
<div id="wrap">
  <div id="sb">
    <div class="team"><div class="chip" id="chipA"></div><span id="nameA"></span><span class="score" id="scoreA">0</span></div>
    <div id="mid"></div>
    <div class="team"><span class="score" id="scoreH">0</span><span id="nameH"></span><div class="chip" id="chipH"></div></div>
  </div>
  <canvas id="field" width="1000" height="300"></canvas>
  <div id="ctl">
    <button id="bRestart" title="Restart">&#9198; Restart</button>
    <button id="bPrev" title="Previous drive">&#9664; Prev</button>
    <button id="bPlay">&#9654; Play</button>
    <button id="bNext" title="Next drive">Next &#9654;</button>
    <label style="font-size:13px;color:#aab">Speed <select id="spd"><option value="1">1x</option><option value="2" selected>2x</option><option value="4">4x</option><option value="8">8x</option></select></label>
    <input type="range" id="slider" min="1" value="1">
  </div>
  <div id="info"></div>
  <div id="cols">
    <div id="statsbox"></div>
    <div id="listbox"><table id="list"></table></div>
  </div>
  <div class="hint">Home team attacks to the right, away team to the left. Click any drive in the list to jump to it.</div>
</div>
<script>
const G = __DATA__;
const D = G.drives, N = D.length;
const cv = document.getElementById('field'), cx = cv.getContext('2d');
const W = cv.width, H = cv.height, YP = W / 120;
const SEG_MS = 520, HOLD_MS = 1700;
let cur = 0, seg = 0, segT = 0, phase = 'move', holdT = 0, shown = -1, playing = false, speed = 2, last = null;
const LABEL = {TD:'TOUCHDOWN', FG:'FIELD GOAL IS GOOD', FGX:'FIELD GOAL MISSED', PUNT:'PUNT', INT:'INTERCEPTION', FUM:'FUMBLE', DOWNS:'TURNOVER ON DOWNS', END:'END OF HALF', KNEEL:'KNEEL DOWNS'};
const BCOL = {TD:'#f5c518', FG:'#7bd88f', FGX:'#f59e0b', PUNT:'#9aa4b5', INT:'#ef4444', FUM:'#ef4444', DOWNS:'#f59e0b', END:'#9aa4b5', KNEEL:'#9aa4b5'};
function lite(hex, a){ const n = parseInt(hex.slice(1), 16); let r = n >> 16, g = (n >> 8) & 255, b = n & 255; r = Math.round(r + (255 - r) * a); g = Math.round(g + (255 - g) * a); b = Math.round(b + (255 - b) * a); return 'rgb(' + r + ',' + g + ',' + b + ')'; }
const colOf = t => (t === G.home ? G.hc : G.ac);
function px(x){ return (10 + x) * YP; }
document.getElementById('nameH').textContent = G.home; document.getElementById('nameA').textContent = G.away;
document.getElementById('chipH').style.background = G.hc; document.getElementById('chipA').style.background = G.ac;
const slider = document.getElementById('slider'); slider.max = N;

function drawField(){
  for (let i = 0; i < 24; i++){ cx.fillStyle = (i % 2) ? '#2f7a3a' : '#2b6b34'; cx.fillRect(px(i * 100 / 20 * 0) + 0, 0, 0, 0); }
  cx.fillStyle = '#2b6b34'; cx.fillRect(0, 0, W, H);
  for (let i = 0; i < 20; i++){ cx.fillStyle = (i % 2) ? '#2f7a3a' : '#2a6833'; cx.fillRect(px(i * 5), 0, 5 * YP, H); }
  cx.fillStyle = G.hc; cx.fillRect(0, 0, 10 * YP, H); cx.fillStyle = G.ac; cx.fillRect(W - 10 * YP, 0, 10 * YP, H);
  cx.fillStyle = 'rgba(255,255,255,.85)'; cx.font = 'bold 22px system-ui'; cx.textAlign = 'center';
  cx.save(); cx.translate(5 * YP, H / 2); cx.rotate(-Math.PI / 2); cx.fillText(G.home, 0, 8); cx.restore();
  cx.save(); cx.translate(W - 5 * YP, H / 2); cx.rotate(Math.PI / 2); cx.fillText(G.away, 0, 8); cx.restore();
  cx.strokeStyle = 'rgba(255,255,255,.75)'; cx.lineWidth = 2;
  for (let y = 0; y <= 100; y += 5){ cx.lineWidth = (y % 10 === 0) ? 2 : 1; cx.beginPath(); cx.moveTo(px(y), 0); cx.lineTo(px(y), H); cx.stroke(); }
  cx.fillStyle = 'rgba(255,255,255,.8)'; cx.font = 'bold 16px system-ui';
  for (let y = 10; y <= 90; y += 10){ const lab = y <= 50 ? y : 100 - y; cx.fillText(lab, px(y), 26); cx.fillText(lab, px(y), H - 12); }
  cx.lineWidth = 1;
  for (let y = 1; y < 100; y++){ if (y % 5){ cx.beginPath(); cx.moveTo(px(y), H * 0.36); cx.lineTo(px(y), H * 0.36 + 6); cx.moveTo(px(y), H * 0.64 - 6); cx.lineTo(px(y), H * 0.64); cx.stroke(); } }
}

function curPos(){
  const d = D[cur], xs = d.xs, ys = d.ys;
  const i = Math.min(seg, xs.length - 2), t = (phase === 'hold') ? 1 : segT;
  const e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  return { x: xs[i] + (xs[i + 1] - xs[i]) * e, y: ys[i] + (ys[i + 1] - ys[i]) * e, i: i };
}

function draw(){
  drawField();
  const d = D[cur], c = curPos(), col = colOf(d.team);
  const pts = []; for (let k = 0; k <= c.i; k++) pts.push([px(d.xs[k]), d.ys[k] * H]); pts.push([px(c.x), c.y * H]);
  // line of scrimmage at the start of the drive
  cx.setLineDash([6, 5]); cx.strokeStyle = '#4da3ff'; cx.lineWidth = 3; cx.beginPath(); cx.moveTo(px(d.xs[0]), 0); cx.lineTo(px(d.xs[0]), H); cx.stroke(); cx.setLineDash([]);
  cx.lineJoin = 'round'; cx.lineCap = 'round';
  cx.strokeStyle = 'rgba(255,255,255,.95)'; cx.lineWidth = 10; cx.beginPath(); pts.forEach((p, k) => k ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1])); cx.stroke();
  cx.strokeStyle = col; cx.lineWidth = 6; cx.beginPath(); pts.forEach((p, k) => k ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1])); cx.stroke();
  for (let k = 0; k <= c.i; k++){ cx.fillStyle = '#fff'; cx.beginPath(); cx.arc(px(d.xs[k]), d.ys[k] * H, 4, 0, 7); cx.fill(); }
  const dir = (D[cur].side === 'home') ? 1 : -1;
  cx.save(); cx.translate(px(c.x), c.y * H); cx.scale(dir, 1);
  cx.fillStyle = '#8b4513'; cx.strokeStyle = '#fff'; cx.lineWidth = 2; cx.beginPath(); cx.ellipse(0, 0, 13, 7.5, 0, 0, 7); cx.fill(); cx.stroke();
  cx.beginPath(); cx.moveTo(-5, 0); cx.lineTo(5, 0); cx.stroke(); cx.restore();
  if (phase === 'hold'){
    let lab = LABEL[d.code] || d.code;
    if (d.def_pts > 0) lab = (d.code === 'INT') ? 'PICK-SIX' : 'SCOOP AND SCORE';
    const bc = d.def_pts > 0 ? '#ef4444' : (BCOL[d.code] || '#fff');
    cx.fillStyle = 'rgba(10,12,16,.78)'; const bw = 520, bh = 74, bx = (W - bw) / 2, by = H / 2 - bh / 2;
    cx.beginPath(); cx.roundRect ? cx.roundRect(bx, by, bw, bh, 12) : cx.rect(bx, by, bw, bh); cx.fill();
    cx.textAlign = 'center'; cx.fillStyle = bc; cx.font = 'bold 30px system-ui'; cx.fillText(lab, W / 2, by + 36);
    cx.fillStyle = '#e8eaed'; cx.font = '15px system-ui';
    const sub = d.def_pts > 0 ? ('+' + d.def_pts + ' ' + (d.team === G.home ? G.away : G.home)) : (d.pts > 0 ? ('+' + d.pts + ' ' + d.team) : (d.plays + ' plays, ' + d.yds + ' yds, ' + d.time));
    cx.fillText(sub, W / 2, by + 60);
  }
}

function statsFor(upto){
  const out = {}; [G.home, G.away].forEach(t => out[t] = {dr: 0, pl: 0, yd: 0, top: 0, td: 0, fg: 0, pu: 0, to: 0});
  for (let i = 0; i <= upto; i++){ const d = D[i], s = out[d.team]; s.dr++; s.pl += d.plays; s.yd += d.yds; s.top += d.mins; if (d.code === 'TD') s.td++; if (d.code === 'FG') s.fg++; if (d.code === 'PUNT') s.pu++; if (d.code === 'INT' || d.code === 'FUM') s.to++; }
  return out;
}
const mm = m => { const s = Math.round(Math.max(m, 0) * 60); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };

function refreshUI(){
  const d = D[cur];
  document.getElementById('scoreH').textContent = shown >= 0 ? D[shown].hs : 0;
  document.getElementById('scoreA').textContent = shown >= 0 ? D[shown].as : 0;
  document.getElementById('mid').innerHTML = d.when + '<br>Drive ' + (cur + 1) + ' of ' + N;
  let s = '<b style="color:' + lite(colOf(d.team), .35) + '">' + d.team + '</b> ball at <b>' + d.start_lbl + '</b>' + (d.note ? ' &middot; <i>' + d.note + '</i>' : '');
  s += (shown >= cur) ? '<br>' + d.text + ' &mdash; ' + d.plays + ' plays, ' + d.yds + ' yds, ' + d.time : '<br><span style="color:#8a93a3">Driving&hellip;</span>';
  document.getElementById('info').innerHTML = s;
  const st = statsFor(shown);
  let h = '<table><tr><th>Drive stats so far</th><th>Drives</th><th>Plays</th><th>Yards</th><th>TOP</th><th>TD</th><th>FG</th><th>Punts</th><th>TO</th></tr>';
  [G.away, G.home].forEach(t => { const x = st[t]; h += '<tr><td style="color:' + lite(colOf(t), .35) + ';font-weight:700">' + t + '</td><td>' + x.dr + '</td><td>' + x.pl + '</td><td>' + x.yd + '</td><td>' + mm(x.top) + '</td><td>' + x.td + '</td><td>' + x.fg + '</td><td>' + x.pu + '</td><td>' + x.to + '</td></tr>'; });
  document.getElementById('statsbox').innerHTML = h + '</table>';
  const rows = document.querySelectorAll('#list tr.row');
  rows.forEach((r, i) => { r.classList.toggle('cur', i === cur); r.classList.toggle('dim', i > shown && i !== cur); });
  if (rows[cur]){ const box = document.getElementById('listbox'); box.scrollTop = Math.max(0, rows[cur].offsetTop - 60); }
  slider.value = cur + 1;
}

function buildList(){
  let h = '<tr><th>#</th><th>Team</th><th>Clock</th><th>Plays-Yds</th><th style="text-align:left">Result</th><th>Score</th></tr>';
  D.forEach((d, i) => { h += '<tr class="row" data-i="' + i + '"><td>' + (i + 1) + '</td><td style="color:' + lite(colOf(d.team), .35) + ';font-weight:700">' + d.team + '</td><td>' + d.when + '</td><td>' + d.plays + '-' + d.yds + '</td><td class="res">' + d.text + '</td><td>' + G.away + ' ' + d.as + ' - ' + G.home + ' ' + d.hs + '</td></tr>'; });
  const t = document.getElementById('list'); t.innerHTML = h;
  t.querySelectorAll('tr.row').forEach(r => r.addEventListener('click', () => jump(parseInt(r.dataset.i))));
}

function startDrive(){ seg = 0; segT = 0; phase = 'move'; holdT = 0; refreshUI(); }
function jump(i){ cur = Math.max(0, Math.min(N - 1, i)); shown = cur - 1; startDrive(); }
function tick(dt){
  const nseg = D[cur].xs.length - 1;
  if (phase === 'move'){
    segT += dt / SEG_MS;
    while (segT >= 1 && seg < nseg - 1){ segT -= 1; seg++; }
    if (seg >= nseg - 1 && segT >= 1){ segT = 1; phase = 'hold'; holdT = 0; shown = cur; refreshUI(); }
  } else {
    holdT += dt;
    if (holdT >= HOLD_MS){ if (cur < N - 1){ cur++; startDrive(); } else { playing = false; setLabel(); } }
  }
}
function setLabel(){ document.getElementById('bPlay').innerHTML = playing ? '&#9208; Pause' : '&#9654; Play'; }
function loop(now){
  if (last === null) last = now;
  const dt = Math.min(now - last, 100) * speed; last = now;
  if (playing) tick(dt);
  draw();
  requestAnimationFrame(loop);
}
document.getElementById('bPlay').onclick = () => { if (!playing && cur === N - 1 && shown === N - 1) jump(0); playing = !playing; setLabel(); };
document.getElementById('bRestart').onclick = () => { jump(0); };
document.getElementById('bPrev').onclick = () => jump(cur - 1);
document.getElementById('bNext').onclick = () => { if (cur < N - 1){ shown = cur; jump(cur + 1); shown = cur - 1; refreshUI(); } };
document.getElementById('spd').onchange = e => { speed = parseFloat(e.target.value); };
slider.oninput = () => jump(parseInt(slider.value) - 1);
buildList(); refreshUI(); setLabel(); requestAnimationFrame(loop);
</script>
"""


def anim_payload(g):
    h, a = g["home"], g["away"]
    r = random.Random(11)
    drives = []
    for d in g["drives"]:
        is_home = d["team"] == h
        s, e = float(d["start"]), float(d["end"])
        k = max(1, min(int(d["plays"]), 9))
        ws = [r.expovariate(1.0) + 0.15 for _ in range(k)]
        tot = sum(ws)
        pos, acc = [s], 0.0
        for w in ws:
            acc += w / tot
            pos.append(s + (e - s) * acc)
        pos[-1] = e
        pos = [min(max(x, 0.0), 100.0) for x in pos]
        xs = pos if is_home else [100.0 - x for x in pos]
        ys = [0.5] + [r.uniform(0.28, 0.72) for _ in range(len(pos) - 1)]
        drives.append({"team": d["team"], "side": "home" if is_home else "away", "when": d["when"], "plays": int(d["plays"]), "yds": int(d["yds"]), "time": mmss(d["mins"]), "mins": float(d["mins"]), "code": d["code"], "text": d["text"], "pts": int(d["pts"]), "def_pts": int(d["def_pts"]), "hs": int(d["hs"]), "as": int(d["as"]), "start_lbl": yl(d["start"]), "note": d["note"], "xs": [round(x, 2) for x in xs], "ys": [round(y, 3) for y in ys]})
    return {"home": h, "away": a, "hc": COLORS.get(h, "#555555"), "ac": COLORS.get(a, "#888888"), "drives": drives}


def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html.strip(), height=height)
    else:
        components.html(html, height=height, scrolling=False)


def build_anim_html(g):
    data = json.dumps(anim_payload(g)).replace("</", "<\\/")
    return ANIM_HTML.replace("__DATA__", data)


# =====================================================================
# TEAM STYLE TABLE
# =====================================================================
def style_table(S, home, away):
    L = S["league"]
    def pct(x):
        return f"{x * 100:.1f}%"
    cols = {"League avg": None}
    rows = [("Touchdown %", "TD"), ("Field goal attempt %", "FG"), ("Punt %", "PUNT"), ("Turnover %", "TO"), ("Turnover on downs %", "DOWNS")]
    out = []
    for lab, key in rows:
        row = {"Per drive": lab, "League": pct(L["rates"][key])}
        for t in (home, away):
            T = S["teams"].get(t)
            row[f"{t} offense"] = pct(T["off"]["rates"][key]) if T else "-"
            row[f"{t} defense (allowed)"] = pct(T["def"]["rates"][key]) if T else "-"
        out.append(row)
    for lab, fn in [("Plays per drive", lambda x: f"{x['plays_pd']:.1f}"), ("Yards per play", lambda x: f"{x['ypp']:.2f}"), ("Minutes per drive", lambda x: f"{x['tpd']:.2f}"), ("Share of TDs by pass", lambda x: pct(x["pass_td_share"]))]:
        row = {"Per drive": lab, "League": fn(L)}
        for t in (home, away):
            T = S["teams"].get(t)
            row[f"{t} offense"] = fn(T["off"]) if T else "-"
            row[f"{t} defense (allowed)"] = fn(T["def"]) if (T and lab in ("Yards per play", "Plays per drive", "Minutes per drive")) else "-"
        out.append(row)
    return pd.DataFrame(out)


# =====================================================================
# APP
# =====================================================================
sched, pl, seasons = load_data()
version = int(sched["home_score"].notna().sum())
latest = seasons[-1]
pool = build_pool(pl, len(pl))
qbs, qb_prior = build_qbs(pl, len(pl))
inj_df = load_injuries(latest)
S_all = load_style(latest, version)

with st.sidebar:
    st.header("Settings")
    decay = st.slider("How much recent seasons matter", 0.4, 1.0, 0.75, 0.05, help="Lower = older seasons count less")
    alpha = st.slider("Rating smoothing", 1.0, 60.0, 20.0, 1.0, help="Higher = ratings pulled harder toward average")
    R = fit_ratings(sched, decay, alpha, version)
    teams = R["teams"]
    home = st.selectbox("Home team", teams, index=teams.index("CHI") if "CHI" in teams else 0)
    away = st.selectbox("Away team", teams, index=teams.index("SEA") if "SEA" in teams else 1)
    neutral = st.checkbox("Neutral site")

    st.subheader("Injuries")
    weeks = sorted(int(w) for w in inj_df["week"].dropna().unique()) if len(inj_df) else []
    gw = game_week(sched, latest, home, away)
    use_inj = False
    inj_week = None
    if weeks:
        use_inj = st.checkbox("Use official injury report", value=True, help="Players listed Out or Doubtful are removed from the default lineups.")
        cand = [w for w in weeks if gw is None or w <= gw] or weeks
        default_week = cand[-1]
        inj_week = st.selectbox("Injury report week", weeks[::-1], index=weeks[::-1].index(default_week), key=f"injwk_{home}_{away}")
        if gw is not None:
            st.caption(f"{away} @ {home} is on the schedule for week {gw}." if sched[(sched['season'] == latest) & (sched['week'] == gw) & (sched['home_team'] == home)].shape[0] else f"{home} and {away} play each other in week {gw}.")
    else:
        st.caption("No injury report data found yet.")
    inj = {home: inj_lookup(inj_df, home, inj_week) if use_inj else {}, away: inj_lookup(inj_df, away, inj_week) if use_inj else {}}
    wk_key = f"{inj_week}_{int(use_inj)}"

    st.subheader("Quarterbacks")
    st.caption("The usual starter is the QB with the most recent snaps. If he is Out or Doubtful on the injury report, the next QB is selected for you.")

    def tag_for(team, pid):
        e = inj[team].get(pid)
        if not e:
            return ""
        return f" [{e['status'].upper()}" + (f": {e['injury']}" if e["injury"] else "") + "]"

    def pick_qb(team, side):
        mine = qbs.get(team, [])
        others = sorted([q for t2, lst in qbs.items() if t2 != team for q in lst], key=lambda z: -z["db"])[:80]
        plist = mine + others
        if not plist:
            return None, None
        labels = []
        for i, q in enumerate(plist):
            base = q["name"] + (" (usual starter)" if (i == 0 and q["team"] == team) else ("" if q["team"] == team else f" ({q['team']})"))
            labels.append(base + (tag_for(team, q["pid"]) if q["team"] == team else ""))
        di = 0
        for i, q in enumerate(mine):
            if not unavailable(inj[team].get(q["pid"], {}).get("status")):
                di = i
                break
        choice = st.selectbox(f"{team} quarterback", labels, index=di, key=f"qb_{side}_{team}_{wk_key}")
        return plist[labels.index(choice)], (mine[0] if mine else plist[0])

    qb_h, s0_h = pick_qb(home, "home")
    qb_a, s0_a = pick_qb(away, "away")
    qb_adj = {home: 0.0, away: 0.0}
    for t, q, s0 in ((home, qb_h, s0_h), (away, qb_a, s0_a)):
        if q is not None and s0 is not None:
            qb_adj[t] = (q["rate"] - s0["rate"]) * DROPBACKS * QB_WEIGHT
            if q["pid"] != s0["pid"]:
                st.caption(f"{t}: {q['name']} instead of {s0['name']} = {qb_adj[t]:+.1f} pts/game")

    st.subheader("Skill players")
    st.caption("Defaults are each team's top players by recent usage, skipping anyone Out or Doubtful. Add or remove anyone you like.")

    def pick_skill(team, side):
        by_pos = pool.get(team, {})
        chosen, changed = {}, []
        with st.expander(f"{team} running backs, receivers, tight ends", expanded=True):
            for pos, title in (("RB", "Running backs"), ("WR", "Wide receivers"), ("TE", "Tight ends")):
                cands = by_pos.get(pos, [])
                labels = {}
                for c in cands:
                    lab = c["name"] + (f" (missed last {c['missed']} games)" if c["missed"] > 0 else "") + tag_for(team, c["pid"])
                    labels[lab] = c
                avail = [lab for lab, c in labels.items() if not unavailable(inj[team].get(c["pid"], {}).get("status"))]
                default = avail[:SLOTS[pos]]
                sel = st.multiselect(f"{title} (usual: {SLOTS[pos]})", list(labels), default=default, key=f"skill_{side}_{team}_{pos}_{wk_key}")
                chosen[pos] = [labels[s] for s in sel]
                if set(sel) != set(default):
                    changed.append(pos)
        return chosen, changed

    ch_h, chg_h = pick_skill(home, "home")
    ch_a, chg_a = pick_skill(away, "away")

    with st.expander("Other injuries / adjustments (points per game)"):
        st.caption("Offense: + means the team scores more. Defense: + means the team allows more. Defenders and linemen on the injury report are not modeled automatically, so use these sliders for them.")
        oh = st.slider(f"{home} offense", -10.0, 10.0, 0.0, 0.5, key="adj_off_home")
        dh = st.slider(f"{home} defense", -10.0, 10.0, 0.0, 0.5, key="adj_def_home")
        oa = st.slider(f"{away} offense", -10.0, 10.0, 0.0, 0.5, key="adj_off_away")
        da = st.slider(f"{away} defense", -10.0, 10.0, 0.0, 0.5, key="adj_def_away")
    adj = {home: qb_adj[home] + oh + da, away: qb_adj[away] + oa + dh}

    st.subheader("Model options")
    use_style = st.checkbox("Team style (from play-by-play)", value=S_all is not None, disabled=S_all is None, help="Each team's pace, turnover rate, field-goal vs touchdown mix, and passing share, learned from the last two seasons of drives.")
    if S_all is None:
        st.caption("Play-by-play style data could not be loaded, so league-average style is used.")
    use_flow = st.checkbox("Game flow", value=True, help="Trailing teams hurry up and throw more, leaders run the clock, and the leader kneels at the end.")

    game_sims = st.select_slider("Game simulations", [10000, 50000, 100000, 250000], value=100000)
    drive_sims = st.select_slider("Games for 'typical game'", [500, 1000, 2000], value=1000)
    run = st.button("Run simulation", type="primary")

st.title("🏈 NFL Game Simulator")
st.caption(f"Based on {seasons[0]}-{seasons[-1]} data ({version:,} completed games). Data refreshes automatically every few hours.")

if run:
    if home == away:
        st.sidebar.error("Pick two different teams.")
    else:
        with st.spinner("Simulating..."):
            S_use = S_all if (use_style and S_all) else None
            prof_used, notes = {}, []
            for t, q, s0, chosen, changed in ((home, qb_h, s0_h, ch_h, chg_h), (away, qb_a, s0_a, ch_a, chg_a)):
                rows = []
                if q is not None:
                    r0 = qb_row(q, pl)
                    if r0 is not None:
                        rows.append(r0)
                    if s0 is not None and q["pid"] != s0["pid"]:
                        notes.append(f"{t}: {q['name']} at QB instead of {s0['name']} ({qb_adj[t]:+.1f} pts per game)")
                prof_used[t] = rows + skill_rows(chosen)
                for pos in changed:
                    notes.append(f"{t} {pos} lineup: " + (", ".join(c["name"] for c in chosen[pos]) or "none"))
                if use_inj:
                    outs = [e for e in inj[t].values() if unavailable(e["status"])]
                    if outs:
                        notes.append(f"{t} injury report (week {inj_week}), Out/Doubtful: " + ", ".join(f"{e['name']} ({e['pos']})" for e in outs))
            for t, o, d in ((home, oh, dh), (away, oa, da)):
                if o or d:
                    notes.append(f"{t}: other adjustments, offense {o:+.1f} / defense {d:+.1f} pts")
            notes.append("Team style: " + ("on" if S_use else "off (league-average style)") + " | Game flow: " + ("on" if use_flow else "off"))
            prep = prepare(R, S_use, prof_used, home, away, neutral, adj, use_flow)
            typ, avg_df = typical_game(prep, drive_sims, use_flow)
            game = simulate_game(R, S_use, home, away, game_sims, neutral, adj)
            st.session_state["res"] = {"home": home, "away": away, "neutral": neutral, "notes": notes, "prep": prep, "flow": use_flow, "S": S_use, "inj_week": inj_week, "use_inj": use_inj,
                                       "game": game, "players": sim_players(R, prof_used, home, away, game), "typical": typ, "avg": avg_df, "drive_sims": drive_sims, "one": run_game(prep, use_flow), "anim": None}

res = st.session_state.get("res")
tab_game, tab_players, tab_drives, tab_anim, tab_inj, tab_style, tab_ratings = st.tabs(["Game result", "Player stats", "Drive by drive", "Drive animation", "Injury report", "Team style", "Team ratings"])

with tab_ratings:
    rt = pd.DataFrame({"Offense": R["off"], "Defense (pts allowed)": R["dfn"]})
    rt["Net"] = rt["Offense"] - rt["Defense (pts allowed)"]
    rt = rt.sort_values("Net", ascending=False).round(2)
    st.caption("Offense = points per game above average scored. Defense = points above average allowed (lower is better). Net = offense minus defense.")
    st.bar_chart(rt["Net"])
    st.dataframe(rt)

with tab_inj:
    if not weeks:
        st.info("No injury report data is available yet.")
    else:
        st.caption(f"Official injury report, week {inj_week}. Out/Doubtful players are removed from the default lineups when 'Use official injury report' is checked. Only quarterbacks, running backs, receivers and tight ends affect the simulation; everyone else is shown for reference.")
        for t in (home, away):
            st.subheader(t)
            x = inj_df[(inj_df["team"] == t) & (inj_df["week"] == inj_week) & inj_df["report_status"].notna()].copy()
            if x.empty:
                st.write("No players listed.")
            else:
                x["Modeled"] = np.where(x["position"].isin(["QB", "RB", "FB", "WR", "TE"]), "Yes", "No")
                x["o"] = x["report_status"].map({"Out": 0, "Doubtful": 1, "Questionable": 2}).fillna(3)
                x = x.sort_values(["o", "position"])
                st.dataframe(x.rename(columns={"full_name": "Player", "position": "Pos", "report_status": "Status", "report_primary_injury": "Injury", "practice_status": "Practice"})[["Player", "Pos", "Status", "Injury", "Practice", "Modeled"]], hide_index=True)

with tab_style:
    if S_all is None:
        st.info("Team style data is not available.")
    else:
        st.caption(f"Learned from {latest - 1}-{latest} regular-season drives (the current season counts double). Each team's numbers are blended with league average so small samples don't swing wildly. Style changes how a team scores and how fast it plays; the team ratings still set how many points it is expected to score.")
        st.dataframe(style_table(S_all, home, away), hide_index=True)

if res is None:
    for t in (tab_game, tab_players, tab_drives, tab_anim):
        with t:
            st.info("Pick two teams in the sidebar and press Run simulation.")
else:
    h, a = res["home"], res["away"]
    with tab_game:
        hs, as_ = res["game"]["hs"], res["game"]["as"]
        for note in res["notes"]:
            st.info(note)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{h} win", f"{(hs > as_).mean() * 100:.1f}%")
        c2.metric(f"{a} win", f"{(as_ > hs).mean() * 100:.1f}%")
        c3.metric(f"Avg score ({h} - {a})", f"{hs.mean():.1f} - {as_.mean():.1f}")
        c4.metric("Most common final", f"{res['game']['common'][0]} - {res['game']['common'][1]}")
        margin = (hs - as_).astype(int)
        top = pd.Series(np.abs(margin)).value_counts(normalize=True).head(3)
        st.caption(f"Average margin ({h}): {(hs - as_).mean():+.1f} | Average total points: {(hs + as_).mean():.1f} | {len(hs):,} simulations | most common margins: " + ", ".join(f"{int(k)} pts ({v * 100:.1f}%)" for k, v in top.items()))
        mc = np.clip(margin, -40, 40)
        dist = pd.Series(mc).value_counts().sort_index().reindex(range(-40, 41), fill_value=0)
        st.subheader(f"How often each margin happens (positive = {h} wins)")
        st.bar_chart(dist / len(mc) * 100)
        st.caption("Scores are built from touchdowns, extra points, field goals, and the occasional defensive score, so common football numbers (3, 7, 10, 14) show up more often than in a smooth bell curve. This headline result runs the fast scoring engine; the drive-by-drive tabs add game flow.")
    with tab_players:
        st.caption("Format: average (10th-90th percentile range). Based on each selected player's last 6 games played, scaled to the simulated score.")
        for t in (h, a):
            st.subheader(f"{t}  (avg simulated score {res['players']['avg_pts'][t]:.1f})")
            st.dataframe(res["players"]["tables"][t], hide_index=True)
    with tab_drives:
        mode = st.radio("Show", ["Most typical game", "One random game"], horizontal=True)
        if mode == "One random game":
            if st.button("Simulate another game"):
                res["one"] = run_game(res["prep"], res["flow"])
            g_show = res["one"]
        else:
            g_show = res["typical"]
            st.subheader(f"Averages across {res['drive_sims']:,} simulated games")
            st.dataframe(res["avg"], hide_index=True)
            st.caption("Below is the simulated game closest to those averages.")
        st.subheader(f"{a} @ {h}" + ("  (went to OT)" if g_show["ot"] else ""))
        st.dataframe(summary_df(g_show), hide_index=True)
        ddf = drive_df(g_show)
        st.dataframe(ddf, hide_index=True, height=36 * (len(ddf) + 1))
    with tab_anim:
        which = st.radio("Game to animate", ["Most typical game", "A random game"], horizontal=True, key="anim_which")
        if which == "A random game":
            if st.button("Simulate a new random game", key="anim_new") or res.get("anim") is None:
                res["anim"] = run_game(res["prep"], res["flow"])
            g_anim = res["anim"]
        else:
            g_anim = res["typical"]
        st.caption("Press Play. The ball follows each drive from where it starts to where it ends, then the result and score update. Use the slider or click a drive in the list to jump around.")
        show_html(build_anim_html(g_anim), 900)