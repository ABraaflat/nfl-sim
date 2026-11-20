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

st.set_page_config(page_title="NFL Simulator", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")

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
# LIVE ROSTERS
# =====================================================================
STAT_TXT = {"RES": "on injured reserve", "DEV": "on the practice squad", "CUT": "released", "RET": "retired", "EXE": "on the exempt list", "TRD": "traded", "TRC": "trade pending", "TRT": "trade pending", "INA": "inactive", "ACT": "active"}
STAT_TAG = {"RES": "on IR", "DEV": "practice squad", "CUT": "released", "RET": "retired", "EXE": "exempt list", "TRD": "traded", "TRC": "trade pending", "TRT": "trade pending"}
ON_ROSTER = ("ACT", "INA")
MODELED = ["QB", "RB", "FB", "WR", "TE"]


@st.cache_data(ttl=3600, show_spinner="Checking the latest team rosters...")
def load_rosters(season):
    try:
        df = to_pandas(nfl.load_rosters_weekly([season]))
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    need = ["team", "week", "gsis_id", "full_name", "position", "status", "jersey_number", "years_exp"]
    for c in need:
        if c not in df.columns:
            df[c] = None
    df = df[need].dropna(subset=["gsis_id", "team", "week"]).copy()
    df["team"] = df["team"].replace(FIX)
    df["week"] = df["week"].astype(int)
    df["status"] = df["status"].fillna("ACT")
    return df.drop_duplicates(["gsis_id", "week"], keep="last").reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_trade_list(season):
    try:
        t = to_pandas(nfl.load_trades())
    except Exception:
        return pd.DataFrame()
    if t is None or len(t) == 0:
        return pd.DataFrame()
    t = t[(t["season"] == season) & t["pfr_name"].fillna("").ne("")].copy()
    if t.empty:
        return t
    t["gave"] = t["gave"].replace(FIX)
    t["received"] = t["received"].replace(FIX)
    return t.sort_values("trade_date", ascending=False)[["trade_date", "pfr_name", "gave", "received"]].rename(columns={"trade_date": "Date", "pfr_name": "Player", "gave": "From", "received": "To"})


def roster_state(ros):
    """Each player's team and status in his team's newest roster week (teams on a bye have no rows that week): {gsis_id: (team, status)}."""
    if ros is None or ros.empty:
        return {}
    tw = ros.groupby("team")["week"].transform("max")
    last = ros[ros["week"] == tw]
    return {r.gsis_id: (r.team, r.status) for r in last.itertuples(index=False)}


@st.cache_data(show_spinner=False)
def roster_moves(ros):
    """Builds a feed of roster changes. Teams on a bye have no roster that week, so each player is compared with his own previous appearance."""
    if ros is None or ros.empty:
        return pd.DataFrame(columns=["week", "gsis_id", "Player", "Pos", "From", "To", "Move", "Detail"])
    r = ros.sort_values(["gsis_id", "week"]).reset_index(drop=True)
    g = r.groupby("gsis_id")
    r["pt"], r["ps"] = g["team"].shift(), g["status"].shift()
    team_weeks = {t: sorted(x["week"].unique()) for t, x in r.groupby("team")}
    first_wk = {t: w[0] for t, w in team_weeks.items()}
    ev = []

    def add(week, row, frm, to, kind, detail):
        ev.append({"week": int(week), "gsis_id": row.gsis_id, "Player": row.full_name, "Pos": row.position, "From": frm, "To": to, "Move": kind, "Detail": detail})

    for row in r.itertuples(index=False):
        extra = f" ({STAT_TXT.get(row.status, row.status)})" if row.status not in ON_ROSTER else ""
        if isinstance(row.pt, str):
            if row.pt != row.team:
                add(row.week, row, row.pt, row.team, "Changed teams", f"{row.pt} to {row.team}{extra}")
            elif row.ps != row.status and {row.ps, row.status} != {"ACT", "INA"}:
                new, old = row.status, row.ps
                if new == "RES":
                    kind = "Placed on IR"
                elif old == "RES" and new in ON_ROSTER:
                    kind = "Activated from IR"
                elif old == "DEV" and new in ON_ROSTER:
                    kind = "Promoted to active roster"
                elif new == "DEV":
                    kind = "Moved to practice squad"
                elif new == "CUT":
                    kind = "Released"
                elif new == "RET":
                    kind = "Retired"
                else:
                    kind = "Status change"
                add(row.week, row, row.team, row.team, kind, f"{STAT_TXT.get(old, old)} to {STAT_TXT.get(new, new)}")
        elif row.week > first_wk.get(row.team, row.week):
            add(row.week, row, None, row.team, "Added to roster", f"signed by {row.team}{extra}")
    last = r.groupby("gsis_id").tail(1)
    for row in last.itertuples(index=False):
        nxt = [w for w in team_weeks.get(row.team, []) if w > row.week]
        if nxt and row.status not in ("CUT", "RET"):
            add(nxt[0], row, row.team, None, "Left roster", f"no longer on {row.team}'s roster")
    out = pd.DataFrame(ev, columns=["week", "gsis_id", "Player", "Pos", "From", "To", "Move", "Detail"])
    return out.sort_values(["week", "Player"], ascending=[False, True]).reset_index(drop=True)


def _tag_candidate(c, team_now, team_before, info, manual=False):
    c = dict(c)
    c["moved_from"] = team_before if team_before != team_now else None
    c["manual"] = bool(manual and team_before != team_now)
    status = info[1] if info else None
    if info is None:
        c["rlabel"] = "not on a roster"
    elif status in ON_ROSTER:
        c["rlabel"] = None
    else:
        c["rlabel"] = STAT_TAG.get(status, status)
    c["blocked"] = c["rlabel"] is not None
    return c


def rehome_pool(pool, rst, ov=()):
    """Moves each skill player to the team the newest roster says he is on, and flags anyone on IR / practice squad / released / not rostered."""
    if not rst:
        return pool
    out = {}
    for team, by_pos in pool.items():
        for pos, lst in by_pos.items():
            for c in lst:
                info = rst.get(c["pid"])
                now = info[0] if info else team
                out.setdefault(now, {}).setdefault(pos, []).append(_tag_candidate(c, now, team, info, c["pid"] in ov))
    for by_pos in out.values():
        for lst in by_pos.values():
            lst.sort(key=lambda c: (not c.get("manual"), -c["score"], -c["games"]))
    return out


def rehome_qbs(qbs, rst, ov=()):
    if not rst:
        return qbs
    out = {}
    for team, lst in qbs.items():
        for i, q in enumerate(lst):
            info = rst.get(q["pid"])
            now = info[0] if info else team
            q2 = _tag_candidate(q, now, team, info, q["pid"] in ov)
            q2["team"] = now
            q2["order"] = i
            out.setdefault(now, []).append(q2)
    for team in out:
        out[team].sort(key=lambda q: (not q.get("manual"), q["moved_from"] is not None, q["order"], -q["db"]))
    return out


def roster_tag(c):
    t = ""
    if c.get("moved_from"):
        t += f" (joined from {c['moved_from']}{', your manual move' if c.get('manual') else ''})"
    if c.get("rlabel"):
        t += f" [{c['rlabel']}]"
    return t


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
        qbs[team] = [{"pid": pid, "name": r["name"], "team": team, "rate": float(r["rate"]), "db": int(r["db"]), "recent": float(r["recent"])} for pid, r in mine.iterrows()]
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
PASS_ENV = {"passing_yards", "passing_tds", "receptions", "receiving_yards", "receiving_tds"}
PASS_RHO = 0.10


def sim_players(R, profiles, home, away, game):
    """Simulates each player's stats game by game. A shared 'passing game' factor per team makes a QB and his receivers rise and fall together,
    and the Anytime TD draw is built from the same rushing and receiving TD draws as the TD props, so same-game parlays stay consistent."""
    from scipy.stats import norm, poisson
    cols = ["Player", "Pos"] + [NICE[s] for s in STATS]
    tables, avg_pts, draws = {}, {}, {}
    for team, arr in ((home, game["hs"]), (away, game["as"])):
        pts = np.asarray(arr)[:10000]
        sims = len(pts)
        avg_pts[team] = float(pts.mean())
        f = np.clip(pts / (R["league_avg"] + R["off"][team]), 0.3, 2.0)
        zt = rng.standard_normal(sims)
        rows = []

        def gen(stat, m, s):
            mean = m[stat] * f
            if stat in PASS_ENV:
                z = np.sqrt(PASS_RHO) * zt + np.sqrt(1.0 - PASS_RHO) * rng.standard_normal(sims)
            else:
                z = rng.standard_normal(sims)
            if stat in DISCRETE:
                return poisson.ppf(np.clip(norm.cdf(z), 1e-9, 1 - 1e-9), np.clip(mean, 1e-9, None))
            return np.clip(np.round(mean + s[stat] * z), 0, None)

        for name, pos, m, s in profiles.get(team, []):
            row = {"Player": name, "Pos": pos}
            made = {}
            for stat in SHOW.get(pos, STATS):
                draw = gen(stat, m, s)
                made[stat] = draw
                lo, hi = np.percentile(draw, [10, 90])
                draws[(team, name, NICE[stat])] = np.asarray(draw).astype(np.int16)
                row[NICE[stat]] = f"{draw.mean():.1f} ({lo:.0f}-{hi:.0f})"
            tdm = float(m["rushing_tds"]) + float(m["receiving_tds"])
            if tdm > 0.02:
                rt = made.get("rushing_tds")
                if rt is None:
                    rt = gen("rushing_tds", m, s)
                ct = made.get("receiving_tds")
                if ct is None:
                    ct = gen("receiving_tds", m, s)
                draws[(team, name, "Anytime TD")] = (np.asarray(rt) + np.asarray(ct)).astype(np.int16)
            rows.append(row)
        tables[team] = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    return {"tables": tables, "avg_pts": avg_pts, "draws": draws}


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
            "ros": {home: make_roster(profiles.get(home, [])), away: make_roster(profiles.get(away, []))},
            "cast": {home: make_cast(profiles.get(home, [])), away: make_cast(profiles.get(away, []))}}
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
        info = {}
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
                rcv = pick_one(rec)
                info = {"kind": "pass", "ln": ln, "who": rcv}
                text = f"TOUCHDOWN - {qb} {ln}-yd pass to {rcv}"
            else:
                ln = min(reach, int(min(50, 1 + rnd.expovariate(1 / 4.0))))
                rsh = pick_one(rush)
                info = {"kind": "run", "ln": ln, "who": rsh}
                text = f"TOUCHDOWN - {rsh} {ln}-yd run"
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
            info = {"land": landing}
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
        drives.append({"team": team, "when": when, "plays": plays, "yds": int(round(yds)), "mins": mins, "code": code, "text": text, "pts": pts, "def_pts": def_pts, "hs": score[home], "as": score[away], "start": start, "end": end, "note": ctx["note"], "info": info})
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
# PLAY-BY-PLAY ENGINE (expands each simulated drive into individual plays)
# =====================================================================
def make_cast(rows):
    """Who can carry, catch and throw for one team, with usage weights taken from each player's recent averages."""
    qb = next((n for n, p, m, s in rows if p == "QB"), None) or "the QB"
    qb_run = next((float(m["rushing_yards"]) for n, p, m, s in rows if p == "QB"), 12.0)
    runners = [(n, p, max(float(m["rushing_yards"]), 1.0)) for n, p, m, s in rows if p in ("RB", "FB")]
    runners.append((qb, "QB", max(qb_run, 3.0) * 0.55))
    if len(runners) == 1:
        runners.insert(0, ("the RB", "RB", 40.0))
    targets, ypr = [], {}
    for n, p, m, s in rows:
        if p in ("WR", "TE", "RB", "FB"):
            rec = float(m["receptions"])
            targets.append((n, p, max(rec, 0.4) * (0.55 if p in ("RB", "FB") else 1.0)))
            ypr[n] = float(np.clip(float(m["receiving_yards"]) / rec, 5.0, 17.0)) if rec > 0.5 else 10.0
    if not targets:
        targets = [("a receiver", "WR", 1.0)]
    order = {pos: [n for n, p, w in sorted([r for r in (targets + [(a, b, c) for a, b, c in runners]) if r[1] == pos], key=lambda z: -z[2])] for pos in ("RB", "WR", "TE")}
    seen = {}
    for pos in order:
        order[pos] = [n for n in dict.fromkeys(order[pos])]
    return {"qb": qb, "runners": runners, "targets": targets, "ypr": ypr, "order": order}


def _wchoice(lst):
    return rnd.choices(lst, weights=[x[2] for x in lst])[0]


def _gain0():
    u = rnd.random()
    if u < 0.47:
        return 0
    if u < 0.54:
        return -int(min(9, 1 + rnd.expovariate(1 / 3.0)))
    return max(1, int(round(rnd.gauss(4.8, 3.0))))


def _series(L, G, kind, togo, last_fixed=None):
    """Gains for one series of L plays. kind: conv (sums to G and converts on the last play), fail (never converts, sums to G),
    td (last play fixed at last_fixed = the touchdown, earlier plays never convert), part (never converts, no total)."""
    for _ in range(40):
        gains, cum, ok = [], 0, True
        free = L - 1 if kind in ("conv", "fail", "td") else L
        if kind == "td":
            free = L - 1
        for i in range(free):
            g = _gain0()
            cum += g
            gains.append(g)
            if cum >= togo:
                ok = False
                break
        if not ok:
            continue
        if kind == "conv":
            last = G - cum
            if last < togo - cum:
                continue
            gains.append(last)
        elif kind == "fail":
            last = G - cum
            if cum + last >= togo or last < -12:
                continue
            gains.append(last)
        elif kind == "td":
            gains.append(last_fixed)
        return gains
    return None


def _plan(s, Y, kind, start, ln):
    """Plans s scrimmage plays covering Y net yards as a list of series; returns the labeled plays or None."""
    for _ in range(80):
        if kind == "fail3":
            f = 3
        elif kind == "fail4":
            f = 4
        else:
            f = min(s, rnd.choices([1, 2, 3], [0.4, 0.35, 0.25])[0])
        r = s - f
        if r < 0:
            return None
        pre, Yf = [], 0
        if kind == "fail3":
            Yf = max(-8, min(9, int(round(rnd.gauss(3.2, 3.6)))))
        elif kind == "fail4":
            Yf = max(-6, min(8, int(round(rnd.gauss(4.0, 2.8)))))
        elif kind == "td":
            pre = [_gain0() for _ in range(f - 1)]
            if any(sum(pre[:i + 1]) >= 10 for i in range(len(pre))):
                continue
            lnx = ln if ln else int(rnd.expovariate(1 / 9.0)) + 1
            Yf = sum(pre) + lnx
        elif kind == "turn":
            pre = [_gain0() for _ in range(f - 1)] + [0]
            if any(sum(pre[:i + 1]) >= 10 for i in range(len(pre))):
                continue
            Yf = sum(pre)
        else:  # open
            pre = [_gain0() for _ in range(f)]
            if any(sum(pre[:i + 1]) >= 10 for i in range(len(pre))):
                continue
            Yf = sum(pre)
        Yc = Y - Yf
        if r == 0:
            if kind == "td":
                lnx = Y - sum(pre)
                if lnx < 1:
                    continue
                return _label([("td", f, pre + [lnx])], start)
            if kind in ("fail3", "fail4"):
                return _label([("fail", f, Y)], start) or None
            if Yc != 0:
                continue
            return _label([("part", f, pre)], start)
        kmin = -(-r // 3)
        kmax = int(Yc // 10)
        if kmax < kmin:
            continue
        k = min(kmax, r, max(kmin, int(round(Yc / 14.0))))
        lens = [1] * k
        for _i in range(r - k):
            cand = [i for i in range(k) if lens[i] < 3]
            if not cand:
                lens = None
                break
            lens[rnd.choice(cand)] += 1
        if lens is None:
            continue
        extra = max(0, Yc - 10 * k)
        w = [rnd.expovariate(1.0) + 0.05 for _ in range(k)]
        ext = [int(extra * x / sum(w)) for x in w]
        ext[-1] += extra - sum(ext)
        series = [("conv", L, 10 + e) for L, e in zip(lens, ext)]
        if kind in ("fail3", "fail4"):
            series.append(("fail", f, Yf))
        elif kind == "td":
            series.append(("td", f, pre + [lnx]))
        else:
            series.append(("part", f, pre))
        res = _label(series, start)
        if res:
            return res
    return None


def _label(series, start):
    """Turns a series plan into per-play gains with down and distance. Returns None if the plan doesn't fit."""
    out, pos = [], float(start)
    for kind, L, spec in series:
        togo0 = min(10.0, 100.0 - pos)
        if kind == "conv":
            g = _series(L, spec, "conv", togo0)
        elif kind == "fail":
            g = _series(L, spec, "fail", togo0)
        elif kind == "td":
            g = list(spec)
            if g[-1] is None:
                return None
        else:
            g = list(spec)
        if g is None or len(g) != L:
            return None
        cum = 0
        for i, gain in enumerate(g):
            conv = kind in ("conv",) and i == L - 1
            tdp = kind == "td" and i == L - 1
            out.append({"gain": int(gain), "down": i + 1, "togo": int(round(togo0 - cum)), "los": pos + cum, "first": conv, "td": tdp, "goal": togo0 >= 100.0 - pos - 0.01 and 100.0 - pos <= 10})
            cum += gain
        pos += cum
    return out


def _fallback(s, Y, start, ln=None, td=False):
    """If no tidy series fits, spread the yards evenly (down and distance still tracked, with a first down awarded when earned)."""
    gains = [int(Y // s)] * s
    for i in range(int(Y - sum(gains))):
        gains[i % s] += 1
    out, pos, down, togo = [], float(start), 1, min(10.0, 100.0 - start)
    for i, g in enumerate(gains):
        first = g >= togo
        out.append({"gain": int(g), "down": down, "togo": int(round(togo)), "los": pos, "first": first and i < s - 1, "td": td and i == s - 1, "goal": False})
        pos += g
        if first:
            down, togo = 1, min(10.0, 100.0 - pos)
        else:
            down, togo = min(down + 1, 4), togo - g
    return out


def _ord(n):
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(n, f"{n}th")


def _yardword(y):
    y = int(y)
    return f"{y} yard{'s' if abs(y) != 1 else ''}" if y != 0 else "no gain"


def expand_drive(d, cast, opp_name="the defense"):
    """Turns one simulated drive into a list of individual plays with players, down and distance, and clock."""
    code, start, end = d["code"], float(d["start"]), float(d["end"])
    s = max(1, int(d["plays"]))
    info = d.get("info") or {}
    team = d["team"]
    qb = cast["qb"]
    plays = []
    per, secs = d["when"].split(" ", 1)
    mm, ss = secs.split(":")
    clock = int(mm) * 60 + int(ss)
    total_secs = max(float(d["mins"]) * 60.0, 3.0)

    def add(**kw):
        plays.append(kw)

    if code == "KNEEL":
        pos = start
        for i in range(s):
            add(type="kneel", down=min(i + 1, 3), togo=10, los=pos, to=max(pos - 1, 0), yds=-1, who=qb, tgt=None, text=f"{qb} kneels", first=False, td=False, w=1.0)
            pos -= 1
        kind, special = None, None
    else:
        td_len = None
        if code == "TD":
            kind = "td"
            s = max(1, s)
            td_len = info.get("ln")
        elif code in ("FG", "FGX"):
            kind, s = "fail3", max(3, s)
        elif code == "PUNT":
            kind, s = "fail3", max(3, s)
        elif code == "DOWNS":
            kind, s = "fail4", max(4, s)
        elif code in ("INT", "FUM"):
            kind = "turn"
        else:
            kind = "open"
        Y = int(round(end - start))
        if code == "TD":
            Y = int(round(100.0 - start))
        plan = None
        ss_ = s
        while plan is None and ss_ >= 1:
            plan = _plan(ss_, Y, kind, start, td_len)
            if plan is None:
                ss_ -= 1 if (kind not in ("fail3", "fail4") or ss_ > (3 if kind == "fail3" else 4)) else 0
                if ss_ < 1 or (kind == "fail3" and ss_ <= 3) or (kind == "fail4" and ss_ <= 4):
                    break
        if plan is None:
            s2 = s
            plan = _fallback(s2, Y, start, td=(code == "TD"))
            if kind in ("fail3", "fail4"):
                plan[-1]["first"] = False
        trail = "trailing" in (d.get("note") or "") or "2-minute" in (d.get("note") or "")
        lead = "leading" in (d.get("note") or "")
        for i, p in enumerate(plan):
            gain, down, togo, pos = p["gain"], p["down"], p["togo"], p["los"]
            last = i == len(plan) - 1
            if p["td"] and info.get("kind"):
                ptype = info["kind"]
            elif code == "INT" and last:
                ptype = "pass"
            elif code == "FUM" and last:
                ptype = "run" if rnd.random() < 0.6 else "pass"
            else:
                pp = {1: 0.40, 2: 0.50, 3: 0.76 if togo >= 5 else 0.36, 4: 0.60}.get(down, 0.5)
                pp += 0.18 if trail else 0.0
                pp -= 0.2 if lead else 0.0
                if gain == 0:
                    pp = max(pp, 0.78)
                elif gain < 0:
                    pp = max(pp, 0.5)
                elif gain >= 18:
                    pp = max(pp, 0.82)
                ptype = "pass" if rnd.random() < min(pp, 0.95) else "run"
                if ptype == "run" and gain > 11 and rnd.random() < 0.7:
                    ptype = "pass"
                if ptype == "pass" and gain in (1, 2, 3) and togo > 6 and rnd.random() < 0.25:
                    ptype = "run"
            q = {"down": down, "togo": togo, "los": pos, "to": pos + gain, "first": p["first"], "td": p["td"], "who": qb, "tgt": None, "air": 0, "w": 1.0}
            at = f"{_ord(down)} & {'Goal' if p.get('goal') else togo} at {yl(pos)}"
            q["at"] = at
            if code == "INT" and last:
                q.update(type="int", yds=0, to=pos, text=f"{qb} pass intercepted by {opp_name}", first=False)
                tg = _wchoice(cast["targets"])
                q["tgt"] = tg[0]
                q["w"] = 0.9
            elif code == "FUM" and last:
                if ptype == "run":
                    rn = _wchoice(cast["runners"])
                    q.update(type="fum", yds=0, to=pos, who=rn[0], text=f"{rn[0]} fumbles - recovered by {opp_name}", first=False, w=0.9)
                else:
                    q.update(type="fum", yds=0, to=pos, who=qb, text=f"{qb} is sacked and fumbles - recovered by {opp_name}", first=False, w=1.0)
            elif ptype == "run":
                if p["td"]:
                    rn = next((x for x in cast["runners"] if x[0] == info.get("who")), None) or _wchoice(cast["runners"])
                else:
                    rn = _wchoice(cast["runners"])
                who = rn[0]
                if p["td"]:
                    q.update(type="run", yds=gain, who=who, text=f"{who} runs {gain} yard{'s' if gain != 1 else ''} for a TOUCHDOWN", w=1.0)
                else:
                    tail = " - FIRST DOWN" if p["first"] else ""
                    q.update(type="run", yds=gain, who=who, text=f"{who} rush {'for ' + _yardword(gain) if gain != 0 else 'for no gain'}{tail}", w=1.0)
            else:
                if gain < 0 and not p["td"]:
                    if rnd.random() < 0.8:
                        q.update(type="sack", yds=gain, who=qb, text=f"{qb} sacked for a loss of {-gain}", w=1.1, first=False)
                    else:
                        q.update(type="pass", yds=gain, who=qb, tgt=_wchoice(cast["targets"])[0], air=min(gain, 0), text="", w=0.9)
                        q["text"] = f"{qb} pass complete to {q['tgt']} for a loss of {-gain}"
                elif gain == 0 and not p["td"]:
                    tg = _wchoice(cast["targets"])
                    q.update(type="inc", yds=0, who=qb, tgt=tg[0], air=int(max(2, rnd.gauss(8, 5))), text=f"{qb} pass incomplete, intended for {tg[0]}", w=0.45)
                else:
                    if p["td"] and info.get("who") and info.get("kind") == "pass":
                        tg = next((x for x in cast["targets"] if x[0] == info["who"]), None) or _wchoice(cast["targets"])
                    else:
                        tw = [(n, pos_, w * (1.7 if (gain >= 12 and pos_ == "WR") else 1.0) * (0.6 if (togo <= 4 and gain >= 8 and pos_ in ("RB", "FB")) else 1.0)) for n, pos_, w in cast["targets"]]
                        tg = _wchoice(tw)
                    air = max(-3, min(gain + 1, int(round(rnd.gauss(0.72 * gain, 2.5)))))
                    if p["td"]:
                        air = max(air, min(gain, 1))
                    if tg[1] in ("RB", "FB"):
                        air = min(air, int(rnd.uniform(-2, 4)))
                    tail = " - FIRST DOWN" if p["first"] else ""
                    word = f"for a {gain}-yard TOUCHDOWN" if p["td"] else f"for {_yardword(gain)}{tail}"
                    q.update(type="pass", yds=gain, who=qb, tgt=tg[0], air=air, text=f"{qb} pass complete to {tg[0]} {word}", w=0.85)
            add(**q)
        special = None
    # extra points, field goals, punts
    pre_h = d["hs"] - (d["pts"] if team == d.get("_home") else d["def_pts"])
    base_pos = plays[-1]["to"] if plays else end
    if code == "TD":
        pts_after = d["pts"]
        if pts_after == 8:
            add(type="twopt", down=0, togo=2, los=98, to=100, yds=2, who=qb, tgt=None, text="Two-point try is GOOD", first=False, td=False, good=True, w=0.6, at="Two-point try")
        elif d["text"].endswith("(2-pt failed)"):
            add(type="twopt", down=0, togo=2, los=98, to=98, yds=0, who=qb, tgt=None, text="Two-point try fails", first=False, td=False, good=False, w=0.6, at="Two-point try")
        elif pts_after == 7:
            add(type="xp", down=0, togo=0, los=85, to=100, yds=0, who=None, tgt=None, text="Extra point is good", first=False, td=False, good=True, w=0.6, at="Extra point")
        else:
            add(type="xp", down=0, togo=0, los=85, to=85, yds=0, who=None, tgt=None, text="Extra point is NO GOOD", first=False, td=False, good=False, w=0.6, at="Extra point")
    elif code in ("FG", "FGX"):
        dist = int(min(max(round(117 - plays[-1]["to"]), 18), 66))
        good = code == "FG"
        add(type="fg", down=4, togo=0, los=plays[-1]["to"], to=plays[-1]["to"], yds=0, who=None, tgt=None, dist=dist, good=good, text=f"{dist}-yard field goal is GOOD" if good else f"{dist}-yard field goal is NO GOOD", first=False, td=False, w=0.7, at=f"4th & {max(plays[-1]['togo'] - plays[-1]['yds'], 1)} at {yl(plays[-1]['to'])}")
    elif code == "PUNT":
        spot = plays[-1]["to"]
        land = info.get("land", spot + 42.0)
        ln = int(round(min(land, 100.0) - spot)) if land < 100 else int(round(100 - spot))
        tb = land >= 100
        add(type="punt", down=4, togo=0, los=spot, to=min(land, 100.0), yds=ln, who=None, tgt=None, text=f"Punt of {ln} yards" + (" - touchback" if tb else ""), first=False, td=False, w=0.8, at=f"4th & {max(plays[-1]['togo'] - plays[-1]['yds'], 1)} at {yl(spot)}")
    # clock for every play
    wt = sum(p["w"] for p in plays) or 1.0
    t = clock
    for p in plays:
        p["per"], p["clock"] = per, int(t)
        t = max(0, t - total_secs * p["w"] / wt)
    return plays


def expand_game(g, prep):
    """Expands every drive and fills in running score and a flat play list with drive numbers."""
    home, away = g["home"], g["away"]
    allp = []
    hs = as_ = 0
    for di, d in enumerate(g["drives"]):
        dd = dict(d)
        dd["_home"] = home
        team = d["team"]
        opp = away if team == home else home
        plays = expand_drive(dd, prep["cast"][team], opp_name=f"{opp}")
        # keep the drive table consistent with the plays
        scrim = [p for p in plays if p["type"] in ("run", "pass", "inc", "sack", "int", "fum", "kneel")]
        d["plays"] = len(scrim)
        d["yds"] = int(round(sum(p["yds"] for p in scrim if p["type"] != "xp")))
        for n, p in enumerate(plays, 1):
            p["d"], p["n"], p["team"], p["opp"] = di, n, team, opp
            p["qtr"] = d["when"].split(" ")[0] if False else p["per"]
            p["hs0"], p["as0"] = hs, as_
            pts = 0
            if p["type"] in ("run", "pass") and p.get("td"):
                pts = 6
            elif p["type"] == "xp" and p.get("good"):
                pts = 1
            elif p["type"] == "twopt" and p.get("good"):
                pts = 2
            elif p["type"] == "fg" and p.get("good"):
                pts = 3
            if pts:
                if team == home:
                    hs += pts
                else:
                    as_ += pts
            if p["type"] in ("int", "fum") and d["def_pts"] == 7:
                p["pick6"] = True
                p["text"] += " and RETURNED FOR A TOUCHDOWN"
                if opp == home:
                    hs += 7
                else:
                    as_ += 7
            p["hs"], p["as"] = hs, as_
            p["key"] = bool(p.get("td") or p["type"] in ("int", "fum", "fg", "twopt") or p.get("pick6") or (p["type"] in ("run", "pass") and p["yds"] >= 20) or p["type"] == "sack" and False)
        allp += plays
    for i, p in enumerate(allp):
        p["i"] = i
    return allp


def box_score(plays, home, away):
    """Player and team stats built only from the plays that happened."""
    out = {}
    for t in (home, away):
        pas, rus, rec, kick = {}, {}, {}, {"FG": [0, 0], "XP": [0, 0], "long": 0}
        tm = {"plays": 0, "yards": 0, "pass_yds": 0, "rush_yds": 0, "first": 0, "turn": 0, "third": [0, 0], "fourth": [0, 0], "sacks": 0, "punts": 0, "punt_yds": 0}
        for p in plays:
            if p["team"] != t:
                continue
            ty = p["type"]
            if ty in ("run", "pass", "inc", "sack", "int", "fum"):
                tm["plays"] += 1
                if ty != "inc" and ty != "int":
                    tm["yards"] += p["yds"]
                if p["down"] == 3:
                    tm["third"][1] += 1
                    tm["third"][0] += int(bool(p["first"] or p.get("td")))
                if p["down"] == 4 and ty != "kneel":
                    tm["fourth"][1] += 1
                    tm["fourth"][0] += int(bool(p["first"] or p.get("td")))
                if p["first"] or p.get("td"):
                    tm["first"] += 1
            if ty in ("int", "fum"):
                tm["turn"] += 1
            if ty == "run":
                r = rus.setdefault(p["who"], [0, 0, 0, 0])
                r[0] += 1
                r[1] += p["yds"]
                r[2] += int(bool(p.get("td")))
                r[3] = max(r[3], p["yds"])
                tm["rush_yds"] += p["yds"]
            elif ty == "pass":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[0] += 1
                a[1] += 1
                a[2] += p["yds"]
                a[3] += int(bool(p.get("td")))
                e = rec.setdefault(p["tgt"], [0, 0, 0, 0, 0])
                e[0] += 1
                e[1] += 1
                e[2] += p["yds"]
                e[3] += int(bool(p.get("td")))
                e[4] = max(e[4], p["yds"])
                tm["pass_yds"] += p["yds"]
            elif ty == "inc":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[1] += 1
                e = rec.setdefault(p["tgt"], [0, 0, 0, 0, 0])
                e[0] += 1
            elif ty == "int":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[1] += 1
                a[4] += 1
                e = rec.setdefault(p["tgt"], [0, 0, 0, 0, 0])
                e[0] += 1
            elif ty == "sack":
                a = pas.setdefault(p["who"], [0, 0, 0, 0, 0, 0])
                a[5] += 1
                a[2] += p["yds"]
                tm["sacks"] += 1
                tm["pass_yds"] += p["yds"]
            elif ty == "fg":
                kick["FG"][1] += 1
                kick["FG"][0] += int(bool(p.get("good")))
                if p.get("good"):
                    kick["long"] = max(kick["long"], p["dist"])
            elif ty == "xp":
                kick["XP"][1] += 1
                kick["XP"][0] += int(bool(p.get("good")))
            elif ty == "punt":
                tm["punts"] += 1
                tm["punt_yds"] += p["yds"]
        out[t] = {"pass": pas, "rush": rus, "rec": rec, "kick": kick, "team": tm}
    return out


def box_frames(box, plays, home, away):
    res = {}
    for t in (home, away):
        b = box[t]
        P = pd.DataFrame([{"Player": n, "C/A": f"{a[0]}/{a[1]}", "Yds": a[2], "TD": a[3], "INT": a[4], "Sacked": a[5]} for n, a in b["pass"].items()])
        R = pd.DataFrame([{"Player": n, "Car": r[0], "Yds": r[1], "Avg": round(r[1] / r[0], 1) if r[0] else 0.0, "TD": r[2], "Long": r[3]} for n, r in sorted(b["rush"].items(), key=lambda kv: -kv[1][1]) if r[0]])
        C = pd.DataFrame([{"Player": n, "Tgt": e[0], "Rec": e[1], "Yds": e[2], "TD": e[3], "Long": e[4]} for n, e in sorted(b["rec"].items(), key=lambda kv: (-kv[1][2], -kv[1][0])) if n])
        tm = b["team"]
        T = {"Total yards": tm["yards"], "Passing": tm["pass_yds"], "Rushing": tm["rush_yds"], "Plays": tm["plays"], "Yards per play": round(tm["yards"] / tm["plays"], 1) if tm["plays"] else 0.0,
             "First downs": tm["first"], "3rd down": f"{tm['third'][0]}/{tm['third'][1]}", "4th down": f"{tm['fourth'][0]}/{tm['fourth'][1]}", "Sacks taken": tm["sacks"], "Turnovers": tm["turn"],
             "Field goals": f"{b['kick']['FG'][0]}/{b['kick']['FG'][1]}", "Punts": tm["punts"]}
        res[t] = {"pass": P, "rush": R, "rec": C, "team": T}
    return res


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
#wp{background:#1b212c;margin-top:8px;cursor:pointer}
@keyframes pop{0%{transform:scale(1)}30%{transform:scale(1.5);color:#f5c518}100%{transform:scale(1)}}
.score.pop{animation:pop .9s ease-out}
label.opt{font-size:13px;color:#aab;display:flex;align-items:center;gap:5px}
</style>
<div id="wrap">
  <div id="sb">
    <div class="team"><div class="chip" id="chipA"></div><span id="nameA"></span><span class="score" id="scoreA">0</span></div>
    <div id="mid"></div>
    <div class="team"><span class="score" id="scoreH">0</span><span id="nameH"></span><div class="chip" id="chipH"></div></div>
  </div>
  <canvas id="field" width="1000" height="300"></canvas>
  <canvas id="wp" width="1000" height="170" title="Click to jump to a drive"></canvas>
  <div id="ctl">
    <button id="bRestart" title="Restart">&#9198; Restart</button>
    <button id="bPrev" title="Previous drive">&#9664; Prev</button>
    <button id="bPlay">&#9654; Play</button>
    <button id="bNext" title="Next drive">Next &#9654;</button>
    <label style="font-size:13px;color:#aab">Speed <select id="spd"><option value="0.5">0.5x</option><option value="1">1x</option><option value="2" selected>2x</option><option value="4">4x</option><option value="8">8x</option></select></label>
    <label class="opt"><input type="checkbox" id="cel" checked> Celebrate scores</label>
    <input type="range" id="slider" min="1" value="1">
  </div>
  <div id="info"></div>
  <div id="cols">
    <div id="statsbox"></div>
    <div id="listbox"><table id="list"></table></div>
  </div>
  <div class="hint">Home team attacks to the right, away team to the left. Click any drive in the list, or a point on the win-chance chart, to jump to it. The win-chance line is an estimate from the score, the time left and each team's pre-game rating.</div>
</div>
<script>
const G = __DATA__;
const D = G.drives, N = D.length;
const cv = document.getElementById('field'), cx = cv.getContext('2d');
const W = cv.width, H = cv.height, YP = W / 120;
const SEG_MS = 520, HOLD_MS = 1700;
let cur = 0, seg = 0, segT = 0, phase = 'move', holdT = 0, shown = -1, playing = false, speed = 2, last = null;
let parts = [], pdt = 16, celebOn = true;
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
  drawParts();
  drawWP();
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

function celebrate(){
  const d = D[cur];
  if (!celebOn || !(d.pts > 0 || d.def_pts > 0)) return;
  const scorer = d.def_pts > 0 ? (d.team === G.home ? G.away : G.home) : d.team;
  const big = d.pts >= 6 || d.def_pts > 0, n = big ? 120 : 45;
  const base = colOf(scorer), pal = [lite(base, .15), '#ffffff', '#f5c518', base];
  const ex = px(d.xs[d.xs.length - 1]), ey = H * 0.5;
  for (let i = 0; i < n; i++){
    const a = Math.random() * Math.PI * 2, v = 2 + Math.random() * (big ? 9 : 5);
    parts.push({x: ex, y: ey, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 3, life: 1, col: pal[i % pal.length], s: 3 + Math.random() * 4, r: Math.random() * 6});
  }
  const el = document.getElementById(scorer === G.home ? 'scoreH' : 'scoreA');
  el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
}
function drawParts(){
  if (!parts.length) return;
  const k = pdt / 16;
  parts = parts.filter(p => p.life > 0);
  parts.forEach(p => {
    p.vy += 0.22 * k; p.x += p.vx * k; p.y += p.vy * k; p.vx *= 0.995; p.life -= pdt / 1700; p.r += 0.2 * k;
    cx.globalAlpha = Math.max(p.life, 0); cx.fillStyle = p.col;
    cx.save(); cx.translate(p.x, p.y); cx.rotate(p.r); cx.fillRect(-p.s / 2, -p.s / 4, p.s, p.s / 2); cx.restore();
  });
  cx.globalAlpha = 1;
}
const wc = document.getElementById('wp'), wx = wc.getContext('2d');
const WPL = 46, WPR = 14, WPT = 18, WPB = 22;
function wpX(i){ return WPL + (wc.width - WPL - WPR) * i / N; }
function wpY(p){ return WPT + (wc.height - WPT - WPB) * (1 - p); }
function drawWP(){
  const w = wc.width, h = wc.height, wp = G.wp;
  wx.fillStyle = '#1b212c'; wx.fillRect(0, 0, w, h);
  wx.font = '12px system-ui'; wx.lineWidth = 1;
  [0, .25, .5, .75, 1].forEach(p => {
    wx.strokeStyle = (p === .5) ? '#56627a' : '#2c3442'; wx.setLineDash(p === .5 ? [5, 4] : []);
    wx.beginPath(); wx.moveTo(WPL, wpY(p)); wx.lineTo(w - WPR, wpY(p)); wx.stroke();
    wx.fillStyle = '#8a93a3'; wx.textAlign = 'right'; wx.fillText(Math.round(p * 100) + '%', WPL - 6, wpY(p) + 4);
  });
  wx.setLineDash([]);
  wx.textAlign = 'left'; wx.font = 'bold 12px system-ui';
  wx.fillStyle = lite(G.hc, .35); wx.fillText(G.home + ' win chance', WPL + 8, WPT + 12);
  wx.fillStyle = lite(G.ac, .35); wx.fillText(G.away + ' win chance', WPL + 8, h - WPB - 6);
  const upto = Math.max(shown + 1, 0);
  wx.strokeStyle = 'rgba(255,255,255,.16)'; wx.lineWidth = 2; wx.beginPath();
  wp.forEach((p, i) => i ? wx.lineTo(wpX(i), wpY(p)) : wx.moveTo(wpX(i), wpY(p))); wx.stroke();
  for (let i = 1; i <= upto && i < wp.length; i++){
    const mid = (wp[i] + wp[i - 1]) / 2;
    wx.fillStyle = mid >= .5 ? G.hc : G.ac; wx.globalAlpha = .45;
    wx.beginPath(); wx.moveTo(wpX(i - 1), wpY(.5)); wx.lineTo(wpX(i - 1), wpY(wp[i - 1])); wx.lineTo(wpX(i), wpY(wp[i])); wx.lineTo(wpX(i), wpY(.5)); wx.closePath(); wx.fill();
    wx.globalAlpha = 1;
  }
  wx.strokeStyle = '#ffffff'; wx.lineWidth = 3; wx.beginPath();
  for (let i = 0; i <= upto && i < wp.length; i++) i ? wx.lineTo(wpX(i), wpY(wp[i])) : wx.moveTo(wpX(i), wpY(wp[i]));
  wx.stroke();
  const k = Math.min(upto, wp.length - 1), pv = wp[k];
  wx.fillStyle = '#ffffff'; wx.beginPath(); wx.arc(wpX(k), wpY(pv), 5, 0, 7); wx.fill();
  const lead = pv >= .5 ? G.home : G.away, shownP = Math.round(Math.max(pv, 1 - pv) * 100);
  wx.fillStyle = '#e8eaed'; wx.font = 'bold 13px system-ui'; wx.textAlign = (k > N * 0.8) ? 'right' : 'left';
  wx.fillText(lead + ' ' + shownP + '%', wpX(k) + ((k > N * 0.8) ? -10 : 10), wpY(pv) + (pv >= .5 ? 16 : -8));
  wx.fillStyle = '#8a93a3'; wx.font = '11px system-ui'; wx.textAlign = 'center';
  wx.fillText('Kickoff', wpX(0) + 10, h - 6); wx.fillText('Final', wpX(N) - 6, h - 6);
}
wc.addEventListener('click', e => {
  const r = wc.getBoundingClientRect(), x = (e.clientX - r.left) * wc.width / r.width;
  const i = Math.round((x - WPL) / (wc.width - WPL - WPR) * N);
  jump(Math.max(0, Math.min(N - 1, i - 1)));
});
document.getElementById('cel').onchange = e => { celebOn = e.target.checked; };
function startDrive(){ seg = 0; segT = 0; phase = 'move'; holdT = 0; refreshUI(); }
function jump(i){ cur = Math.max(0, Math.min(N - 1, i)); shown = cur - 1; startDrive(); }
function tick(dt){
  const nseg = D[cur].xs.length - 1;
  if (phase === 'move'){
    segT += dt / SEG_MS;
    while (segT >= 1 && seg < nseg - 1){ segT -= 1; seg++; }
    if (seg >= nseg - 1 && segT >= 1){ segT = 1; phase = 'hold'; holdT = 0; shown = cur; refreshUI(); celebrate(); }
  } else {
    holdT += dt;
    if (holdT >= HOLD_MS){ if (cur < N - 1){ cur++; startDrive(); } else { playing = false; setLabel(); } }
  }
}
function setLabel(){ document.getElementById('bPlay').innerHTML = playing ? '&#9208; Pause' : '&#9654; Play'; }
function loop(now){
  if (last === null) last = now;
  const raw = Math.min(now - last, 100); pdt = raw; const dt = raw * speed; last = now;
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


def _wp_home(margin, pre, rem):
    """Rough home win chance from the score margin, minutes left in regulation, and the pre-game expected margin."""
    f = max(rem, 0.0) / 60.0
    if f <= 0.0:
        return 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
    z = (margin + pre * f) / (13.3 * math.sqrt(f))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def win_curve(drives, pre):
    """Home win chance before the game and after each drive (list of len(drives)+1)."""
    out = [_wp_home(0.0, pre, 60.0)]
    for d in drives:
        try:
            lab, clk = str(d["when"]).split(" ", 1)
            mm_, ss_ = clk.split(":")
            t = int(mm_) + int(ss_) / 60.0
            rem0 = 0.0 if lab == "OT" else (4 - int(lab[1])) * 15 + t
        except Exception:
            rem0 = 0.0
        rem1 = max(rem0 - float(d["mins"]), 0.0)
        out.append(round(_wp_home(float(d["hs"]) - float(d["as"]), pre, rem1), 4))
    return out


def anim_payload(g, pre=0.0):
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
    return {"home": h, "away": a, "hc": COLORS.get(h, "#555555"), "ac": COLORS.get(a, "#888888"), "drives": drives, "wp": win_curve(g["drives"], pre)}


def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html.strip(), height=height)
    else:
        components.html(html, height=height, scrolling=False)


def build_anim_html(g, pre=0.0):
    data = json.dumps(anim_payload(g, pre)).replace("</", "<\\/")
    return ANIM_HTML.replace("__DATA__", data)


# =====================================================================
# INTERACTIVE FIELD (plays on the field, runs in the browser)
# =====================================================================
FIELD_HTML = r"""
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;color:#e8eaed;background:transparent;-webkit-text-size-adjust:100%}
#wrap{background:#14181f;border-radius:12px;padding:10px;max-width:1120px;margin:0 auto}
#sb{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
.team{display:flex;align-items:center;gap:8px;font-weight:700;font-size:18px;min-width:0}
.team span.nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip{width:10px;height:32px;border-radius:3px;flex:none}
.score{font-size:32px;font-variant-numeric:tabular-nums;min-width:38px;text-align:center;line-height:1}
#mid{text-align:center;font-size:13px;color:#aab;line-height:1.35;flex:none}
#mid b{color:#f1f3f6;font-size:15px}
#fieldbox{position:relative;width:100%}
canvas#field{display:block;width:100%;border-radius:8px;background:#2b6b34;touch-action:manipulation}
#banner{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none;text-align:center;opacity:0;transition:opacity .25s;background:rgba(10,12,16,.82);border-radius:12px;padding:10px 22px;white-space:nowrap}
#banner b{display:block;font-size:26px;letter-spacing:.5px}
#banner span{font-size:13px;color:#d6dbe3}
#strip{width:100%;height:46px;display:block;margin-top:6px;border-radius:6px;background:#1b212c}
#ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}
button,select{background:#2a3140;color:#fff;border:1px solid #3b4558;border-radius:8px;padding:9px 12px;font-size:14px;cursor:pointer;min-height:40px;font-family:inherit}
button:hover{background:#364055}
button.on{background:#3a5fcd;border-color:#5b7fe8}
button.big{background:#2f7d45;border-color:#43a35f;font-weight:700;min-width:96px}
input[type=range]{flex:1 1 100%;min-width:120px;height:28px}
label.opt{font-size:13px;color:#aab;display:flex;align-items:center;gap:5px}
#info{font-size:15px;line-height:1.4;margin:6px 0;min-height:46px}
#info .at{color:#9fb0cc;font-size:13px}
#card{display:none;background:#1f2634;border:1px solid #3b4558;border-radius:10px;padding:8px 12px;margin:6px 0;font-size:14px;line-height:1.45}
#card b{font-size:15px}
.cols{display:flex;gap:10px;flex-wrap:wrap}
.cols>div{flex:1 1 420px;min-width:0}
canvas#wp{width:100%;display:block;background:#1b212c;border-radius:8px;cursor:pointer;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 6px;text-align:right;border-bottom:1px solid #262d3a;white-space:nowrap}
th{position:sticky;top:0;background:#14181f;z-index:1;font-weight:600;color:#9aa4b5}
th:first-child,td:first-child{text-align:left}
td.tx{text-align:left;white-space:normal;line-height:1.3}
#listbox{max-height:300px;overflow-y:auto;border:1px solid #2c3442;border-radius:8px;margin-top:6px}
tr.row{cursor:pointer}
tr.row:hover td{background:#242c3b}
tr.cur td{background:#2a3550}
tr.dim td{opacity:.42}
tr.key td.tx{font-weight:600}
.flt{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.flt button{padding:5px 10px;min-height:32px;font-size:13px}
.hint{font-size:12px;color:#8a93a3;margin-top:8px;line-height:1.4}
@keyframes pop{0%{transform:scale(1)}30%{transform:scale(1.5);color:#f5c518}100%{transform:scale(1)}}
.score.pop{animation:pop .9s ease-out}
@media (max-width:640px){th,td{padding:5px 4px;font-size:12px}.team{font-size:15px}.score{font-size:28px}button,select{padding:8px 9px;font-size:13px}#banner b{font-size:20px}}
</style>
<div id="wrap">
  <div id="sb">
    <div class="team"><div class="chip" id="chipH"></div><span class="nm" id="nameH"></span><span class="score" id="scoreH">0</span></div>
    <div id="mid"></div>
    <div class="team"><span class="score" id="scoreA">0</span><span class="nm" id="nameA"></span><div class="chip" id="chipA"></div></div>
  </div>
  <div id="fieldbox"><canvas id="field"></canvas><div id="banner"></div></div>
  <canvas id="strip"></canvas>
  <div id="info"></div>
  <div id="card"></div>
  <div id="ctl">
    <button id="bPlay" class="big">&#9654; Play</button>
    <button id="bPrev" title="Previous play">&#9664; Play</button>
    <button id="bNext" title="Next play">Play &#9654;</button>
    <button id="bDPrev" title="Previous drive">&#9664;&#9664; Drive</button>
    <button id="bDNext" title="Next drive">Drive &#9654;&#9654;</button>
    <button id="bKey" title="Jump to the next big play">&#11088; Key play</button>
    <button id="bRep" title="Replay this play">&#8635; Replay</button>
    <select id="spd" title="Speed"><option value="0.5">0.5x</option><option value="1">1x</option><option value="2" selected>2x</option><option value="4">4x</option><option value="8">8x</option></select>
    <label class="opt"><input type="checkbox" id="rts" checked> Routes</label>
    <label class="opt"><input type="checkbox" id="cel" checked> Celebrate</label>
    <input type="range" id="slider" min="1" value="1">
  </div>
  <div class="cols">
    <div><canvas id="wp"></canvas><div id="stats" style="overflow-x:auto"></div></div>
    <div>
      <div class="flt" id="flt"></div>
      <div id="listbox"><table id="list"></table></div>
    </div>
  </div>
  <div class="hint">Tap or click any player for his stats so far. Tap the field to pause or resume. Click the win-chance chart, a play in the list, or drag the slider to jump around. Keys: space = play/pause, arrows = step plays. Home team attacks to the right (or up on a phone).</div>
</div>
<script>
const G = __DATA__;
const P = G.plays, NP = P.length, HOME = G.home, AWAY = G.away;
const FW = 53.3, CY = FW / 2;
const $ = id => document.getElementById(id);
const cv = $('field'), cx = cv.getContext('2d'), sc = $('strip'), sx = sc.getContext('2d');
const wc = $('wp'), wx = wc.getContext('2d');
let vertical = false, cssW = 1000, cssH = 444, ppy = 8.3, dpr = 1, VIEW = 42, camX = 50, camTarget = 50;
let pi = 0, t = 0, playing = false, speed = 2, last = null, cur = null, phase = 'play', holdT = 0, parts = [], pdt = 16;
let celebOn = true, routesOn = true, filt = 'all', selected = null, celebrated = -1;
const colOf = t => (t === HOME ? G.hc : G.ac);
function lite(hex, a){ const n = parseInt(hex.slice(1), 16); let r = n >> 16, g = (n >> 8) & 255, b = n & 255; r = Math.round(r + (255 - r) * a); g = Math.round(g + (255 - g) * a); b = Math.round(b + (255 - b) * a); return 'rgb(' + r + ',' + g + ',' + b + ')'; }
function rs(seed){ let s = (seed * 9301 + 49297) % 233280; return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; }; }
const ease = u => u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
$('nameH').textContent = HOME; $('nameA').textContent = AWAY; $('chipH').style.background = G.hc; $('chipA').style.background = G.ac;
$('slider').max = NP;

// ---------- layout ----------
function resize(){
  const box = $('fieldbox'); let w = Math.floor(box.clientWidth || 600);
  vertical = w < 640; dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  VIEW = vertical ? 44 : 120;
  if (vertical){ ppy = w / FW; cssH = Math.round(ppy * VIEW); } else { ppy = w / VIEW; cssH = Math.round(ppy * FW); }
  cssW = w; cv.style.height = cssH + 'px'; cv.width = Math.round(w * dpr); cv.height = Math.round(cssH * dpr);
  const sw = $('strip').clientWidth || w; sc.width = Math.round(sw * dpr); sc.height = Math.round(46 * dpr);
  const ww = wc.clientWidth || w; wc.style.height = (vertical ? 150 : 170) + 'px'; wc.width = Math.round(ww * dpr); wc.height = Math.round((vertical ? 150 : 170) * dpr);
  fit();
}
function fit(){ try { const h = document.documentElement.scrollHeight + 4; if (window.frameElement) window.frameElement.style.height = h + 'px'; } catch (e) {} }
function S(X, Y){ return vertical ? [Y * ppy, cssH / 2 - (X - camX) * ppy] : [(X + 10) * ppy, Y * ppy]; }

// ---------- field drawing ----------
function rect(X0, X1, Y0, Y1){ const a = S(X0, Y0), b = S(X1, Y1); cx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])); }
function line(X0, Y0, X1, Y1){ const a = S(X0, Y0), b = S(X1, Y1); cx.beginPath(); cx.moveTo(a[0], a[1]); cx.lineTo(b[0], b[1]); cx.stroke(); }
function drawField(){
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cx.fillStyle = '#2a6833'; cx.fillRect(0, 0, cssW, cssH);
  for (let i = -2; i < 22; i++){ cx.fillStyle = (i % 2) ? '#2f7a3a' : '#2a6833'; rect(i * 5, i * 5 + 5, 0, FW); }
  cx.fillStyle = G.hc; rect(-10, 0, 0, FW); cx.fillStyle = G.ac; rect(100, 110, 0, FW);
  cx.fillStyle = 'rgba(255,255,255,.88)'; cx.font = 'bold ' + Math.round(Math.max(13, ppy * 3.2)) + 'px system-ui'; cx.textAlign = 'center'; cx.textBaseline = 'middle';
  [[-5, HOME, -1], [105, AWAY, 1]].forEach(([X, nm, sgn]) => { const p = S(X, CY); cx.save(); cx.translate(p[0], p[1]); if (!vertical) cx.rotate(sgn * Math.PI / 2); cx.fillText(nm, 0, 0); cx.restore(); });
  cx.strokeStyle = 'rgba(255,255,255,.8)';
  for (let X = 0; X <= 100; X += 5){ cx.lineWidth = (X % 10 === 0) ? 2 : 1; line(X, 0, X, FW); }
  cx.lineWidth = 1;
  for (let X = 1; X < 100; X++){ if (X % 5){ line(X, CY - 3.6, X, CY - 2.9); line(X, CY + 2.9, X, CY + 3.6); line(X, 0.6, X, 1.4); line(X, FW - 1.4, X, FW - 0.6); } }
  cx.fillStyle = 'rgba(255,255,255,.7)'; cx.font = 'bold ' + Math.round(Math.max(11, ppy * 2.2)) + 'px system-ui';
  for (let X = 10; X <= 90; X += 10){ const lab = X <= 50 ? X : 100 - X; let a = S(X, 8), b = S(X, FW - 8); cx.fillText(lab, a[0], a[1]); cx.fillText(lab, b[0], b[1]); }
  cx.strokeStyle = '#f5c518'; cx.lineWidth = 3;
  [-10, 110].forEach(X => { line(X, CY - 3.1, X, CY + 3.1); });
  cx.lineWidth = 1;
}

// ---------- entities and tracks ----------
function Ent(side, role, name, x, y){ return {side, role, name, x0: x, y0: y, tr: []}; }
function go(e, a, b, x, y){ const L = e.tr.length ? e.tr[e.tr.length - 1] : null; e.tr.push({t0: a, t1: Math.max(b, a + 1), x0: L ? L.x1 : e.x0, y0: L ? L.y1 : e.y0, x1: x, y1: y}); return e; }
function epos(e, tm){
  const tr = e.tr; if (!tr.length) return [e.x0, e.y0];
  for (let i = 0; i < tr.length; i++){ const s = tr[i]; if (tm < s.t0) return [s.x0, s.y0]; if (tm <= s.t1){ const u = ease((tm - s.t0) / (s.t1 - s.t0)); return [s.x0 + (s.x1 - s.x0) * u, s.y0 + (s.y1 - s.y0) * u]; } }
  const L = tr[tr.length - 1]; return [L.x1, L.y1];
}
function Ball(x, y){ return {x0: x, y0: y, tr: [], att: []}; }
function bgo(b, a, c, x, y, arc){ const L = b.tr.length ? b.tr[b.tr.length - 1] : null; b.tr.push({t0: a, t1: Math.max(c, a + 1), x0: L ? L.x1 : b.x0, y0: L ? L.y1 : b.y0, x1: x, y1: y, arc: arc || 0}); }
function battach(b, a, c, ent){ b.att.push({t0: a, t1: c, e: ent}); const p = epos(ent, c); const L = b.tr.length ? b.tr[b.tr.length - 1] : null; b.tr.push({t0: c, t1: c + 1, x0: p[0], y0: p[1], x1: p[0], y1: p[1], arc: 0}); }
function bpos(b, tm){
  for (const a of b.att){ if (tm >= a.t0 && tm <= a.t1){ const p = epos(a.e, tm); return [p[0], p[1], 0]; } }
  const tr = b.tr; if (!tr.length) return [b.x0, b.y0, 0];
  for (let i = 0; i < tr.length; i++){ const s = tr[i]; if (tm < s.t0) return [s.x0, s.y0, 0]; if (tm <= s.t1){ const u = (tm - s.t0) / (s.t1 - s.t0); return [s.x0 + (s.x1 - s.x0) * u, s.y0 + (s.y1 - s.y0) * u, (s.arc || 0) * 4 * u * (1 - u)]; } }
  const L = tr[tr.length - 1]; return [L.x1, L.y1, 0];
}

function nameSlots(p){
  const ro = G.rost[p.team] || {qb: 'QB', rb: [], wr: [], te: [], pos: {}};
  const nm = {QB: ro.qb, RB: ro.rb[0] || 'RB', WR1: ro.wr[0] || 'WR', WR2: ro.wr[1] || 'WR', WR3: ro.wr[2] || 'WR', TE: ro.te[0] || 'TE'};
  function place(n){
    if (!n || n === ro.qb) return null;
    const pos = ro.pos[n];
    if (pos === 'RB' || pos === 'FB'){ nm.RB = n; return 'RB'; }
    if (pos === 'TE'){ nm.TE = n; return 'TE'; }
    for (const k of ['WR1', 'WR2', 'WR3']) if (nm[k] === n) return k;
    nm.WR3 = n; return 'WR3';
  }
  const sw = place(p.who), st = place(p.tgt);
  return {nm, slotWho: p.who === ro.qb ? 'QB' : sw, slotTgt: st};
}

function formation(p, X0, d, kick){
  const sl = nameSlots(p), nm = sl.nm, c = CY;
  const o = {};
  const back = (k, y) => [X0 - d * k, y];
  const place = (role, name, pos) => { o[role] = Ent('o', role, name, pos[0], pos[1]); return o[role]; };
  [-4.2, -2.1, 0, 2.1, 4.2].forEach((dy, i) => place('OL' + (i + 1), 'Offensive line', back(0.9, c + dy)));
  place('TE', nm.TE, back(0.9, c + 7));
  place('WR1', nm.WR1, back(0.9, 5)); place('WR2', nm.WR2, back(0.9, FW - 5)); place('WR3', nm.WR3, back(1.9, c + 13));
  place('QB', nm.QB, back(4.6, c)); place('RB', nm.RB, back(5.6, c + 3.2));
  const def = {};
  const fwd = (k, y) => [X0 + d * k, y];
  [-4.8, -1.6, 1.6, 4.8].forEach((dy, i) => { def['DL' + (i + 1)] = Ent('d', 'DL', 'Defensive line', ...fwd(1.2, c + dy)); });
  [-7.5, 0, 7.5].forEach((dy, i) => { def['LB' + (i + 1)] = Ent('d', 'LB', 'Linebacker', ...fwd(4.6, c + dy)); });
  def.CB1 = Ent('d', 'CB', 'Cornerback', ...fwd(6.5, 5.5)); def.CB2 = Ent('d', 'CB', 'Cornerback', ...fwd(6.5, FW - 5.5));
  def.S1 = Ent('d', 'S', 'Safety', ...fwd(13, c - 9)); def.S2 = Ent('d', 'S', 'Safety', ...fwd(13, c + 9));
  return {o, d: def, sl};
}

// ---------- play choreography ----------
function buildPlay(p){
  const off = p.team, d = off === HOME ? 1 : -1, ab = v => off === HOME ? v : 100 - v, c = CY, r = rs(p.i + 3);
  const X0 = ab(clamp(p.los, 0, 100)), X1 = ab(clamp(p.to, 0, 100));
  const F = formation(p, X0, d), O = F.o, D = F.d, sl = F.sl;
  const ball = Ball(X0, c); const ev = {}; let T = 2600, tackle = null, route = null, carrier = null;
  const snap = 450;
  const endZ = off === HOME ? 100 : 0;
  function lineWork(tEnd, push){ ['OL1', 'OL2', 'OL3', 'OL4', 'OL5', 'TE'].forEach(k => { const e = O[k]; go(e, snap, snap + 500, e.x0 + d * push, e.y0 + (r() - .5)); }); ['DL1', 'DL2', 'DL3', 'DL4'].forEach(k => { const e = D[k]; go(e, snap, snap + 500, e.x0 - d * push * .5, e.y0 + (r() - .5) * 1.2); }); }
  const ty = p.type;
  if (ty === 'run' || ty === 'fum' || ty === 'twopt'){
    const qbCarry = p.who === G.rost[off].qb;
    const car = qbCarry ? O.QB : O.RB; carrier = car;
    const g = (ty === 'twopt') ? (p.good ? 2 : 0) : p.yds;
    let xe = (ty === 'fum') ? X0 + d * clamp(r() * 4, 0, 4) : X1; if (p.td) xe = endZ + d * 3.5;
    const run = clamp(550 + Math.abs(xe - X0) * 55, 650, 2000);
    const t0 = snap + 380, t1 = t0 + run;
    bgo(ball, snap, snap + 180, O.QB.x0, O.QB.y0, 0);
    if (!qbCarry){ go(O.QB, snap + 180, t0, X0 - d * 3.8, c - .4); go(car, snap, t0, X0 - d * 3.4, c + 1.3); battach(ball, snap + 180, t0, O.QB); bgo(ball, t0 - 200, t0, X0 - d * 3.4, c + 1.3, .6); battach(ball, t0, t1, car); }
    else { battach(ball, snap + 180, t1, car); }
    const ye = clamp(c + (r() - .5) * 16, 6, FW - 6), ym = clamp((car.y0 + ye) / 2 + (r() - .5) * 5, 4, FW - 4);
    go(car, t0, t0 + run * .45, X0 + (xe - X0) * .35, ym); go(car, t0 + run * .45, t1, xe, ye);
    lineWork(t1, 1.8);
    O.WR1.tr = []; go(O.WR1, snap, snap + 900, X0 + d * 5, O.WR1.y0 + 3); go(O.WR2, snap, snap + 900, X0 + d * 5, O.WR2.y0 - 3); go(O.WR3, snap, snap + 900, X0 + d * 4, O.WR3.y0 - 2);
    ['LB1', 'LB2', 'LB3'].forEach((k, i) => go(D[k], snap + 250, t1, xe + d * (.7 + i * .5), ye + (i - 1) * 1.6));
    ['S1', 'S2'].forEach((k, i) => go(D[k], snap + 450, t1 + 100, xe + d * (1.4 + i * .6), ye + (i ? 1.8 : -1.8)));
    ['DL1', 'DL2', 'DL3', 'DL4'].forEach((k, i) => go(D[k], snap + 500, t1, xe + d * (.2 + i * .2), ye + (i - 1.5) * 1.3));
    go(D.CB1, snap, snap + 900, X0 + d * 8, 7); go(D.CB2, snap, snap + 900, X0 + d * 8, FW - 7);
    T = t1 + 700; tackle = {t: t1, x: xe, y: ye};
    if (ty === 'fum'){
      const tf = t0 + run * .8; const pf = epos(car, tf);
      ball.att = []; bgo(ball, 0, 1, X0, c, 0); ball.tr = []; battach(ball, snap + 180, tf, car);
      const ex = pf[0] + d * 1.5, ey = pf[1] + 1.5; bgo(ball, tf, tf + 350, ex, ey, 1.2);
      go(D.LB2, tf, tf + 450, ex, ey); battach(ball, tf + 450, tf + 1400, D.LB2); go(D.LB2, tf + 450, tf + 1400, ex - d * (p.pick6 ? 40 : 3), ey);
      T = tf + 2000; tackle = null; ev.shake = tf;
    }
    ev.carrier = car; if (ty === 'twopt') T = Math.max(T, 2400);
  } else if (ty === 'pass' || ty === 'inc' || ty === 'int' || ty === 'sack'){
    const qb = O.QB; bgo(ball, snap, snap + 200, qb.x0, qb.y0, 0); battach(ball, snap + 200, snap + 1100, qb);
    const drop = (ty === 'sack') ? 5.6 : 5.0 + r() * 1.5;
    go(qb, snap + 150, snap + 800, X0 - d * drop, c);
    lineWork(0, .2);
    ['OL1', 'OL2', 'OL3', 'OL4', 'OL5'].forEach(k => { const e = O[k]; go(e, snap + 500, snap + 900, e.x0 - d * .9, e.y0); });
    if (ty === 'sack'){
      const xe = X1, ye = c + (r() - .5) * 3; const tEnd = snap + 1300;
      go(qb, snap + 800, tEnd, xe, ye); battach(ball, snap + 1100, tEnd, qb);
      const rusher = D.DL2; go(rusher, snap + 300, tEnd, xe + d * .9, ye + .6); go(D.DL3, snap + 300, tEnd, xe + d * 1.3, ye - .9);
      ['LB1', 'LB2', 'LB3'].forEach((k, i) => go(D[k], snap + 200, snap + 900, X0 + d * 3.2, D[k].y0 * .8 + c * .2));
      [['WR1', 11], ['WR2', 11], ['WR3', 9], ['TE', 7], ['RB', 6]].forEach(([k, dd], i) => go(O[k], snap + 100, snap + 1100, X0 + d * dd, O[k].y0 + (i % 2 ? 2 : -2)));
      T = tEnd + 800; tackle = {t: tEnd, x: xe, y: ye}; carrier = qb;
    } else {
      const slotT = sl.slotTgt || 'WR1', tg = O[slotT], tgName = p.tgt;
      const air = (ty === 'inc') ? p.air : p.air;
      const xc = ab(clamp(p.los + (ty === 'pass' ? p.air : p.air), 0, 100));
      const yc = clamp(tg.y0 + (r() - .5) * 10 + (tg.y0 < c ? 3 : -3), 5, FW - 5);
      const tCatch = 1250 + Math.abs(air) * 38, tRel = tCatch - (300 + Math.abs(air) * 30);
      route = [];
      go(tg, snap, snap + (tCatch - snap) * .55, X0 + d * Math.max(air * .85, 1.5), tg.y0 + (yc - tg.y0) * .2); go(tg, snap + (tCatch - snap) * .55, tCatch, xc, yc);
      route = {ent: tg, pts: [[tg.x0, tg.y0], [X0 + d * Math.max(air * .85, 1.5), tg.y0 + (yc - tg.y0) * .2], [xc, yc]]};
      const oth = ['WR1', 'WR2', 'WR3', 'TE', 'RB'].filter(k => k !== slotT);
      oth.forEach((k, i) => { const dd = k === 'RB' ? 4 : 9 + i * 2.5; go(O[k], snap, tCatch + 200, X0 + d * dd, O[k].y0 + (i % 2 ? 4 : -4)); });
      const cbNear = (tg.y0 < c) ? 'CB1' : 'CB2', cbFar = cbNear === 'CB1' ? 'CB2' : 'CB1';
      const covers = [cbNear, 'S1', 'LB2'];
      const dbs = {};
      if (slotT === 'TE' || slotT === 'RB') dbs.main = 'LB' + (slotT === 'TE' ? 3 : 2); else dbs.main = cbNear;
      go(D[dbs.main], snap, tCatch, xc + d * 1.4, yc + (tg.y0 < c ? -1.4 : 1.4));
      [cbFar, 'S2', 'S1'].forEach((k, i) => go(D[k], snap, tCatch, X0 + d * (8 + i * 3), D[k].y0 + (i ? (i === 1 ? -4 : 4) : 0)));
      ['LB1', 'LB2', 'LB3'].filter(k => k !== dbs.main).forEach((k, i) => go(D[k], snap, tCatch, X0 + d * (5 + i), D[k].y0 * .9 + c * .1));
      ['DL1', 'DL2', 'DL3', 'DL4'].forEach((k, i) => go(D[k], snap + 100, snap + 1000, X0 - d * 2.8, c + (i - 1.5) * 2.2));
      const qp = epos(qb, tRel);
      if (ty === 'pass'){
        bgo(ball, tRel, tCatch, xc, yc, 2 + Math.abs(air) * .1); battach(ball, tCatch, tCatch + 3000, tg); ball.att = ball.att.filter(a => a.e !== qb || a.t1 <= tRel); ball.att[0].t1 = tRel;
        const xe = p.td ? endZ + d * 3.2 : X1, ye = clamp(yc + (r() - .5) * 8, 5, FW - 5), run = clamp(250 + Math.abs(xe - xc) * 60, 250, 1700);
        go(tg, tCatch, tCatch + run, xe, ye);
        go(D[dbs.main], tCatch, tCatch + run, xe + d * .9, ye + .9); go(D.S1, tCatch, tCatch + run, xe + d * 2, ye - 2); go(D.S2, tCatch, tCatch + run, xe + d * 2.5, ye + 2.5);
        ['LB1', 'LB2', 'LB3'].forEach((k, i) => go(D[k], tCatch, tCatch + run + 200, xe + d * (2 + i), ye + (i - 1) * 2));
        T = tCatch + run + 800; tackle = {t: tCatch + run, x: xe, y: ye}; carrier = tg; ev.catchT = tCatch;
      } else if (ty === 'inc'){
        ball.att[0].t1 = tRel;
        const over = d * (1 + r() * 3);
        bgo(ball, tRel, tCatch, xc + (r() < .5 ? over : -over * .7), yc + (r() - .5) * 4, 2 + Math.abs(air) * .12);
        const bp = bpos(ball, tCatch); bgo(ball, tCatch, tCatch + 450, bp[0] + d * 2, bp[1] + (r() - .5) * 2, .8);
        T = tCatch + 1500; ev.dropT = tCatch;
      } else {
        ball.att[0].t1 = tRel; const D1 = D[dbs.main];
        bgo(ball, tRel, tCatch, xc + d * .7, yc, 2 + Math.abs(air) * .1); battach(ball, tCatch, tCatch + 2500, D1);
        const xr = p.pick6 ? (off === HOME ? 0 : 100) - d * 3 : xc - d * (6 + r() * 8);
        go(D1, tCatch, tCatch + clamp(Math.abs(xr - xc) * 60, 500, 2300), xr, clamp(yc + (r() - .5) * 14, 5, FW - 5));
        go(tg, tCatch, tCatch + 900, xc - d * 6, yc + 2); go(qb, tCatch, tCatch + 1500, X0 - d * 3, c);
        T = tCatch + clamp(Math.abs(xr - xc) * 60, 500, 2300) + 800; ev.turnover = tCatch; carrier = D1;
      }
    }
  } else if (ty === 'punt'){
    const pnt = O.QB, hold = O.RB, land = ab(clamp(p.to, 0, 100));
    pnt.tag = 'P'; pnt.name = 'Punter';
    go(pnt, snap, snap + 400, X0 - d * 14.5, c); go(hold, snap, snap + 300, X0 - d * 2.5, c - 4);
    O.WR1.tr = []; go(O.WR1, snap, snap + 400, X0 - d * .9, 3); go(O.WR2, snap, snap + 400, X0 - d * .9, FW - 3);
    bgo(ball, snap, snap + 380, X0 - d * 14.5, c, 0.8);
    const kick = snap + 700, fl = 1500 + Math.abs(land - X0) * 14;
    battach(ball, snap + 380, kick, pnt); bgo(ball, kick, kick + fl, land, c + (r() - .5) * 8, 14 + Math.abs(land - X0) * .12);
    const ret = D.S1; go(ret, snap, kick + fl, land + d * .6, c + (r() - .5) * 6);
    ['OL1', 'OL2', 'OL3', 'OL4', 'OL5', 'TE'].forEach(k => go(O[k], kick, kick + fl, X0 + d * (Math.abs(land - X0) * .8), O[k].y0 + (r() - .5) * 8));
    go(O.RB, kick, kick + fl, X0 + d * (Math.abs(land - X0) * .8), c); go(O.WR3, kick, kick + fl, land - d * 2, c + 3);
    battach(ball, kick + fl, kick + fl + 1400, ret); T = kick + fl + 1400;
    ev.kickT = kick;
  } else if (ty === 'fg' || ty === 'xp'){
    const kicker = O.QB, holder = O.RB, post = off === HOME ? 110 : -10;
    kicker.tag = 'K'; kicker.name = 'Kicker'; holder.tag = 'H'; holder.name = 'Holder';
    go(kicker, snap, snap + 350, X0 - d * 9.6, c + 1.6); go(holder, snap, snap + 250, X0 - d * 6.8, c - .2);
    ['WR1', 'WR2'].forEach((k, i) => go(O[k], snap, snap + 300, X0 - d * .9, i ? c + 9 : c - 9)); go(O.WR3, snap, snap + 300, X0 - d * .9, c - 11);
    ['DL1', 'DL2', 'DL3', 'DL4'].forEach((k, i) => go(D[k], snap + 350, snap + 1100, X0 + d * .4, D[k].y0 * .9 + c * .1));
    D.S1.tr = []; D.S2.tr = []; go(D.S1, snap, snap + 400, post - d * 2, c - 3); go(D.S2, snap, snap + 400, post - d * 2, c + 3);
    bgo(ball, snap, snap + 280, X0 - d * 6.8, c - .2, .6); battach(ball, snap + 280, snap + 750, holder);
    const kick = snap + 800; go(kicker, kick - 150, kick + 150, X0 - d * 8.2, c + .4);
    const wide = p.good ? 0 : (r() < .5 ? 5 : -5) * (r() < .5 ? 1 : .7);
    const dist = Math.abs(post - X0), fl = 700 + dist * 15;
    bgo(ball, kick, kick + fl, post, c + (p.good ? (r() - .5) * 1.8 : wide + (wide > 0 ? 1 : -1)), 7 + dist * .08);
    T = kick + fl + 1100; ev.kickT = kick; ev.fg = true;
  } else {  // kneel
    const qb = O.QB; bgo(ball, snap, snap + 200, qb.x0, qb.y0, 0); battach(ball, snap + 200, snap + 1500, qb); go(qb, snap + 200, snap + 700, X0 - d * 4.2, c); ev.kneel = snap + 700; carrier = qb; T = 1900;
  }
  return {p, d, X0, X1, O, D, ball, T, tackle, route, ev, carrier, endZ};
}
const allEnts = () => Object.values(cur.O).concat(Object.values(cur.D));

// ---------- stats so far ----------
const CUM = [];
(function(){
  const base = () => ({plays: 0, yds: 0, pass: 0, rush: 0, to: 0, first: 0});
  let s = {}; s[HOME] = base(); s[AWAY] = base(); CUM.push(JSON.parse(JSON.stringify(s)));
  P.forEach(p => {
    const o = s[p.team];
    if (['run', 'pass', 'inc', 'sack', 'int', 'fum'].includes(p.type)){ o.plays++; if (p.type !== 'inc' && p.type !== 'int') o.yds += p.yds; if (p.type === 'run') o.rush += p.yds; if (p.type === 'pass' || p.type === 'sack') o.pass += p.yds; if (p.first || p.td) o.first++; }
    if (p.type === 'int' || p.type === 'fum') o.to++;
    CUM.push(JSON.parse(JSON.stringify(s)));
  });
})();
function lineFor(name, team, upto){
  let pa = [0, 0, 0, 0, 0], ru = [0, 0, 0], rc = [0, 0, 0, 0];
  for (let i = 0; i <= upto && i < NP; i++){
    const p = P[i]; if (p.team !== team) continue;
    if (p.who === name){
      if (p.type === 'run' || p.type === 'fum' && p.who === name && false){ ru[0]++; ru[1] += p.yds; if (p.td) ru[2]++; }
      if (p.type === 'pass'){ pa[0]++; pa[1]++; pa[2] += p.yds; if (p.td) pa[3]++; }
      if (p.type === 'inc'){ pa[1]++; } if (p.type === 'int'){ pa[1]++; pa[4]++; } if (p.type === 'sack'){ pa[2] += p.yds; }
    }
    if (p.tgt === name){ if (p.type === 'pass'){ rc[0]++; rc[1]++; rc[2] += p.yds; if (p.td) rc[3]++; } else if (p.type === 'inc' || p.type === 'int'){ rc[0]++; } }
  }
  const out = [];
  if (pa[1]) out.push('Passing: ' + pa[0] + '/' + pa[1] + ', ' + pa[2] + ' yds' + (pa[3] ? ', ' + pa[3] + ' TD' : '') + (pa[4] ? ', ' + pa[4] + ' INT' : ''));
  if (ru[0]) out.push('Rushing: ' + ru[0] + ' car, ' + ru[1] + ' yds' + (ru[2] ? ', ' + ru[2] + ' TD' : ''));
  if (rc[0]) out.push('Receiving: ' + rc[1] + ' rec on ' + rc[0] + ' tgt, ' + rc[2] + ' yds' + (rc[3] ? ', ' + rc[3] + ' TD' : ''));
  return out;
}
function showCard(e){
  const el = $('card'); if (!e){ el.style.display = 'none'; selected = null; return; }
  selected = e;
  const team = e.side === 'o' ? cur.p.team : cur.p.opp, upto = (t >= cur.T * .85) ? pi : pi - 1;
  let h = '<b style="color:' + lite(colOf(team), .35) + '">' + e.name + '</b> &middot; ' + team + ' &middot; ' + (e.tag ? e.name : {QB: 'Quarterback', RB: 'Running back', WR1: 'Wide receiver', WR2: 'Wide receiver', WR3: 'Wide receiver', TE: 'Tight end', DL: 'Defensive line', LB: 'Linebacker', CB: 'Cornerback', S: 'Safety'}[e.role] || (e.role.startsWith('OL') ? 'Offensive line' : e.role));
  if (e.side === 'o' && !e.role.startsWith('OL')){ const L = lineFor(e.name, team, upto); h += L.length ? '<br>' + L.join('<br>') : '<br><span style="color:#8a93a3">No stats yet this game</span>'; }
  el.innerHTML = h; el.style.display = 'block'; fit();
}

// ---------- drawing the action ----------
function drawAction(tm){
  const p = cur.p, d = cur.d;
  // line of scrimmage and first-down line
  cx.lineWidth = 3; cx.setLineDash([7, 5]); cx.strokeStyle = '#4da3ff'; line(cur.X0, 0, cur.X0, FW);
  if (p.togo > 0 && p.down > 0 && p.type !== 'kneel'){ const fx = clamp(cur.X0 + d * p.togo, -0.2, 100.2); cx.setLineDash([]); cx.strokeStyle = '#f5c518'; cx.lineWidth = 4; line(fx, 0, fx, FW); }
  cx.setLineDash([]); cx.lineWidth = 1;
  if (routesOn && cur.route && tm < cur.T * .8){ cx.strokeStyle = 'rgba(255,255,255,.55)'; cx.setLineDash([5, 5]); cx.lineWidth = 2; cx.beginPath(); cur.route.pts.forEach((q, k) => { const a = S(q[0], q[1]); k ? cx.lineTo(a[0], a[1]) : cx.moveTo(a[0], a[1]); }); cx.stroke(); cx.setLineDash([]); cx.lineWidth = 1; }
  const R = Math.max(6.5, ppy * 1.25), ents = allEnts();
  const bp = bpos(cur.ball, tm), holder = cur.ball.att.find(a => tm >= a.t0 && tm <= a.t1);
  ents.sort((a, b) => (a.side === 'd') - (b.side === 'd'));
  ents.forEach(e => {
    const q = epos(e, tm), a = S(q[0], q[1]); e._sx = a[0]; e._sy = a[1];
    const isO = e.side === 'o', team = isO ? p.team : p.opp, col = colOf(team);
    cx.beginPath(); cx.arc(a[0], a[1], R, 0, 7); cx.fillStyle = col; cx.fill();
    cx.lineWidth = (holder && holder.e === e) ? 3 : 1.6; cx.strokeStyle = (holder && holder.e === e) ? '#f5c518' : (isO ? '#fff' : 'rgba(0,0,0,.55)'); cx.stroke();
    const jn = (isO && G.nums) ? G.nums[e.name] : null;
    const lab = e.tag || (jn ? String(jn) : (isO ? (e.role.startsWith('OL') ? '' : e.role.replace(/[0-9]/g, '')) : e.role));
    if (lab){ cx.fillStyle = '#fff'; cx.font = 'bold ' + Math.round(R * .85) + 'px system-ui'; cx.textAlign = 'center'; cx.textBaseline = 'middle'; cx.fillText(lab, a[0], a[1] + .5); }
    if (selected === e){ cx.strokeStyle = '#4da3ff'; cx.lineWidth = 3; cx.beginPath(); cx.arc(a[0], a[1], R + 4, 0, 7); cx.stroke(); }
  });
  // name tags on the players involved
  const tag = (e) => { if (!e || !e.name || e.name.length < 4 || e.side !== 'o' && false) return; const last = e.name.split(' ').slice(-1)[0]; cx.font = 'bold ' + Math.round(Math.max(11, R * .95)) + 'px system-ui'; cx.textAlign = 'center'; cx.textBaseline = 'alphabetic'; const w = cx.measureText(last).width + 8; cx.fillStyle = 'rgba(10,12,16,.8)'; cx.fillRect(e._sx - w / 2, e._sy - R - 17, w, 15); cx.fillStyle = '#fff'; cx.fillText(last, e._sx, e._sy - R - 5); };
  [cur.O.QB, cur.carrier && cur.carrier.side === 'o' ? cur.carrier : null, cur.route ? cur.route.ent : null].forEach(tag);
  // ball
  const b = S(bp[0], bp[1]), lift = bp[2] * ppy * .9;
  if (bp[2] > 0.05){ cx.fillStyle = 'rgba(0,0,0,.28)'; cx.beginPath(); cx.ellipse(b[0], b[1], R * .7, R * .38, 0, 0, 7); cx.fill(); }
  cx.save(); cx.translate(b[0], b[1] - lift); cx.rotate(vertical ? Math.PI / 2 : 0); const bs = 1 + bp[2] * .035;
  cx.fillStyle = '#8b4513'; cx.strokeStyle = '#fff'; cx.lineWidth = 1.6; cx.beginPath(); cx.ellipse(0, 0, R * .78 * bs, R * .46 * bs, 0, 0, 7); cx.fill(); cx.stroke(); cx.beginPath(); cx.moveTo(-R * .3, 0); cx.lineTo(R * .3, 0); cx.stroke(); cx.restore();
  // tackle burst
  if (cur.tackle && tm >= cur.tackle.t && tm < cur.tackle.t + 500){ const a = S(cur.tackle.x, cur.tackle.y), u = (tm - cur.tackle.t) / 500; cx.strokeStyle = 'rgba(255,255,255,' + (1 - u) + ')'; cx.lineWidth = 3; cx.beginPath(); cx.arc(a[0], a[1], R * (1 + u * 2.2), 0, 7); cx.stroke(); }
}

// ---------- info, banner, lists ----------
const KIND = {td: 'TOUCHDOWN', fg: 'FIELD GOAL', int: 'INTERCEPTION', fum: 'FUMBLE', twopt: 'TWO-POINT TRY', sack: 'SACK', punt: 'PUNT', xp: 'EXTRA POINT'};
function bannerFor(p){
  if (p.pick6) return [p.type === 'int' ? 'PICK-SIX' : 'SCOOP AND SCORE', '#ef4444'];
  if (p.td) return ['TOUCHDOWN', '#f5c518'];
  if (p.type === 'fg') return [p.good ? 'FIELD GOAL IS GOOD' : 'FIELD GOAL NO GOOD', p.good ? '#7bd88f' : '#f59e0b'];
  if (p.type === 'int') return ['INTERCEPTED', '#ef4444'];
  if (p.type === 'fum') return ['FUMBLE - TURNOVER', '#ef4444'];
  if (p.type === 'xp') return [p.good ? 'EXTRA POINT GOOD' : 'EXTRA POINT MISSED', p.good ? '#7bd88f' : '#f59e0b'];
  if (p.type === 'twopt') return [p.good ? 'TWO-POINT GOOD' : 'TWO-POINT NO GOOD', p.good ? '#7bd88f' : '#f59e0b'];
  if (p.type === 'sack') return ['SACK', '#f59e0b'];
  if (p.first && p.yds >= 15) return ['BIG GAIN - FIRST DOWN', '#7bd88f'];
  return null;
}
function qtrLabel(p){ const m = Math.floor(p.clock / 60), s = p.clock % 60; return p.per + ' ' + m + ':' + String(s).padStart(2, '0'); }
function refresh(){
  const p = P[pi], done = (t >= cur.T * .85), tail = done ? p.text : '...';
  $('scoreH').textContent = done ? p.hs : p.hs0; $('scoreA').textContent = done ? p.as : p.as0;
  $('mid').innerHTML = '<b>' + qtrLabel(p) + '</b><br>Play ' + (pi + 1) + ' of ' + NP;
  let sit = (p.at || '') + (p.type === 'kneel' ? ' - victory formation' : '');
  $('info').innerHTML = '<span class="at"><b style="color:' + lite(colOf(p.team), .35) + '">' + p.team + '</b> ball &middot; ' + sit + '</span><br>' + tail;
  const c = CUM[done ? pi + 1 : pi];
  let h = '<table><tr><th>Team stats so far</th><th>Plays</th><th>Yards</th><th>Pass</th><th>Rush</th><th>1st dn</th><th>TO</th></tr>';
  [AWAY, HOME].forEach(tm => { const x = c[tm]; h += '<tr><td style="color:' + lite(colOf(tm), .35) + ';font-weight:700">' + tm + '</td><td>' + x.plays + '</td><td>' + x.yds + '</td><td>' + x.pass + '</td><td>' + x.rush + '</td><td>' + x.first + '</td><td>' + x.to + '</td></tr>'; });
  $('stats').innerHTML = h + '</table>';
  $('slider').value = pi + 1;
  document.querySelectorAll('#list tr.row').forEach(r => { const i = +r.dataset.i; r.classList.toggle('cur', i === pi); r.classList.toggle('dim', i > pi); });
  const row = document.querySelector('#list tr.cur'); if (row){ const box = $('listbox'); const top = row.offsetTop, bot = top + row.offsetHeight; if (top < box.scrollTop + 30 || bot > box.scrollTop + box.clientHeight - 10) box.scrollTop = Math.max(0, top - 80); }
}
function buildList(){
  let h = '<tr><th>#</th><th>Clock</th><th>Team</th><th>Down</th><th style="text-align:left">Play</th><th>Score</th></tr>';
  P.forEach((p, i) => {
    const show = filt === 'all' || (filt === 'key' && p.key) || (filt === 'score' && (p.td || p.pick6 || (p.type === 'fg' && p.good) || (p.type === 'xp' && p.good) || (p.type === 'twopt' && p.good)));
    if (!show) return;
    h += '<tr class="row' + (p.key ? ' key' : '') + '" data-i="' + i + '"><td>' + (i + 1) + '</td><td>' + qtrLabel(p) + '</td><td style="color:' + lite(colOf(p.team), .35) + ';font-weight:700">' + p.team + '</td><td>' + (p.down ? p.down + '&amp;' + (p.togo || 'G') : '-') + '</td><td class="tx">' + p.text + '</td><td>' + p.as + '-' + p.hs + '</td></tr>';
  });
  $('list').innerHTML = h;
  $('list').querySelectorAll('tr.row').forEach(r => r.addEventListener('click', () => jumpTo(+r.dataset.i)));
  const f = $('flt'); f.innerHTML = '';
  [['all', 'All plays'], ['key', 'Key plays'], ['score', 'Scoring']].forEach(([k, lab]) => { const b = document.createElement('button'); b.textContent = lab; b.className = filt === k ? 'on' : ''; b.onclick = () => { filt = k; buildList(); refresh(); }; f.appendChild(b); });
}

// ---------- win chance and drive strip ----------
const WPL = 40, WPR = 10, WPT = 14, WPB = 18;
function wpX(i){ return WPL + (wc.clientWidth - WPL - WPR) * i / NP; }
function wpY(p){ return WPT + (wc.clientHeight - WPT - WPB) * (1 - p); }
function drawWP(){
  wx.setTransform(dpr, 0, 0, dpr, 0, 0); const w = wc.clientWidth, h = wc.clientHeight, wp = G.wp;
  wx.fillStyle = '#1b212c'; wx.fillRect(0, 0, w, h); wx.font = '11px system-ui'; wx.lineWidth = 1;
  [0, .25, .5, .75, 1].forEach(q => { wx.strokeStyle = (q === .5) ? '#56627a' : '#2c3442'; wx.setLineDash(q === .5 ? [5, 4] : []); wx.beginPath(); wx.moveTo(WPL, wpY(q)); wx.lineTo(w - WPR, wpY(q)); wx.stroke(); wx.fillStyle = '#8a93a3'; wx.textAlign = 'right'; wx.fillText(Math.round(q * 100) + '%', WPL - 5, wpY(q) + 4); });
  wx.setLineDash([]); wx.textAlign = 'left'; wx.font = 'bold 11px system-ui';
  wx.fillStyle = lite(G.hc, .35); wx.fillText(HOME + ' win chance', WPL + 6, WPT + 11); wx.fillStyle = lite(G.ac, .35); wx.fillText(AWAY + ' win chance', WPL + 6, h - WPB - 5);
  const upto = Math.max((t >= cur.T * .85 ? pi + 1 : pi), 0);
  wx.strokeStyle = 'rgba(255,255,255,.16)'; wx.lineWidth = 2; wx.beginPath(); wp.forEach((q, i) => i ? wx.lineTo(wpX(i), wpY(q)) : wx.moveTo(wpX(i), wpY(q))); wx.stroke();
  for (let i = 1; i <= upto && i < wp.length; i++){ const mid = (wp[i] + wp[i - 1]) / 2; wx.fillStyle = mid >= .5 ? G.hc : G.ac; wx.globalAlpha = .45; wx.beginPath(); wx.moveTo(wpX(i - 1), wpY(.5)); wx.lineTo(wpX(i - 1), wpY(wp[i - 1])); wx.lineTo(wpX(i), wpY(wp[i])); wx.lineTo(wpX(i), wpY(.5)); wx.closePath(); wx.fill(); wx.globalAlpha = 1; }
  wx.strokeStyle = '#fff'; wx.lineWidth = 2.5; wx.beginPath(); for (let i = 0; i <= upto && i < wp.length; i++) i ? wx.lineTo(wpX(i), wpY(wp[i])) : wx.moveTo(wpX(i), wpY(wp[i])); wx.stroke();
  const k = Math.min(upto, wp.length - 1), pv = wp[k]; wx.fillStyle = '#fff'; wx.beginPath(); wx.arc(wpX(k), wpY(pv), 4.5, 0, 7); wx.fill();
  const lead = pv >= .5 ? HOME : AWAY, shownP = Math.round(Math.max(pv, 1 - pv) * 100);
  wx.fillStyle = '#e8eaed'; wx.font = 'bold 12px system-ui'; const rightSide = k > NP * .75; wx.textAlign = rightSide ? 'right' : 'left'; wx.fillText(lead + ' ' + shownP + '%', wpX(k) + (rightSide ? -9 : 9), wpY(pv) + (pv >= .5 ? 15 : -7));
  P.forEach((p, i) => { if (p.td || p.pick6){ wx.fillStyle = '#f5c518'; wx.beginPath(); wx.arc(wpX(i + 1), h - WPB + 6, 2.6, 0, 7); wx.fill(); } });
  wx.fillStyle = '#8a93a3'; wx.font = '10px system-ui'; wx.textAlign = 'center'; wx.fillText('Kickoff', wpX(0) + 14, h - 3); wx.fillText('Final', wpX(NP) - 8, h - 3);
}
wc.addEventListener('click', e => { const r = wc.getBoundingClientRect(), x = e.clientX - r.left; jumpTo(clamp(Math.round((x - WPL) / (r.width - WPL - WPR) * NP) - 1, 0, NP - 1)); });
function drawStrip(tm){
  sx.setTransform(dpr, 0, 0, dpr, 0, 0); const w = sc.clientWidth, h = 46, L = 28, Wd = w - 56;
  sx.fillStyle = '#1b212c'; sx.fillRect(0, 0, w, h);
  sx.fillStyle = G.hc; sx.fillRect(4, 8, 22, 26); sx.fillStyle = G.ac; sx.fillRect(w - 26, 8, 22, 26);
  sx.fillStyle = '#2a6833'; sx.fillRect(L, 8, Wd, 26); sx.strokeStyle = 'rgba(255,255,255,.35)'; sx.lineWidth = 1;
  for (let i = 1; i < 10; i++){ sx.beginPath(); sx.moveTo(L + Wd * i / 10, 8); sx.lineTo(L + Wd * i / 10, 34); sx.stroke(); }
  const X = v => L + Wd * v / 100, p = cur.p, d = cur.d;
  // drive so far
  let first = pi; while (first > 0 && P[first - 1].d === p.d) first--;
  const sxs = (P[first].team === HOME ? P[first].los : 100 - P[first].los), bp = bpos(cur.ball, tm);
  sx.fillStyle = colOf(p.team); sx.globalAlpha = .55; sx.fillRect(Math.min(X(sxs), X(clamp(bp[0], 0, 100))), 17, Math.abs(X(clamp(bp[0], 0, 100)) - X(sxs)), 8); sx.globalAlpha = 1;
  sx.strokeStyle = '#4da3ff'; sx.lineWidth = 2; sx.beginPath(); sx.moveTo(X(cur.X0), 6); sx.lineTo(X(cur.X0), 36); sx.stroke();
  if (p.togo > 0 && p.down > 0){ sx.strokeStyle = '#f5c518'; sx.beginPath(); sx.moveTo(X(clamp(cur.X0 + d * p.togo, 0, 100)), 6); sx.lineTo(X(clamp(cur.X0 + d * p.togo, 0, 100)), 36); sx.stroke(); }
  sx.fillStyle = '#8b4513'; sx.strokeStyle = '#fff'; sx.lineWidth = 1.5; sx.beginPath(); sx.ellipse(X(clamp(bp[0], 0, 100)), 21, 6, 3.6, 0, 0, 7); sx.fill(); sx.stroke();
  sx.fillStyle = '#9aa4b5'; sx.font = '10px system-ui'; sx.textAlign = 'center'; sx.fillText(HOME + ' end zone', 15, 44); sx.fillText(AWAY + ' end zone', w - 15, 44); sx.fillText('Drive ' + (p.d + 1) + (d > 0 ? '  ▶' : '  ◀'), w / 2, 44);
}

// ---------- celebration ----------
function celebrate(p){
  if (!celebOn) return;
  const scorer = p.pick6 ? p.opp : p.team, big = p.td || p.pick6, n = big ? 130 : 40, base = colOf(scorer), pal = [lite(base, .2), '#ffffff', '#f5c518', base];
  const a = S(cur.endZ + (cur.d > 0 ? 3 : -3) * (p.pick6 ? -1 : 1), CY);
  for (let i = 0; i < n; i++){ const an = Math.random() * 6.28, v = 2 + Math.random() * (big ? 9 : 5); parts.push({x: a[0], y: a[1], vx: Math.cos(an) * v, vy: Math.sin(an) * v - 3, life: 1, col: pal[i % 4], s: 3 + Math.random() * 4, r: Math.random() * 6}); }
  const el = $(scorer === HOME ? 'scoreH' : 'scoreA'); el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
}
function drawParts(){
  if (!parts.length) return; const k = pdt / 16; parts = parts.filter(q => q.life > 0);
  parts.forEach(q => { q.vy += .22 * k; q.x += q.vx * k; q.y += q.vy * k; q.vx *= .995; q.life -= pdt / 1700; q.r += .2 * k; cx.globalAlpha = Math.max(q.life, 0); cx.fillStyle = q.col; cx.save(); cx.translate(q.x, q.y); cx.rotate(q.r); cx.fillRect(-q.s / 2, -q.s / 4, q.s, q.s / 2); cx.restore(); });
  cx.globalAlpha = 1;
}

// ---------- control ----------
function load(i){ pi = clamp(i, 0, NP - 1); cur = buildPlay(P[pi]); t = 0; phase = 'play'; holdT = 0; celebrated = -1; showCard(null); const b = $('banner'); b.style.opacity = 0; camTarget = clamp(cur.X0 + cur.d * 8, 22, 78); if (!vertical) camTarget = 50; refresh(); }
function jumpTo(i){ load(i); camX = camTarget; }
function setLabel(){ $('bPlay').innerHTML = playing ? '&#9208; Pause' : '&#9654; Play'; }
function holdFor(p){ return (p.td || p.pick6 || p.type === 'fg' && p.good) ? 2100 : (p.key ? 1300 : 450); }
function tick(dt){
  if (phase === 'play'){
    t += dt;
    if (t >= cur.T){ t = cur.T; phase = 'hold'; holdT = 0; refresh(); const bn = bannerFor(cur.p); if (bn){ const b = $('banner'); b.innerHTML = '<b style="color:' + bn[1] + '">' + bn[0] + '</b><span>' + cur.p.text + '</span>'; b.style.opacity = 1; } if ((cur.p.td || cur.p.pick6) || (cur.p.type === 'fg' && cur.p.good) || (cur.p.type === 'twopt' && cur.p.good)){ if (celebrated !== pi){ celebrated = pi; celebrate(cur.p); } } }
    else if (t >= cur.T * .85 && celebrated === -1) refresh();
  } else {
    holdT += dt;
    if (holdT >= holdFor(cur.p)){ if (pi < NP - 1){ load(pi + 1); } else { playing = false; setLabel(); } }
  }
}
function loop(now){
  if (last === null) last = now; const raw = Math.min(now - last, 100); pdt = raw; last = now;
  if (playing) tick(raw * speed);
  if (vertical){ const bp = bpos(cur.ball, t); const target = clamp(Math.abs(bp[0] - cur.X0) > 0 ? (cur.X0 + bp[0]) / 2 : cur.X0, 0, 100); camX += (clamp(target + cur.d * 6, VIEW / 2 - 15, 110 - VIEW / 2 + 15) - camX) * Math.min(1, raw / 220); }
  drawField(); drawAction(t); drawParts(); drawStrip(t); drawWP();
  if (!playing && phase === 'hold' && $('banner').style.opacity === '0' && false) {}
  requestAnimationFrame(loop);
}
cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect(), mx = (e.clientX - r.left) * cssW / r.width, my = (e.clientY - r.top) * cssH / r.height; let best = null, bd = 1e9;
  allEnts().forEach(en => { const dd = Math.hypot(en._sx - mx, en._sy - my); if (dd < bd){ bd = dd; best = en; } });
  if (best && bd < Math.max(18, ppy * 2.4)) showCard(best === selected ? null : best); else { showCard(null); playing = !playing; setLabel(); }
});
$('bPlay').onclick = () => { if (!playing && pi === NP - 1 && phase === 'hold') jumpTo(0); playing = !playing; setLabel(); };
$('bPrev').onclick = () => jumpTo(pi - 1); $('bNext').onclick = () => jumpTo(pi + 1);
$('bDPrev').onclick = () => { let i = pi; const dd = P[pi].d; while (i > 0 && P[i - 1].d === dd) i--; if (i === pi && i > 0){ i = pi - 1; while (i > 0 && P[i - 1].d === P[pi - 1].d) i--; } jumpTo(i); };
$('bDNext').onclick = () => { let i = pi; const dd = P[pi].d; while (i < NP - 1 && P[i].d === dd) i++; jumpTo(i); };
$('bKey').onclick = () => { let i = pi + 1; while (i < NP && !P[i].key) i++; jumpTo(i >= NP ? 0 : i); };
$('bRep').onclick = () => { const q = playing; jumpTo(pi); playing = true; setLabel(); };
$('spd').onchange = e => { speed = parseFloat(e.target.value); };
$('cel').onchange = e => { celebOn = e.target.checked; }; $('rts').onchange = e => { routesOn = e.target.checked; };
$('slider').oninput = () => jumpTo(parseInt($('slider').value) - 1);
document.addEventListener('keydown', e => { if (e.code === 'Space'){ e.preventDefault(); playing = !playing; setLabel(); } else if (e.code === 'ArrowRight') jumpTo(pi + 1); else if (e.code === 'ArrowLeft') jumpTo(pi - 1); });
let rzT = null; window.addEventListener('resize', () => { clearTimeout(rzT); rzT = setTimeout(() => { resize(); camX = camTarget; }, 120); });
resize(); load(0); camX = camTarget; buildList(); refresh(); setLabel(); fit(); requestAnimationFrame(loop);
setTimeout(fit, 400);
</script>
"""


def field_payload(g, plays, prep, pre=0.0):
    h, a = g["home"], g["away"]
    hc, ac = COLORS.get(h, "#555555"), COLORS.get(a, "#888888")

    def rgb(x):
        n = int(x[1:], 16)
        return (n >> 16, (n >> 8) & 255, n & 255)
    if sum(abs(p - q) for p, q in zip(rgb(hc), rgb(ac))) < 140:
        ac = "#d9dde3" if sum(rgb(hc)) < 600 else "#26303f"
    wp = [round(_wp_home(0.0, pre, 60.0), 4)]
    for p in plays:
        try:
            per = p["per"]
            rem = 0.0 if per == "OT" else (4 - int(per[1])) * 15 + p["clock"] / 60.0
        except Exception:
            rem = 0.0
        wp.append(round(_wp_home(float(p["hs"] - p["as"]), pre, max(rem - 0.2, 0.0)), 4))
    rost = {}
    for t in (h, a):
        c = prep["cast"][t]
        posmap = {n: p for n, p, w in c["targets"]}
        posmap.update({n: p for n, p, w in c["runners"]})
        rost[t] = {"qb": c["qb"], "rb": c["order"].get("RB", []), "wr": c["order"].get("WR", []), "te": c["order"].get("TE", []), "pos": posmap}
    keep = ("i", "d", "n", "team", "opp", "type", "down", "togo", "los", "to", "yds", "who", "tgt", "air", "first", "td", "key", "pick6", "per", "clock", "hs", "as", "hs0", "as0", "text", "at", "good", "dist")
    out = []
    for p in plays:
        q = {k: p.get(k) for k in keep}
        for k in ("first", "td", "key", "pick6", "good"):
            q[k] = bool(q.get(k))
        for k in ("who", "tgt", "air", "dist", "at", "down", "togo"):
            if q.get(k) is None:
                q[k] = "" if k in ("at",) else 0 if k in ("air", "dist", "down", "togo") else None
        q["los"], q["to"] = round(float(q["los"]), 2), round(float(q["to"]), 2)
        out.append(q)
    return {"home": h, "away": a, "hc": hc, "ac": ac, "plays": out, "wp": wp, "rost": rost}


def build_field_html(g, prep, pre=0.0, plays=None, nums=None):
    plays = plays if plays is not None else expand_game(g, prep)
    _pl = field_payload(g, plays, prep, pre)
    _pl["nums"] = nums or {}
    data = json.dumps(_pl).replace("</", "<\\/")
    return FIELD_HTML.replace("__DATA__", data)


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
# BETTING LINES, WEEK SLATE, BACKTEST
# =====================================================================
def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def ml_implied(ml):
    ml = _num(ml)
    if ml is None or ml == 0:
        return None
    return -ml / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def devig(ml_home, ml_away):
    a, b = ml_implied(ml_home), ml_implied(ml_away)
    if a is None or b is None:
        return None
    return a / (a + b)


def market_from_row(r):
    """Betting line for a scheduled game. spread = points the HOME team is favored by (negative = away favored)."""
    if r is None:
        return None
    sp, tot = _num(r["spread_line"]), _num(r["total_line"])
    mlh, mla = _num(r.get("home_moneyline")), _num(r.get("away_moneyline"))
    if sp is None and tot is None and mlh is None:
        return None
    return {"spread": sp, "total": tot, "p_home": devig(mlh, mla), "ml_home": mlh, "ml_away": mla, "week": int(r["week"]), "date": str(r.get("gameday", ""))}


def find_game(sched, season, home, away):
    x = sched[(sched["season"] == season) & sched["home_score"].isna() & (sched["home_team"] == home) & (sched["away_team"] == away)].sort_values("week")
    return x.iloc[0] if len(x) else None


def spread_label(home, away, s):
    if s is None:
        return "-"
    if abs(s) < 0.25:
        return "Pick'em"
    return f"{home} -{abs(s):g}" if s > 0 else f"{away} -{abs(s):g}"


def line_probs(hs, as_, spread, total):
    m, t = hs - as_, hs + as_
    out = {}
    if spread is not None:
        out["home_cover"], out["away_cover"], out["push_spread"] = float((m > spread).mean()), float((m < spread).mean()), float((m == spread).mean())
    if total is not None:
        out["over"], out["under"], out["push_total"] = float((t > total).mean()), float((t < total).mean()), float((t == total).mean())
    return out


def show_line_box(res):
    h, a = res["home"], res["away"]
    hs, as_ = res["game"]["hs"], res["game"]["as"]
    mk = res.get("line")
    st.subheader("Versus the betting market")
    manual = False
    if mk is None:
        st.caption(f"No posted line was found for {a} @ {h} in the schedule data (lines usually appear a few days before kickoff, and only for games where {h} is the listed home team). You can type one in.")
        with st.expander("Compare to a line I type in"):
            st.caption("Spread = points the home team is favored by (use a negative number if the away team is favored).")
            c1, c2 = st.columns(2)
            ms = c1.number_input("Spread", value=0.0, step=0.5, key="man_spread")
            mt = c2.number_input("Total points", value=44.5, step=0.5, key="man_total")
            manual = st.checkbox("Use my line", key="man_use")
        if not manual:
            return
        mk = {"spread": float(ms), "total": float(mt), "p_home": None, "ml_home": None, "ml_away": None, "week": None, "date": "your line"}
    sim_m, sim_t, sim_p = float((hs - as_).mean()), float((hs + as_).mean()), float((hs > as_).mean())
    rows = []
    if mk["spread"] is not None:
        rows.append({"": "Spread", "Market": spread_label(h, a, mk["spread"]), "Simulator": spread_label(h, a, sim_m), f"Difference ({h} margin)": f"{sim_m - mk['spread']:+.1f} pts"})
    if mk["total"] is not None:
        rows.append({"": "Total points", "Market": f"{mk['total']:g}", "Simulator": f"{sim_t:.1f}", f"Difference ({h} margin)": f"{sim_t - mk['total']:+.1f} pts"})
    if mk["p_home"] is not None:
        rows.append({"": f"{h} win chance", "Market": f"{mk['p_home'] * 100:.1f}%", "Simulator": f"{sim_p * 100:.1f}%", f"Difference ({h} margin)": f"{(sim_p - mk['p_home']) * 100:+.1f} pts"})
    if rows:
        df = pd.DataFrame(rows).rename(columns={f"Difference ({h} margin)": "Sim minus market"})
        st.dataframe(df, hide_index=True)
    pr = line_probs(hs, as_, mk["spread"], mk["total"])
    bits = []
    if "home_cover" in pr:
        bits.append(f"{h} covers {pr['home_cover'] * 100:.1f}% | {a} covers {pr['away_cover'] * 100:.1f}%" + (f" | push {pr['push_spread'] * 100:.1f}%" if pr["push_spread"] > 0.005 else ""))
    if "over" in pr:
        bits.append(f"Over {pr['over'] * 100:.1f}% | Under {pr['under'] * 100:.1f}%" + (f" | push {pr['push_total'] * 100:.1f}%" if pr["push_total"] > 0.005 else ""))
    if bits:
        st.write("  \n".join(bits))
    st.caption("A cover or over needs a profit margin: at standard -110 pricing a bet has to win about 52.4% of the time to break even. The simulator doesn't see late lineup news, weather or betting-market information, so read differences as a starting point, not a pick. The backtest tab shows how the model has done against past lines.")


def default_qb(team, qbs_use, qbs_base, inj_map, use_ros):
    """The QB the lineup picker would choose by default, and the QB the team ratings were built around."""
    mine = qbs_use.get(team, [])
    base_list = qbs_base.get(team, []) if use_ros else mine
    s0 = base_list[0] if base_list else (mine[0] if mine else None)
    pick = None
    for q in mine:
        if not unavailable(inj_map.get(q["pid"], {}).get("status")) and not q.get("blocked"):
            pick = q
            break
    if pick is None and mine:
        pick = mine[0]
    adj = (pick["rate"] - s0["rate"]) * DROPBACKS * QB_WEIGHT if (pick is not None and s0 is not None) else 0.0
    return pick, s0, adj


def run_slate(R, S_use, sched, season, week, sims, qbs_use, qbs_base, use_ros, inj_df, use_inj, inj_weeks, situ=None):
    g = sched[(sched["season"] == season) & (sched["week"] == week) & sched["home_score"].isna()].sort_values(["gameday", "home_team"])
    rows = []
    for r in g.to_dict("records"):
        h, a = r["home_team"], r["away_team"]
        if h not in R["off"] or a not in R["off"]:
            continue
        neutral = str(r.get("location", "")).lower() == "neutral"
        have_inj = use_inj and week in inj_weeks
        im = {t: (inj_lookup(inj_df, t, week) if have_inj else {}) for t in (h, a)}
        qh, s0h, ah = default_qb(h, qbs_use, qbs_base, im[h], use_ros)
        qa, s0a, aa = default_qb(a, qbs_use, qbs_base, im[a], use_ros)
        sa, sn = situ_adj(situ, r, h, a, neutral)
        game = simulate_game(R, S_use, h, a, sims, neutral, {h: ah + sa.get(h, 0.0), a: aa + sa.get(a, 0.0)})
        hs, as_ = game["hs"], game["as"]
        mk = market_from_row(r)
        sim_m, sim_t, sim_p = float((hs - as_).mean()), float((hs + as_).mean()), float((hs > as_).mean())
        row = {"Date": str(r.get("gameday", ""))[5:], "Game": f"{a} @ {h}", "Sim favorite": h if sim_p >= 0.5 else a, "Home win %": sim_p * 100, "Sim score (away-home)": f"{as_.mean():.1f} - {hs.mean():.1f}", "Sim margin (home)": sim_m, "Sim total": sim_t}
        if mk:
            pr = line_probs(hs, as_, mk["spread"], mk["total"])
            row.update({"Line (home margin)": mk["spread"], "Sim minus line": (sim_m - mk["spread"]) if mk["spread"] is not None else None, "Home covers %": pr.get("home_cover", float("nan")) * 100,
                        "Total line": mk["total"], "Over %": pr.get("over", float("nan")) * 100, "Market home win %": (mk["p_home"] * 100) if mk["p_home"] is not None else None,
                        "Win % vs market": ((sim_p - mk["p_home"]) * 100) if mk["p_home"] is not None else None})
        def qtxt(t, q, s0, adj):
            if q is None:
                return f"{t}: ?"
            return f"{t}: {q['name']}" + (f" ({adj:+.1f})" if (s0 is not None and q['pid'] != s0['pid']) else "")
        row["QBs used"] = qtxt(a, qa, s0a, aa) + "  |  " + qtxt(h, qh, s0h, ah)
        row["Rest & travel"] = "; ".join(sn)
        pr0 = line_probs(hs, as_, mk["spread"] if mk else None, mk["total"] if mk else None)
        row.update({"_home": h, "_away": a, "_gameday": str(r.get("gameday", "")), "_p_home": sim_p, "_sim_m": sim_m, "_sim_t": sim_t, "_spread": mk["spread"] if mk else None, "_total": mk["total"] if mk else None,
                    "_ml_h": mk["ml_home"] if mk else None, "_ml_a": mk["ml_away"] if mk else None, "_mkt_p": mk["p_home"] if mk else None,
                    "_sp_h_odds": _num(r.get("home_spread_odds")), "_sp_a_odds": _num(r.get("away_spread_odds")), "_over_odds": _num(r.get("over_odds")), "_under_odds": _num(r.get("under_odds")),
                    "_p_hcov": pr0.get("home_cover"), "_p_acov": pr0.get("away_cover"), "_p_over": pr0.get("over"), "_p_under": pr0.get("under"),
                    "_qb_h": (qh["name"] if qh else None), "_qb_a": (qa["name"] if qa else None)})
        row["Injury report"] = "used" if have_inj else "not posted yet"
        rows.append(row)
    return pd.DataFrame(rows)


def _phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


@st.cache_data(show_spinner="Replaying games week by week...")
def backtest(_sched, decay, alpha, scope, version):
    """Walk-forward test of the team ratings: for every week, refit using only earlier games and predict that week."""
    s = _sched.dropna(subset=["home_score", "away_score"]).copy()
    s = s.sort_values(["season", "week"]).reset_index(drop=True)
    teams = sorted(set(s["home_team"]) | set(s["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(s)
    hi, ai = s["home_team"].map(tidx).values, s["away_team"].map(tidx).values
    hfa = np.where(s["location"].astype(str).str.lower().eq("neutral"), 0, 1) if "location" in s.columns else np.ones(m)
    r = np.arange(m)
    X = np.zeros((2 * m, 2 * n + 1))
    X[r, hi] = 1
    X[r, n + ai] = 1
    X[r, -1] = hfa
    X[m + r, ai] = 1
    X[m + r, n + hi] = 1
    X[m + r, -1] = -hfa
    y = np.concatenate([s["home_score"].values, s["away_score"].values]).astype(float)
    key = (s["season"].astype(int) * 100 + s["week"].astype(int)).values
    latest_s = int(s["season"].max())
    targets = [latest_s] if scope == "cur" else [latest_s - 1, latest_s]
    out = []
    for (sea, wk), g in s[s["season"].isin(targets)].groupby(["season", "week"]):
        k = int(sea) * 100 + int(wk)
        train = key < k
        if train.sum() < 300:
            continue
        w = np.where(train, decay ** (int(sea) - s["season"].values), 0.0)
        model = Ridge(alpha=alpha).fit(X, y, sample_weight=np.tile(w, 2))
        c, b0 = model.coef_, float(model.intercept_)
        for i in g.index:
            ha = c[-1] * hfa[i]
            mu_h = b0 + c[hi[i]] + c[n + ai[i]] + ha
            mu_a = b0 + c[ai[i]] + c[n + hi[i]] - ha
            row = s.loc[i]
            out.append({"Season": int(sea), "Week": int(wk), "Home": row["home_team"], "Away": row["away_team"], "mu_h": mu_h, "mu_a": mu_a, "hs": row["home_score"], "as": row["away_score"],
                        "spread": _num(row.get("spread_line")), "total": _num(row.get("total_line")), "p_mkt": devig(row.get("home_moneyline"), row.get("away_moneyline"))})
    df = pd.DataFrame(out)
    if df.empty:
        return df, 13.5
    df["pred_m"] = df["mu_h"] - df["mu_a"]
    df["act_m"] = df["hs"] - df["as"]
    sigma = float(np.sqrt(np.mean((df["act_m"] - df["pred_m"]) ** 2)))
    df["p_home"] = _phi(df["pred_m"] / sigma)
    return df, sigma


def bt_summary(df):
    d = df[df["act_m"] != 0]
    y = (d["act_m"] > 0).astype(float)
    res = {"n": len(df), "acc": float(((d["pred_m"] > 0) == (d["act_m"] > 0)).mean()), "brier": float(((d["p_home"] - y) ** 2).mean()), "mae": float((df["act_m"] - df["pred_m"]).abs().mean())}
    return res


def bt_market(df):
    d = df[df["spread"].notna()].copy()
    if d.empty:
        return None
    nz = d[d["act_m"] != 0]
    out = {"n": len(d), "mae_model": float((d["act_m"] - d["pred_m"]).abs().mean()), "mae_mkt": float((d["act_m"] - d["spread"]).abs().mean())}
    nz2 = nz[nz["spread"] != 0]
    out["acc_model"] = float(((nz2["pred_m"] > 0) == (nz2["act_m"] > 0)).mean()) if len(nz2) else None
    out["acc_mkt"] = float(((nz2["spread"] > 0) == (nz2["act_m"] > 0)).mean()) if len(nz2) else None
    pm = nz[nz["p_mkt"].notna()]
    if len(pm):
        yy = (pm["act_m"] > 0).astype(float)
        out["brier_model"] = float(((pm["p_home"] - yy) ** 2).mean())
        out["brier_mkt"] = float(((pm["p_mkt"] - yy) ** 2).mean())
    # against the spread
    ats = []
    for label, thr in (("Every game", 0.0), ("Model differs from line by 2+ pts", 2.0), ("3+ pts", 3.0), ("5+ pts", 5.0)):
        z = d[(d["pred_m"] - d["spread"]).abs() >= thr]
        z = z[z["act_m"] != z["spread"]]
        if len(z) == 0:
            continue
        pick_home = z["pred_m"] > z["spread"]
        home_cov = z["act_m"] > z["spread"]
        win = (pick_home == home_cov)
        ats.append({"Games": label, "Bets": len(z), "Model right": int(win.sum()), "Hit rate": f"{win.mean() * 100:.1f}%"})
    out["ats"] = pd.DataFrame(ats)
    return out


def bt_calibration(df):
    d = df[df["act_m"] != 0].copy()
    fav = np.maximum(d["p_home"], 1 - d["p_home"])
    won = np.where(d["p_home"] >= 0.5, d["act_m"] > 0, d["act_m"] < 0)
    d["fav"], d["won"] = fav, won
    bins = [0.5, 0.6, 0.7, 0.8, 1.01]
    labs = ["50-60%", "60-70%", "70-80%", "80%+"]
    d["bin"] = pd.cut(d["fav"], bins=bins, labels=labs, right=False)
    g = d.groupby("bin", observed=True).agg(Games=("won", "size"), Predicted=("fav", "mean"), Actual=("won", "mean")).reset_index()
    g["Predicted"] = (g["Predicted"] * 100).round(1)
    g["Actual"] = (g["Actual"] * 100).round(1)
    return g.rename(columns={"bin": "Favorite's predicted chance"})


def tune_grid(_sched, scope, version):
    rows = []
    for dec in (0.5, 0.65, 0.75, 0.9, 1.0):
        for al in (5.0, 10.0, 20.0, 40.0, 80.0):
            df, _ = backtest(_sched, dec, al, scope, version)
            if df.empty:
                continue
            sm = bt_summary(df)
            rows.append({"decay": dec, "alpha": al, "brier": sm["brier"], "mae": sm["mae"], "acc": sm["acc"]})
    return pd.DataFrame(rows)


# =====================================================================
# SEASON + PLAYOFF SIMULATOR
# =====================================================================
DIVS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"], "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"], "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
CONFS = {"AFC": [t for d, ts in DIVS.items() if d.startswith("AFC") for t in ts], "NFC": [t for d, ts in DIVS.items() if d.startswith("NFC") for t in ts]}
TEAM_DIV = {t: d for d, ts in DIVS.items() for t in ts}


def _phi_arr(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def margin_tables(R):
    """Expected home margin for every home/away pair (with home field) and at a neutral site."""
    teams = R["teams"]
    n = len(teams)
    MU, MUN = np.zeros((n, n)), np.zeros((n, n))
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i == j:
                continue
            x, y = expected_points(R, h, a, False)
            MU[i, j] = x - y
            x, y = expected_points(R, h, a, True)
            MUN[i, j] = x - y
    return MU, MUN


def _play(h, a, MU, sig, rg):
    p = _phi_arr(MU[h, a] / sig)
    return np.where(rg.random(h.shape) < p, h, a)


def seed_conference(key, div_cols):
    """key: (sims, 16) strength of each conference team in each simulated season; returns (sims, 7) team columns by seed."""
    win_idx = np.stack([cols[np.argmax(key[:, cols], axis=1)] for cols in div_cols], axis=1)
    wk = np.take_along_axis(key, win_idx, axis=1)
    order = np.argsort(-wk, axis=1)
    top4 = np.take_along_axis(win_idx, order, axis=1)
    k2 = key.copy()
    np.put_along_axis(k2, win_idx, -1e9, axis=1)
    wc = np.argsort(-k2, axis=1)[:, :3]
    return np.concatenate([top4, wc], axis=1)


def playoff_rounds(seeds, MU, sig, rg):
    """seeds: (sims, 7) GLOBAL team indices ordered by seed. Returns dict of arrays of team indices per round."""
    s = seeds
    w27 = _play(s[:, 1], s[:, 6], MU, sig, rg)
    w36 = _play(s[:, 2], s[:, 5], MU, sig, rg)
    w45 = _play(s[:, 3], s[:, 4], MU, sig, rg)
    n27 = np.where(w27 == s[:, 1], 2, 7)
    n36 = np.where(w36 == s[:, 2], 3, 6)
    n45 = np.where(w45 == s[:, 3], 4, 5)
    T4 = np.stack([s[:, 0], w27, w36, w45], axis=1)
    S4 = np.stack([np.ones_like(n27), n27, n36, n45], axis=1)
    o = np.argsort(S4, axis=1)
    T = np.take_along_axis(T4, o, axis=1)
    Sn = np.take_along_axis(S4, o, axis=1)
    d1 = _play(T[:, 0], T[:, 3], MU, sig, rg)
    sd1 = np.where(d1 == T[:, 0], Sn[:, 0], Sn[:, 3])
    d2 = _play(T[:, 1], T[:, 2], MU, sig, rg)
    sd2 = np.where(d2 == T[:, 1], Sn[:, 1], Sn[:, 2])
    host = np.where(sd1 < sd2, d1, d2)
    vis = np.where(sd1 < sd2, d2, d1)
    champ = _play(host, vis, MU, sig, rg)
    return {"divisional": T4, "conf_final": np.stack([d1, d2], axis=1), "champ": champ}


@st.cache_data(show_spinner="Simulating the rest of the season...")
def sim_season(_R, _sched, season, sims, seed, version, decay, alpha):
    R = _R
    teams = R["teams"]
    n = len(teams)
    tidx = {t: i for i, t in enumerate(teams)}
    reg = _sched[_sched["season"] == season]
    if "game_type" in reg.columns:
        reg = reg[reg["game_type"] == "REG"]
    done = reg.dropna(subset=["home_score", "away_score"])
    rest = reg[reg["home_score"].isna() | reg["away_score"].isna()]
    W0, L0, T0, PD0 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for r in done.itertuples(index=False):
        h, a = tidx.get(r.home_team), tidx.get(r.away_team)
        if h is None or a is None:
            continue
        d = float(r.home_score) - float(r.away_score)
        PD0[h] += d
        PD0[a] -= d
        if d > 0:
            W0[h] += 1
            L0[a] += 1
        elif d < 0:
            W0[a] += 1
            L0[h] += 1
        else:
            T0[h] += 1
            T0[a] += 1
    rg = np.random.default_rng(seed)
    G = len(rest)
    hi = rest["home_team"].map(tidx).values.astype(int)
    ai = rest["away_team"].map(tidx).values.astype(int)
    neutral = rest["location"].astype(str).str.lower().eq("neutral").values if "location" in rest.columns else np.zeros(G, dtype=bool)
    MU, MUN = margin_tables(R)
    mu = np.where(neutral, MUN[hi, ai], MU[hi, ai])
    sig = math.sqrt(2.0) * R["sd"]
    M = np.round(mu[None, :] + sig * rg.standard_normal((sims, G)))
    M = np.where(M == 0, np.where(rg.random((sims, G)) < 0.5, 1.0, -1.0), M)
    hw = (M > 0).astype(float)
    H = np.zeros((G, n))
    A = np.zeros((G, n))
    H[np.arange(G), hi] = 1
    A[np.arange(G), ai] = 1
    wins = W0 + 0.5 * T0 + hw @ H + (1 - hw) @ A
    pdiff = PD0 + M @ H - M @ A
    key_all = wins + pdiff / 1000.0 + rg.random((sims, n)) * 1e-6
    res = {"teams": teams, "W0": W0, "L0": L0, "T0": T0, "games_left": G, "exp_wins": wins.mean(0)}
    conf_seeds = {}
    stat = {k: np.zeros(n) for k in ("playoffs", "division", "seed1", "divisional", "conf_final", "super_bowl", "champion")}
    seedcount = np.zeros((n, 7))
    for conf, tl in CONFS.items():
        cols = np.array([tidx[t] for t in tl])
        div_cols = [np.array([tl.index(t) for t in ts]) for d, ts in DIVS.items() if d.startswith(conf)]
        local = seed_conference(key_all[:, cols], div_cols)
        gl = cols[local]
        conf_seeds[conf] = gl
        for k in range(7):
            seedcount[:, k] += np.bincount(gl[:, k], minlength=n)
        stat["playoffs"] += np.bincount(gl.ravel(), minlength=n)
        stat["division"] += np.bincount(gl[:, :4].ravel(), minlength=n)
        stat["seed1"] += np.bincount(gl[:, 0], minlength=n)
    champs = {}
    for conf in CONFS:
        pr = playoff_rounds(conf_seeds[conf], MU, sig, rg)
        stat["divisional"] += np.bincount(pr["divisional"].ravel(), minlength=n)
        stat["conf_final"] += np.bincount(pr["conf_final"].ravel(), minlength=n)
        champs[conf] = pr["champ"]
        stat["super_bowl"] += np.bincount(pr["champ"], minlength=n)
    sb = _play(champs["AFC"], champs["NFC"], MUN, sig, rg)
    stat["champion"] += np.bincount(sb, minlength=n)
    res["stat"] = {k: v / sims for k, v in stat.items()}
    res["seedprob"] = seedcount / sims
    ta = np.array(teams)
    res["sb_pairs"] = pd.Series([f"{a} vs {b}" for a, b in zip(ta[champs["AFC"]], ta[champs["NFC"]])]).value_counts(normalize=True).head(8)
    res["sims"] = sims
    return res


def projected_seeds(res):
    """Most likely team for each seed slot in each conference, assigned greedily without repeats."""
    teams = res["teams"]
    out = {}
    for conf, tl in CONFS.items():
        idx = [teams.index(t) for t in tl]
        used = set()
        row = []
        for k in range(7):
            best = max((res["seedprob"][i, k], i) for i in idx if i not in used)[1]
            used.add(best)
            row.append(teams[best])
        out[conf] = row
    return out


def run_bracket(R, bracket, sims, seed):
    """bracket: {conf: [7 teams by seed]}. Simulates the playoffs from fixed seeds."""
    teams = R["teams"]
    tidx = {t: i for i, t in enumerate(teams)}
    MU, MUN = margin_tables(R)
    sig = math.sqrt(2.0) * R["sd"]
    rg = np.random.default_rng(seed)
    n = len(teams)
    stat = {k: np.zeros(n) for k in ("divisional", "conf_final", "super_bowl", "champion")}
    champs = {}
    for conf in ("AFC", "NFC"):
        sd = np.tile(np.array([tidx[t] for t in bracket[conf]]), (sims, 1))
        pr = playoff_rounds(sd, MU, sig, rg)
        stat["divisional"] += np.bincount(pr["divisional"].ravel(), minlength=n)
        stat["conf_final"] += np.bincount(pr["conf_final"].ravel(), minlength=n)
        stat["super_bowl"] += np.bincount(pr["champ"], minlength=n)
        champs[conf] = pr["champ"]
    sb = _play(champs["AFC"], champs["NFC"], MUN, sig, rg)
    stat["champion"] += np.bincount(sb, minlength=n)
    games = []
    for conf in ("AFC", "NFC"):
        b = bracket[conf]
        for hs_, as_ in ((2, 7), (3, 6), (4, 5)):
            h, a = tidx[b[hs_ - 1]], tidx[b[as_ - 1]]
            p = float(_phi_arr(MU[h, a] / sig))
            games.append({"Conference": conf, "Wild card game": f"({as_}) {b[as_ - 1]} at ({hs_}) {b[hs_ - 1]}", "Home win chance": f"{p * 100:.1f}%"})
    return {k: v / sims for k, v in stat.items()}, pd.DataFrame(games)


# =====================================================================
# BETTING MATH
# =====================================================================
def american_to_dec(o):
    o = float(o)
    return 1.0 + o / 100.0 if o > 0 else 1.0 + 100.0 / (-o)


def make_bets(sdf, bankroll, kelly_frac):
    """Every moneyline / spread / total bet implied by the slate sims, with edge, expected value and a fractional-Kelly stake."""
    rows = []
    for r in sdf.to_dict("records"):
        h, a = r["_home"], r["_away"]
        legs = []
        mlh, mla = _num(r.get("_ml_h")), _num(r.get("_ml_a"))
        if mlh is not None and mla is not None:
            legs.append((f"{h} moneyline", mlh, r["_p_home"], 1 - r["_p_home"]))
            legs.append((f"{a} moneyline", mla, 1 - r["_p_home"], r["_p_home"]))
        sp = _num(r.get("_spread"))
        if sp is not None and _num(r.get("_p_hcov")) is not None:
            oh = _num(r.get("_sp_h_odds")) or -110.0
            oa = _num(r.get("_sp_a_odds")) or -110.0
            legs.append((f"{h} {-sp:+g}" if sp != 0 else f"{h} pick'em", oh, r["_p_hcov"], r["_p_acov"]))
            legs.append((f"{a} {sp:+g}" if sp != 0 else f"{a} pick'em", oa, r["_p_acov"], r["_p_hcov"]))
        tl = _num(r.get("_total"))
        if tl is not None and _num(r.get("_p_over")) is not None:
            oo = _num(r.get("_over_odds")) or -110.0
            ou = _num(r.get("_under_odds")) or -110.0
            legs.append((f"Over {tl:g}", oo, r["_p_over"], r["_p_under"]))
            legs.append((f"Under {tl:g}", ou, r["_p_under"], r["_p_over"]))
        for name, odds, pw, pl in legs:
            dec = american_to_dec(odds)
            b = dec - 1.0
            if pw + pl <= 0:
                continue
            ev = pw * b - pl
            cond = pw / (pw + pl)
            kelly = max(0.0, (pw * b - pl) / (b * (pw + pl)))
            rows.append({"Game": r["Game"], "Bet": name, "Price": f"{odds:+.0f}", "Sim win %": pw * 100, "Needed to break even %": 100.0 / dec, "Edge (pts)": (cond - 1.0 / dec) * 100,
                         "Expected profit per $100": ev * 100, "Stake ($)": bankroll * kelly * kelly_frac, "_ev": ev})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("_ev", ascending=False).reset_index(drop=True)
    return out


# =====================================================================
# PREDICTION TRACKER
# =====================================================================
PRED_COLS = ["saved_at", "season", "week", "home", "away", "gameday", "p_home", "sim_margin", "sim_total", "line_spread", "line_total", "market_p_home", "qb_home", "qb_away", "late"]


def pred_path():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base = os.getcwd()
    return os.path.join(base, "nfl_predictions.csv")


def _clean_preds(df):
    df = df.copy()
    for c in PRED_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[PRED_COLS].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    for c in ("p_home", "sim_margin", "sim_total", "line_spread", "line_total", "market_p_home"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["late"] = df["late"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.dropna(subset=["season", "week", "home", "away"]).reset_index(drop=True)


def load_preds():
    p = pred_path()
    df = None
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
        except Exception:
            df = None
    if df is None:
        df = st.session_state.get("preds_mem")
    if df is None:
        df = pd.DataFrame(columns=PRED_COLS)
    return _clean_preds(df)


def save_preds(df):
    df = _clean_preds(df)
    st.session_state["preds_mem"] = df
    try:
        df.to_csv(pred_path(), index=False)
        return True
    except Exception:
        return False


def add_preds(new):
    old = load_preds()
    new = _clean_preds(new)
    if len(old) and len(new):
        # a pick saved before the game must not be overwritten by one saved after it started
        k = lambda d: d["season"].astype(int).astype(str) + "|" + d["week"].astype(int).astype(str) + "|" + d["home"] + "|" + d["away"]
        keep_old = set(k(old[~old["late"].astype(bool)]))
        new = new[~(new["late"].astype(bool) & k(new).isin(keep_old))]
    allp = pd.concat([old, new], ignore_index=True)
    allp = allp.drop_duplicates(["season", "week", "home", "away"], keep="last")
    return allp, save_preds(allp)


def slate_to_preds(sl, season):
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for r in sl["df"].to_dict("records"):
        gd = pd.Timestamp(r["_gameday"], tz="UTC") if r.get("_gameday") else None
        late = bool(gd is not None and now >= gd + pd.Timedelta(days=1))
        rows.append({"saved_at": now.strftime("%Y-%m-%d %H:%M"), "season": season, "week": sl["week"], "home": r["_home"], "away": r["_away"], "gameday": r.get("_gameday"), "p_home": r["_p_home"],
                     "sim_margin": r["_sim_m"], "sim_total": r["_sim_t"], "line_spread": r.get("_spread"), "line_total": r.get("_total"), "market_p_home": r.get("_mkt_p"),
                     "qb_home": r.get("_qb_h"), "qb_away": r.get("_qb_a"), "late": late})
    return pd.DataFrame(rows)


def grade_preds(preds, sched, include_late=False):
    p = preds if include_late else preds[~preds["late"]]
    s = sched[["season", "week", "home_team", "away_team", "home_score", "away_score"]].rename(columns={"home_team": "home", "away_team": "away"})
    m = p.merge(s, on=["season", "week", "home", "away"], how="left")
    m["done"] = m["home_score"].notna() & m["away_score"].notna()
    m["act_m"] = m["home_score"] - m["away_score"]
    m["act_t"] = m["home_score"] + m["away_score"]
    return m


def tracker_summary(m):
    d = m[m["done"] & (m["act_m"] != 0)]
    out = {"n": int(m["done"].sum())}
    if len(d):
        y = (d["act_m"] > 0).astype(float)
        out["acc"] = float(((d["p_home"] > 0.5) == (d["act_m"] > 0)).mean())
        out["brier"] = float(((d["p_home"] - y) ** 2).mean())
        dm = d[d["market_p_home"].notna()]
        if len(dm):
            ym = (dm["act_m"] > 0).astype(float)
            out["n_m"] = len(dm)
            out["brier_model_m"] = float(((dm["p_home"] - ym) ** 2).mean())
            out["brier_mkt_m"] = float(((dm["market_p_home"] - ym) ** 2).mean())
            out["acc_model_m"] = float(((dm["p_home"] > 0.5) == (dm["act_m"] > 0)).mean())
            out["acc_mkt_m"] = float(((dm["market_p_home"] > 0.5) == (dm["act_m"] > 0)).mean())
    ats = m[m["done"] & m["line_spread"].notna()]
    if len(ats):
        pick_home = ats["sim_margin"] > ats["line_spread"]
        w = int(((pick_home & (ats["act_m"] > ats["line_spread"])) | (~pick_home & (ats["act_m"] < ats["line_spread"]))).sum())
        pu = int((ats["act_m"] == ats["line_spread"]).sum())
        out["ats"] = (w, len(ats) - w - pu, pu)
    ou = m[m["done"] & m["line_total"].notna()]
    if len(ou):
        pick_over = ou["sim_total"] > ou["line_total"]
        w = int(((pick_over & (ou["act_t"] > ou["line_total"])) | (~pick_over & (ou["act_t"] < ou["line_total"]))).sum())
        pu = int((ou["act_t"] == ou["line_total"]).sum())
        out["ou"] = (w, len(ou) - w - pu, pu)
    return out


def units_110(rec):
    return rec[0] * (100.0 / 110.0) - rec[1]


def compute_slate(week, sims):
    S_sl = S_all if (use_style and S_all) else None
    sl = {"week": week, "sims": sims, "df": run_slate(R, S_sl, sched, latest, week, sims, qbs, qbs0, bool((use_ros and not ros_df.empty) or use_depth), inj_df, use_inj, set(weeks), dict(SITU, apply=use_situ)), "ros": bool(use_ros and not ros_df.empty), "ov": len(ov_eff)}
    st.session_state["slate"] = sl
    return sl


# =====================================================================
# BETS: model probabilities, prop board, bet slip, and a per-profile tracker that grades itself
# =====================================================================
import re as _re

STAT_COL = {"Pass yds": ["passing_yards"], "Pass TD": ["passing_tds"], "INT": ["interceptions"], "Rush yds": ["rushing_yards"], "Rush TD": ["rushing_tds"],
            "Rec": ["receptions"], "Rec yds": ["receiving_yards"], "Rec TD": ["receiving_tds"], "Anytime TD": ["rushing_tds", "receiving_tds"]}
BET_COLS = ["id", "profile", "placed_at", "kind", "season", "week", "home", "away", "gameday", "desc", "market", "pick", "line", "odds", "stake", "model_p", "ev", "player", "team", "stat", "status", "pnl", "settled_at", "legs"]


def safe_profile(name):
    s = _re.sub(r"[^A-Za-z0-9_-]", "", str(name or "").strip())[:24]
    return s or "me"


def bets_path(profile):
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base = os.getcwd()
    return os.path.join(base, f"nfl_bets_{safe_profile(profile)}.csv")


def _clean_bets(df):
    df = df.copy()
    for c in BET_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[BET_COLS].copy()
    for c in ("season", "week", "line", "odds", "stake", "model_p", "ev", "pnl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["status"] = df["status"].fillna("open").replace("", "open")
    return df.dropna(subset=["id"]).reset_index(drop=True)


def load_bets(profile):
    p = bets_path(profile)
    df = None
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
        except Exception:
            df = None
    if df is None:
        df = st.session_state.get(f"bets_mem_{safe_profile(profile)}")
    if df is None:
        df = pd.DataFrame(columns=BET_COLS)
    return _clean_bets(df)


def save_bets(df, profile):
    df = _clean_bets(df)
    st.session_state[f"bets_mem_{safe_profile(profile)}"] = df
    try:
        df.to_csv(bets_path(profile), index=False)
        return True
    except Exception:
        return False


def add_bet(profile, row):
    df = load_bets(profile)
    row = dict(row)
    row.update({"id": f"{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S%f')}", "profile": safe_profile(profile), "placed_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"), "status": "open", "pnl": None})
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return save_bets(df, profile)


def split_probs(win, push, loss=None):
    return float(win), float(push), float(1.0 - win - push if loss is None else loss)


def p_from_draws(d, line, side):
    d = np.asarray(d)
    if side == "Over":
        return split_probs((d > line).mean(), (d == line).mean())
    return split_probs((d < line).mean(), (d == line).mean())


def p_game(hs, as_, market, pick, line, home, away):
    """Win, push and loss chances for a game bet. Spread lines are written the way a sportsbook shows them for the team you pick (e.g. -3.5 or +7)."""
    hs, as_ = np.asarray(hs), np.asarray(as_)
    if market == "Moneyline":
        a, b = (hs, as_) if pick == home else (as_, hs)
        return split_probs((a > b).mean(), (a == b).mean())
    if market == "Spread":
        a, b = (hs, as_) if pick == home else (as_, hs)
        x = a - b + line
        return split_probs((x > 0).mean(), (x == 0).mean())
    t = hs + as_
    return split_probs((t > line).mean(), (t == line).mean()) if pick == "Over" else split_probs((t < line).mean(), (t == line).mean())


def bet_math(pw, pp, pl, odds, stake=0.0, bankroll=1000.0, frac=0.25):
    dec = american_to_dec(odds)
    b = dec - 1.0
    ev = pw * b - pl
    be = 1.0 / dec
    kelly = max(0.0, (pw * b - pl) / (b * (pw + pl))) if (pw + pl) > 0 else 0.0
    return {"dec": dec, "ev": ev, "be": be, "kelly": kelly, "stake_k": bankroll * kelly * frac, "edge": (pw / (pw + pl) - be) * 100 if (pw + pl) > 0 else 0.0, "to_win": stake * b}


def best_line(d):
    d = np.sort(np.asarray(d))
    med = float(np.median(d))
    cands = [np.floor(med) + k + 0.5 for k in (-2, -1, 0, 1)]
    cands = [c for c in cands if c >= 0.5] or [0.5]
    best = min(cands, key=lambda c: abs((1.0 - np.searchsorted(d, c, side="right") / len(d)) - 0.5))
    return float(best)


def prop_board(draws):
    rows = []
    for (team, name, stat), d in draws.items():
        d = np.asarray(d)
        line = 0.5 if stat == "Anytime TD" else best_line(d)
        if stat != "Anytime TD" and d.mean() < 0.05:
            continue
        po, pu = float((d > line).mean()), float((d < line).mean())
        rows.append({"Player": name, "Team": team, "Prop": stat, "Avg": round(float(d.mean()), 1), "Line": line, "Over %": po * 100, "Under %": pu * 100})
    out = pd.DataFrame(rows)
    return out.sort_values(["Team", "Player", "Prop"]).reset_index(drop=True) if len(out) else out


def find_any_game(sched, season, home, away):
    x = sched[(sched["season"] == season) & (((sched["home_team"] == home) & (sched["away_team"] == away)) | ((sched["home_team"] == away) & (sched["away_team"] == home)))].copy()
    if x.empty:
        return None
    x["open"] = x["home_score"].isna()
    return x.sort_values(["open", "week"], ascending=[False, True]).iloc[0]


def _settle(status_p, stake, odds):
    if status_p == "won":
        return stake * (american_to_dec(odds) - 1.0)
    if status_p == "lost":
        return -stake
    return 0.0


def grade_bets(bets, sched, pl):
    """Settles every open bet whose game is final: game bets from the schedule's scores, props from the player's real box-score stats."""
    b = bets.copy()
    changed = 0
    for i, r in b[b["status"] == "open"].iterrows():
        g = sched[(sched["season"] == r["season"]) & (sched["week"] == r["week"]) & (sched["home_team"] == r["home"]) & (sched["away_team"] == r["away"])]
        if g.empty or pd.isna(g.iloc[0]["home_score"]) or pd.isna(g.iloc[0]["away_score"]):
            continue
        hs, as_ = float(g.iloc[0]["home_score"]), float(g.iloc[0]["away_score"])
        res = None
        if r["kind"] == "sgp":
            res = grade_sgp(r, hs, as_, pl)
            if res is None:
                continue
        elif r["kind"] == "game":
            if r["market"] == "Moneyline":
                a, o = (hs, as_) if r["pick"] == r["home"] else (as_, hs)
                res = "won" if a > o else ("push" if a == o else "lost")
            elif r["market"] == "Spread":
                a, o = (hs, as_) if r["pick"] == r["home"] else (as_, hs)
                x = a - o + float(r["line"])
                res = "won" if x > 0 else ("push" if x == 0 else "lost")
            else:
                tt = hs + as_
                res = "push" if tt == float(r["line"]) else ("won" if ((tt > float(r["line"])) == (r["pick"] == "Over")) else "lost")
        else:
            cols = STAT_COL.get(r["stat"], [])
            x = pl[(pl["season"] == r["season"]) & (pl["week"] == r["week"]) & (pl["pname"] == r["player"]) & (pl["tm"] == r["team"])]
            if x.empty or not cols:
                res = "void"
            else:
                v = float(sum(x[c].sum() for c in cols if c in x.columns))
                ln = float(r["line"])
                res = "push" if v == ln else ("won" if ((v > ln) == (r["pick"] == "Over")) else "lost")
        b.at[i, "status"] = res
        b.at[i, "pnl"] = _settle(res, float(r["stake"]), float(r["odds"]))
        b.at[i, "settled_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
        changed += 1
    return b, changed


def bet_summary(b):
    s = b[b["status"].isin(["won", "lost", "push"])]
    out = {"n": len(s), "open": int((b["status"] == "open").sum()), "won": int((s["status"] == "won").sum()), "lost": int((s["status"] == "lost").sum()), "push": int((s["status"] == "push").sum())}
    out["staked"] = float(s["stake"].sum())
    out["profit"] = float(s["pnl"].sum())
    out["roi"] = out["profit"] / out["staked"] * 100 if out["staked"] > 0 else 0.0
    dec = s[s["status"].isin(["won", "lost"])]
    out["hit"] = float((dec["status"] == "won").mean() * 100) if len(dec) else None
    out["avg_p"] = float(dec["model_p"].mean() * 100) if len(dec) and dec["model_p"].notna().any() else None
    return out


# =====================================================================
# V7: depth chart starters, same-game parlays, game cards
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_depth(season):
    """Latest nflverse depth chart: {team: {pos: [(gsis_id, name), ...]}} in depth order (starter first)."""
    try:
        d = to_pandas(nfl.load_depth_charts([int(season)]))
    except Exception:
        return {}
    if d is None or len(d) == 0 or "pos_abb" not in d.columns:
        return {}
    d = d[d["pos_abb"].isin(["QB", "RB", "WR", "TE"])].copy()
    d["team"] = d["team"].replace(FIX)
    d = d[d["dt"] == d.groupby("team")["dt"].transform("max")]
    out = {}
    for (t, pos), g in d.sort_values(["pos_rank", "pos_slot"]).groupby(["team", "pos_abb"]):
        seen, lst = set(), []
        for _, r in g.iterrows():
            key = r.get("gsis_id") or r.get("player_name")
            if key in seen:
                continue
            seen.add(key)
            lst.append((r.get("gsis_id"), r.get("player_name")))
        out.setdefault(t, {})[pos] = lst
    return out


def _dc_sort(cands, order):
    """Players on the depth chart first, in depth order; everyone else keeps their usage order."""
    by_id = {pid: i for i, (pid, nm) in enumerate(order) if pid}
    by_nm = {nm: i for i, (pid, nm) in enumerate(order)}
    def rank(c):
        i = by_id.get(c.get("pid"))
        if i is None:
            i = by_nm.get(c.get("name"))
        return i
    ranked = [(rank(c), j, c) for j, c in enumerate(cands)]
    on = sorted([x for x in ranked if x[0] is not None], key=lambda x: x[0])
    off = [x for x in ranked if x[0] is None]
    return [x[2] for x in on] + [x[2] for x in off]


def apply_depth(pool, qbs, depth):
    """Returns copies of the pool and QB lists reordered so the depth-chart starters come first."""
    p2 = {}
    for t, by_pos in pool.items():
        p2[t] = {pos: (_dc_sort(lst, depth.get(t, {}).get(pos, [])) if depth.get(t, {}).get(pos) else list(lst)) for pos, lst in by_pos.items()}
    q2 = {t: (_dc_sort(lst, depth.get(t, {}).get("QB", [])) if depth.get(t, {}).get("QB") else list(lst)) for t, lst in qbs.items()}
    return p2, q2


def _am_from_dec(dec):
    dec = max(float(dec), 1.0001)
    return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)


def leg_label(leg):
    if leg["type"] == "game":
        if leg["market"] == "Moneyline":
            return f"{leg['pick']} moneyline"
        if leg["market"] == "Spread":
            return f"{leg['pick']} {float(leg['line']):+g}"
        return f"{leg['pick']} {float(leg['line']):g} total points"
    return f"{leg['player']} {leg['pick']} {float(leg['line']):g} {leg['stat']}"


def leg_outcome(leg, hs, as_, draws, home):
    """Boolean arrays (win, push) over the simulations for one parlay leg."""
    hs, as_ = np.asarray(hs), np.asarray(as_)
    line = float(leg.get("line") or 0.0)
    if leg["type"] == "game":
        m, pick = leg["market"], leg["pick"]
        if m in ("Moneyline", "Spread"):
            a, b = (hs, as_) if pick == home else (as_, hs)
            x = a - b + (line if m == "Spread" else 0.0)
            return x > 0, x == 0
        t = hs + as_
        return ((t > line) if pick == "Over" else (t < line)), t == line
    d = np.asarray(draws[(leg["team"], leg["player"], leg["stat"])])[:len(hs)]
    return ((d > line) if leg["pick"] == "Over" else (d < line)), d == line


def sgp_probs(legs, hs, as_, draws, home):
    """Joint chances for all legs from the SAME simulated games, so correlation is built in."""
    n = len(hs)
    outs = [leg_outcome(l, hs, as_, draws, home) for l in legs]
    allwin = np.ones(n, bool)
    anyloss = np.zeros(n, bool)
    for w, p in outs:
        allwin &= w
        anyloss |= ~(w | p)
    pw = float(allwin.mean())
    pl_ = float(anyloss.mean())
    pp = float(1.0 - pw - pl_)
    singles = [(float(w.mean()), float(p.mean())) for w, p in outs]
    indep = float(np.prod([s[0] for s in singles])) if singles else 0.0
    return {"pw": pw, "pp": pp, "pl": pl_, "singles": singles, "indep": indep, "n": n, "n_win": int(allwin.sum())}


def grade_sgp(r, hs, as_, pl):
    """Parlay result from the final score and real player stats: any lost leg loses, every leg won wins, otherwise push."""
    try:
        legs = json.loads(r["legs"])
    except Exception:
        return None
    home = r["home"]
    res = []
    for lg in legs:
        line = float(lg.get("line") or 0.0)
        if lg["type"] == "game":
            m, pick = lg["market"], lg["pick"]
            if m in ("Moneyline", "Spread"):
                a, o = (hs, as_) if pick == home else (as_, hs)
                x = a - o + (line if m == "Spread" else 0.0)
                res.append("won" if x > 0 else ("push" if x == 0 else "lost"))
            else:
                t = hs + as_
                res.append("push" if t == line else ("won" if ((t > line) == (pick == "Over")) else "lost"))
        else:
            cols = STAT_COL.get(lg["stat"], [])
            x = pl[(pl["season"] == r["season"]) & (pl["week"] == r["week"]) & (pl["pname"] == lg["player"]) & (pl["tm"] == lg["team"])]
            if x.empty or not cols:
                res.append("push")      # player never appeared: the leg is voided
            else:
                v = float(sum(x[c].sum() for c in cols if c in x.columns))
                res.append("push" if v == line else ("won" if ((v > line) == (lg["pick"] == "Over")) else "lost"))
    if "lost" in res:
        return "lost"
    if res and all(s == "won" for s in res):
        return "won"
    return "push"


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def card_color(t):
    """Team color, lightened when it would vanish on a dark background."""
    c = COLORS.get(t, "#4b5563")
    r, g, b = _hex_rgb(c)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    if lum < 0.16:
        k = 0.30
        r, g, b = (int(v + (255 - v) * k) for v in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def spread_text(spread, home, away):
    if spread is None or (isinstance(spread, float) and np.isnan(spread)):
        return ""
    if abs(spread) < 0.01:
        return "PK"
    return f"{home} -{abs(spread):g}" if spread > 0 else f"{away} -{abs(spread):g}"


def game_card_html(away, home, p_home, sim_a, sim_h, spread=None, total=None, mkt_p=None, qb_a=None, qb_h=None, date="", note=""):
    ca, ch = card_color(away), card_color(home)
    pa = (1 - p_home) * 100
    ph = p_home * 100
    fav = home if p_home >= 0.5 else away
    tag = ""
    if mkt_p is not None and not (isinstance(mkt_p, float) and np.isnan(mkt_p)):
        d = (p_home - mkt_p) * 100
        if abs(d) >= 4:
            side = home if d > 0 else away
            tag = f'<span class="g7tag">Model vs market: {side} +{abs(d):.0f} pts win chance</span>'
    sp = spread_text(spread, home, away)
    nt = f'<div class="g7sub">{note}</div>' if note else ""
    bits = []
    if sp:
        bits.append(f"Line {sp}")
    if total is not None and not (isinstance(total, float) and np.isnan(total)):
        bits.append(f"O/U {float(total):g}")
    bits.append(f"Sim {away} {sim_a:.0f} - {home} {sim_h:.0f}")
    qbs = ""
    if qb_a or qb_h:
        qbs = f'<div class="g7sub">QB: {qb_a or "?"} / {qb_h or "?"}</div>'
    return (f'<div class="g7card" style="background:linear-gradient(100deg,{ca}d9 0%,#0f172a 46%,#0f172a 54%,{ch}d9 100%)">'
            f'<div class="g7top"><span>{date}</span><span>{fav} favored</span></div>'
            f'<div class="g7row"><div class="g7team">{away}<small>{pa:.0f}%</small></div><div class="g7at">@</div><div class="g7team r">{home}<small>{ph:.0f}%</small></div></div>'
            f'<div class="g7bar"><i style="width:{pa:.1f}%;background:{ca}"></i><i style="width:{ph:.1f}%;background:{ch}"></i></div>'
            f'<div class="g7sub">{"  |  ".join(bits)}</div>{qbs}{nt}{tag}</div>')


G7_CSS = """
<style>
.g7grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:.7rem;margin:.4rem 0 1rem}
.g7card{border-radius:14px;padding:.75rem .9rem;color:#f8fafc;border:1px solid rgba(255,255,255,.08);box-shadow:0 2px 10px rgba(0,0,0,.35)}
.g7top{display:flex;justify-content:space-between;font-size:.72rem;opacity:.8;letter-spacing:.03em;text-transform:uppercase}
.g7row{display:flex;align-items:center;justify-content:space-between;margin:.35rem 0 .45rem}
.g7team{font-size:1.55rem;font-weight:800;line-height:1.05;text-shadow:0 1px 3px rgba(0,0,0,.6)}
.g7team small{display:block;font-size:.95rem;font-weight:700;opacity:.92}
.g7team.r{text-align:right}
.g7at{opacity:.55;font-weight:700}
.g7bar{display:flex;height:7px;border-radius:5px;overflow:hidden;background:#1e293b}
.g7bar i{display:block;height:100%}
.g7sub{font-size:.78rem;opacity:.85;margin-top:.45rem}
.g7tag{display:inline-block;margin-top:.5rem;padding:.12rem .55rem;border-radius:999px;background:rgba(250,204,21,.16);color:#fde68a;font-size:.72rem;font-weight:600;border:1px solid rgba(250,204,21,.35)}
.g7hero{padding:1rem 1.1rem}
.g7hero .g7team{font-size:2.3rem}
@media (max-width:640px){.g7grid{grid-template-columns:1fr}.g7hero .g7team{font-size:1.9rem}}
</style>
"""


# =====================================================================
# V8: REST, BYE WEEKS AND TRAVEL (learned from past games)
# =====================================================================
TEAM_XY = {"ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62), "BUF": (42.77, -78.79), "CAR": (35.23, -80.85), "CHI": (41.86, -87.62), "CIN": (39.10, -84.52),
           "CLE": (41.51, -81.70), "DAL": (32.75, -97.09), "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06), "HOU": (29.68, -95.41), "IND": (39.76, -86.16),
           "JAX": (30.32, -81.64), "KC": (39.05, -94.48), "LA": (33.95, -118.34), "LAC": (33.95, -118.34), "LV": (36.09, -115.18), "MIA": (25.96, -80.24), "MIN": (44.97, -93.26),
           "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07), "NYJ": (40.81, -74.07), "PHI": (39.90, -75.17), "PIT": (40.45, -80.02), "SEA": (47.60, -122.33),
           "SF": (37.40, -121.97), "TB": (27.98, -82.50), "TEN": (36.17, -86.77), "WAS": (38.91, -76.86)}


def trip_miles(away, home):
    a, h = TEAM_XY.get(away), TEAM_XY.get(home)
    if not a or not h:
        return 0.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], h[0], h[1]))
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3959.0 * 2 * math.asin(math.sqrt(d))


def trip_tz_east(away, home):
    """Time zones the visiting team crosses going east (0 if it travels west or stays level)."""
    a, h = TEAM_XY.get(away), TEAM_XY.get(home)
    if not a or not h:
        return 0.0
    return float(min(max((h[1] - a[1]) / 15.0, 0.0), 3.0))


def _wols(X, y, w):
    Xw, yw = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    e = yw - Xw @ b
    s2 = float(e @ e) / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xw.T @ Xw)))
    return b, se


@st.cache_data(show_spinner=False)
def situ_fit(_sched, _R, version, decay, alpha):
    """Measures from past games how much a bye week, a short week, travel distance and time zones moved the final margin beyond what the team ratings predict."""
    p = _sched.dropna(subset=["home_score", "away_score"]).copy()
    p = p[p["location"].astype(str).str.lower() != "neutral"]
    p = p[p["home_team"].isin(_R["off"]) & p["away_team"].isin(_R["off"]) & p["home_rest"].notna() & p["away_rest"].notna()]
    out = {"coef": 0.0, "n": int(len(p)), "rows": [], "applied": 0.0}
    if len(p) < 400:
        return out
    ha = _R["home_adv"]
    mu_h = _R["league_avg"] + p["home_team"].map(_R["off"]) + p["away_team"].map(_R["dfn"]) + ha
    mu_a = _R["league_avg"] + p["away_team"].map(_R["off"]) + p["home_team"].map(_R["dfn"]) - ha
    y = ((p["home_score"] - p["away_score"]) - (mu_h - mu_a)).values.astype(float)
    hr, ar = p["home_rest"].astype(float).values, p["away_rest"].astype(float).values
    w = decay ** (int(p["season"].max()) - p["season"].values)
    feats = {
        "Bye week (home off a bye minus away off a bye)": ((hr >= 10).astype(float) - (ar >= 10).astype(float)),
        "Short week (home on short rest minus away)": ((hr <= 5).astype(float) - (ar <= 5).astype(float)),
        "Visitor's trip, per 1,000 miles": np.array([trip_miles(a_, h_) for a_, h_ in zip(p["away_team"], p["home_team"])]) / 1000.0,
        "Visitor flies east, per time zone": np.array([trip_tz_east(a_, h_) for a_, h_ in zip(p["away_team"], p["home_team"])]),
    }
    rows = []
    for name, x in feats.items():
        X = np.column_stack([np.ones(len(x)), x])
        b, se = _wols(X, y, w)
        rows.append({"Factor": name, "Points for the home side": float(b[1]), "Std. error": float(se[1]), "Games with it": int((x != 0).sum())})
    # the amount applied comes from season-by-season estimates of the bye effect (completed seasons only), damped
    xb = feats["Bye week (home off a bye minus away off a bye)"]
    per = []
    seas = p["season"].values
    for s_ in sorted(set(seas)):
        m_ = seas == s_
        if m_.sum() >= 200 and (xb[m_] != 0).sum() >= 40 and s_ < int(p["season"].max()):
            per.append(float(xb[m_] @ y[m_]) / float(xb[m_] @ xb[m_]))
    k = float(np.mean(per)) if per else 0.0
    t = (k / (np.std(per, ddof=1) / math.sqrt(len(per)))) if len(per) >= 4 and np.std(per, ddof=1) > 0 else 0.0
    use = max(0.0, min(1.5, 0.8 * k)) if t > 1.5 else 0.0
    rows[0]["Typical season"] = k
    for r_ in rows[1:]:
        r_["Typical season"] = None
    k = rows[0]["Points for the home side"]
    out.update({"coef": float(k), "applied": float(use), "rows": rows})
    return out


def situ_adj(situ, row, home, away, neutral):
    """Points to move each team's expected score (home +, away -) plus plain-English notes."""
    adj, notes = {}, []
    if situ is None:
        return adj, notes
    hr = ar = None
    if row is not None:
        try:
            hr, ar = float(row["home_rest"]), float(row["away_rest"])
        except Exception:
            hr = ar = None
    m = 0.0
    if hr is not None and not (math.isnan(hr) or math.isnan(ar)):
        hb, ab = hr >= 10, ar >= 10
        if hb:
            notes.append(f"{home} off a bye")
        if ab:
            notes.append(f"{away} off a bye")
        for t, r_ in ((home, hr), (away, ar)):
            if r_ <= 5:
                notes.append(f"{t} on a short week ({int(r_)} days)")
        m = situ.get("applied", 0.0) * (float(hb) - float(ab)) if situ.get("apply", True) else 0.0
        if m:
            adj = {home: m / 2.0, away: -m / 2.0}
            notes[0:0] = []
    if not neutral:
        mi, tz = trip_miles(away, home), trip_tz_east(away, home)
        if mi >= 1500 or tz >= 2:
            notes.append(f"{away} travels {mi:,.0f} miles" + (f", {tz:.0f} time zones east" if tz >= 2 else ""))
    return adj, notes


# =====================================================================
# V8: PICKS OF THE WEEK, STORAGE, BEAT THE BOT PICK'EM
# =====================================================================
import base64 as _b64
from datetime import datetime as _dt, timezone as _tz


def weekly_picks(sdf):
    """Lock, Upset alert and Coin flip for a simulated week."""
    if sdf is None or len(sdf) == 0:
        return {}
    d = sdf.copy()
    d["_side"] = np.where(d["_p_home"] >= 0.5, d["_home"], d["_away"])
    d["_conf"] = np.maximum(d["_p_home"], 1 - d["_p_home"])
    out = {}
    lk = d.sort_values("_conf", ascending=False).iloc[0]
    out["lock"] = lk
    cf = d.assign(_gap=(d["_p_home"] - 0.5).abs()).sort_values("_gap").iloc[0]
    out["coin"] = cf
    ups = d[d["_mkt_p"].notna()].copy()
    if len(ups):
        mk_side = np.where(ups["_p_home"] >= 0.5, ups["_mkt_p"], 1 - ups["_mkt_p"])
        ups["_edge"] = np.where(ups["_p_home"] >= 0.5, ups["_p_home"], 1 - ups["_p_home"]) - mk_side
        ups = ups[mk_side < 0.5].sort_values("_edge", ascending=False)
        if len(ups):
            out["upset"] = ups.iloc[0]
    return out


def pick_card_html(kind, r):
    side = r["_side"] if "_side" in r.index else (r["_home"] if r["_p_home"] >= 0.5 else r["_away"])
    p_side = r["_p_home"] if side == r["_home"] else 1 - r["_p_home"]
    title = {"lock": "LOCK OF THE WEEK", "upset": "UPSET ALERT", "coin": "COIN FLIP"}[kind]
    icon = {"lock": "&#128274;", "upset": "&#9888;&#65039;", "coin": "&#129689;"}[kind]
    why = {"lock": f"The model's most confident pick: {p_side * 100:.0f}% to win.",
           "upset": f"The market has {side} as the underdog ({(1 - r['_mkt_p'] if side == r['_away'] else r['_mkt_p']) * 100:.0f}%), but the model gives {side} {p_side * 100:.0f}%.",
           "coin": f"As close as it gets: {r['_home']} {r['_p_home'] * 100:.0f}% / {r['_away']} {(1 - r['_p_home']) * 100:.0f}%."}[kind]
    col = card_color(side)
    return (f'<div class="g7card" style="background:linear-gradient(120deg,{col}d9 0%,#0f172a 70%)"><div class="g7top"><span>{icon} {title}</span><span>{r["Date"]}</span></div>'
            f'<div class="g7row"><div class="g7team">{side}<small>{r["_away"]} @ {r["_home"]}</small></div></div><div class="g7sub">{why}</div></div>')


# ---- storage: a data branch on GitHub when a token is set, otherwise a local file ----------------
PICKEM_FILE = "nfl_pickem.csv"
PICKEM_COLS = ["saved_at", "season", "week", "player", "home", "away", "pick"]
GH_BRANCH = "pickem-data"


def _secret(name, default=None):
    try:
        v = st.secrets.get(name, default)
        return v if v not in ("", None) else default
    except Exception:
        return default


def _gh_cfg():
    tok, repo = _secret("GH_TOKEN"), _secret("GH_REPO")
    return (str(tok), str(repo)) if tok and repo else None


def _gh(method, url, **kw):
    import requests
    tok, repo = _gh_cfg()
    return requests.request(method, "https://api.github.com" + url.replace("{repo}", repo), headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}, timeout=20, **kw)


def _gh_ensure_branch():
    r = _gh("GET", f"/repos/{{repo}}/git/ref/heads/{GH_BRANCH}")
    if r.status_code == 200:
        return True
    info = _gh("GET", "/repos/{repo}")
    if info.status_code != 200:
        return False
    base = info.json().get("default_branch", "main")
    sha = _gh("GET", f"/repos/{{repo}}/git/ref/heads/{base}").json().get("object", {}).get("sha")
    if not sha:
        return False
    return _gh("POST", "/repos/{repo}/git/refs", json={"ref": f"refs/heads/{GH_BRANCH}", "sha": sha}).status_code in (200, 201)


def _gh_read(name):
    r = _gh("GET", f"/repos/{{repo}}/contents/{name}?ref={GH_BRANCH}")
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    j = r.json()
    return _b64.b64decode(j["content"]).decode("utf-8"), j["sha"]


def _gh_write(name, text, sha, msg):
    body = {"message": msg, "content": _b64.b64encode(text.encode("utf-8")).decode("ascii"), "branch": GH_BRANCH}
    if sha:
        body["sha"] = sha
    return _gh("PUT", f"/repos/{{repo}}/contents/{name}", json=body)


def _local_path(name):
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base = os.getcwd()
    return os.path.join(base, name)


def _parse_csv(text, cols):
    import io
    try:
        df = pd.read_csv(io.StringIO(text)) if text and text.strip() else pd.DataFrame(columns=cols)
    except Exception:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


@st.cache_data(ttl=20, show_spinner=False)
def store_read(name, cols, mode):
    cols = list(cols)
    if mode == "github":
        try:
            text, _ = _gh_read(name)
            return _parse_csv(text, cols)
        except Exception:
            pass
    p = _local_path(name)
    if os.path.exists(p):
        try:
            return _parse_csv(open(p, encoding="utf-8").read(), cols)
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def store_mode():
    return "github" if _gh_cfg() else "local"


def store_append(name, cols, new, key_cols, msg="update"):
    """Adds rows (newer rows replace older ones with the same key). Returns (ok, where)."""
    cols = list(cols)
    if len(new) == 0:
        return True, store_mode()
    if _gh_cfg():
        try:
            if _gh_ensure_branch():
                for _ in range(4):
                    text, sha = _gh_read(name)
                    cur = _parse_csv(text, cols)
                    allp = pd.concat([cur, new[cols]], ignore_index=True).drop_duplicates(key_cols, keep="last")
                    r = _gh_write(name, allp.to_csv(index=False), sha, msg)
                    if r.status_code in (200, 201):
                        store_read.clear()
                        return True, "github"
                    if r.status_code not in (409, 422):
                        break
        except Exception:
            pass
    cur = store_read(name, tuple(cols), "local")
    allp = pd.concat([cur, new[cols]], ignore_index=True).drop_duplicates(key_cols, keep="last")
    try:
        allp.to_csv(_local_path(name), index=False)
        store_read.clear()
        return True, "local"
    except Exception:
        return False, "none"


def kickoff_utc(row):
    """Kickoff time from the schedule (gameday + Eastern gametime)."""
    try:
        from zoneinfo import ZoneInfo
        d = pd.Timestamp(row["gameday"]).date()
        hh, mm = str(row.get("gametime", "13:00") or "13:00").split(":")[:2]
        loc = _dt(d.year, d.month, d.day, int(hh), int(mm), tzinfo=ZoneInfo("America/New_York"))
        return loc.astimezone(_tz.utc)
    except Exception:
        return None


def is_locked(row, now=None):
    k = kickoff_utc(row)
    now = now or _dt.now(_tz.utc)
    return bool(k is not None and now >= k)


def pickem_score(picks, bot, sched, season):
    """Leaderboard rows and per-player detail from graded games.
    picks: friends' picks (PICKEM_COLS); bot: DataFrame with season, week, home, away, pick for the Bot."""
    s = sched[(sched["season"] == season) & sched["home_score"].notna()][["season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]].rename(columns={"home_team": "home", "away_team": "away"})
    s = s[s["home_score"] != s["away_score"]].copy()
    s["winner"] = np.where(s["home_score"] > s["away_score"], s["home"], s["away"])
    pk = picks[picks["season"] == season].merge(s, on=["season", "week", "home", "away"], how="inner")
    pk["ok"] = pk["pick"] == pk["winner"]
    bt = bot.rename(columns={"pick": "bot"})[["season", "week", "home", "away", "bot"]] if len(bot) else pd.DataFrame(columns=["season", "week", "home", "away", "bot"])
    pk = pk.merge(bt, on=["season", "week", "home", "away"], how="left")
    pk["bot_ok"] = pk["bot"] == pk["winner"]
    bs = s.merge(bt, on=["season", "week", "home", "away"], how="inner")
    bs["ok"] = bs["bot"] == bs["winner"]
    rows, badges = [], {}
    for name, g in pk.sort_values("gameday").groupby("player"):
        ok = g["ok"].tolist()
        cur = 0
        for v in reversed(ok):
            if v:
                cur += 1
            else:
                break
        best = run = 0
        for v in ok:
            run = run + 1 if v else 0
            best = max(best, run)
        gb = g[g["bot"].notna()]
        net = int(((gb["ok"]) & (~gb["bot_ok"])).sum() - ((~gb["ok"]) & (gb["bot_ok"])).sum())
        bd = []
        wk = g.groupby("week").agg(n=("ok", "size"), c=("ok", "sum"))
        if (wk["n"] >= 4).any() and ((wk["c"] == wk["n"]) & (wk["n"] >= 4)).any():
            bd.append("Perfect week")
        wb = gb.groupby("week").agg(n=("ok", "size"), c=("ok", "sum"), b=("bot_ok", "sum"))
        if ((wb["n"] >= 4) & (wb["c"] > wb["b"])).any():
            bd.append("Bot slayer")
        if best >= 5:
            bd.append("Hot streak")
        if int(((gb["ok"]) & (gb["bot"] != gb["pick"]) & (~gb["bot_ok"])).sum()) >= 3:
            bd.append("Rebel (right against the Bot 3+ times)")
        badges[name] = bd
        rows.append({"Player": name, "Right": int(sum(ok)), "Picks": len(ok), "Win %": 100.0 * sum(ok) / len(ok), "Streak": cur, "Best streak": best, "Vs the Bot": net, "Badges": ", ".join(bd)})
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values(["Win %", "Picks"], ascending=False).reset_index(drop=True)
        lb.insert(0, "Rank", range(1, len(lb) + 1))
    brow = {"Right": int(bs["ok"].sum()), "Picks": int(len(bs))}
    return lb, brow, pk


CONFETTI_HTML = """
<canvas id="c" style="width:100%;height:150px;display:block"></canvas>
<script>
const c=document.getElementById('c'),x=c.getContext('2d');const W=c.width=c.clientWidth*2,H=c.height=300;
const cols=['#22c55e','#facc15','#38bdf8','#f472b6','#fb923c','#a78bfa'];
const P=[...Array(140)].map(()=>({x:Math.random()*W,y:-Math.random()*H,r:4+Math.random()*7,vx:(Math.random()-.5)*3,vy:2+Math.random()*4,c:cols[(Math.random()*cols.length)|0],a:Math.random()*6}));
let t=0;(function f(){x.clearRect(0,0,W,H);P.forEach(p=>{p.x+=p.vx;p.y+=p.vy;p.a+=.15;x.save();x.translate(p.x,p.y);x.rotate(p.a);x.fillStyle=p.c;x.fillRect(-p.r/2,-p.r/4,p.r,p.r/2);x.restore();});if(++t<170)requestAnimationFrame(f);else x.clearRect(0,0,W,H);})();
</script>
"""


# =====================================================================
# V8: COMMENTARY, PLAYERS OF THE GAME, RIVALRIES, ODDS RACE
# =====================================================================
def _ln(n):
    return str(n).split(" ")[-1] if n else "the offense"


def commentary_lines(plays, home, away, only_key=True):
    out = []
    for i, p in enumerate(plays):
        ty = p.get("type")
        if only_key and not (p.get("key") or p.get("td") or ty in ("fg", "int", "fum", "sack")):
            continue
        per, clk = p.get("per", ""), int(p.get("clock", 0) or 0)
        head = f"**{per} {clk // 60}:{clk % 60:02d}** &middot; {p.get('team', '')} &middot; {p.get('at', '')}"
        who, tgt, yds = _ln(p.get("who")), _ln(p.get("tgt")), int(p.get("yds", 0) or 0)
        v = i % 3
        if ty == "pass":
            if p.get("td"):
                line = [f"{who} finds {tgt} in the end zone for {yds}! Touchdown, {p['team']}!", f"{tgt} hauls it in and scores! {yds} yards from {who}.", f"{who} to {tgt}, {yds} yards, and that's six!"][v]
            elif p.get("first"):
                line = [f"{who} to {tgt} for {yds}, moving the chains.", f"{tgt} makes the catch for {yds} and it's a first down.", f"Good protection, {who} hits {tgt} for {yds} and a fresh set of downs."][v]
            else:
                line = [f"{who} completes it to {tgt} for {yds}.", f"{tgt} makes the grab, {yds} yards.", f"Short of the sticks: {who} to {tgt}, {yds}."][v]
        elif ty == "run":
            if p.get("td"):
                line = [f"{who} punches it in! Touchdown, {p['team']}!", f"{who} finds the end zone from {abs(yds)} out!", f"{who} bursts through for the score!"][v]
            elif yds >= 15:
                line = [f"{who} breaks free for {yds} yards!", f"Huge hole and {who} is off for {yds}!", f"{who} rips off a {yds}-yard run."][v]
            elif p.get("first"):
                line = f"{who} gets the first down with a {yds}-yard carry."
            else:
                line = [f"{who} takes it for {yds}.", f"{who} up the middle, {yds} yards.", f"{who} cuts outside for {yds}."][v]
        elif ty == "inc":
            line = [f"{who}'s pass for {tgt} falls incomplete.", f"{who} looks for {tgt} but can't connect.", f"Incomplete. {who} and {tgt} just miss."][v]
        elif ty == "sack":
            line = [f"{who} is sacked for a loss of {abs(yds)}!", f"The pocket collapses and {who} goes down.", f"Sack! {who} is buried for -{abs(yds)}."][v]
        elif ty == "int":
            line = [f"INTERCEPTED! {who}'s pass is picked off.", f"{who} throws it right to the defense, a turnover!", f"Trouble for {p['team']}: {who} is intercepted."][v]
        elif ty == "fum":
            line = f"FUMBLE! The ball is loose and {p.get('text', '').split('recovered by')[-1].strip() or 'the defense'} comes up with it."
        elif ty in ("fg", "xp"):
            line = p.get("text", "Kick attempt.")
        elif ty == "punt":
            line = f"{p['team']} punts, {yds} yards."
        else:
            line = p.get("text", "")
        if p.get("hs") is not None:
            line += f" ({away} {p['as']}, {home} {p['hs']})"
        out.append(head + "  \n" + line)
    return out


def three_stars(box, home, away):
    sc = {}
    for t in (home, away):
        for n, a in box[t]["pass"].items():
            d = sc.setdefault((n, t), {"s": 0.0, "bits": []})
            d["s"] += a[2] / 25.0 + 4 * a[3] - 2 * a[4]
            if a[1]:
                d["bits"].append(f"{a[0]}/{a[1]}, {a[2]} pass yds" + (f", {a[3]} TD" if a[3] else "") + (f", {a[4]} INT" if a[4] else ""))
        for n, r in box[t]["rush"].items():
            d = sc.setdefault((n, t), {"s": 0.0, "bits": []})
            d["s"] += r[1] / 10.0 + 6 * r[2]
            d["bits"].append(f"{r[0]} car, {r[1]} rush yds" + (f", {r[2]} TD" if r[2] else ""))
        for n, e in box[t]["rec"].items():
            if not n:
                continue
            d = sc.setdefault((n, t), {"s": 0.0, "bits": []})
            d["s"] += e[2] / 10.0 + 6 * e[3] + 0.5 * e[1]
            d["bits"].append(f"{e[1]} rec, {e[2]} yds" + (f", {e[3]} TD" if e[3] else ""))
    top = sorted(sc.items(), key=lambda kv: -kv[1]["s"])[:3]
    return [{"name": n, "team": t, "line": " | ".join(v["bits"]), "score": v["s"]} for (n, t), v in top]


def stars_html(stars):
    meds = ["#facc15", "#cbd5e1", "#f59e0b"]
    cards = "".join(f'<div class="g7card" style="background:linear-gradient(120deg,{card_color(s["team"])}d9 0%,#0f172a 75%)"><div class="g7top"><span style="color:{meds[i]}">&#9733; STAR {i + 1}</span><span>{s["team"]}</span></div>'
                    f'<div class="g7row"><div class="g7team" style="font-size:1.25rem">{s["name"]}</div></div><div class="g7sub">{s["line"]}</div></div>' for i, s in enumerate(stars))
    return '<div class="g7grid">' + cards + '</div>'


# ---- rivalries -------------------------------------------------------------------------------
def series_games(sched, a, b):
    s = sched.dropna(subset=["home_score", "away_score"])
    g = s[((s["home_team"] == a) & (s["away_team"] == b)) | ((s["home_team"] == b) & (s["away_team"] == a))].sort_values(["season", "week"]).copy()
    g["a_pts"] = np.where(g["home_team"] == a, g["home_score"], g["away_score"])
    g["b_pts"] = np.where(g["home_team"] == a, g["away_score"], g["home_score"])
    g["a_margin"] = g["a_pts"] - g["b_pts"]
    return g


def series_summary(g):
    if g.empty:
        return None
    aw, bw = int((g["a_margin"] > 0).sum()), int((g["a_margin"] < 0).sum())
    return {"games": len(g), "a_w": aw, "b_w": bw, "t": len(g) - aw - bw, "avg_margin": float(g["a_margin"].mean()), "avg_total": float((g["a_pts"] + g["b_pts"]).mean()),
            "close": int((g["a_margin"].abs() <= 7).sum())}


def division_rivalry_table(sched):
    rows = []
    for conf_div in sorted(set(TEAM_DIV.values())):
        ts = sorted(t for t, d in TEAM_DIV.items() if d == conf_div)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                g = series_games(sched, ts[i], ts[j])
                sm = series_summary(g)
                if sm:
                    rows.append({"Division": conf_div, "Matchup": f"{ts[i]} vs {ts[j]}", "Games": sm["games"], "Record": f"{ts[i]} {sm['a_w']}-{sm['b_w']}" + (f"-{sm['t']}" if sm["t"] else ""),
                                 "Avg margin (first team)": sm["avg_margin"], "Avg total": sm["avg_total"], "Decided by 7 or less %": 100.0 * sm["close"] / sm["games"]})
    return pd.DataFrame(rows)


# ---- odds race ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Rebuilding the odds after each week...")
def retro_odds(_sched, season, decay, alpha, sims, last_week, version):
    """Title odds as they stood after each completed week, using only the games played up to then."""
    out = []
    for w in range(1, int(last_week) + 1):
        m = _sched.copy()
        mask = (m["season"] == season) & (m["week"] > w)
        m.loc[mask, ["home_score", "away_score"]] = np.nan
        ver = int(900000 + season * 100 + w)
        Rw = fit_ratings(m, decay, alpha, ver)
        ss = sim_season(Rw, m, season, sims, 11, ver, decay, alpha)
        st_ = ss["stat"]
        for i, t in enumerate(ss["teams"]):
            out.append({"week": w, "team": t, "Make the playoffs": float(st_["playoffs"][i]) * 100, "Win the division": float(st_["division"][i]) * 100,
                        "Get the #1 seed": float(st_["seed1"][i]) * 100, "Reach the Super Bowl": float(st_["super_bowl"][i]) * 100, "Win the Super Bowl": float(st_["champion"][i]) * 100,
                        "Projected wins": float(ss["exp_wins"][i])})
    return pd.DataFrame(out)


RACE_HTML = r"""
<style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,'Segoe UI',Arial,sans-serif;color:#e8eaed;background:transparent}
#w{background:#0f172a;border-radius:12px;padding:12px 12px 8px}
#t{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:6px}
#ttl{font-weight:700;font-size:15px}#wk{font-size:26px;font-weight:800;color:#facc15;font-variant-numeric:tabular-nums;white-space:nowrap}@media(max-width:520px){#ttl{font-size:12px}#wk{font-size:17px}}
#bars{position:relative}
.b{position:absolute;left:0;right:0;height:26px;transition:top .55s ease}
.b i{position:absolute;left:44px;top:2px;height:22px;border-radius:5px;transition:width .55s ease;min-width:2px}
.b em{position:absolute;left:0;top:4px;width:40px;font-style:normal;font-weight:700;font-size:13px}
.b s{position:absolute;top:4px;text-decoration:none;font-size:12px;font-weight:700;transition:left .55s ease;white-space:nowrap}
#c{display:flex;gap:8px;align-items:center;margin-top:8px}
button{background:#1e293b;color:#fff;border:1px solid #334155;border-radius:8px;padding:6px 12px;font-size:14px;cursor:pointer}
input[type=range]{flex:1}
</style>
<div id="w"><div id="t"><div id="ttl"></div><div id="wk"></div></div><div id="bars"></div>
<div id="c"><button id="pl">Play</button><input id="sl" type="range" min="0" value="0"><span id="lb" style="font-size:12px;color:#94a3b8;min-width:70px;text-align:right"></span></div></div>
<script>
const D=__DATA__;const N=D.top, RH=30;
const bars=document.getElementById('bars');bars.style.height=(N*RH+4)+'px';
document.getElementById('ttl').textContent=D.title;
const els={};D.teams.forEach(t=>{const d=document.createElement('div');d.className='b';d.innerHTML='<em>'+t+'</em><i style="background:'+D.col[t]+'"></i><s></s>';bars.appendChild(d);els[t]=d;});
const sl=document.getElementById('sl');sl.max=D.frames.length-1;sl.value=sl.max;
let timer=null;
function draw(k){const f=D.frames[k];document.getElementById('wk').textContent='After week '+f.w;document.getElementById('lb').textContent='Week '+f.w;
  const order=D.teams.slice().sort((a,b)=>f.v[b]-f.v[a]);const mx=Math.max(1,D.max);const W=bars.clientWidth-44-70;
  D.teams.forEach(t=>{const r=order.indexOf(t),e=els[t];e.style.top=(r*RH)+'px';e.style.opacity=r<N?1:0;e.style.pointerEvents='none';const w=Math.max(2,(f.v[t]/mx)*W);e.children[1].style.width=w+'px';
    const s=e.children[2];s.textContent=f.v[t].toFixed(1)+'%';s.style.left=(44+w+6)+'px';s.style.color='#e8eaed';});}
sl.oninput=()=>{stop();draw(+sl.value)};
function stop(){if(timer){clearInterval(timer);timer=null;document.getElementById('pl').textContent='Play'}}
document.getElementById('pl').onclick=()=>{if(timer){stop();return}if(+sl.value>=+sl.max)sl.value=0;document.getElementById('pl').textContent='Pause';draw(+sl.value);
  timer=setInterval(()=>{if(+sl.value>=+sl.max){stop();return}sl.value=+sl.value+1;draw(+sl.value)},900)};
draw(+sl.value);
</script>
"""


def race_html(df, metric, conf, title):
    sub = df
    if conf in ("AFC", "NFC"):
        sub = df[df["team"].isin(CONFS[conf])]
    weeks = sorted(sub["week"].unique())
    teams = sorted(sub["team"].unique())
    last = sub[sub["week"] == weeks[-1]].sort_values(metric, ascending=False)
    keep = list(last["team"].head(14))
    # include anyone who led at some point
    for w in weeks:
        top = sub[sub["week"] == w].sort_values(metric, ascending=False).head(6)["team"]
        keep += [t for t in top if t not in keep]
    keep = keep[:18]
    frames = [{"w": int(w), "v": {t: round(float(sub[(sub["week"] == w) & (sub["team"] == t)][metric].iloc[0]), 2) for t in keep}} for w in weeks]
    mx = max(max(f["v"].values()) for f in frames)
    data = {"title": title, "top": min(10, len(keep)), "teams": keep, "col": {t: card_color(t) for t in keep}, "frames": frames, "max": mx}
    return RACE_HTML.replace("__DATA__", json.dumps(data))


def jersey_numbers(ros, teams):
    """{player name: jersey number} for the two teams, from the latest weekly roster."""
    try:
        d = ros[ros["team"].isin(teams)].dropna(subset=["jersey_number"])
        d = d[d["week"] == d.groupby("team")["week"].transform("max")]
        return {str(r["full_name"]): int(float(r["jersey_number"])) for r in d.to_dict("records") if str(r["jersey_number"]).replace(".0", "").isdigit()}
    except Exception:
        return {}


HIST_FILE = "nfl_odds_history.csv"
HIST_COLS = ["date", "week", "team", "proj_wins", "playoffs", "division", "seed1", "super_bowl"]


def load_odds_history():
    p = _local_path(HIST_FILE)
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p)
        return d if len(d) else None
    except Exception:
        return None


# =====================================================================
# APP
# =====================================================================
sched, pl, seasons = load_data()
version = int(sched["home_score"].notna().sum())
latest = seasons[-1]
pool0 = build_pool(pl, len(pl))
qbs0, qb_prior = build_qbs(pl, len(pl))
pool, qbs = pool0, qbs0
inj_df = load_injuries(latest)
ros_df = load_rosters(latest)
rst = roster_state(ros_df)
ros_week = int(ros_df["week"].max()) if not ros_df.empty else None
S_all = load_style(latest, version)

CSS = '\n<style>\n.block-container{padding-top:3.6rem;padding-bottom:4rem;max-width:1240px}\nh1{font-size:1.7rem !important;padding:0 0 .2rem 0 !important}\ndiv[data-testid="stMetricValue"]{font-size:1.6rem}\n.st-key-matchbar [data-testid="stHorizontalBlock"]{flex-wrap:nowrap !important;gap:.6rem}\n.st-key-matchbar [data-testid="stColumn"]{min-width:0 !important;width:auto !important;flex:1 1 0 !important}\n.st-key-matchbar [data-testid="stButton"] button{width:100%;font-weight:700}\n.st-key-matchbar div[data-testid="stButton"]{width:100%}\n.stTabs [data-baseweb="tab-list"]{gap:.15rem;overflow-x:auto}\n.stTabs [data-baseweb="tab"]{padding:.45rem .8rem;white-space:nowrap}\n[class*="st-key-kpis"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important}\n@media (max-width:640px){\n  [class*="st-key-kpis"] [data-testid="stColumn"]{min-width:calc(50% - .5rem) !important;flex:1 1 calc(50% - .5rem) !important;width:auto !important}\n  .block-container{padding:3.3rem .55rem 4rem}\n  .st-key-topcap{display:none}\n  h1{font-size:1.35rem !important}\n  div[data-testid="stMetricValue"]{font-size:1.15rem}\n  div[data-testid="stMetricLabel"]{font-size:.72rem}\n  .stTabs [data-baseweb="tab"]{padding:.4rem .55rem;font-size:.85rem}\n  div[data-testid="stHorizontalBlock"]{gap:.4rem}\n  .stButton>button,.stDownloadButton>button{width:100%;min-height:2.6rem}\n  div[data-testid="stDataFrame"]{font-size:.8rem}\n}\n</style>\n'
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(G7_CSS, unsafe_allow_html=True)
st.title("🏈 NFL Simulator")
with st.container(key="topcap"):
    st.caption(f"{seasons[0]}-{seasons[-1]} data, {version:,} completed games. Simulate any game, watch it play out on the field, and track your predictions and bets.")
teams = fit_ratings(sched, 0.75, 20.0, version)["teams"]
with st.container(key="matchbar"):
    mb1, mb2 = st.columns(2)
    away = mb1.selectbox("Away team", teams, index=teams.index("SEA") if "SEA" in teams else 1)
    home = mb2.selectbox("Home team", teams, index=teams.index("CHI") if "CHI" in teams else 0)
    mb3, mb4 = st.columns([1, 1])
    neutral = mb3.checkbox("Neutral site")
    run = mb4.button("▶ Simulate game", type="primary", key="run_btn")
t_game, t_field, t_players, t_bets, t_week, t_pickem, t_season, t_teams, t_setup = st.tabs(["Game", "Field", "Players", "Bets", "Week", "Pick'em", "Season", "Teams", "Setup"])
with t_season:
    tab_season, tab_race, tab_riv = st.tabs(["Odds", "Race", "Rivalries"])
with t_players:
    tab_players, tab_box, tab_pbp = st.tabs(["Projections", "Box score", "Plays"])
with t_bets:
    tab_props, tab_slip, tab_sgp, tab_mybets, tab_bet, tab_track = st.tabs(["Props board", "Place a bet", "Parlay", "My bets", "Value finder", "Model picks"])
with t_teams:
    tab_bt, tab_roster, tab_inj, tab_style, tab_ratings = st.tabs(["Backtest", "Rosters", "Injuries", "Team style", "Ratings"])
tab_game, tab_anim, tab_slate, tab_drives = t_game, t_field, t_week, tab_pbp

with t_setup:
    st.subheader("Model settings")
    decay = st.slider("How much recent seasons matter", 0.4, 1.0, 0.75, 0.05, help="Lower = older seasons count less")
    alpha = st.slider("Rating smoothing", 1.0, 60.0, 20.0, 1.0, help="Higher = ratings pulled harder toward average")
    R = fit_ratings(sched, decay, alpha, version)
    teams = R["teams"]

    st.subheader("Rosters")
    ov_all = st.session_state.setdefault("ros_ov", {})
    ov_eff = {}
    for _pid, _v in list(ov_all.items()):
        if rst.get(_pid, (None, None))[0] == _v["team"]:
            del ov_all[_pid]          # the data caught up, so the manual move is no longer needed
        else:
            ov_eff[_pid] = _v
    rst_eff = dict(rst)
    for _pid, _v in ov_eff.items():
        rst_eff[_pid] = (_v["team"], "ACT")
    use_ros = st.checkbox("Use current team rosters", value=not ros_df.empty, disabled=ros_df.empty, help="Moves traded or newly signed players to their new team and keeps players on IR, the practice squad, or off a roster out of the default lineups.")
    with st.expander("Manual roster moves (trades the data doesn't show yet)"):
        if ros_df.empty:
            st.caption("Needs the roster data, which could not be loaded right now.")
        else:
            st.caption("Pick a player and the team he is now on. The simulator treats him as active there (and as the starter, for a QB). A move disappears by itself once the roster data shows him on that team.")
            names = {}
            for _t, _by in pool0.items():
                for _lst in _by.values():
                    for _c in _lst:
                        names[f"{_c['name']} ({_t})"] = _c["pid"]
            for _t, _lst in qbs0.items():
                for _q in _lst:
                    names[f"{_q['name']} ({_t})"] = _q["pid"]
            who = st.selectbox("Player (type to search)", [""] + sorted(names), key="ov_who")
            to_team = st.selectbox("Now plays for", teams, key="ov_to")
            if st.button("Add move", key="ov_add") and who:
                ov_all[names[who]] = {"team": to_team, "name": who.rsplit(" (", 1)[0]}
                st.rerun()
            for _pid, _v in list(ov_all.items()):
                c1, c2 = st.columns([3, 1])
                c1.write(f"{_v['name']} to {_v['team']}")
                if c2.button("Undo", key=f"ov_rm_{_pid}"):
                    del ov_all[_pid]
                    st.rerun()
    if ros_df.empty:
        st.caption("Roster data could not be loaded, so lineups use each player's last team.")
    elif use_ros:
        pool, qbs = rehome_pool(pool0, rst_eff, set(ov_eff)), rehome_qbs(qbs0, rst_eff, set(ov_eff))
        st.caption(f"Rosters as of week {ros_week} (rechecked hourly)" + (f", plus {len(ov_eff)} manual move(s)." if ov_eff else "."))

    depth = load_depth(latest)
    use_depth = st.checkbox("Use depth chart starters", value=bool(depth), disabled=not depth, help="Puts each team's official depth chart starters first: the QB, the top RB, the top three WRs and the TE. Players who are Out or Doubtful are still skipped for the next man up.")
    if not depth:
        st.caption("The depth chart could not be loaded right now, so lineups use recent usage.")
    elif use_depth:
        pool, qbs = apply_depth(pool, qbs, depth)
        st.caption("Default lineups follow the latest nflverse depth chart (rechecked hourly). You can still change any lineup below.")

    use_situ = st.checkbox("Rest and bye weeks", value=True, help="Moves the margin a little when one team is coming off a bye and the other is not. The size is measured from past games (see below).")
    SITU = situ_fit(sched, R, version, decay, alpha)
    with st.expander("What past games say about rest and travel"):
        if SITU["rows"]:
            st.dataframe(pd.DataFrame(SITU["rows"]), hide_index=True, column_config={"Points for the home side": st.column_config.NumberColumn(format="%+.2f"), "Std. error": st.column_config.NumberColumn(format="%.2f"), "Typical season": st.column_config.NumberColumn(format="%+.2f")})
            st.caption(f"Each row is how many points that factor moved the home team's margin beyond what the ratings predicted, across {SITU['n']:,} games. An effect smaller than about twice its standard error is just noise. "
                       f"Only the bye-week effect is applied, using the average of each completed season's own estimate (the Typical season column) and damped by 20%: currently {SITU['applied']:.1f} points when exactly one team is off a bye. Short weeks, miles and time zones are shown on the game cards but do not change the odds, because the data doesn't show a reliable effect.")
        else:
            st.caption("Not enough games with rest data yet.")

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
    wk_key = f"{inj_week}_{int(use_inj)}_{int(bool(use_ros))}_{abs(hash(tuple(sorted((k, v['team']) for k, v in ov_eff.items())))) % 100000}"

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
            base = q["name"] + (" (usual starter)" if (i == 0 and q["team"] == team and not q.get("blocked") and not q.get("moved_from")) else ("" if q["team"] == team else f" ({q['team']})"))
            labels.append(base + roster_tag(q) + (tag_for(team, q["pid"]) if q["team"] == team else ""))
        di = 0
        for i, q in enumerate(mine):
            if not unavailable(inj[team].get(q["pid"], {}).get("status")) and not q.get("blocked"):
                di = i
                break
        choice = st.selectbox(f"{team} quarterback", labels, index=di, key=f"qb_{side}_{team}_{wk_key}")
        base_list = qbs0.get(team, []) if (use_ros or use_depth) else mine
        return plist[labels.index(choice)], (base_list[0] if base_list else (mine[0] if mine else plist[0]))

    qb_h, s0_h = pick_qb(home, "home")
    qb_a, s0_a = pick_qb(away, "away")
    qb_adj = {home: 0.0, away: 0.0}
    for t, q, s0 in ((home, qb_h, s0_h), (away, qb_a, s0_a)):
        if q is not None and s0 is not None:
            qb_adj[t] = (q["rate"] - s0["rate"]) * DROPBACKS * QB_WEIGHT
            if q["pid"] != s0["pid"]:
                st.caption(f"{t}: {q['name']} instead of {s0['name']} = {qb_adj[t]:+.1f} pts/game")

    st.subheader("Skill players")
    st.caption("Defaults are each team's top players by recent usage, skipping anyone Out or Doubtful, and (with current rosters on) anyone on IR, the practice squad, or not on a roster. Add or remove anyone you like.")

    def pick_skill(team, side):
        by_pos = pool.get(team, {})
        chosen, changed = {}, []
        with st.expander(f"{team} running backs, receivers, tight ends", expanded=True):
            for pos, title in (("RB", "Running backs"), ("WR", "Wide receivers"), ("TE", "Tight ends")):
                cands = by_pos.get(pos, [])
                labels = {}
                for c in cands:
                    lab = c["name"] + (f" (missed last {c['missed']} games)" if c["missed"] > 0 else "") + roster_tag(c) + tag_for(team, c["pid"])
                    labels[lab] = c
                avail = [lab for lab, c in labels.items() if not unavailable(inj[team].get(c["pid"], {}).get("status")) and not c.get("blocked")]
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
    _sa, _sn = situ_adj(dict(SITU, apply=use_situ), find_game(sched, latest, home, away), home, away, neutral)
    for _t_, _v_ in _sa.items():
        adj[_t_] = adj.get(_t_, 0.0) + _v_

    st.subheader("Model options")
    use_style = st.checkbox("Team style (from play-by-play)", value=S_all is not None, disabled=S_all is None, help="Each team's pace, turnover rate, field-goal vs touchdown mix, and passing share, learned from the last two seasons of drives.")
    if S_all is None:
        st.caption("Play-by-play style data could not be loaded, so league-average style is used.")
    use_flow = st.checkbox("Game flow", value=True, help="Trailing teams hurry up and throw more, leaders run the clock, and the leader kneels at the end.")

    game_sims = st.select_slider("Game simulations", [10000, 50000, 100000, 250000], value=100000)
    drive_sims = st.select_slider("Games for 'typical game'", [500, 1000, 2000], value=1000)


if run:
    if home == away:
        st.error("Pick two different teams.")
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
            if use_ros and not ros_df.empty and ov_eff:
                notes.append("Manual roster moves in effect: " + ", ".join(f"{v['name']} to {v['team']}" for v in ov_eff.values()))
            if _sn:
                notes.append("Rest and travel: " + "; ".join(_sn) + (f" (moves the margin {_sa.get(home, 0.0) * 2:+.1f} for {home})" if _sa else " (shown for context, no change to the odds)"))
            notes.append("Team style: " + ("on" if S_use else "off (league-average style)") + " | Game flow: " + ("on" if use_flow else "off") + " | Rosters: " + (f"current through week {ros_week}" if (use_ros and not ros_df.empty) else "each player's last team"))
            prep = prepare(R, S_use, prof_used, home, away, neutral, adj, use_flow)
            typ, avg_df = typical_game(prep, drive_sims, use_flow)
            game = simulate_game(R, S_use, home, away, game_sims, neutral, adj)
            st.session_state["res"] = {"adj": dict(adj), "qbcur": {home: qb_h, away: qb_a}, "rest_note": "; ".join(_sn), "home": home, "away": away, "neutral": neutral, "notes": notes, "prep": prep, "flow": use_flow, "S": S_use, "inj_week": inj_week, "use_inj": use_inj,
                                       "game": game, "line": market_from_row(find_game(sched, latest, home, away)), "players": sim_players(R, prof_used, home, away, game), "typical": typ, "avg": avg_df, "drive_sims": drive_sims, "one": run_game(prep, use_flow), "anim": None}

res = st.session_state.get("res")
pass

with tab_slate:
    open_weeks = sorted(int(w) for w in sched[(sched["season"] == latest) & sched["home_score"].isna()]["week"].unique())
    if not open_weeks:
        st.info("There are no unplayed games left on this season's schedule.")
    else:
        st.caption("Simulates every game in a week at once using the same settings as the Setup tab (ratings, team style, current rosters, depth chart, and each team's default quarterback from the injury report). Skill players don't change the score, so the slate only needs the QB. Lines come from the schedule data when they are posted.")
        s1, s2 = st.columns([1, 1])
        sl_week = s1.selectbox("Week", open_weeks, index=0, key="slate_week")
        sl_sims = s2.select_slider("Simulations per game", [2000, 5000, 10000, 20000], value=5000, key="slate_sims")
        if st.button("Simulate this week", type="primary", key="slate_go"):
            with st.spinner("Simulating every game..."):
                compute_slate(sl_week, sl_sims)
        sl = st.session_state.get("slate")
        if sl is None:
            st.info("Pick a week and press Simulate this week.")
        else:
            sdf = sl["df"]
            st.subheader(f"Week {sl['week']} ({sl['sims']:,} simulations per game)")
            if sdf.empty:
                st.write("No games found.")
            else:
                if (sdf["Injury report"] == "not posted yet").all():
                    st.caption("The injury report for this week isn't posted yet, so default lineups only reflect the rosters. Re-run after the report comes out (usually Wednesday or Thursday).")
                _wp = weekly_picks(sdf)
                if _wp:
                    st.markdown('<div class="g7grid">' + "".join(pick_card_html(k, _wp[k]) for k in ("lock", "upset", "coin") if k in _wp) + '</div>', unsafe_allow_html=True)
                    if "upset" not in _wp:
                        st.caption("No upset alert: the model sides with the market's favorite in every game that has a line.")
                    _gm = grade_preds(load_preds(), sched)
                    _gd = _gm[_gm["done"] & (_gm["act_m"] != 0)]
                    if len(_gd):
                        _w_ = int(((_gd["p_home"] > 0.5) == (_gd["act_m"] > 0)).sum())
                        st.caption(f"Model record on picks saved before kickoff: {_w_}-{len(_gd) - _w_} ({100.0 * _w_ / len(_gd):.0f}%).")
                    else:
                        st.caption("The model's record will appear here once saved picks have been graded. The daily robot saves them automatically.")
                with st.expander("Check a price from your sportsbook"):
                    _pg = st.selectbox("Game", [r_["Game"] for r_ in sdf.to_dict("records")], key="pc_game")
                    _pr = [r_ for r_ in sdf.to_dict("records") if r_["Game"] == _pg][0]
                    _ps = st.radio("Side", [_pr["_home"], _pr["_away"]], horizontal=True, key="pc_side")
                    _po = st.number_input("American odds you were offered", value=-110.0, step=5.0, key="pc_odds")
                    _pw = _pr["_p_home"] if _ps == _pr["_home"] else 1 - _pr["_p_home"]
                    _pm = bet_math(_pw, 0.0, 1 - _pw, _po, 10.0, float(st.session_state.get("bet_bank0", 1000.0)), float(st.session_state.get("bet_frac", 0.25)))
                    _pk = st.container(key="kpis_pc")
                    _pc = _pk.columns(4)
                    _pc[0].metric("Model win chance", f"{_pw * 100:.1f}%")
                    _pc[1].metric("Break-even chance", f"{_pm['be'] * 100:.1f}%", help="The win chance this price needs just to break even")
                    _pc[2].metric("Profit per $100 (model)", f"{_pm['ev'] * 100:+.1f}", help="Expected profit on a $100 bet if the model's chance were exactly right")
                    _pc[3].metric("Kelly stake", f"${_pm['stake_k']:,.0f}" if _pm["kelly"] > 0 else "$0", help="Fractional-Kelly size for your bankroll; zero means the model sees no edge")
                    st.caption("Win chance is the simulator's moneyline chance for that side. A positive profit number means the model thinks the price is too generous; the model has trailed the market in the Backtest, so treat it as a prompt to look closer.")
                _cards = "".join(game_card_html(r["_away"], r["_home"], r["_p_home"], (r["_sim_t"] - r["_sim_m"]) / 2, (r["_sim_t"] + r["_sim_m"]) / 2, r.get("_spread"), r.get("_total"), r.get("_mkt_p"),
                                                 r.get("_qb_a"), r.get("_qb_h"), str(r.get("Date", "")), r.get("Rest & travel", "")) for r in sdf.to_dict("records"))
                st.markdown('<div class="g7grid">' + _cards + '</div>', unsafe_allow_html=True)
                cfg = {
                    "Home win %": st.column_config.NumberColumn(format="%.1f"),
                    "Sim margin (home)": st.column_config.NumberColumn(format="%.1f", help="Average home-team margin in the simulations"),
                    "Sim total": st.column_config.NumberColumn(format="%.1f"),
                    "Line (home margin)": st.column_config.NumberColumn(format="%.1f", help="Points the home team is favored by (negative = away favored)"),
                    "Sim minus line": st.column_config.NumberColumn(format="%+.1f", help="Positive = the simulator likes the home team more than the market does"),
                    "Home covers %": st.column_config.NumberColumn(format="%.1f"),
                    "Total line": st.column_config.NumberColumn(format="%.1f"),
                    "Over %": st.column_config.NumberColumn(format="%.1f"),
                    "Market home win %": st.column_config.NumberColumn(format="%.1f"),
                    "Win % vs market": st.column_config.NumberColumn(format="%+.1f"),
                }
                cols = [c for c in ["Date", "Game", "Sim favorite", "Home win %", "Sim score (away-home)", "Sim margin (home)", "Line (home margin)", "Sim minus line", "Home covers %", "Sim total", "Total line", "Over %", "Market home win %", "Win % vs market", "QBs used"] if c in sdf.columns]
                cols = cols + (["Rest & travel"] if "Rest & travel" in sdf.columns else [])
                st.dataframe(sdf[cols], hide_index=True, column_config=cfg, height=min(36 * (len(sdf) + 1) + 3, 700))
                b1, b2 = st.columns([1, 1])
                b1.download_button("Download as CSV", sdf[[c for c in sdf.columns if not c.startswith("_")]].to_csv(index=False).encode("utf-8"), file_name=f"nfl_week_{sl['week']}_sim.csv", mime="text/csv")
                if b2.button("Save these picks to my tracker", key="slate_save"):
                    _allp, _ok = add_preds(slate_to_preds(sl, latest))
                    st.success(f"Saved {len(sdf)} picks for week {sl['week']}. See the Prediction tracker tab." + ("" if _ok else " (Could not write the file here, so they are kept for this session only; download a backup from the tracker tab.)"))
                st.caption("Sim minus line: positive means the simulator likes the home team more than the market does. QBs used shows each team's default starter, with the points-per-game effect in brackets when it isn't the QB the team ratings were built around. Not betting advice: the market knows things the model doesn't.")

with tab_bet:
    open_w = sorted(int(w) for w in sched[(sched["season"] == latest) & sched["home_score"].isna()]["week"].unique())
    if not open_w:
        st.info("There are no unplayed games left on this season's schedule.")
    else:
        st.caption("Turns the week slate's simulated probabilities into bets: how likely each side is to win, what price the book offers, and whether the gap is big enough to matter. Prices come from the schedule data (usually standard -110 on spreads and totals when a specific price isn't listed).")
        bc1, bc2 = st.columns([1, 1])
        bet_week = bc1.selectbox("Week", open_w, key="bet_week")
        bet_sims = bc2.select_slider("Simulations per game", [2000, 5000, 10000, 20000], value=5000, key="bet_sims")
        if st.button("Simulate this week", type="primary", key="bet_go"):
            with st.spinner("Simulating every game..."):
                compute_slate(bet_week, bet_sims)
        slb = st.session_state.get("slate")
        if slb is None:
            st.info("Pick a week and press Simulate this week (or run it on the Week slate tab).")
        else:
            st.subheader(f"Week {slb['week']} bets ({slb['sims']:,} simulations per game)")
            btt, btm = backtest(sched, decay, alpha, "cur", version)
            mkt = bt_market(btt) if len(btt) else None
            if mkt is not None and len(mkt["ats"]):
                r0 = mkt["ats"].iloc[0]
                st.warning(f"Reality check from the backtest: this season, on {mkt['n']} games with a posted line, the model's side covered {r0['Hit rate']} of the time ({r0['Model right']} of {r0['Bets']}). Breaking even at -110 takes 52.4%. The market was also more accurate than the model at picking winners and margins, so most of the 'edges' below are likely the model's blind spots, not real advantages.")
            k1, k2, k3 = st.columns(3)
            bankroll = k1.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=100.0, key="bet_bank")
            kfrac = k2.select_slider("Kelly fraction", [0.1, 0.25, 0.5, 1.0], value=0.25, key="bet_kelly", help="Full Kelly is very aggressive. Most people use a quarter or less.")
            min_edge = k3.slider("Minimum edge (points)", 0.0, 15.0, 3.0, 0.5, key="bet_edge", help="Edge = the simulator's win chance minus the chance the price needs to break even.")
            bets = make_bets(slb["df"], bankroll, kfrac)
            if bets.empty:
                st.write("No lines are posted for these games yet.")
            else:
                shown_b = bets[(bets["Edge (pts)"] >= min_edge) & (bets["_ev"] > 0)]
                st.write(f"{len(shown_b)} of {len(bets)} possible bets clear a {min_edge:g}-point edge.")
                cfgb = {"Sim win %": st.column_config.NumberColumn(format="%.1f"), "Needed to break even %": st.column_config.NumberColumn(format="%.1f"), "Edge (pts)": st.column_config.NumberColumn(format="%+.1f"),
                        "Expected profit per $100": st.column_config.NumberColumn(format="%+.1f"), "Stake ($)": st.column_config.NumberColumn(format="%.0f")}
                st.dataframe(shown_b.drop(columns=["_ev"]), hide_index=True, column_config=cfgb, height=min(36 * (len(shown_b) + 1) + 3, 650))
                with st.expander("Every bet, including the ones with no edge"):
                    st.dataframe(bets.drop(columns=["_ev"]), hide_index=True, column_config=cfgb)
            st.caption("Expected profit per $100 = simulated win chance times the payout, minus the loss chance (pushes count as zero). Stake is the Kelly-criterion size scaled by your fraction, capped at zero when there is no edge. This is a model's opinion, not advice, and betting can lose money.")

with tab_season:
    left = sched[(sched["season"] == latest) & sched["home_score"].isna()]
    st.caption("Plays out every remaining regular-season game thousands of times using the team ratings, then seeds the playoffs and plays them out too. Ties between teams are broken by wins, then net points (a simplification of the NFL's real tiebreakers), and quarterback and injury changes aren't included.")
    s1, s2 = st.columns([1, 1])
    ss_sims = s1.select_slider("Simulated seasons", [2000, 5000, 10000, 20000], value=5000, key="ss_sims")
    seed_n = st.session_state.setdefault("ss_seed", 1)
    if s2.button("Run season simulation", type="primary", key="ss_go"):
        st.session_state["ss_seed"] = seed_n + 1
        st.session_state["ss_on"] = True
    if len(left) == 0:
        st.info("The regular season is over. Use the bracket section below with the final seeds.")
    if st.session_state.get("ss_on"):
        ss = sim_season(R, sched, latest, ss_sims, st.session_state["ss_seed"], version, decay, alpha)
        teams_s = ss["teams"]
        st.write(f"{ss['games_left']} regular-season games left, {ss['sims']:,} simulated seasons.")
        rows = []
        for i, t in enumerate(teams_s):
            c = "AFC" if t in CONFS["AFC"] else "NFC"
            rec = f"{int(ss['W0'][i])}-{int(ss['L0'][i])}" + (f"-{int(ss['T0'][i])}" if ss["T0"][i] else "")
            stt = ss["stat"]
            rows.append({"Conf": c, "Division": TEAM_DIV.get(t, ""), "Team": t, "Record": rec, "Proj wins": float(ss["exp_wins"][i]), "Playoffs %": stt["playoffs"][i] * 100, "Win division %": stt["division"][i] * 100,
                         "#1 seed %": stt["seed1"][i] * 100, "Reach conf final %": stt["conf_final"][i] * 100, "Reach Super Bowl %": stt["super_bowl"][i] * 100, "Win Super Bowl %": stt["champion"][i] * 100})
        df_s = pd.DataFrame(rows)
        st.session_state["ss_df"] = df_s
        cfgs = {c: st.column_config.NumberColumn(format="%.1f") for c in df_s.columns if "%" in c or c == "Proj wins"}
        for conf in ("AFC", "NFC"):
            st.subheader(conf)
            show = df_s[df_s["Conf"] == conf].sort_values(["Playoffs %", "Proj wins"], ascending=False).drop(columns=["Conf"])
            st.dataframe(show, hide_index=True, column_config=cfgs, height=36 * 17 + 3)
        top = df_s.sort_values("Win Super Bowl %", ascending=False).head(12).set_index("Team")["Win Super Bowl %"]
        st.subheader("Super Bowl favorites")
        st.bar_chart(top)
        with st.expander("Most common Super Bowl matchups"):
            sp = ss["sb_pairs"].rename("Share of seasons").to_frame()
            sp["Share of seasons"] = (sp["Share of seasons"] * 100).round(1)
            st.dataframe(sp.rename(columns={"Share of seasons": "% of seasons"}))
        with st.expander("Seed odds (chance of finishing in each playoff seed)"):
            sd_tbl = pd.DataFrame(ss["seedprob"] * 100, index=teams_s, columns=[f"Seed {k}" for k in range(1, 8)]).round(1)
            sd_tbl.insert(0, "Conf", ["AFC" if t in CONFS["AFC"] else "NFC" for t in teams_s])
            sd_tbl["Any seed"] = sd_tbl[[f"Seed {k}" for k in range(1, 8)]].sum(axis=1).round(1)
            st.dataframe(sd_tbl.sort_values(["Conf", "Any seed"], ascending=[True, False]))
        st.subheader("Playoff bracket simulator")
        st.caption("Starts from the most likely seeds from the simulation above. Change any seed to test a scenario (for example, who wins if a particular team gets the #1 seed), then simulate that bracket on its own.")
        proj = projected_seeds(ss)
        bracket = {}
        for conf in ("AFC", "NFC"):
            st.markdown(f"**{conf} seeds**")
            cols_b = st.columns(7)
            sel = []
            for k in range(7):
                opts = CONFS[conf]
                sel.append(cols_b[k].selectbox(f"Seed {k + 1}", opts, index=opts.index(proj[conf][k]), key=f"br_{conf}_{k}_{st.session_state['ss_seed']}"))
            bracket[conf] = sel
        dup = [c for c in bracket if len(set(bracket[c])) < 7]
        if dup:
            st.error(f"Each team can only be used once per conference ({', '.join(dup)} has a repeat).")
        elif st.button("Simulate this bracket", key="br_go"):
            br_stat, br_games = run_bracket(R, bracket, 20000, int(st.session_state["ss_seed"]) + 7)
            rows_b = []
            for conf in ("AFC", "NFC"):
                for k, t in enumerate(bracket[conf]):
                    i = teams_s.index(t)
                    rows_b.append({"Conf": conf, "Seed": k + 1, "Team": t, "Win first game / bye %": br_stat["divisional"][i] * 100, "Reach conf final %": br_stat["conf_final"][i] * 100, "Reach Super Bowl %": br_stat["super_bowl"][i] * 100, "Win Super Bowl %": br_stat["champion"][i] * 100})
            st.session_state["br_res"] = (pd.DataFrame(rows_b), br_games)
        brr = st.session_state.get("br_res")
        if brr is not None and not dup:
            cfgb2 = {c: st.column_config.NumberColumn(format="%.1f") for c in brr[0].columns if "%" in c}
            st.dataframe(brr[0], hide_index=True, column_config=cfgb2, height=36 * 15 + 3)
            st.markdown("**Wild card round**")
            st.dataframe(brr[1], hide_index=True)
            st.caption("Single-elimination: the #1 seed gets a first-round bye, the higher seed hosts, the divisional round reseeds, and the Super Bowl is at a neutral site. 20,000 simulated playoffs.")
    else:
        st.info("Press Run season simulation.")

with t_pickem:
    st.subheader("Beat the Bot")
    st.caption("Pick the winner of every game before kickoff and see if you can beat the simulator. The Bot's pick is the model's favorite, saved by the daily robot before the games. "
               "The Bot's pick stays hidden until you've saved yours (or the game has started), so nobody can copy it.")
    _mode = store_mode()
    _need_code = _secret("PICKEM_CODE")
    _preds = load_preds()
    _preds = _preds[~_preds["late"].astype(bool)] if len(_preds) else _preds
    if len(_preds):
        _bot = _preds[["season", "week", "home", "away", "p_home"]].copy()
        _bot["pick"] = np.where(_bot["p_home"] >= 0.5, _bot["home"], _bot["away"])
        _bot = _bot[["season", "week", "home", "away", "pick"]].drop_duplicates(["season", "week", "home", "away"], keep="last")
    else:
        _bot = pd.DataFrame(columns=["season", "week", "home", "away", "pick"])
    _all = store_read(PICKEM_FILE, tuple(PICKEM_COLS), _mode)
    pe_weeks = sorted(int(w) for w in sched[sched["season"] == latest]["week"].unique())
    _open_w = [w for w in pe_weeks if sched[(sched["season"] == latest) & (sched["week"] == w) & sched["home_score"].isna()].shape[0] > 0]
    c1, c2, c3 = st.columns([2, 1, 1])
    pe_name = c1.text_input("Your name", key="pe_name", max_chars=20, placeholder="e.g. Alan").strip()
    pe_week = c2.selectbox("Week", _open_w or pe_weeks, key="pe_week")
    pe_code = c3.text_input("Group code", key="pe_code", type="password") if _need_code else ""
    _g = sched[(sched["season"] == latest) & (sched["week"] == pe_week)].sort_values(["gameday", "gametime", "home_team"])
    _mine = _all[(_all["player"].astype(str) == pe_name) & (_all["season"] == latest) & (_all["week"] == pe_week)] if pe_name else _all.iloc[0:0]
    _mine_map = {(r["home"], r["away"]): r["pick"] for r in _mine.to_dict("records")}
    _botmap = {(r["home"], r["away"]): r["pick"] for r in _bot[(_bot["season"] == latest) & (_bot["week"] == pe_week)].to_dict("records")}
    if not pe_name:
        st.info("Type your name to make picks. Use the same name each week so your record adds up.")
    choices = {}
    st.markdown("**Your picks**")
    for r in _g.to_dict("records"):
        hh, aa = r["home_team"], r["away_team"]
        final = pd.notna(r["home_score"])
        locked = final or is_locked(r)
        saved = _mine_map.get((hh, aa))
        lab = f"{aa} @ {hh}  ({str(r['gameday'])[5:]} {r.get('gametime') or ''} ET)" + ("  locked" if locked else "")
        idx = [aa, hh].index(saved) if saved in (aa, hh) else None
        sel = st.radio(lab, [aa, hh], index=idx, horizontal=True, key=f"pe_{pe_name}_{pe_week}_{hh}_{aa}", disabled=(locked or not pe_name))
        if not locked and sel:
            choices[(hh, aa)] = sel
        bp = _botmap.get((hh, aa))
        if bp and (saved or locked):
            tail = ""
            if final:
                win = hh if r["home_score"] > r["away_score"] else (aa if r["away_score"] > r["home_score"] else None)
                tail = f"  | final {aa} {int(r['away_score'])}-{int(r['home_score'])}: Bot {'right' if bp == win else 'wrong'}" + (f", you {'right' if saved == win else 'wrong'}" if saved else "")
            st.caption(f"Bot picked {bp}{tail}")
        elif not bp and (saved or locked):
            st.caption("The Bot has no saved pick for this game.")
    if st.button("Save my picks", type="primary", key="pe_save", disabled=not pe_name):
        if _need_code and pe_code != str(_need_code):
            st.error("That group code isn't right.")
        elif not choices:
            st.warning("Pick at least one game that hasn't started.")
        else:
            now = _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M")
            new = pd.DataFrame([{"saved_at": now, "season": int(latest), "week": int(pe_week), "player": pe_name, "home": k[0], "away": k[1], "pick": v} for k, v in choices.items()])
            ok, where = store_append(PICKEM_FILE, PICKEM_COLS, new, ["season", "week", "player", "home", "away"], msg=f"pick'em: {pe_name} week {pe_week}")
            if ok:
                st.success(f"Saved {len(new)} picks for {pe_name}." + (" Stored safely on GitHub." if where == "github" else " Stored in this app's own file, which can reset when the app restarts (see the setup notes to keep picks permanently)."))
                show_html(CONFETTI_HTML, 160)
            else:
                st.error("Couldn't save the picks.")
    st.markdown("---")
    st.subheader("Leaderboard")
    _all = store_read(PICKEM_FILE, tuple(PICKEM_COLS), _mode)
    if _all.empty:
        st.info("No picks yet. Be the first!")
    else:
        lb, brow, pk_g = pickem_score(_all, _bot, sched, latest)
        if brow["Picks"]:
            st.caption(f"The Bot is {brow['Right']}-{brow['Picks'] - brow['Right']} ({100.0 * brow['Right'] / brow['Picks']:.0f}%) on the graded games it has picks for.")
        if lb.empty:
            st.info("Nothing to grade yet. Scores appear once the games you picked have finished.")
        else:
            st.dataframe(lb, hide_index=True, column_config={"Win %": st.column_config.NumberColumn(format="%.1f")})
            if pe_name and pe_name in set(lb["Player"]) and "Bot slayer" in lb.loc[lb["Player"] == pe_name, "Badges"].iloc[0] and not st.session_state.get("pe_celebrated"):
                st.session_state["pe_celebrated"] = True
                st.success(f"{pe_name}, you've beaten the Bot in a week!")
                show_html(CONFETTI_HTML, 160)
        st.markdown("**This week's picks (shown once each game has started)**")
        wk_rows = []
        for r in _g.to_dict("records"):
            hh, aa = r["home_team"], r["away_team"]
            if not (pd.notna(r["home_score"]) or is_locked(r)):
                continue
            win = None
            if pd.notna(r["home_score"]) and r["home_score"] != r["away_score"]:
                win = hh if r["home_score"] > r["away_score"] else aa
            row_ = {"Game": f"{aa} @ {hh}"}
            bp = _botmap.get((hh, aa))
            row_["Bot"] = (bp + (" ok" if bp == win else "") if bp and win else bp) if bp else "-"
            for pl_name, gg in _all[(_all["season"] == latest) & (_all["week"] == pe_week) & (_all["home"] == hh) & (_all["away"] == aa)].groupby("player"):
                p_ = gg.iloc[-1]["pick"]
                row_[str(pl_name)] = p_ + (" ok" if (win and p_ == win) else "")
            wk_rows.append(row_)
        if wk_rows:
            st.dataframe(pd.DataFrame(wk_rows).fillna("-"), hide_index=True)
            st.caption('"ok" marks a correct pick once the game is final.')
        else:
            st.caption("No games from this week have started yet.")
    if _mode == "local":
        st.caption("Picks are being saved to a file inside the app. On Streamlit Cloud that file can reset when the app restarts. Add GH_TOKEN and GH_REPO in the app's Secrets to keep them permanently on a separate GitHub branch (setup notes included).")

with tab_race:
    st.caption("Watch the title race unfold. For every completed week, the ratings are rebuilt using only the games played by then, the rest of the season is simulated, and the odds are plotted. Nothing here looks ahead.")
    _done = sched[(sched["season"] == latest) & sched["home_score"].notna()]
    _lw = int(_done["week"].max()) if len(_done) else 0
    if _lw < 2:
        st.info("The race starts once two weeks have been played.")
    else:
        r1, r2 = st.columns(2)
        rc_metric = r1.selectbox("Race for", ["Win the Super Bowl", "Make the playoffs", "Win the division", "Get the #1 seed", "Reach the Super Bowl", "Projected wins"], key="race_metric")
        rc_conf = r2.selectbox("Teams", ["All 32", "AFC", "NFC"], key="race_conf")
        if st.button("Build the race", type="primary", key="race_go") or st.session_state.get("race_on"):
            st.session_state["race_on"] = True
            _rdf = retro_odds(sched, latest, decay, alpha, 3000, _lw, version)
            show_html(race_html(_rdf, rc_metric, rc_conf, f"{rc_metric}: {rc_conf}"), 440)
            st.caption("Press Play to watch it move week by week, or drag the slider. Each bar is a team's chance in the simulated seasons after that week.")
            with st.expander("Table: odds after the latest week"):
                _lt = _rdf[_rdf["week"] == _lw].drop(columns=["week"]).set_index("team").sort_values("Win the Super Bowl", ascending=False)
                st.dataframe(_lt.style.format("{:.1f}") if hasattr(_lt, "style") else _lt)
        else:
            st.info("Press Build the race. The first time takes a few seconds per week of football; after that it's saved.")
    _hist = load_odds_history()
    if _hist is not None and _hist["date"].nunique() >= 2:
        st.subheader("Daily snapshots")
        st.caption("Taken each morning by the daily robot, so they include moves from injuries, lineups and rest as well as results.")
        _pick = st.multiselect("Teams", sorted(_hist["team"].unique()), default=list(_hist.sort_values("super_bowl", ascending=False)["team"].drop_duplicates().head(5)), key="race_hist_teams")
        if _pick:
            _pv = _hist[_hist["team"].isin(_pick)].pivot_table(index="date", columns="team", values="super_bowl")
            st.line_chart(_pv)
            st.caption("Chance to win the Super Bowl (%).")

with tab_riv:
    st.caption(f"Results since 2017 (the seasons in this app's data), plus what the simulator says about the next meeting.")
    _tl = sorted(R["off"].keys())
    ra, rb = st.columns(2)
    _ra = ra.selectbox("Team", _tl, index=_tl.index(home) if home in _tl else 0, key="riv_a")
    _rb = rb.selectbox("Rival", [t for t in _tl if t != _ra], index=0 if away == _ra or away not in _tl else [t for t in _tl if t != _ra].index(away), key="riv_b")
    _sg = series_games(sched, _ra, _rb)
    _ss = series_summary(_sg)
    if _ss is None:
        st.info("No completed games between these teams in the data.")
    else:
        kc = st.container(key="kpis_riv")
        c = kc.columns(4)
        c[0].metric("Series", f"{_ra} {_ss['a_w']}-{_ss['b_w']}" + (f"-{_ss['t']}" if _ss["t"] else ""))
        c[1].metric(f"Avg margin ({_ra})", f"{_ss['avg_margin']:+.1f}")
        c[2].metric("Avg total points", f"{_ss['avg_total']:.1f}")
        c[3].metric("Decided by 7 or less", f"{_ss['close']} of {_ss['games']}")
        _chart = _sg.assign(label=_sg["season"].astype(int).astype(str) + " wk" + _sg["week"].astype(int).astype(str)).set_index("label")["a_margin"]
        st.bar_chart(_chart)
        st.caption(f"Margin in each game, {_ra} perspective (above zero means {_ra} won).")
        _tbl = _sg.assign(Game=_sg["away_team"] + " @ " + _sg["home_team"], Score=_sg["away_score"].astype(int).astype(str) + "-" + _sg["home_score"].astype(int).astype(str))[["season", "week", "Game", "Score"]].rename(columns={"season": "Season", "week": "Week"})
        st.dataframe(_tbl.iloc[::-1], hide_index=True)
    _nx = sched[(sched["season"] == latest) & sched["home_score"].isna() & (((sched["home_team"] == _ra) & (sched["away_team"] == _rb)) | ((sched["home_team"] == _rb) & (sched["away_team"] == _ra)))].sort_values("week")
    st.markdown("**What the simulator says**")
    _S_r = S_all if (use_style and S_all) else None
    _sims = []
    for _h2, _a2 in ((_ra, _rb), (_rb, _ra)):
        _g2 = simulate_game(R, _S_r, _h2, _a2, 10000, False)
        _sims.append(f"At {_h2}: {_h2} {float((_g2['hs'] > _g2['as']).mean()) * 100:.0f}% / {_a2} {float((_g2['as'] > _g2['hs']).mean()) * 100:.0f}% (avg {_h2} {_g2['hs'].mean():.0f} - {_a2} {_g2['as'].mean():.0f})")
    st.write("  \n".join(_sims))
    if len(_nx):
        _r0 = _nx.iloc[0]
        st.caption(f"Next meeting: week {int(_r0['week'])}, {_r0['away_team']} @ {_r0['home_team']} ({str(_r0['gameday'])[5:]}).")
    else:
        st.caption("These teams don't meet again this regular season.")
    with st.expander("Every division rivalry"):
        _dv = division_rivalry_table(sched)
        st.dataframe(_dv, hide_index=True, column_config={"Avg margin (first team)": st.column_config.NumberColumn(format="%+.1f"), "Avg total": st.column_config.NumberColumn(format="%.1f"),
                                                        "Decided by 7 or less %": st.column_config.NumberColumn(format="%.0f")})


with tab_track:
    st.caption("Save a week's slate picks before kickoff, and the tracker grades them against the real results as games finish: how often the simulator picked the winner, how good its percentages were, and its record against the spread and totals.")
    preds = load_preds()
    inc_late = st.checkbox("Include picks saved after the game day", value=False, key="trk_late", help="Picks saved a day or more after the game day are flagged late and left out by default, since the result may have been known.")
    pcol = st.columns([1, 1, 1])
    pcol[0].download_button("Download my picks (backup)", preds.to_csv(index=False).encode("utf-8"), file_name="nfl_predictions.csv", mime="text/csv", disabled=preds.empty)
    up = pcol[1].file_uploader("Restore from a backup file", type=["csv"], key="trk_up", label_visibility="collapsed")
    if up is not None and st.session_state.get("trk_up_done") != getattr(up, "file_id", up.name):
        try:
            add_preds(pd.read_csv(up))
            st.session_state["trk_up_done"] = getattr(up, "file_id", up.name)
            st.rerun()
        except Exception:
            st.error("That file didn't look like a picks backup.")
    st.caption("Where picks are stored: a file next to the app. If the app runs on Streamlit Cloud, that file is shared by everyone who opens your link and is erased whenever the app restarts, so download a backup now and then and restore it here.")
    if preds.empty:
        st.info("No saved picks yet. Simulate a week on the Week slate tab and press Save these picks to my tracker.")
    else:
        gm = grade_preds(preds, sched, inc_late)
        sm = tracker_summary(gm)
        if sm["n"] == 0:
            st.info(f"{len(preds)} picks saved, but none of those games have finished yet.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Graded games", sm["n"])
            if "acc" in sm:
                k2.metric("Picked the winner", f"{sm['acc'] * 100:.1f}%")
                k3.metric("Brier score", f"{sm['brier']:.3f}", help="Lower is better. Always saying 50% scores 0.250.")
            if "ats" in sm:
                w, l, pu = sm["ats"]
                k4.metric("Against the spread", f"{w}-{l}" + (f"-{pu}" if pu else ""), help="Wins-losses-pushes for backing the side the simulator liked versus the posted line")
            if "n_m" in sm:
                st.write(f"Versus the market on the {sm['n_m']} graded games with moneylines: model picked the winner {sm['acc_model_m'] * 100:.1f}% (market {sm['acc_mkt_m'] * 100:.1f}%), Brier {sm['brier_model_m']:.3f} (market {sm['brier_mkt_m']:.3f}).")
            extra = []
            if "ats" in sm:
                extra.append(f"Spread picks at -110: {units_110(sm['ats']):+.1f} units")
            if "ou" in sm:
                w, l, pu = sm["ou"]
                extra.append(f"Over/under picks: {w}-{l}" + (f"-{pu}" if pu else "") + f" ({units_110(sm['ou']):+.1f} units at -110)")
            if extra:
                st.write(" | ".join(extra) + ". One unit is one bet's stake. A small sample like this says very little.")
        g2 = gm.copy()
        g2["Game"] = g2["away"] + " @ " + g2["home"]
        g2["Sim home win %"] = g2["p_home"] * 100
        g2["Sim pick"] = np.where(g2["p_home"] >= 0.5, g2["home"], g2["away"])
        g2["Result"] = np.where(g2["done"], g2["away"] + " " + g2["away_score"].fillna(0).astype(int).astype(str) + " - " + g2["home"] + " " + g2["home_score"].fillna(0).astype(int).astype(str), "not played")
        g2["Correct"] = np.where(~g2["done"], "", np.where(g2["act_m"] == 0, "tie", np.where((g2["p_home"] > 0.5) == (g2["act_m"] > 0), "yes", "no")))
        g2 = g2.sort_values(["season", "week", "Game"], ascending=[False, False, True])
        cfg_t = {"Sim home win %": st.column_config.NumberColumn(format="%.1f"), "week": st.column_config.NumberColumn("Week", format="%d")}
        st.dataframe(g2[["week", "Game", "Sim pick", "Sim home win %", "Result", "Correct", "saved_at", "late"]].rename(columns={"saved_at": "Saved (UTC)", "late": "Saved late"}), hide_index=True, column_config=cfg_t, height=min(36 * (len(g2) + 1) + 3, 600))
        with st.expander("Delete all saved picks"):
            if st.checkbox("Yes, delete everything I've saved", key="trk_del_ok") and st.button("Delete", key="trk_del"):
                save_preds(pd.DataFrame(columns=PRED_COLS))
                st.rerun()

with tab_bt:
    st.caption("A fair test of the team ratings: for every week, the ratings are refit using only games played before that week, then used to predict that week's games. Injury, QB and team-style adjustments can't be replayed from history, so this tests the core ratings, not those add-ons. Win chances come from each predicted margin and the model's typical miss.")
    scope_lab = st.radio("Games to replay", ["This season", "Last season and this season"], horizontal=True, key="bt_scope")
    scope = "cur" if scope_lab == "This season" else "two"
    bt, bt_sigma = backtest(sched, decay, alpha, scope, version)
    if bt.empty:
        st.info("Not enough finished games to replay yet.")
    else:
        sm = bt_summary(bt)
        mk = bt_market(bt)
        st.subheader("Model results")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Games replayed", f"{sm['n']}")
        k2.metric("Picked the winner", f"{sm['acc'] * 100:.1f}%")
        k3.metric("Average miss on the margin", f"{sm['mae']:.1f} pts", help="Average of |actual margin - predicted margin|")
        k4.metric("Brier score", f"{sm['brier']:.3f}", help="Lower is better. Always saying 50% scores 0.250.")
        st.caption(f"Settings used: recent-season weight {decay}, smoothing {alpha:g} (your sidebar sliders). Typical miss on a game's margin: {bt_sigma:.1f} points.")
        if mk is not None and mk["n"] >= 10:
            st.subheader(f"Against the market ({mk['n']} games with a posted line)")
            rows = [{"": "Average miss on the margin", "Model": f"{mk['mae_model']:.2f} pts", "Market": f"{mk['mae_mkt']:.2f} pts"}]
            if mk.get("acc_model") is not None:
                rows.append({"": "Picked the winner", "Model": f"{mk['acc_model'] * 100:.1f}%", "Market": f"{mk['acc_mkt'] * 100:.1f}%"})
            if "brier_model" in mk:
                rows.append({"": "Brier score (lower is better)", "Model": f"{mk['brier_model']:.3f}", "Market": f"{mk['brier_mkt']:.3f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            if len(mk["ats"]):
                st.markdown("**Picking sides against the spread** (the model backs whichever side it rates better than the line; pushes are left out)")
                st.dataframe(mk["ats"], hide_index=True)
                st.caption("A bet needs about 52.4% to break even at -110. With this few games a hit rate within a few points of 50% is noise, and the closing market is very hard to beat.")
        st.subheader("Are the percentages honest?")
        cal = bt_calibration(bt)
        st.dataframe(cal, hide_index=True)
        if len(cal):
            st.bar_chart(cal.set_index("Favorite's predicted chance")[["Predicted", "Actual"]])
        st.caption("Each row groups games by how likely the model said the favorite was to win, then shows how often the favorite really won. Predicted and Actual should be close.")
        wk = bt.assign(right=((bt["pred_m"] > 0) == (bt["act_m"] > 0)) & (bt["act_m"] != 0)).groupby(["Season", "Week"]).agg(Games=("right", "size"), Right=("right", "sum"), MAE=("act_m", lambda s: 0.0)).reset_index()
        wk["MAE"] = bt.assign(e=(bt["act_m"] - bt["pred_m"]).abs()).groupby(["Season", "Week"])["e"].mean().values.round(1)
        wk["Accuracy"] = (wk["Right"] / wk["Games"] * 100).round(1)
        with st.expander("Week by week"):
            st.dataframe(wk.rename(columns={"MAE": "Avg miss (pts)", "Right": "Winners picked", "Accuracy": "Accuracy %"}), hide_index=True)
        with st.expander("Every game"):
            g = bt.assign(Predicted=(bt["pred_m"]).round(1), Actual=bt["act_m"].round(0), **{"Home win %": (bt["p_home"] * 100).round(1)})
            st.dataframe(g[["Season", "Week", "Away", "Home", "Predicted", "Actual", "Home win %", "spread"]].rename(columns={"Predicted": "Predicted home margin", "Actual": "Actual home margin", "spread": "Line (home margin)"}), hide_index=True)
        st.subheader("Try other settings")
        st.caption("Replays the same games with different recent-season weights and smoothing. It takes a short while. Treat small differences as noise: with this many games, gaps under about 0.003 in Brier score don't mean much, and tuning hard on a few hundred games just fits luck.")
        if st.button("Compare settings", key="bt_tune"):
            with st.spinner("Replaying with 25 setting combinations..."):
                st.session_state["bt_grid"] = (scope, tune_grid(sched, scope, version))
        gd = st.session_state.get("bt_grid")
        if gd is not None and gd[0] == scope and len(gd[1]):
            grid = gd[1]
            best = grid.loc[grid["brier"].idxmin()]
            st.write(f"Best Brier score in this test: recent-season weight {best['decay']:g}, smoothing {best['alpha']:g} (Brier {best['brier']:.3f}). You are using {decay}, {alpha:g}.")
            pv = grid.pivot(index="decay", columns="alpha", values="brier").round(4)
            pv.index.name, pv.columns.name = "Recent-season weight", "Smoothing"
            st.markdown("Brier score (lower is better)")
            st.dataframe(pv)
            pm = grid.pivot(index="decay", columns="alpha", values="mae").round(2)
            pm.index.name, pm.columns.name = "Recent-season weight", "Smoothing"
            st.markdown("Average miss on the margin, points (lower is better)")
            st.dataframe(pm)

with tab_roster:
    if ros_df.empty:
        st.info("Roster data could not be loaded right now. Try again in a few minutes.")
    else:
        moves_all = roster_moves(ros_df)
        if ov_eff:
            st.info("Manual moves active in the simulator (not shown in the tables below): " + ", ".join(f"{v['name']} to {v['team']}" for v in ov_eff.values()))
        st.caption(f"Built from the nflverse weekly rosters, which currently run through week {ros_week}. The app rechecks hourly, so new signings, trades, cuts and IR moves appear as soon as nflverse publishes them. Moves are found by comparing each week's roster with the one before it, so a move shows up in the first week it is visible in the data.")
        cA, cB = st.columns([1, 1])
        rteam = cA.selectbox("Team", teams, index=teams.index(home), key="ros_team")
        only_model = cB.checkbox("Only QB / RB / WR / TE (the positions the simulator uses)", value=True, key="ros_model")
        team_rows = ros_df[ros_df["team"] == rteam]
        team_wk = int(team_rows["week"].max()) if len(team_rows) else ros_week
        cur = team_rows[team_rows["week"] == team_wk].copy()
        if only_model:
            cur = cur[cur["position"].isin(MODELED)]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active", int(cur["status"].isin(ON_ROSTER).sum()))
        m2.metric("Injured reserve", int((cur["status"] == "RES").sum()))
        m3.metric("Practice squad", int((cur["status"] == "DEV").sum()))
        m4.metric("Other", int((~cur["status"].isin(list(ON_ROSTER) + ["RES", "DEV"])).sum()))

        st.subheader(f"{rteam} roster (week {team_wk})")
        joined = {}
        if not moves_all.empty:
            arr = moves_all[(moves_all["To"] == rteam) & moves_all["Move"].isin(["Changed teams", "Added to roster"])]
            joined = arr.groupby("gsis_id")["week"].max().to_dict()
            frm = arr.sort_values("week").groupby("gsis_id")["From"].last().to_dict()
        else:
            frm = {}
        order = {p: i for i, p in enumerate(["QB", "RB", "FB", "WR", "TE", "T", "G", "C", "OL", "DE", "DT", "DL", "NT", "LB", "OLB", "ILB", "MLB", "CB", "S", "SS", "FS", "DB", "K", "P", "LS"])}
        cur["o1"] = cur["position"].map(order).fillna(99)
        cur["o2"] = cur["status"].map({"ACT": 0, "INA": 1, "RES": 2, "DEV": 3}).fillna(4)
        cur = cur.sort_values(["o1", "o2", "full_name"])
        tbl = pd.DataFrame({
            "Player": cur["full_name"].values,
            "Pos": cur["position"].values,
            "#": cur["jersey_number"].values,
            "Status": [STAT_TXT.get(s, s).capitalize() for s in cur["status"]],
            "On team since": [(f"week {joined[p]}" + (f" (from {frm[p]})" if isinstance(frm.get(p), str) else "")) if p in joined else "start of season" for p in cur["gsis_id"]],
            "Years": cur["years_exp"].values,
        })
        st.dataframe(tbl, hide_index=True, height=min(36 * (len(tbl) + 1), 600))

        st.subheader("Recent moves")
        f1, f2 = st.columns([1, 1])
        scope = f1.radio("Show moves for", [f"{rteam} only", "Every team"], horizontal=True, key="ros_scope")
        span = f2.slider("How many recent weeks", 1, max(ros_week, 1), min(3, max(ros_week, 1)), key="ros_span")
        mv = moves_all[moves_all["week"] > ros_week - span].copy()
        if scope != "Every team":
            mv = mv[(mv["From"] == rteam) | (mv["To"] == rteam)]
        if only_model:
            mv = mv[mv["Pos"].isin(MODELED)]
        mv["Team"] = np.where(mv["To"].notna(), mv["To"], mv["From"])
        if mv.empty:
            st.write("No moves found for these filters.")
        else:
            st.dataframe(mv.rename(columns={"week": "Week"})[["Week", "Team", "Player", "Pos", "Move", "Detail"]], hide_index=True, height=min(36 * (len(mv) + 1), 600))

        with st.expander("Official trades list (nflverse)"):
            tr = load_trade_list(latest)
            if tr.empty:
                st.write("No trades found for this season in the trades table.")
            else:
                st.caption("Players only (draft picks left out). This list can lag behind the weekly rosters, so a recent trade may show up in Recent moves first.")
                if scope != "Every team":
                    tr = tr[(tr["From"] == rteam) | (tr["To"] == rteam)]
                st.dataframe(tr.head(60), hide_index=True)

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

def game_view(res):
    """The game the Field, Box score and Plays tabs are showing, with its expanded plays (built once per game)."""
    mode = st.session_state.get("gsel", "Most typical game")
    key = "typical" if mode.startswith("Most") else "random"
    g = res["typical"] if key == "typical" else res["one"]
    fx = res.setdefault("fx", {})
    if key not in fx or fx[key][0] is not g:
        fx[key] = (g, expand_game(g, res["prep"]))
    return g, fx[key][1], key


def _slip_odds_default(mk, market, pick, home):
    if market == "Moneyline" and mk:
        v = mk.get("ml_home") if pick == home else mk.get("ml_away")
        if v:
            return float(v)
    return -110.0


def show_math(pw, pp, pl, odds, stake, bankroll, frac):
    m = bet_math(pw, pp, pl, odds, stake, bankroll, frac)
    kc = st.container(key="kpis_math")
    c = kc.columns(4)
    c[0].metric("Model win chance", f"{pw * 100:.1f}%", help="From the simulated games for this matchup" + (f" (push {pp * 100:.1f}%)" if pp > 0.002 else ""))
    c[1].metric("Break-even chance", f"{m['be'] * 100:.1f}%", help="The win chance this price needs just to break even")
    c[2].metric("Profit per $100 (model)", f"{m['ev'] * 100:+.1f}", help="Expected profit on a $100 bet if the model's chances were exactly right")
    c[3].metric("Kelly stake", f"${m['stake_k']:,.0f}" if m["kelly"] > 0 else "$0", help="Fractional-Kelly size for your bankroll; zero means the model sees no edge")
    return m


if res is None:
    for t in (tab_game, tab_anim, tab_box, tab_pbp, tab_players, tab_props, tab_slip, tab_sgp):
        with t:
            st.info("Pick the two teams at the top and press Simulate game.")
else:
    h, a = res["home"], res["away"]
    with tab_game:
        hs, as_ = res["game"]["hs"], res["game"]["as"]
        for note in res["notes"]:
            st.info(note)
        _mk = res.get("line") or {}
        _ph = float((res["game"]["hs"] > res["game"]["as"]).mean())
        st.markdown(game_card_html(a, h, _ph, float(res["game"]["as"].mean()), float(res["game"]["hs"].mean()), _mk.get("spread"), _mk.get("total"), _mk.get("p_home"),
                                   date="Neutral site" if res.get("neutral") else "Home field: " + h, note=res.get("rest_note", "")).replace('class="g7card"', 'class="g7card g7hero"'), unsafe_allow_html=True)
        with st.container(key="kpis_game"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{a} win", f"{(as_ > hs).mean() * 100:.1f}%")
            c2.metric(f"{h} win", f"{(hs > as_).mean() * 100:.1f}%")
            c3.metric(f"Avg score ({a}-{h})", f"{as_.mean():.1f} - {hs.mean():.1f}")
            c4.metric("Most common final", f"{res['game']['common'][1]} - {res['game']['common'][0]}", help=f"{a} - {h}")
        margin = (hs - as_).astype(int)
        top = pd.Series(np.abs(margin)).value_counts(normalize=True).head(3)
        st.caption(f"Average margin ({h}): {(hs - as_).mean():+.1f} | Average total points: {(hs + as_).mean():.1f} | {len(hs):,} simulations | most common margins: " + ", ".join(f"{int(k)} pts ({v * 100:.1f}%)" for k, v in top.items()))
        mc = np.clip(margin, -40, 40)
        dist = pd.Series(mc).value_counts().sort_index().reindex(range(-40, 41), fill_value=0)
        st.subheader(f"How often each margin happens (positive = {h} wins)")
        st.bar_chart(dist / len(mc) * 100)
        st.caption("Scores are built from touchdowns, extra points, field goals, and the occasional defensive score, so common football numbers (3, 7, 10, 14) show up more often than in a smooth bell curve. Watch a full game play out on the Field tab.")
        show_line_box(res)
        st.subheader("What-if lab")
        st.caption("Change one thing and see how the odds move. This quick version re-simulates the game from the team ratings (and team style, if it's on) but skips the play-by-play game flow, so it's best for comparing scenarios against each other.")
        _base_adj = dict(res.get("adj") or {h: 0.0, a: 0.0})
        _qd = {}
        wq = st.columns(2)
        for _col, _t in zip(wq, (h, a)):
            _cur = (res.get("qbcur") or {}).get(_t)
            _opts = ["Same QB"] + [q["name"] for q in qbs.get(_t, []) if not _cur or q["name"] != _cur["name"]]
            _ch = _col.selectbox(f"{_t} quarterback", _opts, key=f"wif_qb_{_t}")
            _qd[_t] = 0.0
            if _ch != "Same QB" and _cur is not None:
                _q2 = next(q for q in qbs[_t] if q["name"] == _ch)
                _qd[_t] = (_q2["rate"] - _cur["rate"]) * DROPBACKS * QB_WEIGHT
        wo = st.columns(4)
        _oh = wo[0].slider(f"{h} offense", -7.0, 7.0, 0.0, 0.5, key="wif_oh")
        _dh = wo[1].slider(f"{h} defense", -7.0, 7.0, 0.0, 0.5, key="wif_dh", help="+ means the defense allows more points")
        _oa = wo[2].slider(f"{a} offense", -7.0, 7.0, 0.0, 0.5, key="wif_oa")
        _da = wo[3].slider(f"{a} defense", -7.0, 7.0, 0.0, 0.5, key="wif_da", help="+ means the defense allows more points")
        _neu = st.checkbox("Play it at a neutral site", value=bool(res["neutral"]), key="wif_neu")
        if st.button("Run what-if", key="wif_go"):
            _new = {h: _base_adj.get(h, 0.0) + _qd[h] + _oh + _da, a: _base_adj.get(a, 0.0) + _qd[a] + _oa + _dh}
            _b0 = simulate_game(R, res["S"], h, a, 20000, res["neutral"], _base_adj)
            _b1 = simulate_game(R, res["S"], h, a, 20000, _neu, _new)
            st.session_state["wif"] = {"h": h, "a": a, "b0": (_b0["hs"], _b0["as"]), "b1": (_b1["hs"], _b1["as"])}
        _w = st.session_state.get("wif")
        if _w and _w["h"] == h and _w["a"] == a:
            def _sm(hs_, as_):
                return {f"{h} win %": float((hs_ > as_).mean()) * 100, f"{a} win %": float((as_ > hs_).mean()) * 100, f"Avg margin ({h})": float((hs_ - as_).mean()), "Avg total": float((hs_ + as_).mean())}
            _s0, _s1 = _sm(*_w["b0"]), _sm(*_w["b1"])
            _tb = pd.DataFrame({"Metric": list(_s0), "Now": [_s0[k] for k in _s0], "What-if": [_s1[k] for k in _s0], "Change": [_s1[k] - _s0[k] for k in _s0]})
            st.dataframe(_tb, hide_index=True, column_config={c_: st.column_config.NumberColumn(format="%.1f") for c_ in ("Now", "What-if")} | {"Change": st.column_config.NumberColumn(format="%+.1f")})
            _d = _s1[f"{h} win %"] - _s0[f"{h} win %"]
            st.caption(f"The change moves {h}'s win chance by {_d:+.1f} points.")
    with tab_anim:
        st.radio("Game to watch", ["Most typical game", "A new random game"], horizontal=True, key="gsel", help="The typical game is the simulated game closest to the averages. A random game is one draw from the same simulator.")
        if st.session_state.get("gsel", "").startswith("A new"):
            if st.button("Simulate another random game", key="new_rand"):
                res["one"] = run_game(res["prep"], res["flow"])
                res.setdefault("fx", {}).pop("random", None)
        g_view, plays_view, vkey = game_view(res)
        st.caption(f"{a} @ {h}" + (" (OT)" if g_view["ot"] else "") + f" final: {a} {g_view['score'][a]}, {h} {g_view['score'][h]}. Tap a player for his stats.")
        show_html(build_field_html(g_view, res["prep"], float(res["game"]["mu"][0] - res["game"]["mu"][1]), plays_view, jersey_numbers(ros_df, (h, a))), 1500)
    with tab_players:
        st.caption("Format: average (10th-90th percentile range). Based on each selected player's last 6 games played, scaled to the simulated score.")
        for t in (h, a):
            st.subheader(f"{t}  (avg simulated score {res['players']['avg_pts'][t]:.1f})")
            st.dataframe(res["players"]["tables"][t], hide_index=True)
    with tab_box:
        g_view, plays_view, vkey = game_view(res)
        st.caption(f"Box score of the {'most typical' if vkey == 'typical' else 'random'} simulated game (switch games on the Field tab). Every number is added up from the individual plays you can watch there.")
        bx = box_score(plays_view, h, a)
        fr = box_frames(bx, plays_view, h, a)
        st.markdown('**Players of the game**')
        st.markdown(stars_html(three_stars(bx, h, a)), unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        k1.metric(a, g_view["score"][a])
        k2.metric(h, g_view["score"][h])
        team_df = pd.DataFrame({a: pd.Series(fr[a]["team"]).astype(str), h: pd.Series(fr[h]["team"]).astype(str)})
        st.dataframe(team_df)
        for t in (a, h):
            st.subheader(t)
            for lab, key in (("Passing", "pass"), ("Rushing", "rush"), ("Receiving", "rec")):
                if len(fr[t][key]):
                    st.caption(lab)
                    st.dataframe(fr[t][key], hide_index=True)
    with tab_pbp:
        g_view, plays_view, vkey = game_view(res)
        if vkey == "typical":
            st.subheader(f"Averages across {res['drive_sims']:,} simulated games")
            st.dataframe(res["avg"], hide_index=True)
        st.subheader(f"{a} @ {h}" + ("  (went to OT)" if g_view["ot"] else ""))
        st.dataframe(summary_df(g_view), hide_index=True)
        if st.checkbox("Broadcast commentary", key="pbp_comm", help="A play-by-play call of the key moments in this simulated game."):
            _all_c = st.checkbox("Call every play, not just the big ones", key="pbp_comm_all")
            _cl = commentary_lines(plays_view, h, a, only_key=not _all_c)
            st.markdown("\n\n".join(_cl[:80]) if _cl else "Nothing to call yet.", unsafe_allow_html=True)
            if len(_cl) > 80:
                st.caption(f"Showing the first 80 of {len(_cl)} calls.")
        opts = ["All drives"] + [f"{i + 1}. {d['team']} - {d['text']}" for i, d in enumerate(g_view["drives"])]
        pick_d = st.selectbox("Show plays from", opts, key="pbp_drive")
        only_key = st.checkbox("Key plays only", key="pbp_key")
        rows = []
        for p in plays_view:
            if pick_d != "All drives" and p["d"] != int(pick_d.split(".")[0]) - 1:
                continue
            if only_key and not p["key"]:
                continue
            rows.append({"Drive": p["d"] + 1, "Clock": f"{p['per']} {p['clock'] // 60}:{p['clock'] % 60:02d}", "Team": p["team"], "Situation": p.get("at", ""), "Play": p["text"], "Score": f"{a} {p['as']} - {h} {p['hs']}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(36 * (len(rows) + 1) + 3, 600))
        with st.expander("Drive summary table"):
            ddf = drive_df(g_view)
            st.dataframe(ddf, hide_index=True, height=36 * (len(ddf) + 1))
    draws = res["players"]["draws"]
    with tab_props:
        board = prop_board(draws)
        st.caption("Every number here comes from simulating the game thousands of times. The line is the model's own fair line (the over is about 50/50). No sportsbook prop lines are in the free data, so to price a line you actually see, open Place a bet.")
        if board.empty:
            st.info("No player projections for this game.")
        else:
            f1, f2, f3 = st.columns(3)
            tsel = f1.selectbox("Team", ["Both", h, a], key="pb_team")
            psel = f2.selectbox("Prop", ["All"] + sorted(board["Prop"].unique()), key="pb_prop")
            qsel = f3.text_input("Player search", key="pb_q")
            b2 = board
            if tsel != "Both":
                b2 = b2[b2["Team"] == tsel]
            if psel != "All":
                b2 = b2[b2["Prop"] == psel]
            if qsel:
                b2 = b2[b2["Player"].str.contains(qsel, case=False, regex=False)]
            st.dataframe(b2, hide_index=True, column_config={"Line": st.column_config.NumberColumn(format="%.1f"), "Over %": st.column_config.NumberColumn(format="%.0f"), "Under %": st.column_config.NumberColumn(format="%.0f")}, height=min(36 * (len(b2) + 1) + 3, 620))
    with tab_slip:
        prof = safe_profile(st.session_state.get("bet_profile", "me"))
        bankroll = float(st.session_state.get("bet_bank0", 1000.0))
        frac = float(st.session_state.get("bet_frac", 0.25))
        gm = find_any_game(sched, latest, h, a)
        mk = res.get("line")
        st.caption(f"Betting as **{prof}** (change the profile name, bankroll and Kelly fraction on My bets). The model prices any line you type against this simulated game. It is a model's opinion, not advice, and bets can lose money.")
        kind = st.radio("What are you betting on?", ["Game", "Player prop"], horizontal=True, key="slip_kind")
        row = None
        if kind == "Game":
            market = st.selectbox("Bet type", ["Moneyline", "Spread", "Total points"], key="slip_market")
            if market == "Total points":
                pick = st.radio("Pick", ["Over", "Under"], horizontal=True, key="slip_pick_t")
                dflt_line = float(mk["total"]) if mk and mk.get("total") is not None else float(np.round((res["game"]["hs"] + res["game"]["as"]).mean() * 2) / 2)
                line = st.number_input("Total", value=dflt_line, step=0.5, key=f"slip_line_t_{dflt_line}")
                mname = "Total"
            else:
                pick = st.radio("Pick", [h, a], horizontal=True, key=f"slip_pick_{market}")
                if market == "Spread":
                    base = (-float(mk["spread"]) if pick == h else float(mk["spread"])) if mk and mk.get("spread") is not None else 0.0
                    line = st.number_input(f"{pick} spread (as the sportsbook shows it, e.g. -3.5 or +7)", value=base, step=0.5, key=f"slip_line_{pick}_{base}")
                else:
                    line = 0.0
                mname = market
            odds = st.number_input("Odds (American, e.g. -110 or +150)", value=_slip_odds_default(mk, market, pick, h), step=5.0, key=f"slip_odds_{market}_{pick}")
            stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0, key="slip_stake")
            pw, pp, pl_ = p_game(res["game"]["hs"], res["game"]["as"], mname if mname != "Total" else "Total", pick, float(line), h, a)
            m = show_math(pw, pp, pl_, odds, stake, bankroll, frac)
            desc = f"{pick} {line:+g} (spread)" if market == "Spread" else (f"{pick} {line:g} total points" if market == "Total points" else f"{pick} moneyline")
            row = {"kind": "game", "market": mname if market != "Total points" else "Total", "pick": pick, "line": float(line), "player": None, "team": None, "stat": None, "desc": desc}
        else:
            opts = sorted({(t, n) for (t, n, s_) in draws}, key=lambda z: (z[0], z[1]))
            who = st.selectbox("Player", [f"{n} ({t})" for t, n in opts], key="slip_player")
            pname_, pteam = who.rsplit(" (", 1)[0], who.rsplit(" (", 1)[1][:-1]
            stats = [s_ for (t, n, s_) in draws if t == pteam and n == pname_]
            stat = st.selectbox("Stat", stats, key=f"slip_stat_{who}")
            d = draws[(pteam, pname_, stat)]
            side = st.radio("Pick", ["Over", "Under"], horizontal=True, key="slip_side")
            dflt = 0.5 if stat == "Anytime TD" else best_line(d)
            line = st.number_input("Line", value=dflt, step=0.5, key=f"slip_pline_{who}_{stat}")
            odds = st.number_input("Odds (American, e.g. -110 or +150)", value=-110.0, step=5.0, key=f"slip_podds_{who}_{stat}")
            stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0, key="slip_stake_p")
            pw, pp, pl_ = p_from_draws(d, float(line), side)
            st.caption(f"Simulated average for {pname_}: {float(np.mean(d)):.1f} {stat}.")
            m = show_math(pw, pp, pl_, odds, stake, bankroll, frac)
            row = {"kind": "prop", "market": "Prop", "pick": side, "line": float(line), "player": pname_, "team": pteam, "stat": stat, "desc": f"{pname_} {side} {line:g} {stat}"}
        if gm is None:
            st.warning(f"{a} @ {h} isn't on this season's schedule, so this bet can't be graded automatically. You can mark its result yourself on My bets.")
        else:
            if pd.notna(gm["home_score"]):
                st.warning(f"{gm['away_team']} @ {gm['home_team']} (week {int(gm['week'])}) has already been played, so a bet added now will grade right away from the real result. That is fine for practice, but it isn't a real prediction.")
            else:
                st.caption(f"Scheduled: {gm['away_team']} @ {gm['home_team']}, week {int(gm['week'])}. The bet grades itself from the real result once the game is final.")
        if st.button("Add to my bets", type="primary", key="slip_add"):
            row.update({"season": int(latest), "week": int(gm["week"]) if gm is not None else None, "home": gm["home_team"] if gm is not None else h, "away": gm["away_team"] if gm is not None else a,
                        "gameday": str(gm["gameday"]) if gm is not None else None, "odds": float(odds), "stake": float(stake), "model_p": float(pw), "ev": float(m["ev"])})
            ok = add_bet(prof, row)
            st.success("Saved to My bets." + ("" if ok else " (Kept in this session only; the file could not be written.)"))

    with tab_sgp:
        prof = safe_profile(st.session_state.get("bet_profile", "me"))
        bankroll = float(st.session_state.get("bet_bank0", 1000.0))
        frac = float(st.session_state.get("bet_frac", 0.25))
        gm = find_any_game(sched, latest, h, a)
        mk = res.get("line")
        if not draws:
            st.info("No player projections for this game, so only game legs are available.")
        n_sim = min([len(res["game"]["hs"])] + [len(v) for v in draws.values()]) if draws else len(res["game"]["hs"])
        HS, AS = np.asarray(res["game"]["hs"])[:n_sim], np.asarray(res["game"]["as"])[:n_sim]
        st.caption(f"Build a same-game parlay for {a} @ {h}. All legs are scored on the same {n_sim:,} simulated games, so legs that move together (a team winning and its QB throwing for a lot) are priced as a group, not as separate coin flips. "
                   "Only the game score and a shared passing-game factor link the players, so treat the correlation as a rough guide, not what a sportsbook's own model would price.")
        legs = st.session_state.setdefault(f"sgp_legs_{h}_{a}", [])
        lk = st.radio("Add a leg", ["Game", "Player prop"], horizontal=True, key="sgp_kind")
        new_leg = None
        if lk == "Game":
            gmk = st.selectbox("Bet type", ["Moneyline", "Spread", "Total points"], key="sgp_market")
            if gmk == "Total points":
                gp = st.radio("Pick", ["Over", "Under"], horizontal=True, key="sgp_pick_t")
                dl = float(mk["total"]) if mk and mk.get("total") is not None else float(np.round((HS + AS).mean() * 2) / 2)
                gl = st.number_input("Total", value=dl, step=0.5, key=f"sgp_line_t_{dl}")
                new_leg = {"type": "game", "market": "Total", "pick": gp, "line": float(gl)}
            else:
                gp = st.radio("Pick", [h, a], horizontal=True, key=f"sgp_pick_{gmk}")
                if gmk == "Spread":
                    base = (-float(mk["spread"]) if gp == h else float(mk["spread"])) if mk and mk.get("spread") is not None else 0.0
                    gl = st.number_input(f"{gp} spread (as the sportsbook shows it, e.g. -3.5 or +7)", value=base, step=0.5, key=f"sgp_line_{gp}_{base}")
                else:
                    gl = 0.0
                new_leg = {"type": "game", "market": gmk, "pick": gp, "line": float(gl)}
        elif draws:
            opts = sorted({(t, nme) for (t, nme, s_) in draws}, key=lambda z: (z[0], z[1]))
            who = st.selectbox("Player", [f"{nme} ({t})" for t, nme in opts], key="sgp_player")
            pn, pt = who.rsplit(" (", 1)[0], who.rsplit(" (", 1)[1][:-1]
            stats = [s_ for (t, nme, s_) in draws if t == pt and nme == pn]
            stat = st.selectbox("Stat", stats, key=f"sgp_stat_{who}")
            dd = draws[(pt, pn, stat)]
            side = st.radio("Pick", ["Over", "Under"], horizontal=True, key="sgp_side")
            pl_line = st.number_input("Line", value=0.5 if stat == "Anytime TD" else best_line(dd), step=0.5, key=f"sgp_pline_{who}_{stat}")
            new_leg = {"type": "prop", "player": pn, "team": pt, "stat": stat, "pick": side, "line": float(pl_line)}
        if new_leg is not None and st.button("Add this leg", key="sgp_add_leg"):
            new_leg["label"] = leg_label(new_leg)
            if new_leg["label"] not in [l["label"] for l in legs]:
                legs.append(new_leg)
            st.rerun()
        if not legs:
            st.info("Add two or more legs to price the parlay.")
        else:
            for i, lg in enumerate(list(legs)):
                c1, c2 = st.columns([5, 1])
                c1.write(f"{i + 1}. {lg['label']}")
                if c2.button("Remove", key=f"sgp_rm_{i}"):
                    legs.pop(i)
                    st.rerun()
            if st.button("Clear all legs", key="sgp_clear"):
                legs.clear()
                st.rerun()
        if len(legs) >= 1:
            sp = sgp_probs(legs, HS, AS, draws, h)
            tbl = pd.DataFrame({"Leg": [l["label"] for l in legs], "Win %": [s[0] * 100 for s in sp["singles"]]})
            st.dataframe(tbl, hide_index=True, column_config={"Win %": st.column_config.NumberColumn(format="%.1f")})
        if len(legs) >= 2:
            lift = sp["pw"] / sp["indep"] if sp["indep"] > 0 else float("nan")
            kc = st.container(key="kpis_sgp")
            c = kc.columns(4)
            c[0].metric("Parlay win chance", f"{sp['pw'] * 100:.2f}%", help=f"All legs hit in {sp['n_win']:,} of {sp['n']:,} simulated games")
            c[1].metric("If legs were independent", f"{sp['indep'] * 100:.2f}%", help="Each leg's win chance multiplied together, the way a naive parlay is priced")
            c[2].metric("Correlation lift", f"{lift:.2f}x" if lift == lift else "n/a", help="Above 1 means the legs tend to happen together; below 1 means they pull against each other")
            c[3].metric("Fair odds", f"{_am_from_dec(1.0 / sp['pw']):+,.0f}" if sp["pw"] > 0 else "n/a", help="The American price at which this parlay would be a break-even bet according to the model")
            if sp["n_win"] < 30:
                st.warning(f"Only {sp['n_win']} of {sp['n']:,} simulated games hit every leg, so this number is shaky. A rare combination needs more simulations than this tab has.")
            naive = _am_from_dec(1.909 ** len(legs))
            odds = st.number_input("Odds the book offers for this parlay (American)", value=float(round(naive / 5.0) * 5.0), step=5.0, key=f"sgp_odds_{len(legs)}_{'|'.join(l['label'] for l in legs)}",
                                   help="The default is what separate -110 legs would pay. Type the real price from your book.")
            stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0, key="sgp_stake")
            m = bet_math(sp["pw"], sp["pp"], sp["pl"], odds, stake, bankroll, frac)
            kc2 = st.container(key="kpis_sgp2")
            d2 = kc2.columns(4)
            d2[0].metric("Break-even chance", f"{m['be'] * 100:.2f}%")
            d2[1].metric("Profit per $100 (model)", f"{m['ev'] * 100:+.1f}")
            d2[2].metric("To win", f"${m['to_win']:,.2f}")
            d2[3].metric("Kelly stake", f"${m['stake_k']:,.0f}" if m["kelly"] > 0 else "$0")
            st.caption("Parlays carry a big built-in house edge, and this model trails the market (see the Backtest). Treat a positive number as a reason to look closer, not a signal to bet. A pushed leg here counts as a push for the whole parlay; a sportsbook would usually drop that leg and pay the rest.")
            if gm is None:
                st.warning(f"{a} @ {h} isn't on this season's schedule, so this parlay can't be graded automatically.")
            elif pd.notna(gm["home_score"]):
                st.warning(f"{gm['away_team']} @ {gm['home_team']} (week {int(gm['week'])}) has already been played, so a parlay added now will grade right away from the real result.")
            if st.button("Add parlay to my bets", type="primary", key="sgp_save"):
                row = {"kind": "sgp", "market": "SGP", "pick": f"{len(legs)} legs", "line": None, "player": None, "team": None, "stat": None, "desc": "SGP: " + " + ".join(l["label"] for l in legs),
                       "legs": json.dumps(legs), "season": int(latest), "week": int(gm["week"]) if gm is not None else None, "home": gm["home_team"] if gm is not None else h,
                       "away": gm["away_team"] if gm is not None else a, "gameday": str(gm["gameday"]) if gm is not None else None, "odds": float(odds), "stake": float(stake),
                       "model_p": float(sp["pw"]), "ev": float(m["ev"])}
                ok = add_bet(prof, row)
                st.success("Parlay saved to My bets." + ("" if ok else " (Kept in this session only; the file could not be written.)"))


with tab_mybets:
    mc1, mc2, mc3 = st.columns(3)
    prof_in = mc1.text_input("Profile name", value="me", key="bet_profile", help="Each name keeps its own list of bets. Friends can use their own names.")
    prof = safe_profile(prof_in)
    bank0 = mc2.number_input("Starting bankroll ($)", min_value=10.0, value=1000.0, step=100.0, key="bet_bank0")
    kfr = mc3.select_slider("Kelly fraction", [0.1, 0.25, 0.5, 1.0], value=0.25, key="bet_frac", help="Full Kelly is very aggressive. Most people use a quarter or less.")
    bets = load_bets(prof)
    bets, n_new = grade_bets(bets, sched, pl)
    if n_new:
        save_bets(bets, prof)
        st.toast(f"Graded {n_new} bet(s) from final results.")
    if bets.empty:
        st.info("No bets yet. Simulate a game, then use Place a bet. Open bets grade themselves when the game is final.")
    else:
        sm = bet_summary(bets)
        kb = st.container(key="kpis_bets")
        k1, k2, k3, k4 = kb.columns(4)
        k1.metric("Settled bets", sm["n"], help=f"{sm['open']} still open")
        k2.metric("Record", f"{sm['won']}-{sm['lost']}" + (f"-{sm['push']}" if sm["push"] else ""))
        k3.metric("Profit", f"${sm['profit']:+,.2f}")
        k4.metric("ROI", f"{sm['roi']:+.1f}%", help="Profit divided by the total amount staked on settled bets")
        if sm["hit"] is not None and sm["avg_p"] is not None:
            st.caption(f"You won {sm['hit']:.0f}% of decided bets; the model's average win chance on them was {sm['avg_p']:.0f}%. Over a small sample those can differ a lot.")
        done = bets[bets["status"].isin(["won", "lost", "push", "void"])].copy()
        if len(done) >= 2:
            done["when"] = done["settled_at"].fillna(done["placed_at"])
            done = done.sort_values("when")
            curve = pd.DataFrame({"Bankroll ($)": bank0 + done["pnl"].fillna(0).cumsum().to_numpy()}, index=range(1, len(done) + 1))
            st.line_chart(curve)
        view = bets.copy()
        view["Status"] = view["status"].map({"open": "⏳ open", "won": "✅ won", "lost": "❌ lost", "push": "➖ push", "void": "➖ void"}).fillna(view["status"])
        view["Game"] = view["away"].astype(str) + " @ " + view["home"].astype(str) + view["week"].apply(lambda w: f" (wk {int(w)})" if pd.notna(w) else "")
        view["Edge"] = view["model_p"] * 100
        view = view.sort_values("placed_at", ascending=False)
        st.dataframe(view[["placed_at", "desc", "Game", "odds", "stake", "Edge", "Status", "pnl"]].rename(columns={"placed_at": "Placed (UTC)", "desc": "Bet", "odds": "Odds", "stake": "Stake", "Edge": "Model win %", "pnl": "Profit"}), hide_index=True,
                     column_config={"Model win %": st.column_config.NumberColumn(format="%.0f"), "Odds": st.column_config.NumberColumn(format="%+.0f"), "Stake": st.column_config.NumberColumn(format="$%.2f"), "Profit": st.column_config.NumberColumn(format="$%+.2f")}, height=min(36 * (len(view) + 1) + 3, 560))
        with st.expander("Manage bets"):
            opts = {f"{r['desc']}  |  {r['Game']}  |  {r['status']}": r["id"] for _, r in view.iterrows()}
            chosen = st.selectbox("Pick a bet", list(opts), key="mb_pick")
            cid = opts[chosen] if opts else None
            cc1, cc2, cc3, cc4 = st.columns(4)
            for col, lab, stt in ((cc1, "Mark won", "won"), (cc2, "Mark lost", "lost"), (cc3, "Mark push", "push")):
                if col.button(lab, key=f"mb_{stt}") and cid:
                    i = bets.index[bets["id"] == cid][0]
                    bets.at[i, "status"] = stt
                    bets.at[i, "pnl"] = _settle(stt, float(bets.at[i, "stake"]), float(bets.at[i, "odds"]))
                    bets.at[i, "settled_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
                    save_bets(bets, prof)
                    st.rerun()
            if cc4.button("Delete", key="mb_del") and cid:
                save_bets(bets[bets["id"] != cid], prof)
                st.rerun()
    bc1, bc2 = st.columns(2)
    bc1.download_button("Download my bets (backup)", bets.to_csv(index=False).encode("utf-8"), file_name=f"nfl_bets_{prof}.csv", mime="text/csv", disabled=bets.empty)
    upb = bc2.file_uploader("Restore from a backup file", type=["csv"], key="bet_up", label_visibility="collapsed")
    if upb is not None and st.session_state.get("bet_up_done") != getattr(upb, "file_id", upb.name):
        try:
            old = pd.read_csv(upb)
            allb = pd.concat([load_bets(prof), _clean_bets(old)], ignore_index=True).drop_duplicates("id", keep="last")
            save_bets(allb, prof)
            st.session_state["bet_up_done"] = getattr(upb, "file_id", upb.name)
            st.rerun()
        except Exception:
            st.error("That file didn't look like a bets backup.")
    st.caption("Bets are stored in a file next to the app, one file per profile name. If the app runs on Streamlit Cloud, those files are shared by everyone with your link and are erased whenever the app restarts, so pick your own profile name and download a backup now and then.")
