"""Daily robot for the NFL app. No browser needed.

It runs the real app headlessly (Streamlit's AppTest), so every number matches what you'd get by pressing the buttons yourself:
  1. Simulates the next unplayed week and saves the picks to nfl_predictions.csv (these are also the Bot's picks in Beat the Bot).
  2. Takes a daily snapshot of everyone's playoff and Super Bowl odds in nfl_odds_history.csv (feeds the Race page).
  3. Prints the model's record on its saved picks (also shown on the GitHub run summary).

Usage:   python save_predictions.py [app_file.py] [--weeks N] [--sims 10000] [--no-odds]
"""
import argparse, os, sys, time
from datetime import datetime, timezone

APP_CANDIDATES = ["nfl_dashboard_v8.py", "nfl_dashboard_v7.py", "nfl_dashboard.py", "streamlit_app.py", "app.py"]
HIST_COLS = ["date", "week", "team", "proj_wins", "playoffs", "division", "seed1", "super_bowl"]


def record_summary():
    """The model's straight-up record on saved (non-late) picks that have been played."""
    try:
        import pandas as pd
        import nflreadpy as nfl
        p = pd.read_csv("nfl_predictions.csv")
        p = p[~p["late"].astype(bool)]
        s = nfl.load_schedules([int(x) for x in sorted(p["season"].unique())]).to_pandas()
        s = s[["season", "week", "home_team", "away_team", "home_score", "away_score"]].rename(columns={"home_team": "home", "away_team": "away"})
        m = p.merge(s, on=["season", "week", "home", "away"]).dropna(subset=["home_score"])
        m = m[m["home_score"] != m["away_score"]]
        if m.empty:
            return "No saved picks have been played yet."
        w = int(((m["p_home"] > 0.5) == (m["home_score"] > m["away_score"])).sum())
        return f"Model record on saved picks: {w}-{len(m) - w} ({100 * w / len(m):.0f}%) over {len(m)} games."
    except Exception as e:
        return f"(Could not compute the record: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("app", nargs="?")
    ap.add_argument("--weeks", type=int, default=1)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--no-odds", action="store_true")
    ns = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    app = ns.app or next((os.path.join(here, c) for c in APP_CANDIDATES if os.path.exists(os.path.join(here, c))), None)
    if not app or not os.path.exists(app):
        sys.exit("Could not find the app file. Pass its name: python save_predictions.py nfl_dashboard_v8.py")
    os.chdir(os.path.dirname(os.path.abspath(app)))
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.abspath(app), default_timeout=1500)
    t0 = time.time()
    at.run()
    if at.exception:
        sys.exit("The app failed to load: " + "; ".join(str(e.value)[:300] for e in at.exception))
    print(f"app loaded in {time.time() - t0:.0f}s")
    lines = []

    sel = [s for s in at.selectbox if s.key == "slate_week"]
    saved, first_week = 0, None
    if not sel:
        lines.append("No unplayed games left, so no picks to save.")
    else:
        weeks = list(sel[0].options)[:max(1, ns.weeks)]
        first_week = weeks[0]
        for wk in weeks:
            at.selectbox(key="slate_week").set_value(wk)
            try:
                at.select_slider(key="slate_sims").set_value(ns.sims)
            except Exception:
                pass
            at.run()
            [b for b in at.button if b.key == "slate_go"][0].click()
            at.run()
            if at.exception:
                sys.exit(f"Week {wk} failed: " + "; ".join(str(e.value)[:300] for e in at.exception))
            sl = at.session_state["slate"]
            n = len(sl["df"])
            if n == 0:
                continue
            [b for b in at.button if b.key == "slate_save"][0].click()
            at.run()
            if at.exception:
                sys.exit("Saving failed: " + "; ".join(str(e.value)[:300] for e in at.exception))
            lines.append(f"Week {wk}: saved {n} picks ({ns.sims:,} sims each; injury report {sl['df']['Injury report'].iloc[0]}).")
            saved += n

    if not ns.no_odds:
        try:
            import pandas as pd
            at.select_slider(key="ss_sims").set_value(5000)
            [b for b in at.button if b.key == "ss_go"][0].click()
            at.run()
            if at.exception:
                raise RuntimeError("; ".join(str(e.value)[:200] for e in at.exception))
            d = at.session_state["ss_df"]
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            snap = pd.DataFrame({"date": today, "week": first_week if first_week is not None else 0, "team": d["Team"], "proj_wins": d["Proj wins"], "playoffs": d["Playoffs %"],
                                 "division": d["Win division %"], "seed1": d["#1 seed %"], "super_bowl": d["Win Super Bowl %"]}).round(2)
            old = pd.read_csv("nfl_odds_history.csv") if os.path.exists("nfl_odds_history.csv") else pd.DataFrame(columns=HIST_COLS)
            allh = pd.concat([old, snap[HIST_COLS]], ignore_index=True).drop_duplicates(["date", "team"], keep="last")
            allh.to_csv("nfl_odds_history.csv", index=False)
            top = snap.sort_values("super_bowl", ascending=False).head(3)
            lines.append("Odds snapshot saved. Super Bowl favorites: " + ", ".join(f"{r.team} {r.super_bowl:.1f}%" for r in top.itertuples()))
        except Exception as e:
            lines.append(f"(Odds snapshot skipped: {e})")

    lines.append(record_summary())
    text = "\n".join(lines)
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write("### NFL daily robot\n\n" + "\n\n".join(lines) + "\n")
    print(f"Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()