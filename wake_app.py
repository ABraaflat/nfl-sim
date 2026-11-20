"""Visits the Streamlit app in a real (headless) browser so Streamlit Community Cloud doesn't put it to sleep.
If the app is already asleep, it clicks the 'get this app back up' button."""
import os, re, sys
from playwright.sync_api import sync_playwright

url = os.environ.get("APP_URL", "").strip()
if not url:
    sys.exit("Set the APP_URL repository variable to your app's address.")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(url, timeout=120000)
    pg.wait_for_timeout(8000)
    woke = False
    for fr in [pg] + list(pg.frames):
        try:
            btn = fr.get_by_role("button", name=re.compile("get this app back up", re.I))
            if btn.count():
                btn.first.click()
                woke = True
                break
        except Exception:
            pass
    pg.wait_for_timeout(60000 if woke else 20000)
    print("App was asleep, woke it up." if woke else "App was already awake.", "Page title:", pg.title())
    b.close()