#!/usr/bin/env python3
"""Fetch the Kaggle leaderboard, find our team's rank, and update the README
badge + a live-standing line. Runs in GitHub Actions.
Env: KAGGLE_USERNAME, KAGGLE_KEY (secrets), KAGGLE_TEAM (LB display name),
     GITHUB_REPOSITORY / GITHUB_REF_NAME (auto in Actions).
"""
import csv, io, json, os, re, subprocess, sys, zipfile, datetime

COMP = "biohub-cell-tracking-during-development"
TEAM = os.environ.get("KAGGLE_TEAM", "Pathik Patel").strip()
REPO = os.environ.get("GITHUB_REPOSITORY", "OWNER/REPO")
BRANCH = os.environ.get("GITHUB_REF_NAME", "main")

os.makedirs("/tmp/lb", exist_ok=True)
subprocess.run(["kaggle", "competitions", "leaderboard", COMP, "-d", "-p", "/tmp/lb"], check=True)

data = None
for f in os.listdir("/tmp/lb"):
    p = os.path.join("/tmp/lb", f)
    if f.endswith(".zip"):
        z = zipfile.ZipFile(p); data = z.read(z.namelist()[0]).decode("utf-8", "replace")
    elif f.endswith(".csv"):
        data = open(p, encoding="utf-8", errors="replace").read()
if data is None:
    sys.exit("no leaderboard file downloaded")

raw = list(csv.DictReader(io.StringIO(data)))
if not raw:
    sys.exit("empty leaderboard csv")
print("columns:", list(raw[0].keys()), "| rows:", len(raw))

def g(row, *names):
    low = {k.lower(): v for k, v in row.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return ""

def score_of(row):
    try:
        return float(g(row, "score", "publicscore"))
    except ValueError:
        return float("-inf")

rows = sorted(raw, key=lambda r: (-score_of(r), g(r, "submissiondate", "date")))
total = len(rows)
rank = score = None
for i, r in enumerate(rows, 1):
    if g(r, "teamname", "team").strip().lower() == TEAM.lower():
        rank, score = i, score_of(r); break

if rank is None:
    names = sorted({g(r, "teamname", "team") for r in rows})
    print("TEAM '%s' not found. %d teams; sample: %s" % (TEAM, total, names[:8]))
    msg, color = f"team not found", "lightgrey"
else:
    msg = f"#{rank} of {total} · {score:.3f}"
    color = "brightgreen" if rank <= 10 else ("blue" if rank <= 50 else "informational")
    print("found:", msg)

os.makedirs(".github", exist_ok=True)
json.dump({"schemaVersion": 1, "label": "Kaggle rank", "message": msg, "color": color},
          open(".github/kaggle-rank.json", "w"))

badge = (f"![Kaggle rank](https://img.shields.io/endpoint?url="
         f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/.github/kaggle-rank.json)")
updated = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
block = (f"<!-- KAGGLE-RANK:START -->\n{badge}\n\n"
         f"**Live Kaggle standing:** {msg} — "
         f"[leaderboard](https://www.kaggle.com/competitions/{COMP}/leaderboard) "
         f"· updated {updated}\n<!-- KAGGLE-RANK:END -->")

readme = open("README.md", encoding="utf-8").read().replace("\r\n", "\n").replace("\r", "\n")
if "<!-- KAGGLE-RANK:START -->" in readme:
    readme = re.sub(r"<!-- KAGGLE-RANK:START -->.*?<!-- KAGGLE-RANK:END -->", block, readme, flags=re.S)
else:
    readme = block + "\n\n" + readme
open("README.md", "w", encoding="utf-8", newline="\n").write(readme)
print("done")
