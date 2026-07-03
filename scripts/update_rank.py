#!/usr/bin/env python3
"""Fetch the Kaggle leaderboard, find our team's rank, and write:
  - .github/kaggle-rank.json  (shields.io endpoint badge data)
  - README.md                 (between the KAGGLE-RANK markers)
Runs in GitHub Actions (Ubuntu has Python; not meant for the Windows box).
Env: KAGGLE_USERNAME, KAGGLE_KEY (secrets), KAGGLE_TEAM (your LB display name),
     GITHUB_REPOSITORY (auto-set in Actions).
"""
import csv, io, json, os, subprocess, sys, zipfile, datetime

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
        z = zipfile.ZipFile(p); data = z.read(z.namelist()[0]).decode()
    elif f.endswith(".csv"):
        data = open(p, encoding="utf-8").read()
if data is None:
    sys.exit("no leaderboard file downloaded")

rows = list(csv.DictReader(io.StringIO(data)))
# Kaggle ranks ties by earliest submission reaching the score:
rows.sort(key=lambda r: (-float(r["score"]), r.get("submissionDate", "")))
total = len(rows)
rank = score = None
for i, r in enumerate(rows, 1):
    if r.get("teamName", "").strip().lower() == TEAM.lower():
        rank, score = i, r["score"]; break

if rank is None:
    msg, color = f"team '{TEAM}' not found", "lightgrey"
    label_msg = msg
else:
    label_msg = f"#{rank} of {total} · {float(score):.3f}"
    color = "brightgreen" if rank <= 10 else ("blue" if rank <= 50 else "informational")

os.makedirs(".github", exist_ok=True)
json.dump({"schemaVersion": 1, "label": "Kaggle rank", "message": label_msg, "color": color},
          open(".github/kaggle-rank.json", "w"))

badge = (f"![Kaggle rank](https://img.shields.io/endpoint?url="
         f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/.github/kaggle-rank.json)")
updated = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
block = (f"<!-- KAGGLE-RANK:START -->\n{badge}\n\n"
         f"**Live Kaggle standing:** {label_msg} — "
         f"[leaderboard](https://www.kaggle.com/competitions/{COMP}/leaderboard) "
         f"· updated {updated}\n<!-- KAGGLE-RANK:END -->")

readme = open("README.md", encoding="utf-8").read()
import re
if "<!-- KAGGLE-RANK:START -->" in readme:
    readme = re.sub(r"<!-- KAGGLE-RANK:START -->.*?<!-- KAGGLE-RANK:END -->", block, readme, flags=re.S)
else:
    readme = block + "\n\n" + readme
open("README.md", "w", encoding="utf-8").write(readme)
print("updated:", label_msg)
