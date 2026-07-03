#!/usr/bin/env python3
"""Fetch the Kaggle leaderboard, find our team's rank, and write ONLY the
shields.io endpoint badge data (.github/kaggle-rank.json). The README badge
image reads this JSON live, so the rank updates without ever editing README.
Env: KAGGLE_TEAM (LB display name). Auth via KAGGLE_API_TOKEN / access_token.
"""
import csv, io, json, os, subprocess, sys, zipfile

COMP = "biohub-cell-tracking-during-development"
TEAM = os.environ.get("KAGGLE_TEAM", "Pathik Patel").strip()

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
    msg, color = "team not found", "lightgrey"
else:
    msg = f"#{rank} of {total} · {score:.3f}"
    color = "brightgreen" if rank <= 10 else ("blue" if rank <= 50 else "informational")

os.makedirs(".github", exist_ok=True)
with open(".github/kaggle-rank.json", "w", encoding="utf-8", newline="\n") as fh:
    json.dump({"schemaVersion": 1, "label": "Kaggle rank", "message": msg, "color": color}, fh)
print("updated badge:", msg)
