#!/usr/bin/env python3
"""Download the competition data via the Kaggle API.

Prerequisites (one-time):
  1. pip install kaggle
  2. Create an API token: kaggle.com -> Account -> "Create New API Token".
     Put the downloaded kaggle.json at ~/.kaggle/kaggle.json
     (Windows: %USERPROFILE%\\.kaggle\\kaggle.json), chmod 600.
  3. Accept the competition rules on the competition page (required to download).

Usage:
    python scripts/download_data.py            # full competition data -> data/
    python scripts/download_data.py --dir data

Data is large; it lands under data/ which is git-ignored.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

COMP = "biohub-cell-tracking-during-development"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMP, "-p", args.dir]
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("kaggle CLI not found. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        sys.exit(f"kaggle download failed ({e}). Did you accept the rules and place kaggle.json?")

    # unzip
    for f in os.listdir(args.dir):
        if f.endswith(".zip"):
            zp = os.path.join(args.dir, f)
            print("Unzipping", zp, flush=True)
            subprocess.run(["python", "-c",
                            f"import zipfile;zipfile.ZipFile(r'{zp}').extractall(r'{args.dir}')"],
                           check=True)
    print("Done. Data under", args.dir)


if __name__ == "__main__":
    main()
