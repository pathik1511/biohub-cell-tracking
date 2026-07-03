#!/usr/bin/env bash
# Initialise the repo, verify the private winning solution is excluded, push to
# GitHub, and create a Projects board. Run from Windows (Git Bash / WSL) in the
# project root:  C:\Claude\Kaggle\Biohub_Kaggle_comp
#
# Prereqs: git, and GitHub CLI `gh` (https://cli.github.com) authenticated:
#     gh auth login          # choose GitHub.com, and grant 'project' scope
set -e

REPO_NAME="${1:-biohub-cell-tracking}"
VISIBILITY="${2:-public}"     # public | private

# 1. clean any broken partial repo, init fresh
rm -rf .git
git init -q
git add -A

# 2. SAFETY CHECK — make sure no private (0.872) files got staged
echo "== staged files =="
git diff --cached --name-only
if git diff --cached --name-only | grep -E \
   '^(src/biohub/(unet|detect|pipeline)\.py|scripts/(train_detector|tune_unet|sweep)\.py|kernels/inference_(kernel|hybrid|hybrid_tta|hybrid_ds2|ens)\.py|configs/(hybrid|unet|ens|detect_best).*\.json|ROADMAP\.md|RESULTS\.md|docs/SUBMITTING\.md|.*\.pt)$'; then
  echo "ERROR: private winning-solution files are staged. Fix .gitignore before pushing."
  exit 1
fi
echo "OK: no private winning-solution files staged."

# 3. first commit
git commit -q -m "Public toolkit: baseline + metric + I/O + linking + tooling"

# 4. create the GitHub repo and push
gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push

# 5. create a GitHub Projects board (v2) and seed a few columns/items
PROJ_URL=$(gh project create --owner @me --title "Biohub Cell Tracking" --format json | python -c "import sys,json;print(json.load(sys.stdin)['url'])" 2>/dev/null || true)
echo "Project board: ${PROJ_URL:-'(create manually: gh project create --owner @me --title \"Biohub Cell Tracking\")'}"
# Example items (uncomment and set <NUMBER> from the created project):
# gh project item-create <NUMBER> --owner @me --title "Baseline harness + local CV"
# gh project item-create <NUMBER> --owner @me --title "Detector improvements"
# gh project item-create <NUMBER> --owner @me --title "Linking / divisions"
# gh project item-create <NUMBER> --owner @me --title "Submission calibration"

echo "Done. Repo pushed as '$REPO_NAME' ($VISIBILITY). Private solution kept local."
