#!/usr/bin/env bash
# Populate the "Biohub Cell Tracking" GitHub Projects board with starter cards.
# Uses gh's built-in --jq (no local Python needed).
# Prereq: gh authed with the 'project' scope  (gh auth refresh -s project)
set -e
TITLE="Biohub Cell Tracking"

NUM=$(gh project list --owner @me --format json --jq ".projects[] | select(.title==\"$TITLE\") | .number" | head -1)

if [ -z "$NUM" ]; then
  echo "No project titled '$TITLE'. Create it first:"
  echo "    gh project create --owner @me --title \"$TITLE\""
  exit 1
fi
echo "Seeding project #$NUM ..."

add() { gh project item-create "$NUM" --owner @me --title "$1" --body "$2" >/dev/null && echo "  + $1"; }
add "Baseline harness + local CV"  "Rule-based baseline; local metric re-implementation as an LB proxy."
add "Detector improvements"        "Recall/precision of the cell-centre detector."
add "Node-count calibration"       "Keep predicted node count near estimated true count (over-prediction penalty)."
add "Linking / divisions"          "Temporal linking quality; division handling (low priority)."
add "Submission calibration"       "Turn local wins into confirmed LB gains; manage daily submission budget."
add "GitHub + docs"                "Public toolkit repo, README, project board."

echo "Done. Board URL:"
gh project list --owner @me --format json --jq ".projects[] | select(.title==\"$TITLE\") | .url"
