#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

stage=${1:-audit}
config=${2:-}
python=.transformer_env/bin/python
export MPLCONFIGDIR=/tmp/top-reconstruction-mpl
export XDG_CACHE_HOME=/tmp/top-reconstruction-cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

git diff --check
"$python" - <<'PY'
import ast
from pathlib import Path
for root in (Path("src"), Path("tests"), Path("scripts")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
"$python" -m unittest discover -s tests -v
while IFS= read -r script; do bash -n "$script"; done < <(
  find . -maxdepth 2 -type f \( -name '*.sh' -o -name '*.sbatch' \) -print
)

if [[ "$stage" == training ]]; then
  [[ -n "$config" ]] || { echo "training preflight requires a config" >&2; exit 2; }
  "$python" scripts/plan_gate.py training --config "$config"
  "$python" scripts/validate_config.py "$config"
else
  "$python" scripts/plan_gate.py audit
fi

tracked_changes=$(git status --porcelain --untracked-files=no)
untracked_code=$(git ls-files --others --exclude-standard -- \
  src tests scripts config submit.sh submit_true_jets.sh)
[[ -z "$tracked_changes" && -z "$untracked_code" ]] || {
  echo "BLOCKED: jobs require audited code/config to be committed in a clean worktree" >&2
  exit 2
}
receipt_args=(issue "$stage")
[[ -z "$config" ]] || receipt_args+=(--config "$config")
"$python" scripts/preflight_receipt.py "${receipt_args[@]}"

echo "PREFLIGHT OK: $stage"
