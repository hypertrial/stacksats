#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

doc_files_tmp="$(mktemp)"
tokens_tmp="$(mktemp)"
missing_tmp="$(mktemp)"
trap 'rm -f "${doc_files_tmp}" "${tokens_tmp}" "${missing_tmp}"' EXIT

{
  printf '%s\n' README.md CONTRIBUTING.md CHANGELOG.md SECURITY.md
  if [[ -d docs ]]; then
    if command -v rg >/dev/null 2>&1; then
      rg --files docs -g '*.md' | sort
    else
      python - <<'PY'
from pathlib import Path
for path in sorted(Path("docs").rglob("*.md")):
    print(path.as_posix())
PY
    fi
  fi
} > "${doc_files_tmp}"

if command -v rg >/dev/null 2>&1; then
  rg --no-filename -o '`[^`]+`' $(<"${doc_files_tmp}") \
    | sed -e 's/^`//' -e 's/`$//' \
    | sort -u > "${tokens_tmp}"
else
  python - "${doc_files_tmp}" <<'PY' > "${tokens_tmp}"
import re
import sys
from pathlib import Path

token_re = re.compile(r"`([^`]+)`")
tokens = set()

with open(sys.argv[1], "r", encoding="utf-8") as f:
    files = [line.strip() for line in f if line.strip()]

for rel in files:
    path = Path(rel)
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    for token in token_re.findall(text):
        tokens.add(token)

for token in sorted(tokens):
    print(token)
PY
fi

missing_count=0
while IFS= read -r token; do
  [[ -z "${token}" ]] && continue
  candidate="${token}"

  if [[ ! "${candidate}" =~ ^(\.\./)?(docs|tests|stacksats|scripts|examples|\.github|README\.md|CONTRIBUTING\.md|CHANGELOG\.md|SECURITY\.md|CODE_OF_CONDUCT\.md|LICENSE)(/.*)?$ ]]; then
    continue
  fi

  if [[ "${candidate}" == *"<"* || "${candidate}" == *">"* || "${candidate}" == *"*"* || "${candidate}" == *"{"* || "${candidate}" == *"}"* || "${candidate}" == *"$"* ]]; then
    continue
  fi

  path_ref="${candidate}"
  if [[ "${path_ref}" == *.py:* ]]; then
    path_ref="${path_ref%%:*}"
  fi
  if [[ "${path_ref}" == ../* ]]; then
    path_ref="${path_ref#../}"
  fi

  if [[ ! -e "${path_ref}" ]]; then
    printf ' - %s -> %s\n' "${candidate}" "${path_ref}" >> "${missing_tmp}"
    missing_count=$((missing_count + 1))
  fi
done < "${tokens_tmp}"

if ((missing_count > 0)); then
  echo "Missing doc references:"
  cat "${missing_tmp}"
  exit 1
fi

echo "Docs reference check passed."
