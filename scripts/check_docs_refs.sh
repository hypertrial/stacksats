#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DOC_FILES=(
  README.md
  CONTRIBUTING.md
  CHANGELOG.md
  SECURITY.md
)

while IFS= read -r file; do
  DOC_FILES+=("${file}")
done < <(rg --files docs -g '*.md' | sort)

TOKENS=()
while IFS= read -r line; do
  TOKENS+=("${line}")
done < <(
  rg --no-filename -o '`[^`]+`' "${DOC_FILES[@]}" \
    | sed -e 's/^`//' -e 's/`$//' \
    | sort -u
)

MISSING=()

for token in "${TOKENS[@]}"; do
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
    MISSING+=("${candidate} -> ${path_ref}")
  fi
done

if ((${#MISSING[@]} > 0)); then
  echo "Missing doc references:"
  printf ' - %s\n' "${MISSING[@]}"
  exit 1
fi

echo "Docs reference check passed."
