## Summary

Describe what changed and why.

## Validation

- [ ] `venv/bin/python -m ruff check .`
- [ ] `venv/bin/python -m pytest -q`
- [ ] `venv/bin/python -m build`
- [ ] `venv/bin/python -m twine check dist/*`
- [ ] `bash scripts/release_check.sh` (required for release/tooling changes)

## Documentation

- [ ] Updated `README.md` (if user-facing behavior changed)
- [ ] Updated `docs/` pages (if workflows changed)
- [ ] Updated `CHANGELOG.md` (`Unreleased` section)

## Risk and rollout

- [ ] Backward compatible
- [ ] No secrets added
- [ ] Release notes impact understood
