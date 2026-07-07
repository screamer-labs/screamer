## Summary

<!-- What does this PR change, and why? -->

## Checklist

- [ ] Tests pass locally (`poetry run pytest -q`), suite is green with no new skips
- [ ] Batch and streaming produce identical results (causality preserved, no look-ahead)
- [ ] New/changed public functions have a `docs/functions_*/<Name>.md` page with valid
      frontmatter, at least one `topics:` slug, and a `nan_policy`
- [ ] Docs and `CHANGELOG.md` updated where relevant
- [ ] For new functions: a reference baseline added under `devtools/baselines/` (if feasible)

## Notes for reviewers

<!-- Anything that needs extra attention, trade-offs, or follow-ups. -->
