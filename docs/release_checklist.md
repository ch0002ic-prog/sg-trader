# Release Checklist

## Scope

Use this checklist before merging or promoting changes to production-adjacent usage.

## Pre-Merge (PR)

- Confirm `smoke` and `unit-gates` checks pass in GitHub Actions.
- Verify no unexpected file churn in `reports/` or generated artifacts.
- Confirm strategy docs remain aligned:
  - `README.md`
  - `docs/runbook.md`
  - `docs/new-strategy.md`
- Ensure command examples in docs match actual CLI flags in `main.py`.
- Validate any new scripts have executable bits if intended (`chmod +x`).

## Pre-Push (Local)

- Run strict local CI:
  - `bash scripts/local_ci.sh --strict`
- Run machine-readable local CI summary:
  - `bash scripts/local_ci.sh --strict --json | python scripts/local_ci_parse.py`
- (Optional) Generate combined artifact summary for triage:
  - `python scripts/print_ci_artifact_summary.py --smoke-path reports/ci_smoke_summary.json --local-ci-path reports/local_ci_result.json`

## Release Readiness

- Confirm branch protection requires:
  - `smoke`
  - `unit-gates`
- Confirm risk-control defaults are suitable for environment (`.env` / runtime configuration).
- Confirm no secrets are committed.
- Confirm rollback path is clear (revert commit or rollback to previous known-good tag/commit).

## Post-Release Verification

- Run `python main.py --healthcheck`.
- Run a deterministic replay smoke path:
  - `python main.py --execution-ci-smoke --execution-broker paper --paper-symbol SPX_PUT --paper-side SELL --paper-qty 1 --paper-reference-price 1.25 --paper-seed 1`
- Confirm expected artifacts are produced under `reports/`.
- Check CI summary line in Actions logs for quick PASS/FAIL triage.
