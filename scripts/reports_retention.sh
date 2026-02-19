#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KEEP=20
APPLY=0
DRY_RUN=0
ARCHIVE_BASE="reports/archive"

usage() {
	cat <<'EOF'
Usage: bash scripts/reports_retention.sh [--keep N] [--apply] [--dry-run]

Archives older generated report artifacts while preserving the newest N per pattern.
Safety rule: tracked git files are never moved.

Options:
	--keep N    Keep the newest N files per pattern (default: 20)
	--apply     Apply changes (without this flag, script runs in dry mode)
	--dry-run   Print planned moves only
	-h, --help  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--keep)
			if [[ $# -lt 2 ]]; then
				echo "Missing value for --keep" >&2
				exit 2
			fi
			KEEP="$2"
			shift 2
			;;
		--apply)
			APPLY=1
			shift
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

if ! [[ "$KEEP" =~ ^[0-9]+$ ]]; then
	echo "--keep must be a non-negative integer" >&2
	exit 2
fi

if [[ $APPLY -eq 0 ]]; then
	DRY_RUN=1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="$ARCHIVE_BASE/$TIMESTAMP"
MOVED=0

move_candidates_for_pattern() {
	local pattern="$1"
	local -a files=()
	while IFS= read -r file; do
		[[ -z "$file" ]] && continue
		files+=("$file")
	done < <(find reports -maxdepth 1 -type f -name "$pattern" -print | sort)

	local count="${#files[@]}"
	if (( count <= KEEP )); then
		return 0
	fi

	local cutoff=$((count - KEEP))
	for ((i = 0; i < cutoff; i++)); do
		local src="${files[$i]}"
		if git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
			echo "[skip tracked] $src"
			continue
		fi

		local dst="$ARCHIVE_DIR/$(basename "$src")"
		if [[ $DRY_RUN -eq 1 ]]; then
			echo "[dry-run] mv $src -> $dst"
		else
			mkdir -p "$ARCHIVE_DIR"
			mv "$src" "$dst"
			echo "[moved] $src -> $dst"
		fi
		MOVED=$((MOVED + 1))
	done
}

move_candidates_for_pattern "ledger_universe_allocation*.json"
move_candidates_for_pattern "portfolio_dashboard_all*.json"
move_candidates_for_pattern "portfolio_dashboard_all*.md"
move_candidates_for_pattern "portfolio_dashboard_all*.csv"
move_candidates_for_pattern "ci_smoke_summary*.json"
move_candidates_for_pattern "local_ci_result*.json"

if [[ $DRY_RUN -eq 1 ]]; then
	echo "Retention dry-run complete. Planned moves: $MOVED"
else
	echo "Retention complete. Files moved: $MOVED"
fi
