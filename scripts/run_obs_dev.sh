#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-5174}"
API_URL="http://127.0.0.1:8787"
UI_URL="http://${UI_HOST}:${UI_PORT}"
UI_OPEN_HOST="${UI_OPEN_HOST:-}"
LAN_IP=""
LOG_DIR="${REPO_ROOT}/.build/obs-dev"
API_LOG="${LOG_DIR}/api.log"
UI_LOG="${LOG_DIR}/ui.log"

mkdir -p "${LOG_DIR}"

if [[ -f "${REPO_ROOT}/.local/gdal-kakadu/env.sh" ]]; then
	source "${REPO_ROOT}/.local/gdal-kakadu/env.sh"
else
	echo "warning: ${REPO_ROOT}/.local/gdal-kakadu/env.sh not found; raster-backed review paths may fail" >&2
fi

if ! command -v uv >/dev/null 2>&1; then
	echo "error: uv is required to run the observability API" >&2
	exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
	echo "error: npm is required to run the SvelteKit UI" >&2
	exit 1
fi

if [[ ! -d "${REPO_ROOT}/web/node_modules" ]]; then
	echo "error: web/node_modules is missing. Run 'cd web && npm install' first." >&2
	exit 1
fi

if command -v hostname >/dev/null 2>&1; then
	LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

if [[ -z "${LAN_IP}" ]] && command -v ip >/dev/null 2>&1; then
	LAN_IP="$(
		ip route get 1.1.1.1 2>/dev/null | awk '
			/src/ {
				for (i = 1; i <= NF; i++) {
					if ($i == "src") {
						print $(i + 1);
						exit;
					}
				}
			}
		'
	)"
fi

if [[ -n "${UI_OPEN_HOST}" ]]; then
	UI_BROWSER_URL="http://${UI_OPEN_HOST}:${UI_PORT}"
elif [[ "${UI_HOST}" == "0.0.0.0" && -n "${LAN_IP}" ]]; then
	UI_BROWSER_URL="http://${LAN_IP}:${UI_PORT}"
else
	UI_BROWSER_URL="${UI_URL}"
fi

API_PID=""
UI_PID=""
API_TAIL_PID=""
UI_TAIL_PID=""

cleanup() {
	local exit_code=$?
	for pid in "${API_TAIL_PID}" "${UI_TAIL_PID}" "${API_PID}" "${UI_PID}"; do
		if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
			kill "${pid}" >/dev/null 2>&1 || true
		fi
	done
	wait >/dev/null 2>&1 || true
	exit "${exit_code}"
}

trap cleanup INT TERM EXIT

: >"${API_LOG}"
: >"${UI_LOG}"

(
	cd "${REPO_ROOT}"
	uv run --extra obs geogrok-obs-api >"${API_LOG}" 2>&1
) &
API_PID=$!

(
	cd "${REPO_ROOT}/web"
	npm run dev -- --host "${UI_HOST}" --port "${UI_PORT}" >"${UI_LOG}" 2>&1
) &
UI_PID=$!

tail -n 0 -F "${API_LOG}" 2>/dev/null | sed -u 's/^/[api] /' &
API_TAIL_PID=$!

tail -n 0 -F "${UI_LOG}" 2>/dev/null | sed -u 's/^/[ui] /' &
UI_TAIL_PID=$!

echo "Observability dev stack starting..."
echo "API: ${API_URL}"
echo "UI:  ${UI_URL}"
if [[ "${UI_BROWSER_URL}" != "${UI_URL}" ]]; then
	echo "Open from another machine: ${UI_BROWSER_URL}"
fi
echo
echo "Open one of:"
echo "  ${UI_BROWSER_URL}/chips"
echo "  ${UI_BROWSER_URL}/pairs"
echo "  ${UI_BROWSER_URL}/review"
echo "  ${UI_BROWSER_URL}/runs"
echo
echo "Press Ctrl-C to stop both services."

wait "${API_PID}" "${UI_PID}"
