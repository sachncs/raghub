#!/usr/bin/env bash
# raghub e2e smoke — exercises the home page + sign-in page
# against the local web server via agent-browser.
#
# Usage: pnpm e2e   (after `pnpm --filter @raghub/web start` &
#                    `pnpm --filter @raghub/api start`)
set -euo pipefail

BASE="${RAGHUB_WEB_BASE:-http://localhost:3001}"
OUT="${RAGHUB_E2E_OUT:-./e2e-report}"
mkdir -p "$OUT"

run() {
  local name="$1"
  shift
  echo "→ $name"
  if ! "$@" >"$OUT/$name.log" 2>&1; then
    echo "  FAIL: $name (log: $OUT/$name.log)"
    return 1
  fi
  echo "  ok"
}

agent-browser open "$BASE/" >/dev/null
run home_snapshot agent-browser snapshot -i
run home_signin_link agent-browser find role link click --name 'Sign in'

# After Sign in click we should be at /sign-in
sleep 1
run signin_url agent-browser get url

agent-browser find label "Email" fill "noreply@example.com" >/dev/null
agent-browser find label "Password" fill "this-is-a-test" >/dev/null
agent-browser find label "Workspace passphrase" fill "test-passphrase" >/dev/null
run signin_snapshot agent-browser snapshot -i

agent-browser close >/dev/null

echo "e2e: ok — report in $OUT"