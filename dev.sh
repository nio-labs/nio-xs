#!/usr/bin/env bash
#
# NioXS development hot-reload runner.
#
# Rebuilds the workspace and restarts the selected process whenever a Rust
# source, manifest, or model file changes. It also resolves its own
# dependencies: it loads .env, finds the `niodb` binary, links a credentials
# file into a stable path, and reuses or starts the local NioDB dev store.
# Debug profile only.
#
# Usage:
#   ./dev.sh [serve|mcp|test|check] [--listen ADDR] [--db-url URL]
#            [--token-file PATH] [--niodb-bin PATH] [--no-db]
#            [--interval SECONDS]
#
#   serve    Run the NioXS gateway (`nioxs serve`) and restart on change (default)
#   mcp      Run the MCP JSON-RPC stdio server and restart on change
#   test     Re-run `cargo test --workspace` on change
#   check    Re-run fmt + clippy on change
#
# Dependency resolution for serve/mcp:
#   niodb binary   $NIOXS_NIODB_BIN, .runtime/bin/niodb,
#                  ../nio-db/target/{release,debug}/niodb, ~/.nioxs/bin/niodb,
#                  ~/.local/bin/niodb, then PATH
#   credentials    .runtime/local/niodb-client.token, .runtime/db.token,
#                  ~/.nioxs/db.token, ~/.config/nioxs/db.token, or the
#                  client_token in .runtime/local/niodb-credentials.json
#   dev store      .runtime/local/niodb (existing data is reused, never reset).
#                  A missing store is provisioned with `niodb init-auth`.
# The found credentials file is linked at .runtime/db.token so the provider
# always receives one stable path.
#
# Environment:
#   NIOXS_DB_URL           NioDB base URL        (default http://127.0.0.1:7432)
#   NIOXS_LISTEN           gateway listen addr   (default 127.0.0.1:7460)
#   NIOXS_DEPLOYMENT_ID    deployment identifier (default local)
#   NIOXS_API_TOKEN_FILE   provider auth token   (required for non-loopback)
#   NIOXS_NIODB_BIN        explicit niodb binary path
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE=serve
INTERVAL=1
MANAGE_DB=1
RUNTIME_DIR="$ROOT/.runtime"
LOCAL_DIR="$RUNTIME_DIR/local"
DB_DATA_DIR="$LOCAL_DIR/niodb"
DEFAULT_TOKEN="$RUNTIME_DIR/db.token"
NIODB_BIN=""

export NIOXS_LISTEN="${NIOXS_LISTEN:-127.0.0.1:7460}"
export NIOXS_DB_URL="${NIOXS_DB_URL:-http://127.0.0.1:7432}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        serve|mcp|test|check) MODE="$1"; shift ;;
        --interval)   INTERVAL="${2:?--interval needs a value}"; shift 2 ;;
        --listen)     export NIOXS_LISTEN="${2:?--listen needs a value}"; shift 2 ;;
        --db-url)     export NIOXS_DB_URL="${2:?--db-url needs a value}"; shift 2 ;;
        --token-file) export NIOXS_DB_TOKEN_FILE="${2:?--token-file needs a value}"; shift 2 ;;
        --niodb-bin)  export NIOXS_NIODB_BIN="${2:?--niodb-bin needs a value}"; shift 2 ;;
        --no-db)      MANAGE_DB=0; shift ;;
        -h|--help)    awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "$0"; exit 0 ;;
        *) echo "dev: unknown argument '$1' (see --help)" >&2; exit 2 ;;
    esac
done

BIN="$ROOT/target/debug/nioxs"
CHILD_PID=""
DB_PID=""

log() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dev]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; exit 2; }

stop_pid() {
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

stop_children() {
    stop_pid "$CHILD_PID"; CHILD_PID=""
    stop_pid "$DB_PID"; DB_PID=""
}

on_exit() { stop_children; }
on_signal() { stop_children; log "stopped"; exit 0; }
# Cleanup on exit must not rewrite the exit status, so it never calls exit.
trap on_exit EXIT
trap on_signal INT TERM

# Parse .env as data: KEY=VALUE only, no shell evaluation, existing env wins.
load_env_file() {
    local file="$ROOT/.env" line key value
    [[ -f "$file" ]] || return 0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
        [[ "$line" != *=* ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        key="${key//[[:space:]]/}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        if [[ -z "${!key:-}" ]]; then export "$key=$value"; fi
    done < "$file"
    log "loaded .env"
}

check_tools() {
    command -v cargo >/dev/null 2>&1 || die "cargo not found; install Rust (see rust-toolchain.toml)"
    command -v rustc >/dev/null 2>&1 || die "rustc not found; cargo is present but incomplete"
    command -v curl >/dev/null 2>&1 || warn "curl not found; NioDB readiness will use a fixed grace period"
}

find_niodb() {
    local candidate
    if [[ -n "${NIOXS_NIODB_BIN:-}" ]]; then
        [[ -x "$NIOXS_NIODB_BIN" ]] || return 1
        printf '%s' "$NIOXS_NIODB_BIN"
        return 0
    fi
    for candidate in \
        "$RUNTIME_DIR/bin/niodb" \
        "$ROOT/../nio-db/target/release/niodb" \
        "$ROOT/../nio-db/target/debug/niodb" \
        "$HOME/.nioxs/bin/niodb" \
        "$HOME/.local/bin/niodb"; do
        if [[ -x "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    if command -v niodb >/dev/null 2>&1; then
        command -v niodb
        return 0
    fi
    return 1
}

db_is_loopback() {
    local url="$1"
    [[ "$url" == *"127.0.0.1"* || "$url" == *"localhost"* || "$url" == *"0.0.0.0"* ]]
}

db_reachable() {
    command -v curl >/dev/null 2>&1 || return 1
    curl -sf --max-time 1 "$NIOXS_DB_URL/health" >/dev/null 2>&1
}

wait_http() {
    local url="$1" label="$2" tries=40
    if ! command -v curl >/dev/null 2>&1; then sleep 2; return 0; fi
    until curl -sf --max-time 1 "$url" >/dev/null 2>&1; do
        tries=$((tries - 1))
        [[ "$tries" -le 0 ]] && die "$label did not become ready at $url"
        kill -0 "${DB_PID:-$$}" 2>/dev/null || die "$label exited before becoming ready"
        sleep 0.25
    done
}

# Point .runtime/db.token at the credentials we found so the provider always
# reads one stable path, and report which real file is in use.
link_token() {
    local source="$1"
    mkdir -p "$RUNTIME_DIR"
    chmod 700 "$RUNTIME_DIR" 2>/dev/null || true
    if [[ "$source" != "$DEFAULT_TOKEN" ]]; then
        ln -sfn "$source" "$DEFAULT_TOKEN" && log "linked $DEFAULT_TOKEN -> $source"
    fi
    export NIOXS_DB_TOKEN_FILE="$DEFAULT_TOKEN"
}

# Reads a JSON credential bundle on stdin and prints its client_token.
extract_client_token() {
    grep -oE '"client_token"[[:space:]]*:[[:space:]]*"[^"]+"' 2>/dev/null \
        | head -1 | sed -E 's/.*:[[:space:]]*"([^"]+)"$/\1/'
}

resolve_token() {
    if [[ -n "${NIOXS_DB_TOKEN_FILE:-}" ]]; then
        [[ -r "$NIOXS_DB_TOKEN_FILE" ]] || die "NIOXS_DB_TOKEN_FILE is not readable: $NIOXS_DB_TOKEN_FILE"
        log "using token file $NIOXS_DB_TOKEN_FILE"
        return 0
    fi

    local candidate
    for candidate in \
        "$LOCAL_DIR/niodb-client.token" \
        "$DEFAULT_TOKEN" \
        "$HOME/.nioxs/db.token" \
        "$HOME/.config/nioxs/db.token"; do
        [[ -r "$candidate" ]] || continue
        link_token "$candidate"
        log "using token file $candidate"
        return 0
    done

    # Fall back to the credential bundle written by `niodb init-auth`.
    local credentials="$LOCAL_DIR/niodb-credentials.json"
    if [[ -r "$credentials" ]]; then
        local token
        token="$(extract_client_token < "$credentials")"
        if [[ -n "$token" ]]; then
            mkdir -p "$RUNTIME_DIR"
            ( umask 077; printf '%s' "$token" > "$DEFAULT_TOKEN" )
            chmod 600 "$DEFAULT_TOKEN"
            export NIOXS_DB_TOKEN_FILE="$DEFAULT_TOKEN"
            log "derived token file from $credentials"
            return 0
        fi
    fi
    return 1
}

provision_token() {
    [[ -n "$NIODB_BIN" ]] || return 1
    mkdir -p "$LOCAL_DIR"
    chmod 700 "$RUNTIME_DIR" "$LOCAL_DIR" 2>/dev/null || true
    local creds token
    log "provisioning NioDB credentials in $LOCAL_DIR"
    creds="$("$NIODB_BIN" init-auth --dir "$LOCAL_DIR" --name nioxs-dev --workspace nioxs-dev 2>/dev/null)" \
        || die "niodb init-auth failed"
    token="$(printf '%s' "$creds" | extract_client_token)"
    [[ -n "$token" ]] || die "niodb init-auth did not return a client_token"
    ( umask 077; printf '%s' "$token" > "$DEFAULT_TOKEN" )
    chmod 600 "$DEFAULT_TOKEN"
    export NIOXS_DB_TOKEN_FILE="$DEFAULT_TOKEN"
    log "wrote dev credentials to $DEFAULT_TOKEN (mode 0600)"
}

start_db() {
    local port="${NIOXS_DB_URL##*:}"
    port="${port%%/*}"
    mkdir -p "$DB_DATA_DIR"
    log "starting NioDB ($NIODB_BIN) on 127.0.0.1:$port (data: $DB_DATA_DIR)"
    "$NIODB_BIN" serve --dir "$DB_DATA_DIR" --listen "127.0.0.1:$port" --no-pg --no-demo \
        --nio-bin nioxs-assistance-disabled --node-bin nioxs-sql-disabled \
        >"$LOCAL_DIR/niodb.log" 2>&1 &
    DB_PID=$!
    wait_http "$NIOXS_DB_URL/health" "NioDB"
    log "NioDB ready at $NIOXS_DB_URL"
}

ensure_dependencies() {
    # Validate an explicit override before discovery so a typo fails loudly
    # instead of being swallowed by the command substitution.
    if [[ -n "${NIOXS_NIODB_BIN:-}" && ! -x "$NIOXS_NIODB_BIN" ]]; then
        die "NIOXS_NIODB_BIN is not executable: $NIOXS_NIODB_BIN"
    fi
    NIODB_BIN="$(find_niodb || true)"
    if [[ -n "$NIODB_BIN" ]]; then
        log "niodb binary: $NIODB_BIN"
    else
        warn "no niodb binary found (checked .runtime/bin, ../nio-db/target/*, ~/.nioxs/bin, ~/.local/bin, PATH)"
    fi

    if ! resolve_token; then
        provision_token || true
    fi

    if db_reachable; then
        log "NioDB already reachable at $NIOXS_DB_URL"
    elif [[ "$MANAGE_DB" == 0 ]]; then
        warn "NioDB is not reachable at $NIOXS_DB_URL (--no-db given)"
    elif ! db_is_loopback "$NIOXS_DB_URL"; then
        warn "NioDB is not reachable at the remote URL $NIOXS_DB_URL; not starting a local one"
    elif [[ -n "$NIODB_BIN" ]]; then
        start_db
    else
        die "NioDB is not reachable at $NIOXS_DB_URL and no niodb binary was found. Pass --niodb-bin PATH, install niodb, or start it yourself."
    fi

    if [[ -z "${NIOXS_DB_TOKEN_FILE:-}" ]]; then
        die "no credentials found; set NIOXS_DB_TOKEN_FILE, or put a token at $LOCAL_DIR/niodb-client.token"
    fi
}

# Fingerprint the tree so we can detect edits without extra tooling.
fingerprint() {
    { find crates models -type f \( -name '*.rs' -o -name '*.toml' -o -name '*.json' \) -print0 \
        | xargs -0 stat -c '%Y:%n' 2>/dev/null
      stat -c '%Y:%n' Cargo.toml Cargo.lock 2>/dev/null
    } | sort | md5sum
}

wait_for_change() {
    if command -v inotifywait >/dev/null 2>&1; then
        inotifywait -qq -r -e modify,create,delete,move crates models Cargo.toml Cargo.lock
        return
    fi
    local before after
    before="$(fingerprint)"
    while :; do
        sleep "$INTERVAL"
        after="$(fingerprint)"
        [[ "$before" != "$after" ]] && return
        before="$after"
    done
}

run_once() {
    case "$MODE" in
        serve)
            "$BIN" serve &
            CHILD_PID=$!
            ;;
        mcp)
            "$BIN" mcp &
            CHILD_PID=$!
            ;;
        test)
            cargo test --workspace --locked
            ;;
        check)
            cargo fmt --all --check
            cargo clippy --workspace --all-targets --locked -- -D warnings
            ;;
    esac
}

load_env_file
check_tools
if [[ "$MODE" == serve || "$MODE" == mcp ]]; then
    ensure_dependencies
fi

log "mode=$MODE${NIOXS_LISTEN:+ listen=$NIOXS_LISTEN} — watching for changes (Ctrl-C to stop)"
while :; do
    if cargo build --workspace --locked; then
        log "starting $MODE"
        if [[ "$MODE" == serve || "$MODE" == mcp ]]; then
            run_once
            wait "$CHILD_PID" || true
            CHILD_PID=""
        else
            run_once
        fi
    else
        log "build failed; fix the error and save to retry"
    fi
    log "waiting for changes"
    wait_for_change
    stop_pid "$CHILD_PID"; CHILD_PID=""
done
