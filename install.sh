#!/usr/bin/env bash
# ==============================================================================
# NioXS Universal Installer
# Supports: Linux (x86_64, aarch64), macOS (Apple Silicon / Intel), Termux, Windows
# ==============================================================================

set -e

REPO="nio-labs/nio-xs"
VERSION="${NIOXS_VERSION:-latest}"

BOLD="$(tput bold 2>/dev/null || echo '')"
GREEN="$(tput setaf 2 2>/dev/null || echo '')"
YELLOW="$(tput setaf 3 2>/dev/null || echo '')"
RED="$(tput setaf 1 2>/dev/null || echo '')"
RESET="$(tput sgr0 2>/dev/null || echo '')"

info() {
    echo -e "${GREEN}==>${RESET} ${BOLD}$1${RESET}"
}

warn() {
    echo -e "${YELLOW}Warning:${RESET} $1"
}

error() {
    echo -e "${RED}Error:${RESET} $1" >&2
    exit 1
}

# 1. Detect Environment & OS
IS_TERMUX=false
IS_WINDOWS=false
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

if [ -n "$TERMUX_VERSION" ] || [ -d "/data/data/com.termux" ]; then
    IS_TERMUX=true
    OS="termux"
elif [[ "$OS" == *"mingw"* ]] || [[ "$OS" == *"msys"* ]] || [[ "$OS" == *"cygwin"* ]]; then
    IS_WINDOWS=true
    OS="windows"
fi

# Normalize architecture
case "$ARCH" in
    x86_64|amd64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        error "Unsupported architecture: $ARCH"
        ;;
esac

info "Detected platform: OS=${BOLD}$OS${RESET}, Arch=${BOLD}$ARCH${RESET}"

# 2. Determine target installation directory
BIN_NAME="nioxs"
if [ "$IS_WINDOWS" = true ]; then
    BIN_NAME="nioxs.exe"
fi

if [ "$IS_TERMUX" = true ]; then
    INSTALL_DIR="${PREFIX:-/data/data/com.termux/files/usr}/bin"
elif [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
elif [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin" 2>/dev/null; then
    INSTALL_DIR="$HOME/.local/bin"
else
    INSTALL_DIR="$HOME/.nioxs/bin"
    mkdir -p "$INSTALL_DIR"
fi

TARGET_FILE="$INSTALL_DIR/$BIN_NAME"

# 3. Check for local build first (if running from source tree)
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SOURCE_DIR/target/release/nioxs" ] || [ -f "$SOURCE_DIR/target/release/nioxs.exe" ]; then
    info "Found local release binary in workspace. Installing directly..."
    if [ -f "$SOURCE_DIR/target/release/nioxs.exe" ]; then
        cp "$SOURCE_DIR/target/release/nioxs.exe" "$TARGET_FILE"
    else
        cp "$SOURCE_DIR/target/release/nioxs" "$TARGET_FILE"
    fi
    chmod +x "$TARGET_FILE"
else
    # 4. Fetch release asset from GitHub
    info "Downloading $BIN_NAME for $OS-$ARCH..."

    if [ "$VERSION" = "latest" ]; then
        DOWNLOAD_URL="https://github.com/$REPO/releases/latest/download/nioxs-$OS-$ARCH"
    else
        DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/nioxs-$OS-$ARCH"
    fi

    if [ "$IS_WINDOWS" = true ]; then
        DOWNLOAD_URL="${DOWNLOAD_URL}.exe"
    fi

    TEMP_FILE="$(mktemp 2>/dev/null || echo "/tmp/nioxs_download")"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$DOWNLOAD_URL" -o "$TEMP_FILE" || {
            warn "Remote asset not found at $DOWNLOAD_URL."
            warn "Attempting to build from source via cargo..."
            if command -v cargo >/dev/null 2>&1; then
                cargo install --path "$SOURCE_DIR/crates/provider" --bin nioxs --root "$(dirname "$INSTALL_DIR")"
            else
                error "Cargo not found. Please install Rust or download a pre-built release binary."
            fi
        }
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$TEMP_FILE" "$DOWNLOAD_URL"
    else
        error "Neither curl nor wget found. Please install one of them."
    fi

    if [ -f "$TEMP_FILE" ]; then
        mv "$TEMP_FILE" "$TARGET_FILE"
        chmod +x "$TARGET_FILE"
    fi
fi

# 5. Bundle and install NioDB for state persistence
NIODB_NAME="niodb"
if [ "$IS_WINDOWS" = true ]; then
    NIODB_NAME="niodb.exe"
fi
NIODB_TARGET="$INSTALL_DIR/$NIODB_NAME"

if ! command -v niodb >/dev/null 2>&1 && [ ! -f "$NIODB_TARGET" ]; then
    info "Bundling NioDB state store ($OS-$ARCH)..."
    DB_OS="$OS"
    if [ "$OS" = "windows" ]; then
        DB_OS="win32"
    fi
    DB_ARCH="$ARCH"
    if [ "$ARCH" = "amd64" ]; then
        DB_ARCH="x64"
    fi

    DB_ASSET="niodb-$DB_OS-$DB_ARCH"
    if [ "$IS_WINDOWS" = true ]; then
        DB_ASSET="${DB_ASSET}.exe"
    fi
    DB_URL="https://github.com/nio-labs/nio-db/releases/latest/download/$DB_ASSET"

    TEMP_DB="$(mktemp 2>/dev/null || echo "/tmp/niodb_download")"
    if curl -fsSL "$DB_URL" -o "$TEMP_DB" 2>/dev/null || wget -qO "$TEMP_DB" "$DB_URL" 2>/dev/null; then
        mv "$TEMP_DB" "$NIODB_TARGET"
        chmod +x "$NIODB_TARGET"
        info "NioDB bundled successfully at ${BOLD}$NIODB_TARGET${RESET}"
    else
        warn "Could not bundle prebuilt NioDB directly; nioxs will manage state or prompt on start."
    fi
fi

# 6. Check PATH availability
info "NioXS binary installed successfully to: ${BOLD}$TARGET_FILE${RESET}"

if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    warn "The directory '$INSTALL_DIR' is not in your PATH."
    echo ""
    echo "Add it by adding this line to your ~/.bashrc or ~/.zshrc:"
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    echo ""
fi

# 6. Verify Installation
if command -v nioxs >/dev/null 2>&1 || [ -x "$TARGET_FILE" ]; then
    info "Verifying installation..."
    "$TARGET_FILE" --help | head -n 5
    echo ""
    echo -e "${GREEN}${BOLD}✔ NioXS is ready!${RESET}"
    echo "Start the gateway with: ${BOLD}nioxs${RESET}"
fi
