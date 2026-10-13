# ==========================================
# Multi-stage Dockerfile for NioXS Gateway
# Embeds NioNano closed-source model weights
# ==========================================

# --- Stage 1: Build binary ---
FROM rust:1.85-slim-bookworm AS builder

WORKDIR /usr/src/nio-xs

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests
COPY Cargo.toml Cargo.lock dependency-pins.json rust-toolchain.toml ./

# Copy all source crates and model assets
COPY crates ./crates
COPY contracts ./contracts
COPY models ./models
COPY .agents/skills ./.agents/skills

# Compile release binary
RUN cargo build --release --bin nioxs

# --- Stage 2: Minimal Runtime ---
FROM debian:bookworm-slim AS runtime

# Install CA certs for HTTPS web scraping and Python for companion plugins
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    bash \
    python3 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled nioxs binary from builder
COPY --from=builder /usr/src/nio-xs/target/release/nioxs /usr/local/bin/nioxs
RUN ln -sf /usr/local/bin/nioxs /usr/local/bin/nio-xs

# Pre-install NioDB for embedded local state persistence
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then DB_ARCH="x64"; \
    elif [ "$ARCH" = "arm64" ]; then DB_ARCH="arm64"; \
    else DB_ARCH="$ARCH"; fi && \
    (curl -fsSL "https://github.com/nio-labs/nio-db/releases/latest/download/niodb-linux-${DB_ARCH}" -o /usr/local/bin/niodb && \
     chmod +x /usr/local/bin/niodb) || true

# Setup non-root working environment
WORKDIR /workspace
RUN mkdir -p /root/.nioxs/plugins /root/.nioxs/data

# OpenAI-compatible API Gateway port
EXPOSE 7460

ENV NIOXS_LISTEN="0.0.0.0:7460" \
    NIOXS_ALLOW_UNAUTHENTICATED="true"

# Healthcheck against unauthenticated health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:7460/health || exit 1

CMD ["nioxs", "serve"]
