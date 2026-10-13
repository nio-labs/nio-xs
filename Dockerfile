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
    && rm -rf /var/lib/apt/lists/*

# Copy compiled nioxs binary from builder
COPY --from=builder /usr/src/nio-xs/target/release/nioxs /usr/local/bin/nioxs

# Setup non-root working environment
WORKDIR /workspace
RUN mkdir -p /root/.nioxs/plugins

# OpenAI-compatible API Gateway port
EXPOSE 7460

ENV NIOXS_LISTEN="0.0.0.0:7460"

# Healthcheck against OpenAI-compatible models endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:7460/v1/models || exit 1

ENTRYPOINT ["nioxs"]
