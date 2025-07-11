#!/usr/bin/env bash
set -e

echo "=== Building / launching dockerised FHE benchmark ==="
cd TenSEAL
docker compose up --build   # CTR‑C to stop or run detached with -d
