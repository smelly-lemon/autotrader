#!/bin/bash
# Periodic sync of parquet data from Jetson (m4) to this Mac.
# Run via cron: */30 * * * * /path/to/jetson_sync.sh
#
# Usage:
#   ./scripts/jetson_sync.sh                    # one-shot sync
#   ./scripts/jetson_sync.sh --continuous 1800  # sync every 30 min

set -euo pipefail

REMOTE_HOST="m4"
REMOTE_DIR="/Users/tim/Development/auto-trader/data/raw"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"

mkdir -p "$LOCAL_DIR"

sync_once() {
    echo "$(date): Syncing from $REMOTE_HOST:$REMOTE_DIR -> $LOCAL_DIR"
    rsync -avz --progress \
        --include="*.parquet" \
        --include=".parts_*/" \
        --include=".parts_*/part_*.parquet" \
        --exclude="*" \
        "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
    echo "$(date): Sync complete"
}

if [[ "${1:-}" == "--continuous" ]]; then
    INTERVAL="${2:-1800}"
    echo "Continuous sync mode: every ${INTERVAL}s"
    while true; do
        sync_once
        sleep "$INTERVAL"
    done
else
    sync_once
fi
