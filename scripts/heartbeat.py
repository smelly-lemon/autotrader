#!/usr/bin/env python3
"""Daily data-collection heartbeat.

Checks that the tick collector is actually producing fresh parquet data.
Logs a status line and raises a macOS notification if the data is stale.

Run via launchd (see deploy/launchd/) or manually:
    python scripts/heartbeat.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STALE_AFTER_HOURS = 2.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def newest_parquet_age_hours() -> tuple[float, Path | None]:
    newest_mtime = 0.0
    newest_path: Path | None = None
    for p in DATA_DIR.glob("*.parquet"):
        m = p.stat().st_mtime
        if m > newest_mtime:
            newest_mtime, newest_path = m, p
    if newest_path is None:
        return float("inf"), None
    return (time.time() - newest_mtime) / 3600, newest_path


def notify(title: str, message: str):
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}"'],
            timeout=10, check=False,
        )
    except Exception:
        pass


def main() -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    age_h, newest = newest_parquet_age_hours()

    if newest is None:
        print(f"{now} CRITICAL no parquet files found in {DATA_DIR}")
        notify("auto-trader: NO DATA", f"No parquet files in {DATA_DIR}")
        return 1

    if age_h > STALE_AFTER_HOURS:
        print(f"{now} STALE newest file {newest.name} is {age_h:.1f}h old "
              f"(threshold {STALE_AFTER_HOURS}h)")
        notify("auto-trader: data STALE",
               f"{newest.name} last written {age_h:.1f}h ago — collector may be down")
        return 1

    total_gb = sum(p.stat().st_size for p in DATA_DIR.glob("*.parquet")) / 1e9
    print(f"{now} OK newest={newest.name} age={age_h * 60:.0f}min "
          f"parquet_total={total_gb:.2f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
