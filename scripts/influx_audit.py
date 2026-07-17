"""
InfluxDB Data Audit Script
Run on hesiod: python3 influx_audit.py
Produces a JSON report of data coverage, gaps, and quality metrics.
"""

import json
import sys
import subprocess
from datetime import datetime, timedelta, timezone

try:
    from influxdb_client import InfluxDBClient
    import pandas as pd
except ImportError:
    print("ERROR: requires influxdb-client and pandas. Install with:")
    print("  pip install influxdb-client pandas")
    sys.exit(1)

INFLUX_URL = "http://localhost:8086"
INFLUX_ORG = "Sybl"
INFLUX_BUCKET = "coinbase_data"
SECRETS_FILE = "secrets.json"
HEALTH_LOG = "health_check.log"

def load_token():
    with open(SECRETS_FILE) as f:
        return json.load(f)["if-db-token"]

def get_disk_usage():
    try:
        r = subprocess.run(
            "sudo du -sh /var/lib/docker/volumes/influxdb_data/_data",
            shell=True, capture_output=True, text=True, timeout=30
        )
        return r.stdout.split()[0] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

def audit_restarts():
    """Parse health_check.log for restart events (grace period = just rebooted)."""
    restarts = []
    try:
        with open(HEALTH_LOG) as f:
            for line in f:
                if "grace period" in line and "Skipping" in line:
                    ts_str = line.split(" - ")[0].strip()
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                        restarts.append(ts.isoformat())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass

    # Deduplicate: group restarts within 5 minutes of each other
    if not restarts:
        return []
    deduped = [restarts[0]]
    for r in restarts[1:]:
        prev = datetime.fromisoformat(deduped[-1])
        curr = datetime.fromisoformat(r)
        if (curr - prev).total_seconds() > 300:
            deduped.append(r)
    return deduped

def query_current_range(query_api):
    """Get the absolute first and last timestamps and total counts per measurement."""
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: 0)
        |> filter(fn: (r) => r._field == "price")
        |> group(columns: ["_measurement"])
        |> reduce(
            identity: {{count: 0, first: 2099-01-01T00:00:00Z, last: 1970-01-01T00:00:00Z}},
            fn: (r, accumulator) => ({{
                count: accumulator.count + 1,
                first: if r._time < accumulator.first then r._time else accumulator.first,
                last: if r._time > accumulator.last then r._time else accumulator.last
            }})
        )
        |> keep(columns: ["_measurement", "count", "first", "last"])
    '''
    try:
        df = query_api.query_data_frame(query=flux)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        df = df.drop(columns=["result", "table"], errors="ignore")
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"  WARNING: reduce query failed ({e}), falling back to simpler query...")
        return query_current_range_simple(query_api)

def query_current_range_simple(query_api):
    """Fallback: get first/last/count per measurement using separate queries."""
    results = []
    for meas in ["ticker", "matches", "level2"]:
        print(f"  Querying {meas}...")
        # Get count + last (fast: just tail)
        flux_last = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: 0)
            |> filter(fn: (r) => r._measurement == "{meas}" and r._field == "price")
            |> group()
            |> last()
        '''
        flux_first = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: 0)
            |> filter(fn: (r) => r._measurement == "{meas}" and r._field == "price")
            |> group()
            |> first()
        '''
        try:
            df_last = query_api.query_data_frame(query=flux_last)
            df_first = query_api.query_data_frame(query=flux_first)
            if isinstance(df_last, list):
                df_last = pd.concat(df_last)
            if isinstance(df_first, list):
                df_first = pd.concat(df_first)

            first_time = df_first["_time"].min().isoformat() if not df_first.empty else None
            last_time = df_last["_time"].max().isoformat() if not df_last.empty else None
            results.append({
                "_measurement": meas,
                "first": first_time,
                "last": last_time,
            })
        except Exception as e:
            results.append({"_measurement": meas, "error": str(e)})
    return results

def query_daily_density(query_api, measurement="ticker"):
    """Count data points per day for a measurement to find gaps."""
    print(f"  Querying daily density for {measurement}...")
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: 0)
        |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "price")
        |> group()
        |> aggregateWindow(every: 1d, fn: count, createEmpty: true)
        |> yield(name: "daily_counts")
    '''
    try:
        df = query_api.query_data_frame(query=flux)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        df = df.drop(columns=["result", "table"], errors="ignore")
        if "_time" in df.columns:
            df["date"] = pd.to_datetime(df["_time"]).dt.strftime("%Y-%m-%d")
            df = df.rename(columns={"_value": "count"})
            return df[["date", "count"]].to_dict(orient="records")
    except Exception as e:
        print(f"  WARNING: daily density query failed: {e}")
    return []

def find_gaps(daily_data, min_gap_hours=4):
    """Identify days with zero or very low data as gaps."""
    if not daily_data:
        return []
    gaps = []
    median_count = sorted([d["count"] for d in daily_data if d["count"] and d["count"] > 0])
    if not median_count:
        return gaps
    median_val = median_count[len(median_count) // 2]
    threshold = median_val * 0.1  # less than 10% of median = gap

    for d in daily_data:
        count = d["count"] if d["count"] else 0
        if count < threshold:
            gaps.append({"date": d["date"], "count": count, "median": median_val})
    return gaps

def query_sample_data(query_api, measurement="ticker"):
    """Get a few recent rows to show the data schema."""
    print(f"  Sampling recent {measurement} data...")
    flux = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -5m)
        |> filter(fn: (r) => r._measurement == "{measurement}")
        |> pivot(rowKey:["_time", "product_id"], columnKey: ["_field"], valueColumn: "_value")
        |> group()
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: 5)
    '''
    try:
        df = query_api.query_data_frame(query=flux)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        df = df.drop(columns=["result", "table", "_start", "_stop"], errors="ignore")
        return {
            "columns": list(df.columns),
            "sample_rows": df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 60)
    print("InfluxDB Data Audit")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    token = load_token()
    report = {"generated_at": datetime.now().isoformat()}

    # Disk usage
    print("\n[1/6] Checking disk usage...")
    report["disk_usage"] = get_disk_usage()
    print(f"  Disk: {report['disk_usage']}")

    # Restart history
    print("\n[2/6] Scanning health_check.log for restarts...")
    restarts = audit_restarts()
    report["restarts"] = restarts
    report["restart_count"] = len(restarts)
    print(f"  Found {len(restarts)} restart events")
    for r in restarts:
        print(f"    - {r}")

    # System uptime
    print("\n[3/6] Checking system uptime...")
    try:
        r = subprocess.run("uptime", capture_output=True, text=True, shell=True, timeout=5)
        report["uptime"] = r.stdout.strip()
        print(f"  {report['uptime']}")
    except Exception:
        report["uptime"] = "unknown"

    # InfluxDB queries
    print("\n[4/6] Querying data range (this may take a while on 59GB)...")
    client = InfluxDBClient(url=INFLUX_URL, token=token, org=INFLUX_ORG, timeout=120_000)
    query_api = client.query_api()

    range_data = query_current_range(query_api)
    report["data_range"] = range_data
    for item in range_data:
        print(f"  {item.get('_measurement', '?')}: {item.get('first', '?')} -> {item.get('last', '?')}")

    # Daily density (use ticker — much smaller than level2)
    print("\n[5/6] Querying daily data density (ticker)...")
    daily = query_daily_density(query_api, "ticker")
    report["daily_density_ticker"] = daily
    gaps = find_gaps(daily)
    report["gaps"] = gaps
    print(f"  Total days with data: {len([d for d in daily if d.get('count', 0) > 0])}")
    print(f"  Days with gaps (<10% median volume): {len(gaps)}")
    if gaps:
        print("  Gap dates:")
        for g in gaps:
            print(f"    - {g['date']}: {g['count']} points (median: {g['median']})")

    # Sample data
    print("\n[6/6] Sampling recent data to show schema...")
    for meas in ["ticker", "matches", "level2"]:
        sample = query_sample_data(query_api, meas)
        report[f"sample_{meas}"] = sample
        if "columns" in sample:
            print(f"  {meas} columns: {sample['columns']}")

    client.close()

    # Write report
    output_file = "audit_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Audit complete. Full report written to {output_file}")
    print(f"{'=' * 60}")

    # Print summary
    print(f"\nSUMMARY:")
    print(f"  Disk usage: {report['disk_usage']}")
    print(f"  Restarts detected: {report['restart_count']}")
    print(f"  Data gaps: {len(gaps)}")
    print(f"  Uptime: {report['uptime']}")

if __name__ == "__main__":
    main()
