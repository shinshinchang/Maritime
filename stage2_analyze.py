#!/usr/bin/env python3
"""Analyze multi-vessel snapshot candidates without needing the 10 GiB AIS input."""
import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path


def quantile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)] if values else None


def run(input_csv, summary_json, cells_csv, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    daily, daily_bins, sizes, group_points = Counter(), defaultdict(set), Counter(), []
    speeds, diameters = [], []
    # Restrict exact-member run tracking to interpretable small group candidates.
    active = {}
    runs = []
    all_count, small_count = 0, 0

    def close_run(record):
        if record["snapshots"] >= 6:
            runs.append(record)

    with open(input_csv, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        expected = {"time_bin_start_utc", "group_size", "vessel_ids", "centroid_lat", "centroid_lon", "group_diameter_nm", "mean_sog_kn"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing candidate fields: {sorted(expected - set(reader.fieldnames or []))}")
        for row in reader:
            all_count += 1
            when = dt.datetime.fromisoformat(row["time_bin_start_utc"].replace("Z", "+00:00"))
            date = when.date().isoformat()
            day_bin = row["time_bin_start_utc"]
            daily[date] += 1
            daily_bins[date].add(day_bin)
            size = int(row["group_size"])
            sizes[size] += 1
            lat, lon = float(row["centroid_lat"]), float(row["centroid_lon"])
            diameter = float(row["group_diameter_nm"])
            sog = float(row["mean_sog_kn"]) if row["mean_sog_kn"] else None
            diameters.append(diameter)
            if sog is not None:
                speeds.append(sog)
            if 3 <= size <= 8 and sog is not None and sog < 3 and diameter <= 0.5:
                small_count += 1
                # Exact membership: stricter than Week 11's eventual Jaccard tracking.
                ids = tuple(sorted(row["vessel_ids"].split(";")))
                key = ids
                time_slot = int(when.timestamp() // 300)
                prev = active.get(key)
                if prev and time_slot == prev["last_slot"] + 1:
                    prev["snapshots"] += 1
                    prev["last_slot"] = time_slot
                    prev["end"] = when.isoformat()
                    prev["lat_sum"] += lat
                    prev["lon_sum"] += lon
                    prev["max_diameter_nm"] = max(prev["max_diameter_nm"], diameter)
                    prev["sog_sum"] += sog
                else:
                    if prev:
                        close_run(prev)
                    active[key] = {"member_ids": ids, "group_size": size, "snapshots": 1, "start": when.isoformat(), "end": when.isoformat(), "last_slot": time_slot, "lat_sum": lat, "lon_sum": lon, "sog_sum": sog, "max_diameter_nm": diameter}
            if all_count % 100000 == 0:
                print(f"Read {all_count:,} candidate snapshots ...", flush=True)
    for prev in active.values():
        close_run(prev)

    # Longer exact-member runs first; avoid displaying five runs from one neighborhood.
    runs.sort(key=lambda r: (r["snapshots"], -r["max_diameter_nm"]), reverse=True)
    selected = []
    for record in runs:
        lat = record["lat_sum"] / record["snapshots"]
        lon = record["lon_sum"] / record["snapshots"]
        if any(abs(lat - c["centroid_lat"]) < 0.25 and abs(lon - c["centroid_lon"]) < 0.25 for c in selected):
            continue
        selected.append({"case_id": f"CASE-{len(selected)+1:02d}", "start_utc": record["start"], "end_utc": record["end"], "group_size": record["group_size"], "snapshot_count": record["snapshots"], "observed_span_min": (record["snapshots"]-1)*5, "centroid_lat": round(lat, 6), "centroid_lon": round(lon, 6), "mean_sog_kn": round(record["sog_sum"]/record["snapshots"], 3), "max_snapshot_diameter_nm": round(record["max_diameter_nm"], 4), "vessel_ids": ";".join(record["member_ids"]), "note": "Exact same ship set in consecutive snapshot bins; not confirmed anomalous"})
        if len(selected) == 5:
            break

    with (output / "selected_cases_private.csv").open("w", newline="", encoding="utf-8") as file:
        columns = ["case_id", "start_utc", "end_utc", "group_size", "snapshot_count", "observed_span_min", "centroid_lat", "centroid_lon", "mean_sog_kn", "max_snapshot_diameter_nm", "vessel_ids", "note"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)
    with (output / "selected_cases_public.csv").open("w", newline="", encoding="utf-8") as file:
        columns = [c for c in (selected[0].keys() if selected else ["case_id", "start_utc", "end_utc", "group_size", "snapshot_count", "observed_span_min", "centroid_lat", "centroid_lon", "mean_sog_kn", "max_snapshot_diameter_nm", "note"]) if c != "vessel_ids"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in selected:
            writer.writerow({c: record[c] for c in columns})

    with (output / "daily_candidate_counts.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date_utc", "candidate_snapshots", "bins_with_candidates", "candidates_per_positive_bin"])
        for date in sorted(daily):
            writer.writerow([date, daily[date], len(daily_bins[date]), round(daily[date]/len(daily_bins[date]), 2)])

    stats = {"total_candidates": all_count, "unique_days_with_candidates": len(daily), "size_3": sizes[3], "size_4": sizes[4], "size_5": sizes[5], "greater_than_100_ships": sum(v for k,v in sizes.items() if k > 100), "small_slow_compact_snapshots": small_count, "exact_member_runs_6plus_bins": len(runs), "selected_distinct_cases": len(selected), "median_group_diameter_nm": quantile(diameters,.5), "median_group_mean_speed_kn": quantile(speeds,.5), "data_time_bins": summary["time_bins"], "dataset_time_start_utc": summary["time_min_utc"], "dataset_time_end_utc": summary["time_max_utc"], "limitations": ["A case is identified only when exactly the same 3–8 vessels occur in six or more consecutive 5-minute snapshots.", "Observed span is (number of snapshot intervals - 1) * 5 minutes; actual physical duration needs raw AIS verification.", "A group may appear in many snapshots; results are not independent events.", "Daily bins counted here are only bins with candidates, not every bin with AIS.", "Scatter plots use latitude/longitude without coastline or port basemap; not evidence of being offshore.", "A CSV file part may cover sparse, noncontiguous dates, so daily comparisons are not comparable exposure estimates."]}
    (output / "stage2_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(daily, sizes, selected, cells_csv, output)
    print(json.dumps({k:stats[k] for k in ["total_candidates", "small_slow_compact_snapshots", "exact_member_runs_6plus_bins", "selected_distinct_cases"]}, indent=2))
    return stats


def plot(daily, sizes, selected, cells_csv, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    days = sorted(daily)
    ax.bar(days, [daily[d] for d in days], color="#277da1")
    ax.set_ylabel("Candidate snapshots")
    ax.set_title("Candidate snapshots by UTC day (uneven source coverage)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output / "daily_candidates.png", dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = ["3", "4", "5", "6–8", "9–20", "21–100", ">100"]
    vals = [sizes[3], sizes[4], sizes[5], sum(sizes[x] for x in range(6,9)), sum(sizes[x] for x in range(9,21)), sum(v for k,v in sizes.items() if 21 <= k <= 100), sum(v for k,v in sizes.items() if k > 100)]
    ax.bar(bins, vals, color="#43aa8b")
    ax.set(xlabel="Group size (ships)", ylabel="Candidate snapshots", title="Detected group size distribution")
    fig.tight_layout()
    fig.savefig(output / "group_sizes.png", dpi=175)
    plt.close(fig)

    with open(cells_csv, newline="", encoding="utf-8-sig") as file:
        cells = list(csv.DictReader(file))
    fig, ax = plt.subplots(figsize=(9, 6))
    max_count = max(int(x["records"]) for x in cells)
    lon = [(int(x["gx"])+0.5)*0.01 for x in cells]
    lat = [(int(x["gy"])+0.5)*0.01 for x in cells]
    area = [70 + 500*int(x["records"])/max_count for x in cells]
    ax.scatter(lon, lat, s=area, alpha=.35, color="#277da1", label="Top 100 AIS record cells")
    for c in selected:
        ax.scatter(c["centroid_lon"], c["centroid_lat"], s=110, marker="x", color="#d1495b", linewidths=2)
        ax.annotate(c["case_id"], (c["centroid_lon"], c["centroid_lat"]), xytext=(5,5), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Top AIS activity cells and selected cases (no coastline layer)")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "case_locations.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--out", default="outputs/stage2")
    args = parser.parse_args()
    run(args.candidates, args.summary, args.cells, args.out)
