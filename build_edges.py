#!/usr/bin/env python3
"""
Convert events.csv from a simulation run into a Gephi-compatible edge list.

Usage:
    python build_edges.py experiments_adaptive/lineage_nolife
    # produces experiments_adaptive/lineage_nolife/edges.csv
"""
import os
import sys
import csv
from collections import defaultdict


def build_edges(run_dir, mode="accepted_only"):
    """
    mode options:
      'all_shares'        -> every share attempt, weight = count
      'accepted_only'     -> only accepted shares, weight = count
      'accepted_weighted' -> only accepted shares, weight = sum of interest
    """
    events_path = os.path.join(run_dir, "events.csv")
    out_path = os.path.join(run_dir, "edges.csv")

    if not os.path.exists(events_path):
        print(f"No events.csv in {run_dir}")
        return

    edges = defaultdict(float)

    with open(events_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event_type") != "share":
                continue
            sender = row.get("sender_id")
            recipient = row.get("recipient_id")
            if not sender or not recipient or sender == recipient:
                continue

            if mode == "all_shares":
                edges[(sender, recipient)] += 1.0

            elif mode == "accepted_only":
                if row.get("accepted") == "True":
                    edges[(sender, recipient)] += 1.0

            elif mode == "accepted_weighted":
                if row.get("accepted") == "True":
                    try:
                        w = float(row.get("evaluated_interest") or 0.0)
                    except ValueError:
                        w = 0.0
                    edges[(sender, recipient)] += w

    # columns: Source, Target, Weight, Type
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target", "Weight", "Type"])
        for (src, tgt), weight in edges.items():
            if weight > 0:  # drop zero-weight edges
                writer.writerow([src, tgt, f"{weight:.4f}", "Directed"])

    print(f"Wrote {len(edges)} edges to {out_path} (mode={mode})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_edges.py <run_dir> [mode]")
        print("Modes: all_shares, accepted_only, accepted_weighted (default)")
        sys.exit(1)
    run_dir = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "accepted_weighted"
    build_edges(run_dir, mode)