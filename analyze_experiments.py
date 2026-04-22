#!/usr/bin/env python3
"""
Experiment Analysis: Domain Structure + Adaptive Strategy + Lifespan
=====================================================================

Reads all experiment runs from experiments_adaptive/ and produces:
  1. experiments_adaptive/analysis_summary.csv
  2. experiments_adaptive/analysis_report.txt

Focuses on the core comparison for this research direction:
  - 3 domain structures (flat / similarity / lineage)
  - With vs without generational turnover (lifespan)
  - How agents' learned strategy_pref drifts in each structure

Usage:
    python analyze_experiments.py
"""

import os
import sys
import csv
import numpy as np
from collections import Counter
from datetime import datetime

EXPERIMENTS_DIR = "experiments_adaptive"
OUTPUT_SUMMARY = os.path.join(EXPERIMENTS_DIR, "analysis_summary.csv")
OUTPUT_REPORT = os.path.join(EXPERIMENTS_DIR, "analysis_report.txt")


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=0.0):
    if val is None or val == '' or val == 'None':
        return default
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    if val is None or val == '' or val == 'None':
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def parse_run_name(name):
    info = {
        'name': name,
        'domain_mode': 'flat',
        'lifespan': False,
    }
    if name.startswith('flat'):
        info['domain_mode'] = 'flat'
    elif name.startswith('similarity'):
        info['domain_mode'] = 'similarity'
    elif name.startswith('lineage'):
        info['domain_mode'] = 'lineage'
    if 'life' in name and 'nolife' not in name:
        info['lifespan'] = True
    return info


def compute_metrics(run_dir, run_info):
    events = load_csv(os.path.join(run_dir, "events.csv"))
    agent_inits = load_csv(os.path.join(run_dir, "agent_init.csv"))
    agent_states = load_csv(os.path.join(run_dir, "agent_state.csv"))
    domain_events = load_csv(os.path.join(run_dir, "domain.csv"))

    if not events:
        return None

    m = dict(run_info)

    generations = [e for e in events if e.get('event_type') == 'generation']
    shares = [e for e in events if e.get('event_type') == 'share']
    boredom_adoptions = [e for e in events if e.get('event_type') == 'boredom_adoption']
    deaths = [e for e in events if e.get('event_type') == 'agent_death']
    births = [e for e in events if e.get('event_type') == 'agent_birth']

    # ── Domain growth ─────────────────────────────────────────
    domain_sizes = [safe_int(e.get('domain_size')) for e in events if e.get('domain_size')]
    m['final_domain_size'] = domain_sizes[-1] if domain_sizes else 0
    m['max_domain_size'] = max(domain_sizes) if domain_sizes else 0

    accepted_shares = [s for s in shares if s.get('accepted') == 'True']
    rejected_shares = [s for s in shares if s.get('accepted') == 'False']
    m['total_shares'] = len(shares)
    m['total_accepted'] = len(accepted_shares)
    m['acceptance_rate'] = len(accepted_shares) / max(1, len(shares))

    # ── Interest & novelty ────────────────────────────────────
    gen_interests = [safe_float(e.get('interest')) for e in generations]
    gen_novelties = [safe_float(e.get('novelty')) for e in generations]
    m['mean_interest_generated'] = np.mean(gen_interests) if gen_interests else 0
    m['mean_novelty_generated'] = np.mean(gen_novelties) if gen_novelties else 0

    if len(gen_interests) > 20:
        q = len(gen_interests) // 4
        m['interest_first_quarter'] = np.mean(gen_interests[:q])
        m['interest_last_quarter'] = np.mean(gen_interests[-q:])
        m['interest_trend'] = m['interest_last_quarter'] - m['interest_first_quarter']
    else:
        m['interest_first_quarter'] = m['interest_last_quarter'] = m['interest_trend'] = 0

    # ── Boredom ───────────────────────────────────────────────
    m['total_boredom_adoptions'] = len(boredom_adoptions)
    m['boredom_rate'] = len(boredom_adoptions) / max(1, len(generations))

    # ── Adaptive strategy: drift & final distribution ─────────
    # Pull strategy_pref from domain.csv retrieve events.
    domain_retrievals = [d for d in domain_events if d.get('operation') == 'retrieve']
    strategy_prefs = [safe_float(d.get('agent_strategy_pref'), default=None)
                      for d in domain_retrievals
                      if d.get('agent_strategy_pref') not in (None, '', 'None')]

    if strategy_prefs:
        m['mean_strategy_pref'] = np.mean(strategy_prefs)
        m['std_strategy_pref'] = np.std(strategy_prefs)
        m['min_strategy_pref'] = min(strategy_prefs)
        m['max_strategy_pref'] = max(strategy_prefs)
        # Drift: late-run preferences vs. 0.5 baseline
        late = strategy_prefs[-max(10, len(strategy_prefs)//4):]
        m['late_mean_strategy_pref'] = np.mean(late)
        m['strategy_drift_from_neutral'] = abs(np.mean(late) - 0.5)
    else:
        m['mean_strategy_pref'] = m['std_strategy_pref'] = 0
        m['min_strategy_pref'] = m['max_strategy_pref'] = 0
        m['late_mean_strategy_pref'] = 0.5
        m['strategy_drift_from_neutral'] = 0

    # ── Retrieval patterns ────────────────────────────────────
    relation_types = Counter(d.get('relation_type', 'unknown') for d in domain_retrievals)
    m['retrieval_count'] = len(domain_retrievals)
    for rtype in ['parent', 'child', 'sibling', 'ancestor', 'descendant',
                  'same_lineage', 'random']:
        m[f'retrieval_{rtype}'] = relation_types.get(rtype, 0)
    fallbacks = [d for d in domain_retrievals
                 if d.get('retrieval_fallback_random') == 'True']
    m['retrieval_fallback_rate'] = len(fallbacks) / max(1, len(domain_retrievals))

    # ── Lineage depth ─────────────────────────────────────────
    depths = [safe_int(e.get('lineage_depth')) for e in events
              if e.get('lineage_depth') not in (None, '', 'None')]
    m['max_lineage_depth'] = max(depths) if depths else 0
    m['mean_lineage_depth'] = np.mean(depths) if depths else 0

    # ── Creator diversity ─────────────────────────────────────
    domain_adds = [d for d in domain_events if d.get('operation') == 'add']
    m['unique_domain_creators'] = len(set(d.get('creator_id') for d in domain_adds
                                          if d.get('creator_id')))

    # ── Agent state ───────────────────────────────────────────
    if agent_states:
        last_step = max(safe_int(s.get('step')) for s in agent_states)
        final = [s for s in agent_states if safe_int(s.get('step')) == last_step]
        ci = [safe_float(s.get('cumulative_interest')) for s in final]
        rs = [safe_float(s.get('repository_size')) for s in final]
        m['final_mean_cumulative_interest'] = np.mean(ci) if ci else 0
        m['final_mean_repo_size'] = np.mean(rs) if rs else 0
        if ci and len(ci) > 1:
            sc = sorted(ci)
            n = len(sc)
            cum = sum((2*(i+1)-n-1)*v for i, v in enumerate(sc))
            m['interest_gini'] = max(0, min(1, cum / (n * max(1e-8, sum(sc)))))
        else:
            m['interest_gini'] = 0
    else:
        m['final_mean_cumulative_interest'] = m['final_mean_repo_size'] = m['interest_gini'] = 0

    # ── Lifespan & gatekeeper hints ───────────────────────────
    m['agent_deaths'] = len(deaths)
    m['agent_births'] = len(births)

    # Gatekeeper proxy: share activity per agent id.
    # If a few agents send disproportionately many shares AND their shares
    # are accepted at higher rates, they're acting as gatekeepers.
    share_counts = Counter(s.get('sender_id') for s in shares if s.get('sender_id'))
    if share_counts:
        share_totals = sorted(share_counts.values(), reverse=True)
        top_3 = sum(share_totals[:3])
        m['share_concentration_top3'] = top_3 / max(1, sum(share_totals))
    else:
        m['share_concentration_top3'] = 0

    # Age-of-sender on successful shares (only meaningful with lifespan on).
    # Proxy: count shares sent by agents whose id < num_agents (the initial
    # cohort). These are the "elders" who never got replaced.
    return m


def generate_report(all_metrics):
    L = []

    def sec(t):
        L.append(""); L.append("=" * 80); L.append(f"  {t}"); L.append("=" * 80)

    def tr(label, *vals):
        s = "  {:<42s}".format(label)
        for v in vals:
            if isinstance(v, float):
                s += " {:>14.4f}".format(v)
            elif isinstance(v, int):
                s += " {:>14d}".format(v)
            else:
                s += " {:>14s}".format(str(v))
        L.append(s)

    L.append("+" + "=" * 78 + "+")
    L.append(f"|  ADAPTIVE STRATEGY TRIAL REPORT")
    L.append(f"|  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"|  Experiments: {len(all_metrics)}")
    L.append("+" + "=" * 78 + "+")

    def get(runs, key, agg='mean'):
        vals = [m.get(key, 0) for m in runs]
        if not vals:
            return 0
        if agg == 'mean':
            return float(np.mean(vals))
        if agg == 'max':
            return max(vals)
        return vals

    def by(mode, life):
        return [m for m in all_metrics
                if m['domain_mode'] == mode and m['lifespan'] == life]

    # ── Section 1: Structure comparison without lifespan ─────
    sec("1. DOMAIN STRUCTURE — LIFESPAN OFF")
    L.append("")
    L.append("  Baseline comparison: flat (unstructured) vs similarity (low structure)")
    L.append("  vs lineage (high structure), agents immortal.")
    L.append("")

    L.append("  {:<42s} {:>14s} {:>14s} {:>14s}".format(
        "Metric", "Flat", "Similarity", "Lineage"))
    L.append("  " + "-" * 90)

    flat = by('flat', False)
    sim = by('similarity', False)
    lin = by('lineage', False)

    for label, key, *rest in [
        ('Final domain size', 'final_domain_size'),
        ('Acceptance rate', 'acceptance_rate'),
        ('Mean interest generated', 'mean_interest_generated'),
        ('Interest trend (late - early)', 'interest_trend'),
        ('Mean novelty generated', 'mean_novelty_generated'),
        ('Boredom rate', 'boredom_rate'),
        ('Max lineage depth', 'max_lineage_depth', 'max'),
        ('Unique domain creators', 'unique_domain_creators'),
        ('Retrieval fallback rate', 'retrieval_fallback_rate'),
        ('Interest Gini', 'interest_gini'),
        ('Share concentration (top 3)', 'share_concentration_top3'),
    ]:
        agg = rest[0] if rest else 'mean'
        tr(label, get(flat, key, agg), get(sim, key, agg), get(lin, key, agg))

    # ── Section 2: Adaptive strategy learning signal ─────────
    sec("2. ADAPTIVE STRATEGY — HOW FAR DID AGENTS DRIFT?")
    L.append("")
    L.append("  strategy_pref starts at 0.5 for all agents. After learning,")
    L.append("  where do agents end up? Large drift = structure provides signal.")
    L.append("")

    L.append("  {:<42s} {:>14s} {:>14s} {:>14s}".format(
        "Metric", "Flat", "Similarity", "Lineage"))
    L.append("  " + "-" * 90)

    for label, key in [
        ('Mean strategy_pref across run', 'mean_strategy_pref'),
        ('Std of strategy_pref (spread)', 'std_strategy_pref'),
        ('Late-run mean strategy_pref', 'late_mean_strategy_pref'),
        ('Drift from 0.5 baseline', 'strategy_drift_from_neutral'),
        ('Min strategy_pref observed', 'min_strategy_pref'),
        ('Max strategy_pref observed', 'max_strategy_pref'),
    ]:
        tr(label, get(flat, key), get(sim, key), get(lin, key))

    # ── Section 3: Structure × Lifespan interaction ──────────
    sec("3. STRUCTURE x LIFESPAN INTERACTION")
    L.append("")
    L.append("  Does generational turnover change the picture? Compare each")
    L.append("  structure with and without agent replacement.")
    L.append("")

    flat_l = by('flat', True)
    sim_l = by('similarity', True)
    lin_l = by('lineage', True)

    L.append("  {:<42s} {:>14s} {:>14s} {:>14s}".format(
        "Metric (with lifespan)", "Flat", "Similarity", "Lineage"))
    L.append("  " + "-" * 90)

    for label, key, *rest in [
        ('Final domain size', 'final_domain_size'),
        ('Acceptance rate', 'acceptance_rate'),
        ('Mean interest generated', 'mean_interest_generated'),
        ('Interest trend', 'interest_trend'),
        ('Boredom rate', 'boredom_rate'),
        ('Max lineage depth', 'max_lineage_depth', 'max'),
        ('Agent deaths', 'agent_deaths'),
        ('Agent births', 'agent_births'),
        ('Unique domain creators', 'unique_domain_creators'),
        ('Share concentration (top 3)', 'share_concentration_top3'),
        ('Late mean strategy_pref', 'late_mean_strategy_pref'),
    ]:
        agg = rest[0] if rest else 'mean'
        tr(label, get(flat_l, key, agg), get(sim_l, key, agg), get(lin_l, key, agg))

    # ── Section 4: Deltas ────────────────────────────────────
    sec("4. LIFESPAN IMPACT (ON minus OFF)")
    L.append("")
    L.append("  Positive delta = lifespan helped this metric. Negative = hurt.")
    L.append("")

    L.append("  {:<42s} {:>14s} {:>14s} {:>14s}".format(
        "Delta", "Flat D", "Similarity D", "Lineage D"))
    L.append("  " + "-" * 90)

    for label, key, *rest in [
        ('Domain size', 'final_domain_size'),
        ('Acceptance rate', 'acceptance_rate'),
        ('Mean interest', 'mean_interest_generated'),
        ('Interest trend', 'interest_trend'),
        ('Boredom rate', 'boredom_rate'),
        ('Max lineage depth', 'max_lineage_depth', 'max'),
        ('Strategy drift from 0.5', 'strategy_drift_from_neutral'),
        ('Share concentration top 3', 'share_concentration_top3'),
    ]:
        agg = rest[0] if rest else 'mean'
        tr(label,
           get(flat_l, key, agg) - get(flat, key, agg),
           get(sim_l, key, agg) - get(sim, key, agg),
           get(lin_l, key, agg) - get(lin, key, agg))

    # ── Section 5: Lineage relation use ──────────────────────
    sec("5. LINEAGE RELATION PATTERNS (lineage mode only)")
    L.append("")
    lin_all = [m for m in all_metrics if m['domain_mode'] == 'lineage']
    if lin_all:
        L.append("  {:<25s} {:>12s} {:>12s}".format("Relation", "Count", "% of total"))
        L.append("  " + "-" * 50)
        rkeys = ['parent', 'child', 'sibling', 'ancestor',
                 'descendant', 'same_lineage', 'random']
        tots = {k: sum(m.get(f'retrieval_{k}', 0) for m in lin_all) for k in rkeys}
        gt = sum(tots.values())
        for k in rkeys:
            L.append("  {:<25s} {:>12d} {:>11.1f}%".format(
                k, tots[k], 100 * tots[k] / max(1, gt)))

    # ── Section 6: Auto-findings ─────────────────────────────
    sec("6. KEY OBSERVATIONS")
    L.append("")
    if flat and sim and lin:
        best_i = max([
            ('flat', get(flat, 'mean_interest_generated')),
            ('similarity', get(sim, 'mean_interest_generated')),
            ('lineage', get(lin, 'mean_interest_generated'))], key=lambda x: x[1])
        L.append(f"  * Highest mean interest: {best_i[0]} ({best_i[1]:.4f})")

        biggest_drift = max([
            ('flat', get(flat, 'strategy_drift_from_neutral')),
            ('similarity', get(sim, 'strategy_drift_from_neutral')),
            ('lineage', get(lin, 'strategy_drift_from_neutral'))], key=lambda x: x[1])
        L.append(f"  * Most strategy learning: {biggest_drift[0]} "
                 f"(drift={biggest_drift[1]:.4f})")

        most_concentrated = max([
            ('flat', get(flat, 'share_concentration_top3')),
            ('similarity', get(sim, 'share_concentration_top3')),
            ('lineage', get(lin, 'share_concentration_top3'))], key=lambda x: x[1])
        L.append(f"  * Most share concentration (gatekeeper proxy): "
                 f"{most_concentrated[0]} ({most_concentrated[1]:.1%})")

    L.append(""); L.append("=" * 80); L.append("  END OF REPORT"); L.append("=" * 80)
    return "\n".join(L)


def main():
    if not os.path.isdir(EXPERIMENTS_DIR):
        print(f"Error: '{EXPERIMENTS_DIR}' not found.")
        sys.exit(1)

    run_dirs = sorted([
        d for d in os.listdir(EXPERIMENTS_DIR)
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))
        and os.path.exists(os.path.join(EXPERIMENTS_DIR, d, "events.csv"))
    ])
    if not run_dirs:
        print("No runs found.")
        sys.exit(1)

    print(f"Found {len(run_dirs)} runs.")

    all_metrics = []
    for rn in run_dirs:
        rd = os.path.join(EXPERIMENTS_DIR, rn)
        ri = parse_run_name(rn)
        print(f"  {rn}  (mode={ri['domain_mode']}, lifespan={ri['lifespan']})")
        met = compute_metrics(rd, ri)
        if met:
            all_metrics.append(met)

    if not all_metrics:
        sys.exit(1)

    # Summary CSV
    fnames = sorted(set().union(*(m.keys() for m in all_metrics)))
    priority = ['name', 'domain_mode', 'lifespan']
    ordered = [f for f in priority if f in fnames] + [f for f in fnames if f not in priority]
    with open(OUTPUT_SUMMARY, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for m in all_metrics:
            w.writerow(m)
    print(f"\nSummary CSV: {OUTPUT_SUMMARY}")

    report = generate_report(all_metrics)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    print(f"Report: {OUTPUT_REPORT}\n")
    print(report)


if __name__ == "__main__":
    main()