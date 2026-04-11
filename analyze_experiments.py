#!/usr/bin/env python3
"""
Experiment Analysis: Domain Structure & Cumulative Cultural Evolution
=====================================================================

Reads all experiment runs from experiments/ and produces:
  1. A comparative summary CSV (experiments/analysis_summary.csv)
  2. A human-readable report (experiments/analysis_report.txt)

Usage:
    python analyze_experiments.py
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

EXPERIMENTS_DIR = "experiments"
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
        'name': name, 'domain_mode': 'flat', 'strategy': 'none',
        'policy': 'none', 'lifespan': False, 'uniform': False, 'strategy_value': None,
    }
    if name.startswith('sim') or 'ctrl_sim' in name:
        info['domain_mode'] = 'similarity'
    elif name.startswith('lin') or 'ctrl_lin' in name:
        info['domain_mode'] = 'lineage'
    elif name.startswith('img_similarity'):
        info['domain_mode'] = 'similarity'
    elif name.startswith('img_lineage'):
        info['domain_mode'] = 'lineage'

    for s in ['nearest', 'mid', 'far']:
        if f'_{s}_' in name or name.endswith(f'_{s}'):
            info['strategy'] = s
    for p in ['novelty_match', 'simple']:
        if p in name:
            info['policy'] = p
    if 'lifespan' in name:
        info['lifespan'] = True
    if 'uniform' in name:
        info['uniform'] = True
    if '_sv' in name:
        try:
            sv_part = name.split('_sv')[1].split('_')[0]
            info['strategy_value'] = float(sv_part.replace('p', '.'))
            info['strategy'] = f'sv={info["strategy_value"]}'
        except (ValueError, IndexError):
            pass
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

    # Domain growth
    domain_sizes = [safe_int(e.get('domain_size')) for e in events if e.get('domain_size')]
    m['final_domain_size'] = domain_sizes[-1] if domain_sizes else 0
    m['max_domain_size'] = max(domain_sizes) if domain_sizes else 0

    accepted_shares = [s for s in shares if s.get('accepted') == 'True']
    rejected_shares = [s for s in shares if s.get('accepted') == 'False']
    m['total_shares'] = len(shares)
    m['total_accepted'] = len(accepted_shares)
    m['total_rejected'] = len(rejected_shares)
    m['acceptance_rate'] = len(accepted_shares) / max(1, len(shares))

    # Interest & novelty
    gen_interests = [safe_float(e.get('interest')) for e in generations]
    gen_novelties = [safe_float(e.get('novelty')) for e in generations]
    m['mean_interest_generated'] = np.mean(gen_interests) if gen_interests else 0
    m['std_interest_generated'] = np.std(gen_interests) if gen_interests else 0
    m['mean_novelty_generated'] = np.mean(gen_novelties) if gen_novelties else 0
    m['std_novelty_generated'] = np.std(gen_novelties) if gen_novelties else 0

    if len(gen_interests) > 20:
        q = len(gen_interests) // 4
        m['interest_first_quarter'] = np.mean(gen_interests[:q])
        m['interest_last_quarter'] = np.mean(gen_interests[-q:])
        m['interest_trend'] = m['interest_last_quarter'] - m['interest_first_quarter']
    else:
        m['interest_first_quarter'] = m['interest_last_quarter'] = m['interest_trend'] = 0

    if len(gen_interests) > 50:
        w = 50
        rolling = [np.mean(gen_interests[i:i+w]) for i in range(len(gen_interests) - w)]
        m['peak_sustained_interest'] = max(rolling) if rolling else 0
    else:
        m['peak_sustained_interest'] = np.mean(gen_interests) if gen_interests else 0

    shared_interests = [safe_float(e.get('interest')) for e in shares]
    accepted_interests = [safe_float(e.get('evaluated_interest')) for e in accepted_shares]
    rejected_interests = [safe_float(e.get('evaluated_interest')) for e in rejected_shares]
    m['mean_shared_interest'] = np.mean(shared_interests) if shared_interests else 0
    m['mean_accepted_interest'] = np.mean(accepted_interests) if accepted_interests else 0
    m['mean_rejected_interest'] = np.mean(rejected_interests) if rejected_interests else 0

    # Boredom
    m['total_boredom_adoptions'] = len(boredom_adoptions)
    m['boredom_rate'] = len(boredom_adoptions) / max(1, len(generations))
    boredom_sources = Counter(e.get('source', 'unknown') for e in boredom_adoptions)
    m['boredom_domain_retrievals'] = sum(v for k, v in boredom_sources.items() if k in ('flat', 'similarity', 'lineage', 'domain_exploration'))
    m['boredom_random_restarts'] = boredom_sources.get('random_restart', 0)
    m['boredom_hedonic_retreats'] = boredom_sources.get('hedonic_retreat', 0)

    # Domain retrieval patterns
    domain_retrievals = [d for d in domain_events if d.get('operation') == 'retrieve']
    relation_types = Counter(d.get('relation_type', 'unknown') for d in domain_retrievals)
    m['retrieval_count'] = len(domain_retrievals)
    for rtype in ['parent', 'child', 'sibling', 'ancestor', 'descendant', 'same_lineage', 'random']:
        m[f'retrieval_{rtype}'] = relation_types.get(rtype, 0)
    fallbacks = [d for d in domain_retrievals if d.get('retrieval_fallback_random') == 'True']
    m['retrieval_fallback_rate'] = len(fallbacks) / max(1, len(domain_retrievals))
    buckets = Counter(d.get('retrieval_bucket', 'unknown') for d in domain_retrievals)
    for bucket in ['close', 'moderate', 'far', 'random']:
        m[f'bucket_{bucket}'] = buckets.get(bucket, 0)
    ret_scores = [safe_float(d.get('relevance_score')) for d in domain_retrievals if d.get('relevance_score') and d.get('relevance_score') != 'None']
    m['mean_retrieval_score'] = np.mean(ret_scores) if ret_scores else 0

    # Lineage depth
    lineage_depths = [safe_int(e.get('lineage_depth')) for e in events if e.get('lineage_depth') and e.get('lineage_depth') != 'None']
    m['max_lineage_depth'] = max(lineage_depths) if lineage_depths else 0
    m['mean_lineage_depth'] = np.mean(lineage_depths) if lineage_depths else 0

    # Creator diversity
    domain_adds = [d for d in domain_events if d.get('operation') == 'add']
    m['unique_domain_creators'] = len(set(d.get('creator_id') for d in domain_adds if d.get('creator_id')))
    m['unique_root_creators'] = len(set(d.get('root_creator_id') for d in domain_adds if d.get('root_creator_id')))

    # Agent init
    if agent_inits:
        pn = [safe_float(a.get('preferred_novelty', 0.5)) for a in agent_inits]
        m['novelty_pref_mean'] = np.mean(pn)
        m['novelty_pref_std'] = np.std(pn)
    else:
        m['novelty_pref_mean'] = 0.5
        m['novelty_pref_std'] = 0

    # Agent state
    if agent_states:
        last_step = max(safe_int(s.get('step')) for s in agent_states)
        final = [s for s in agent_states if safe_int(s.get('step')) == last_step]
        ci = [safe_float(s.get('cumulative_interest')) for s in final]
        rs = [safe_float(s.get('repository_size')) for s in final]
        m['final_mean_cumulative_interest'] = np.mean(ci) if ci else 0
        m['final_std_cumulative_interest'] = np.std(ci) if ci else 0
        m['final_mean_repo_size'] = np.mean(rs) if rs else 0
        if ci and len(ci) > 1:
            sc = sorted(ci)
            n = len(sc)
            cum = sum((2*(i+1)-n-1)*v for i, v in enumerate(sc))
            m['interest_gini'] = max(0, min(1, cum / (n * max(1e-8, sum(sc)))))
        else:
            m['interest_gini'] = 0
    else:
        m['final_mean_cumulative_interest'] = m['final_std_cumulative_interest'] = m['final_mean_repo_size'] = m['interest_gini'] = 0

    # Lifespan
    m['agent_deaths'] = len(deaths)
    m['agent_births'] = len(births)

    # Domain entry temporal pattern
    if domain_adds:
        entry_steps = sorted(safe_int(d.get('step')) for d in domain_adds)
        if len(entry_steps) > 1:
            half = max(entry_steps) // 2
            fh = sum(1 for s in entry_steps if s <= half)
            sh = sum(1 for s in entry_steps if s > half)
            m['domain_entries_first_half'] = fh
            m['domain_entries_second_half'] = sh
            m['domain_entry_ratio'] = sh / max(1, fh)
        else:
            m['domain_entries_first_half'] = len(entry_steps)
            m['domain_entries_second_half'] = 0
            m['domain_entry_ratio'] = 0
    else:
        m['domain_entries_first_half'] = m['domain_entries_second_half'] = 0
        m['domain_entry_ratio'] = 0

    return m


def generate_report(all_metrics):
    L = []
    def sec(t): L.append(""); L.append("=" * 80); L.append(f"  {t}"); L.append("=" * 80)
    def sub(t): L.append(""); L.append(f"  --- {t} ---")
    def tr(label, *vals):
        s = "  {:<45s}".format(label)
        for v in vals:
            if isinstance(v, float): s += " {:>12.4f}".format(v)
            elif isinstance(v, int): s += " {:>12d}".format(v)
            else: s += " {:>12s}".format(str(v))
        L.append(s)

    L.append("+" + "=" * 78 + "+")
    L.append("|  ANALYSIS REPORT: Domain Structure & Cumulative Cultural Evolution" + " " * 10 + "|")
    L.append(f"|  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 44 + "|")
    L.append(f"|  Experiments analyzed: {len(all_metrics)}" + " " * (78 - 24 - len(str(len(all_metrics)))) + "|")
    L.append("+" + "=" * 78 + "+")

    def by_mode(mode): return [m for m in all_metrics if m['domain_mode']==mode and not m['lifespan'] and not m['uniform'] and m.get('strategy_value') is None and not m['name'].startswith('img_')]
    def by_mode_ls(mode): return [m for m in all_metrics if m['domain_mode']==mode and m['lifespan'] and not m['uniform']]
    def get(runs, key, agg='mean'):
        vals = [m.get(key, 0) for m in runs]
        if not vals: return 0
        if agg == 'mean': return np.mean(vals)
        elif agg == 'max': return max(vals)
        return vals

    flat, sim, lin = by_mode('flat'), by_mode('similarity'), by_mode('lineage')

    # Section 1
    sec("1. DOMAIN STRUCTURE COMPARISON (without lifespan)")
    L.append("")
    L.append("  flat = no structure (Motz baseline), similarity = low structure, lineage = high structure")
    L.append("")
    hdr = "  {:<45s} {:>12s} {:>12s} {:>12s}".format("Metric", "Flat", "Similarity", "Lineage")
    L.append(hdr); L.append("  " + "-" * 81)

    for label, key, *rest in [
        ('Final domain size', 'final_domain_size'),
        ('Acceptance rate', 'acceptance_rate'),
        ('Mean interest (generated)', 'mean_interest_generated'),
        ('Interest trend (late - early)', 'interest_trend'),
        ('Peak sustained interest', 'peak_sustained_interest'),
        ('Mean novelty (generated)', 'mean_novelty_generated'),
        ('Boredom rate', 'boredom_rate'),
        ('Max lineage depth', 'max_lineage_depth', 'max'),
        ('Mean lineage depth', 'mean_lineage_depth'),
        ('Unique domain creators', 'unique_domain_creators'),
        ('Retrieval fallback rate', 'retrieval_fallback_rate'),
        ('Interest Gini (inequality)', 'interest_gini'),
        ('Final cumulative interest', 'final_mean_cumulative_interest'),
        ('Domain entries (2nd half ratio)', 'domain_entry_ratio'),
    ]:
        agg = rest[0] if rest else 'mean'
        tr(label, get(flat, key, agg), get(sim, key, agg), get(lin, key, agg))

    # Section 2
    sec("2. AGENT STRATEGY COMPARISON")
    L.append(""); L.append("  nearest = exploit close, mid = balanced, far = explore distant"); L.append("")
    for ml, mk in [("Similarity", "similarity"), ("Lineage", "lineage")]:
        sub(f"{ml} Domain")
        L.append("  {:<45s} {:>12s} {:>12s} {:>12s}".format("Metric", "Nearest", "Mid", "Far"))
        L.append("  " + "-" * 81)
        strats = {s: [m for m in all_metrics if m['domain_mode']==mk and m['strategy']==s and not m['lifespan'] and not m['uniform'] and m.get('strategy_value') is None] for s in ['nearest','mid','far']}
        for label, key, *rest in [
            ('Mean interest (generated)', 'mean_interest_generated'),
            ('Interest trend', 'interest_trend'),
            ('Peak sustained interest', 'peak_sustained_interest'),
            ('Acceptance rate', 'acceptance_rate'),
            ('Boredom rate', 'boredom_rate'),
            ('Final domain size', 'final_domain_size'),
            ('Max lineage depth', 'max_lineage_depth', 'max'),
            ('Retrieval fallback rate', 'retrieval_fallback_rate'),
            ('Mean retrieval relevance', 'mean_retrieval_score'),
        ]:
            agg = rest[0] if rest else 'mean'
            tr(label, get(strats['nearest'], key, agg), get(strats['mid'], key, agg), get(strats['far'], key, agg))

    # Section 3
    sec("3. SELECTION POLICY: simple vs novelty_match")
    L.append(""); L.append("  Does matching retrieval to agent novelty preference help?"); L.append("")
    for ml, mk in [("Similarity", "similarity"), ("Lineage", "lineage")]:
        sub(f"{ml} Domain")
        L.append("  {:<45s} {:>12s} {:>12s}".format("Metric", "Simple", "Nov.Match"))
        L.append("  " + "-" * 69)
        pols = {p: [m for m in all_metrics if m['domain_mode']==mk and m['policy']==p and not m['lifespan'] and not m['uniform'] and m.get('strategy_value') is None] for p in ['simple','novelty_match']}
        for label, key in [
            ('Mean interest (generated)', 'mean_interest_generated'),
            ('Interest trend', 'interest_trend'),
            ('Peak sustained interest', 'peak_sustained_interest'),
            ('Acceptance rate', 'acceptance_rate'),
            ('Boredom rate', 'boredom_rate'),
            ('Final domain size', 'final_domain_size'),
            ('Interest Gini', 'interest_gini'),
        ]:
            tr(label, get(pols['simple'], key), get(pols['novelty_match'], key))

    # Section 4
    sec("4. GENERATIONAL TURNOVER (lifespan effects)")
    L.append(""); L.append("  Does agent replacement affect cultural accumulation differently by structure?"); L.append("")
    flat_ls, sim_ls, lin_ls = by_mode_ls('flat'), by_mode_ls('similarity'), by_mode_ls('lineage')
    L.append("  {:<45s} {:>12s} {:>12s} {:>12s}".format("Metric", "Flat", "Similarity", "Lineage"))
    L.append("  " + "-" * 81)
    for label, key, *rest in [
        ('Final domain size (w/ lifespan)', 'final_domain_size'),
        ('Acceptance rate (w/ lifespan)', 'acceptance_rate'),
        ('Mean interest (w/ lifespan)', 'mean_interest_generated'),
        ('Interest trend (w/ lifespan)', 'interest_trend'),
        ('Boredom rate (w/ lifespan)', 'boredom_rate'),
        ('Max lineage depth (w/ lifespan)', 'max_lineage_depth', 'max'),
        ('Agent deaths', 'agent_deaths'),
        ('Agent births', 'agent_births'),
        ('Unique domain creators (w/ lifespan)', 'unique_domain_creators'),
    ]:
        agg = rest[0] if rest else 'mean'
        tr(label, get(flat_ls, key, agg), get(sim_ls, key, agg), get(lin_ls, key, agg))

    sub("Lifespan Impact (delta: ON - OFF)")
    L.append("  {:<45s} {:>12s} {:>12s} {:>12s}".format("Metric", "Flat D", "Sim D", "Lin D"))
    L.append("  " + "-" * 81)
    for label, key, *rest in [
        ('D Domain size', 'final_domain_size'),
        ('D Acceptance rate', 'acceptance_rate'),
        ('D Mean interest', 'mean_interest_generated'),
        ('D Interest trend', 'interest_trend'),
        ('D Boredom rate', 'boredom_rate'),
        ('D Max lineage depth', 'max_lineage_depth', 'max'),
    ]:
        agg = rest[0] if rest else 'mean'
        tr(label, get(flat_ls,key,agg)-get(flat,key,agg), get(sim_ls,key,agg)-get(sim,key,agg), get(lin_ls,key,agg)-get(lin,key,agg))

    # Section 5
    sec("5. LINEAGE RETRIEVAL RELATION PATTERNS")
    L.append(""); L.append("  What ancestral relationships are agents exploiting? (lineage mode only)"); L.append("")
    lin_all = [m for m in all_metrics if m['domain_mode']=='lineage' and not m['uniform']]
    if lin_all:
        L.append("  {:<25s} {:>10s} {:>10s}".format("Relation Type", "Count", "% of total"))
        L.append("  " + "-" * 45)
        rkeys = ['parent','child','sibling','ancestor','descendant','same_lineage','random']
        tots = {k: sum(m.get(f'retrieval_{k}',0) for m in lin_all) for k in rkeys}
        gt = sum(tots.values())
        for k in rkeys:
            L.append("  {:<25s} {:>10d} {:>9.1f}%".format(k, tots[k], 100*tots[k]/max(1,gt)))

    # Section 6
    sec("6. CONTINUOUS STRATEGY VALUE SWEEP")
    L.append(""); L.append("  Strategy position: 0=closest, 1=farthest"); L.append("")
    sv_runs = [m for m in all_metrics if m.get('strategy_value') is not None]
    for ml, mk in [("Similarity","similarity"),("Lineage","lineage")]:
        sub(f"{ml} Domain")
        mode_sv = sorted([m for m in sv_runs if m['domain_mode']==mk], key=lambda m: m.get('strategy_value',0))
        if mode_sv:
            L.append("  {:<10s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format("SV","Interest","Trend","Boredom","DomainSz","Fallback%"))
            L.append("  " + "-" * 70)
            for m in mode_sv:
                L.append("  {:<10.2f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12d} {:>11.1f}%".format(
                    m.get('strategy_value',0), m.get('mean_interest_generated',0),
                    m.get('interest_trend',0), m.get('boredom_rate',0),
                    int(m.get('final_domain_size',0)), 100*m.get('retrieval_fallback_rate',0)))

    # Section 7
    sec("7. CONTROL CONDITIONS (uniform novelty preference)")
    L.append("")
    ctrl = [m for m in all_metrics if m['uniform']]
    if ctrl:
        L.append("  {:<45s} {:>12s} {:>12s} {:>12s}".format("Metric","Flat(U)","Sim(U)","Lin(U)"))
        L.append("  " + "-" * 81)
        cf = [m for m in ctrl if m['domain_mode']=='flat']
        cs = [m for m in ctrl if m['domain_mode']=='similarity']
        cl = [m for m in ctrl if m['domain_mode']=='lineage']
        for label, key in [
            ('Mean interest','mean_interest_generated'),('Interest trend','interest_trend'),
            ('Acceptance rate','acceptance_rate'),('Final domain size','final_domain_size'),
            ('Boredom rate','boredom_rate'),('Interest Gini','interest_gini'),
        ]:
            tr(label, get(cf,key), get(cs,key), get(cl,key))

    # Section 8
    sec("8. KEY FINDINGS SUMMARY")
    L.append("")
    if flat and sim and lin:
        best_i = max([('flat',get(flat,'mean_interest_generated')),('similarity',get(sim,'mean_interest_generated')),('lineage',get(lin,'mean_interest_generated'))], key=lambda x:x[1])
        L.append(f"  * Highest mean interest: {best_i[0]} domain ({best_i[1]:.4f})")
        best_t = max([('flat',get(flat,'interest_trend')),('similarity',get(sim,'interest_trend')),('lineage',get(lin,'interest_trend'))], key=lambda x:x[1])
        L.append(f"  * Best interest trajectory: {best_t[0]} domain (D={best_t[1]:+.4f})")
        best_d = max([('flat',get(flat,'max_lineage_depth','max')),('similarity',get(sim,'max_lineage_depth','max')),('lineage',get(lin,'max_lineage_depth','max'))], key=lambda x:x[1])
        L.append(f"  * Deepest cultural lineage: {best_d[0]} domain (depth={int(best_d[1])})")
        low_b = min([('flat',get(flat,'boredom_rate')),('similarity',get(sim,'boredom_rate')),('lineage',get(lin,'boredom_rate'))], key=lambda x:x[1])
        L.append(f"  * Lowest boredom rate: {low_b[0]} domain ({low_b[1]:.4f})")
        big_d = max([('flat',get(flat,'final_domain_size')),('similarity',get(sim,'final_domain_size')),('lineage',get(lin,'final_domain_size'))], key=lambda x:x[1])
        L.append(f"  * Largest final domain: {big_d[0]} ({int(big_d[1])} artifacts)")

    if flat_ls and sim_ls and lin_ls and flat and sim and lin:
        L.append(""); L.append("  Lifespan impact summary:")
        for ml, off, on in [("flat",flat,flat_ls),("similarity",sim,sim_ls),("lineage",lin,lin_ls)]:
            di = get(on,'mean_interest_generated') - get(off,'mean_interest_generated')
            dd = get(on,'final_domain_size') - get(off,'final_domain_size')
            arrow = "^" if di > 0.001 else ("v" if di < -0.001 else "=")
            L.append(f"    {ml:>12s}: interest {arrow} ({di:+.4f}), domain size D={dd:+.0f}")

    L.append(""); L.append("=" * 80); L.append("  END OF REPORT"); L.append("=" * 80)
    return "\n".join(L)


def main():
    if not os.path.isdir(EXPERIMENTS_DIR):
        print(f"Error: '{EXPERIMENTS_DIR}' not found. Run run_experiments.sh first.")
        sys.exit(1)

    run_dirs = sorted([
        d for d in os.listdir(EXPERIMENTS_DIR)
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))
        and os.path.exists(os.path.join(EXPERIMENTS_DIR, d, "events.csv"))
    ])
    if not run_dirs:
        print("No experiment runs found. Run run_experiments.sh first.")
        sys.exit(1)

    print(f"Found {len(run_dirs)} experiment runs. Analyzing...")

    all_metrics = []
    for rn in run_dirs:
        rd = os.path.join(EXPERIMENTS_DIR, rn)
        ri = parse_run_name(rn)
        print(f"  {rn} ({ri['domain_mode']}/{ri['strategy']}/{ri['policy']})")
        met = compute_metrics(rd, ri)
        if met:
            all_metrics.append(met)
        else:
            print(f"    WARNING: no data, skipping")

    if not all_metrics:
        print("No valid data. Exiting.")
        sys.exit(1)

    # Summary CSV
    fnames = sorted(set().union(*(m.keys() for m in all_metrics)))
    priority = ['name','domain_mode','strategy','policy','lifespan','uniform','strategy_value']
    ordered = [f for f in priority if f in fnames] + [f for f in fnames if f not in priority]
    with open(OUTPUT_SUMMARY, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for m in all_metrics:
            w.writerow(m)
    print(f"\nSummary CSV: {OUTPUT_SUMMARY}")

    # Report
    report = generate_report(all_metrics)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    print(f"Report: {OUTPUT_REPORT}")
    print("\n" + "=" * 60)
    print("REPORT PREVIEW:")
    print("=" * 60)
    for line in report.split("\n")[:80]:
        print(line)
    print(f"...\n\nFull report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
