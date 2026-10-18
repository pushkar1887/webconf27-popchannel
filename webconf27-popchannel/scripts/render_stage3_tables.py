#!/usr/bin/env python
"""Render final Stage-3 RESULTS sections 1-2; preserve all other document bytes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from render_stage12_tables import f, first, index_rows, metric, number, replace_sections, stop, table, type_label

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("gowalla", "yelp2018", "amazonbook")
METHODS = ("full", "graphhash", "random", "dfreq")
LOSSES = ("bpr", "directau")


def final_id(dataset, method, loss):
    rid = f"{dataset}__mf__{method}__{loss}__s0"
    if dataset == "gowalla" and loss == "bpr":
        if method == "full":
            rid += "-long"
        elif method == "graphhash":
            rid += "-p500"
    return rid


MAIN_RUN_IDS = [final_id(d, m, loss) for d in DATASETS for m in METHODS for loss in LOSSES]


def lineage(row):
    if row.get("lineage"):
        return row["lineage"]
    rid = row["run_id"]
    return "modern" if (PROJECT / "runs" / rid / "metrics.json").is_file() else "legacy"


def status(row):
    return f"{row.get('status') or 'unavailable'} / {stop(row)}"


def render_sections(rows):
    by = index_rows(rows)
    main = []
    for rid in MAIN_RUN_IDS:
        modes = by.get(rid, {})
        row = first(modes) if modes else {"run_id": rid}
        parts = rid.split("__")
        main.append([parts[0], f"{parts[2]} × {parts[3]}", f"`{rid}`", metric(modes.get("raw")),
                     metric(modes.get("cosine")), metric(modes.get("cos_prior"), True),
                     metric(modes.get("raw_prior"), True), f(row.get("rho_item"), 3),
                     f"{lineage(row)} / {type_label(row)}", status(row)])
    out = ["## 1. Headline: final Stage 3 and retained references", "",
           "Generated from `results/master_metrics_v2.csv` by `scripts/render_stage3_tables.py`. "
           "Cells are test capped Recall@20 / NDCG@20 in percent. Both prior modes show their own validation-selected β; "
           "DirectAU cosine+prior versus BPR raw+prior is the registered repaired comparison. "
           "The 24-cell matrix uses final seed-0 checkpoints, with retained seeds and initialization variants below. "
           "The AmazonBook GraphHash BPR row is the retained legacy patience-500 checkpoint; all other matrix rows are modern.", "",
           table(["Dataset", "Method × loss", "Run ID", "Raw", "Cosine", "Cos + prior (β)",
                  "Raw + prior (β)", "ρ_item", "Lineage / comparison", "Status / stopping"], main), "",
           "**Capped baseline:** `amazonbook__mf__full__bpr__s0` reached 3000 epochs (best 2690); "
           "its scores are descriptive and do not establish convergence. The final Gowalla Full BPR reference is "
           "`s0-long`, and GraphHash BPR seed 0 is `s0-p500`; their earlier capped/patience-100 runs remain in the ledger.", ""]
    seed_rows = []
    for loss in LOSSES:
        ids = [f"gowalla__mf__graphhash__{loss}__s{s}" + ("-p500" if loss == "bpr" and s == 0 else "")
               for s in (0, 1, 2)]
        for mode in ("raw", "cosine", "cos_prior", "raw_prior"):
            cells = [by.get(rid, {}).get(mode) for rid in ids]
            if not all(cells):
                seed_rows.append([loss, mode, "—", "—", "incomplete final seed set"])
                continue
            def mean_sd(key):
                vals = [number(c.get(key)) for c in cells]
                if any(v is None for v in vals):
                    return "—"
                return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"
            seed_rows.append([loss, mode, mean_sd("test_recall20"), mean_sd("test_ndcg20"),
                              ", ".join(str(c.get("beta")) for c in cells)])
    out += ["**Final Gowalla GraphHash seeds 0/1/2:** mean ± sample SD over trained seeds. "
            "DirectAU uses A1/s1/s2; BPR uses s0-p500/s1/s2. These are training-seed spreads, "
            "separate from paired-user bootstrap intervals conditional on each checkpoint.", "",
            table(["Loss", "Scoring mode", "Recall@20 mean ± SD", "NDCG@20 mean ± SD", "β by seed 0/1/2"], seed_rows), ""]
    pairs = [
        ("Gowalla controlled Full reference A2", "gowalla__mf__full__directau__s0"),
        ("Gowalla matched-init GraphHash I1", "gowalla__mf__graphhash__directau__s0-initmatch"),
        ("Gowalla A1: initialization sensitivity versus I1", "gowalla__mf__graphhash__directau__s0"),
        ("AmazonBook controlled Full partner", "amazonbook__mf__full__directau__s0"),
        ("AmazonBook matched-init GraphHash I2", "amazonbook__mf__graphhash__directau__s0-initmatch"),
        ("AmazonBook B9: initialization sensitivity versus I2", "amazonbook__mf__graphhash__directau__s0")]
    pair_rows = []
    for label, rid in pairs:
        modes = by.get(rid, {})
        r = first(modes) if modes else {"run_id": rid}
        pair_rows.append([label, f"`{rid}`", metric(modes.get("raw")), metric(modes.get("cos_prior"), True),
                          type_label(r), status(r)])
    out += ["**Controlled and initialization comparisons:** A2 versus I1 and AmazonBook Full versus I2 "
            "are the eligible controlled pairs. A1 versus I1 and B9 versus I2 assess initialization sensitivity. "
            "A `controlled` label denotes the declared configuration and does not alone establish a causal mechanism.", "",
            table(["Role", "Run", "Raw", "Cos + prior (β)", "Comparison", "Status / stopping"], pair_rows), "", "---", "",
            "## 2. Complete run and control ledger (CSV-derived)", "",
            "`results/master_metrics.csv` preserves the v1 schema. `results/master_metrics_v2.csv` adds explicit lineage, "
            "capped/converged flags and separate E5/E7/E9 fields; its contract is `results/master_metrics_v2.schema.json`. "
            "Blank analysis values are unavailable, never zero. Weighted standard-denominator recall never replaces main capped Recall@20. "
            "Distinct legacy checkpoints retain their own IDs and stopping labels.", "",
            "### 2.1 Training, baseline and retained checkpoint lineages", ""]
    ledger = []
    for rid in sorted(by):
        modes = {m: r for m, r in by[rid].items() if not m.startswith("controls/") and r.get("row_kind") != "analysis_only"}
        if not modes:
            continue
        r = first(modes)
        ledger.append([f"`{rid}`", metric(modes.get("raw", modes.get("pop_only"))), metric(modes.get("cosine")),
                       metric(modes.get("cos_prior"), True), metric(modes.get("raw_prior"), True),
                       f(r.get("rho_item"), 3), lineage(r), type_label(r), status(r)])
    out += [table(["Run", "Raw / pop-only", "Cosine", "Cos + prior (β)", "Raw + prior (β)",
                   "ρ_item", "Lineage", "Comparison", "Status / stopping"], ledger), "",
            "### 2.2 Retained scoring-control artifacts", ""]
    controls = []
    for rid in sorted(by):
        modes = {m: r for m, r in by[rid].items() if m.startswith("controls/")}
        if not modes:
            continue
        r = first(modes)
        controls.append([f"`{rid}`", len(modes), lineage(r), type_label(r), status(r), r.get("session_slug") or "—"])
    out += [table(["Checkpoint", "Scored control cells", "Lineage", "Comparison", "Source status / stopping", "Session"], controls), "",
            "Every individual control cell remains in both CSVs, including negative, capped and fixed-budget results. "
            "Validation of a control file does not establish training convergence. E5 tail denominators, "
            "E7 precision deviations and storage scale, and E9 weighted/original-retuned-beta results have named columns in v2 "
            "and source-policy/status fields. Analysis-only legacy rows have no invented main score. "
            "The existing unconstrained learned-bias ablation remains distinct from the unexecuted C3 regularized residual model.", "", "---", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=PROJECT / "results/master_metrics_v2.csv")
    ap.add_argument("--results", type=Path, default=PROJECT / "RESULTS.md")
    args = ap.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    before = args.results.read_bytes()
    args.results.write_bytes(replace_sections(before, render_sections(rows)))
    print(f"Regenerated RESULTS sections 1-2 from {len(rows)} rows; all other bytes preserved.")


if __name__ == "__main__":
    main()
