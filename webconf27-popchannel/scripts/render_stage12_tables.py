#!/usr/bin/env python
"""Regenerate RESULTS sections 1-2 from the aggregated CSV, preserving all other bytes."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parent.parent


def number(value):
    return None if value in (None, "") else float(value)


def f(value, decimals=2, sign=False):
    x = number(value)
    if x is None:
        return "—"
    return format(x, ("+" if sign else "") + f".{decimals}f")


def metric(row, prior=False):
    if not row or number(row.get("test_recall20")) is None:
        return "—"
    text = f"{f(row.get('test_recall20'))} / {f(row.get('test_ndcg20'))}"
    if prior:
        text += f" (β {row.get('beta', '—')})"
    return text


def table(headers, body):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                       *["| " + " | ".join(str(v) for v in row) + " |" for row in body]])


def index_rows(rows):
    result = {}
    for row in rows:
        rid, mode = row["run_id"], row["scoring_mode"]
        if mode in result.setdefault(rid, {}):
            raise ValueError(f"duplicate ledger key: {rid} / {mode}")
        result[rid][mode] = row
    return result


def first(modes):
    return next(iter(modes.values()))


def stop(row):
    text = str(row.get("stop_reason") or "not recorded")
    if row.get("epochs_run") not in (None, ""):
        text += f"; best {row.get('best_epoch') or '—'} / ran {row['epochs_run']}"
    return text


def type_label(row):
    return str(row.get("comparison_type") or "reproduction")


def render_sections(rows):
    by = index_rows(rows)
    headline_ids = [
        "gowalla__mf__full__directau__s0", "gowalla__mf__full__bpr__s0",
        "gowalla__mf__graphhash__directau__s0", "gowalla__mf__graphhash__directau__s0-initmatch",
        "gowalla__mf__graphhash__bpr__s0", "gowalla__poponly__none__none__s0",
        "yelp2018__mf__full__directau__s0", "yelp2018__mf__full__bpr__s0",
        "yelp2018__mf__graphhash__directau__s0", "yelp2018__mf__graphhash__bpr__s0",
        "yelp2018__poponly__none__none__s0", "amazonbook__mf__full__directau__s0",
        "amazonbook__mf__graphhash__directau__s0", "amazonbook__mf__graphhash__bpr__s0",
        "amazonbook__poponly__none__none__s0"]
    body = []
    for rid in headline_ids:
        if rid not in by:
            continue
        modes = by[rid]
        row = first(modes)
        model = ("popularity only" if "pop_only" in modes
                 else f"{row.get('compression', '—')} × {row.get('loss', '—')}")
        if "initmatch" in rid:
            model += " (I1)"
        base = modes.get("raw", modes.get("pop_only", {}))
        corrected_mode = "raw_prior" if row.get("loss") == "bpr" else "cos_prior"
        body.append([row.get("dataset"), model, metric(base), metric(modes.get("cosine")),
                     metric(modes.get(corrected_mode), True), f(row.get("rho_item"), 3),
                     type_label(row), f"{row.get('status')} / {stop(row)}"])
    out = ["## 1. Headline so far (legacy and completed Stage 1–2)", "",
           "Generated from `results/master_metrics.csv` by `scripts/render_stage12_tables.py`. Test Recall@20 / NDCG@20 are percentages; β is validation-selected. The correction column uses cosine + prior for DirectAU and raw-dot + prior for BPR. It is not a post-hoc choice of the better test result.", "",
           table(["Dataset", "Model", "Raw / pop-only", "Cosine", "Correction (β)", "ρ_item", "Comparison", "Status / stopping"], body), "",
           "**Interpretation of comparison labels:** `controlled` marks an eligible controlled configuration, not an already completed mapping-only comparison. I1 is an **initialization sensitivity** comparison against A1; its matched Full partner A2 is still planned. Other rows are `reproduction` settings; popularity-only is a reproduced non-personalized baseline. Patience means the specified early-stopping rule fired; caps and fixed short budgets remain visible.", ""]
    seeds = []
    for seed in (0, 1, 2):
        rid = f"gowalla__mf__graphhash__directau__s{seed}"
        candidate = by.get(rid + "-legacy30", by.get(rid, {}))
        if candidate.get("cos_prior", {}).get("stop_reason") == "fixed_epochs":
            seeds.append(candidate)
    if len(seeds) == 3:
        def mean_sd(mode, key):
            vals = [number(s[mode][key]) for s in seeds]
            return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"
        out += ["**Legacy 30-epoch GraphHash–DirectAU, seeds 0/1/2 (reproduction; valid-short):** "
                f"cosine {mean_sd('cosine', 'test_recall20')} / {mean_sd('cosine', 'test_ndcg20')}; "
                f"cosine + prior {mean_sd('cos_prior', 'test_recall20')} / {mean_sd('cos_prior', 'test_ndcg20')}. "
                "These are training-seed standard deviations. The later A1 seed-0 run is a separate lineage; it is not substituted into this mean.", ""]
    gap_body = []
    for rid in headline_ids:
        modes = by.get(rid, {})
        if "raw" in modes and "cosine" in modes:
            raw = modes["raw"]
            gap_body.append([f"`{rid}`", f(raw.get("test_gap"), sign=True), f(raw.get("val_gap"), sign=True), type_label(raw)])
    out += ["**Raw-minus-cosine gaps:** compute from the unrounded source values, then round once.", "",
            table(["Run", "Test gap (points)", "Val gap (points)", "Comparison"], gap_body), "",
            "The legacy Gowalla Full DirectAU cosine is **13.00**, and its raw-minus-cosine test gap is **4.90** points. This corrects the earlier truncation to 12.99 and subtraction of displayed values.", "", "---", "",
            "## 2. Complete run and control ledger (CSV-derived)", "",
            "Every row is traceable to `results/master_metrics.csv`. Invalid, exploratory, fixed-budget and capped runs are retained with status; they are not promoted to final results. Legacy checkpoint labels and later runs are kept separate. A `-legacy30` suffix is added only where the combined training ledger would otherwise collide with a modern run ID.", "",
            "### 2.1 Training and popularity-only runs", ""]
    ledger = []
    for rid in sorted(by):
        modes = {m: r for m, r in by[rid].items() if not m.startswith("controls/")}
        if not modes:
            continue
        row = first(modes)
        ledger.append([f"`{rid}`", metric(modes.get("raw", modes.get("pop_only"))), metric(modes.get("cosine")),
                       metric(modes.get("cos_prior"), True), metric(modes.get("raw_prior"), True),
                       f(row.get("rho_item"), 3), f(row.get("train_min"), 1), type_label(row),
                       f"{row.get('status')} / {stop(row)}"])
    out += [table(["Run", "Raw / pop-only", "Cosine", "Cos + prior (β)", "Raw + prior (β)", "ρ_item", "Train min", "Comparison", "Status / stop"], ledger), "",
            "**Negative learned-bias experiment:** the extra `cos_b` / `cos_b_prior` scoring rows are retained in the master CSV. The original unconstrained-bias training is an ablation, not the regularized residual-bias Stage 3 proposal.", ""]
    bias = []
    for rid, modes in by.items():
        for mode in ("cos_b", "cos_b_prior"):
            if mode in modes:
                row = modes[mode]
                bias.append([f"`{rid}`", mode, f(row.get("val_recall20")), metric(row), type_label(row), row.get("status")])
    if bias:
        out += [table(["Run", "Scoring", "Val R@20", "Test R@20 / NDCG@20", "Comparison", "Status"], bias), ""]
    controls = []
    length = []
    for rid in sorted(by):
        modes = {m: r for m, r in by[rid].items() if m.startswith("controls/")}
        if not modes:
            continue
        row = first(modes)
        def cr(mode):
            return f(modes.get("controls/" + mode, {}).get("test_recall20"))
        controls.append([f"`{rid}`", cr("tie_keys/random"), cr("tie_keys/index"), cr("tie_keys/pop"),
                         cr("decomposition_fixed_beta/mu"), cr("decomposition_fixed_beta/eps"),
                         cr("decomposition_fixed_beta/both"), cr("within_bucket_shuffle"), type_label(row), row.get("status")])
        for name in ("unmodified", "len_shuffle_decile", "len_shuffle_global", "len_replace_fit"):
            rr = modes.get("controls/e2e/" + name)
            if rr:
                length.append([f"`{rid}`", name, f(rr.get("val_recall20")), metric(rr), type_label(rr), rr.get("status")])
    out += ["### 2.2 Stage 1 scoring controls", "",
            "Control rows inherit their source checkpoint's capped/short/valid status; validation of the scoring file does not make the training converge. Columns below are test Recall@20; fixed-β cells share the full prior's validation-selected β. Their gains need not add. Default legacy-topk, retuned, raw-prior, uint8 and all other scored cells are retained in the master CSV and the exhaustive companion table.", "",
            table(["Checkpoint label", "Random ties", "Index ties", "Pop ties", "Bucket μ", "Item ε", "Both", "Shuffled", "Comparison", "Source status"], controls), ""]
    if length:
        out += ["### 2.3 Full-model length interventions", "",
                table(["Checkpoint label", "Intervention", "Val R@20", "Test R@20 / NDCG@20", "Comparison", "Source status"], length), "",
                "Within-decile shuffling caused a substantial Gowalla drop, contrary to its prediction; the fitted-popularity replacement exceeded the learned-norm score. These interventions do not establish that popularity is the only useful content in vector length.", ""]
    out += ["**Complete machine-readable ledger:** `results/master_metrics.csv`. **Every scored control cell:** `results/stage12_controls_table.md` (generated from the same CSV). The complete artifacts retain validation scores, NDCG, random-key standard deviations, β, memory fields, mapping provenance, session, comparison type, stopping status and negative runs.", "", "---", ""]
    return "\n".join(out)


def render_controls(rows):
    body = [[f"`{r['run_id']}`", r["scoring_mode"], str(r.get("beta") or "0"),
             f(r.get("val_recall20")), metric(r), f(r.get("test_std_recall20"), 4),
             type_label(r), r.get("status")] for r in rows if r["scoring_mode"].startswith("controls/")]
    return "# Stage 1 control cells (generated from master_metrics.csv)\n\n" + table(
        ["Checkpoint", "Mode", "β", "Val R@20", "Test R@20 / NDCG@20", "Random-key SD", "Comparison", "Source status"], body) + "\n"


def replace_sections(original, rendered):
    start = original.index(b"## 1.")
    end = original.index(b"## 3.", start)
    newline = "\r\n" if b"\r\n" in original else "\n"
    middle = rendered.replace("\r\n", "\n").replace("\n", newline).encode("utf-8")
    return original[:start] + middle + original[end:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=PROJECT / "results" / "master_metrics.csv")
    ap.add_argument("--results", type=Path, default=PROJECT / "RESULTS.md")
    ap.add_argument("--controls-table", type=Path, default=PROJECT / "results" / "stage12_controls_table.md")
    args = ap.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    before = args.results.read_bytes()
    after = replace_sections(before, render_sections(rows))
    args.results.write_bytes(after)
    args.controls_table.write_text(render_controls(rows), encoding="utf-8", newline="\n")
    print(f"Regenerated sections 1-2 from {len(rows)} CSV rows; all other document bytes preserved.")


if __name__ == "__main__":
    main()
