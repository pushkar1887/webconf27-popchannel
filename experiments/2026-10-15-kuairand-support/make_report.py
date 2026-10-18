"""Generate the throwaway feasibility-probe report from saved receipts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    audit = json.loads((ROOT / "support_audit.json").read_text())
    baseline = json.loads((ROOT / "baseline_results.json").read_text())
    release = json.loads((ROOT / "release_receipt.json").read_text())
    verify = json.loads((ROOT / "verification.json").read_text())
    availability = json.loads((ROOT / "baseline_artifact_availability.json").read_text())
    metrics = pd.read_csv(ROOT / "baseline_metrics.csv")
    mismatch = sum(x["target_rule_mismatches"] for x in audit["sources"])
    zero_bad = sum(x["mismatches_zero_duration"] for x in audit["sources"])
    positive_bad = sum(x["mismatches_positive_duration"] for x in audit["sources"])
    split = audit["split_counts"]
    support = audit["context_support"]["tab"]["thresholds"]
    display_names = {
        "global_rate": "Global training rate",
        "context_rate": "Interface/context rate",
        "item_context_rate": "Item + interface counts",
        "user_context_rate": "User + interface counts",
        "item_plus_user_marginal": "Item/context + user-marginal effect",
        "item_plus_user_context": "Item/context + user/context effect",
        "item_plus_random_shared_s11": "Item/context + 256 random user buckets, seed 11",
        "item_plus_random_shared_s29": "Item/context + 256 random user buckets, seed 29",
        "item_plus_random_shared_s47": "Item/context + 256 random user buckets, seed 47",
    }
    text = [
        "# KuaiRand-Pure: executed support audit and minimal baseline probe",
        "",
        "Date: 15 October 2026. Scope: the user-approved next step, a local feasibility probe. All records are real released data. All models below are smoothed count-based component baselines, not learned embedding models or reproductions of GraphHash, CCE, or BACO.",
        "",
        "**[Likely] Verdict:** the data support a narrow chronological native-feedback prediction pilot after explicit integrity amendments. Context-specific user effects improve the standard-traffic count baseline, but the ordering reverses against item-only/random-sharing baselines on random exposures. This does not establish the proposed compression mechanism, a context-frequency-only effect, or a novel method. Do not launch the full campaign on the strength of this probe.",
        "",
        "## 1. Provenance and execution",
        "",
        f"[Certain] Downloaded the published {release['archive']['bytes']:,}-byte archive from Zenodo record {release['record']}; its MD5 matches the published checksum. Extracted only the three interaction CSVs. DOI: {release['doi']}. Record license: CC BY 4.0. The exact files, URLs, hashes and retrieval time are in release_receipt.json; the source record is in zenodo_record.json.",
        "",
        f"[Certain] Runtime: Python {audit['environment']['python']}, NumPy {audit['environment']['numpy']}, pandas {audit['environment']['pandas']}. No dependency installation, Kaggle operation, NVIDIA-key access, model download, GPU run or paid inference occurred. The Sonatype checking connector was unavailable; existing runtime libraries were used without a new dependency recommendation.",
        "",
        f"[Certain] Measured durations: download {release['download_seconds']:.2f} s; successful support audit {audit['elapsed_seconds']:.2f} s; count fitting/tuning/evaluation {baseline['elapsed_seconds']:.2f} s. These exclude investigation, scripting, failed integrity attempts and verification. They do not estimate neural baseline runtime.",
        "",
        "## 2. Integrity findings and recorded amendments",
        "",
        f"[Certain] Raw rows: **{audit['raw_total_rows']:,}**. Quarantined **{audit['duplicate_key_quarantine']['rows']:,} rows** belonging to **{audit['duplicate_key_quarantine']['event_key_groups']:,}** repeated (user_id, video_id, time_ms, tab) groups. Clean cohort: **{audit['total_rows']:,}**. Every row in such a group was quarantined, regardless of label. This conservative choice does not establish that every repeated key is an erroneous event.",
        "",
        f"[Certain] The native long_view label disagrees with the documented duration formula on **{mismatch:,} raw rows**: {zero_bad:,} with zero duration and {positive_bad:,} with positive duration. Labels were retained, not recomputed. No row was filtered according to target-rule agreement. Positive-duration sensitivity metrics are saved in baseline_results.json. Exact operational label semantics remain partly unresolved.",
        "",
        "[Certain] Splitting by the supplied date field did not produce strict raw event-time separation. The amended clock is Unix time_ms interpreted in UTC, with fixed calendar boundaries. No model had been fitted when either amendment was recorded. Protocol.json is the original record; protocol_amendment_01.json and protocol_amendment_02.json document the changes.",
        "",
        "[Certain] Only user_id, video_id and tab enter the fitted count predictors. Duration is used for an outcome-independent sensitivity subset; UTC hour bins are audited separately. Current play time, all other current feedback, all side-information files, and month-aggregated statistics are excluded from the models. The target is native long_view.",
        "",
        "## 3. Canonical chronological splits",
        "",
        "| Split | Clean rows | Users | Native positive rate | Warm user/item event coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["train", "validation", "test_standard", "test_random"]:
        s = split[name]
        text.append(f"| {name} | {s['rows']:,} | {s['users']:,} | {s['target_rate']:.4%} | {s['warm_user_item_fraction']:.4%} |")
    text.extend([
        "",
        "[Certain] Cutoffs are train before 2022-04-19 00:00 UTC, validation from then until 2022-04-22 00:00 UTC, and evaluation from then through the released period ending 8 May. Training and validation use standard traffic. Standard and random evaluation policies are reported separately. Clean raw timestamp ranges do not overlap between train, validation and evaluation.",
        "",
        "## 4. Support for conditional user behavior",
        "",
        "| Minimum observations in each of at least two training interfaces | Users | Fraction of training users | Standard evaluation event coverage | Random evaluation event coverage |",
        "|---|---:|---:|---:|---:|",
    ])
    for threshold, values in support.items():
        text.append(f"| {threshold} | {values['users_with_two_supported_contexts']:,} | {values['fraction_train_users']:.2%} | {values['test_standard_event_coverage']:.2%} | {values['test_random_event_coverage']:.2%} |")
    random_dominant = split["test_random"]["context_counts"]["1"] / split["test_random"]["rows"]
    text.extend([
        "",
        f"[Certain] The random evaluation is **{random_dominant:.4%} tab 1**. The median training user has two observed interfaces, and the median observed user/interface pair has eight records. Most users do not have strongly supported responses in several interfaces. Additional hour-bin support statistics are in support_audit.json; they did not change the primary tab context.",
        "",
        "[Likely] This limits a broad rare-context claim. Start with a training-defined supported cohort and report its coverage; do not silently discard sparsely observed users or treat missing behavior as agreement. The random policy provides an exposure-change test, not a clean isolated manipulation of context frequency.",
        "",
        "## 5. Minimal baseline results",
        "",
        "[Certain] The models use shrinkage probability tables. Combined models add a validation-selected user log-odds effect to the item/interface log odds. Smoothing candidates are 10, 50 and 200; effect weights are 0, .25, .5 and 1. Three random-sharing assignments use 256 buckets. All hyperparameters were selected on validation only; validation_selection.json was saved before test evaluation. There was no subsequent test-driven refit.",
        "",
        "**Lower log loss is better.** Probability-array bytes are not total process memory or a neural-compression benchmark.",
        "",
        "| Count baseline | Standard log loss | Random log loss | Fitted probability-array bytes |",
        "|---|---:|---:|---:|",
    ])
    for name, label in display_names.items():
        a = metrics.loc[(metrics.split == "test_standard") & (metrics.model == name)].iloc[0]
        b = metrics.loc[(metrics.split == "test_random") & (metrics.model == name)].iloc[0]
        text.append(f"| {label} | {a.log_loss:.6f} | {b.log_loss:.6f} | {int(a.probability_array_bytes):,} |")
    text.extend([
        "",
        "[Certain] Per-context metrics, Brier scores, calibration summaries, warm-user/item results, and the positive-duration subset are saved in baseline_results.json. These comparisons have unequal table capacities. They show what the specified count models do; they do not demonstrate the superiority of a compression algorithm.",
        "",
        "### Paired component contrasts on warm-user/warm-item rows",
        "",
        "[Certain] Intervals use 500 paired user-cluster bootstrap replicates. They condition on these periods, models and hyperparameter choices; they do not account for every source of temporal dependence or establish generalization across future periods.",
        "",
        "| Loss difference, positive favors individual user/context effects | Standard mean [95% interval] | Random mean [95% interval] |",
        "|---|---|---|",
    ])
    for key, label in [
        ("item_only_minus_individual_context", "Item/context baseline minus individual user/context"),
        ("marginal_user_minus_contextual_user", "User-marginal control minus user/context"),
        ("mean_random_assignment_loss_minus_individual_context", "Mean loss across random assignments minus user/context"),
    ]:
        cells = []
        for policy in ["test_standard", "test_random"]:
            c = baseline["paired_contrasts"][policy][key]
            interval = c["user_cluster_bootstrap_95_interval"]
            cells.append(f"{c['mean_loss_difference']:.6f} [{interval[0]:.6f}, {interval[1]:.6f}]")
        text.append(f"| {label} | {cells[0]} | {cells[1]} |")
    text.extend([
        "",
        "[Likely] The standard-traffic result supplies a reason for a small next-stage probe: the contextual user-count effect improves over both item/context and a marginal-user control. Its larger capacity and smoothing behavior remain alternative explanations, so this is not evidence for the proposed shared-embedding mechanism.",
        "",
        "[Certain] The ordering against item-only and random-sharing controls reverses on random exposures. The native positive rate is also much lower there. The fitted models were not recalibrated on random traffic; policy/exposure differences and calibration are confounded with any proposed context mechanism. A claim that user-specific representations are universally more robust is contradicted by this probe.",
        "",
        "## 6. Baseline artifact availability",
        "",
        "| Author repository | Read-only revision pin | Executed in this probe? |",
        "|---|---|---|",
    ])
    for record in availability:
        text.append(f"| {record['repository']} | {record.get('revision', 'unavailable')} | No |")
    text.extend([
        "",
        "[Certain] GraphHash declares an older pinned GPU environment; CCE provides a compressed-embedding interface with clustering calls; BACO documents a Cython compilation step and preprocessed-data workflow. Their README/manifests were saved as source receipts. Availability is not executable compatibility with this feedback task. Task adapters, environment validation and a shared neural backbone remain unfinished.",
        "",
        "## 7. What this proves, and what remains open",
        "",
        "- [Certain] Executed: official-release checksum verification, real-data support/integrity audit, chronological cleaned-cohort construction, validation-selected count-baseline evaluation, saved-prediction checks, and read-only author-code availability checks.",
        "- [Certain] Not executed: a learned embedding model, GraphHash/CCE/BACO reproduction, behavior-aware sharing method, equal-total-byte comparison, maintenance-cost benchmark, pure context-frequency intervention, or independent-dataset confirmation.",
        "- [Likely] Next experiment: a small common-backbone study on the training-defined supported cohort, comparing an uncompressed user representation, fixed/random sharing, an existing dynamic comparator, and conditional-count grouping before adding a proposed mechanism.",
        "- [Likely] Do not use random-traffic rank reversal as evidence of the proposed mechanism. First separate calibration/exposure effects; keep future confirmation independent of these now-consumed exploratory evaluation sets.",
        "- [Certain] This probe does not establish novelty, causal preference changes, user welfare, or main-track readiness. It resolves part of the data/execution uncertainty while exposing real limitations.",
        "",
        "## 8. Reproduction and verification",
        "",
        "Run with the existing recorded Python environment from this directory:",
        "",
        "```powershell",
        "python -X utf8 .\\fetch_release.py",
        "python -X utf8 .\\audit_support.py",
        "python -X utf8 .\\count_baselines.py",
        "python -X utf8 .\\verify_probe.py",
        "python -X utf8 .\\make_report.py",
        "```",
        "",
        f"[Certain] Saved-artifact verification status: **{verify['status']}**. Maximum metric recomputation error: {verify['max_metric_recomputation_error']:.3g}; maximum saved-model replay error: {verify['max_saved_model_replay_error']:.3g}. The documented target-rule consistency check remains failed and is explicitly preserved in verification.json.",
        "",
        "[Certain] Original records and source hashes are preserved. Scripts are a throwaway research-probe harness, not production software. No paper result should be copied without retaining the split, cohort, target, amendments, selection and uncertainty qualifications above.",
    ])
    (ROOT / "AUDIT_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    summary = {
        "status": "local_exploratory_probe_executed_and_artifacts_verified",
        "verdict": "narrow_same_policy_component_signal; context-robust-compression hypothesis not validated",
        "raw_rows": audit["raw_total_rows"], "clean_rows": audit["total_rows"],
        "quarantined_rows": audit["duplicate_key_quarantine"]["rows"],
        "native_label_documentation_mismatches": mismatch,
        "two_context_users_min10": support["10"]["users_with_two_supported_contexts"],
        "random_test_dominant_context_fraction": random_dominant,
        "elapsed_seconds": {"download": release["download_seconds"], "successful_support_audit": audit["elapsed_seconds"], "count_baselines": baseline["elapsed_seconds"]},
        "workspace_bytes_at_report": sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()),
        "report": str(ROOT / "AUDIT_REPORT.md"),
    }
    (ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
