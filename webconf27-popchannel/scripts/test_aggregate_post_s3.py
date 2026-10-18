"""Post-Stage-3 ledger contracts: distinct metric families and final run selection."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate as agg


def row(rid="gowalla__mf__graphhash__directau__s0", mode="cos_prior", **changes):
    r = agg.blank_row()
    r.update(agg.base_fields(rid))
    r.update(run_id=rid, scoring_mode=mode, beta=.2, test_recall20=13.3,
             test_ndcg20=8.5, status="valid", stop_reason="patience", table_params=742272,
             comparison_type="reproduction")
    r.update(changes)
    return r


def test_v2_is_an_explicit_writer_and_does_not_change_v1(tmp_path):
    assert hasattr(agg, "build_v2_rows"), "v2 metrics need a separate schema and writer"
    base = [row()]
    modern = tmp_path / "runs" / base[0]["run_id"]
    modern.mkdir(parents=True)
    (modern / "metrics.json").write_text("{}")
    enriched, warnings = agg.build_v2_rows(base, tmp_path, modern.parent)
    assert not warnings
    assert enriched[0]["schema_version"] == "master_metrics/2"
    assert enriched[0]["lineage"] == "modern"
    assert enriched[0]["converged"] is True
    assert enriched[0]["metric_family"] == "capped_recall20"
    assert enriched[0]["test_recall20"] == 13.3
    agg.write_csv(base, tmp_path / "v1.csv")
    agg.write_csv_v2(enriched, tmp_path / "v2.csv")
    assert next(csv.reader((tmp_path / "v1.csv").open())) == agg.COLUMNS
    assert next(csv.reader((tmp_path / "v2.csv").open())) == agg.V2_COLUMNS
    assert "lineage" not in base[0]


def test_v2_keeps_weighted_recall_out_of_capped_fields(tmp_path):
    assert hasattr(agg, "build_v2_rows"), "weighted metrics must be joined explicitly"
    r = row()
    (tmp_path / "E9_final.json").write_text(json.dumps({"records": [{
        "run_id": r["run_id"], "scoring_mode": "cos_prior", "original_beta": .2,
        "retuned_beta": .1, "original_beta_test": {"weighted_recall": .09, "standard_recall": .10},
        "retuned_beta_test": {"weighted_recall": .11}, "analysis_status": "complete",
        "source_policy": "same frozen CPU policy"}]}))
    enriched, _ = agg.build_v2_rows([r], tmp_path, tmp_path / "runs")
    v = enriched[0]
    assert v["test_recall20"] == 13.3
    assert v["e9_weighted_recall_original_pct"] == pytest.approx(9.)
    assert v["e9_weighted_recall_retuned_pct"] == pytest.approx(11.)
    assert v["e9_standard_recall_original_pct"] == pytest.approx(10.)
    assert v["e9_beta_retuned"] == .1
    assert "sum_truth_weights" in v["e9_denominator"]


def test_v2_tail_denominators_storage_and_missing_cells(tmp_path):
    assert hasattr(agg, "build_v2_rows"), "tail and quantization families need separate fields"
    r = row()
    (tmp_path / "E5_final.json").write_text(json.dumps({"records": [{
        "run_id": r["run_id"], "scoring_mode": "cos_prior", "tail_primary": .01,
        "tail_standard": .008, "tail_contribution": .004, "catalog_coverage": .20,
        "tail_eligible_users": 5, "n_users": 8, "analysis_status": "complete"}]}))
    (tmp_path / "E7_final.json").write_text(json.dumps({"records": [{
        "run_id": r["run_id"], "scoring_mode": "cos_prior", "val_delta_pp": -.026,
        "test_delta_pp": .025, "pass": False, "uint8_payload_bytes": 40981,
        "uint8_scale_bytes": 4, "analysis_status": "complete"}]}))
    enriched, _ = agg.build_v2_rows([r], tmp_path, tmp_path / "runs")
    v = enriched[0]
    assert (v["e5_tail_primary_pct"], v["e5_tail_standard_pct"], v["e5_tail_contribution_pct"]) == (1., .8, .4)
    assert v["e5_catalog_coverage_pct"] == 20.
    assert v["e7_val_delta_pp"] == -.026
    assert v["e7_pass"] is False
    assert v["e7_uint8_total_bytes"] == 40985
    assert v["e9_weighted_recall_original_pct"] is None
    assert v["e9_analysis_status"] == "not_available"


def test_legacy_analysis_only_row_stays_distinct_and_capped_not_converged(tmp_path):
    assert hasattr(agg, "build_v2_rows"), "legacy-only saved analyses must remain explicit"
    rid = "amazonbook__mf__graphhash__bpr__s0-legacy"
    base = [row("amazonbook__mf__full__bpr__s0", "raw", status="capped", stop_reason="max_epochs")]
    (tmp_path / "E5_final.json").write_text(json.dumps({"records": [{
        "run_id": rid, "scoring_mode": "cosine", "analysis_status": "missing_gpu"}]}))
    enriched, _ = agg.build_v2_rows(base, tmp_path, tmp_path / "runs")
    assert enriched[0]["capped"] is True and enriched[0]["converged"] is False
    legacy = next(r for r in enriched if r["run_id"] == rid)
    assert legacy["lineage"] == "legacy"
    assert legacy["row_kind"] == "analysis_only"
    assert legacy["test_recall20"] is None
    assert legacy["e5_analysis_status"] == "missing_gpu"


def test_final_table_roster_and_seed_summary_use_homogeneous_final_runs():
    import importlib.util
    path = HERE / "render_stage3_tables.py"
    assert path.is_file(), "final tables need an explicit Stage-3 renderer"
    spec = importlib.util.spec_from_file_location("render_stage3", path)
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    assert len(render.MAIN_RUN_IDS) == 24
    assert "gowalla__mf__full__bpr__s0-long" in render.MAIN_RUN_IDS
    assert "gowalla__mf__graphhash__bpr__s0-p500" in render.MAIN_RUN_IDS
    rows = []
    for loss in ("directau", "bpr"):
        for seed in (0, 1, 2):
            rid = f"gowalla__mf__graphhash__{loss}__s{seed}" + ("-p500" if loss == "bpr" and seed == 0 else "")
            for mode in ("raw", "cosine", "cos_prior", "raw_prior"):
                rows.append(row(rid, mode, test_recall20=10+seed, test_ndcg20=5+seed))
    rows.append(row("gowalla__mf__graphhash__bpr__s0", "raw_prior", test_recall20=99))
    text = render.render_sections(rows)
    assert "11.000 ± 1.000" in text
    assert "Final Gowalla" in text
    assert "A2" in text and "I1" in text and "I2" in text
    assert "A2 is still planned" not in text


def test_current_stage3_runs_all_survive_aggregation_and_ids_are_unique():
    rows, _ = agg.build_rows()
    keys = [(r["run_id"], r["scoring_mode"]) for r in rows]
    assert len(keys) == len(set(keys))
    state = json.loads((agg.PROJECT / "results/queue_state_s3.json").read_text())
    required = {rid for session in state["sessions"].values() for rid in session["run_dirs"]}
    assert len(required) == 25
    assert required <= {r["run_id"] for r in rows}


def test_cli_custom_output_keeps_versioned_artifacts_in_that_directory(tmp_path):
    registry = tmp_path / "registry.csv"
    registry.write_text("file,dir,sub,run_id,status,role,parser,session_slug,stop_reason,note\n")
    out = tmp_path / "custom.csv"
    cmd = [sys.executable, str(HERE / "aggregate.py"), "--out", str(out),
           "--runs", str(tmp_path / "runs"), "--controls", str(tmp_path / "controls"),
           "--results", str(tmp_path / "results"), "--registry", str(registry)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "custom_v2.csv").is_file()
    assert (tmp_path / "custom_v2.schema.json").is_file()

