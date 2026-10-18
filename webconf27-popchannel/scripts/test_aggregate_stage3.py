"""Regression tests for Stage 3's refresh of the Stage 1-2 aggregation."""
import copy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aggregate as agg
from test_t5 import make_run, write_controls, controls_doc


def test_nested_session_controls_discovered_once(tmp_path):
    runs = tmp_path / "runs"
    target = runs / "_sessions" / "pc-x1-1017" / "controls" / "gowalla__mf__graphhash__directau__s0-legacy30"
    write_controls(target)
    rows, warnings = agg.all_controls_rows(runs, target.parent)
    assert len(rows) == len(agg.controls_cells(controls_doc()))
    assert {r["session_slug"] for r in rows} == {"pc-x1-1017"}
    assert warnings == []


def test_new_run_and_colliding_legacy_both_keep_lineage(tmp_path, monkeypatch):
    rid = "gowalla__mf__graphhash__directau__s0"
    make_run(tmp_path / "runs", name=rid)
    legacy = agg.blank_row()
    legacy.update(run_id=rid, dataset="gowalla", compression="graphhash", loss="directau", seed=0,
                  init="xavier", scoring_mode="raw", test_recall20=7.1, epochs_run=30,
                  stop_reason="fixed_epochs", comparison_type="reproduction", status="valid-short")
    untouched = copy.deepcopy(legacy)
    monkeypatch.setattr(agg, "legacy_rows", lambda *a, **kw: ([legacy], []))
    rows, warnings = agg.build_rows(tmp_path / "runs", tmp_path / "controls", tmp_path / "results")
    keys = [(r["run_id"], r["scoring_mode"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert next(r for r in rows if r["run_id"] == rid + "-legacy30")["test_recall20"] == 7.1
    assert next(r for r in rows if r["run_id"] == rid and r["scoring_mode"] == "raw")["test_recall20"] != 7.1
    assert legacy == untouched  # standalone legacy_rows contract stays unchanged
    assert any("legacy30" in w and "lineage" in w for w in warnings)


def test_legacy_controls_inherit_capped_source_metadata(tmp_path):
    rid = "gowalla__mf__full__directau__s0"
    target = tmp_path / "runs" / "_sessions" / "pc-x1-1017" / "controls" / (rid + "-legacy120")
    write_controls(target, controls_doc(rid + "-legacy120", e2e=True, method="Full"))
    rows, warnings = agg.all_controls_rows(tmp_path / "runs", run_info={rid: ("capped", "reproduction")})
    assert rows and {r["status"] for r in rows} == {"capped"}
    assert {r["comparison_type"] for r in rows} == {"reproduction"}
    assert warnings == []


def test_recursive_controls_still_flag_invalid_documents(tmp_path):
    target = tmp_path / "runs" / "_sessions" / "pc-x1-1017" / "controls" / "invalid"
    doc = controls_doc()
    del doc["tie_stats"]
    write_controls(target, doc)
    rows, warnings = agg.all_controls_rows(tmp_path / "runs")
    assert rows and {r["status"] for r in rows} == {"invalid"}
    assert any("tie_stats" in w for w in warnings)


def test_results_renderer_rounds_source_values_and_preserves_other_sections(tmp_path):
    import render_stage12_tables as render
    rid = "gowalla__mf__full__directau__s0"
    common = {**agg.blank_row(), "run_id": rid, "dataset": "gowalla", "compression": "full",
              "loss": "directau", "seed": 0, "status": "capped", "comparison_type": "reproduction",
              "test_gap": 4.89914, "rho_item": .814957, "stop_reason": "time_cap"}
    rows = [{**common, "scoring_mode": "raw", "test_recall20": 17.895065, "test_ndcg20": 11.52},
            {**common, "scoring_mode": "cosine", "test_recall20": 12.995925, "test_ndcg20": 7.44}]
    table = render.render_sections(rows)
    assert "13.00 / 7.44" in table and "+4.90" in table
    assert "initialization sensitivity" in table and "A2" in table
    original = b"# preface\r\n## 0. Board\r\nkeep before\r\n## 1. old\r\nold 1\r\n## 2. old\r\nold 2\r\n## 3. untouched\r\nkeep after\r\n"
    updated = render.replace_sections(original, table)
    assert updated.split(b"## 1.", 1)[0] == original.split(b"## 1.", 1)[0]
    assert updated[updated.index(b"## 3."):] == original[original.index(b"## 3."):]
    assert b"## 1. Headline" in updated


def test_combined_ledger_popularity_baselines_have_explicit_comparison_type(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "A6_poponly_Yelp2018.json").write_text(json.dumps({
        "dataset": "Yelp2018", "val": {"recall": 1.61, "ndcg": .97},
        "test": {"recall": 1.68, "ndcg": 1.05}}))
    rows = agg.poponly_rows(tmp_path / "results")
    assert rows and {r["comparison_type"] for r in rows} == {"reproduction"}
