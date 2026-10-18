"""Final-checkpoint and precision contracts, without real model scoring."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("final_e7e9_test", Path(__file__).with_name("analyze_e7_e9.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_final_roster_and_seed_zero_policy():
    rows = M.final_modern_specs()
    ids = [r[1] for r in rows]
    assert len(ids) == len(set(ids)) == 11
    assert "gowalla__mf__graphhash__bpr__s0-p500" in ids
    assert "gowalla__mf__graphhash__bpr__s0" not in ids
    assert sum(r[2] == "raw" for r in rows) == 4
    assert sum(r[1].endswith("-initmatch") for r in rows) == 2


def test_seed_specific_checkpoint_never_falls_back_to_zero(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"seeds": {"2": {}}}))
    (tmp_path / "best_seed2.pt").touch()
    (tmp_path / "best_seed0.pt").touch()
    assert M.modern_checkpoint_path(tmp_path).name == "best_seed2.pt"
    (tmp_path / "best_seed2.pt").unlink()
    with pytest.raises(FileNotFoundError):
        M.modern_checkpoint_path(tmp_path)


def test_precision_verdict_cannot_average_away_failure_or_missing():
    good = {"analysis_status": "complete", "val_delta_pp": .01, "test_delta_pp": -.01}
    bad = {"analysis_status": "complete", "val_delta_pp": -.026, "test_delta_pp": .024}
    missing = {"analysis_status": "missing_gpu"}
    assert M.final_precision_verdict([good, bad, missing]) == "FAIL"
    assert M.final_precision_verdict([good, missing]) == "INCONCLUSIVE"
    assert M.final_precision_verdict([good]) == "PASS"
    assert M.final_precision_verdict([]) == "INCONCLUSIVE"
