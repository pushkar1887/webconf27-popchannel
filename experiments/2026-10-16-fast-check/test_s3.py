"""Stage 3 safety/contract tests; these never submit a Kaggle job."""
from pathlib import Path
import importlib.util
import shlex
import sys

import pytest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "webconf27-popchannel"
MODULE = HERE / "make_sessions_s3.py"


def load_builder():
    assert MODULE.is_file(), "Stage 3 builder has not been implemented"
    spec = importlib.util.spec_from_file_location("stage3_builder_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def options(job):
    tokens = shlex.split(job["args"])
    result = {}
    for i, value in enumerate(tokens):
        if value.startswith("--"):
            result[value] = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else True
    return result


def all_jobs(sessions):
    return [job for session in sessions for lane in session["gpu_lists"] for job in lane]


def test_builder_exists():
    assert MODULE.is_file(), "Stage 3 builder has not been implemented"


def test_requested_roster_and_unsupported_c3(tmp_path):
    builder = load_builder()
    sessions, report = builder.build_stage3(runs_dir=tmp_path)
    jobs = all_jobs(sessions)
    expected = {
        "gowalla__mf__full__directau__s0", "gowalla__mf__full__bpr__s0-long",
        "gowalla__mf__graphhash__bpr__s0-p500",
        "amazonbook__mf__graphhash__directau__s0", "amazonbook__mf__full__bpr__s0",
        "amazonbook__mf__graphhash__directau__s0-initmatch", "amazonbook__mf__full__directau__s0",
        "gowalla__mf__full__directau__s0-wd0", "gowalla__mf__full__directau__s0-wd1e-4",
    }
    expected.update(f"gowalla__mf__graphhash__{loss}__s{seed}" for loss in ["directau", "bpr"] for seed in [1, 2])
    expected.update(f"{dataset}__mf__{method}__{loss}__s0"
                    for dataset in ["gowalla", "yelp2018", "amazonbook"]
                    for method in ["random", "dfreq"] for loss in ["directau", "bpr"])
    assert {j["run_id"] for j in jobs} == expected
    assert len(jobs) == len(expected) == 25
    assert any(x["id"] == "C3" and "unsupported" in x["reason"].lower() for x in report["omitted"])


def test_four_sessions_two_gpus_and_authorized_accounts(tmp_path):
    sessions, _ = load_builder().build_stage3(runs_dir=tmp_path)
    assert [s["slug"] for s in sessions] == [f"pc-s3{x}-1017" for x in "abcd"]
    assert len(sessions) == 4
    assert sorted(s["account"] for s in sessions) == ["account-3", "account-3", "account-4", "account-4"]
    assert all(len(s["gpu_lists"]) == 2 and all(s["gpu_lists"]) for s in sessions)
    assert all(s["kernel_sources"] == [] for s in sessions)
    assert all(not j.get("inputs") for j in all_jobs(sessions))


def test_run_ids_priorities_and_both_time_budgets(tmp_path):
    sessions, _ = load_builder().build_stage3(runs_dir=tmp_path)
    jobs = all_jobs(sessions)
    assert len({j["run_id"] for j in jobs}) == len(jobs)
    for session in sessions:
        for lane in session["gpu_lists"]:
            assert [j["priority"] for j in lane] == sorted(j["priority"] for j in lane)
            assert sum(j["estimated_minutes"] for j in lane) <= 540
            assert sum(j["budget_minutes"] for j in lane) <= 660
            for job in lane:
                args = options(job)
                assert args["--run-id"] == job["run_id"]
                assert job["script"] == "exp_run.py"
                assert job["budget_minutes"] >= float(args["--cap-min"]) + 5


def test_hyperparameters_match_current_plan_cell_by_cell(tmp_path):
    sessions, _ = load_builder().build_stage3(runs_dir=tmp_path)
    # Parse the independent document, not the builder's own hyperparameter dictionary.
    hparams = {}
    for line in (PROJECT / "PLAN.md").read_text(encoding="utf8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] not in ["Gowalla", "Yelp2018", "AmazonBook"]:
            continue
        if "·" not in cells[2] or "/" not in cells[2]:
            continue
        for loss, cell in [("bpr_author", cells[2]), ("dau", cells[3])]:
            for method, pair in zip(["Full", "GraphHash", "Random", "DoubleFreq"], cell.split("·")):
                hparams[(cells[0], method, loss)] = tuple(map(float, pair.strip().split("/")))
    assert len(hparams) == 24
    d7 = {"gowalla__mf__full__directau__s0-wd0": 0.0,
          "gowalla__mf__full__directau__s0-wd1e-4": 1e-4}
    for job in all_jobs(sessions):
        args = options(job)
        lr, wd = hparams[(args["--dataset"], args["--method"], args["--loss"])]
        assert float(args["--lr"]) == lr
        assert float(args["--wd"]) == d7.get(job["run_id"], wd)
        assert args["--patience"] == ("30" if args["--loss"] == "dau" else "500")
        assert args["--min-delta"] == ("0.0005" if args["--loss"] == "dau" else "0")
        expected_cap = "300" if args["--dataset"] == "AmazonBook" and args["--loss"] == "dau" else "180" if args["--loss"] == "dau" else "90"
        assert args["--cap-min"] == expected_cap


def test_exact_five_amazon_directau_cap_overrides(tmp_path):
    builder = load_builder()
    sessions, report = builder.build_stage3(runs_dir=tmp_path)
    expected = {
        "amazonbook__mf__graphhash__directau__s0",
        "amazonbook__mf__graphhash__directau__s0-initmatch",
        "amazonbook__mf__full__directau__s0",
        "amazonbook__mf__random__directau__s0",
        "amazonbook__mf__dfreq__directau__s0",
    }
    actual = set()
    for job in all_jobs(sessions):
        args = options(job)
        if args["--cap-min"] == "300":
            actual.add(job["run_id"])
            assert args["--epochs"] == "150"
            assert job["training_estimate_minutes"] == 300
            assert job["estimated_minutes"] == 300 + job["overhead_minutes"]
            assert job["budget_minutes"] == 300 + job["overhead_minutes"]
    assert actual == expected
    # Shared defaults must stay unchanged: only these five jobs receive overrides.
    assert "--cap-min 180" in builder.DAU
    assert "--cap-min 90" in builder.BPR
    assert report["estimate_limit_minutes"] == 540
    assert report["budget_limit_minutes"] == 660
    for session in sessions:
        for lane in session["gpu_lists"]:
            assert sum(j["run_id"] in expected for j in lane) <= 1


def test_bpr_final_three_seed_contract_and_colocation(tmp_path):
    builder = load_builder()
    sessions, _ = builder.build_stage3(runs_dir=tmp_path)
    seed_runs = {
        "gowalla__mf__graphhash__bpr__s0-p500",
        "gowalla__mf__graphhash__bpr__s1",
        "gowalla__mf__graphhash__bpr__s2",
    }
    locations = {}
    for s, session in enumerate(sessions):
        for gpu, lane in enumerate(session["gpu_lists"]):
            for job in lane:
                if job["run_id"] in seed_runs:
                    locations[job["run_id"]] = (s, gpu)
                    args = options(job)
                    assert args["--patience"] == "500"
                    assert args["--epochs"] == "3000"
                    assert args["--lr"] == "0.01"
                    assert args["--wd"] == "1e-06"
    assert set(locations) == seed_runs
    # Three BPR caps + reserves fit as a group, so honor the requested co-location.
    assert len(set(locations.values())) == 1
    rerun = next(j for j in all_jobs(sessions) if j["run_id"].endswith("s0-p500"))
    assert rerun["training_estimate_minutes"] == 10


def test_new_seed_zero_directory_is_never_overwritten(tmp_path):
    builder = load_builder()
    run_id = "gowalla__mf__graphhash__bpr__s0-p500"
    (tmp_path / run_id).mkdir()
    sentinel = tmp_path / run_id / "keep.txt"
    sentinel.write_text("preserve new seed zero")
    with pytest.raises(FileExistsError, match=run_id):
        builder.build_stage3(runs_dir=tmp_path)
    assert sentinel.read_text() == "preserve new seed zero"


def test_control_partner_and_long_bpr_contracts(tmp_path):
    sessions, _ = load_builder().build_stage3(runs_dir=tmp_path)
    jobs = {j["run_id"]: options(j) for j in all_jobs(sessions)}
    long_bpr = jobs["gowalla__mf__full__bpr__s0-long"]
    assert long_bpr["--epochs"] == "20000"
    full = jobs["amazonbook__mf__full__directau__s0"]
    init = jobs["amazonbook__mf__graphhash__directau__s0-initmatch"]
    assert init["--init-match"] is True
    assert "--init-match" not in full
    for key in ["--epochs", "--cap-min", "--patience", "--lr", "--wd", "--eval-every", "--min-delta"]:
        assert init[key] == full[key]


def test_real_valid_run_is_skipped_without_modification():
    builder = load_builder()
    run_id = "gowalla__mf__graphhash__directau__s0"
    run_dir = PROJECT / "runs" / run_id
    before = (run_dir / "metrics.json").read_bytes()
    accepted, skipped = builder.filter_existing([{"run_id": run_id}], PROJECT / "runs")
    assert accepted == []
    assert skipped[0]["run_id"] == run_id
    assert (run_dir / "metrics.json").read_bytes() == before


def test_invalid_existing_directory_is_never_overwritten(tmp_path):
    builder = load_builder()
    run_id = "gowalla__mf__full__directau__s0"
    (tmp_path / run_id).mkdir()
    sentinel = tmp_path / run_id / "keep.txt"
    sentinel.write_text("existing data")
    with pytest.raises(FileExistsError, match=run_id):
        builder.build_stage3(runs_dir=tmp_path)
    assert sentinel.read_text() == "existing data"


def test_real_workspace_has_no_queued_run_collisions():
    sessions, report = load_builder().build_stage3()
    assert all(not (PROJECT / "runs" / j["run_id"]).exists() for j in all_jobs(sessions))
    assert report["new_jobs"] == len(all_jobs(sessions))
