#!/usr/bin/env python
"""Strict validation of one run directory (PLAN section 8, TASKS rule 6).

Usage
    python validate_run.py RUN_DIR [--controls]

Prints one JSON object and exits 0 if and only if ``ok`` is true.

    {"run_dir", "run_id", "kind" ("run" | "controls"), "ok", "problems": [...],
     "warnings": [...], "stop_reason", "converged", "exit_code", "seed", "modes"}

``converged`` is ``stop_reason == "patience"`` and nothing else, so a run that hit a time cap or
max_epochs can never pass a convergence gate. It does NOT include ``ok``: a gate must test both.

A run directory (written by the patched runner) is valid only if ALL of these hold:
  1. exit_status.json exists and ``exit_code`` == 0;
  2. metrics.json exists, ``"status" == "complete"`` and
     ``seeds[<seed>]["eval"]`` has every mode raw, cosine, cos_prior, raw_prior (plus cos_b and
     cos_b_prior if the loss is dau_bias), each with best_beta, val, test and val_by_beta.
     Extra consistency checks: val/test have finite recall+ndcg in [0, 1]; ``val`` equals
     ``val_by_beta[best_beta]``; best_beta is the validation argmax (rule 2: validation-only selection);
  3. best_seed{N}.pt, mapping_users.npy and mapping_items.npy exist, are non-empty, and the .npy load;
  4. the sha256 of the mapping arrays matches config.json (see ``mapping_hash_candidates``);
  5. stop_reason is set (exit_status.json, else metrics seed ``stop_reason``/``stopped``).
Missing train_log.csv / norm_log.csv / per_user_test__{mode}.npz / config device_count are warnings only.

Interface with the patched runner (exp_run.py, T1) -- read from its code, tolerant of small variations:
  * exit_status.json: {"exit_code", "stop_reason", "end_time", "error"}; the stop reason may also be in
    metrics ``seeds[seed]["stop_reason"|"stopped"]`` (a conflict between the two is a problem);
  * config.json: ``args.seeds`` = [seed], ``mapping_sha256`` = {"users": hex, "items": hex} where hex is the
    sha256 of the .npy FILE bytes. Also accepted: the sha256 of the array bytes (``np.ascontiguousarray(a)
    .tobytes()``, native or int64), flat keys ``mapping_users_sha256`` / ``mapping_items_sha256``, or one string
    for both (a single mapping hash, or the sha256 of users+items concatenated).

A controls directory (``controls.json`` written by score_controls.py, no metrics.json) is ok if controls.json
is a JSON object with ``status == "complete"``, has the sections in ``CONTROL_SECTIONS`` (plus ``e2e`` when it was
requested), its alias assertion (if any) passed, and the metric cross-check against repro_core is exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REQUIRED_MODES = ("raw", "cosine", "cos_prior", "raw_prior")
BIAS_MODES = ("cos_b", "cos_b_prior")
BIAS_LOSSES = ("dau_bias", "directau_bias")
EVAL_KEYS = ("best_beta", "val", "test", "val_by_beta")
CAPPED_REASONS = ("time_cap", "max_epochs")  # status "capped" in aggregate.py
UNSET_STOP = {"", "none", "null", "unknown", "unset", "n/a"}

# Top-level sections score_controls.py always writes (schema "score_controls/1"). ``e2e`` is added with --e2e
# (Full models) and ``{cos,raw}_prior_u8`` carries the uint8 prior check (E7).
CONTROL_SECTIONS = ("legacy_topk", "tie_keys", "full_prior", "decomposition_fixed_beta", "decomposition_retuned",
                    "within_bucket_shuffle", "cos_prior", "raw_prior", "tie_stats")
CROSSCHECK_TOL = 1e-6


# ----------------------------------------------------------------------------- small helpers
def load_json(path: Path):
    """Return (object, error_string). Never raises."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{Path(path).name} is missing"
    except (OSError, ValueError) as exc:
        return None, f"{Path(path).name} is unreadable: {exc}"


def _walk(obj, path=(), depth=0, max_depth=4):
    """Yield (path, value) for every entry of nested dicts, to a bounded depth."""
    if isinstance(obj, dict) and depth <= max_depth:
        for key, value in obj.items():
            yield path + (str(key),), value
            yield from _walk(value, path + (str(key),), depth + 1, max_depth)


def find_key(obj, names, max_depth=3):
    """First value whose key is in ``names`` (searched breadth-first-ish through nested dicts)."""
    names = {n.lower() for n in names}
    if not isinstance(obj, dict):
        return None
    for key, value in obj.items():
        if str(key).lower() in names and value is not None:
            return value
    for path, value in _walk(obj, max_depth=max_depth):
        if path[-1].lower() in names and value is not None:
            return value
    return None


def _is_hex64(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def parse_run_id(run_id: str) -> dict:
    """``[x__]{dataset}__{backbone}__{compression}__{loss}__s{seed}[-suffix...]`` -> parts (None if malformed)."""
    parts = str(run_id).split("__")
    exploratory = False
    if parts and parts[0] == "x":
        exploratory, parts = True, parts[1:]
    if len(parts) != 5:
        return {"run_id": run_id, "valid": False, "exploratory": exploratory}
    match = re.fullmatch(r"s(\d+)((?:-[A-Za-z0-9._]+)*)", parts[4])
    if not match:
        return {"run_id": run_id, "valid": False, "exploratory": exploratory}
    suffix = match.group(2)
    res = re.search(r"-r(\d+)", suffix)
    return {"run_id": run_id, "valid": True, "exploratory": exploratory, "dataset": parts[0],
            "backbone": parts[1], "compression": parts[2], "loss": parts[3], "seed": int(match.group(1)),
            "suffix": suffix, "init": "initmatch" if "-initmatch" in suffix else "xavier",
            "resolution": int(res.group(1)) if res else None}


# ----------------------------------------------------------------------------- mapping hashes
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mapping_hash_candidates(path: Path) -> dict:
    """{kind: hex} for one mapping .npy: sha256 of the file, of the array bytes, of the int64 array bytes."""
    import numpy as np
    raw = Path(path).read_bytes()
    arr = np.load(path, allow_pickle=False)
    return {"file": _sha(raw),
            "array": _sha(np.ascontiguousarray(arr).tobytes()),
            "array_int64": _sha(np.ascontiguousarray(arr.astype(np.int64)).tobytes())}


def declared_mapping_hashes(cfg) -> dict:
    """Mapping hashes found in config.json: {"users": hex|None, "items": hex|None, "single": hex|None}."""
    out = {"users": None, "items": None, "single": None}
    if not isinstance(cfg, dict):
        return out
    for path, value in _walk(cfg):
        joined = "/".join(path).lower()
        if not _is_hex64(value) or "map" not in joined or not ("sha" in joined or "hash" in joined):
            continue
        slot = "users" if "user" in joined else "items" if "item" in joined else "single"
        out[slot] = out[slot] or value.lower()
    return out


def check_mapping_hashes(run_dir: Path, cfg, problems: list) -> None:
    decl = declared_mapping_hashes(cfg)
    if not any(decl.values()):
        problems.append("config.json has no mapping sha256")
        return
    users, items = run_dir / "mapping_users.npy", run_dir / "mapping_items.npy"
    try:
        cu, ci = mapping_hash_candidates(users), mapping_hash_candidates(items)
    except Exception as exc:  # unreadable arrays are reported by the existence checks too
        problems.append(f"cannot hash mapping arrays: {exc}")
        return
    if decl["users"] or decl["items"]:
        for name, want, cand in (("users", decl["users"], cu), ("items", decl["items"], ci)):
            if not want:
                problems.append(f"config.json has no mapping sha256 for {name}")
            elif want not in cand.values():
                problems.append(f"mapping_{name}.npy sha256 does not match config.json")
        return
    want = decl["single"]
    combined = {_sha(users.read_bytes() + items.read_bytes())}
    import numpy as np
    ua, ia = np.load(users, allow_pickle=False), np.load(items, allow_pickle=False)
    combined.add(_sha(np.ascontiguousarray(ua).tobytes() + np.ascontiguousarray(ia).tobytes()))
    combined.add(_sha(np.ascontiguousarray(ua.astype(np.int64)).tobytes()
                      + np.ascontiguousarray(ia.astype(np.int64)).tobytes()))
    if want not in set(cu.values()) | set(ci.values()) | combined:
        problems.append("mapping sha256 does not match config.json")


# ----------------------------------------------------------------------------- eval block
def validate_eval(eval_block, required_modes=REQUIRED_MODES) -> list:
    """Problems with an ``eval`` dict {mode: {best_beta, val, test, val_by_beta}} (empty list = fine)."""
    problems = []
    if not isinstance(eval_block, dict):
        return ["eval block is missing"]
    for mode in required_modes:
        block = eval_block.get(mode)
        if not isinstance(block, dict):
            problems.append(f"eval mode '{mode}' is missing")
            continue
        missing = [k for k in EVAL_KEYS if block.get(k) is None]
        if missing:
            problems.append(f"eval mode '{mode}' lacks {', '.join(missing)}")
            continue
        for split in ("val", "test"):
            cell = block[split]
            if not isinstance(cell, dict) or not all(_finite(cell.get(m)) and 0.0 <= cell[m] <= 1.0
                                                    for m in ("recall", "ndcg")):
                problems.append(f"eval mode '{mode}': {split} recall/ndcg are not finite values in [0, 1]")
        by_beta = block["val_by_beta"]
        if not isinstance(by_beta, dict) or not by_beta:
            problems.append(f"eval mode '{mode}': val_by_beta is empty")
            continue
        try:
            grid = {float(k): v for k, v in by_beta.items()}
            best = float(block["best_beta"])
        except (TypeError, ValueError):
            problems.append(f"eval mode '{mode}': best_beta / val_by_beta keys are not numeric")
            continue
        if best not in grid:
            problems.append(f"eval mode '{mode}': best_beta {best} is not in val_by_beta")
            continue
        try:
            at_best = grid[best]["recall"]
            if abs(at_best - block["val"]["recall"]) > 1e-9:
                problems.append(f"eval mode '{mode}': val does not equal val_by_beta[best_beta]")
            if any(v["recall"] > at_best + 1e-12 for v in grid.values()):
                problems.append(f"eval mode '{mode}': best_beta is not the validation argmax")
        except (KeyError, TypeError):
            problems.append(f"eval mode '{mode}': val_by_beta entries lack recall")
    return problems


# ----------------------------------------------------------------------------- stop reason / exit code / seed
def _clean_stop(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in UNSET_STOP else text


def _as_seed(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    try:
        return None if value is None or isinstance(value, bool) else int(str(value).split(",")[0])
    except ValueError:
        return None


def config_seed(cfg):
    """Seed written by the runner: top level or ``args`` (``seed`` or a one-element ``seeds``)."""
    if not isinstance(cfg, dict):
        return None
    for scope in (cfg, cfg.get("args") if isinstance(cfg.get("args"), dict) else {}):
        for key in ("seed", "seeds"):
            seed = _as_seed(scope.get(key))
            if seed is not None:
                return seed
    return None


def resolve_seed(run_dir: Path, cfg, metrics):
    """Seed of the run: config.json, else the run id (``..__s{N}``), else the only key of metrics['seeds']."""
    seed = config_seed(cfg)
    if seed is not None:
        return seed
    parsed = parse_run_id(Path(run_dir).name)
    if parsed.get("valid"):
        return parsed["seed"]
    if isinstance(metrics, dict) and isinstance(metrics.get("seeds"), dict) and len(metrics["seeds"]) == 1:
        return int(next(iter(metrics["seeds"])))
    return None


def resolve_loss(run_dir: Path, cfg) -> str:
    parsed = parse_run_id(Path(run_dir).name)
    loss = parsed.get("loss") if parsed.get("valid") else None
    return str(loss or find_key(cfg, ("loss",)) or "").lower()


# ----------------------------------------------------------------------------- run directory
def validate_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    problems, warnings = [], []
    result = {"run_dir": str(run_dir), "run_id": run_dir.name, "kind": "run", "ok": False, "problems": problems,
              "warnings": warnings, "stop_reason": None, "converged": False, "exit_code": None,
              "seed": None, "modes": []}
    if not run_dir.is_dir():
        problems.append("run directory does not exist")
        return result

    cfg, err = load_json(run_dir / "config.json")
    if err:
        problems.append(err)
    elif not isinstance(cfg, dict):
        problems.append("config.json is not a JSON object")
        cfg = None
    exit_status, err = load_json(run_dir / "exit_status.json")
    if err:
        problems.append(err)
    metrics, err = load_json(run_dir / "metrics.json")
    if err:
        problems.append(err)
    elif not isinstance(metrics, dict):
        problems.append("metrics.json is not a JSON object")
        metrics = None

    # 1. exit code
    if isinstance(exit_status, dict):
        code = find_key(exit_status, ("exit_code", "returncode", "exit"))
        result["exit_code"] = code
        if code is None:
            problems.append("exit_status.json has no exit_code")
        elif isinstance(code, bool) or not isinstance(code, int) or code != 0:
            problems.append(f"exit_code is {code!r}, not 0")

    seed = resolve_seed(run_dir, cfg, metrics)
    result["seed"] = seed
    parsed_id = parse_run_id(run_dir.name)
    if parsed_id.get("valid") and config_seed(cfg) is not None and parsed_id["seed"] != config_seed(cfg):
        problems.append(f"seed in the run id ({parsed_id['seed']}) differs from config.json ({config_seed(cfg)})")
    loss = resolve_loss(run_dir, cfg)
    row = None

    # 2. metrics.json
    if metrics is not None:
        if metrics.get("status") != "complete":
            problems.append(f"metrics.json status is {metrics.get('status')!r}, not 'complete'")
        seeds = metrics.get("seeds")
        if seed is None:
            problems.append("cannot determine the seed")
        elif not isinstance(seeds, dict) or str(seed) not in seeds:
            problems.append(f"metrics.json has no seeds['{seed}']")
        else:
            row = seeds[str(seed)]
            required = REQUIRED_MODES + (BIAS_MODES if loss in BIAS_LOSSES else ())
            eval_block = row.get("eval") if isinstance(row, dict) else None
            if isinstance(eval_block, dict):  # bias modes present for another loss are checked too
                required += tuple(m for m in BIAS_MODES if m in eval_block and m not in required)
            problems.extend(validate_eval(eval_block, required))
            if isinstance(row, dict) and isinstance(row.get("eval"), dict):
                result["modes"] = list(row["eval"])

    # 3. checkpoint + mappings
    mapping_ok = True
    if seed is None:
        problems.append("checkpoint cannot be checked without a seed")
    else:
        checkpoint = run_dir / f"best_seed{seed}.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            problems.append(f"{checkpoint.name} is missing or empty")
    for name in ("mapping_users.npy", "mapping_items.npy"):
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{name} is missing or empty")
            mapping_ok = False
    if mapping_ok:
        try:
            import numpy as np
            for name in ("mapping_users.npy", "mapping_items.npy"):
                np.load(run_dir / name, allow_pickle=False)
        except Exception as exc:
            problems.append(f"mapping array is unreadable: {exc}")
            mapping_ok = False

    # 4. sha256 of the mapping arrays vs config.json
    if cfg is not None and mapping_ok:
        check_mapping_hashes(run_dir, cfg, problems)

    # 5. stop reason
    candidates = []
    if isinstance(exit_status, dict):
        candidates.append(("exit_status.json", _clean_stop(exit_status.get("stop_reason"))))
    if isinstance(row, dict):
        candidates.append(("metrics.json", _clean_stop(row.get("stop_reason"))))
        candidates.append(("metrics.json", _clean_stop(row.get("stopped"))))
    if metrics is not None:
        candidates.append(("metrics.json", _clean_stop(metrics.get("stop_reason"))))
    found = [(src, val) for src, val in candidates if val]
    if not found:
        problems.append("stop_reason is not set")
    else:
        if len({val for _, val in found}) > 1:
            problems.append("conflicting stop_reason values: " + ", ".join(f"{s}={v}" for s, v in found))
        result["stop_reason"] = found[0][1]
    result["converged"] = result["stop_reason"] == "patience"

    # warnings (not part of PLAN section 8)
    for name in ("train_log.csv", "norm_log.csv"):
        if not (run_dir / name).is_file():
            warnings.append(f"{name} is missing")
    for mode in REQUIRED_MODES:  # the runner writes per-user files for these four modes only
        if not (run_dir / f"per_user_test__{mode}.npz").is_file():
            warnings.append(f"per_user_test__{mode}.npz is missing")
    if isinstance(cfg, dict) and find_key(cfg, ("device_count",)) is None:
        warnings.append("config.json has no device_count")

    result["ok"] = not problems
    return result


# ----------------------------------------------------------------------------- controls directory
def validate_controls(ctrl_dir) -> dict:
    ctrl_dir = Path(ctrl_dir)
    problems, warnings = [], []
    result = {"run_dir": str(ctrl_dir), "run_id": ctrl_dir.name, "kind": "controls", "ok": False,
              "problems": problems, "warnings": warnings, "stop_reason": None, "converged": False,
              "exit_code": None, "seed": None, "modes": []}
    path = ctrl_dir / "controls.json" if ctrl_dir.is_dir() else ctrl_dir
    data, err = load_json(path)
    if err:
        problems.append(err)
    elif not isinstance(data, dict) or not data:
        problems.append("controls.json is not a non-empty JSON object")
    else:
        if data.get("status") not in (None, "complete"):
            problems.append(f"controls.json status is {data.get('status')!r}, not 'complete'")
        for section in CONTROL_SECTIONS:
            if section in data:
                result["modes"].append(section)
            else:
                problems.append(f"controls.json lacks the '{section}' section")
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        if not meta.get("run_id"):
            warnings.append("controls.json has no meta.run_id")
        else:
            result["run_id"] = str(meta["run_id"])
        args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
        if args.get("e2e") and "e2e" not in data:
            problems.append("controls.json lacks the 'e2e' section although --e2e was requested")
        elif "e2e" in data:
            result["modes"].append("e2e")
        alias = meta.get("alias_assert")
        if isinstance(alias, dict) and alias.get("equal") is not True:
            problems.append("alias assertion did not pass (mapping differs from the stored aliases)")
        legacy = data.get("legacy_topk") if isinstance(data.get("legacy_topk"), dict) else {}
        for base, block in legacy.items():
            check = block.get("crosscheck_vs_repro_core_metrics_from_topk") if isinstance(block, dict) else None
            diff = check.get("max_abs_diff") if isinstance(check, dict) else None
            if diff is None:
                warnings.append(f"legacy_topk/{base} has no metric cross-check")
            elif not _finite(diff) or diff > CROSSCHECK_TOL:
                problems.append(f"legacy_topk/{base}: metric cross-check differs from repro_core by {diff}")
    result["ok"] = not problems
    return result


def validate(path, controls: bool = False) -> dict:
    """Dispatch: a directory with controls.json and no metrics.json (or ``controls=True``) is a controls dir."""
    path = Path(path)
    if controls or (path.is_dir() and (path / "controls.json").is_file() and not (path / "metrics.json").is_file()):
        return validate_controls(path)
    return validate_run(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--controls", action="store_true", help="treat RUN_DIR as a score_controls output dir")
    args = parser.parse_args(argv)
    result = validate(args.run_dir, args.controls)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
