"""One-seed, no-grid GraphHash reproduction. Private runner; deliver two files."""
import ast
import csv
import gc
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
import requests
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from repro_core import metrics_from_topk, pair_diagnostics, rank_topk, split_statistics, expired

COMMIT = "7a1d5eb48dce9ab4c4a453d4482dd0d3bdec4467"
SEED = 0
ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).parent
WORK = ROOT / "_work"
WORK.mkdir(exist_ok=True)
ROWS, DATA_INFO = [], {}
SESSION_START = time.perf_counter()
NOTES = [
    "One seed (0), no grid and no loss fixes. Author-published per-condition command settings.",
    "Primary Recall follows released quality_metrics: hits/min(20, positives); standard Recall is separate.",
    "Model selection: validation raw-dot Recall@20, every 10 epochs; 50-epoch patience, maximum 1000 epochs.",
    "Diagnostic trajectories use one fixed 1024-positive-edge probe per validation snapshot; gradient norms are diagnostic objective gradients, including for BPR.",
    "Raw-dot and cosine final metrics use the same validation-selected checkpoint; cosine does not select checkpoints.",
    "CUDA matrix precision float32, TF32 disabled. Evaluation chunking is reduced for memory and vectorized for speed.",
    "Sonatype authentication returned HTTP 401; user explicitly authorized the pinned clustering package after PyPI hash verification.",
]
CONFIGS = {
    ("Full", "BPR"): (.005, 1e-8),
    ("Random", "BPR"): (.001, 1e-6),
    ("GraphHash", "BPR"): (.01, 1e-6),
    ("Full", "DirectAU"): (.01, 1e-8),
    ("Random", "DirectAU"): (.01, 1e-6),
    ("GraphHash", "DirectAU"): (.01, 1e-8),
}


def settings_for(dataset, method, loss):
    lr, wd = CONFIGS[(method, loss)]
    if dataset == "AmazonBook":
        lr, wd = (.01, 1e-6) if loss == "BPR" else (.001, 1e-8)
    batch = 1024 if loss == "DirectAU" else (2984108 if dataset == "AmazonBook" else 1027370)
    return lr, wd, batch, 1.


def mark_completed_before_deadline(row, deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("90-minute per-run cap reached during final diagnostics")
    row["status"] = "complete"


def log(event, **values):
    print(json.dumps({"event": event, **values}, default=str), flush=True)


def fetch(url, timeout=90, tries=6):
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == tries-1:
                raise
            time.sleep(20*(attempt+1))


def ensure_clustering():
    # Reuse an earlier in-session source rebuild instead of rebuilding concurrently.
    vendor_dir = WORK / "sknetwork_vendor"
    if vendor_dir.exists() and any(vendor_dir.rglob("louvain_core*.so")) and str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
        importlib.invalidate_caches()
    try:
        import sknetwork
        if sknetwork.__version__ == "0.32.1":
            return
    except ImportError:
        pass
    # User authorized this exact package, no other dependency changes.
    metadata = requests.get("https://pypi.org/pypi/scikit-network/0.32.1/json", timeout=30).json()
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels = [x for x in metadata["urls"] if tag in x["filename"] and
              "manylinux" in x["filename"] and "x86_64" in x["filename"]]
    for package in wheels[:1]:
        path = WORK / package["filename"]
        payload = fetch(package["url"])
        assert hashlib.sha256(payload).hexdigest() == package["digests"]["sha256"]
        path.write_bytes(payload)
        p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                            "--no-build-isolation", "--force-reinstall", str(path)],
                           capture_output=True, text=True, timeout=300)
        (WORK / "clustering_install.log").write_text(p.stdout + p.stderr)
        if p.returncode == 0:
            # Import in a fresh process first, so a failed binary import does not poison this one.
            check = subprocess.run([sys.executable, "-c",
                    "import sknetwork; assert sknetwork.__version__=='0.32.1'"],
                    capture_output=True, text=True, timeout=30)
            if check.returncode == 0:
                importlib.invalidate_caches()
                log("clustering_ready", version="0.32.1", artifact=package["filename"])
                return
    # Release 0.32.1 has wheels but no sdist. Its official wheels include the
    # original .pyx sources; rebuild those unchanged for Kaggle Python 3.13/NumPy 2.
    if importlib.util.find_spec("Cython") is None:
        raise RuntimeError("Pinned package needs a source rebuild, but preinstalled Cython is unavailable.")
    package = next(x for x in metadata["urls"] if
                   "manylinux" in x["filename"] and "x86_64" in x["filename"])
    payload = fetch(package["url"])
    assert hashlib.sha256(payload).hexdigest() == package["digests"]["sha256"]
    vendor = WORK / "sknetwork_vendor"
    vendor.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert any(n.endswith("clustering/louvain_core.pyx") for n in z.namelist())
        z.extractall(vendor)
    setup = (
        "from pathlib import Path\n"
        "from setuptools import setup,Extension\n"
        "from Cython.Build import cythonize\n"
        "import numpy as np\n"
        "ext=[Extension('.'.join(p.with_suffix('').parts),[str(p)],language='c++',"
        "include_dirs=[np.get_include()]) for p in Path('sknetwork').rglob('*.pyx')]\n"
        "setup(name='scikit-network',version='0.32.1',ext_modules=cythonize("
        "ext,compiler_directives={'language_level':3},nthreads=2))\n"
    )
    (vendor / "setup_rebuild.py").write_text(setup)
    p = subprocess.run([sys.executable, "setup_rebuild.py", "build_ext", "--inplace", "-j", "2"],
                       cwd=vendor, capture_output=True, text=True, timeout=300)
    (WORK/"clustering_rebuild.log").write_text(p.stdout+p.stderr)
    if p.returncode:
        raise RuntimeError("Pinned clustering source rebuild failed; see internal rebuild log.")
    sys.path.insert(0, str(vendor))
    importlib.invalidate_caches()
    import sknetwork
    assert sknetwork.__version__ == "0.32.1"
    NOTES.append("scikit-network 0.32.1 rebuilt from unchanged .pyx sources bundled in its SHA256-verified official PyPI wheel, using preinstalled Cython/NumPy.")
    log("clustering_ready", version="0.32.1", artifact=package["filename"], mode="source_rebuild")


def prepare():
    ensure_clustering()
    repo = WORK / "GraphHash"
    if not repo.exists():
        payload = fetch(f"https://codeload.github.com/snap-research/GraphHash/zip/{COMMIT}")
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            z.extractall(WORK)
        (WORK / f"GraphHash-{COMMIT}").rename(repo)
    archive = WORK / "datasets.zip"
    if not archive.exists():
        payload = fetch("https://drive.usercontent.google.com/download?id=1vcyDljpxG3-TJdzgbhVdoWIK-eA_wIF-&export=download&confirm=t")
        tmp = archive.with_suffix(f".part{os.getpid()}")
        tmp.write_bytes(payload)
        os.replace(tmp, archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == "b6098e3b3e5d46e5e4313490b02904c365c1a7d270c73e6bd6ecf1af89e894a8"
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if any(name == f"datasets/{ds}-processed/{split}.pt"
                   for ds in ["Gowalla", "AmazonBook", "Yelp2018"] for split in ["train", "val", "test"]):
                target = WORK / name
                if not (target.exists() and target.stat().st_size == z.getinfo(name).file_size):
                    z.extract(name, WORK)
    sys.path.insert(0, str(repo))
    from models.mf import MF
    from models.mfdau import MFDAU
    from loss import BPRLoss, DAULoss
    from sknetwork.clustering import Louvain
    # Execute the pinned author's two mapping functions without importing unrelated optional libraries.
    tree = ast.parse((repo / "utils.py").read_text())
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef)
                and x.name in {"graphClustering", "map_to_consecutive_integers"}]
    namespace = {"np": np, "Louvain": Louvain}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "author_mapping_functions", "exec"), namespace)
    return MF, MFDAU, BPRLoss, DAULoss, namespace


def load_data(dataset, namespace, resolution=200):
    path = WORK / "datasets" / f"{dataset}-processed"
    parts = [torch.load(path / f"{s}.pt", weights_only=True).long() for s in ["train", "val", "test"]]
    info = split_statistics(parts)
    nu, ni = info["user_id_max"]+1, info["item_id_max"]+1
    assert nu == info["users"] and ni == info["items"], "Nonconsecutive IDs require an explicit mapping."
    info["file_sha256"] = {s: hashlib.sha256((path / f"{s}.pt").read_bytes()).hexdigest()
                           for s in ["train", "val", "test"]}
    np.random.seed(SEED)
    graph = csr_matrix((np.ones(parts[0].shape[1]), parts[0].numpy()), shape=(nu, ni))
    u, i, ku, ki = namespace["graphClustering"](graph, resolution=resolution)
    u = namespace["map_to_consecutive_integers"](u)
    i = namespace["map_to_consecutive_integers"](i)
    maps = {"Full": (np.arange(nu), np.arange(ni)),
            "GraphHash": (u, i),
            "Random": (np.arange(nu) % ku, np.arange(ni) % ki)}
    info.update({"user_buckets": ku, "item_buckets": ki,
                 "compressed_parameters": (ku+ki)*64,
                 "full_parameters": (nu+ni)*64})
    dictionaries = []
    for part in parts:
        d = {}
        for uid, iid in part.t().tolist():
            d.setdefault(uid, []).append(iid)
        dictionaries.append(d)
    DATA_INFO[dataset] = info
    log("data_ready", dataset=dataset, **info)
    return parts, maps, dictionaries, info


def evaluate(model, aliases, truth, masks, device, cosine=False, deadline=None):
    keys = list(truth)
    preds = []
    u_map, i_map = aliases
    with torch.no_grad():
        items = model.item_emb_table(i_map)
        for start in range(0, len(keys), 512):
            if deadline and time.perf_counter() >= deadline:
                raise TimeoutError("90-minute per-run cap reached during evaluation")
            users = keys[start:start+512]
            u = model.user_emb_table(u_map[torch.tensor(users, device=device)])
            p = rank_topk(u, items, [masks.get(x, []) for x in users], 20, cosine=cosine)
            preds.append(p.cpu().numpy())
    topk = np.concatenate(preds)
    return metrics_from_topk(topk, [truth[x] for x in keys], 20), topk


def diagnostics(model, aliases, probe, DAULoss):
    uid, iid = probe
    u_map, i_map = aliases
    ua, ia = u_map[uid], i_map[iid]
    u = model.user_emb_table(ua)
    i = model.item_emb_table(ia)
    align = DAULoss.alignment(u, i)
    lu, li = DAULoss.uniformity(u), DAULoss.uniformity(i)
    params = [model.user_emb_table.weight, model.item_emb_table.weight]
    ag = torch.autograd.grad(align, params, retain_graph=True)
    ug = torch.autograd.grad((lu+li)/2, params)
    out = {"alignment": float(align.detach()), "uniformity_user": float(lu.detach()),
           "uniformity_item": float(li.detach())}
    for idx, (side, ids, code, table) in enumerate([
            ("user", uid, ua, params[0]), ("item", iid, ia, params[1])]):
        out.update({f"{side}_{key}": val for key, val in
                    pair_diagnostics(ids, code, table.detach()).items()})
        out[f"{side}_alignment_grad_norm"] = float(ag[idx].norm())
        out[f"{side}_uniformity_grad_norm"] = float(ug[idx].norm())
        amap = u_map if side == "user" else i_map
        out[f"{side}_mean_embedding_norm"] = float(table.detach()[amap].norm(dim=1).mean())
    return out


def write_outputs(status, detail=""):
    base = ["dataset", "method", "loss", "seed", "status", "lr", "weight_decay",
            "batch_size", "gamma", "parameters", "epochs", "best_epoch", "wall_seconds",
            "peak_cuda_mib", "val_raw_recall", "val_raw_ndcg", "test_raw_recall",
            "test_raw_ndcg", "val_cosine_recall", "val_cosine_ndcg",
            "test_cosine_recall", "test_cosine_ndcg"]
    columns = base + sorted(set().union(*(set(x) for x in ROWS))-set(base)) if ROWS else base
    with (ROOT / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(ROWS)
    lines = ["# Loss × embedding sharing: reproduction only", "",
             f"Status: **{status}**. {detail}", "",
             f"Pinned GraphHash commit: {COMMIT}. Seed: {SEED}.",
             f"Runtime: Python {sys.version.split()[0]}, Torch {torch.__version__}; "
             f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}.", "",
             "## Contract", ""] + ["- "+x for x in NOTES]
    lines += ["", "## Loaded author data", "", json.dumps(DATA_INFO, indent=2), "",
              "Gowalla item count in the loaded split is 40,981; the earlier specification listed 49,981. "
              "The loaded 4.534M full parameters agree with Table 1. No resplitting or padding was performed.", "",
              "## Results", "",
              "Metrics below are percentage units. CSV Recall/NDCG values are fractions.", "",
              "| Dataset | Method | Loss | Raw test R@20 | Cosine test R@20 | Raw test NDCG@20 | Minutes | Status |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for r in ROWS:
        fmt = lambda k: f"{100*r[k]:.3f}" if r.get(k) is not None else "NA"
        lines.append(f"| {r['dataset']} | {r['method']} | {r['loss']} | {fmt('test_raw_recall')} | "
                     f"{fmt('test_cosine_recall')} | {fmt('test_raw_ndcg')} | "
                     f"{r.get('wall_seconds', 0)/60:.2f} | {r['status']} |")
    lines += ["", "## Direction check", ""]
    table = {(r["method"], r["loss"]): r for r in ROWS
             if r["dataset"] == "Gowalla" and r["status"] == "complete"}
    all_pass = True
    for method, expected in [("Full", ">"), ("GraphHash", "<"), ("Random", "<")]:
        b, d = table.get((method, "BPR")), table.get((method, "DirectAU"))
        if not b or not d:
            all_pass = False
            lines.append(f"- {method}: incomplete.")
            continue
        bv, dv = b["test_raw_recall"], d["test_raw_recall"]
        passed = dv > bv if expected == ">" else dv < bv
        all_pass &= passed
        lines.append(f"- {method}: DirectAU {dv*100:.3f}% {expected} BPR {bv*100:.3f}%: "
                     f"{'PASS' if passed else 'FAIL'}.")
    lines += ["", "This checks directions for one seed and fixed released configurations. It does not "
              "establish statistical robustness, a collision mechanism, repair effectiveness or novelty.",
              "", "## Diagnostics", "",
              "metrics.csv contains separate repeat-ID and extra-alias fractions, Z, alpha, gradient norms "
              "and mean vector norms for users and items, on the selected checkpoint and as a mean over "
              "validation snapshots. BPR uniformity/alignment gradients are diagnostic gradients, not BPR training terms.",
              "", "## Timing and limits", "",
              f"Elapsed session seconds including setup: {time.perf_counter()-SESSION_START:.1f}. "
              "Each training condition has a 5400-second cap. No tuning or loss fixes were run."]
    (ROOT / "REPRO_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return all_pass


def run_one(dataset, method, loss, data, classes):
    MF, MFDAU, BPRLoss, DAULoss = classes
    parts, maps, (train_truth, val_truth, test_truth), info = data
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    deadline = started+5400
    aliases = tuple(torch.tensor(x, dtype=torch.long, device=device) for x in maps[method])
    cls = MF if loss == "BPR" else MFDAU
    model = cls(int(aliases[0].max())+1, int(aliases[1].max())+1, 64).to(device)
    lr, wd, batch_size, gamma = settings_for(dataset, method, loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = BPRLoss() if loss == "BPR" else DAULoss(gamma)
    edges = parts[0].to(device)
    probe_indices = np.random.RandomState(314159).choice(edges.shape[1], min(1024, edges.shape[1]), replace=False)
    probe = edges[:, torch.tensor(probe_indices, device=device)]
    run_id = f"{dataset}_{method}_{loss}_seed0"
    checkpoint = WORK / (run_id+".pt")
    row = {"dataset": dataset, "method": method, "loss": loss, "seed": SEED,
           "status": "running", "lr": lr, "weight_decay": wd, "batch_size": batch_size,
           "gamma": gamma if loss == "DirectAU" else None,
           "parameters": sum(x.numel() for x in model.parameters())}
    ROWS.append(row)
    write_outputs("running", run_id)
    log("run_started", **row)
    best, best_epoch, snapshots = -math.inf, 0, []
    try:
        for epoch in range(1000):
            if expired(time.perf_counter()-started):
                raise TimeoutError("90-minute training cap reached")
            model.train()
            order = torch.randperm(edges.shape[1], device=device)
            for first in range(0, edges.shape[1], batch_size):
                if expired(time.perf_counter()-started):
                    raise TimeoutError("90-minute training cap reached")
                uid, iid = edges[:, order[first:first+batch_size]]
                ua, ia = aliases[0][uid], aliases[1][iid]
                optimizer.zero_grad(set_to_none=True)
                if loss == "BPR":
                    negatives = torch.randint(len(aliases[1]), iid.shape, device=device)
                    value = criterion(*model(ua, ia, aliases[1][negatives]))
                else:
                    value = criterion(*model(ua, ia))
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite objective")
                value.backward()
                optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                val, _ = evaluate(model, aliases, val_truth, train_truth, device, deadline=deadline)
                diag = diagnostics(model, aliases, probe, DAULoss)
                snapshots.append(diag)
                if val["recall"] > best:
                    best, best_epoch = val["recall"], epoch
                    torch.save(model.state_dict(), checkpoint)
                torch.cuda.synchronize()
                log("validation", run=run_id, epoch=epoch, val_recall=val["recall"],
                    best=best, elapsed_seconds=round(time.perf_counter()-started, 2))
            if epoch-best_epoch >= 50:
                break
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.eval()
        row.update({"epochs": epoch+1, "best_epoch": best_epoch})
        merged = {u: train_truth.get(u, [])+val_truth.get(u, [])
                  for u in set(train_truth)|set(val_truth)}
        for split, truth, mask in [("val", val_truth, train_truth), ("test", test_truth, merged)]:
            for scoring in ["raw", "cosine"]:
                metric, pred = evaluate(model, aliases, truth, mask, device,
                                        cosine=scoring=="cosine", deadline=deadline)
                for key in ["recall", "ndcg", "standard_recall"]:
                    row[f"{split}_{scoring}_{key}"] = metric[key]
                np.savez_compressed(WORK/f"{run_id}_{split}_{scoring}.npz",
                                    users=np.array(list(truth)), top20=pred)
        row.update(diagnostics(model, aliases, probe, DAULoss))
        for key in snapshots[0]:
            vals = [x[key] for x in snapshots if x[key] is not None]
            row["trajectory_mean_"+key] = float(np.mean(vals)) if vals else None
        mark_completed_before_deadline(row, deadline)
    except Exception as exc:
        row["status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
        row["error"] = str(exc)
        raise
    finally:
        torch.cuda.synchronize()
        row["wall_seconds"] = time.perf_counter()-started
        row["peak_cuda_mib"] = torch.cuda.max_memory_allocated()/1024**2
        write_outputs(row["status"], run_id)
        log("run_finished", **row)
        del model, optimizer, edges, aliases
        gc.collect()
        torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; CPU training not launched.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    MF, MFDAU, BPRLoss, DAULoss, namespace = prepare()
    classes = MF, MFDAU, BPRLoss, DAULoss
    data = load_data("Gowalla", namespace)
    # First run supplies the measured runtime gate; no other training starts beforehand.
    run_one("Gowalla", "Full", "DirectAU", data, classes)
    if ROWS[0]["wall_seconds"] > 5400:
        raise TimeoutError("First run exceeded 90 minutes; remaining conditions not launched")
    for method, loss in [("Full", "BPR"), ("Random", "BPR"), ("Random", "DirectAU"),
                         ("GraphHash", "BPR"), ("GraphHash", "DirectAU")]:
        run_one("Gowalla", method, loss, data, classes)
    passed = write_outputs("primary_complete")
    # Boundary case is optional; avoid extending a slow primary pilot.
    elapsed = time.perf_counter()-SESSION_START
    if passed and elapsed < 3600:
        log("boundary_case_started", elapsed_seconds=elapsed)
        boundary = load_data("AmazonBook", namespace)
        run_one("AmazonBook", "GraphHash", "BPR", boundary, classes)
        run_one("AmazonBook", "GraphHash", "DirectAU", boundary, classes)
    else:
        NOTES.append("Optional AmazonBook skipped: primary directions did not all pass, or primary session exceeded one hour.")
    write_outputs("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (WORK/"failure.log").write_text(traceback.format_exc())
        write_outputs("stopped", f"{type(exc).__name__}: {exc}")
        log("stopped", error_type=type(exc).__name__, message=str(exc))
        raise
