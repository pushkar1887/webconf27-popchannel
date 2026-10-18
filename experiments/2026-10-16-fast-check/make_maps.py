"""T4: build and summarize the item/user mappings (runs on the Kaggle GPU image; clustering needs sknetwork).

python make_maps.py --datasets Gowalla,Yelp2018,AmazonBook --out-root /kaggle/working/maps
    [--gowalla-resolutions 400,800,1600,3200] [--assert-aliases NAME=PATH_OR_URL ...]

Maps per dataset: graphhash (r200, from repro_run.load_data), random (load_data), dfreq (authors' double-frequency hash,
GraphHash's bucket counts, seed 0) and, for Gowalla, graphhash at the extra resolutions (graphClustering called
directly with the same seeding as load_data). Files: <dataset>__<compression>[-r<res>]__{users,items}.npy
(dataset and compression in lower case; the r200 default has no suffix).  A reference entry "full" (identity map,
not saved) is added to the summary: eta^2 must be 1 there.

maps_summary.json, per map:
  * user/item table sizes, number of groups, items(users)-per-group mean / median / max / share of items in groups of
    size >= 2. The group is the bucket (1-D) or the unique UNORDERED alias pair (dfreq: v_i is a sum, so (a,b) = (b,a));
    the ordered-tuple counts are reported too.
  * eta^2 = Var(E[X|B]) / Var(X) for X = rank(c) (average ranks) and X = p = log(1+c) (train counts c), by the group
    (PLAN 4.5); Var(mu)/Var(p) is the same number as eta2_p (PLAN 4.6).
  * rho_proxy = Spearman(c_i, N_{a(i)}), N_b = sum of c over items in bucket b (dfreq: N_{a1} + N_{a2}, each N_a over
    the distinct items having alias a in either slot); rho_proxy^2 <= eta2_rank_c is checked (Theorem 2, g = rank o N).
  * sha256 of the int64 arrays. Per dataset: split sizes, evaluated users in val / test.
--assert-aliases NAME=PATH_OR_URL compares the aliases stored in a checkpoint with the map NAME (e.g.
gowalla__graphhash, amazonbook__graphhash) and records equal true/false (never aborts).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

import score_controls as SC

EXPECTED_R200_BUCKETS = {"gowalla": (5792, 5806)}      # TASKS T4 acceptance test
RHO_PROXY_DEFINITION = ("Spearman(c_i, N_a(i)); N_b = sum of c over the distinct items in bucket b; "
                        "dfreq: N_a1 + N_a2")


def eta_squared(x, labels):
    """Var(E[X|B]) / Var(X) over items (population variances, every item weight 1/n). None if Var(X) = 0."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    g = int(labels.max()) + 1
    cnt = np.bincount(labels, minlength=g).astype(np.float64)
    mean_g = np.bincount(labels, weights=x, minlength=g) / np.maximum(cnt, 1)
    xbar = x.mean()
    between = float((cnt * (mean_g - xbar) ** 2).sum() / len(x))
    total = float(((x - xbar) ** 2).mean())
    return None if total <= 0 else between / total


def group_size_stats(labels, n_groups):
    sizes = np.bincount(labels, minlength=n_groups)
    return {"n_groups": int(n_groups), "mean": float(sizes.mean()), "median": float(np.median(sizes)),
            "max": int(sizes.max()), "n_singleton_groups": int((sizes == 1).sum()),
            "share_in_groups_ge2": float((sizes[labels] >= 2).mean())}


def traffic_proxy(counts, item_alias):
    """Per-item N_{a(i)} (dfreq: N_{a1(i)} + N_{a2(i)})."""
    c = np.asarray(counts, dtype=np.float64)
    a = np.asarray(item_alias)
    if a.ndim == 1:
        return np.bincount(a, weights=c, minlength=int(a.max()) + 1)[a]
    rows = int(a.max()) + 1
    a1, a2 = a[:, 0], a[:, 1]
    n = np.bincount(a1, weights=c, minlength=rows) + np.bincount(a2, weights=c * (a2 != a1), minlength=rows)
    return n[a1] + n[a2]


def spearman(x, y):
    r = spearmanr(x, y).correlation
    return None if r is None or not np.isfinite(r) else float(r)


def summarize_map(user_alias, item_alias, counts):
    """All model-free statistics of one mapping."""
    ua, ia = np.asarray(user_alias), np.asarray(item_alias)
    c = np.asarray(counts, dtype=np.float64)
    p = np.log1p(c)
    labels, g, _ = SC.group_labels(ia, canonical=True)
    ulabels, ug, _ = SC.group_labels(ua, canonical=True)
    out = {"multi_alias": bool(ia.ndim == 2),
           "user_table_rows": int(ua.max()) + 1, "item_table_rows": int(ia.max()) + 1,
           "user_rows_used": int(len(np.unique(ua))), "item_rows_used": int(len(np.unique(ia))),
           "items_per_group": group_size_stats(labels, g), "users_per_group": group_size_stats(ulabels, ug),
           "n_item_groups_ordered_tuples": SC.group_labels(ia, canonical=False)[1],
           "n_user_groups_ordered_tuples": SC.group_labels(ua, canonical=False)[1],
           "eta2_rank_c": eta_squared(rankdata(c, method="average"), labels), "eta2_p": eta_squared(p, labels)}
    if ia.ndim == 2:
        lo, _, _ = SC.group_labels(ia, canonical=False)
        out["eta2_rank_c_ordered_tuples"] = eta_squared(rankdata(c, method="average"), lo)
        out["eta2_p_ordered_tuples"] = eta_squared(p, lo)
    _, mu, eps = SC.mu_eps(p, labels, g)
    out.update({"var_p": float(p.var()), "var_mu": float(mu.var()), "var_eps": float(eps.var()),
                "var_mu_over_var_p": float(mu.var() / p.var()) if p.var() > 0 else None,
                "rho_proxy": spearman(c, traffic_proxy(c, ia)), "rho_proxy_definition": RHO_PROXY_DEFINITION})
    if out["rho_proxy"] is not None and out["eta2_rank_c"] is not None:
        out["theorem2_check_rho_proxy_sq_le_eta2_rank_c"] = bool(out["rho_proxy"] ** 2 <= out["eta2_rank_c"] + 1e-9)
    out["users_sha256"], out["items_sha256"] = SC.sha256_array(ua), SC.sha256_array(ia)
    return out


def map_key(dataset, compression, resolution=None):
    suffix = f"-r{resolution}" if resolution is not None else ""
    return f"{dataset.lower()}__{compression}{suffix}"


def build_dataset_maps(R, ns, dataset, extra_resolutions):
    """{key: (user_alias, item_alias, info)} for one dataset, plus the dataset block of the summary."""
    from scipy.sparse import csr_matrix
    parts, maps, (train_truth, val_truth, test_truth), info = R.load_data(dataset, ns)
    nu, ni = int(info["user_id_max"]) + 1, int(info["item_id_max"]) + 1
    counts = np.bincount(parts[0][1].numpy(), minlength=ni).astype(np.int64)
    ku, ki = int(info["user_buckets"]), int(info["item_buckets"])
    out = {map_key(dataset, "graphhash"): (SC.to_alias(maps["GraphHash"][0]), SC.to_alias(maps["GraphHash"][1]), {"resolution": 200}),
           map_key(dataset, "random"): (SC.to_alias(maps["Random"][0]), SC.to_alias(maps["Random"][1]), {})}
    uc, ic = SC.double_freq_maps(Path(R.WORK) / "GraphHash" / "utils.py", parts, nu, ni, ku, ki)
    out[map_key(dataset, "dfreq")] = (uc, ic, {"cutoff": "author default", "bucket_counts_from": "graphhash r200"})
    block = {"users": nu, "items": ni, "split_counts": info["split_counts"], "interactions": info["interactions"],
             "train_interactions": int(parts[0].shape[1]), "eval_users_val": len(val_truth),
             "eval_users_test": len(test_truth), "graphhash_r200_buckets": [ku, ki],
             "items_with_zero_train_count": int((counts == 0).sum())}
    exp = EXPECTED_R200_BUCKETS.get(dataset.lower())
    if exp is not None:
        block["expected_graphhash_r200_buckets"] = list(exp)
        block["graphhash_r200_buckets_match_expected"] = (ku, ki) == exp
    graph = None
    for res in extra_resolutions if dataset.lower() == "gowalla" else []:
        if graph is None:
            graph = csr_matrix((np.ones(parts[0].shape[1]), parts[0].numpy()), shape=(nu, ni))
        np.random.seed(getattr(R, "SEED", 0))                # same seeding as load_data
        u, i, k_u, k_i = ns["graphClustering"](graph, resolution=res)
        u, i = ns["map_to_consecutive_integers"](u), ns["map_to_consecutive_integers"](i)
        out[map_key(dataset, "graphhash", res)] = (SC.to_alias(u), SC.to_alias(i), {"resolution": res})
        SC.log("resolution_done", dataset=dataset, resolution=res, buckets=[int(k_u), int(k_i)])
    return out, counts, block


def parse_assert(items):
    out = {}
    for it in items or []:
        name, sep, path = it.partition("=")
        if not sep or not path:
            raise SystemExit("--assert-aliases expects NAME=PATH_OR_URL")
        out[name.strip().lower()] = path
    return out


def run_asserts(asserts, arrays, out_root):
    """arrays: {key: (user_alias, item_alias)}. Records equality for every requested name; never aborts."""
    res = {}
    for name, src in asserts.items():
        r = {"source": SC.source_label(src)}
        try:
            if name not in arrays:
                r.update(equal=None, error=f"map {name} was not built (known: {sorted(arrays)})")
            else:
                ck = SC.load_checkpoint(SC.fetch_to(src, Path(out_root) / "_inputs", f"assert_{name}.pt"))
                if ck["aliases"] is None:
                    r.update(equal=None, error="checkpoint has no stored aliases")
                else:
                    ou, oi = ck["aliases"]
                    mu_, mi_ = arrays[name]
                    r["users_equal"] = bool(np.array_equal(np.asarray(mu_), ou))
                    r["items_equal"] = bool(np.array_equal(np.asarray(mi_), oi))
                    r["equal"] = r["users_equal"] and r["items_equal"]
                    r["stored_users_sha256"], r["stored_items_sha256"] = SC.sha256_array(ou), SC.sha256_array(oi)
        except Exception as e:
            r.update(equal=None, error=SC.scrub(f"{type(e).__name__}: {e}"))
        res[name] = r
        SC.log("assert_aliases", name=name, equal=r.get("equal"))
    return res


def write_summary(path, summary):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=SC._jd))
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="Gowalla,Yelp2018,AmazonBook")
    p.add_argument("--out-root", default="/kaggle/working/maps")
    p.add_argument("--gowalla-resolutions", default="400,800,1600,3200")
    p.add_argument("--assert-aliases", nargs="+", action="extend", default=[], metavar="NAME=PATH_OR_URL")
    args = p.parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    resolutions = [int(x) for x in args.gowalla_resolutions.split(",") if x.strip()]
    asserts = parse_assert(args.assert_aliases)
    summary = {"schema": "make_maps/1", "datasets": {}, "maps": {}, "assert_aliases": {}, "errors": {}}
    arrays = {}
    started = time.perf_counter()
    for ds in datasets:
        try:
            R, (_, _, _, _, ns) = SC.prepare_repro()
            maps, counts, block = build_dataset_maps(R, ns, ds, resolutions)
            ref = summarize_map(np.arange(block["users"]), np.arange(block["items"]), counts)
            ref["saved"] = False
            summary["maps"][map_key(ds, "full")] = {"dataset": ds, "compression": "full (identity reference)", **ref}
            block["train_count_max"] = int(counts.max())
            summary["datasets"][ds.lower()] = block
            for key, (ua, ia, extra) in maps.items():
                comp = key.split("__", 1)[1]
                np.save(out_root / f"{key.split('__')[0]}__{comp}__users.npy", ua)
                np.save(out_root / f"{key.split('__')[0]}__{comp}__items.npy", ia)
                arrays[key] = (ua, ia)
                summary["maps"][key] = {"dataset": ds, "compression": comp, **extra, "saved": True,
                                        **summarize_map(ua, ia, counts)}
                SC.log("map_done", key=key, eta2_p=summary["maps"][key]["eta2_p"],
                       rho_proxy=summary["maps"][key]["rho_proxy"])
                write_summary(out_root / "maps_summary.json", summary)
        except Exception:
            summary["errors"][ds] = SC.scrub(traceback.format_exc())
            SC.log("dataset_failed", dataset=ds)
        write_summary(out_root / "maps_summary.json", summary)
    summary["assert_aliases"] = run_asserts(asserts, arrays, out_root)
    summary["wall_seconds"] = round(time.perf_counter() - started, 1)
    write_summary(out_root / "maps_summary.json", summary)
    SC.log("done", out=str(out_root / "maps_summary.json"), errors=sorted(summary["errors"]))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
