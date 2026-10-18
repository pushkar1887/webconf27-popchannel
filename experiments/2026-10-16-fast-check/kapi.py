"""Kaggle API worker, one call per process so KAGGLE_CONFIG_DIR is fixed before kaggle is imported.

python kapi.py <account-dir-name> whoami
python kapi.py <account> push <kernel_folder>
python kapi.py <account> status <slug>
python kapi.py <account> fetch <slug> <outdir> <tag>                  # legacy: <tag>.log + exp_*/p1_*.json + *failure.log
python kapi.py <account> fetch <slug> <outdir> <tag> <prefix> [...]   # every output file whose path starts with a prefix,
                                                                      # saved under <outdir>/<path> (+ the kernel log as
                                                                      # <outdir>/<tag>.log); datasets/ and _work/ never
Prints one JSON object on the last stdout line. Never print the account's username anywhere but in that JSON
(the scheduler parses it and redacts it from everything it prints).
"""
import json
import os
import sys
import time
from pathlib import Path

account, cmd, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
os.environ["KAGGLE_CONFIG_DIR"] = r"C:\Users\PUSHKAR\.kaggle-accounts" + "\\" + account
os.environ.pop("KAGGLE_USERNAME", None)
os.environ.pop("KAGGLE_KEY", None)
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

SKIP_PREFIXES = ("datasets/", "_work/")


def norm(name):
    """Output file name -> path relative to /kaggle/working, with forward slashes."""
    name = str(name).replace("\\", "/").lstrip("/")
    for lead in ("kaggle/working/",):
        if name.startswith(lead):
            name = name[len(lead):]
    return name


def wanted(rel, prefixes):
    return not rel.startswith(SKIP_PREFIXES) and any(rel.startswith(p) for p in prefixes)


def download(url, dest, tries=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(tmp, dest)
            return dest.stat().st_size
        except requests.RequestException:
            if attempt == tries-1:
                raise
            time.sleep(10*(attempt+1))


def list_output(api, owner, slug, page_size=500):      # 1000 -> HTTP 400; 500 works (checked read-only)
    """All output files of the latest session (follows pagination) and the kernel log."""
    files, log, token = [], "", ""
    while True:
        q = ApiListKernelSessionOutputRequest()
        q.user_name, q.kernel_slug, q.page_size = owner, slug, page_size
        if token:
            q.page_token = token
        with api.build_kaggle_client() as c:
            r = c.kernels.kernels_api_client.list_kernel_session_output(q)
        log = log or (r.log or "")
        files.extend(r.files or [])
        token = getattr(r, "next_page_token", "") or ""
        if not token:
            return files, log


def run(cmd, rest, api, owner):
    out = {"owner": owner}
    if cmd == "whoami":
        pass
    elif cmd == "push":
        try:
            r = api.kernels_push(rest[0])
            out.update(version=getattr(r, "version_number", None), error=getattr(r, "error", None) or None)
        except Exception as exc:  # e.g. "Maximum batch GPU session count of 2 reached"
            out["error"] = f"{type(exc).__name__}: {exc}".replace(owner, "<owner>")
    elif cmd == "status":
        s = api.kernels_status(f"{owner}/{rest[0]}")
        out["status"] = str(s.status).split(".")[-1].upper()
        out["failure_message"] = str(getattr(s, "failure_message", "") or "")
    elif cmd == "fetch":
        slug, outdir, tag, prefixes = rest[0], Path(rest[1]), rest[2], rest[3:]
        outdir.mkdir(parents=True, exist_ok=True)
        saved = []
        if not prefixes:                                   # legacy behaviour (unchanged)
            q = ApiListKernelSessionOutputRequest()
            q.user_name, q.kernel_slug = owner, slug
            with api.build_kaggle_client() as c:
                r = c.kernels.kernels_api_client.list_kernel_session_output(q)
            if r.log:
                (outdir / f"{tag}.log").write_text(r.log, encoding="utf-8")
                saved.append(f"{tag}.log")
            for f in r.files:
                name = Path(f.file_name).name
                if (name.startswith(("exp_", "p1_")) and name.endswith(".json")) or name.endswith("failure.log"):
                    (outdir / name).write_bytes(requests.get(f.url, timeout=60).content)
                    saved.append(name)
            out["all_files"] = [f.file_name for f in r.files][:30]
        else:
            files, log = list_output(api, owner, slug)
            if log:
                (outdir / f"{tag}.log").write_text(log, encoding="utf-8")
                saved.append(f"{tag}.log")
            root = outdir.resolve()
            skipped = 0
            for f in files:
                rel = norm(f.file_name)
                if not wanted(rel, prefixes):
                    skipped += 1
                    continue
                dest = (outdir / rel).resolve()
                if root not in dest.parents:               # never write outside outdir
                    raise SystemExit(f"refusing path {rel!r}")
                download(f.url, dest)
                saved.append(rel)
            out.update(n_listed=len(files), n_skipped=skipped, sample_listed=[norm(f.file_name) for f in files][:30])
        out["saved"] = saved
    else:
        raise SystemExit(f"unknown command {cmd}")
    return out


api = KaggleApi()
api.authenticate()
owner = api.config_values["username"]
try:
    result = run(cmd, rest, api, owner)
except Exception as exc:        # never let a traceback (its URL contains the username) reach a log
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}".replace(owner, "<owner>")[-800:]}))
    raise SystemExit(1)
print(json.dumps(result))
