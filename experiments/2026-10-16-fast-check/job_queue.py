"""Priority job scheduler for the GraphHash follow-up experiments on Kaggle (2 accounts, 2 GPU sessions each).

Session bundles (v2, T4 x2 sessions; PLAN 7.4):
    python job_queue.py --sessions sessions_X.py [--state FILE] [--dry-run]
sessions_X.py defines SESSIONS = [{"slug", "account", "kernel_sources", "gpu_lists": [[job, ...], [job, ...]]}, ...].
A job is {"run_id", "script", "args", optional "inputs": {name: url}} or {"parallel": [job, job]} (both jobs run at the
same time on the same GPU, a co-location test). The two gpu_lists of a session run in parallel (one per GPU,
CUDA_VISIBLE_DEVICES=0/1); each list is sequential. --dry-run builds kernels_v2/<slug>/ and pushes nothing.
Default state file: <project>/results/queue_state_v2.json. Events go to stdout and <project>/results/queue_events_v2.log.

Legacy single-job mode (the JOBS list below; restartable via results/queue_state.json), no --sessions:
    python job_queue.py            # run until every job is done/failed
    python job_queue.py --dry-run  # only build the kernel folders under kernels/, no Kaggle calls

(Named job_queue.py, not queue.py: a queue.py in this folder would shadow the stdlib `queue` module and break
`import torch` / `import kaggle` for every script run from here.)
Every Kaggle call runs in a `kapi.py` subprocess, which sets KAGGLE_CONFIG_DIR before importing kaggle, so one
scheduler process can talk to either account. Kaggle usernames are never printed or stored in the v2 state file.
"""
import argparse
import base64
import json
import py_compile
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "2026-10-16-loss-sharing-repro" / "_work"
RESULTS = HERE / "results"
STATE_FILE = RESULTS / "queue_state.json"
EVENTS = RESULTS / "queue_events.log"
ACCOUNTS = ["account-3", "account-4"]          # account-2 is invalid
GPU_SLOTS = 2                                   # max concurrent GPU sessions per account
POLL_SECONDS = 60
SOURCE_ACCOUNT = "account-3"                    # owner of graphhash-fast-bpr-author-1016
GPU_FULL_MSG = "Maximum batch GPU session count"
MAX_PUSH_ERRORS = 3
MAX_FETCH_ERRORS = 10
COMMON_FILES = [("repro_core.py", SRC), ("repro_run.py", SRC), ("exp_run.py", HERE)]

# --- v2 (session bundles)
PROJECT = HERE.parents[1] / "webconf27-popchannel"          # D:\web_2\webconf27-popchannel
V2_STATE = PROJECT / "results" / "queue_state_v2.json"
V2_EVENTS = PROJECT / "results" / "queue_events_v2.log"
RUNS_DIR = PROJECT / "runs"
KERNELS_V2 = HERE / "kernels_v2"
FETCH_PREFIXES = ["runs/", "logs/", "maps/", "controls/", "session_info.json", "session_status.json"]
# embedded in every session kernel when the file exists (looked up in HERE, then in the loss-sharing-repro _work dir)
EMBED_NAMES = ["repro_core.py", "repro_run.py", "exp_run.py", "score_controls.py", "make_maps.py"]
OWNERS = {}                                     # account -> Kaggle username; in memory only, never printed or stored

# ----------------------------------------------------------------------------------------------
# EDIT HERE. Priority order (first = highest). Fields:
#   tag            -> kernel slug gh-exp-{tag}-1017 and output exp_{tag}.json (use the SAME tag in --tag)
#   script         -> exp_run.py | score_bpr_ext.py (any file in this folder)
#   args           -> command-line string passed to the script
#   needs_gpu      -> False pushes with enable_gpu=false (no GPU slot used)
#   kernel_sources -> optional list of kernel slugs owned by SOURCE_ACCOUNT; forces that account
# ----------------------------------------------------------------------------------------------
JOBS = [
    # P1 (no training)
    dict(tag="p1-bpr-ext", script="score_bpr_ext.py", args="", needs_gpu=False,
         kernel_sources=["graphhash-fast-bpr-author-1016"]),
    # P1 rerun on GPU image: CPU image produced a different GraphHash bucket labelling
    dict(tag="p1-bpr-ext-gpu", script="score_bpr_ext.py", args="", needs_gpu=True,
         kernel_sources=["graphhash-fast-bpr-author-1016"]),
    # retry: Google Drive 500 during dataset download (download now retries)
    dict(tag="p1-bpr-ext-gpu2", script="score_bpr_ext.py", args="", needs_gpu=True,
         kernel_sources=["graphhash-fast-bpr-author-1016"]),
    # P2: GraphHash x DirectAU seeds 0,1 (seed 2 exists), same settings as before
    dict(tag="p2-gh-dau-s0", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset Gowalla --method GraphHash --loss dau --seeds 0 --epochs 30 --eval-every 5 --lr 0.01 --wd 1e-8 --tag p2-gh-dau-s0"),
    dict(tag="p2-gh-dau-s1", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset Gowalla --method GraphHash --loss dau --seeds 1 --epochs 30 --eval-every 5 --lr 0.01 --wd 1e-8 --tag p2-gh-dau-s1"),
    # P3: authors' per-method DirectAU settings (commands.sh)
    dict(tag="p3-random-dau", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset Gowalla --method Random --loss dau --seeds 0 --epochs 30 --eval-every 5 --lr 0.01 --wd 1e-6 --tag p3-random-dau"),
    dict(tag="p3-dfreq-dau", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset Gowalla --method DoubleFreq --loss dau --seeds 0 --epochs 30 --eval-every 5 --lr 0.005 --wd 1e-6 --tag p3-dfreq-dau"),
    # P4: AmazonBook, authors' settings; DAU patience 10 / cap 75 min; BPR full batch to convergence
    dict(tag="p4-amz-gh-dau", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset AmazonBook --method GraphHash --loss dau --seeds 0 --epochs 1000 --eval-every 5 --lr 0.001 --wd 1e-8 --patience 10 --cap-min 75 --tag p4-amz-gh-dau"),
    dict(tag="p4-amz-full-dau", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset AmazonBook --method Full --loss dau --seeds 0 --epochs 1000 --eval-every 5 --lr 0.001 --wd 1e-8 --patience 10 --cap-min 75 --tag p4-amz-full-dau"),
    dict(tag="p4-amz-gh-bpr", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset AmazonBook --method GraphHash --loss bpr_author --seeds 0 --epochs 3000 --eval-every 10 --lr 0.01 --wd 1e-6 --patience 100 --cap-min 75 --tag p4-amz-gh-bpr"),
    # P4 rerun: patience 100 stopped on an early epoch-10 spike while val was still rising
    dict(tag="p4-amz-gh-bpr-long", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset AmazonBook --method GraphHash --loss bpr_author --seeds 0 --epochs 3000 --eval-every 10 --lr 0.01 --wd 1e-6 --patience 500 --cap-min 75 --tag p4-amz-gh-bpr-long"),
    # P5: learned per-item scalar b_i
    dict(tag="p5-gh-dau-bias", script="exp_run.py", needs_gpu=True, kernel_sources=[],
         args="--dataset Gowalla --method GraphHash --loss dau_bias --seeds 0 --epochs 30 --eval-every 5 --lr 0.01 --wd 1e-8 --tag p5-gh-dau-bias"),
]


# ----------------------------------------------------------------------------------------------
def now():
    return datetime.now().isoformat(timespec="seconds")


def redact(text):
    """Remove any known Kaggle username from text that is about to be printed or logged."""
    text = str(text)
    for owner in OWNERS.values():
        if owner:
            text = text.replace(owner, "<owner>")
    return text


def event(line):
    line = redact(line)
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(f"{line}  # {now()}\n")
    print(f"[{now()}] {line}", flush=True)


def slugify(tag):
    return f"gh-exp-{re.sub(r'[^a-z0-9]+', '-', tag.lower()).strip('-')}-1017"


def kapi(account, *args, timeout=900):
    """Run one Kaggle API call in a subprocess. Returns (parsed_json_or_None, combined_text)."""
    p = subprocess.run([sys.executable, str(HERE / "kapi.py"), account, *map(str, args)],
                       capture_output=True, text=True, timeout=timeout, cwd=str(HERE))
    text = (p.stdout or "") + (p.stderr or "")
    lines = [x for x in (p.stdout or "").strip().splitlines() if x.startswith("{")]
    try:
        return (json.loads(lines[-1]) if lines and p.returncode == 0 else None), text
    except json.JSONDecodeError:
        return None, text


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"owners": {}, "jobs": {}}


def save_state(state):
    RESULTS.mkdir(exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE_FILE)


def owner_of(state, account):
    if account not in state["owners"]:
        res, text = kapi(account, "whoami")
        if res is None:
            raise RuntimeError(f"whoami failed for {account}: {text[-300:]}")
        state["owners"][account] = res["owner"]
        OWNERS[account] = res["owner"]
        save_state(state)
    OWNERS[account] = state["owners"][account]
    return state["owners"][account]


def build_kernel(job, owner, source_owner=None):
    slug = slugify(job["tag"])
    folder = HERE / "kernels" / slug
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, root in COMMON_FILES + [(job["script"], HERE)]:
        files[name] = base64.b64encode((root / name).read_bytes()).decode()
    argv = [job["script"]] + shlex.split(job.get("args", ""))
    entry = ("import base64,runpy,sys\nfrom pathlib import Path\n"
             "p=Path('/kaggle/working/_work');p.mkdir(exist_ok=True)\n"
             f"files={files!r}\n"
             "for n,d in files.items(): (p/n).write_bytes(base64.b64decode(d))\n"
             f"sys.path.insert(0,str(p));sys.argv={argv!r}\n"
             f"runpy.run_path(str(p/{job['script']!r}),run_name='__main__')\n")
    (folder / "run.py").write_text(entry, encoding="utf-8")
    sources = [f"{source_owner}/{s}" for s in job.get("kernel_sources") or []]
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": bool(job["needs_gpu"]),
        "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": sources}))
    return slug, folder


def gpu_running(state, account):
    return sum(1 for t, j in state["jobs"].items()
               if j["state"] == "running" and j["account"] == account and j["needs_gpu"])


def finish(state, tag, status):
    """Fetch outputs + logs into results/, then record DONE. Returns False if the fetch must be retried."""
    j = state["jobs"][tag]
    res, text = kapi(j["account"], "fetch", j["slug"], RESULTS, tag)
    if res is None:
        event(f"FETCH_RETRY {tag} {status}: {text[-200:].strip()!r}")
        return False
    j["saved"] = res["saved"]
    j["state"] = "done" if status == "COMPLETE" else "failed"
    j["final_status"] = status
    event(f"DONE {tag} {status} saved={res['saved']}")
    return True


def poll_running(state):
    for tag, j in state["jobs"].items():
        if j["state"] not in ("running", "fetching"):
            continue
        if j["state"] == "running":
            res, text = kapi(j["account"], "status", j["slug"])
            if res is None:
                print(f"status check failed for {tag}: {redact(text[-200:].strip())}", flush=True)
                continue
            j["status"] = res["status"]
            if res["status"] not in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
                continue
            j["state"], j["final_status"] = "fetching", res["status"]
            if res.get("failure_message"):
                j["failure_message"] = res["failure_message"]
        finish(state, tag, j["final_status"])
        save_state(state)


def try_schedule(state, dry_run=False):
    pushed_any, blocked = False, set()
    for job in JOBS:
        j = state["jobs"][job["tag"]]
        if j["state"] != "pending":
            continue
        sources = job.get("kernel_sources") or []
        candidates = [SOURCE_ACCOUNT] if sources else sorted(ACCOUNTS, key=lambda a: gpu_running(state, a))
        for account in candidates:
            if account in blocked or (job["needs_gpu"] and gpu_running(state, account) >= GPU_SLOTS):
                continue
            owner = owner_of(state, account)
            slug, folder = build_kernel(job, owner, owner_of(state, SOURCE_ACCOUNT) if sources else None)
            res, text = kapi(account, "push", folder)
            error = (res or {}).get("error") if res else text
            if res is not None and not error:
                j.update(state="running", account=account, slug=slug, pushed_at=now(), status="PUSHED")
                event(f"PUSHED {job['tag']} {account} {slug}")
                pushed_any = True
                save_state(state)
                break
            if GPU_FULL_MSG in (error or "") or GPU_FULL_MSG in text:
                blocked.add(account)          # slot count was stale; leave job pending, try the other account
                print(f"GPU slots full on {account}; {job['tag']} stays pending", flush=True)
                continue
            j["push_errors"] = j.get("push_errors", 0) + 1
            j["last_error"] = (error or "")[-400:]
            event(f"PUSH_ERROR {job['tag']} {account} #{j['push_errors']}: {j['last_error']!r}")
            if j["push_errors"] >= MAX_PUSH_ERRORS:
                j["state"], j["final_status"] = "failed", "PUSH_ERROR"
                event(f"DONE {job['tag']} PUSH_ERROR")
            save_state(state)
            break
    return pushed_any


def legacy_main(args):
    global STATE_FILE
    if args.state:
        STATE_FILE = Path(args.state)
    if args.dry_run:
        for job in JOBS:
            slug, folder = build_kernel(job, "OWNER", "SRCOWNER")
            print("built", slug, "->", folder, "| gpu:", job["needs_gpu"], "| argv:",
                  [job["script"]] + shlex.split(job.get("args", "")))
        return
    RESULTS.mkdir(exist_ok=True)
    state = load_state()
    for job in JOBS:  # new jobs may be added between restarts; existing job states are kept
        state["jobs"].setdefault(job["tag"], {"state": "pending", "needs_gpu": bool(job["needs_gpu"]),
                                              "account": None, "slug": slugify(job["tag"])})
    save_state(state)
    event(f"QUEUE_START jobs={len(JOBS)}")
    while True:
        poll_running(state)
        while try_schedule(state):
            pass
        save_state(state)
        if all(j["state"] in ("done", "failed") for j in state["jobs"].values()):
            event("QUEUE_DONE " + json.dumps({t: j["state"] for t, j in state["jobs"].items()}))
            return
        time.sleep(POLL_SECONDS)


# ==============================================================================================
# v2: session bundles (Kaggle GPU T4 x2: one session = two GPU job lists run in parallel)
# ==============================================================================================
# The kernel entry. __FILES__ -> {name: base64}, __SPEC__ -> the session spec. It ALWAYS finishes normally (never
# sys.exit / raise): Kaggle cannot mount the outputs of an ERROR'd kernel, so failures go to session_status.json.
ENTRY_TEMPLATE = r'''# Session bundle entry, generated by job_queue.py. Do not edit.
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FILES = __FILES__
SPEC = __SPEC__
WORKING = Path(os.environ.get("SESSION_WORKING", "/kaggle/working"))
WORK = WORKING / "_work"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


class Status:
    """session_status.json, rewritten atomically after every change (threads share it)."""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.data = {"slug": SPEC["slug"], "state": "starting", "started": now(), "ended": None, "device_count": None,
                     "mode": None, "inputs": {}, "jobs": {}, "errors": []}

    def _write(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def set(self, **kw):
        with self.lock:
            self.data.update(kw)
            self._write()

    def job(self, rid, **kw):
        with self.lock:
            self.data["jobs"].setdefault(rid, {}).update(kw)
            self._write()

    def input(self, name, **kw):
        with self.lock:
            self.data["inputs"][name] = kw
            self._write()

    def error(self, text):
        with self.lock:
            self.data["errors"].append(text[-3000:])
            self._write()


def flat(items):
    for it in items:
        if "parallel" in it:
            yield from it["parallel"]
        else:
            yield it


def all_jobs():
    for lst in SPEC["gpu_lists"]:
        yield from flat(lst)


def query_devices():
    info = {"device_count": 0, "device_names": [], "torch_version": None, "torch_error": None, "nvidia_smi_L": None}
    code = ("import json, torch; n = torch.cuda.device_count(); "
            "print(json.dumps({'device_count': n, 'torch_version': torch.__version__, "
            "'device_names': [torch.cuda.get_device_name(i) for i in range(n)]}))")
    try:   # in a subprocess: the entry process itself must never hold a CUDA context
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
        if p.returncode == 0 and lines:
            info.update(json.loads(lines[-1]))
        else:
            info["torch_error"] = (p.stdout + p.stderr)[-500:]
    except Exception as exc:
        info["torch_error"] = repr(exc)
    try:
        info["nvidia_smi_L"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                              timeout=60).stdout.strip() or None
    except Exception:
        pass
    if not info["device_count"] and info["torch_error"] and info["nvidia_smi_L"]:
        info["device_count"] = len([x for x in info["nvidia_smi_L"].splitlines() if x.startswith("GPU ")])
    return info


def download(url, dest, tries=6):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, dest)
            return dest.stat().st_size
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))


def fetch_inputs(status):
    """Download every job input up front (signed URLs expire within hours) into WORKING/inputs/<name>."""
    wanted = {}
    for job in all_jobs():
        wanted.update(job.get("inputs") or {})
    for name, url in wanted.items():
        try:
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("bad input name")
            size = download(url, WORKING / "inputs" / rel)
            status.input(name, ok=True, bytes=size)
            say(f"input {name}: {size} bytes")
        except Exception as exc:   # never put the URL into the status file (signed)
            status.input(name, ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            say(f"input {name}: FAILED {type(exc).__name__}")


def warm_up(status):
    """Download / install shared things ONCE before two jobs start in parallel (repro_run.prepare)."""
    log = WORKING / "logs" / "_prepare.log"
    try:
        with open(log, "wb") as lf:
            p = subprocess.run([sys.executable, "-u", "-c", "import repro_run as R; R.prepare(); print('prepare ok')"],
                               cwd=str(WORKING), env=dict(os.environ, PYTHONPATH=str(WORK)), stdout=lf,
                               stderr=subprocess.STDOUT, timeout=3600)
        status.set(prepare_exit_code=p.returncode)
        say(f"warm-up prepare exit code {p.returncode}")
    except Exception as exc:
        status.set(prepare_exit_code=-1)
        status.error("warm-up failed: " + repr(exc))


def run_job(job, gpu, status, colocated_with=None):
    run_id = job["run_id"]
    rec = {"run_id": run_id, "gpu": gpu, "script": job["script"], "args": job.get("args", ""), "state": "running",
           "started": now(), "ended": None, "duration_s": None, "exit_code": None, "log": f"logs/{run_id}.log"}
    if colocated_with:
        rec["colocated_with"] = colocated_with
    try:
        bad = [n for n in (job.get("inputs") or {}) if not status.data["inputs"].get(n, {}).get("ok")]
        if bad:
            status.job(run_id, **{**rec, "state": "skipped", "ended": now(), "error": f"input download failed: {bad}"})
            say(f"{run_id}: SKIPPED (inputs {bad} missing)")
            return
        status.job(run_id, **rec)
        cmd = [sys.executable, "-u", str(WORK / job["script"])] + shlex.split(job.get("args", ""))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        say(f"{run_id}: start on GPU {gpu}: {job['script']}")
        t0 = time.time()
        with open(WORKING / "logs" / f"{run_id}.log", "wb") as lf:
            rc = subprocess.Popen(cmd, cwd=str(WORKING), env=env, stdout=lf, stderr=subprocess.STDOUT).wait()
        status.job(run_id, state="done", ended=now(), duration_s=round(time.time() - t0, 1), exit_code=rc)
        say(f"{run_id}: exit code {rc} after {time.time() - t0:.0f}s")
    except BaseException as exc:   # a broken job must not stop the other jobs
        status.job(run_id, state="error", ended=now(), exit_code=-1, error=repr(exc)[:300])
        status.error(f"{run_id}: " + traceback.format_exc())


def run_list(items, gpu, status):
    for item in items:
        if "parallel" in item:   # co-location: both jobs at the same time on the same GPU
            a, b = item["parallel"]
            threads = [threading.Thread(target=run_job, args=(a, gpu, status, b["run_id"])),
                       threading.Thread(target=run_job, args=(b, gpu, status, a["run_id"]))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            run_job(item, gpu, status)


def cleanup():
    """Keep the outputs small: drop the dataset copies / downloaded inputs (the scheduler skips them anyway)."""
    for rel in ("_work/datasets", "_work/datasets.zip", "_work/sknetwork_vendor", "inputs"):
        p = WORKING / rel
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        except Exception:
            pass


def run_session(device_info=None):
    WORKING.mkdir(parents=True, exist_ok=True)
    (WORKING / "logs").mkdir(exist_ok=True)
    status = Status(WORKING / "session_status.json")
    try:
        WORK.mkdir(exist_ok=True)
        for name, data in FILES.items():
            (WORK / name).write_bytes(base64.b64decode(data))
        info = dict(device_info) if device_info is not None else query_devices()
        info.update(slug=SPEC["slug"], start_time=now(), python=sys.version.split()[0],
                    cuda_visible_devices_env=os.environ.get("CUDA_VISIBLE_DEVICES"))
        (WORKING / "session_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        n = int(info.get("device_count") or 0)
        lists = SPEC["gpu_lists"]
        mode = "parallel_2gpu" if (len(lists) >= 2 and n >= 2) else "sequential_gpu0"
        status.set(state="running", device_count=n, mode=mode)
        say(f"session {SPEC['slug']}: device_count={n} mode={mode}")
        fetch_inputs(status)
        if SPEC.get("prepare", True):
            warm_up(status)
        if mode == "parallel_2gpu":
            threads = [threading.Thread(target=run_list, args=(lst, g, status)) for g, lst in enumerate(lists)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:   # fewer than 2 GPUs: both lists one after the other on GPU 0
            for lst in lists:
                run_list(lst, 0, status)
        state = "complete"
    except BaseException:
        status.error(traceback.format_exc())
        state = "entry_error"
    try:
        jobs = status.data["jobs"]
        failed = [r for r, j in jobs.items() if j.get("exit_code") != 0]
        status.set(state=state, ended=now(), n_jobs=len(jobs), n_failed=len(failed), failed_run_ids=failed)
        say(f"session finished: state={state} jobs={len(jobs)} failed={failed}")
    except BaseException:
        pass
    cleanup()


def entry():
    try:
        run_session()
    except BaseException:   # last resort: still finish normally
        traceback.print_exc()


if os.environ.get("SESSION_NO_AUTORUN") != "1":   # (tests import this file without running the session)
    entry()
'''

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{4,49}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def normalize_job(job):
    out = {"run_id": job["run_id"], "script": job["script"], "args": job.get("args", "") or ""}
    if job.get("inputs"):
        out["inputs"] = dict(job["inputs"])
    return out


def normalize_session(s):
    """Defaults + plain structure (what is stored in the kernel and used by the scheduler)."""
    lists = []
    for lst in s["gpu_lists"]:
        items = []
        for it in lst:
            if "parallel" in it:
                items.append({"parallel": [normalize_job(j) for j in it["parallel"]]})
            else:
                items.append(normalize_job(it))
        lists.append(items)
    return {"slug": s["slug"], "account": s.get("account"), "kernel_sources": list(s.get("kernel_sources") or []),
            "prepare": bool(s.get("prepare", True)), "enable_gpu": bool(s.get("enable_gpu", True)), "gpu_lists": lists}


def session_jobs(s):
    for lst in s["gpu_lists"]:
        for it in lst:
            yield from (it["parallel"] if "parallel" in it else [it])


def collect_files(names=EMBED_NAMES):
    """name -> bytes for the files embedded in every session kernel (those that exist)."""
    files = {}
    for name in names:
        roots = (SRC, HERE) if name in ("repro_core.py", "repro_run.py") else (HERE, SRC)
        for root in roots:
            if (root / name).exists():
                files[name] = (root / name).read_bytes()
                break
    missing = [n for n in ("repro_core.py", "repro_run.py", "exp_run.py") if n not in files]
    if missing:
        raise FileNotFoundError(f"required files missing: {missing}")
    return files


def validate_sessions(sessions, embedded):
    """Raise ValueError listing every problem. `sessions` as written by the user; `embedded` = file names."""
    errs, slugs, run_ids = [], set(), {}
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("SESSIONS must be a non-empty list")
    for k, s in enumerate(sessions):
        where = f"session #{k} ({s.get('slug') if isinstance(s, dict) else '?'})"
        if not isinstance(s, dict) or "slug" not in s or "gpu_lists" not in s:
            errs.append(f"{where}: needs 'slug' and 'gpu_lists'")
            continue
        if not SLUG_RE.match(str(s["slug"])):
            errs.append(f"{where}: slug must match {SLUG_RE.pattern}")
        if s["slug"] in slugs:
            errs.append(f"{where}: duplicate slug")
        slugs.add(s["slug"])
        if s.get("account") not in (None, *ACCOUNTS):
            errs.append(f"{where}: account must be None or one of {ACCOUNTS}")
        if not all(isinstance(x, str) for x in (s.get("kernel_sources") or [])):
            errs.append(f"{where}: kernel_sources must be a list of slugs")
        lists = s["gpu_lists"]
        if not isinstance(lists, list) or not 1 <= len(lists) <= 2 or not all(isinstance(x, list) for x in lists):
            errs.append(f"{where}: gpu_lists must be 1 or 2 lists of jobs")
            continue
        inputs = {}
        for g, lst in enumerate(lists):
            for it in lst:
                if not isinstance(it, dict):
                    errs.append(f"{where}: GPU{g} item is not a dict")
                    continue
                if "parallel" in it:
                    group = it["parallel"]
                    if not (isinstance(group, list) and len(group) == 2 and all(
                            isinstance(j, dict) and "parallel" not in j for j in group)):
                        errs.append(f"{where}: GPU{g} 'parallel' needs exactly 2 plain jobs")
                        continue
                else:
                    group = [it]
                for job in group:
                    rid = job.get("run_id")
                    if not isinstance(rid, str) or not RUN_ID_RE.match(rid):
                        errs.append(f"{where}: bad run_id {rid!r}")
                        continue
                    if rid in run_ids:
                        errs.append(f"{where}: run_id {rid} already used in {run_ids[rid]}")
                    run_ids[rid] = s["slug"]
                    if job.get("script") not in embedded:
                        errs.append(f"{where}/{rid}: script {job.get('script')!r} is not an embedded file {sorted(embedded)}")
                    if not isinstance(job.get("args", ""), str):
                        errs.append(f"{where}/{rid}: args must be a string")
                        continue
                    if job.get("script") == "exp_run.py":
                        argv = shlex.split(job.get("args", ""))
                        if "--run-id" not in argv or argv[argv.index("--run-id") + 1:][:1] != [rid]:
                            errs.append(f"{where}/{rid}: exp_run.py args must contain '--run-id {rid}'")
                    for name, url in (job.get("inputs") or {}).items():
                        if Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(url, str):
                            errs.append(f"{where}/{rid}: bad input {name!r}")
                        elif inputs.setdefault(name, url) != url:
                            errs.append(f"{where}/{rid}: input {name!r} has two different URLs in one session")
    if errs:
        raise ValueError("invalid sessions file:\n  " + "\n  ".join(errs))


def load_sessions(path):
    """Execute a Python file that defines SESSIONS and return the validated list (as written)."""
    ns = runpy.run_path(str(path))
    sessions = ns.get("SESSIONS")
    validate_sessions(sessions, collect_files(EMBED_NAMES))
    return sessions


def source_account(s):
    """Account a session must run on: the requested one; sessions that mount kernel outputs default to SOURCE_ACCOUNT."""
    return s.get("account") or (SOURCE_ACCOUNT if s.get("kernel_sources") else None)


def build_session_kernel(session, owner, files=None, root=None):
    """Write <root>/<slug>/{run.py, kernel-metadata.json}. kernel_sources: bare slugs are owned by `owner`
    (a kernel can mount only its own account's outputs); entries containing '/' are used as given."""
    s = normalize_session(session)
    files = files if files is not None else collect_files(EMBED_NAMES)
    embedded = {n: base64.b64encode(b).decode() for n, b in files.items()}
    spec = {"slug": s["slug"], "prepare": s["prepare"], "gpu_lists": s["gpu_lists"]}
    entry = ENTRY_TEMPLATE.replace("__FILES__", repr(embedded)).replace("__SPEC__", repr(spec))
    folder = Path(root or KERNELS_V2) / s["slug"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(entry, encoding="utf-8", newline="\n")
    sources = [x if "/" in x else f"{owner}/{x}" for x in s["kernel_sources"]]
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{s['slug']}", "title": s["slug"], "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": s["enable_gpu"], "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": sources}, indent=1), encoding="utf-8")
    return s["slug"], folder


def scrub_metadata(folder, owner):
    """After a push: keep no Kaggle username in the stored kernel folder."""
    p = Path(folder) / "kernel-metadata.json"
    if p.exists() and owner:
        p.write_text(p.read_text(encoding="utf-8").replace(owner, "OWNER"), encoding="utf-8")


def describe_session(session, folder):
    s = normalize_session(session)
    run_py = Path(folder) / "run.py"
    text = run_py.read_text(encoding="utf-8")
    compile(text, str(run_py), "exec")                       # SyntaxError -> the dry run fails loudly
    lines = [f"SESSION {s['slug']}  account={source_account(s) or 'any'}  kernel_sources={s['kernel_sources']}  "
             f"gpu={s['enable_gpu']}  run.py={len(text)//1024} KiB  syntax=OK  -> {folder}"]
    for g, lst in enumerate(s["gpu_lists"]):
        parts = []
        for it in lst:
            one = lambda j: f"{j['run_id']} [{j['script']}{', inputs=' + str(sorted(j['inputs'])) if j.get('inputs') else ''}]"
            parts.append(" || ".join(one(j) for j in it["parallel"]) + "  (parallel)" if "parallel" in it else one(it))
        lines.append(f"  GPU{g}: " + "  ->  ".join(parts) if parts else f"  GPU{g}: (empty)")
    return "\n".join(lines)


def hoist_session(sdir, runs_dir, slug):
    """Move <sdir>/runs/<run_id>/ to <runs_dir>/<run_id>/ and copy logs/<run_id>.log to <run_id>/stdout.log.
    An existing run dir from ANOTHER session is kept as <runs_dir>/_superseded/<run_id>__<time>."""
    sdir, runs_dir = Path(sdir), Path(runs_dir)
    moved = []
    src = sdir / "runs"
    if not src.is_dir():
        return moved
    for d in sorted(x for x in src.iterdir() if x.is_dir()):
        dest = runs_dir / d.name
        marker = dest / "fetched_from_session.txt"
        if dest.exists() and not (marker.exists() and marker.read_text().strip() == slug):
            old = runs_dir / "_superseded" / f"{d.name}__{datetime.now().strftime('%Y%m%d%H%M%S')}"
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(old))
        shutil.copytree(d, dest, dirs_exist_ok=True)
        shutil.rmtree(d)
        log = sdir / "logs" / f"{d.name}.log"
        if log.exists():
            shutil.copy2(log, dest / "stdout.log")
        marker.write_text(slug + "\n")
        moved.append(d.name)
    return moved


def session_summary(sdir):
    p = Path(sdir) / "session_status.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return {"state": d.get("state"), "n_jobs": d.get("n_jobs"), "failed": d.get("failed_run_ids") or [],
                "device_count": d.get("device_count")}
    except ValueError:
        return None


def v2_owner(account):
    if account not in OWNERS:
        res, text = kapi(account, "whoami")
        if res is None:
            raise RuntimeError(f"whoami failed for {account}: {text[-300:]}")
        OWNERS[account] = res["owner"]
    return OWNERS[account]


def v2_running(state, account):
    return sum(1 for j in state["sessions"].values() if j["state"] == "running" and j.get("account") == account)


def v2_finish(state, slug, status):
    """Fetch outputs into runs/, then record DONE / FAILED. Returns False if the fetch must be retried."""
    j = state["sessions"][slug]
    sdir = RUNS_DIR / "_sessions" / slug
    res, text = kapi(j["account"], "fetch", slug, sdir, "kernel", *FETCH_PREFIXES, timeout=7200)
    if res is None:
        j["fetch_errors"] = j.get("fetch_errors", 0) + 1
        event(f"FETCH_RETRY {slug} {status} #{j['fetch_errors']}: {text[-200:].strip()!r}")
        if j["fetch_errors"] < MAX_FETCH_ERRORS:
            return False
        j["state"], j["final_status"] = "failed", "FETCH_ERROR"
        event(f"FAILED {slug} FETCH_ERROR (outputs stay on Kaggle)")
        return True
    runs = hoist_session(sdir, RUNS_DIR, slug)
    summary = session_summary(sdir)
    ok = status == "COMPLETE"
    j.update(state="done" if ok else "failed", final_status=status, files_saved=len(res.get("saved", [])),
             run_dirs=runs, session_summary=summary)
    detail = (f"jobs={summary['n_jobs']} failed_jobs={summary['failed']} device_count={summary['device_count']}"
              if summary else "session_status.json=missing")
    event(f"{'DONE' if ok else 'FAILED'} {slug} {status} files={j['files_saved']} runs={len(runs)} {detail}")
    return True


def v2_poll_running(state):
    for slug, j in state["sessions"].items():
        if j["state"] not in ("running", "fetching"):
            continue
        if j["state"] == "running":
            res, text = kapi(j["account"], "status", slug)
            if res is None:
                print(f"status check failed for {slug}: {redact(text[-200:].strip())}", flush=True)
                continue
            j["status"] = res["status"]
            if res["status"] not in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
                continue
            j["state"], j["final_status"] = "fetching", res["status"]
            if res.get("failure_message"):
                j["failure_message"] = redact(res["failure_message"])
        v2_finish(state, slug, j["final_status"])
        save_state(state)


def v2_try_schedule(state, sessions, folder_root=None):
    pushed_any, blocked = False, set()
    for s in sessions:
        j = state["sessions"][s["slug"]]
        if j["state"] != "pending":
            continue
        pinned = source_account(s)
        candidates = [pinned] if pinned else sorted(ACCOUNTS, key=lambda a: v2_running(state, a))
        for account in candidates:
            if account in blocked or v2_running(state, account) >= GPU_SLOTS:
                continue
            owner = v2_owner(account)
            slug, folder = build_session_kernel(s, owner, root=folder_root)
            res, text = kapi(account, "push", folder)
            scrub_metadata(folder, owner)
            error = (res or {}).get("error") if res else text
            if res is not None and not error:
                j.update(state="running", account=account, pushed_at=now(), status="PUSHED")
                event(f"PUSHED {slug} {account}")
                pushed_any = True
                save_state(state)
                break
            if GPU_FULL_MSG in (error or "") or GPU_FULL_MSG in text:
                blocked.add(account)          # slot count was stale; leave the session pending, try the other account
                print(f"GPU slots full on {account}; {slug} stays pending", flush=True)
                continue
            j["push_errors"] = j.get("push_errors", 0) + 1
            j["last_error"] = redact((error or "")[-400:])
            event(f"PUSH_ERROR {slug} {account} #{j['push_errors']}: {j['last_error']!r}")
            if j["push_errors"] >= MAX_PUSH_ERRORS:
                j["state"], j["final_status"] = "failed", "PUSH_ERROR"
                event(f"FAILED {slug} PUSH_ERROR")
            save_state(state)
            break
    return pushed_any


def sessions_main(args):
    global STATE_FILE, EVENTS
    sessions = [normalize_session(s) for s in load_sessions(args.sessions)]
    if args.dry_run:
        for s in sessions:
            slug, folder = build_session_kernel(s, "OWNER")
            print(describe_session(s, folder))
        print(f"dry run: {len(sessions)} session(s) built under {KERNELS_V2}; nothing pushed")
        return
    STATE_FILE, EVENTS = Path(args.state) if args.state else V2_STATE, V2_EVENTS
    state = load_state() if STATE_FILE.exists() else {}
    state.pop("owners", None)                       # never keep usernames in the v2 state
    state.setdefault("sessions", {})
    for s in sessions:                              # sessions may be added between restarts; existing states are kept
        state["sessions"].setdefault(s["slug"], {"state": "pending", "account": None})
    save_state(state)
    event(f"QUEUE_START sessions={len(sessions)}")
    while True:
        v2_poll_running(state)
        while v2_try_schedule(state, sessions):
            pass
        save_state(state)
        live = [j for j in state["sessions"].values() if j["state"] in ("running", "fetching")]
        if not live and all(state["sessions"][s["slug"]]["state"] in ("done", "failed") for s in sessions):
            event("QUEUE_DONE " + json.dumps({k: j["state"] for k, j in state["sessions"].items()}))
            return
        time.sleep(POLL_SECONDS)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Kaggle job scheduler (session bundles; legacy JOBS mode without --sessions)")
    ap.add_argument("--sessions", help="Python file defining SESSIONS = [...] (T4 x2 session bundles)")
    ap.add_argument("--dry-run", action="store_true", help="build the kernel folders only; no Kaggle calls")
    ap.add_argument("--state", help=f"state file (default {V2_STATE}; legacy mode: {STATE_FILE})")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.sessions:
        sessions_main(args)
    else:
        legacy_main(args)


if __name__ == "__main__":
    main()
