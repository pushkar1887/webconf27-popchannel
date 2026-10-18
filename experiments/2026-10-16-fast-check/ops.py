"""Push/status/fetch the three fast-check kernels on the University account."""
import base64, json, os, sys
from pathlib import Path
import requests

os.environ["KAGGLE_CONFIG_DIR"] = r"C:\Users\PUSHKAR\.kaggle-accounts" + "\\" + os.environ.get("KACCT", "account-3")
os.environ.pop("KAGGLE_USERNAME", None); os.environ.pop("KAGGLE_KEY", None)
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

HERE = Path(__file__).parent
SRC = HERE.parent / "2026-10-16-loss-sharing-repro" / "_work"
RUNS = ["bpr", "dau", "dau_fix"]
a = KaggleApi(); a.authenticate()
owner = a.config_values["username"]
slug = lambda r: f"graphhash-fast-{r.replace('_', '-').replace(':', '-s').replace(',', '')}-1016"

cmd = sys.argv[1]
for run in (sys.argv[2:] or RUNS):
    if cmd == "push":
        folder = HERE / "kernels" / slug(run)
        folder.mkdir(parents=True, exist_ok=True)
        files = {n: base64.b64encode(p.read_bytes()).decode() for n, p in
                 [("repro_core.py", SRC/"repro_core.py"), ("repro_run.py", SRC/"repro_run.py"),
                  ("fast_run.py", HERE/"fast_run.py")]}
        entry = ("import base64,runpy,sys\nfrom pathlib import Path\n"
                 "p=Path('/kaggle/working/_work');p.mkdir(exist_ok=True)\n"
                 f"files={files!r}\n"
                 "for n,d in files.items(): (p/n).write_bytes(base64.b64decode(d))\n"
                 f"sys.path.insert(0,str(p));sys.argv=['fast_run.py']+{run.split(':')!r}\n"
                 "runpy.run_path(str(p/'fast_run.py'),run_name='__main__')\n")
        (folder/"run.py").write_text(entry, encoding="utf-8")
        (folder/"kernel-metadata.json").write_text(json.dumps({
            "id": f"{owner}/{slug(run)}", "title": slug(run), "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": []}))
        r = a.kernels_push(str(folder))
        print(run, "pushed version", getattr(r, "version_number", None), "error:", getattr(r, "error", None))
    elif cmd == "status":
        print(run, str(a.kernels_status(f"{owner}/{slug(run)}")).replace(owner, "<account>"))
    elif cmd == "fetch":
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = owner, slug(run)
        with a.build_kaggle_client() as c:
            r = c.kernels.kernels_api_client.list_kernel_session_output(req)
        out = HERE / "results"; out.mkdir(exist_ok=True)
        if r.log:
            (out/f"{run.replace(':', '_s').replace(',', '')}.log").write_text(r.log, encoding="utf-8")
        for f in r.files:
            name = run.split(":")[0]
            if (Path(f.file_name).name.startswith(f"fast_{name}") and f.file_name.endswith(".json")) or f.file_name.endswith("failure.log"):
                (out/Path(f.file_name).name.replace("failure", f"{slug(run)}_failure")).write_bytes(
                    requests.get(f.url, timeout=30).content)
        print(run, "fetched", [f.file_name for f in r.files][:10])
