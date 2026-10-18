"""Private University-only push/status/output helper; never prints account identifiers."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import requests

os.environ["KAGGLE_CONFIG_DIR"] = r"C:\Users\PUSHKAR\.kaggle-accounts\account-3"
os.environ.pop("KAGGLE_USERNAME", None)
os.environ.pop("KAGGLE_KEY", None)
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

ROOT = Path(__file__).parent
OUT = ROOT.parent
SLUG = "graphhash-loss-sharing-repro-1016"
a = KaggleApi()
a.authenticate()
owner = a.config_values["username"]
ref = owner+"/"+SLUG

if sys.argv[1] == "push":
    folder = ROOT/"kaggle-run"
    metadata = json.loads((folder/"kernel-metadata.json").read_text())
    assert metadata["id"] == ref and metadata["enable_gpu"] and metadata["enable_internet"]
    assert metadata["code_file"] == "reproduction.py" and metadata["kernel_type"] == "script"
    payload = {name: base64.b64encode((ROOT/name).read_bytes()).decode()
               for name in ["repro_core.py", "repro_run.py"]}
    entry = ("import base64,runpy,sys\nfrom pathlib import Path\n"
             "p=Path('/kaggle/working/_work');p.mkdir(exist_ok=True)\n"
             f"files={payload!r}\n"
             "for name,data in files.items(): (p/name).write_bytes(base64.b64decode(data))\n"
             "sys.path.insert(0,str(p))\n"
             "runpy.run_path(str(p/'repro_run.py'),run_name='__main__')\n")
    (folder/"reproduction.py").write_text(entry, encoding="utf-8")
    r = a.kernels_push(str(folder))
    receipt = {"alias":"University","slug":SLUG,"version":getattr(r,"version_number",None),
               "error":getattr(r,"error",None),"source_sha256":hashlib.sha256(entry.encode()).hexdigest()}
    (ROOT/"submission.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
elif sys.argv[1] == "status":
    r = a.kernels_status(ref)
    print(json.dumps({"alias":"University","slug":SLUG,"status":str(r).replace(owner,"<account>")}))
elif sys.argv[1] == "fetch":
    version = json.loads((ROOT/"submission.json").read_text())["version"]
    target = ROOT/f"gpu-output-v{version}"
    target.mkdir(exist_ok=True)
    token, files, events = None, [], []
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = owner, SLUG
        if token:
            req.page_token = token
        with a.build_kaggle_client() as client:
            r = client.kernels.kernels_api_client.list_kernel_session_output(req)
        if r.log:
            (target/"kernel.log").write_text(r.log,encoding="utf-8")
            try:
                for e in json.loads(r.log):
                    s=e.get("data","").strip()
                    if s.startswith('{"event":'):
                        events.append(json.loads(s))
            except (ValueError,TypeError):
                pass
        for f in r.files:
            retained = f.file_name.startswith(("_work/Gowalla_", "_work/AmazonBook_")) and f.file_name.endswith((".npz", ".pt"))
            if retained or f.file_name in ["REPRO_REPORT.md","metrics.csv","_work/failure.log",
                                          "_work/clustering_install.log","_work/clustering_rebuild.log"]:
                content=requests.get(f.url,timeout=30).content
                path=target/f.file_name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(content)
                files.append(f.file_name)
                if f.file_name in ["REPRO_REPORT.md","metrics.csv"]:
                    (OUT/f.file_name).write_bytes(content)
        token=r.next_page_token
        if not token:
            break
    print(json.dumps({"alias":"University","version":version,"downloaded":files,
                      "latest_events":events[-5:]},default=str))
    failure=target/"_work/failure.log"
    if failure.exists():
        print(failure.read_text(errors="replace")[-4000:].replace(owner,"<account>"))
