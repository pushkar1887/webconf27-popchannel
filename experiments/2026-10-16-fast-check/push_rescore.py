import base64,json,os,sys
from pathlib import Path
os.environ["KAGGLE_CONFIG_DIR"]=r"C:\Users\PUSHKAR\.kaggle-accounts"+"\\"+os.environ.get("KACCT","account-3")
from kaggle.api.kaggle_api_extended import KaggleApi
a=KaggleApi();a.authenticate();o=a.config_values["username"]
HERE=Path(__file__).parent;SRC=HERE.parent/"2026-10-16-loss-sharing-repro"/"_work"
src=sys.argv[1];script=sys.argv[2] if len(sys.argv)>2 else "rescore.py";slug=sys.argv[3] if len(sys.argv)>3 else "graphhash-rescore-"+src.split("graphhash-fast-")[1]
d=HERE/"kernels"/slug;d.mkdir(parents=True,exist_ok=True)
files={n:base64.b64encode(p.read_bytes()).decode() for n,p in [("repro_core.py",SRC/"repro_core.py"),("repro_run.py",SRC/"repro_run.py"),(script,HERE/script)]}
(d/"run.py").write_text("import base64,runpy,sys\nfrom pathlib import Path\np=Path('/kaggle/working/_work');p.mkdir(exist_ok=True)\n"+f"files={files!r}\n"+"for n,x in files.items(): (p/n).write_bytes(base64.b64decode(x))\nsys.path.insert(0,str(p))\nrunpy.run_path(str(p/'"+script+"'),run_name='__main__')\n")
(d/"kernel-metadata.json").write_text(json.dumps({"id":f"{o}/{slug}","title":slug,"code_file":"run.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":[],"competition_sources":[],"kernel_sources":[f"{o}/{src}"]}))
r=a.kernels_push(str(d));print(slug,getattr(r,"version_number",None),getattr(r,"error",None))
