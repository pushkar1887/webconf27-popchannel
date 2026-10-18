import os,sys,json
os.environ["KAGGLE_CONFIG_DIR"]=r"C:\Users\PUSHKAR\.kaggle-accounts"+"\\"+os.environ.get("KACCT","account-3")
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
import requests
a=KaggleApi();a.authenticate();o=a.config_values["username"]
for slug in sys.argv[2:]:
    q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=o,slug
    with a.build_kaggle_client() as c: r=c.kernels.kernels_api_client.list_kernel_session_output(q)
    pts=[f for f in r.files if f.file_name.endswith(".pt")]
    print(slug,[f.file_name for f in pts])
    if sys.argv[1]=="get":
        os.makedirs("ckpt",exist_ok=True)
        for f in pts:
            if "best" in f.file_name: open(f"ckpt/{slug}.pt","wb").write(requests.get(f.url,timeout=120).content)
