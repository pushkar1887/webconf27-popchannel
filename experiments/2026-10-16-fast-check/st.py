import os,sys,json,requests
os.environ["KAGGLE_CONFIG_DIR"]=r"C:\Users\PUSHKAR\.kaggle-accounts"+"\\"+os.environ.get("KACCT","account-3")
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
a=KaggleApi();a.authenticate();o=a.config_values["username"];s=sys.argv[2]
if sys.argv[1]=="status": print(s,a.kernels_status(f"{o}/{s}").status)
else:
    q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=o,s
    with a.build_kaggle_client() as c: r=c.kernels.kernels_api_client.list_kernel_session_output(q)
    for f in r.files:
        if f.file_name.endswith(("rescore.json","popprior.json")): open(f"results/{s}.json","wb").write(requests.get(f.url).content)
    open(f"results/{s}.log","w",encoding="utf-8").write(r.log or "")
