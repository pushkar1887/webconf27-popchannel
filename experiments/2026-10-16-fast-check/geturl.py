import os,sys
os.environ["KAGGLE_CONFIG_DIR"]=r"C:\Users\PUSHKAR\.kaggle-accounts"+"\\"+sys.argv[1]
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
a=KaggleApi();a.authenticate()
q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=a.config_values["username"],sys.argv[2]
with a.build_kaggle_client() as c: r=c.kernels.kernels_api_client.list_kernel_session_output(q)
print([f.url for f in r.files if f.file_name.endswith(sys.argv[3])][0])
