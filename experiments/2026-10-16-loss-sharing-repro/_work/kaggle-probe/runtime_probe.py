import importlib.util,json,subprocess,torch
from pathlib import Path
r={'cuda_available':torch.cuda.is_available(),'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'modules':{n:importlib.util.find_spec(n) is not None for n in ['sknetwork','numpy','scipy','hydra','omegaconf','wandb','h5py','requests']}}
print(json.dumps(r,indent=2))
Path('/kaggle/working/runtime.json').write_text(json.dumps(r,indent=2))
