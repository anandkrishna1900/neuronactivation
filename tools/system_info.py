"""Collect system info for benchmark report."""
import torch, platform, psutil, os, sys, json

info = {}
info["python"] = platform.python_version()
info["os"] = platform.platform()
info["cpu"] = platform.processor()
info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
info["cpu_cores_logical"] = psutil.cpu_count()
info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
info["pytorch"] = torch.__version__
info["cuda_available"] = torch.cuda.is_available()

if torch.cuda.is_available():
    info["cuda_version"] = torch.version.cuda
    info["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    info["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
    info["gpu_sm_count"] = props.multi_processor_count
    info["gpu_capability"] = list(torch.cuda.get_device_capability())
    info["cudnn_version"] = torch.backends.cudnn.version()

for k, v in info.items():
    print(f"  {k}: {v}")

with open("results/phase7/benchmark/system_info.json", "w") as f:
    json.dump(info, f, indent=2)
