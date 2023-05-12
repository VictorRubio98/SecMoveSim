import torch
import os


with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=1,
        active=3,
        repeat=2
    ),
    profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")
) as prof:
    os.system('python main.py --pretrain --data=geolife --cuda=0')

print(prof.key_averages().table(sort_by="self_cuda_time_total"))