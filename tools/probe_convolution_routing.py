"""Observe forward/backward provider names and actual accelerator profiler events."""
import argparse
import json
import os
from pathlib import Path
import sys


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--backend',choices=('nvidia','metax'),required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assert not args.output.exists()
sys.path.insert(0,str(args.source.resolve()))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import torch.nn.functional as F
from up.accelerator import configure_accelerator
torch.set_num_threads(2)
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
provider=configure_accelerator({'runtime': {'accelerator': {'backend': args.backend,'numerics':'portable_v1'}}},reset=True)
provider.configure_convolution(True)
generator=torch.Generator().manual_seed(131)
x=torch.randn(2,32,32,40,generator=generator).cuda().requires_grad_()
w=torch.randn(64,32,3,3,generator=generator).cuda().requires_grad_()
F.conv2d(x,w,padding=1).sum().backward()
torch.cuda.synchronize()
x.grad=w.grad=None
activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
with torch.profiler.profile(activities=activities,record_shapes=True) as profile:
    y=F.conv2d(x,w,padding=1)
    forward_flag=torch.backends.cudnn.enabled
    y.sum().backward()
    backward_flag=torch.backends.cudnn.enabled
    torch.cuda.synchronize()
events=profile.events()
result={'status':'completed','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'forward_exit_convolution_enabled':forward_flag,'backward_exit_convolution_enabled':backward_flag,
        'cpu_event_names':sorted({e.name for e in events if 'conv' in e.name.lower() or 'cudnn' in e.name.lower()}),
        'accelerator_events':[{'name':e.name,'device_type':str(e.device_type),'device_index':e.device_index} for e in events if str(e.device_type) not in ('DeviceType.CPU','CPU')],
        'numerics':provider.numerics_state()}
args.output.write_text(json.dumps(result,indent=2,sort_keys=True))
print(json.dumps({key:value for key,value in result.items() if key!='accelerator_events'},indent=2))
print(json.dumps({'accelerator_events':len(result['accelerator_events'])}))
