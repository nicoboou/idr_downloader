"""Check free GPUs and allocate it to cuda"""

import subprocess
import sys
import torch
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

    try:
        str_gpu_stats = StringIO(gpu_stats)
    except:
        str_gpu_stats = StringIO(gpu_stats.decode("utf-8"))

    gpu_df = pd.read_csv(str_gpu_stats,
                         names=['memory.used', 'memory.free'],
                         skiprows=1)

    print('GPU usage:\n{}'.format(gpu_df))

    gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: int(x.rstrip(' [MiB]')))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = (gpu_df['memory.free']-gpu_df['memory.used']).idxmax()

    print('Returning GPU{} with {} used MiB and {} free MiB'.format(idx,gpu_df.iloc[idx]['memory.used'], gpu_df.iloc[idx]['memory.free']))
    
    free_gpu_id = int(get_free_gpu()) # trouve le gpu libre grace a la fonction precedente
    torch.cuda.set_device(free_gpu_id) # definie le gpu libre trouvee comme gpu de defaut pour PyTorch
    set_gpu="cuda:"+str(free_gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device