import umap
import umap.plot
import wandb
import math
from .helpfuns import *
from .dist_utills import *
from PIL import ImageOps, ImageFilter, ImageFile
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import RandomApply, RandomResizedCrop, InterpolationMode

# For truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def compute_stats(dataloader):
    from tqdm import tqdm

    channels = dataloader.dataset[0][0].size(0)
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x in tqdm(dataloader):
        x_tot += x[0].mean([0, 2, 3]).cpu().numpy()
        x2_tot += (x[0] ** 2).mean([0, 2, 3]).cpu().numpy()

    channel_avr = x_tot / len(dataloader)
    channel_std = np.sqrt(x2_tot / len(dataloader) - channel_avr**2)
    return channel_avr, channel_std

def pil_loader(img_path, n_channels):
    with open(img_path, "rb") as f:
        img = Image.open(f)
        if n_channels == 3:
            return img.convert("RGB")
        elif n_channels == 1:
            return img.convert("L")
        elif n_channels == 4:
            return img.convert("RGBA")
        else:
            raise NotImplementedError(
                "PIL only supports 1,3 and 4 channel inputs. Use cv2 instead"
            )

def model_to_CPU_state(net):
    """Gets the state_dict to CPU for easier save/load later."""
    if is_parallel(net):
        state_dict = {k: deepcopy(v.cpu()) for k, v in net.module.state_dict().items()}
    else:
        state_dict = {k: deepcopy(v.cpu()) for k, v in net.state_dict().items()}
    return OrderedDict(state_dict)

def optimizer_to_CPU_state(opt):
    """Gets the state_dict to CPU for easier save/load later."""
    state_dict = {}
    state_dict["state"] = {}
    state_dict["param_groups"] = deepcopy(opt.state_dict()["param_groups"])

    for k, v in opt.state_dict()["state"].items():
        state_dict["state"][k] = {}
        if v:
            for _k, _v in v.items():
                if torch.is_tensor(_v):
                    elem = deepcopy(_v.cpu())
                else:
                    elem = deepcopy(_v)
                state_dict["state"][k][_k] = elem
    return state_dict

class MovingMeans:
    def __init__(self, window=5):
        self.window = window
        self.values = []

    def add(self, val):
        self.values.append(val)

    def get_value(self):
        return np.convolve(
            np.array(self.values), np.ones((self.window,)) / self.window, mode="valid"
        )[-1]

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __call__(self, running_module, ma_module):
        if ma_module is None:
            return new
        return ma_module * self.beta + (1 - self.beta) * running_module

def conv2d_kaiming_uniform_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")


def conv2d_kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")


def create_umap_embeddings(
    feature_bank, targets, label_mapper=None, umap_path="umap_emb"
):
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)
    if label_mapper is None:
        n_classes = len(np.unique(targets))
        label_mapper = {val: "class_" + str(val) for val in range(n_classes)}
    named_targets = np.asarray([label_mapper[trgt] for trgt in targets])

    # create umaps and plot
    embedding = umap.UMAP().fit(feature_bank)
    figure_handler = umap.plot.points(embedding, named_targets)
    plt.savefig(umap_path)

def modules_are_equal(m1, m2):
    for w1, w2 in zip(m1.parameters(), m2.parameters()):
        if w1.data.ne(w2.data).sum() > 0:
            return False
    return True

def show_unused_params(subnet):
    for name, param in subnet.named_parameters():
        if param.grad is None:
            print(name)


def cancel_gradients(subnet, attr):
    for n, p in subnet.named_parameters():
        if attr in n:
            p.grad = None