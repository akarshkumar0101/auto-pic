# %load_ext autoreload
# %autoreload 2

import sys

from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from PIL import Image
from torchvision import transforms

import utils
from utils import to_np, count_params
from image_cppn import ImageCPPN, BatchImageCPPN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

