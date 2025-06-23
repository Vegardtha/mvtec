import sys
sys.path.append('patchcore-inspection')

import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Torch imported successfully")

from patchcore.patchcore import PatchCore
print("PatchCore imported successfully")
from torchvision import datasets, transforms
print("Torchvision imported successfully")
from torch.utils.data import DataLoader
print("DataLoader imported successfully")