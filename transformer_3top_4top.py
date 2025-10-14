import h5py
import pandas as pd
import numpy as np
import torch
import pytorch_lightning
import torch.nn as nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateFinder
from pathlib import Path


def load_data():
    