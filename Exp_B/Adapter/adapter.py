import torch
import torch.nn as nn

from torch.utils.data import Dataset
from PIL import Image
import torchaudio
import os

class Adapter(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)



from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from transformers import ClapModel, ClapProcessor

imagebind = ImageBindModel().eval().cuda()
clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().cuda()
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

adapter = Adapter().cuda()


