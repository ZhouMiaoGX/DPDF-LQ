from .GlobalFusionClassifier import GlobalFusionClassifier
from .LocalFusionClassifier import LocalFusionClassifier
import torch
from torch import nn


class Gate_fusion(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.Global_path = GlobalFusionClassifier(args) 
        self.Local_path = LocalFusionClassifier(args)
        

        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        self.regression = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, video, audio, text):
        #global feature
        global_feat = self.Global_path(video, audio, text)  # [B,128]
        
        #local feature 
        local_feat = self.Local_path(video, audio, text)  # [B,128]
        
        #gate fusion
        combined = torch.cat([global_feat, local_feat], dim=1)
        gate = self.fusion_gate(combined)  # [B,2]
        fused = gate[:,0:1]*global_feat + gate[:,1:2]*local_feat
        fused = self.regression(fused)
        return fused
def build_model(args):
    model = Gate_fusion(args)

    return model